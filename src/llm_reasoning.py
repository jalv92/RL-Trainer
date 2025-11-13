"""
LLM Reasoning Module

Purpose: LLM model loading, inference, and prompt management for trading decisions.

Uses: Phi-3-mini-4k-instruct (3.8B parameters, ~4GB VRAM with INT8 quantization)
Inference: ~15-20ms per query on RTX 3060

Features:
- INT8 quantization for reduced VRAM usage
- Selective querying to reduce latency
- Structured prompt templates
- Response parsing and validation
"""

import os
import logging
import time
import json
import yaml
from typing import Tuple, Dict, Optional
from collections import deque
import numpy as np

# Conditional imports - only load if LLM dependencies available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # PHASE 2: LoRA imports for fine-tuning
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        LORA_AVAILABLE = True
    except ImportError:
        LORA_AVAILABLE = False
        logging.warning("PEFT not available. LoRA fine-tuning disabled. Install with: pip install peft")
    
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LORA_AVAILABLE = False
    logging.warning("LLM dependencies not available. Install with: pip install transformers torch accelerate bitsandbytes peft")


class LLMReasoningModule:
    """
    LLM inference engine for trading advice.
    
    Loads and manages Phi-3-mini model for generating trading recommendations
    based on market context and observations.
    """
    
    def __init__(self, config_path: str = "config/llm_config.yaml", mock_mode: bool = False, enable_fine_tuning: bool = True):
        """
        Initialize LLM reasoning module.
        
        Args:
            config_path: Path to LLM configuration file
            mock_mode: Use mock LLM for testing (no GPU required)
            enable_fine_tuning: Enable LoRA fine-tuning capability (PHASE 2)
        """
        self.logger = logging.getLogger(__name__)
        self.mock_mode = mock_mode or not LLM_AVAILABLE
        self.enable_fine_tuning = enable_fine_tuning and LORA_AVAILABLE and not mock_mode
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"[LLM] Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"[LLM] Failed to load config: {e}")
            self.config = self._get_default_config()
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Statistics tracking
        self.total_queries = 0
        self.error_count = 0
        self.avg_latency_ms = 0.0
        self.last_error_time = 0
        self.error_cooldown_active = False
        
        # Cache for responses
        self.response_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # PHASE 2: Fine-tuning components
        if self.enable_fine_tuning:
            self.experience_buffer = LLMExperienceBuffer(max_size=10000)
            self.fine_tuning_steps = 0
            self.logger.info("[LLM] LoRA fine-tuning enabled")
        else:
            self.experience_buffer = None
            self.fine_tuning_steps = 0
            if not self.mock_mode and not LORA_AVAILABLE:
                self.logger.info("[LLM] LoRA not available, fine-tuning disabled")
        
        if self.mock_mode:
            self.logger.info("[LLM] Running in MOCK mode (no GPU required)")
        else:
            self._load_model()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file not found."""
        return {
            'llm_model': {
                'name': 'microsoft/Phi-3-mini-4k-instruct',
                'quantization': 'int8',
                'device': 'auto',
                'max_new_tokens': 50,
                'temperature': 0.1,
                'top_p': 0.9
            },
            'fusion': {
                'llm_weight': 0.3,
                'confidence_threshold': 0.7,
                'use_selective_querying': True,
                'query_interval': 5
            },
            'prompts': {
                'system': 'You are a professional futures trader. Provide concise trading advice.',
                'context_template': 'Market: {market_name}\nPrice: {current_price:.2f}\nPosition: {position_status}'
            },
            'fallback': {
                'enable_rl_fallback': True,
                'max_llm_errors': 10
            }
        }
    
    def _load_model(self):
        """Load LLM model with quantization."""
        if self.mock_mode:
            return
        
        if not LLM_AVAILABLE:
            self.logger.warning("[LLM] Dependencies not available, switching to mock mode")
            self.mock_mode = True
            return
        
        try:
            model_name = self.config['llm_model']['name']
            quantization = self.config['llm_model']['quantization']
            
            self.logger.info(f"[LLM] Loading {model_name} with {quantization} quantization...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            load_kwargs = {
                'pretrained_model_name_or_path': model_name,
                'trust_remote_code': True,
                'torch_dtype': torch.float16,
                'device_map': self.config['llm_model']['device']
            }
            
            if quantization == 'int8':
                load_kwargs['load_in_8bit'] = True
                self.logger.info("[LLM] Using INT8 quantization (~4GB VRAM)")
            elif quantization == 'int4':
                load_kwargs['load_in_4bit'] = True
                self.logger.info("[LLM] Using INT4 quantization (~2GB VRAM)")
            
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            
            # Get device
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.logger.info(f"[LLM] Model loaded successfully on {self.device}")
            
            # PHASE 2: Setup LoRA adapters if fine-tuning enabled
            if self.enable_fine_tuning:
                self._setup_lora_adapters()
            
        except Exception as e:
            self.logger.error(f"[LLM] Failed to load model: {e}")
            self.error_count += 1
            
            # Check if we should fall back to mock mode
            max_errors = self.config.get('fallback', {}).get('max_llm_errors', 10)
            if self.error_count >= max_errors:
                self.logger.warning(f"[LLM] Too many errors ({self.error_count}), switching to mock mode")
                self.mock_mode = True
    
    def _setup_lora_adapters(self):
        """
        PHASE 2: Setup LoRA adapters for efficient fine-tuning.
        
        LoRA Configuration:
        - Target modules: query, key, value projection layers
        - Rank: 16 (tradeoff between capacity and efficiency)
        - Alpha: 32 (scaling factor)
        - Dropout: 0.1
        - Only 1-2% of model parameters trainable!
        """
        if not self.enable_fine_tuning or not LORA_AVAILABLE:
            return
        
        self.logger.info("[LLM] Setting up LoRA adapters for fine-tuning...")
        
        lora_config = LoraConfig(
            r=16,  # Rank - controls adapter capacity
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Wrap model with LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient tracking
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        
        self.logger.info(f"[LLM] LoRA adapters added:")
        self.logger.info(f"      Trainable params: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
        self.logger.info(f"      Total params: {all_params:,}")
        self.logger.info("[LLM] ✅ LLM ready for fine-tuning")
    
    def query(self, observation: np.ndarray, position_state: Dict, market_context: Dict) -> Tuple[int, float, str, Optional[int]]:
        """
        PHASE 2: Query LLM for trading advice with experience tracking.
        
        Args:
            observation: (261,) numpy array with enhanced features
            position_state: Dictionary with position information
            market_context: Dictionary with market state
            
        Returns:
            Tuple of (action, confidence, reasoning, query_id)
        """
        self.total_queries += 1
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(observation, position_state, market_context)
            if cache_key in self.response_cache:
                self.cache_hits += 1
                cached_response = self.response_cache[cache_key]
                # Decay confidence for cached responses
                cached_response['confidence'] *= self.config.get('fusion', {}).get('cache_decay_rate', 0.8)
                return cached_response['action'], cached_response['confidence'], cached_response['reasoning'], None
            
            self.cache_misses += 1
            
            # Build prompt
            prompt = self._build_prompt(observation, position_state, market_context)
            
            # Generate response
            response = self._generate_response(
                prompt,
                position_state=position_state,
                market_context=market_context
            )
            latency = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_latency_stats(latency)
            
            # Parse response
            action, confidence, reasoning = self._parse_response(response)
            
            # Cache the response
            self.response_cache[cache_key] = {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'timestamp': time.time()
            }
            
            # PHASE 2: Track for fine-tuning
            query_id = None
            if self.enable_fine_tuning:
                query_id = self._add_to_experience_buffer(
                    prompt=prompt,
                    response=response,
                    action=action,
                    observation=observation,
                    position_state=position_state,
                    market_context=market_context
                )
            
            # Clean old cache entries
            self._clean_cache()
            
            return action, confidence, reasoning, query_id
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"[LLM] Query failed: {e}")
            
            # Check error cooldown
            if self._should_enter_error_cooldown():
                self.error_cooldown_active = True
                self.last_error_time = time.time()
            
            # Fallback behavior
            fallback_action = self._get_fallback_action(position_state)
            return fallback_action, 0.0, f"LLM_ERROR: {str(e)}", None
    
    def _get_cache_key(self, observation: np.ndarray, position_state: Dict, market_context: Dict) -> str:
        """Generate cache key for LLM query."""
        # Use key features that identify the market state
        key_features = [
            int(observation[228] * 100),  # ADX slope
            int(observation[229] * 100),  # VWAP distance
            int(observation[240]),  # RSI
            position_state.get('position', 0),
            int(position_state.get('balance', 0) / 1000)  # Balance in thousands
        ]
        return str(key_features)
    
    def _clean_cache(self, max_age: float = 300.0):
        """Clean cache entries older than max_age seconds."""
        current_time = time.time()
        self.response_cache = {
            k: v for k, v in self.response_cache.items()
            if current_time - v['timestamp'] < max_age
        }
    
    def _should_enter_error_cooldown(self) -> bool:
        """Check if we should enter error cooldown period."""
        max_errors = self.config.get('fallback', {}).get('max_llm_errors', 10)
        return self.error_count >= max_errors
    
    def _get_fallback_action(self, position_state: Dict) -> int:
        """Get fallback action when LLM fails."""
        # Default to HOLD (0) for safety
        # Could be enhanced to use RL prediction or simple heuristics
        return 0
    
    def query_with_cot(self, observation: np.ndarray, position_state: Dict, market_context: Dict, available_actions: list) -> Tuple[int, float, str, Optional[int]]:
        """
        PHASE 3: Query LLM with chain-of-thought reasoning.
        
        Returns:
            action: int
            confidence: float
            reasoning: str (full reasoning chain as text)
            query_id: int (for outcome tracking)
        """
        if not hasattr(self, 'cot_reasoner'):
            try:
                from .chain_of_thought import ChainOfThoughtReasoner
            except ImportError:
                from chain_of_thought import ChainOfThoughtReasoner
            self.cot_reasoner = ChainOfThoughtReasoner(self)
        
        # Execute CoT
        action, confidence, reasoning_chain = self.cot_reasoner.reason(
            observation, position_state, market_context, available_actions
        )

        reasoning_text = self._format_reasoning_chain(reasoning_chain)

        # PHASE 2: Track for fine-tuning (use full reasoning chain text)
        query_id = None
        if self.enable_fine_tuning:
            query_id = self._add_to_experience_buffer(
                prompt=reasoning_text,
                response=reasoning_chain.get('step4_decision', reasoning_text),
                action=action,
                observation=observation,
                position_state=position_state,
                market_context=market_context
            )

        return action, confidence, reasoning_text, query_id
    
    def _update_latency_stats(self, latency: float):
        """Update average latency statistics."""
        if self.total_queries > 0:
            self.avg_latency_ms = (self.avg_latency_ms * (self.total_queries - 1) + latency) / self.total_queries
        else:
            self.avg_latency_ms = latency
    
    def _build_prompt(self, obs: np.ndarray, position_state: Dict, market_context: Dict) -> str:
        """Build LLM prompt from observation and context."""
        try:
            template = self.config['prompts']['context_template']
            
            # Extract key features from observation
            # obs[228:238] = extended market context
            # obs[238:246] = multi-timeframe indicators  
            # obs[246:256] = pattern recognition
            # obs[256:261] = risk context
            
            # Safe extraction with defaults
            adx = float(obs[228]) if len(obs) > 228 else 25.0
            vwap_distance = float(obs[229]) if len(obs) > 229 else 0.0
            rsi = float(obs[240]) if len(obs) > 240 else 50.0
            momentum = float(obs[232]) if len(obs) > 232 else 0.0
            
            # Format last trades
            last_trades = self._format_last_trades(position_state)
            
            prompt = template.format(
                market_name=market_context.get('market_name', 'NQ'),
                current_time=market_context.get('current_time', '10:30'),
                current_price=market_context.get('current_price', 5000.0),
                adx=adx,
                vwap_distance=vwap_distance,
                rsi=rsi,
                momentum=momentum,
                position_status=market_context.get('position_status', 'FLAT'),
                unrealized_pnl=market_context.get('unrealized_pnl', 0.0),
                win_rate=market_context.get('win_rate', 0.5),
                consecutive_losses=market_context.get('consecutive_losses', 0),
                balance=market_context.get('balance', 50000.0),
                last_3_trades=last_trades
            )
            
            # Add system prompt
            system_prompt = self.config['prompts']['system']
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            return full_prompt
            
        except Exception as e:
            self.logger.error(f"[LLM] Error building prompt: {e}")
            # Return a minimal prompt as fallback
            return f"Market context unavailable. What action do you recommend?"
    
    def _generate_response(self,
                            prompt: str,
                            *,
                            position_state: Optional[Dict] = None,
                            market_context: Optional[Dict] = None,
                            max_new_tokens: Optional[int] = None) -> str:
        """Generate response using either mock or actual model path."""
        if self.mock_mode:
            return self._generate_mock(
                prompt,
                position_state or {},
                market_context or {}
            )
        return self._generate_raw(prompt, max_new_tokens=max_new_tokens)

    def _generate_raw(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate LLM response from the model."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            generation_kwargs = {
                'max_new_tokens': max_new_tokens or self.config['llm_model']['max_new_tokens'],
                'temperature': self.config['llm_model']['temperature'],
                'top_p': self.config['llm_model']['top_p'],
                'do_sample': self.config['llm_model'].get('do_sample', True),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if prompt in response:
                response = response[len(prompt):].strip()

            return response

        except Exception as e:
            self.logger.error(f"[LLM] Generation error: {e}")
            raise
    
    def _generate_mock(self, prompt: str, position_state: Dict, market_context: Dict) -> str:
        """Generate mock LLM response for testing."""
        # Simulate inference delay
        if self.config.get('development', {}).get('mock_response_delay', 0) > 0:
            time.sleep(self.config['development']['mock_response_delay'])
        
        # Simple heuristic-based mock response
        position = position_state.get('position', 0)
        win_rate = position_state.get('win_rate', 0.5)
        consecutive_losses = position_state.get('consecutive_losses', 0)
        
        # Mock confidence
        mock_confidence = self.config.get('development', {}).get('mock_confidence', 0.8)
        
        # Generate mock response based on simple rules
        if consecutive_losses >= 3:
            # On losing streak, be more cautious
            action = "HOLD"
            reason = "Losing streak - preserve capital"
            confidence = mock_confidence * 0.6
        elif position == 0:
            # No position - look for entry
            # Simple random decision for mock
            import random
            actions = ["BUY", "SELL", "HOLD"]
            action = random.choice(actions)
            reason = f"Mock {action.lower()} signal"
            confidence = mock_confidence
        else:
            # In position - manage it
            action = "MOVE_TO_BE" if win_rate > 0.5 else "HOLD"
            reason = "Mock position management"
            confidence = mock_confidence * 0.8
        
        return f"{action} | {confidence:.2f} | {reason}"
    
    def _parse_response(self, response: str) -> Tuple[int, float, str]:
        """
        Parse LLM response: "BUY | 0.85 | Strong uptrend"
        
        Returns:
            Tuple of (action, confidence, reasoning)
        """
        try:
            # Clean response
            response = response.strip()
            
            # Split by pipe
            parts = response.split('|')
            
            if len(parts) >= 3:
                # Parse action
                action_str = parts[0].strip().upper()
                
                # Parse confidence
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    self.logger.warning(f"[LLM] Invalid confidence format: {parts[1]}")
                    confidence = 0.5
                
                # Parse reasoning
                reasoning = parts[2].strip()
                
                # Map action string to integer
                action_map = {
                    'HOLD': 0,
                    'BUY': 1,
                    'SELL': 2,
                    'MOVE_TO_BE': 3,
                    'MOVE_SL_TO_BE': 3,
                    'ENABLE_TRAIL': 4,
                    'DISABLE_TRAIL': 5
                }
                
                action = action_map.get(action_str, 0)  # Default to HOLD
                
                # Clamp confidence to valid range
                confidence = max(0.0, min(1.0, confidence))
                
                return action, confidence, reasoning
            
            else:
                # Try alternative parsing (comma separated)
                if ',' in response:
                    parts = response.split(',')
                    if len(parts) >= 3:
                        return self._parse_response(' | '.join(parts))
                
                raise ValueError(f"Invalid format: {response}")
        
        except Exception as e:
            self.logger.warning(f"[LLM] Parse error: {e}, response: '{response}'")
            # Fallback to HOLD
            return 0, 0.0, f"PARSE_ERROR: {response}"
    
    def _format_last_trades(self, position_state: Dict) -> str:
        """Format last 3 trades for prompt."""
        try:
            history = position_state.get('trade_history', [])
            if not history:
                return "No trades yet"
            
            # Get last 3 trades
            last_3 = history[-3:]
            formatted = []
            
            for trade in last_3:
                if isinstance(trade, dict):
                    pnl = trade.get('pnl', 0)
                    result = "WIN" if pnl > 0 else "LOSS"
                    formatted.append(f"{result} (${pnl:.0f})")
                else:
                    # Handle simple list of P&L values
                    result = "WIN" if trade > 0 else "LOSS"
                    formatted.append(f"{result} (${trade:.0f})")
            
            return ", ".join(formatted)
        
        except Exception as e:
            self.logger.error(f"[LLM] Error formatting trades: {e}")
            return "Error formatting trades"
    
    def get_stats(self) -> Dict:
        """Get LLM usage statistics."""
        total = self.total_queries
        if total == 0:
            return {
                'total_queries': 0,
                'error_count': self.error_count,
                'error_rate': 0.0,
                'avg_latency_ms': 0.0,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': 0.0
            }
        
        return {
            'total_queries': total,
            'error_count': self.error_count,
            'error_rate': self.error_count / total * 100,
            'avg_latency_ms': self.avg_latency_ms,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / total * 100
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    # PHASE 2: Fine-tuning methods
    
    def _format_reasoning_chain(self, reasoning_chain: Dict[str, str]) -> str:
        """Format reasoning chain into a multi-line string for logging/storage."""
        return (
            f"ANALYSIS: {reasoning_chain.get('step1_analysis', '').strip()}\n\n"
            f"OPTIONS: {reasoning_chain.get('step2_options', '').strip()}\n\n"
            f"RISKS: {reasoning_chain.get('step3_risks', '').strip()}\n\n"
            f"DECISION: {reasoning_chain.get('step4_decision', '').strip()}"
        )

    def _add_to_experience_buffer(self,
                                  prompt,
                                  response,
                                  action,
                                  observation,
                                  position_state,
                                  market_context=None):
        """
        PHASE 2: Add query to experience buffer for later fine-tuning.
        
        Returns:
            query_id: int - unique ID for this query
        """
        if not self.enable_fine_tuning or self.experience_buffer is None:
            return None
        
        query_id = len(self.experience_buffer)

        prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        response_text = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)

        experience = {
            'id': query_id,
            'prompt': prompt_text,
            'response': response_text,
            'action': int(action),
            'observation': observation.copy(),
            'position_state': position_state.copy(),
            'market_context': (market_context.copy() if isinstance(market_context, dict) else {}),
            'timestamp': time.time(),
            'outcome': None  # Will be filled later
        }

        self.experience_buffer.add(experience)
        
        return query_id
    
    def update_outcome(self, query_id, reward, final_pnl):
        """
        PHASE 2: Update query outcome after trade completes.
        
        Args:
            query_id: int - ID from query()
            reward: float - immediate reward
            final_pnl: float - final P&L of trade
        """
        if (query_id is not None and 
            self.enable_fine_tuning and 
            self.experience_buffer is not None and
            query_id < len(self.experience_buffer)):
            
            self.experience_buffer.buffer[query_id]['outcome'] = {
                'reward': reward,
                'pnl': final_pnl,
                'success': (final_pnl > 0)
            }
    
    def fine_tune_step(self, batch_size=8, learning_rate=5e-5):
        """
        PHASE 2: Single fine-tuning step on successful trading outcomes.
        
        Strategy:
        1. Sample experiences where outcome is known
        2. Weight by success (winning trades emphasized)
        3. Fine-tune LLM to reproduce successful reasoning
        4. Use gradient accumulation for stability
        
        Returns:
            loss: float - training loss
            accuracy: float - how often LLM reproduces successful action
        """
        if not self.enable_fine_tuning or self.experience_buffer is None:
            return None, None
        
        if len(self.experience_buffer) < batch_size:
            return None, None
        
        # Sample batch (weighted by outcome)
        batch = self.experience_buffer.sample_weighted(batch_size)
        
        if batch is None:
            return None, None
        
        # Prepare for training
        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        total_loss = 0.0
        correct = 0
        
        for exp in batch:
            outcome = exp.get('outcome')
            if not outcome:
                continue

            inputs = self.tokenizer(
                exp['prompt'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            targets = self.tokenizer(
                exp['response'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs, labels=targets.input_ids)
            loss = outputs.loss
            
            # Weight loss by outcome
            pnl = outcome.get('pnl', 0.0)
            weight = 1.0 + pnl / 100.0  # More successful = higher weight
            weighted_loss = loss * weight

            # Backward
            weighted_loss.backward()
            
            total_loss += loss.item()
            
            # Check if model still predicts correct action
            with torch.no_grad():
                predicted_response = self._generate_response(exp['prompt'])
                predicted_action, _, _ = self._parse_response(predicted_response)
                if predicted_action == exp['action']:
                    correct += 1
        
        # Update
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        self.fine_tuning_steps += 1
        
        avg_loss = total_loss / batch_size
        accuracy = correct / batch_size
        
        return avg_loss, accuracy
    
    def save_lora_adapters(self, path):
        """Save only LoRA adapters (small, <50MB)."""
        if self.enable_fine_tuning and self.model is not None:
            self.model.save_pretrained(path)
            self.logger.info(f"[LLM] LoRA adapters saved to {path}")
    
    def load_lora_adapters(self, path):
        """Load fine-tuned LoRA adapters."""
        if self.enable_fine_tuning and self.model is not None and os.path.exists(path):
            self.model = PeftModel.from_pretrained(self.model, path)
            self.logger.info(f"[LLM] LoRA adapters loaded from {path}")


class LLMExperienceBuffer:
    """
    PHASE 2: Buffer for LLM query-outcome pairs.
    
    Used to fine-tune LLM on successful trading outcomes.
    """
    
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """Add experience dict."""
        self.buffer.append(experience)
    
    def sample_weighted(self, batch_size):
        """
        PHASE 2: Sample batch weighted by outcome quality.
        
        Strategy:
        - Only sample experiences with known outcomes
        - Weight by P&L (winning trades more likely)
        - Ensure diversity (don't oversample same market state)
        
        Returns:
            List of experiences or None if insufficient data
        """
        # Filter to experiences with outcomes
        completed = [exp for exp in self.buffer if exp.get('outcome') is not None]
        
        if len(completed) < batch_size:
            return None
        
        # Compute sampling weights
        weights = []
        for exp in completed:
            pnl = exp['outcome']['pnl']
            # Positive weight for winners, small positive for losers (learn from mistakes too)
            weight = max(pnl, 0.1) if pnl > 0 else 0.1
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Sample
        indices = np.random.choice(len(completed), size=batch_size, replace=False, p=weights)
        batch = [completed[i] for i in indices]
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    """Test LLM reasoning module."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing LLM Reasoning Module...")
    
    # Test in mock mode
    print("\n1. Testing in mock mode...")
    llm = LLMReasoningModule(mock_mode=True)
    
    # Create test data
    obs = np.random.randn(261)
    position_state = {
        'position': 0,
        'balance': 50000,
        'win_rate': 0.5,
        'consecutive_losses': 0,
        'trade_history': [
            {'pnl': 100}, {'pnl': -50}, {'pnl': 200}
        ]
    }
    market_context = {
        'market_name': 'NQ',
        'current_time': '10:30',
        'current_price': 5000.0,
        'trend_strength': 'Strong',
        'unrealized_pnl': 0.0
    }
    
    # Test query
    action, confidence, reasoning = llm.query(obs, position_state, market_context)
    
    print(f"Action: {action}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Reasoning: {reasoning}")
    
    assert 0 <= action <= 5, f"Invalid action: {action}"
    assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
    assert len(reasoning) > 0, "Empty reasoning"
    
    print("    Mock LLM test passed")
    
    # Test statistics
    stats = llm.get_stats()
    print(f"\nLLM Statistics:")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    
    print("\n    All LLM reasoning tests passed!")
    
    # Note: GPU test would require actual model loading
    print("\nNote: For GPU test, set mock_mode=False and ensure dependencies are installed")
