#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 Hybrid Agent Evaluation Script

Evaluates hybrid RL + LLM trading agent with component ablation study.
Compares performance of:
- Full hybrid agent (RL + LLM)
- RL-only baseline
- LLM contribution analysis

Generates comprehensive evaluation report with metrics and statistics.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from environment_phase3_llm import TradingEnvironmentPhase3LLM
    from llm_reasoning import LLMReasoningModule
    from hybrid_agent import HybridTradingAgent
    from feature_engineering import add_market_regime_features
    from market_specs import get_market_spec
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Phase 3 modules not available: {e}")
    PHASE3_AVAILABLE = False


def safe_print(message=""):
    """Print with fallback for encoding errors."""
    try:
        print(message)
    except UnicodeEncodeError:
        replacements = {
            '✓': '[OK]', '✅': '[OK]', '✗': '[X]', '❌': '[X]', '→': '->',
            '⚠': '[WARN]', '⚠️': '[WARN]', '—': '-', '–': '-', '’': "'", '“': '"', '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


def load_evaluation_data(market_name: str, test_period: str = "recent") -> pd.DataFrame:
    """
    Load evaluation data for testing.
    
    Args:
        market_name: Market to evaluate (e.g., 'NQ', 'ES')
        test_period: 'recent' for last 3 months, 'all' for all data
    
    Returns:
        DataFrame with evaluation data
    """
    import glob
    
    data_pattern = f"./data/processed/{market_name}_*.csv"
    data_files = glob.glob(data_pattern)
    
    if not data_files:
        raise ValueError(f"No data files found for {market_name}")
    
    # Load and combine data
    all_data = []
    for file in sorted(data_files):
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        all_data.append(df)
    
    data = pd.concat(all_data, axis=0)
    data.sort_index(inplace=True)
    
    # Add features
    data = add_market_regime_features(data)
    
    # Select test period
    if test_period == "recent":
        # Use last 3 months or 30% of data, whichever is smaller
        test_start = max(data.index[0], data.index[-1] - pd.Timedelta(days=90))
        data = data[data.index >= test_start]
    
    safe_print(f"[DATA] Loaded {len(data)} rows for evaluation")
    safe_print(f"[DATA] Date range: {data.index[0]} to {data.index[-1]}")
    
    return data


def create_evaluation_env(data: pd.DataFrame, market_name: str = None, use_llm_features: bool = True):
    """Create evaluation environment."""
    from environment_phase3_llm import TradingEnvironmentPhase3LLM
    
    market_spec = get_market_spec(market_name) if market_name else None
    
    env = TradingEnvironmentPhase3LLM(
        data=data,
        use_llm_features=use_llm_features,
        initial_balance=50000,
        window_size=20,
        second_data=None,
        market_spec=market_spec,
        commission_override=None,
        initial_sl_multiplier=1.5,
        initial_tp_ratio=3.0,
        position_size_contracts=1.0,
        trailing_drawdown_limit=2500,
        tighten_sl_step=0.5,
        extend_tp_step=1.0,
        trailing_activation_profit=1.0
    )

    env = ActionMasker(env, lambda env: env.action_masks())
    env = Monitor(env)
    
    return env


def evaluate_agent(agent, env, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
    """
    Evaluate an agent on the environment.
    
    Args:
        agent: Agent to evaluate (can be hybrid, RL-only, or any callable)
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        render: Whether to render episodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    from collections import defaultdict
    
    episode_returns = []
    episode_lengths = []
    episode_trades = []
    episode_win_rates = []
    
    # For hybrid agents
    fusion_stats = defaultdict(list)
    llm_stats = defaultdict(list)
    
    safe_print(f"[EVAL] Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        episode_return = 0
        episode_length = 0
        episode_trades_list = []
        
        while not (done or truncated):
            # Get action from agent
            if hasattr(agent, 'predict'):
                # Hybrid agent
                position_state = {
                    'position': env.position,
                    'balance': env.balance,
                    'win_rate': env._calculate_win_rate(),
                    'consecutive_losses': env.consecutive_losses,
                    'dd_buffer_ratio': (env.balance - (env.peak_balance - 2500)) / env.balance if env.peak_balance > 0 else 1.0
                }
                market_context = env.get_llm_context()
                action_mask = env.action_masks()
                
                action, meta = agent.predict(obs, action_mask, position_state, market_context)
                
                # Track fusion stats
                if 'fusion_method' in meta:
                    fusion_stats[meta['fusion_method']].append(1)
                if 'risk_veto' in meta and meta['risk_veto']:
                    fusion_stats['risk_veto'].append(1)
                if 'llm_queried' in meta and meta['llm_queried']:
                    fusion_stats['llm_queried'].append(1)
            else:
                # RL-only agent
                action, _ = agent.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_return += reward
            episode_length += 1
            
            # Track trades
            if 'trade_pnl' in info and info['trade_pnl'] != 0:
                episode_trades_list.append(info['trade_pnl'])
        
        # Episode summary
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_trades.append(len(episode_trades_list))
        
        if episode_trades_list:
            win_rate = sum(1 for pnl in episode_trades_list if pnl > 0) / len(episode_trades_list)
            episode_win_rates.append(win_rate)
        else:
            episode_win_rates.append(0.0)
        
        if (episode + 1) % max(1, num_episodes // 5) == 0:
            safe_print(f"[EVAL] Completed {episode + 1}/{num_episodes} episodes...")
    
    # Calculate metrics
    returns_array = np.array(episode_returns)
    lengths_array = np.array(episode_lengths)
    trades_array = np.array(episode_trades)
    win_rates_array = np.array(episode_win_rates)
    
    metrics = {
        'mean_return': np.mean(returns_array),
        'std_return': np.std(returns_array),
        'sharpe_ratio': np.mean(returns_array) / (np.std(returns_array) + 1e-8),
        'mean_length': np.mean(lengths_array),
        'mean_trades': np.mean(trades_array),
        'win_rate': np.mean(win_rates_array),
        'total_return': np.sum(returns_array),
        'max_drawdown': np.max(np.maximum.accumulate(returns_array) - returns_array),
        'profit_factor': calculate_profit_factor(episode_trades_list) if episode_trades_list else 0.0
    }
    
    # Add fusion statistics if available
    if fusion_stats:
        total_steps = sum(len(v) for v in fusion_stats.values())
        metrics['fusion_stats'] = {
            k: len(v) / total_steps * 100 if total_steps > 0 else 0
            for k, v in fusion_stats.items()
        }
    
    return metrics


def calculate_profit_factor(trades_list: List[float]) -> float:
    """Calculate profit factor from list of trade P&Ls."""
    if not trades_list:
        return 0.0
    
    gross_profit = sum(pnl for pnl in trades_list if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trades_list if pnl < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def run_component_ablation_study(data: pd.DataFrame, model_path: str, market_name: str, config: Dict) -> Dict[str, Any]:
    """
    Run component ablation study comparing:
    - Full hybrid agent (RL + LLM)
    - RL-only baseline
    - LLM-only (if feasible)
    
    Args:
        data: Evaluation data
        model_path: Path to trained RL model
        market_name: Market name
        config: Configuration dictionary
    
    Returns:
        Dictionary with results for each component
    """
    safe_print("=" * 70)
    safe_print("COMPONENT ABLATION STUDY")
    safe_print("=" * 70)
    
    results = {}
    
    # Load RL model
    safe_print("[LOAD] Loading RL model...")
    try:
        rl_model = MaskablePPO.load(model_path)
        safe_print("[OK] RL model loaded")
    except Exception as e:
        safe_print(f"[ERROR] Failed to load RL model: {e}")
        return results
    
    # 1. Evaluate RL-only baseline
    safe_print("\n[EVAL] Evaluating RL-only baseline...")
    rl_env = create_evaluation_env(data, market_name, use_llm_features=False)
    
    rl_results = evaluate_agent(rl_model, rl_env, num_episodes=10)
    results['rl_only'] = rl_results
    
    safe_print("[OK] RL-only evaluation completed")
    print_metrics("RL-Only", rl_results)
    
    # 2. Evaluate full hybrid agent
    safe_print("\n[EVAL] Evaluating full hybrid agent...")
    
    # Initialize LLM
    try:
        llm_model = LLMReasoningModule(
            config_path=config['llm_config_path'],
            mock_mode=config.get('mock_llm', True)
        )
        safe_print("[OK] LLM advisor initialized")
    except Exception as e:
        safe_print(f"[ERROR] Failed to initialize LLM: {e}")
        return results
    
    # Create hybrid agent
    hybrid_agent = HybridTradingAgent(
        rl_model=rl_model,
        llm_model=llm_model,
        config=config
    )
    
    # Evaluate hybrid
    hybrid_env = create_evaluation_env(data, market_name, use_llm_features=True)
    hybrid_results = evaluate_agent(hybrid_agent, hybrid_env, num_episodes=10)
    results['hybrid'] = hybrid_results
    
    safe_print("[OK] Hybrid agent evaluation completed")
    print_metrics("Hybrid", hybrid_results)
    
    # 3. Close environments
    try:
        rl_env.close()
        hybrid_env.close()
    except:
        pass
    
    return results


def print_metrics(agent_name: str, metrics: Dict[str, Any]):
    """Print evaluation metrics in formatted way."""
    safe_print(f"\n{agent_name} Results:")
    safe_print(f"  Mean Return: ${metrics['mean_return']:,.2f}")
    safe_print(f"  Std Return: ${metrics['std_return']:,.2f}")
    safe_print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    safe_print(f"  Win Rate: {metrics['win_rate']:.1%}")
    safe_print(f"  Mean Trades/Episode: {metrics['mean_trades']:.1f}")
    safe_print(f"  Max Drawdown: ${metrics['max_drawdown']:,.2f}")
    safe_print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Print fusion stats if available
    if 'fusion_stats' in metrics:
        safe_print(f"  Fusion Stats:")
        for stat, value in metrics['fusion_stats'].items():
            safe_print(f"    {stat}: {value:.1f}%")


def generate_evaluation_report(results: Dict[str, Any], output_path: str = None):
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Evaluation results dictionary
        output_path: Optional path to save report
    """
    safe_print("\n" + "=" * 70)
    safe_print("EVALUATION REPORT")
    safe_print("=" * 70)
    
    if not results:
        safe_print("No results to report")
        return
    
    # Compare hybrid vs RL-only
    if 'hybrid' in results and 'rl_only' in results:
        hybrid = results['hybrid']
        rl_only = results['rl_only']
        
        safe_print("\nComponent Comparison:")
        safe_print("-" * 70)
        
        # Performance improvements
        return_improvement = hybrid['mean_return'] - rl_only['mean_return']
        sharpe_improvement = hybrid['sharpe_ratio'] - rl_only['sharpe_ratio']
        winrate_improvement = hybrid['win_rate'] - rl_only['win_rate']
        
        safe_print(f"Mean Return:     ${hybrid['mean_return']:,.2f} vs ${rl_only['mean_return']:,.2f} " +
                  f"({return_improvement:+.2f})")
        safe_print(f"Sharpe Ratio:    {hybrid['sharpe_ratio']:.2f} vs {rl_only['sharpe_ratio']:.2f} " +
                  f"({sharpe_improvement:+.2f})")
        safe_print(f"Win Rate:        {hybrid['win_rate']:.1%} vs {rl_only['win_rate']:.1%} " +
                  f"({winrate_improvement:+.1%})")
        
        # Statistical significance (simple t-test approximation)
        if hybrid['std_return'] > 0 and rl_only['std_return'] > 0:
            hybrid_sem = hybrid['std_return'] / np.sqrt(10)  # 10 episodes
            rl_sem = rl_only['std_return'] / np.sqrt(10)
            
            # Rough significance check (overlap of confidence intervals)
            hybrid_ci = 1.96 * hybrid_sem
            rl_ci = 1.96 * rl_sem
            
            safe_print(f"\nStatistical Significance (95% CI):")
            safe_print(f"  Hybrid:  ${hybrid['mean_return']:,.2f} ± ${hybrid_ci:,.2f}")
            safe_print(f"  RL-Only: ${rl_only['mean_return']:,.2f} ± ${rl_ci:,.2f}")
            
            if abs(return_improvement) > (hybrid_ci + rl_ci):
                safe_print(f"  → Improvement is statistically significant")
            else:
                safe_print(f"  → Improvement is not statistically significant")
        
        # LLM contribution analysis
        if 'fusion_stats' in hybrid:
            safe_print("\nLLM Contribution Analysis:")
            safe_print("-" * 70)
            fusion_stats = hybrid['fusion_stats']
            
            safe_print(f"LLM Query Rate: {fusion_stats.get('llm_queried', 0):.1f}%")
            safe_print(f"Agreement Rate: {fusion_stats.get('agreement', 0):.1f}%")
            safe_print(f"Risk Veto Rate: {fusion_stats.get('risk_veto', 0):.1f}%")
            
            # Estimate LLM impact
            llm_override_rate = fusion_stats.get('llm_weighted', 0) + fusion_stats.get('llm_confident', 0)
            safe_print(f"LLM Override Rate: {llm_override_rate:.1f}%")
    
    # Save report if path provided
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write("Phase 3 Hybrid Agent Evaluation Report\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if 'hybrid' in results and 'rl_only' in results:
                    f.write("Component Comparison:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Mean Return Improvement: ${return_improvement:,.2f}\n")
                    f.write(f"Sharpe Ratio Improvement: {sharpe_improvement:+.2f}\n")
                    f.write(f"Win Rate Improvement: {winrate_improvement:+.1%}\n\n")
            
            safe_print(f"\n[SAVE] Report saved to: {output_path}")
        except Exception as e:
            safe_print(f"[WARNING] Could not save report: {e}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Phase 3 Hybrid Agent Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--market', type=str, required=True, help='Market to evaluate')
    parser.add_argument('--config', type=str, default="./config/llm_config.yaml", help='LLM config path')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--output', type=str, help='Output report path')
    parser.add_argument('--mock-llm', action='store_true', help='Use mock LLM')
    
    args = parser.parse_args()
    
    if not PHASE3_AVAILABLE:
        safe_print("[ERROR] Phase 3 modules not available")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        safe_print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)
    
    safe_print("=" * 70)
    safe_print("Phase 3 Hybrid Agent Evaluation")
    safe_print("=" * 70)
    safe_print(f"Model: {args.model}")
    safe_print(f"Market: {args.market}")
    safe_print(f"Episodes: {args.episodes}")
    safe_print(f"Mock LLM: {args.mock_llm}")
    safe_print("=" * 70)
    
    # Load configuration
    config = {
        'llm_config_path': args.config,
        'mock_llm': args.mock_llm
    }
    
    try:
        # Load evaluation data
        safe_print("[LOAD] Loading evaluation data...")
        data = load_evaluation_data(args.market, test_period="recent")
        
        # Run ablation study
        results = run_component_ablation_study(data, args.model, args.market, config)
        
        # Generate report
        generate_evaluation_report(results, args.output)
        
        safe_print("\n✅ Evaluation completed successfully")
        
    except Exception as e:
        safe_print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()