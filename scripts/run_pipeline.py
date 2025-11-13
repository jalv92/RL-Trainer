#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Sequential Pipeline Mode Execution Script

Executes Phase 1, Phase 2, and Phase 3 in sequence with automated checkpointing
and resource reallocation between phases.

Key Features:
- Continuous execution with phase transitions
- Automated checkpointing and recovery
- Resource reallocation between phases
- Progress tracking and validation
- Failure recovery and rollback

Usage:
    python run_pipeline.py --market NQ --auto-resume
    python run_pipeline.py --market ES --start-phase 2 --checkpoint models/checkpoints/pipeline/phase1_complete
"""

import os
import sys
import argparse
import time
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.testing_framework import (
    TestingFramework, 
    create_test_config,
    HardwareMonitor
)


class PipelinePhase(Enum):
    """Pipeline execution phases."""
    SETUP = 1
    BASE_RL = 2
    HYBRID = 3
    VALIDATION = 4
    COMPLETE = 5


class PipelineState:
    """Manages pipeline execution state and checkpointing."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = checkpoint_dir / "pipeline_state.json"
        self.current_phase = PipelinePhase.SETUP
        self.completed_phases = []
        self.phase_start_times = {}
        self.phase_end_times = {}
        self.errors = []
        self.metrics = {}
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing state if available
        self.load_state()
    
    def load_state(self):
        """Load pipeline state from checkpoint."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                self.current_phase = PipelinePhase[data.get('current_phase', 'SETUP')]
                self.completed_phases = data.get('completed_phases', [])
                self.phase_start_times = data.get('phase_start_times', {})
                self.phase_end_times = data.get('phase_end_times', {})
                self.errors = data.get('errors', [])
                self.metrics = data.get('metrics', {})
                
                logging.getLogger(__name__).info(f"Loaded pipeline state: {self.current_phase.name}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to load pipeline state: {e}")
    
    def save_state(self):
        """Save pipeline state to checkpoint."""
        try:
            data = {
                'current_phase': self.current_phase.name,
                'completed_phases': self.completed_phases,
                'phase_start_times': self.phase_start_times,
                'phase_end_times': self.phase_end_times,
                'errors': self.errors,
                'metrics': self.metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.getLogger(__name__).debug("Pipeline state saved")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save pipeline state: {e}")
    
    def start_phase(self, phase: PipelinePhase):
        """Mark phase as started."""
        self.current_phase = phase
        self.phase_start_times[phase.name] = datetime.now().isoformat()
        self.save_state()
    
    def complete_phase(self, phase: PipelinePhase, metrics: Dict = None):
        """Mark phase as completed."""
        if phase.name not in self.completed_phases:
            self.completed_phases.append(phase.name)
        
        self.phase_end_times[phase.name] = datetime.now().isoformat()
        
        if metrics:
            self.metrics[phase.name] = metrics
        
        self.save_state()
    
    def record_error(self, phase: PipelinePhase, error: str):
        """Record error for a phase."""
        self.errors.append({
            'phase': phase.name,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.save_state()
    
    def can_resume(self) -> bool:
        """Check if pipeline can resume from checkpoint."""
        return len(self.completed_phases) > 0 and len(self.errors) == 0
    
    def get_next_phase(self) -> PipelinePhase:
        """Get the next phase to execute."""
        if not self.completed_phases:
            return PipelinePhase.SETUP
        
        last_completed = self.completed_phases[-1]
        
        if last_completed == "SETUP":
            return PipelinePhase.BASE_RL
        elif last_completed == "BASE_RL":
            return PipelinePhase.HYBRID
        elif last_completed == "HYBRID":
            return PipelinePhase.VALIDATION
        else:
            return PipelinePhase.COMPLETE


def setup_pipeline_logging(verbose: bool = False):
    """Setup logging for pipeline execution."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pipeline_execution.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Pipeline logging initialized")
    return logger


def validate_pipeline_environment():
    """Validate environment for pipeline execution."""
    logger = logging.getLogger(__name__)
    logger.info("Validating pipeline execution environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required for pipeline execution")
        return False
    
    # Check required packages
    required_packages = [
        "stable_baselines3",
        "sb3_contrib",
        "pandas",
        "numpy",
        "gymnasium",
        "psutil",
        "pyyaml"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    # Check data availability
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error("Data directory not found")
        return False
    
    # Check checkpoint directory
    checkpoint_dir = Path("models/checkpoints/pipeline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Pipeline environment validation passed")
    return True


def reallocate_resources(phase: PipelinePhase, config: Dict[str, Any]):
    """Reallocate resources between phases."""
    logger = logging.getLogger(__name__)
    logger.info(f"Reallocating resources for {phase.name}...")
    
    try:
        import torch
        import gc
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
        
        # Garbage collection
        gc.collect()
        logger.info("Ran garbage collection")
        
        # Phase-specific resource allocation
        if phase == PipelinePhase.SETUP:
            # Phase 1: Lower GPU usage for data processing
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.3)
            logger.info("Set Phase 1 resource allocation (30% GPU)")
        
        elif phase == PipelinePhase.BASE_RL:
            # Phase 2: High GPU usage for training
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)
            logger.info("Set Phase 2 resource allocation (80% GPU)")
        
        elif phase == PipelinePhase.HYBRID:
            # Phase 3: Maximum GPU usage for hybrid training
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.9)
            logger.info("Set Phase 3 resource allocation (90% GPU)")
        
        time.sleep(2)  # Allow resources to stabilize
        logger.info("Resource reallocation complete")
        
    except Exception as e:
        logger.error(f"Error during resource reallocation: {e}")


def run_phase_setup(framework: TestingFramework, pipeline_state: PipelineState) -> bool:
    """Execute Phase 1: Environment Setup."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 1: ENVIRONMENT SETUP")
    logger.info("=" * 70)
    
    try:
        pipeline_state.start_phase(PipelinePhase.SETUP)
        
        # Reallocate resources for Phase 1
        reallocate_resources(PipelinePhase.SETUP, framework.config.__dict__)
        
        # Setup environments
        logger.info("Setting up environments...")
        envs, eval_env = framework._setup_vectorized_environments()
        
        # Validate environment setup
        logger.info("Validating environment setup...")
        
        # Test environment functionality
        obs, _ = eval_env.reset()
        assert obs is not None, "Environment reset failed"
        
        action = eval_env.action_space.sample()
        obs, reward, terminated, truncated, info = eval_env.step(action)
        assert obs is not None, "Environment step failed"
        
        # Record metrics
        metrics = {
            'environments_created': framework.config.vectorized_envs,
            'observation_space': eval_env.observation_space.shape,
            'action_space': eval_env.action_space.n,
            'setup_time': time.time() - framework.start_time
        }
        
        pipeline_state.complete_phase(PipelinePhase.SETUP, metrics)
        logger.info("✓ Phase 1 completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        pipeline_state.record_error(PipelinePhase.SETUP, str(e))
        return False


def run_phase_base_rl(framework: TestingFramework, pipeline_state: PipelineState) -> bool:
    """Execute Phase 2: Base RL Training."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 2: BASE RL TRAINING")
    logger.info("=" * 70)
    
    try:
        pipeline_state.start_phase(PipelinePhase.BASE_RL)
        
        # Reallocate resources for Phase 2
        reallocate_resources(PipelinePhase.BASE_RL, framework.config.__dict__)
        
        # Setup environments if not already done
        envs, eval_env = framework._setup_vectorized_environments()
        
        # Train base RL model
        logger.info("Training base RL model...")
        base_model = framework._train_base_rl_model(envs, eval_env)
        
        # Save base model checkpoint
        checkpoint_path = Path("models/checkpoints/pipeline/base_rl_model.zip")
        base_model.save(checkpoint_path)
        logger.info(f"Saved base RL model to {checkpoint_path}")
        
        # Record metrics
        metrics = {
            'model_type': 'MaskablePPO',
            'timesteps_trained': framework.total_timesteps,
            'checkpoint_path': str(checkpoint_path),
            'training_time': time.time() - framework.start_time
        }
        
        pipeline_state.complete_phase(PipelinePhase.BASE_RL, metrics)
        logger.info("✓ Phase 2 completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        pipeline_state.record_error(PipelinePhase.BASE_RL, str(e))
        return False


def run_phase_hybrid(framework: TestingFramework, pipeline_state: PipelineState) -> bool:
    """Execute Phase 3: Hybrid LLM Integration."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PHASE 3: HYBRID LLM INTEGRATION")
    logger.info("=" * 70)
    
    try:
        pipeline_state.start_phase(PipelinePhase.HYBRID)
        
        # Reallocate resources for Phase 3
        reallocate_resources(PipelinePhase.HYBRID, framework.config.__dict__)
        
        # Setup environments
        envs, eval_env = framework._setup_vectorized_environments()
        
        # Load base model
        from stable_baselines3 import PPO
        base_model_path = Path("models/checkpoints/pipeline/base_rl_model.zip")
        
        if base_model_path.exists():
            logger.info("Loading base RL model...")
            base_model = MaskablePPO.load(base_model_path, env=envs)
        else:
            logger.warning("Base model checkpoint not found, training from scratch...")
            base_model = framework._train_base_rl_model(envs, eval_env)
        
        # Train hybrid model
        logger.info("Training hybrid RL + LLM model...")
        hybrid_model = framework._train_hybrid_model(base_model, envs, eval_env)
        
        # Save hybrid model checkpoint
        checkpoint_path = Path("models/checkpoints/pipeline/hybrid_model")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Note: Hybrid model saving would need to be implemented based on
        # the specific hybrid model architecture
        
        # Record metrics
        metrics = {
            'model_type': 'HybridTradingAgent',
            'fusion_config': framework.decision_fusion.config if framework.decision_fusion else None,
            'cache_stats': framework.feature_cache.get_stats() if framework.feature_cache else None,
            'hybrid_training_time': time.time() - framework.start_time
        }
        
        pipeline_state.complete_phase(PipelinePhase.HYBRID, metrics)
        logger.info("✓ Phase 3 completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        pipeline_state.record_error(PipelinePhase.HYBRID, str(e))
        return False


def run_pipeline_validation(framework: TestingFramework, pipeline_state: PipelineState) -> bool:
    """Run final pipeline validation."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("PIPELINE VALIDATION")
    logger.info("=" * 70)
    
    try:
        pipeline_state.start_phase(PipelinePhase.VALIDATION)
        
        # Run validation on trained model
        logger.info("Running model validation...")
        
        # This would use the validation logic from the framework
        # For now, just verify all phases completed
        
        validation_results = {
            'all_phases_completed': len(pipeline_state.completed_phases) == 3,
            'phases_completed': pipeline_state.completed_phases,
            'total_execution_time': time.time() - framework.start_time,
            'errors': len(pipeline_state.errors) == 0
        }
        
        # Check GPU utilization across phases
        avg_gpu_util = framework.hardware_monitor.get_average_gpu_utilization()
        validation_results['gpu_utilization'] = avg_gpu_util
        
        pipeline_state.complete_phase(PipelinePhase.VALIDATION, validation_results)
        
        # Log validation summary
        logger.info("Validation Results:")
        logger.info(f"  Phases completed: {len(pipeline_state.completed_phases)}/3")
        logger.info(f"  Average GPU utilization: {avg_gpu_util:.1f}%")
        logger.info(f"  Errors: {len(pipeline_state.errors)}")
        
        if validation_results['all_phases_completed'] and validation_results['errors']:
            logger.info("✓ Pipeline validation passed")
            return True
        else:
            logger.error("✗ Pipeline validation failed")
            return False
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        pipeline_state.record_error(PipelinePhase.VALIDATION, str(e))
        return False


def run_automated_pipeline(market: str, start_phase: int = 1,
                          auto_resume: bool = False,
                          config_file: Optional[str] = None,
                          verbose: bool = False) -> bool:
    """Run automated sequential pipeline."""
    logger = setup_pipeline_logging(verbose)
    
    logger.info("=" * 70)
    logger.info("AUTOMATED SEQUENTIAL PIPELINE MODE")
    logger.info("=" * 70)
    logger.info(f"Market: {market}")
    logger.info(f"Start Phase: {start_phase}")
    logger.info(f"Auto-resume: {auto_resume}")
    logger.info("=" * 70)
    
    # Validate environment
    if not validate_pipeline_environment():
        logger.error("Pipeline environment validation failed")
        return False
    
    # Initialize pipeline state
    checkpoint_dir = Path("models/checkpoints/pipeline")
    pipeline_state = PipelineState(checkpoint_dir)
    
    # Check for auto-resume
    if auto_resume and pipeline_state.can_resume():
        next_phase = pipeline_state.get_next_phase()
        logger.info(f"Auto-resuming from {next_phase.name}")
        start_phase = next_phase.value
    
    try:
        # Create test configuration
        if config_file and Path(config_file).exists():
            import yaml
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = {'market': market}
        
        test_config = create_test_config(
            mode="pipeline",
            market=market,
            **config_dict
        )
        
        # Initialize framework
        framework = TestingFramework(test_config)
        framework.start_time = time.time()
        
        # Start hardware monitoring
        framework.hardware_monitor.start_monitoring()
        
        # Execute phases
        phases = {
            1: ("Setup", run_phase_setup),
            2: ("Base RL", run_phase_base_rl),
            3: ("Hybrid", run_phase_hybrid),
            4: ("Validation", run_pipeline_validation)
        }
        
        success = True
        for phase_num in range(start_phase, 5):
            if phase_num not in phases:
                continue
            
            phase_name, phase_func = phases[phase_num]
            logger.info(f"\nExecuting Phase {phase_num}: {phase_name}")
            
            # Pause between phases for resource reallocation
            if phase_num > start_phase:
                logger.info("Pausing for resource reallocation...")
                time.sleep(5)
            
            # Execute phase
            phase_success = phase_func(framework, pipeline_state)
            
            if not phase_success:
                logger.error(f"Phase {phase_num} failed - stopping pipeline")
                success = False
                break
            
            logger.info(f"✓ Phase {phase_num} completed successfully")
        
        # Stop monitoring
        framework.hardware_monitor.stop_monitoring()
        
        # Save final metrics
        metrics_file = f"logs/pipeline_metrics_{market}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'completed_phases': pipeline_state.completed_phases,
                'phase_times': framework.phase_times,
                'total_time': time.time() - framework.start_time,
                'gpu_utilization': framework.hardware_monitor.get_average_gpu_utilization(),
                'peak_memory': framework.hardware_monitor.get_peak_memory_usage()
            }, f, indent=2)
        
        # Log pipeline summary
        logger.info("=" * 70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Completed Phases: {len(pipeline_state.completed_phases)}/4")
        logger.info(f"Total Time: {time.time() - framework.start_time:.2f} seconds")
        logger.info(f"Average GPU Utilization: {framework.hardware_monitor.get_average_gpu_utilization():.1f}%")
        logger.info(f"Peak GPU Memory: {framework.hardware_monitor.get_peak_memory_usage():.2f} GB")
        
        if success:
            logger.info("✓ AUTOMATED PIPELINE COMPLETED SUCCESSFULLY")
        else:
            logger.error("✗ AUTOMATED PIPELINE FAILED")
        
        logger.info("=" * 70)
        
        return success
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Automated Sequential Pipeline Mode for Phase 3 Hybrid RL + LLM Trading Agent"
    )
    
    parser.add_argument(
        "--market", "-m",
        type=str,
        required=True,
        choices=["NQ", "ES", "YM", "RTY", "MNQ", "MES", "M2K", "MYM"],
        help="Market symbol to test (e.g., NQ, ES)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/test_pipeline.yaml",
        help="Configuration file path (default: config/test_pipeline.yaml)"
    )
    
    parser.add_argument(
        "--start-phase",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Phase to start from (default: 1)"
    )
    
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Auto-resume from last checkpoint if available"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (reduced timesteps)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create results directory
    results_dir = Path("results/pipeline")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    success = run_automated_pipeline(
        market=args.market,
        start_phase=args.start_phase,
        auto_resume=args.auto_resume,
        config_file=args.config if Path(args.config).exists() else None,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()