#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware-Maximized Validation Mode Execution Script

Executes Phase 3 Hybrid RL + LLM Trading Agent testing with maximum GPU utilization.
Targets >90% GPU utilization while using only 10% of production timesteps.

Key Features:
- 32 parallel vectorized environments
- Batch size optimized for GPU (1024)
- Cached LLM feature calculations
- Vectorized decision fusion
- Real-time hardware monitoring
- Automated GPU utilization validation

Usage:
    python run_hardware_maximized.py --market NQ --mock-llm
    python run_hardware_maximized.py --market ES --config config/test_hardware_maximized.yaml
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.testing_framework import (
    TestingFramework, 
    create_test_config, 
    HardwareMonitor
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/hardware_maximized_execution.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def validate_environment():
    """Validate execution environment."""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        return False
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} ({gpu_count} device(s))")
        else:
            logger.warning("No GPU detected - CPU mode will be very slow")
    except ImportError:
        logger.error("PyTorch not installed - required for GPU acceleration")
        return False
    
    # Check required packages
    required_packages = [
        "stable_baselines3",
        "sb3_contrib", 
        "pandas",
        "numpy",
        "gymnasium",
        "psutil"
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
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error("Data directory not found. Run data processing first.")
        return False
    
    logger.info("Environment validation passed")
    return True


def check_gpu_memory():
    """Check available GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            available = total - reserved
            
            logger = logging.getLogger(__name__)
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, "
                       f"{reserved:.2f}GB reserved, "
                       f"{available:.2f}GB available of {total:.2f}GB total")
            
            return available, total
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not check GPU memory: {e}")
    
    return 0, 0


def run_pre_execution_checks(config: Dict[str, Any]) -> bool:
    """Run pre-execution checks and optimizations."""
    logger = logging.getLogger(__name__)
    logger.info("Running pre-execution checks...")
    
    # Check GPU memory
    available_memory, total_memory = check_gpu_memory()
    
    if available_memory > 0:
        # Adjust configuration based on available memory
        required_memory = 8.0  # GB minimum recommended
        
        if available_memory < required_memory:
            logger.warning(f"Low GPU memory: {available_memory:.2f}GB available. "
                          f"Recommended: {required_memory}GB+")
            
            # Reduce batch size and environments
            config['batch_size'] = min(config['batch_size'], 256)
            config['vectorized_envs'] = min(config['vectorized_envs'], 4)
            logger.info(f"Adjusted config: batch_size={config['batch_size']}, "
                       f"n_envs={config['vectorized_envs']}")
        
        # Verify we can achieve target utilization
        if available_memory / total_memory < 0.5:
            logger.warning("Limited GPU memory may prevent reaching target utilization")
    
    # Check CPU cores
    import psutil
    cpu_count = psutil.cpu_count(logical=False)
    logger.info(f"CPU cores: {cpu_count}")
    
    if cpu_count < 4:
        logger.warning("Low CPU core count may limit parallel processing")
    
    # Check system memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    logger.info(f"System memory: {memory_gb:.1f}GB")
    
    if memory_gb < 16:
        logger.warning("Low system memory may cause performance issues")
    
    # Clear GPU cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")
    except:
        pass
    
    logger.info("Pre-execution checks completed")
    return True


def monitor_execution(framework: TestingFramework, duration: int = 10):
    """Monitor execution and provide real-time feedback."""
    import threading
    
    def monitor():
        logger = logging.getLogger(__name__)
        monitor = HardwareMonitor(log_interval=5)
        monitor.start_monitoring()
        
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(10)
            
            avg_gpu = monitor.get_average_gpu_utilization()
            peak_memory = monitor.get_peak_memory_usage()
            
            logger.info(f"Live Metrics - GPU: {avg_gpu:.1f}%, "
                       f"Peak Memory: {peak_memory:.2f}GB")
            
            if avg_gpu < 85.0:
                logger.warning(f"GPU utilization below target: {avg_gpu:.1f}%")
        
        monitor.stop_monitoring()
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    
    return monitor_thread


def run_hardware_maximized_test(market: str, mock_llm: bool = False, 
                               config_file: Optional[str] = None,
                               verbose: bool = False) -> bool:
    """Run hardware-maximized validation test."""
    logger = setup_logging(verbose)
    
    logger.info("=" * 70)
    logger.info("HARDWARE-MAXIMIZED VALIDATION MODE")
    logger.info("=" * 70)
    logger.info(f"Market: {market}")
    logger.info(f"Mock LLM: {mock_llm}")
    logger.info(f"Config: {config_file or 'Default'}")
    logger.info("=" * 70)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return False
    
    # Load configuration
    if config_file:
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return False
    else:
        # Use default configuration
        config_dict = {
            'market': market,
            'mock_llm': mock_llm,
            'vectorized_envs': 32,
            'batch_size': 1024,
            'timesteps_reduction': 0.1
        }
    
    # Run pre-execution checks
    if not run_pre_execution_checks(config_dict):
        logger.warning("Pre-execution checks identified issues but continuing...")
    
    try:
        # Create test configuration
        test_config = create_test_config(
            mode="hardware_maximized",
            market=market,
            mock_llm=mock_llm,
            **config_dict
        )
        
        # Initialize framework
        framework = TestingFramework(test_config)
        
        # Start monitoring
        monitor_thread = monitor_execution(framework, duration=3600)  # 1 hour max
        
        # Run test
        start_time = time.time()
        framework.run_hardware_maximized_validation()
        execution_time = time.time() - start_time
        
        # Wait for monitor to finish
        monitor_thread.join(timeout=10)
        
        # Get final metrics
        avg_gpu_util = framework.hardware_monitor.get_average_gpu_utilization()
        peak_memory = framework.hardware_monitor.get_peak_memory_usage()
        
        # Validate results
        success = True
        
        if avg_gpu_util < 90.0:
            logger.error(f"GPU utilization below target: {avg_gpu_util:.1f}% (target: >90%)")
            success = False
        else:
            logger.info(f"✓ GPU utilization target achieved: {avg_gpu_util:.1f}%")
        
        if execution_time > 3600:  # 1 hour
            logger.warning(f"Execution time exceeded target: {execution_time:.0f}s (target: <3600s)")
        else:
            logger.info(f"✓ Execution time within target: {execution_time:.0f}s")
        
        # Log final summary
        logger.info("=" * 70)
        logger.info("FINAL RESULTS")
        logger.info("=" * 70)
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        logger.info(f"Average GPU Utilization: {avg_gpu_util:.1f}%")
        logger.info(f"Peak GPU Memory: {peak_memory:.2f} GB")
        logger.info(f"Total Timesteps: {framework.total_timesteps}")
        
        if success:
            logger.info("✓ HARDWARE-MAXIMIZED TEST PASSED")
        else:
            logger.error("✗ HARDWARE-MAXIMIZED TEST FAILED")
        
        logger.info("=" * 70)
        
        return success
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Hardware-Maximized Validation Mode for Phase 3 Hybrid RL + LLM Trading Agent"
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
        default="config/test_hardware_maximized.yaml",
        help="Configuration file path (default: config/test_hardware_maximized.yaml)"
    )
    
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM for testing without GPU (faster testing)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (1% timesteps instead of 10%)"
    )
    
    parser.add_argument(
        "--vectorized-envs", "-n",
        type=int,
        default=32,
        help="Number of vectorized environments (default: 32)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training (default: 1024)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Override config with command line arguments
    config_overrides = {
        'vectorized_envs': args.vectorized_envs,
        'batch_size': args.batch_size
    }
    
    if args.quick:
        config_overrides['timesteps_reduction'] = 0.01  # 1% for quick test
    
    # Create results directory
    results_dir = Path("results/hardware_maximized")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run test
    success = run_hardware_maximized_test(
        market=args.market,
        mock_llm=args.mock_llm,
        config_file=args.config if Path(args.config).exists() else None,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()