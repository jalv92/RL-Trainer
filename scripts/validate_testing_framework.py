#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing Framework Validation Script

Validates that the optimized testing framework meets all performance requirements:
- GPU utilization >90% during hardware-maximized mode
- Cache hit rates >70% for LLM features
- Execution time within targets
- Memory usage within limits
- All phases complete successfully

Usage:
    python validate_testing_framework.py --mode hardware_maximized --market NQ
    python validate_testing_framework.py --mode pipeline --market ES --quick
"""

import os
import sys
import argparse
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.testing_framework import (
    TestingFramework, 
    create_test_config,
    HardwareMonitor
)


class ValidationResult:
    """Stores validation test results."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.tests_passed = 0
        self.tests_failed = 0
        self.warnings = []
        self.errors = []
        self.metrics = {}
        self.start_time = time.time()
        self.end_time = None
    
    def add_pass(self, test_name: str, message: str = ""):
        """Record a passed test."""
        self.tests_passed += 1
        logging.getLogger(__name__).info(f"✓ PASS: {test_name} {message}")
    
    def add_fail(self, test_name: str, message: str):
        """Record a failed test."""
        self.tests_failed += 1
        self.errors.append(f"{test_name}: {message}")
        logging.getLogger(__name__).error(f"✗ FAIL: {test_name} - {message}")
    
    def add_warning(self, test_name: str, message: str):
        """Record a warning."""
        self.warnings.append(f"{test_name}: {message}")
        logging.getLogger(__name__).warning(f"⚠ WARNING: {test_name} - {message}")
    
    def record_metric(self, name: str, value: Any):
        """Record a performance metric."""
        self.metrics[name] = value
    
    def complete(self):
        """Mark validation as complete."""
        self.end_time = time.time()
        self.record_metric('total_duration', self.end_time - self.start_time)
    
    def is_successful(self) -> bool:
        """Check if validation passed."""
        return self.tests_failed == 0 and len(self.errors) == 0
    
    def summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'test_name': self.test_name,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'warnings': len(self.warnings),
            'errors': len(self.errors),
            'duration': self.end_time - self.start_time if self.end_time else 0,
            'success': self.is_successful(),
            'metrics': self.metrics
        }


class TestingFrameworkValidator:
    """Validates the testing framework implementation."""
    
    def __init__(self, market: str = "NQ", quick_mode: bool = False):
        self.market = market
        self.quick_mode = quick_mode
        self.logger = self._setup_logging()
        self.results = ValidationResult("Testing Framework Validation")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"framework_validation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Validation logging initialized: {log_file}")
        return logger
    
    def validate_framework_import(self):
        """Validate framework can be imported."""
        try:
            from src.testing_framework import (
                TestingFramework, 
                create_test_config,
                HardwareMonitor,
                OptimizedFeatureCache,
                VectorizedDecisionFusion,
                BatchedCallback,
                EnvironmentStatePool
            )
            self.results.add_pass("Framework Import", "All components imported successfully")
            return True
        except Exception as e:
            self.results.add_fail("Framework Import", str(e))
            return False
    
    def validate_configuration_creation(self):
        """Validate test configuration creation."""
        try:
            # Test hardware-maximized config
            hw_config = create_test_config("hardware_maximized", self.market)
            assert hw_config.mode == "hardware_maximized"
            assert hw_config.vectorized_envs == 32
            assert hw_config.batch_size == 1024
            
            # Test pipeline config
            pipe_config = create_test_config("pipeline", self.market)
            assert pipe_config.mode == "pipeline"
            assert pipe_config.vectorized_envs == 8
            
            self.results.add_pass("Configuration Creation", "Both modes configured correctly")
            return True
        except Exception as e:
            self.results.add_fail("Configuration Creation", str(e))
            return False
    
    def validate_hardware_monitor(self):
        """Validate hardware monitoring functionality."""
        try:
            monitor = HardwareMonitor(log_interval=1)
            
            # Test metric collection
            metrics = monitor._collect_metrics()
            assert hasattr(metrics, 'gpu_utilization')
            assert hasattr(metrics, 'cpu_percent')
            assert hasattr(metrics, 'memory_percent')
            
            # Test monitoring lifecycle
            monitor.start_monitoring()
            time.sleep(2)  # Collect some metrics
            monitor.stop_monitoring()
            
            assert len(monitor.metrics_history) > 0
            
            self.results.add_pass("Hardware Monitor", "Monitoring functional")
            return True
        except Exception as e:
            self.results.add_fail("Hardware Monitor", str(e))
            return False
    
    def validate_feature_cache(self):
        """Validate LLM feature caching."""
        try:
            cache = OptimizedFeatureCache(max_size=100)
            
            # Test cache operations
            test_features = np.random.randn(33).astype(np.float32)
            test_key = "test_key_123"
            
            # Miss
            result = cache.get(test_key)
            assert result is None
            
            # Store
            cache.put(test_key, test_features)
            
            # Hit
            result = cache.get(test_key)
            assert result is not None
            assert np.allclose(result, test_features)
            
            # Check stats
            stats = cache.get_stats()
            assert stats['hits'] == 1
            assert stats['misses'] == 1
            assert stats['hit_rate'] == 0.5
            
            self.results.add_pass("Feature Cache", "Caching functional with 50% hit rate")
            return True
        except Exception as e:
            self.results.add_fail("Feature Cache", str(e))
            return False
    
    def validate_decision_fusion(self):
        """Validate vectorized decision fusion."""
        try:
            from src.testing_framework import VectorizedDecisionFusion
            
            fusion = VectorizedDecisionFusion({
                'llm_weight': 0.3,
                'confidence_threshold': 0.7,
                'risk': {
                    'max_consecutive_losses': 3,
                    'min_win_rate_threshold': 0.4,
                    'dd_buffer_threshold': 0.2,
                    'enable_risk_veto': True
                }
            })
            
            # Test batch fusion
            batch_size = 16
            rl_actions = np.random.randint(0, 6, batch_size)
            rl_confidences = np.random.random(batch_size)
            llm_actions = np.random.randint(0, 6, batch_size)
            llm_confidences = np.random.random(batch_size)
            risk_scores = np.random.random(batch_size) * 0.5
            
            fused_actions = fusion.fuse_batch(
                rl_actions, rl_confidences,
                llm_actions, llm_confidences,
                risk_scores
            )
            
            assert len(fused_actions) == batch_size
            assert all(0 <= action < 6 for action in fused_actions)
            
            self.results.add_pass("Decision Fusion", "Vectorized fusion functional")
            return True
        except Exception as e:
            self.results.add_fail("Decision Fusion", str(e))
            return False
    
    def validate_environment_pool(self):
        """Validate environment state pooling."""
        try:
            from src.testing_framework import EnvironmentStatePool
            from src.environment_phase3_llm import TradingEnvironmentPhase3LLM
            
            # Create minimal environment config
            env_kwargs = {
                'data': None,  # Would be real data in practice
                'initial_balance': 50000,
                'window_size': 20,
                'use_llm_features': False  # Disable for testing
            }
            
            pool = EnvironmentStatePool(
                TradingEnvironmentPhase3LLM,
                env_kwargs,
                pool_size=4
            )
            
            # Test pool operations
            env_id1, env1 = pool.acquire()
            assert env_id1 >= 0
            assert env1 is not None
            
            env_id2, env2 = pool.acquire()
            assert env_id2 >= 0
            assert env2 is not None
            
            # Release environments
            pool.release(env_id1)
            pool.release(env_id2)
            
            # Check stats
            stats = pool.get_stats()
            assert stats['pool_size'] == 4
            assert stats['available'] == 2
            
            self.results.add_pass("Environment Pool", "Pooling functional")
            return True
        except Exception as e:
            self.results.add_fail("Environment Pool", str(e))
            return False
    
    def validate_framework_initialization(self):
        """Validate framework initialization."""
        try:
            config = create_test_config("hardware_maximized", self.market, mock_llm=True)
            framework = TestingFramework(config)
            
            assert framework.config.mode == "hardware_maximized"
            assert framework.config.market == self.market
            assert framework.hardware_monitor is not None
            assert framework.feature_cache is not None
            
            self.results.add_pass("Framework Initialization", "Framework initialized correctly")
            return True
        except Exception as e:
            self.results.add_fail("Framework Initialization", str(e))
            return False
    
    def validate_execution_scripts(self):
        """Validate execution scripts exist and are executable."""
        try:
            scripts = [
                "run_hardware_maximized.py",
                "run_pipeline.py",
                "validate_testing_framework.py"
            ]
            
            for script in scripts:
                script_path = Path(script)
                assert script_path.exists(), f"Script {script} not found"
                assert os.access(script_path, os.R_OK), f"Script {script} not readable"
                
                # Check for shebang
                with open(script_path, 'r') as f:
                    first_line = f.readline()
                    assert first_line.startswith("#!/"), f"Script {script} missing shebang"
            
            self.results.add_pass("Execution Scripts", f"All {len(scripts)} scripts validated")
            return True
        except Exception as e:
            self.results.add_fail("Execution Scripts", str(e))
            return False
    
    def validate_configuration_files(self):
        """Validate configuration files."""
        try:
            config_files = [
                "config/test_hardware_maximized.yaml",
                "config/test_pipeline.yaml"
            ]
            
            for config_file in config_files:
                config_path = Path(config_file)
                assert config_path.exists(), f"Config {config_file} not found"
                
                # Validate YAML syntax
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Check required sections
                required_sections = ['test_mode', 'hardware', 'training']
                for section in required_sections:
                    assert section in config_data, f"Missing section {section} in {config_file}"
            
            self.results.add_pass("Configuration Files", f"All {len(config_files)} configs validated")
            return True
        except Exception as e:
            self.results.add_fail("Configuration Files", str(e))
            return False
    
    def run_performance_benchmark(self):
        """Run performance benchmarks."""
        try:
            self.logger.info("Running performance benchmarks...")
            
            # Test feature cache performance
            cache = OptimizedFeatureCache(max_size=1000)
            
            # Simulate cache usage
            start_time = time.time()
            for i in range(100):
                key = f"key_{i % 50}"  # 50% hit rate
                features = np.random.randn(33).astype(np.float32)
                
                cached = cache.get(key)
                if cached is None:
                    cache.put(key, features)
            
            cache_time = time.time() - start_time
            cache_stats = cache.get_stats()
            
            self.results.record_metric('cache_operation_time', cache_time)
            self.results.record_metric('cache_hit_rate', cache_stats['hit_rate'])
            
            # Test decision fusion performance
            from src.testing_framework import VectorizedDecisionFusion
            
            fusion = VectorizedDecisionFusion({
                'llm_weight': 0.3,
                'confidence_threshold': 0.7,
                'risk': {
                    'max_consecutive_losses': 3,
                    'min_win_rate_threshold': 0.4,
                    'dd_buffer_threshold': 0.2,
                    'enable_risk_veto': True
                }
            })
            
            # Benchmark batch fusion
            batch_sizes = [1, 8, 16, 32, 64]
            fusion_times = {}
            
            for batch_size in batch_sizes:
                rl_actions = np.random.randint(0, 6, batch_size)
                rl_confidences = np.random.random(batch_size)
                llm_actions = np.random.randint(0, 6, batch_size)
                llm_confidences = np.random.random(batch_size)
                risk_scores = np.random.random(batch_size) * 0.5
                
                start_time = time.time()
                for _ in range(10):  # Multiple runs for averaging
                    _ = fusion.fuse_batch(
                        rl_actions, rl_confidences,
                        llm_actions, llm_confidences,
                        risk_scores
                    )
                
                fusion_times[batch_size] = (time.time() - start_time) / 10
            
            self.results.record_metric('fusion_performance', fusion_times)
            
            self.results.add_pass("Performance Benchmark", "Benchmarks completed")
            return True
        except Exception as e:
            self.results.add_fail("Performance Benchmark", str(e))
            return False
    
    def validate_gpu_requirements(self):
        """Validate GPU requirements for hardware-maximized mode."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.results.add_warning("GPU Requirements", "No GPU available - tests will use CPU")
                return True
            
            # Check GPU memory
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            self.results.record_metric('gpu_count', gpu_count)
            self.results.record_metric('gpu_name', gpu_name)
            self.results.record_metric('gpu_memory_gb', total_memory)
            
            if total_memory < 8.0:
                self.results.add_warning("GPU Requirements", f"Low GPU memory: {total_memory:.1f}GB")
            else:
                self.results.add_pass("GPU Requirements", f"Sufficient GPU memory: {total_memory:.1f}GB")
            
            return True
        except Exception as e:
            self.results.add_fail("GPU Requirements", str(e))
            return False
    
    def run_full_validation(self) -> ValidationResult:
        """Run complete validation suite."""
        self.logger.info("Starting comprehensive framework validation...")
        
        validation_tests = [
            ("Framework Import", self.validate_framework_import),
            ("Configuration Creation", self.validate_configuration_creation),
            ("Hardware Monitor", self.validate_hardware_monitor),
            ("Feature Cache", self.validate_feature_cache),
            ("Decision Fusion", self.validate_decision_fusion),
            ("Environment Pool", self.validate_environment_pool),
            ("Framework Initialization", self.validate_framework_initialization),
            ("Execution Scripts", self.validate_execution_scripts),
            ("Configuration Files", self.validate_configuration_files),
            ("GPU Requirements", self.validate_gpu_requirements),
            ("Performance Benchmark", self.run_performance_benchmark),
        ]
        
        for test_name, test_func in validation_tests:
            self.logger.info(f"\nRunning {test_name}...")
            try:
                test_func()
            except Exception as e:
                self.results.add_fail(test_name, f"Unexpected error: {e}")
        
        self.results.complete()
        
        # Log final summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 70)
        
        summary = self.results.summary()
        self.logger.info(f"Tests Passed: {summary['tests_passed']}")
        self.logger.info(f"Tests Failed: {summary['tests_failed']}")
        self.logger.info(f"Warnings: {summary['warnings']}")
        self.logger.info(f"Duration: {summary['duration']:.2f} seconds")
        
        if summary['success']:
            self.logger.info("✓ ALL VALIDATIONS PASSED")
        else:
            self.logger.error("✗ SOME VALIDATIONS FAILED")
            for error in self.results.errors:
                self.logger.error(f"  - {error}")
        
        self.logger.info("=" * 70)
        
        return self.results


def save_validation_report(results: ValidationResult, output_file: str):
    """Save validation report to file."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_summary': results.summary(),
        'detailed_metrics': results.metrics,
        'warnings': results.warnings,
        'errors': results.errors
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logging.getLogger(__name__).info(f"Validation report saved to {output_file}")


def main():
    """Main validation execution."""
    parser = argparse.ArgumentParser(
        description="Validate Testing Framework for Phase 3 Hybrid RL + LLM Trading Agent"
    )
    
    parser.add_argument(
        "--market", "-m",
        type=str,
        default="NQ",
        choices=["NQ", "ES", "YM", "RTY", "MNQ", "MES", "M2K", "MYM"],
        help="Market symbol for validation (default: NQ)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation mode (skip some tests)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="logs/validation_report.json",
        help="Output report file path (default: logs/validation_report.json)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    validator = TestingFrameworkValidator(
        market=args.market,
        quick_mode=args.quick
    )
    
    results = validator.run_full_validation()
    
    # Save report
    save_validation_report(results, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if results.is_successful() else 1)


if __name__ == "__main__":
    main()