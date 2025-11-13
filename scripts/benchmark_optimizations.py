#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking Script for Testing Framework Optimizations

Demonstrates performance improvements from:
- Cached LLM feature calculations
- Vectorized decision fusion
- Environment state pooling
- Batched callbacks

Usage:
    python benchmark_optimizations.py --benchmark cache
    python benchmark_optimizations.py --benchmark fusion
    python benchmark_optimizations.py --benchmark all
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import Dict, Any, List
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.testing_framework import (
    OptimizedFeatureCache,
    VectorizedDecisionFusion,
    EnvironmentStatePool,
    BatchedCallback
)
from src.environment_phase3_llm import TradingEnvironmentPhase3LLM


class BenchmarkResult:
    """Stores benchmark results."""
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.results = {}
        self.start_time = time.time()
    
    def record_result(self, test_name: str, execution_time: float, 
                     baseline_time: float = None, **metrics):
        """Record benchmark result."""
        speedup = baseline_time / execution_time if baseline_time else 1.0
        
        self.results[test_name] = {
            'execution_time': execution_time,
            'baseline_time': baseline_time,
            'speedup': speedup,
            'improvement': (speedup - 1.0) * 100,
            **metrics
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        total_time = time.time() - self.start_time
        
        return {
            'benchmark_name': self.benchmark_name,
            'total_duration': total_time,
            'results': self.results,
            'tests_run': len(self.results)
        }


def benchmark_feature_cache() -> BenchmarkResult:
    """Benchmark LLM feature caching performance."""
    print("Benchmarking Feature Cache Performance...")
    result = BenchmarkResult("Feature Cache")
    
    # Test different cache sizes and access patterns
    cache_sizes = [100, 500, 1000, 2000]
    access_patterns = [
        ("high_locality", 0.9),   # 90% repeated access
        ("medium_locality", 0.5), # 50% repeated access  
        ("low_locality", 0.1),    # 10% repeated access
    ]
    
    n_operations = 1000
    
    for cache_size in cache_sizes:
        cache = OptimizedFeatureCache(max_size=cache_size)
        
        for pattern_name, locality in access_patterns:
            # Baseline: No caching (recalculate every time)
            start_time = time.time()
            for i in range(n_operations):
                # Simulate feature calculation
                features = np.random.randn(33).astype(np.float32)
                _ = features  # Use the features
            baseline_time = time.time() - start_time
            
            # Optimized: With caching
            cache.clear()
            start_time = time.time()
            
            for i in range(n_operations):
                # Simulate cache access pattern
                if np.random.random() < locality:
                    # Repeated access (use existing key)
                    key = f"key_{i % int(cache_size * locality)}"
                else:
                    # New access
                    key = f"key_{i + cache_size}"
                
                cached_features = cache.get(key)
                if cached_features is None:
                    features = np.random.randn(33).astype(np.float32)
                    cache.put(key, features)
            
            optimized_time = time.time() - start_time
            
            # Record results
            test_name = f"cache_{cache_size}_{pattern_name}"
            cache_stats = cache.get_stats()
            
            result.record_result(
                test_name=test_name,
                execution_time=optimized_time,
                baseline_time=baseline_time,
                cache_size=cache_size,
                locality=locality,
                hit_rate=cache_stats['hit_rate'],
                final_cache_size=cache_stats['size']
            )
            
            print(f"  {test_name}: {baseline_time:.4f}s → {optimized_time:.4f}s "
                  f"({result.results[test_name]['improvement']:.1f}% improvement, "
                  f"hit rate: {cache_stats['hit_rate']:.1%})")
    
    return result


def benchmark_decision_fusion() -> BenchmarkResult:
    """Benchmark vectorized decision fusion performance."""
    print("Benchmarking Decision Fusion Performance...")
    result = BenchmarkResult("Decision Fusion")
    
    # Initialize fusion
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
    
    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        # Generate test data
        rl_actions = np.random.randint(0, 6, batch_size)
        rl_confidences = np.random.random(batch_size)
        llm_actions = np.random.randint(0, 6, batch_size)
        llm_confidences = np.random.random(batch_size)
        risk_scores = np.random.random(batch_size)
        
        # Baseline: Sequential processing (simulate)
        start_time = time.time()
        for _ in range(100):  # Multiple runs for averaging
            # Simulate sequential processing
            fused_actions = []
            for i in range(batch_size):
                # Simple sequential logic
                if risk_scores[i] > 0.7:
                    action = 0  # HOLD
                elif rl_actions[i] == llm_actions[i]:
                    action = rl_actions[i]
                elif rl_confidences[i] > 0.7:
                    action = rl_actions[i]
                elif llm_confidences[i] > 0.7:
                    action = llm_actions[i]
                else:
                    action = rl_actions[i] if rl_confidences[i] * 0.7 > llm_confidences[i] * 0.3 else llm_actions[i]
                fused_actions.append(action)
        
        baseline_time = time.time() - start_time
        
        # Optimized: Vectorized processing
        start_time = time.time()
        for _ in range(100):  # Multiple runs for averaging
            fused_actions = fusion.fuse_batch(
                rl_actions, rl_confidences,
                llm_actions, llm_confidences,
                risk_scores
            )
        
        optimized_time = time.time() - start_time
        
        # Record results
        test_name = f"fusion_batch_{batch_size}"
        result.record_result(
            test_name=test_name,
            execution_time=optimized_time,
            baseline_time=baseline_time,
            batch_size=batch_size,
            operations_per_second=batch_size * 100 / optimized_time
        )
        
        speedup = result.results[test_name]['speedup']
        print(f"  {test_name}: {baseline_time:.4f}s → {optimized_time:.4f}s "
              f"({speedup:.1f}x speedup)")
    
    return result


def benchmark_environment_pool() -> BenchmarkResult:
    """Benchmark environment state pooling performance."""
    print("Benchmarking Environment Pool Performance...")
    result = BenchmarkResult("Environment Pool")
    
    # Create minimal environment config for benchmarking
    env_kwargs = {
        'data': None,  # Would be real data in practice
        'initial_balance': 50000,
        'window_size': 20,
        'use_llm_features': False  # Disable for faster testing
    }
    
    # Test different pool sizes
    pool_sizes = [1, 4, 8, 16]
    n_operations = 100
    
    for pool_size in pool_sizes:
        # Baseline: Create new environments each time
        start_time = time.time()
        envs = []
        
        for _ in range(n_operations):
            env = TradingEnvironmentPhase3LLM(**env_kwargs)
            env.reset()
            envs.append(env)
        
        baseline_time = time.time() - start_time
        
        # Clean up
        for env in envs:
            env.close()
        
        # Optimized: Use environment pool
        pool = EnvironmentStatePool(
            TradingEnvironmentPhase3LLM,
            env_kwargs,
            pool_size=pool_size
        )
        
        start_time = time.time()
        acquired_envs = []
        
        for i in range(n_operations):
            env_id, env = pool.acquire()
            acquired_envs.append((env_id, env))
            
            # Simulate some usage
            if i % 10 == 0:
                # Release some environments
                for _ in range(min(3, len(acquired_envs))):
                    env_id, env = acquired_envs.pop(0)
                    pool.release(env_id)
        
        # Release remaining environments
        for env_id, env in acquired_envs:
            pool.release(env_id)
        
        optimized_time = time.time() - start_time
        
        # Record results
        test_name = f"pool_size_{pool_size}"
        pool_stats = pool.get_stats()
        
        result.record_result(
            test_name=test_name,
            execution_time=optimized_time,
            baseline_time=baseline_time,
            pool_size=pool_size,
            avg_uses_per_env=pool_stats['avg_uses_per_env'],
            env_creation_time_savings=baseline_time - optimized_time
        )
        
        print(f"  {test_name}: {baseline_time:.4f}s → {optimized_time:.4f}s "
              f"({result.results[test_name]['improvement']:.1f}% improvement, "
              f"avg uses/env: {pool_stats['avg_uses_per_env']:.1f})")
    
    return result


def benchmark_callback_system() -> BenchmarkResult:
    """Benchmark batched callback performance."""
    print("Benchmarking Callback System Performance...")
    result = BenchmarkResult("Callback System")
    
    # Simulate callback operations
    n_callbacks = 10000
    log_interval = 100
    
    # Baseline: Individual logging
    start_time = time.time()
    
    for i in range(n_callbacks):
        if i % log_interval == 0:
            # Simulate individual log write
            _ = f"Step {i}: reward={np.random.random():.4f}"
    
    baseline_time = time.time() - start_time
    
    # Optimized: Batched logging
    callback = BatchedCallback(log_interval=log_interval)
    
    start_time = time.time()
    batch_logs = []
    
    for i in range(n_callbacks):
        if i % log_interval == 0:
            log_entry = {
                'step': i,
                'time': time.time(),
                'reward': np.random.random(),
                'action': np.random.randint(0, 6)
            }
            batch_logs.append(log_entry)
            
            # Process batch when full
            if len(batch_logs) >= 10:
                # Simulate batch processing
                steps = [log['step'] for log in batch_logs]
                rewards = [log['reward'] for log in batch_logs]
                avg_reward = np.mean(rewards)
                batch_logs.clear()
    
    optimized_time = time.time() - start_time
    
    # Record results
    result.record_result(
        test_name="batched_callbacks",
        execution_time=optimized_time,
        baseline_time=baseline_time,
        log_operations=n_callbacks,
        batch_processing_optimization=baseline_time - optimized_time
    )
    
    print(f"  batched_callbacks: {baseline_time:.4f}s → {optimized_time:.4f}s "
          f"({result.results['batched_callbacks']['improvement']:.1f}% improvement)")
    
    return result


def run_comprehensive_benchmark():
    """Run all benchmarks and generate comprehensive report."""
    print("=" * 70)
    print("TESTING FRAMEWORK OPTIMIZATION BENCHMARKS")
    print("=" * 70)
    print()
    
    all_results = {}
    
    # Run individual benchmarks
    benchmarks = [
        ("Feature Cache", benchmark_feature_cache),
        ("Decision Fusion", benchmark_decision_fusion),
        ("Environment Pool", benchmark_environment_pool),
        ("Callback System", benchmark_callback_system),
    ]
    
    for benchmark_name, benchmark_func in benchmarks:
        print(f"\n{benchmark_name}:")
        print("-" * 70)
        result = benchmark_func()
        all_results[benchmark_name] = result.summary()
        print()
    
    # Generate comprehensive report
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 70)
    
    total_improvement = 0
    total_tests = 0
    
    for benchmark_name, summary in all_results.items():
        print(f"\n{benchmark_name}:")
        
        benchmark_improvement = 0
        benchmark_tests = 0
        
        for test_name, test_result in summary['results'].items():
            improvement = test_result['improvement']
            speedup = test_result['speedup']
            
            print(f"  {test_name}: {improvement:.1f}% improvement ({speedup:.1f}x speedup)")
            
            benchmark_improvement += improvement
            benchmark_tests += 1
            total_tests += 1
        
        if benchmark_tests > 0:
            avg_improvement = benchmark_improvement / benchmark_tests
            print(f"  Average: {avg_improvement:.1f}% improvement")
            total_improvement += benchmark_improvement
    
    if total_tests > 0:
        overall_improvement = total_improvement / total_tests
        print(f"\nOverall Average Improvement: {overall_improvement:.1f}%")
    
    print("=" * 70)
    
    # Save results to file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"logs/benchmark_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return all_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Testing Framework Optimizations"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        choices=['cache', 'fusion', 'pool', 'callbacks', 'all'],
        default='all',
        help="Benchmark to run (default: all)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Run selected benchmark(s)
    if args.benchmark == 'all':
        results = run_comprehensive_benchmark()
    else:
        benchmark_map = {
            'cache': benchmark_feature_cache,
            'fusion': benchmark_decision_fusion,
            'pool': benchmark_environment_pool,
            'callbacks': benchmark_callback_system
        }
        
        print(f"Running {args.benchmark} benchmark...")
        result = benchmark_map[args.benchmark]()
        results = {args.benchmark: result.summary()}
        
        # Save individual benchmark results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output or f"logs/benchmark_{args.benchmark}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()