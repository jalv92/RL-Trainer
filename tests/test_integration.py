#!/usr/bin/env python3
"""
Integration Tests for Trading RL System

Tests the full pipeline from data loading through training to evaluation.
Based on COMPREHENSIVE_IMPROVEMENT_PLAN.md Phase 4.1.2
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment_phase1 import TradingEnvironmentPhase1
from environment_phase2 import TradingEnvironmentPhase2
from apex_compliance_checker import ApexComplianceChecker
from feature_engineering import add_market_regime_features, validate_features


class TestDataPipeline(unittest.TestCase):
    """Test data loading and feature engineering pipeline."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01 09:30', periods=500, freq='1min', tz='US/Eastern')

        self.test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 500),
            'high': np.random.uniform(4100, 4150, 500),
            'low': np.random.uniform(3950, 4000, 500),
            'close': np.random.uniform(4000, 4100, 500),
            'volume': np.random.randint(100, 1000, 500),
            'sma_5': np.random.uniform(4000, 4100, 500),
            'sma_20': np.random.uniform(4000, 4100, 500),
            'rsi': np.random.uniform(30, 70, 500),
            'macd': np.random.uniform(-10, 10, 500),
            'momentum': np.random.uniform(-50, 50, 500),
            'atr': np.random.uniform(10, 30, 500)
        }, index=dates)

    def test_feature_engineering_pipeline(self):
        """Test feature engineering adds all expected features."""
        # Add features
        enhanced_data = add_market_regime_features(self.test_data)

        # Check new features exist
        expected_features = [
            'adx', 'vol_regime', 'vol_percentile', 'vwap', 'price_to_vwap',
            'spread', 'efficiency_ratio', 'trend_strength',
            'session_morning', 'session_midday', 'session_afternoon'
        ]

        for feat in expected_features:
            self.assertIn(feat, enhanced_data.columns, f"Missing feature: {feat}")

        # Validate features
        is_valid = validate_features(enhanced_data)
        self.assertTrue(is_valid, "Feature validation failed")

        # Check no excessive NaNs
        nan_pct = enhanced_data.isna().sum() / len(enhanced_data)
        max_nan_pct = nan_pct.max()
        self.assertLess(max_nan_pct, 0.15, f"Too many NaNs: {max_nan_pct:.2%}")

    def test_environment_with_enhanced_features(self):
        """Test environment can use enhanced features."""
        # Add features
        enhanced_data = add_market_regime_features(self.test_data)

        # Create environment
        env = TradingEnvironmentPhase1(
            data=enhanced_data,
            initial_balance=50000,
            window_size=20
        )

        # Reset and step
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertFalse(np.isnan(obs).any(), "Observation contains NaN")

        # Take action
        obs2, reward, terminated, truncated, info = env.step(0)
        self.assertIsInstance(obs2, np.ndarray)
        self.assertFalse(np.isnan(obs2).any(), "Next observation contains NaN")


class TestTrainingPipeline(unittest.TestCase):
    """Test training pipeline components."""

    def setUp(self):
        """Set up test environment."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01 09:30', periods=200, freq='1min', tz='US/Eastern')

        self.test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 200),
            'high': np.random.uniform(4100, 4150, 200),
            'low': np.random.uniform(3950, 4000, 200),
            'close': np.random.uniform(4000, 4100, 200),
            'volume': np.random.randint(100, 1000, 200),
            'sma_5': np.random.uniform(4000, 4100, 200),
            'sma_20': np.random.uniform(4000, 4100, 200),
            'rsi': np.random.uniform(30, 70, 200),
            'macd': np.random.uniform(-10, 10, 200),
            'momentum': np.random.uniform(-50, 50, 200),
            'atr': np.random.uniform(10, 30, 200)
        }, index=dates)

        self.test_data = add_market_regime_features(self.test_data)

    def test_phase1_environment_full_episode(self):
        """Test Phase 1 environment can complete a full episode."""
        env = TradingEnvironmentPhase1(
            data=self.test_data,
            initial_balance=50000,
            window_size=20
        )

        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        max_steps = len(self.test_data) - 20 - 1

        while not terminated and not truncated and step_count < max_steps:
            action = np.random.choice([0, 1, 2])  # Random actions
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        # Should complete without errors
        self.assertGreater(step_count, 0)
        self.assertIsInstance(info['portfolio_value'], float)
        self.assertIsInstance(info['num_trades'], int)

    def test_phase2_environment_full_episode(self):
        """Test Phase 2 environment can complete a full episode."""
        env = TradingEnvironmentPhase2(
            data=self.test_data,
            initial_balance=50000,
            window_size=20
        )

        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        max_steps = len(self.test_data) - 20 - 1

        while not terminated and not truncated and step_count < max_steps:
            action = np.random.choice(range(6))  # Random actions (0-5)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        # Should complete without errors
        self.assertGreater(step_count, 0)
        self.assertIsInstance(info['portfolio_value'], float)

    def test_random_episode_starts(self):
        """Test that random episode starts produce different data slices."""
        # Simulate two "parallel environments" with different random starts
        np.random.seed(42)
        start1 = np.random.randint(0, 50)

        np.random.seed(43)
        start2 = np.random.randint(0, 50)

        # Should be different (with high probability)
        self.assertNotEqual(start1, start2, "Random starts should be different")

        # Create environments with different slices
        env1_data = self.test_data.iloc[start1:start1+100].copy()
        env2_data = self.test_data.iloc[start2:start2+100].copy()

        # First observations should be different
        if start1 != start2:
            self.assertFalse(env1_data.index[0] == env2_data.index[0])


class TestApexCompliance(unittest.TestCase):
    """Test Apex compliance checking integration."""

    def test_apex_compliance_checker_integration(self):
        """Test Apex compliance checker works with environment output."""
        # Create test data
        portfolio_values = [50000, 50500, 51000, 50800, 51500, 52000]
        trailing_dd_levels = [47500, 48000, 48500, 48300, 49000, 49500]

        trade_history = [
            {'pnl': 500, 'entry_price': 4000, 'exit_price': 4010, 'exit_reason': 'take_profit'},
            {'pnl': -200, 'entry_price': 4020, 'exit_price': 4016, 'exit_reason': 'stop_loss'},
            {'pnl': 700, 'entry_price': 4010, 'exit_price': 4024, 'exit_reason': 'take_profit'}
        ]

        timestamps = pd.date_range('2024-01-01 09:30', periods=6, freq='1H', tz='US/Eastern')
        position_sizes = [0.5] * 6

        # Create checker
        checker = ApexComplianceChecker(
            account_size=50000,
            trailing_dd_limit=2500,
            profit_target=3000,
            max_contracts=10
        )

        # Run compliance check
        results = checker.check_episode(
            portfolio_values=portfolio_values,
            trailing_dd_levels=trailing_dd_levels,
            trade_history=trade_history,
            timestamps=timestamps,
            position_sizes=position_sizes
        )

        # Verify results structure
        self.assertIn('passed', results)
        self.assertIn('violations', results)
        self.assertIn('metrics', results)
        self.assertIsInstance(results['passed'], bool)
        self.assertIsInstance(results['violations'], list)
        self.assertIsInstance(results['metrics'], dict)

    def test_4_59_pm_rule_enforcement(self):
        """Test 4:59 PM auto-close is enforced."""
        dates = pd.date_range('2024-01-01 16:58', periods=3, freq='1min', tz='US/Eastern')
        test_data = pd.DataFrame({
            'open': [4000, 4001, 4002],
            'high': [4005, 4006, 4007],
            'low': [3995, 3996, 3997],
            'close': [4000, 4001, 4002],
            'volume': [100, 100, 100],
            'sma_5': [4000, 4001, 4002],
            'sma_20': [4000, 4001, 4002],
            'rsi': [50, 50, 50],
            'macd': [0, 0, 0],
            'momentum': [0, 0, 0],
            'atr': [20, 20, 20]
        }, index=dates)

        test_data = add_market_regime_features(test_data)

        env = TradingEnvironmentPhase1(data=test_data, window_size=1)
        env.reset()

        # Open position
        env.position = 1
        env.entry_price = 4000
        env.current_step = 1  # At 4:59 PM

        # Check time rule
        current_time = test_data.index[1]
        terminated = env._check_apex_time_rules(current_time)

        # Should auto-close
        self.assertTrue(terminated)
        self.assertEqual(env.position, 0)
        self.assertGreater(len(env.apex_violations), 0)


class TestPositionManagement(unittest.TestCase):
    """Test Phase 2 position management features."""

    def setUp(self):
        """Set up Phase 2 test environment."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01 09:30', periods=100, freq='1min', tz='US/Eastern')

        self.test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 100),
            'high': np.random.uniform(4100, 4150, 100),
            'low': np.random.uniform(3950, 4000, 100),
            'close': np.random.uniform(4000, 4100, 100),
            'volume': np.random.randint(100, 1000, 100),
            'sma_5': np.random.uniform(4000, 4100, 100),
            'sma_20': np.random.uniform(4000, 4100, 100),
            'rsi': np.random.uniform(30, 70, 100),
            'macd': np.random.uniform(-10, 10, 100),
            'momentum': np.random.uniform(-50, 50, 100),
            'atr': np.random.uniform(10, 30, 100)
        }, index=dates)

        self.test_data = add_market_regime_features(self.test_data)

    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing based on volatility."""
        env = TradingEnvironmentPhase2(
            data=self.test_data,
            initial_balance=50000,
            window_size=20,
            position_size_contracts=1.0
        )

        env.reset()
        env.current_step = 50  # After warmup

        # Get position size
        size = env._calculate_position_size()

        # Should be between 0.5 and 1.0 (max)
        self.assertGreaterEqual(size, 0.5)
        self.assertLessEqual(size, 1.0)

        # Should adjust based on volatility
        self.assertIsInstance(size, float)

    def test_position_management_action_validation(self):
        """Test PM action validation prevents invalid actions."""
        env = TradingEnvironmentPhase2(
            data=self.test_data,
            initial_balance=50000,
            window_size=20
        )

        env.reset()
        env.current_step = 50

        # Open a position
        env.position = 1
        env.entry_price = 4000
        env.sl_price = 3980
        env.tp_price = 4040

        current_price = 4010
        atr = 20

        # Test valid tighten SL
        is_valid, reason = env._validate_position_management_action(
            env.ACTION_TIGHTEN_SL, current_price, atr
        )
        self.assertTrue(is_valid)

        # Test valid enable trailing
        is_valid, reason = env._validate_position_management_action(
            env.ACTION_ENABLE_TRAIL, current_price, atr
        )
        self.assertTrue(is_valid)


class TestMetricsCalculation(unittest.TestCase):
    """Test comprehensive metrics calculation."""

    def test_comprehensive_metrics_calculation(self):
        """Test all metrics are calculated correctly."""
        from evaluate_phase2 import calculate_comprehensive_metrics

        # Create test equity curve
        equity_curve = [50000, 50500, 51000, 50800, 51500, 52000, 51800]

        # Create test trade history
        trade_history = [
            {'pnl': 500, 'exit_reason': 'take_profit'},
            {'pnl': -200, 'exit_reason': 'stop_loss'},
            {'pnl': 700, 'exit_reason': 'take_profit'},
            {'pnl': -300, 'exit_reason': 'stop_loss'},
            {'pnl': 800, 'exit_reason': 'take_profit'}
        ]

        metrics = calculate_comprehensive_metrics(equity_curve, trade_history)

        # Check all required metrics exist
        required_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'profit_factor',
            'win_rate', 'avg_win', 'avg_loss', 'recovery_factor',
            'consistency_score', 'total_trades', 'apex_compliance_rate'
        ]

        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")

        # Verify metrics are reasonable
        self.assertGreater(metrics['total_return'], 0)  # Made profit
        self.assertEqual(metrics['total_trades'], 5)
        self.assertEqual(metrics['win_rate'], 0.6)  # 3 wins out of 5
        self.assertGreater(metrics['profit_factor'], 1.0)  # Profitable
        self.assertGreaterEqual(metrics['apex_compliance_rate'], 0.0)
        self.assertLessEqual(metrics['apex_compliance_rate'], 100.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
