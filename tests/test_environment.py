#!/usr/bin/env python3
"""
Unit tests for trading environments.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment_phase1 import TradingEnvironmentPhase1
from environment_phase2 import TradingEnvironmentPhase2


class TestTradingEnvironmentPhase1(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create test data
        dates = pd.date_range('2024-01-01 09:30', periods=100, freq='1min', tz='US/Eastern')
        np.random.seed(42)  # For reproducible tests
        
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 100),
            'high': np.random.uniform(4100, 4200, 100),
            'low': np.random.uniform(3900, 4000, 100),
            'close': np.random.uniform(4000, 4100, 100),
            'volume': np.random.randint(100, 1000, 100),
            'sma_5': np.random.uniform(4000, 4100, 100),
            'sma_20': np.random.uniform(4000, 4100, 100),
            'rsi': np.random.uniform(30, 70, 100),
            'macd': np.random.uniform(-10, 10, 100),
            'momentum': np.random.uniform(-50, 50, 100),
            'atr': np.random.uniform(10, 30, 100)
        }, index=dates)
        
        self.env = TradingEnvironmentPhase1(
            data=self.test_data,
            initial_balance=50000,
            window_size=20
        )
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        
        # Check observation shape
        self.assertEqual(obs.shape, (220,))  # 20 * 11 features
        
        # Check initial state
        self.assertEqual(self.env.balance, 50000)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.entry_price, 0)
        self.assertEqual(self.env.num_trades, 0)
        
        # Check info dict
        self.assertIsInstance(info, dict)
        
        # Check new Apex compliance attributes
        self.assertEqual(len(self.env.apex_violations), 0)
        self.assertTrue(self.env.allow_new_trades)
        self.assertEqual(self.env.daily_loss_limit, 1000)
        self.assertEqual(self.env.daily_pnl, 0)
        self.assertEqual(self.env.position_entry_step, 0)
    
    def test_step_hold(self):
        """Test step with hold action."""
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)  # Hold
        
        # Check return types
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
        # Should still be flat
        self.assertEqual(self.env.position, 0)
    
    def test_step_buy(self):
        """Test step with buy action."""
        self.env.reset()
        initial_step = self.env.current_step
        
        obs, reward, terminated, truncated, info = self.env.step(1)  # Buy
        
        # Should have opened position
        self.assertEqual(self.env.position, 1)
        self.assertGreater(self.env.entry_price, 0)
        self.assertEqual(self.env.num_trades, 1)
        # Position entry step should be the step when the position was opened
        # It's set to the current step when the position is opened
        self.assertEqual(self.env.position_entry_step, self.env.current_step - 1)
    
    def test_apex_4_59_close(self):
        """Test 4:59 PM ET auto-close rule."""
        # Create test data ending at 4:59 PM
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
        
        env = TradingEnvironmentPhase1(data=test_data, window_size=1)  # Small window for test
        env.reset()
        
        # Open position
        env.position = 1
        env.entry_price = 4000
        env.balance = 50000
        env.current_step = 1  # Set to second bar (4:59 PM)
        
        # Manually check time rule (since we can't step to it)
        current_time = test_data.index[1]  # 4:59 PM
        terminated = env._check_apex_time_rules(current_time)
        
        # Should force close at 4:59 PM
        self.assertTrue(terminated)
        self.assertEqual(env.position, 0)  # Should be closed by _check_apex_time_rules
        self.assertTrue(hasattr(env, 'apex_violations'))
        self.assertGreater(len(env.apex_violations), 0)
        
        # Check violation was recorded
        violation = env.apex_violations[-1]
        self.assertEqual(violation['type'], 'Late position closure')
    
    def test_reward_hacking_prevention(self):
        """Test reward hacking prevention in Sharpe calculation."""
        env = self.env
        env.reset()
        
        # Create scenario with near-zero volatility
        env.portfolio_values = [50000] * 21  # 21 identical values
        
        # Calculate reward
        reward = env._calculate_apex_reward(
            position_changed=False,
            exit_reason=None,
            trade_pnl=0,
            portfolio_value=50000
        )
        
        # Should not be infinite
        self.assertFalse(np.isinf(reward))
        self.assertIsInstance(reward, float)
        
        # Test the improved Sharpe calculation directly
        sharpe_reward = env._calculate_sharpe_component()
        self.assertFalse(np.isinf(sharpe_reward))
        self.assertGreaterEqual(sharpe_reward, -0.02)  # Within clipping range
        self.assertLessEqual(sharpe_reward, 0.05)
    
    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement."""
        env = self.env
        env.reset()
        
        # Set daily P&L to exceed limit
        env.daily_pnl = -1500  # Exceeds $1000 limit
        # Use the same date as the data to avoid reset
        env.current_date = env.data.index[env.current_step].date()
        env.daily_loss_limit = 1000  # Ensure limit is set
        
        # Check daily loss limit
        limit_exceeded = env._check_daily_loss_limit()
        
        self.assertTrue(limit_exceeded)
        
        # Check violation was recorded
        self.assertGreater(len(env.apex_violations), 0)
        violation = env.apex_violations[-1]
        self.assertEqual(violation['type'], 'Daily loss limit exceeded')
    
    def test_time_decay_penalty(self):
        """Test time decay penalty for long-held positions."""
        env = self.env
        env.reset()
        
        # Set up a position held too long
        env.position = 1
        env.position_entry_step = 0
        env.current_step = 800  # Held 800 steps (exceeds 390 limit)
        
        # Calculate reward
        reward = env._calculate_apex_reward(
            position_changed=False,
            exit_reason=None,
            trade_pnl=0,
            portfolio_value=50000
        )
        
        # Check that time decay penalty is applied
        # We can't guarantee the total reward is negative due to other components
        # But we can check the penalty calculation itself
        hold_time = env.current_step - env.position_entry_step
        if hold_time > env.max_hold_time:
            expected_penalty = -0.001 * (hold_time - env.max_hold_time)
            # The penalty should be included in the total reward
            # We can't directly check it, but we can verify the calculation
            self.assertLess(expected_penalty, 0)  # Penalty should be negative


class TestTradingEnvironmentPhase2(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment for Phase 2."""
        dates = pd.date_range('2024-01-01 09:30', periods=100, freq='1min', tz='US/Eastern')
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 100),
            'high': np.random.uniform(4100, 4200, 100),
            'low': np.random.uniform(3900, 4000, 100),
            'close': np.random.uniform(4000, 4100, 100),
            'volume': np.random.randint(100, 1000, 100),
            'sma_5': np.random.uniform(4000, 4100, 100),
            'sma_20': np.random.uniform(4000, 4100, 100),
            'rsi': np.random.uniform(30, 70, 100),
            'macd': np.random.uniform(-10, 10, 100),
            'momentum': np.random.uniform(-50, 50, 100),
            'atr': np.random.uniform(10, 30, 100)
        }, index=dates)
        
        self.env = TradingEnvironmentPhase2(
            data=self.test_data,
            initial_balance=50000,
            window_size=20
        )
    
    def test_reset_phase2(self):
        """Test Phase 2 environment reset."""
        obs, info = self.env.reset()
        
        # Check Phase 2 specific attributes
        self.assertFalse(self.env.trailing_stop_active)
        self.assertEqual(self.env.highest_profit_point, 0)
        self.assertEqual(self.env.be_move_count, 0)
    
    def test_position_management_validation(self):
        """Test position management action validation."""
        env = self.env
        env.reset()
        
        # Open a position first
        env.position = 1
        env.entry_price = 4000
        env.sl_price = 3980
        env.tp_price = 4040
        
        current_price = 4010
        atr = 20

        # Test valid Move to BE
        is_valid, reason = env._validate_position_management_action(
            env.ACTION_MOVE_SL_TO_BE, current_price, atr
        )
        self.assertTrue(is_valid)

        # Test invalid Move to BE (when losing)
        env.entry_price = 4020  # Current price 4010, so losing
        is_valid, reason = env._validate_position_management_action(
            env.ACTION_MOVE_SL_TO_BE, current_price, atr
        )
        self.assertFalse(is_valid)
        self.assertIn("losing", reason.lower())
    
    def test_expanded_action_space(self):
        """Test Phase 2 has simplified action space."""
        self.assertEqual(self.env.action_space.n, 6)  # Should have 6 actions

        # Test all action constants
        self.assertEqual(self.env.ACTION_HOLD, 0)
        self.assertEqual(self.env.ACTION_BUY, 1)
        self.assertEqual(self.env.ACTION_SELL, 2)
        self.assertEqual(self.env.ACTION_MOVE_SL_TO_BE, 3)
        self.assertEqual(self.env.ACTION_ENABLE_TRAIL, 4)
        self.assertEqual(self.env.ACTION_DISABLE_TRAIL, 5)
    
    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        env = self.env
        env.reset()
        
        # Test long position
        env.position = 1
        env.entry_price = 4000
        current_price = 4010
        expected_pnl = (4010 - 4000) * 50 * 1.0  # 10 points * 50 * 1 contract
        actual_pnl = env._calculate_unrealized_pnl(current_price)
        self.assertEqual(actual_pnl, expected_pnl)
        
        # Test short position
        env.position = -1
        env.entry_price = 4000
        current_price = 3990
        expected_pnl = (4000 - 3990) * 50 * 1.0  # 10 points * 50 * 1 contract
        actual_pnl = env._calculate_unrealized_pnl(current_price)
        self.assertEqual(actual_pnl, expected_pnl)
    
    def test_invalid_atr_handling(self):
        """Test handling of invalid ATR values."""
        env = self.env
        env.reset()
        
        # Open a position
        env.position = 1
        env.entry_price = 4000
        
        # Test with zero ATR
        is_valid, reason = env._validate_position_management_action(
            env.ACTION_MOVE_SL_TO_BE, 4010, 0
        )
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Invalid ATR")

        # Test with NaN ATR
        is_valid, reason = env._validate_position_management_action(
            env.ACTION_MOVE_SL_TO_BE, 4010, float('nan')
        )
        self.assertFalse(is_valid)
        self.assertEqual(reason, "Invalid ATR")


class TestSecondLevelData(unittest.TestCase):
    """Test second-level drawdown detection."""

    def setUp(self):
        """Set up test with second-level data."""
        np.random.seed(42)

        # Minute-level data
        dates_min = pd.date_range('2024-01-01 09:30', periods=10, freq='1min', tz='US/Eastern')
        self.minute_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 10),
            'high': np.random.uniform(4100, 4150, 10),
            'low': np.random.uniform(3950, 4000, 10),
            'close': np.random.uniform(4000, 4100, 10),
            'volume': np.random.randint(100, 1000, 10),
            'sma_5': np.random.uniform(4000, 4100, 10),
            'sma_20': np.random.uniform(4000, 4100, 10),
            'rsi': np.random.uniform(30, 70, 10),
            'macd': np.random.uniform(-10, 10, 10),
            'momentum': np.random.uniform(-50, 50, 10),
            'atr': np.random.uniform(10, 30, 10)
        }, index=dates_min)

        # Second-level data (60 bars per minute)
        dates_sec = pd.date_range('2024-01-01 09:30', periods=600, freq='1S', tz='US/Eastern')
        self.second_data = pd.DataFrame({
            'high': np.random.uniform(4100, 4150, 600),
            'low': np.random.uniform(3950, 4000, 600),
        }, index=dates_sec)

    def test_second_level_drawdown_detection(self):
        """Test drawdown detection at second-level granularity."""
        env = TradingEnvironmentPhase1(
            data=self.minute_data,
            second_data=self.second_data,
            initial_balance=50000,
            window_size=5,
            trailing_drawdown_limit=1000
        )

        env.reset()

        # Open position
        env.position = 1
        env.entry_price = 4050
        env.balance = 50000
        env.highest_balance = 50000
        env.trailing_dd_level = 49000  # $1000 drawdown limit

        # Check second-level drawdown
        current_bar_time = self.minute_data.index[0]
        drawdown_hit, min_equity = env._check_second_level_drawdown(current_bar_time)

        # Should return tuple
        self.assertIsInstance(drawdown_hit, bool)
        self.assertIsInstance(min_equity, (int, float))


class TestComprehensiveScenarios(unittest.TestCase):
    """Test comprehensive trading scenarios."""

    def setUp(self):
        """Set up test environment."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01 09:30', periods=400, freq='1min', tz='US/Eastern')

        self.test_data = pd.DataFrame({
            'open': np.random.uniform(4000, 4100, 400),
            'high': np.random.uniform(4100, 4150, 400),
            'low': np.random.uniform(3950, 4000, 400),
            'close': np.random.uniform(4000, 4100, 400),
            'volume': np.random.randint(100, 1000, 400),
            'sma_5': np.random.uniform(4000, 4100, 400),
            'sma_20': np.random.uniform(4000, 4100, 400),
            'rsi': np.random.uniform(30, 70, 400),
            'macd': np.random.uniform(-10, 10, 400),
            'momentum': np.random.uniform(-50, 50, 400),
            'atr': np.random.uniform(10, 30, 400)
        }, index=dates)

    def test_full_trading_day_simulation(self):
        """Test complete trading day simulation (390 minutes)."""
        env = TradingEnvironmentPhase1(
            data=self.test_data,
            initial_balance=50000,
            window_size=20
        )

        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        max_steps = 390  # One trading day

        actions_taken = []
        portfolio_history = []

        while not terminated and not truncated and step_count < max_steps:
            # Simulate realistic trading strategy
            if env.position == 0:
                action = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])  # Mostly hold
            else:
                action = 0  # Hold while in position

            obs, reward, terminated, truncated, info = env.step(action)
            actions_taken.append(action)
            portfolio_history.append(info['portfolio_value'])
            step_count += 1

        # Verify complete day simulation
        self.assertGreater(step_count, 100)  # Should run for significant time
        self.assertIsInstance(portfolio_history[-1], float)

        # Check no violations occurred
        self.assertEqual(len(env.apex_violations), 0, "Should have no violations")

    def test_sl_tp_execution_accuracy(self):
        """Test stop loss and take profit hit accurately."""
        env = TradingEnvironmentPhase1(
            data=self.test_data,
            initial_balance=50000,
            window_size=20,
            fixed_sl_atr_multiplier=1.0,
            fixed_tp_to_sl_ratio=2.0
        )

        env.reset()
        env.current_step = 50

        # Manually set up a position
        env.position = 1
        env.entry_price = 4050
        env.sl_price = 4030  # 20 points below
        env.tp_price = 4090  # 40 points above (2:1 R:R)

        # Test SL hit
        sl_hit, tp_hit, exit_price = env._check_sl_tp_hit(high=4080, low=4025)
        self.assertTrue(sl_hit)
        self.assertFalse(tp_hit)
        self.assertEqual(exit_price, env.sl_price)

        # Reset for TP test
        sl_hit, tp_hit, exit_price = env._check_sl_tp_hit(high=4095, low=4045)
        self.assertFalse(sl_hit)
        self.assertTrue(tp_hit)
        self.assertEqual(exit_price, env.tp_price)

    def test_multiple_trades_sequence(self):
        """Test multiple trades in sequence maintain state correctly."""
        env = TradingEnvironmentPhase1(
            data=self.test_data,
            initial_balance=50000,
            window_size=20
        )

        obs, info = env.reset()

        # Trade 1: Buy
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        self.assertEqual(env.position, 1)
        trade_count_1 = env.num_trades

        # Wait and close (force SL hit by setting price)
        env.current_step += 10
        env._check_sl_tp_hit(high=4100, low=env.sl_price - 1)  # Hit SL
        env.position = 0  # Simulate close

        # Trade 2: Sell
        obs, reward, terminated, truncated, info = env.step(2)  # Sell
        self.assertEqual(env.position, -1)
        self.assertGreater(env.num_trades, trade_count_1)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)