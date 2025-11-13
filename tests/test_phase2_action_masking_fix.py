"""
Test Phase 2 Action Masking Enhancement (RL FIX #11)

Verifies that ENABLE_TRAIL action requires SL to be at or past break-even first.
This prevents suboptimal risk management sequences.

Test Date: November 10, 2025
Related: PHASE2_ACTION_MASKING_ISSUE.md
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from environment_phase2 import TradingEnvironmentPhase2


def create_test_data(direction='up', start_price=4000, num_bars=100):
    """
    Create test market data for action masking tests.

    Args:
        direction: 'up' for rising prices, 'down' for falling prices
        start_price: Starting price level
        num_bars: Number of bars to generate

    Returns:
        pd.DataFrame: Test market data with proper timezone

    Note: Uses minimal wicks (0.1 points) to avoid hitting BE stops during tests
    """
    dates = pd.date_range(
        '2024-01-02 10:00',
        periods=num_bars,
        freq='1min',
        tz='America/New_York'
    )

    if direction == 'up':
        close_prices = np.linspace(start_price, start_price + 50, num_bars)
    else:
        close_prices = np.linspace(start_price, start_price - 50, num_bars)

    # Use minimal wicks (0.1 points) to avoid hitting BE stops
    data = pd.DataFrame({
        'close': close_prices,
        'open': close_prices,
        'high': close_prices + 0.1,  # Minimal wick up
        'low': close_prices - 0.1,   # Minimal wick down (won't hit BE stop)
        'volume': np.full(num_bars, 1000.0),
        'sma_5': close_prices,
        'sma_20': close_prices,
        'rsi': np.full(num_bars, 50.0),
        'macd': np.full(num_bars, 0.0),
        'momentum': np.full(num_bars, 0.0),
        'atr': np.full(num_bars, 10.0)
    }, index=dates)

    return data


class TestPhase2ActionMaskingEnhancement:
    """Test suite for RL FIX #11 - Enhanced action masking dependencies."""

    def test_trailing_blocked_before_breakeven_long(self):
        """
        Test that ENABLE_TRAIL is blocked when SL is NOT at break-even (LONG).

        Expected behavior:
        1. Open long position
        2. Price moves up (profitable)
        3. MOVE_SL_TO_BE is valid
        4. ENABLE_TRAIL is INVALID (SL not at BE yet)
        """
        data = create_test_data(direction='up', start_price=4000)
        env = TradingEnvironmentPhase2(data=data, window_size=20)
        env.reset()

        # Open long position
        obs, reward, term, trunc, info = env.step(1)  # BUY
        assert env.position == 1, "Position should be long"

        initial_sl = env.sl_price
        entry = env.entry_price

        print(f"\nLONG Position Test:")
        print(f"  Entry: ${entry:.2f}, SL: ${initial_sl:.2f}")

        # Advance to profitable state
        for _ in range(5):
            env.step(0)  # HOLD

        current_price = env.data['close'].iloc[env.current_step]
        unrealized = (current_price - entry) * env.contract_size

        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Unrealized P&L: ${unrealized:.2f}")
        print(f"  SL Position: ${env.sl_price:.2f} (entry: ${entry:.2f})")

        # Get action mask
        mask = env.action_masks()

        print(f"\nAction Mask (SL NOT at BE):")
        print(f"  [0] HOLD:           {mask[0]}")
        print(f"  [1] BUY:            {mask[1]}")
        print(f"  [2] SELL:           {mask[2]}")
        print(f"  [3] MOVE_SL_TO_BE:  {mask[3]}  <- Should be TRUE")
        print(f"  [4] ENABLE_TRAIL:   {mask[4]}  <- Should be FALSE (RL FIX #11)")
        print(f"  [5] DISABLE_TRAIL:  {mask[5]}")

        # Verify MOVE_SL_TO_BE is valid
        assert mask[3] == True, (
            "MOVE_SL_TO_BE should be valid when profitable and SL not at BE"
        )

        # RL FIX #11: Verify ENABLE_TRAIL is BLOCKED (SL not at BE yet)
        assert mask[4] == False, (
            "ENABLE_TRAIL should be BLOCKED when SL is not at break-even yet! "
            "This is RL FIX #11 - must move to BE before trailing."
        )

        print("\n✅ PASS: ENABLE_TRAIL correctly blocked before break-even (LONG)")

    def test_trailing_allowed_after_breakeven_long(self):
        """
        Test that ENABLE_TRAIL is allowed AFTER SL moved to break-even (LONG).

        Expected behavior:
        1. Open long position
        2. Price moves up (profitable)
        3. Move SL to break-even
        4. ENABLE_TRAIL is now VALID
        5. MOVE_SL_TO_BE is now INVALID (already at BE)
        """
        data = create_test_data(direction='up', start_price=4000)
        env = TradingEnvironmentPhase2(data=data, window_size=20)
        env.reset()

        # Open long position
        env.step(1)  # BUY
        entry = env.entry_price

        # Advance to profitable state
        for _ in range(5):
            env.step(0)  # HOLD

        print(f"\nLONG Position After BE Test:")
        print(f"  Entry: ${entry:.2f}, SL: ${env.sl_price:.2f}")

        # Move SL to break-even
        obs, reward, term, trunc, info = env.step(3)  # MOVE_SL_TO_BE

        print(f"  SL After BE Move: ${env.sl_price:.2f}")
        assert env.sl_price >= entry, "SL should be at or past entry after BE move"

        # Get action mask after BE move
        mask = env.action_masks()

        current_price = env.data['close'].iloc[env.current_step]
        print(f"  Current Price: ${current_price:.2f}")

        print(f"\nAction Mask (SL AT BE):")
        print(f"  [0] HOLD:           {mask[0]}")
        print(f"  [1] BUY:            {mask[1]}")
        print(f"  [2] SELL:           {mask[2]}")
        print(f"  [3] MOVE_SL_TO_BE:  {mask[3]}  <- Should be FALSE (already at BE)")
        print(f"  [4] ENABLE_TRAIL:   {mask[4]}  <- Should be TRUE (RL FIX #11)")
        print(f"  [5] DISABLE_TRAIL:  {mask[5]}")

        # Verify MOVE_SL_TO_BE is now INVALID (already at BE)
        assert mask[3] == False, (
            "MOVE_SL_TO_BE should be invalid when SL already at break-even"
        )

        # RL FIX #11: Verify ENABLE_TRAIL is now ALLOWED
        assert mask[4] == True, (
            "ENABLE_TRAIL should be ALLOWED after SL moved to break-even! "
            "This is RL FIX #11 - logical sequence enforced: BE → Trail."
        )

        print("\n✅ PASS: ENABLE_TRAIL correctly allowed after break-even (LONG)")

    def test_trailing_blocked_before_breakeven_short(self):
        """
        Test that ENABLE_TRAIL is blocked when SL is NOT at break-even (SHORT).
        """
        data = create_test_data(direction='down', start_price=4050)
        env = TradingEnvironmentPhase2(data=data, window_size=20)
        env.reset()

        # Open short position
        env.step(2)  # SELL
        assert env.position == -1, "Position should be short"

        initial_sl = env.sl_price
        entry = env.entry_price

        print(f"\nSHORT Position Test:")
        print(f"  Entry: ${entry:.2f}, SL: ${initial_sl:.2f}")

        # Advance to profitable state
        for _ in range(5):
            env.step(0)  # HOLD

        current_price = env.data['close'].iloc[env.current_step]
        unrealized = (entry - current_price) * env.contract_size

        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Unrealized P&L: ${unrealized:.2f}")

        # Get action mask
        mask = env.action_masks()

        print(f"\nAction Mask (SHORT, SL NOT at BE):")
        print(f"  [3] MOVE_SL_TO_BE:  {mask[3]}  <- Should be TRUE")
        print(f"  [4] ENABLE_TRAIL:   {mask[4]}  <- Should be FALSE (RL FIX #11)")

        # Verify masking
        assert mask[3] == True, "MOVE_SL_TO_BE should be valid"
        assert mask[4] == False, (
            "ENABLE_TRAIL should be BLOCKED when SL not at BE (SHORT)"
        )

        print("\n✅ PASS: ENABLE_TRAIL correctly blocked before break-even (SHORT)")

    def test_trailing_allowed_after_breakeven_short(self):
        """
        Test that ENABLE_TRAIL is allowed AFTER SL moved to break-even (SHORT).
        """
        data = create_test_data(direction='down', start_price=4050)
        env = TradingEnvironmentPhase2(data=data, window_size=20)
        env.reset()

        # Open short position
        env.step(2)  # SELL
        entry = env.entry_price

        # Advance to profitable state
        for _ in range(5):
            env.step(0)  # HOLD

        print(f"\nSHORT Position After BE Test:")
        print(f"  Entry: ${entry:.2f}, SL: ${env.sl_price:.2f}")

        # Move SL to break-even
        env.step(3)  # MOVE_SL_TO_BE

        print(f"  SL After BE Move: ${env.sl_price:.2f}")
        assert env.sl_price <= entry, "SL should be at or below entry for short"

        # Get action mask
        mask = env.action_masks()

        print(f"\nAction Mask (SHORT, SL AT BE):")
        print(f"  [3] MOVE_SL_TO_BE:  {mask[3]}  <- Should be FALSE")
        print(f"  [4] ENABLE_TRAIL:   {mask[4]}  <- Should be TRUE (RL FIX #11)")

        # Verify masking
        assert mask[3] == False, "MOVE_SL_TO_BE should be invalid (already at BE)"
        assert mask[4] == True, (
            "ENABLE_TRAIL should be ALLOWED after BE move (SHORT)"
        )

        print("\n✅ PASS: ENABLE_TRAIL correctly allowed after break-even (SHORT)")

    def test_disable_trail_only_when_active(self):
        """
        Test that DISABLE_TRAIL is only valid when trailing is active.
        """
        data = create_test_data(direction='up', start_price=4000)
        env = TradingEnvironmentPhase2(data=data, window_size=20)
        env.reset()

        # Open long position
        env.step(1)  # BUY

        # Advance to profitable state
        for _ in range(5):
            env.step(0)

        # Before enabling trailing
        mask = env.action_masks()
        print(f"\nDISABLE_TRAIL Test (before enabling):")
        print(f"  Trailing Active: {env.trailing_stop_active}")
        print(f"  [5] DISABLE_TRAIL: {mask[5]}  <- Should be FALSE")

        assert mask[5] == False, "DISABLE_TRAIL should be invalid when not active"

        # Move to BE
        env.step(3)  # MOVE_SL_TO_BE

        # Enable trailing
        env.step(4)  # ENABLE_TRAIL

        # After enabling trailing
        mask = env.action_masks()
        print(f"\nDISABLE_TRAIL Test (after enabling):")
        print(f"  Trailing Active: {env.trailing_stop_active}")
        print(f"  [5] DISABLE_TRAIL: {mask[5]}  <- Should be TRUE")

        assert mask[5] == True, "DISABLE_TRAIL should be valid when trailing active"

        print("\n✅ PASS: DISABLE_TRAIL correctly gated by trailing state")

    def test_complete_position_management_sequence(self):
        """
        Test complete logical sequence: Entry → Profitable → BE → Trail → Disable.

        This verifies the entire position management flow with RL FIX #11.
        """
        data = create_test_data(direction='up', start_price=4000)
        env = TradingEnvironmentPhase2(data=data, window_size=20)
        env.reset()

        print(f"\n{'='*60}")
        print("COMPLETE POSITION MANAGEMENT SEQUENCE TEST")
        print(f"{'='*60}")

        # Step 1: Open position
        env.step(1)  # BUY
        print(f"\n1. Position Opened:")
        print(f"   Entry: ${env.entry_price:.2f}, SL: ${env.sl_price:.2f}")

        # Step 2: Wait for profit
        for _ in range(5):
            env.step(0)

        mask = env.action_masks()
        print(f"\n2. Profitable (SL NOT at BE):")
        print(f"   Current: ${env.data['close'].iloc[env.current_step]:.2f}")
        print(f"   MOVE_SL_TO_BE: {mask[3]} (should be TRUE)")
        print(f"   ENABLE_TRAIL:  {mask[4]} (should be FALSE - RL FIX #11)")

        assert mask[3] == True
        assert mask[4] == False

        # Step 3: Move to break-even
        env.step(3)  # MOVE_SL_TO_BE
        print(f"\n3. Moved to Break-Even:")
        print(f"   SL: ${env.sl_price:.2f} (entry: ${env.entry_price:.2f})")

        mask = env.action_masks()
        print(f"   MOVE_SL_TO_BE: {mask[3]} (should be FALSE - already at BE)")
        print(f"   ENABLE_TRAIL:  {mask[4]} (should be TRUE - RL FIX #11)")

        assert mask[3] == False
        assert mask[4] == True

        # Step 4: Enable trailing
        env.step(4)  # ENABLE_TRAIL
        print(f"\n4. Trailing Enabled:")
        print(f"   Trailing Active: {env.trailing_stop_active}")

        mask = env.action_masks()
        print(f"   ENABLE_TRAIL:   {mask[4]} (should be FALSE - already enabled)")
        print(f"   DISABLE_TRAIL:  {mask[5]} (should be TRUE)")

        assert mask[4] == False  # Already enabled
        assert mask[5] == True   # Can disable

        # Step 5: Disable trailing
        env.step(5)  # DISABLE_TRAIL
        print(f"\n5. Trailing Disabled:")
        print(f"   Trailing Active: {env.trailing_stop_active}")

        mask = env.action_masks()
        print(f"   ENABLE_TRAIL:   {mask[4]} (should be TRUE - can re-enable)")
        print(f"   DISABLE_TRAIL:  {mask[5]} (should be FALSE - not active)")

        assert mask[4] == True   # Can re-enable
        assert mask[5] == False  # Can't disable (not active)

        print(f"\n{'='*60}")
        print("✅ COMPLETE SEQUENCE TEST PASSED")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
