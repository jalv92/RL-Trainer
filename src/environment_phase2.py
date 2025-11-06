"""
Phase 2: Advanced Position Management Environment

Goal: Learn HOW to dynamically manage risk and optimize positions
- Actions: 8 discrete (Hold, Buy, Sell, Close, Tighten SL, Move to BE, Extend TP, Toggle Trail)
- SL/TP: DYNAMIC (learned & adaptive)
- Focus: Risk management, position optimization, Sharpe ratio
- Reward: Risk-adjusted returns, max drawdown minimization, consistency

Based on: OpenAI Spinning Up PPO + Transfer Learning + Advanced RL
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from src.environment_phase1 import TradingEnvironmentPhase1
from src.market_specs import MarketSpecification


class TradingEnvironmentPhase2(TradingEnvironmentPhase1):
    """
    Phase 2: Advanced Position Management

    RL FIX #10: Simplified action space from 9 to 6 actions

    Observation Space: (window_size*11 + 5,) - Market features + position state
    Action Space: Discrete(6) - Streamlined position management
    Reward: Risk-adjusted returns, Sharpe ratio, consistency
    """

    # Simplified action space
    # RL FIX #10: Reduced from 9 to 6 actions for better sample efficiency
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    ACTION_MOVE_SL_TO_BE = 3  # Was ACTION 5
    ACTION_ENABLE_TRAIL = 4   # Was ACTION 7
    ACTION_DISABLE_TRAIL = 5  # Was ACTION 8

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 50000,
        window_size: int = 20,
        second_data: pd.DataFrame = None,
        # Market specifications
        market_spec: MarketSpecification = None,
        commission_override: float = None,
        # Phase 2 specific
        initial_sl_multiplier: float = 1.5,
        initial_tp_ratio: float = 3.0,
        position_size_contracts: float = 1.0,  # Full size in Phase 2
        trailing_drawdown_limit: float = 2500,  # Strict Apex rules
        tighten_sl_step: float = 0.5,  # How much to tighten SL (in ATR)
        extend_tp_step: float = 1.0,   # How much to extend TP (in ATR)
        trailing_activation_profit: float = 1.0  # Activate trail after 1R profit
    ):
        """
        Initialize Phase 2 environment with position management.

        Args:
            data: OHLCV + indicators DataFrame
            initial_balance: Starting capital
            window_size: Lookback window
            second_data: Optional second-level data
            market_spec: Market specification object (defaults to ES if None)
            commission_override: Override default commission
            initial_sl_multiplier: Initial SL distance (can be adjusted)
            initial_tp_ratio: Initial TP ratio (can be adjusted)
            position_size_contracts: Full position size (1.0 for Apex)
            trailing_drawdown_limit: Strict Apex limit ($2,500)
            tighten_sl_step: Amount to tighten SL by (in ATR units)
            extend_tp_step: Amount to extend TP by (in ATR units)
            trailing_activation_profit: Min profit (in R) to enable trailing
        """
        # Initialize parent class (Phase 1) with market specs
        super().__init__(
            data, initial_balance, window_size, second_data,
            market_spec, commission_override,  # Pass market specs
            initial_sl_multiplier, initial_tp_ratio, position_size_contracts,
            trailing_drawdown_limit
        )

        # Phase 2: Simplified action space (3 -> 6 actions)
        # RL FIX #10: Reduced from 9 to 6 for improved sample efficiency
        self.action_space = spaces.Discrete(6)

        # Phase 2 position management parameters
        self.tighten_step_atr = tighten_sl_step
        self.extend_step_atr = extend_tp_step
        self.trail_activation_r = trailing_activation_profit

        # NEW: Dynamic position sizing parameters
        self.max_position_size = position_size_contracts
        self.position_scaling_enabled = True
        self.volatility_adjustment = True

        # Position management state
        self.trailing_stop_active = False
        self.highest_profit_point = 0
        self.be_move_count = 0

        # Action diversity tracking (fix sell bias)
        self.buy_count = 0
        self.sell_count = 0
        self.action_diversity_window = 100  # Track last 100 actions

        # Diagnostics
        self.last_action_mask = None
        self.last_done_reason = None


    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset with Phase 2 tracking."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset position management state
        self.trailing_stop_active = False
        self.highest_profit_point = 0
        self.be_move_count = 0

        # Action diversity tracking (fix sell bias)
        self.buy_count = 0
        self.sell_count = 0
        self.action_diversity_window = 100  # Track last 100 actions

        # Diagnostics
        self.last_action_mask = None
        self.last_done_reason = None

        if info is None:
            info = {}

        info.update({
            'data_length': len(self.data),
            'window_size': self.window_size,
            'market_symbol': getattr(self, 'market_symbol', None),
        })

        # Phase 2 specific position management parameters are already set in __init__
        # No need to re-assign them here

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action with Phase 2 position management.

        RL FIX #10: Simplified action space (9 -> 6) for better sample efficiency

        Actions:
            0: Hold
            1: Buy (open long)
            2: Sell (open short)
            3: Move SL to break-even (was 5)
            4: Enable trailing stop (was 7)
            5: Disable trailing stop (was 8)
        """
        if self.current_step >= len(self.data) - 1:
            obs = self._get_observation()
            info = {
                'portfolio_value': self.balance,
                'position': self.position,
                'num_trades': self.num_trades,
                'balance': self.balance,
                'done_reason': 'data_exhausted',
                'action_mask': self._last_action_mask_as_list(),
                'observation_quality': self._observation_quality(obs),
                'step_index': self.current_step,
                'remaining_bars': 0,
            }
            self.last_done_reason = 'data_exhausted'
            return obs, 0.0, False, True, info

        current_price = self.data['close'].iloc[self.current_step]
        high = self.data['high'].iloc[self.current_step]
        low = self.data['low'].iloc[self.current_step]
        atr = self.data['atr'].iloc[self.current_step]

        reward = 0.0
        terminated = False
        truncated = False
        done_reason = None
        position_changed = False
        trade_pnl = 0.0
        exit_reason = None
        pm_action_taken = None  # Position management action

        # ============================================================
        # 1. UPDATE TRAILING STOP (if active)
        # ============================================================
        if self.trailing_stop_active and self.position != 0:
            self._update_trailing_stop(current_price, atr)

        # ============================================================
        # 2. CHECK SL/TP (now potentially dynamic)
        # ============================================================
        if self.position != 0:
            sl_hit, tp_hit, exit_price = self._check_sl_tp_hit(high, low)

            if sl_hit or tp_hit:
                # Close position
                if self.position == 1:
                    trade_pnl = (exit_price - self.entry_price) * self.contract_size * self.position_size
                else:
                    trade_pnl = (self.entry_price - exit_price) * self.contract_size * self.position_size

                trade_pnl -= (self.commission_per_side * 2 * self.position_size)
                self.balance += trade_pnl
                exit_reason = "take_profit" if tp_hit else "stop_loss"

                if trade_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                self.trade_history.append({
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': trade_pnl,
                    'be_moves': self.be_move_count,
                    'trailing_used': self.trailing_stop_active,
                    'step': self.current_step
                })

                # Reset
                self._reset_position_state()
                position_changed = True

        # ============================================================
        # 3. EXECUTE ACTION
        # ============================================================
        current_time = self.data.index[self.current_step]

        # RTH gating
        if current_time.tzinfo is None:
            ts = current_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = current_time.tz_convert('America/New_York')

        rth_open = ts.replace(hour=9, minute=30, second=0)
        rth_close = ts.replace(hour=16, minute=59, second=0)
        allowed_to_open = (ts >= rth_open) and (ts <= rth_close)

        # PHASE 1 ACTIONS (Hold, Buy, Sell)
        if action == self.ACTION_BUY and self.position == 0 and allowed_to_open:
            # Open long with dynamic position sizing
            self.position = 1
            self.entry_price = current_price + self.slippage_points
            self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(self.entry_price, 1)

            # NEW: Calculate dynamic position size based on volatility
            self.position_size = self._calculate_position_size()

            self.balance -= (self.commission_per_side * self.position_size)
            self.num_trades += 1
            self.buy_count += 1  # Track for diversity
            position_changed = True

        elif action == self.ACTION_SELL and self.position == 0 and allowed_to_open:
            # Open short with dynamic position sizing
            self.position = -1
            self.entry_price = current_price - self.slippage_points
            self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(self.entry_price, -1)

            # NEW: Calculate dynamic position size based on volatility
            self.position_size = self._calculate_position_size()

            self.balance -= (self.commission_per_side * self.position_size)
            self.num_trades += 1
            self.sell_count += 1  # Track for diversity
            # ❌ REMOVED BUG: self.buy_count += 1  # This was a copy-paste error
            position_changed = True

        # PHASE 2 ACTIONS (Position Management)
        elif action == self.ACTION_MOVE_SL_TO_BE and self.position != 0:
            is_valid, reason = self._validate_position_management_action(
                action, current_price, atr
            )
            if is_valid:
                success = self._move_sl_to_breakeven(current_price)
                if success:
                    pm_action_taken = "move_to_be"
                    self.be_move_count += 1
            else:
                reward -= 0.01
                pm_action_taken = f"invalid_move_to_be: {reason}"

        elif action == self.ACTION_ENABLE_TRAIL and self.position != 0:
            is_valid, reason = self._validate_position_management_action(
                action, current_price, atr
            )
            if is_valid:
                if not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    pm_action_taken = "trail_enabled"
                else:
                    # Already enabled - no-op but not an error
                    pm_action_taken = "trail_already_on"
            else:
                reward -= 0.01
                pm_action_taken = f"invalid_enable_trail: {reason}"

        elif action == self.ACTION_DISABLE_TRAIL and self.position != 0:
            is_valid, reason = self._validate_position_management_action(
                action, current_price, atr
            )
            if is_valid:
                if self.trailing_stop_active:
                    self.trailing_stop_active = False
                    pm_action_taken = "trail_disabled"
                else:
                    # Already disabled - no-op but not an error
                    pm_action_taken = "trail_already_off"
            else:
                reward -= 0.01
                pm_action_taken = f"invalid_disable_trail: {reason}"

        # ============================================================
        # 4. UPDATE PORTFOLIO & DRAWDOWN
        # ============================================================
        unrealized_pnl = 0.0
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.contract_size * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.contract_size * self.position_size

            # Track highest profit for trailing
            if unrealized_pnl > self.highest_profit_point:
                self.highest_profit_point = unrealized_pnl

        portfolio_value = self.balance + unrealized_pnl
        self.portfolio_values.append(portfolio_value)

        # APEX RITHMIC RULE: Stop trailing when profit target reached
        if portfolio_value >= self.profit_target and not self.profit_target_reached:
            self.profit_target_reached = True
            self.profit_target_reached_step = self.current_step
            self.trailing_stopped = True

        # Update trailing drawdown (Apex rules)
        if portfolio_value > self.highest_balance:
            self.highest_balance = portfolio_value

            if not self.trailing_stopped:
                # Normal trailing behavior (before profit target)
                self.trailing_dd_level = self.highest_balance - self.trailing_dd_limit
            # else: trailing_dd_level remains frozen at profit target level

        # Track trailing DD level history for compliance checking
        self.trailing_dd_levels.append(self.trailing_dd_level)

        # Check violation at minute-level
        if portfolio_value < self.trailing_dd_level:
            terminated = True
            reward = -0.1
            done_reason = 'minute_trailing_drawdown'

        # Check violation at second-level (Apex compliance)
        if not terminated and self.second_data is not None:
            current_bar_time = self.data.index[self.current_step]
            drawdown_hit, _ = self._check_second_level_drawdown(current_bar_time)
            if drawdown_hit:
                terminated = True
                reward = -0.1  # Heavy penalty for Apex violation
                done_reason = 'second_level_trailing_drawdown'

        # ============================================================
        # 5. CALCULATE REWARD (Apex-Optimized)
        # ============================================================
        if not terminated:
            reward = self._calculate_apex_reward(
                position_changed, exit_reason, trade_pnl,
                portfolio_value, pm_action_taken
            )

        # ============================================================
        # 6. ADVANCE TIME
        # ============================================================
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            truncated = True
            if done_reason is None:
                done_reason = 'end_of_data'

        obs = self._get_observation()
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'pm_action': pm_action_taken,
            'be_moves': self.be_move_count,
            'num_trades': self.num_trades,
            'balance': self.balance,
            'unrealized_pnl': unrealized_pnl,
            'action_mask': self._last_action_mask_as_list(),
            'observation_quality': self._observation_quality(obs),
            'step_index': self.current_step,
            'prev_step_index': self.current_step - 1,
            'remaining_bars': max(len(self.data) - self.current_step, 0),
            'bar_timestamp': str(self.data.index[min(self.current_step - 1, len(self.data) - 1)]),
            'done_reason': done_reason,
        }

        if done_reason:
            self.last_done_reason = done_reason

        return obs, reward, terminated, truncated, info

    def _last_action_mask_as_list(self):
        if self.last_action_mask is None:
            return None
        return self.last_action_mask.astype(bool).tolist()

    def _observation_quality(self, obs: np.ndarray) -> Dict[str, float]:
        if obs is None:
            return {
                'has_nan': False,
                'has_inf': False,
                'min': float('nan'),
                'max': float('nan'),
            }

        obs = np.asarray(obs)
        has_nan = bool(np.isnan(obs).any())
        has_inf = bool(np.isinf(obs).any())
        finite_mask = np.isfinite(obs)
        if finite_mask.any():
            finite_vals = obs[finite_mask]
            min_val = float(finite_vals.min())
            max_val = float(finite_vals.max())
        else:
            min_val = float('nan')
            max_val = float('nan')

        return {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'min': min_val,
            'max': max_val,
        }

    def _move_sl_to_breakeven(self, current_price: float) -> bool:
        """
        Move stop loss to entry price (break-even).
        Only allowed if profitable.

        Returns:
            True if SL was moved to BE
        """
        if self.position == 0:
            return False

        unrealized = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)

        # Only move to BE if profitable
        if unrealized <= 0:
            return False

        # Check if already at or past break-even
        if self.position == 1 and self.sl_price >= self.entry_price:
            return False
        if self.position == -1 and self.sl_price <= self.entry_price:
            return False

        # Add small buffer (commission coverage)
        buffer = 0.25

        if self.position == 1:
            self.sl_price = self.entry_price + buffer
        else:
            self.sl_price = self.entry_price - buffer

        return True

    def _update_trailing_stop(self, current_price: float, atr: float):
        """
        Update trailing stop if profit increases.
        Trails by 1× ATR behind current price.
        """
        if self.position == 0 or not self.trailing_stop_active or atr <= 0:
            return

        unrealized = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
        unrealized_pnl = unrealized * self.contract_size * self.position_size

        if unrealized_pnl > self.highest_profit_point:
            # New high - move SL up
            trail_distance = atr * 1.0  # Trail 1 ATR behind

            if self.position == 1:
                new_sl = current_price - trail_distance
                self.sl_price = max(self.sl_price, new_sl)
            else:
                new_sl = current_price + trail_distance
                self.sl_price = min(self.sl_price, new_sl)

    def _calculate_position_size(self):
        """
        Calculate dynamic position size based on volatility.

        Reduces position size in high volatility environments to manage risk.
        This implements volatility-adjusted position sizing from the improvement plan.

        Returns:
            float: Adjusted position size (between 0 and max_position_size)
        """
        if not self.position_scaling_enabled:
            return self.max_position_size

        # Get current ATR and average ATR
        current_atr = self.data['atr'].iloc[self.current_step]
        avg_atr = self.data['atr'].rolling(50, min_periods=10).mean().iloc[self.current_step]

        # Handle edge cases
        if np.isnan(current_atr) or np.isnan(avg_atr) or avg_atr <= 0:
            return self.max_position_size

        if self.volatility_adjustment:
            # Calculate volatility ratio
            vol_ratio = current_atr / avg_atr

            # Reduce size in high volatility (vol_ratio > 1)
            # When volatility is 2x normal, size becomes 0.5x max
            # Formula: size = max_size / (1 + vol_ratio)
            size_multiplier = 1.0 / (1.0 + max(0, vol_ratio - 1.0))

            # Ensure we don't go below 50% of max size (always trade something)
            size_multiplier = max(0.5, size_multiplier)
        else:
            size_multiplier = 1.0

        return self.max_position_size * size_multiplier

    def _reset_position_state(self):
        """Reset position and management state after exit."""
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0
        self.trailing_stop_active = False
        self.highest_profit_point = 0
        # Note: Don't reset counters (tracked across episode)

    def _validate_position_management_action(self, action, current_price, atr):
        """
        Validate position management actions before execution.

        CRITICAL: Prevent invalid actions that could violate trading rules
        """
        if atr <= 0 or np.isnan(atr):
            return False, "Invalid ATR"

        if action == self.ACTION_MOVE_SL_TO_BE:
            # Only allow if currently profitable
            unrealized = self._calculate_unrealized_pnl(current_price)
            if unrealized <= 0:
                return False, "Cannot move to BE when losing"
            
            # Check if already at or past break-even
            if self.position == 1 and self.sl_price >= self.entry_price:
                return False, "SL already at or past BE"
            if self.position == -1 and self.sl_price <= self.entry_price:
                return False, "SL already at or past BE"

        elif action == self.ACTION_ENABLE_TRAIL:
            # Only allow enabling trailing if profitable
            unrealized = self._calculate_unrealized_pnl(current_price)
            if unrealized <= 0:
                return False, "Cannot enable trailing when losing"

        elif action == self.ACTION_DISABLE_TRAIL:
            # Disabling trailing is always valid (but may be no-op)
            pass  # Always valid

        return True, "Valid"

    def _calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L at current price."""
        if self.position == 0:
            return 0.0

        if self.position == 1:  # Long
            return (current_price - self.entry_price) * self.contract_size * self.position_size
        else:  # Short
            return (self.entry_price - current_price) * self.contract_size * self.position_size

    def action_masks(self) -> np.ndarray:
        """
        Get action mask for current state.

        RL FIX #4: Action masking prevents wasted exploration on invalid actions.
        RL FIX #10: Updated for simplified 6-action space.

        Returns:
            np.ndarray: Boolean mask of shape (6,) where True = valid action

        Called by MaskablePPO before each action selection.
        """
        # Start with all actions valid
        mask = np.ones(6, dtype=bool)

        current_price = self.data['close'].iloc[self.current_step]
        current_atr = self.data['atr'].iloc[self.current_step]

        # Handle invalid ATR
        if current_atr <= 0 or np.isnan(current_atr):
            current_atr = current_price * 0.01

        # Get current time for RTH gating
        current_time = self.data.index[self.current_step]
        if current_time.tzinfo is None:
            ts = current_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = current_time.tz_convert('America/New_York')

        rth_open = ts.replace(hour=9, minute=30, second=0)
        rth_close = ts.replace(hour=16, minute=59, second=0)
        in_rth = (ts >= rth_open) and (ts <= rth_close) and self.allow_new_trades

        if self.position == 0:
            # Agent is FLAT - only entry actions valid
            mask[0] = True  # Hold always valid
            mask[1] = in_rth  # Buy only in RTH
            mask[2] = in_rth  # Sell only in RTH
            # Disable all position management actions
            mask[3:6] = False  # Move BE, Enable Trail, Disable Trail

        else:
            # Agent HAS POSITION - validate each position management action
            mask[0] = True  # Hold always valid
            mask[1] = False  # Can't open new long when in position
            mask[2] = False  # Can't open new short when in position

            unrealized_pnl = self._calculate_unrealized_pnl(current_price)

            # Move to BE: only if profitable and not already at BE
            if unrealized_pnl > 0:
                if self.position == 1:
                    mask[3] = self.sl_price < self.entry_price
                else:
                    mask[3] = self.sl_price > self.entry_price
            else:
                mask[3] = False

            # Enable trailing: only if profitable and not already enabled
            mask[4] = (unrealized_pnl > 0) and (not self.trailing_stop_active)

            # Disable trailing: only if currently enabled
            mask[5] = self.trailing_stop_active

        return mask

    # Phase 2 now uses unified _calculate_apex_reward() inherited from Phase 1
    # No separate reward function needed - pm_action parameter handles Phase 2 bonuses
