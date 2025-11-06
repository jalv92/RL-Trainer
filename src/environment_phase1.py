"""
Phase 1: Foundational Trading Patterns Environment

Goal: Learn WHEN to enter trades (not HOW to manage exits)
- Actions: 3 discrete (Hold=0, Buy=1, Sell=2)
- SL/TP: FIXED on entry (never adjusted)
- Focus: Entry signal quality, pattern recognition
- Reward: Entry accuracy, R:R ratio, survival

Based on: OpenAI Spinning Up PPO + Curriculum Learning
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from src.market_specs import MarketSpecification, ES_SPEC


class TradingEnvironmentPhase1(gym.Env):
    """
    Phase 1: Foundational Trading Patterns

    Observation Space: (window_size, 11) - OHLC + indicators + time features
    Action Space: Discrete(3) - Hold, Buy, Sell
    Reward: Entry quality focused (asymmetric TP/SL rewards)
    """

    metadata = {"render_modes": []}

    # Action space constants
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 50000,
        window_size: int = 20,
        second_data: pd.DataFrame = None,
        # Market specifications
        market_spec: MarketSpecification = None,
        commission_override: float = None,
        # Phase 1 specific parameters (FIXED SL/TP)
        initial_sl_multiplier: float = 1.5,
        initial_tp_ratio: float = 3.0,
        position_size_contracts: float = 0.5,
        trailing_drawdown_limit: float = 5000  # Relaxed for Phase 1
    ):
        """
        Initialize Phase 1 trading environment.

        Args:
            data: OHLCV + indicators DataFrame (minute bars)
            initial_balance: Starting capital ($50,000 for Apex)
            window_size: Lookback window for observations (20 bars)
            second_data: Optional second-level data for precise drawdown
            market_spec: Market specification object (defaults to ES if None)
            commission_override: Override default commission (None = use market default)
            initial_sl_multiplier: SL distance in ATR (1.5 = 1.5× ATR) - FIXED
            initial_tp_ratio: TP as multiple of SL (3.0 = 3:1 R:R) - FIXED
            position_size_contracts: Contract quantity (0.5 for safety)
            trailing_drawdown_limit: Max drawdown allowed ($5,000 relaxed)
        """
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.second_data = second_data

        # PHASE 1 CONSTRAINT: Fixed SL/TP (not adaptive)
        self.sl_atr_mult = initial_sl_multiplier
        self.tp_sl_ratio = initial_tp_ratio
        self.position_size = position_size_contracts
        self.trailing_dd_limit = trailing_drawdown_limit

        # APEX PROFIT TARGET (RITHMIC rule)
        self.profit_target = initial_balance + 3000  # $53,000 for $50K account
        self.profit_target_reached = False
        self.profit_target_reached_step = None
        self.trailing_stopped = False  # Stops when profit target reached

        # Market specifications (DYNAMIC - supports all futures markets)
        if market_spec is None:
            market_spec = ES_SPEC  # Default to ES for backward compatibility

        self.market_symbol = market_spec.symbol
        self.market_name = market_spec.name
        self.contract_size = market_spec.contract_multiplier
        self.tick_size = market_spec.tick_size
        self.tick_value = market_spec.tick_value

        # Commission (user can override, otherwise use market default)
        self.commission_per_side = commission_override if commission_override is not None else market_spec.commission

        # Slippage (market-dependent: liquid markets = 1 tick, less liquid = 2 ticks)
        self.slippage_points = market_spec.tick_size * market_spec.slippage_ticks

        # Gymnasium spaces
        self.action_space = spaces.Discrete(3)
        # RL FIX #2: Added 5 position-aware features to observation space
        # Previous: (window_size * 11,) = (20 * 11,) = (220,)
        # New: (window_size * 11 + 5,) = (225,)
        # Additional features: [position, entry_price_ratio, sl_distance_atr, tp_distance_atr, time_in_position]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * 11 + 5,),  # Flattened for SB3 + position state
            dtype=np.float32
        )

        # State tracking
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.sl_price = 0  # Fixed on entry
        self.tp_price = 0  # Fixed on entry

        # Tracking
        self.trade_history = []
        self.portfolio_values = [self.initial_balance]
        self.trailing_dd_levels = []  # Track trailing DD history for compliance
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Apex rules
        self.highest_balance = self.initial_balance
        self.trailing_dd_level = self.initial_balance - self.trailing_dd_limit

        # Reset profit target tracking
        self.profit_target_reached = False
        self.profit_target_reached_step = None
        self.trailing_stopped = False

        # NEW: Apex compliance tracking
        self.apex_violations = []
        self.allow_new_trades = True
        self.daily_loss_limit = 1000  # $1,000 daily loss limit
        self.daily_pnl = 0
        self.current_date = None
        self.position_entry_step = 0
        self.max_hold_time = 390  # Maximum 1 day (390 minutes)

        # Action diversity tracking (fix sell bias)
        self.buy_count = 0
        self.sell_count = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation window with position-aware features.

        RL FIX #2: Added 5 position-aware features
        RL FIX #6: Removed instance normalization (let VecNormalize handle it)

        Returns:
            np.ndarray: Flattened shape (window_size*11 + 5,) = (225,)
                Market features (per timestep × 20 bars):
                    [close, volume, sma_5, sma_20, rsi, macd, momentum, atr,
                     hour_norm, min_from_open, min_to_close]
                Position features (global):
                    [position, entry_price_ratio, sl_distance_atr, tp_distance_atr, time_in_position]
        """
        if self.current_step < self.window_size:
            # Return zero observation for early steps
            return np.zeros((self.window_size * 11 + 5,), dtype=np.float32)

        start = self.current_step - self.window_size
        end = self.current_step

        # Price features
        feature_cols = ['close', 'volume', 'sma_5', 'sma_20', 'rsi',
                       'macd', 'momentum', 'atr']
        obs = self.data[feature_cols].iloc[start:end].values

        # Time features (for RTH awareness)
        timestamps = self.data.index[start:end]
        time_features = np.zeros((len(timestamps), 3))

        for i, ts in enumerate(timestamps):
            hour_decimal = ts.hour + ts.minute / 60.0
            time_features[i, 0] = (hour_decimal - 9.5) / (16.98 - 9.5)

            market_open = ts.replace(hour=9, minute=30, second=0)
            min_open = (ts - market_open).total_seconds() / 60.0
            time_features[i, 1] = min_open / 449.0

            market_close = ts.replace(hour=16, minute=59, second=0)
            min_close = (market_close - ts).total_seconds() / 60.0
            time_features[i, 2] = min_close / 449.0

        # Combine market features (11 features per timestep)
        market_obs = np.concatenate([obs, time_features], axis=1)

        # RL FIX #6: REMOVED instance normalization
        # VecNormalize wrapper will handle all normalization with running statistics
        # This prevents double-normalization conflict and improves stability
        # DELETED:
        # obs_mean = obs.mean(axis=0)
        # obs_std = obs.std(axis=0) + 1e-8
        # obs = (obs - obs_mean) / obs_std

        # Flatten market features
        market_obs_flat = market_obs.flatten()

        # RL FIX #2: Add position-aware features (5 features)
        # These allow the agent to "see" its own position state
        current_price = self.data['close'].iloc[self.current_step]
        current_atr = self.data['atr'].iloc[self.current_step]

        # Handle invalid ATR
        if current_atr <= 0 or np.isnan(current_atr):
            current_atr = current_price * 0.01  # Fallback to 1% of price

        if self.position != 0:
            # Agent has a position - provide detailed position information
            position_features = np.array([
                float(self.position),  # -1 (short), 0 (flat), +1 (long)
                self.entry_price / current_price,  # Entry price ratio (< 1 if underwater for longs)
                abs(self.sl_price - current_price) / current_atr,  # SL distance in ATR units
                abs(self.tp_price - current_price) / current_atr,  # TP distance in ATR units
                (self.current_step - self.position_entry_step) / 390.0  # Time in position (normalized to 1 day)
            ], dtype=np.float32)
        else:
            # Agent is flat - zero out position-specific features
            position_features = np.array([
                0.0,  # No position
                1.0,  # Entry price ratio = 1 (neutral)
                0.0,  # No SL
                0.0,  # No TP
                0.0   # No time in position
            ], dtype=np.float32)

        # Combine market features + position features
        full_observation = np.concatenate([market_obs_flat, position_features])

        return full_observation.astype(np.float32)

    def _check_second_level_drawdown(self, current_bar_time) -> Tuple[bool, float]:
        """
        Check if trailing drawdown was hit at second-level granularity.

        Apex rules require real-time (second-level) precision for drawdown enforcement.
        This method checks all second-level bars within the current minute bar for violations.

        Args:
            current_bar_time: Timestamp of current minute bar

        Returns:
            (drawdown_hit, min_equity_reached) - Whether drawdown limit was violated
        """
        if self.second_data is None or self.position == 0:
            return False, self.balance

        # SAFETY: Don't check second-level drawdown on the same step we entered
        # This prevents false violations from timing mismatches
        if self.position_entry_step == self.current_step:
            return False, self.balance

        # Get all second-level bars for this minute
        # IMPORTANT: Ensure timezone-aware comparison
        start_time = current_bar_time
        end_time = current_bar_time + pd.Timedelta(minutes=1)

        # Normalize timezones for comparison
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC').tz_convert('America/New_York')
            end_time = end_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            start_time = start_time.tz_convert('America/New_York')
            end_time = end_time.tz_convert('America/New_York')

        # Filter second data for this minute bar
        try:
            mask = (self.second_data.index >= start_time) & (self.second_data.index < end_time)
            second_bars = self.second_data[mask]
        except Exception as e:
            # If timezone comparison fails, skip second-level check
            # print(f"[WARNING] Second-level drawdown check failed: {e}")
            return False, self.balance

        if len(second_bars) == 0:
            # No second-level data for this minute - not a violation
            return False, self.balance

        # Track minimum equity reached during this minute bar
        min_equity = float('inf')

        for _, bar in second_bars.iterrows():
            high = bar['high']
            low = bar['low']

            # Calculate unrealized PnL at second-level precision
            if self.position == 1:  # Long
                unrealized_pnl = (low - self.entry_price) * self.contract_size * self.position_size
            else:  # Short
                unrealized_pnl = (self.entry_price - high) * self.contract_size * self.position_size

            current_equity = self.balance + unrealized_pnl

            # Check if drawdown violated at this second
            if current_equity < self.trailing_dd_level:
                return True, current_equity

            # Track minimum equity
            if current_equity < min_equity:
                min_equity = current_equity

        return False, min_equity

    def _calculate_fixed_sl_tp(self, entry_price: float, position_type: int) -> Tuple[float, float]:
        """
        Calculate FIXED stop loss and take profit.

        PHASE 1 CONSTRAINT: Once set, these NEVER change.

        Args:
            entry_price: Entry execution price
            position_type: 1=long, -1=short

        Returns:
            (sl_price, tp_price)
        """
        atr = self.data['atr'].iloc[self.current_step]
        if atr <= 0 or np.isnan(atr):
            atr = entry_price * 0.01  # Fallback to 1% of price

        sl_distance = atr * self.sl_atr_mult
        tp_distance = sl_distance * self.tp_sl_ratio

        if position_type == 1:  # Long
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # Short
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        return sl, tp

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return (obs, reward, terminated, truncated, info).

        Args:
            action: 0=hold, 1=buy, 2=sell

        Returns:
            obs: Next observation
            reward: Immediate reward
            terminated: Episode ended due to failure
            truncated: Episode ended due to time limit
            info: Additional diagnostics
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, False, True, {}

        # Current market state
        current_price = self.data['close'].iloc[self.current_step]
        high = self.data['high'].iloc[self.current_step]
        low = self.data['low'].iloc[self.current_step]

        reward = 0.0
        terminated = False
        truncated = False
        position_changed = False
        trade_pnl = 0.0
        exit_reason = None

        # ============================================================
        # 1. CHECK SL/TP ON EXISTING POSITION (if any)
        # ============================================================
        if self.position != 0:
            sl_hit, tp_hit, exit_price = self._check_sl_tp_hit(high, low)

            if sl_hit or tp_hit:
                # Close position at SL or TP
                if self.position == 1:
                    trade_pnl = (exit_price - self.entry_price) * self.contract_size * self.position_size
                else:
                    trade_pnl = (self.entry_price - exit_price) * self.contract_size * self.position_size

                # Deduct commissions
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
                    'position_type': 'long' if self.position == 1 else 'short',
                    'step': self.current_step
                })

                # Reset position
                self.position = 0
                self.entry_price = 0
                self.sl_price = 0
                self.tp_price = 0
                position_changed = True

        # ============================================================
        # 2. CHECK APEX TIME RULES (CRITICAL: must be before action execution)
        # ============================================================
        current_time = self.data.index[self.current_step]
        if self._check_apex_time_rules(current_time):
            terminated = True
            reward = -0.1  # Heavy penalty for rule violation
            return self._get_observation(), reward, terminated, False, {
                'portfolio_value': self.balance,
                'position': self.position,
                'apex_violation': True
            }
        
        # ============================================================
        # 3. EXECUTE NEW ACTION (if flat)
        # ============================================================
        # RTH gating: allow new entries only between 09:30 and 16:59 ET
        if current_time.tzinfo is None:
            ts = current_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = current_time.tz_convert('America/New_York')

        rth_open = ts.replace(hour=9, minute=30, second=0)
        rth_close = ts.replace(hour=16, minute=59, second=0)
        allowed_to_open = (ts >= rth_open) and (ts <= rth_close) and self.allow_new_trades

        if self.position == 0 and allowed_to_open:
            if action == self.ACTION_BUY:
                # Open long
                self.position = 1
                self.entry_price = current_price + self.slippage_points
                self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(
                    self.entry_price, 1
                )

                # Deduct entry commission
                self.balance -= (self.commission_per_side * self.position_size)
                self.num_trades += 1
                self.buy_count += 1  # Track for diversity
                self.position_entry_step = self.current_step
                position_changed = True

            elif action == self.ACTION_SELL:
                # Open short
                self.position = -1
                self.entry_price = current_price - self.slippage_points
                self.sl_price, self.tp_price = self._calculate_fixed_sl_tp(
                    self.entry_price, -1
                )

                # Deduct entry commission
                self.balance -= (self.commission_per_side * self.position_size)
                self.num_trades += 1
                self.sell_count += 1  # Track for diversity
                self.position_entry_step = self.current_step
                position_changed = True

        # ============================================================
        # 3. UPDATE PORTFOLIO VALUE & TRAILING DRAWDOWN
        # ============================================================
        unrealized_pnl = 0.0
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.contract_size * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.contract_size * self.position_size

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
            reward = -0.1  # Heavy penalty for account failure

        # Check violation at second-level (Apex compliance)
        if not terminated and self.second_data is not None:
            current_bar_time = self.data.index[self.current_step]
            drawdown_hit, _ = self._check_second_level_drawdown(current_bar_time)
            if drawdown_hit:
                terminated = True
                reward = -0.1  # Heavy penalty for Apex violation

        # Update daily P&L after trade
        if trade_pnl != 0:
            self._update_daily_pnl(trade_pnl)
            
            # Check daily loss limit
            if self._check_daily_loss_limit():
                terminated = True
                reward = -0.1  # Heavy penalty
                return self._get_observation(), reward, terminated, False, {
                    'portfolio_value': self.balance,
                    'position': self.position,
                    'daily_loss_violation': True
                }
        
        # Track position hold time
        if self.position != 0:
            if self.position_entry_step == 0:
                self.position_entry_step = self.current_step
        else:
            self.position_entry_step = 0

        # ============================================================
        # 4. CALCULATE REWARD (Apex-Optimized)
        # ============================================================
        if not terminated:
            reward = self._calculate_apex_reward(
                position_changed, exit_reason, trade_pnl, portfolio_value
            )

        # ============================================================
        # 5. ADVANCE TIME
        # ============================================================
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            truncated = True

        obs = self._get_observation()
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'num_trades': self.num_trades,
            'balance': self.balance,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }

        return obs, reward, terminated, truncated, info

    def _check_apex_time_rules(self, current_time) -> bool:
        """
        Enforce Apex trading time rules.

        RL FIX #10: Only terminate on VIOLATION (holding position at 4:59)
        This teaches agent that compliance matters, not just "time ends at 4:59"

        CRITICAL: Apex requires ALL positions closed by 4:59 PM ET
        """
        # Convert to America/New_York if needed
        if current_time.tzinfo is None:
            ts = current_time.tz_localize('UTC').tz_convert('America/New_York')
        else:
            ts = current_time.tz_convert('America/New_York')

        # RL FIX #10: Only terminate if VIOLATING (still holding at 4:59)
        if ts.hour == 16 and ts.minute >= 59:
            if self.position != 0:
                # VIOLATION: Still holding position at 4:59 PM
                # Force close and terminate episode with violation penalty
                self._force_close_position(reason="VIOLATION: 4:59 PM AUTO-CLOSE")
                self._add_apex_violation("Late position closure - held past 4:59 PM ET")

                # Cancel all pending orders
                self._cancel_all_pending_orders()

                # Terminate with penalty (reward already set to -0.1 in step())
                return True  # terminated

            else:
                # COMPLIANT: Agent already closed position before 4:59
                # Don't terminate - this was good behavior!
                # Let episode continue naturally or mark as truncated if out of data
                self._cancel_all_pending_orders()
                return False  # not terminated (compliant)

        # No new trades after 4:00 PM (give agent time to close)
        if ts.hour >= 16:
            self.allow_new_trades = False

        return False  # not terminated

    def _force_close_position(self, reason="Auto-close"):
        """Force close position immediately at market price."""
        if self.position == 0:
            return
        
        current_price = self.data['close'].iloc[self.current_step]
        
        if self.position == 1:  # Long
            exit_price = current_price - self.slippage_points
            trade_pnl = (exit_price - self.entry_price) * self.contract_size * self.position_size
        else:  # Short
            exit_price = current_price + self.slippage_points
            trade_pnl = (self.entry_price - exit_price) * self.contract_size * self.position_size
        
        # Deduct exit commission
        trade_pnl -= (self.commission_per_side * self.position_size)
        self.balance += trade_pnl
        
        # Record trade
        self.trade_history.append({
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'exit_reason': reason,
            'pnl': trade_pnl,
            'position_type': 'long' if self.position == 1 else 'short',
            'step': self.current_step,
            'forced': True
        })
        
        # Update trade counts
        if trade_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.sl_price = 0
        self.tp_price = 0

    def _add_apex_violation(self, violation_type, description=""):
        """Record Apex rule violation."""
        violation = {
            'step': self.current_step,
            'timestamp': self.data.index[self.current_step],
            'type': violation_type,
            'description': description,
            'portfolio_value': self.balance + self._calculate_unrealized_pnl()
        }
        self.apex_violations.append(violation)

    def _cancel_all_pending_orders(self):
        """Cancel all pending orders (placeholder for future implementation)."""
        # Placeholder for order management system
        pass

    def _calculate_unrealized_pnl(self):
        """Calculate current unrealized P&L."""
        if self.position == 0:
            return 0.0
        
        current_price = self.data['close'].iloc[self.current_step]
        
        if self.position == 1:  # Long
            return (current_price - self.entry_price) * self.contract_size * self.position_size
        else:  # Short
            return (self.entry_price - current_price) * self.contract_size * self.position_size

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded."""
        current_time = self.data.index[self.current_step]
        
        # Reset daily PnL on new day
        if self.current_date is None or self.current_date != current_time.date():
            self.daily_pnl = 0
            self.current_date = current_time.date()
        
        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            self._add_apex_violation(
                "Daily loss limit exceeded",
                f"Loss: ${self.daily_pnl:.2f}, Limit: ${self.daily_loss_limit}"
            )
            return True  # Terminate episode
        
        return False

    def _update_daily_pnl(self, trade_pnl):
        """Update daily P&L after trade."""
        if trade_pnl != 0:
            self.daily_pnl += trade_pnl

    def _check_sl_tp_hit(self, high: float, low: float) -> Tuple[bool, bool, float]:
        """
        Check if SL or TP was hit during the bar.

        Returns:
            (sl_hit, tp_hit, exit_price)
        """
        sl_hit = False
        tp_hit = False
        exit_price = 0

        if self.position == 1:  # Long
            if low <= self.sl_price:
                sl_hit = True
                exit_price = self.sl_price
            elif high >= self.tp_price:
                tp_hit = True
                exit_price = self.tp_price
        else:  # Short
            if high >= self.sl_price:
                sl_hit = True
                exit_price = self.sl_price
            elif low <= self.tp_price:
                tp_hit = True
                exit_price = self.tp_price

        return sl_hit, tp_hit, exit_price

    def _calculate_apex_reward(
        self,
        position_changed: bool,
        exit_reason: str,
        trade_pnl: float,
        portfolio_value: float,
        pm_action: str = None
    ) -> float:
        """
        Unified Apex-Optimized Reward Function

        RL FIX #5: Added dense intermediate rewards (portfolio growth)
        RL FIX #11: Made longevity bonus conditional on portfolio growth

        Optimizes for:
        1. Risk-adjusted returns (Sharpe ratio) - Weight: 35% (reduced from 40%)
        2. Profit target achievement + maintenance - Weight: 30%
        3. Trailing drawdown avoidance - Weight: 20%
        4. Trade quality (R-multiples, win rate) - Weight: 10%
        5. Portfolio growth (step-wise) - Weight: 5% (NEW)
        6. Position management skill (Phase 2 only) - Bonus
        7. Episode longevity (conditional) - Intrinsic

        Based on: Modern Portfolio Theory + Apex Evaluation Rules + OpenAI RL Best Practices
        """
        reward = 0.0

        # ================================================================
        # COMPONENT 1: RISK-ADJUSTED RETURNS (Sharpe Ratio) - Weight: 35%
        # ================================================================
        if len(self.portfolio_values) > 20:
            # Use improved Sharpe calculation with hacking prevention (Fix #1)
            sharpe_reward = self._calculate_sharpe_component()
            reward += sharpe_reward

        # ================================================================
        # COMPONENT 1B: DENSE INTERMEDIATE REWARDS - Weight: 5% (NEW - Fix #5)
        # ================================================================
        # Provide every-step feedback for faster learning
        # This addresses sparse reward signal issue
        # FIX #10: Reduced scaling from 100->20 to prevent overemphasis on short-term growth
        if len(self.portfolio_values) >= 2:
            # Calculate recent portfolio change vs. previous step
            recent_change = (portfolio_value - self.portfolio_values[-1]) / self.initial_balance
            # Small reward for positive portfolio growth every step
            # Clip to prevent dominating other components
            growth_reward = np.clip(recent_change * 20, -0.005, 0.005)  # REDUCED from 100
            reward += growth_reward

        # ================================================================
        # COMPONENT 2: PROFIT TARGET ACHIEVEMENT - Weight: 30%
        # ================================================================
        profit_above_start = portfolio_value - self.initial_balance

        # 2A. Progress towards profit target (before reaching)
        if not self.profit_target_reached:
            target_progress = profit_above_start / 3000  # 3000 = profit target
            progress_reward = np.clip(target_progress * 0.01, 0, 0.03)
            reward += progress_reward

        # 2B. MASSIVE bonus for reaching profit target (one-time)
        if self.profit_target_reached and self.current_step == self.profit_target_reached_step:
            reward += 0.10  # Large one-time bonus

        # 2C. Maintenance bonus (after reaching target)
        if self.profit_target_reached:
            # Reward for staying above target
            cushion = portfolio_value - self.profit_target
            if cushion >= 0:
                # Bonus for maintaining profit
                maintenance_bonus = min(cushion / 1000 * 0.005, 0.02)
                reward += maintenance_bonus
            else:
                # Penalty for dropping below (but not termination per user request)
                penalty = abs(cushion) / 1000 * 0.01
                reward -= penalty

        # ================================================================
        # COMPONENT 3: TRAILING DRAWDOWN AVOIDANCE - Weight: 20%
        # ================================================================
        distance_to_dd = portfolio_value - self.trailing_dd_level
        dd_buffer_ratio = distance_to_dd / self.trailing_dd_limit

        if dd_buffer_ratio < 0.2:  # Within 20% of drawdown
            # Exponential penalty as we approach drawdown
            proximity_penalty = 0.03 * (0.2 - dd_buffer_ratio) / 0.2
            reward -= proximity_penalty
        elif dd_buffer_ratio > 0.8:  # Far from drawdown (safe zone)
            safety_bonus = 0.005
            reward += safety_bonus

        # ================================================================
        # COMPONENT 4: TRADE QUALITY - Weight: 10%
        # ================================================================
        # 4A. Trade outcome (asymmetric - favor winners)
        if exit_reason == "take_profit":
            reward += 0.01  # FIXED: Symmetric with SL
        elif exit_reason == "stop_loss":
            reward -= 0.01  # FIXED: Symmetric with TP  # Smaller penalty if stop hit (risk management working)
        elif exit_reason == "manual_close" and trade_pnl != 0:
            # Reward profitable discretionary exits
            reward += 0.01 if trade_pnl > 0 else -0.008

        # 4B. R-multiple quality (risk-adjusted trade returns)
        if exit_reason in ["take_profit", "stop_loss", "manual_close"] and trade_pnl != 0:
            initial_risk = abs(self.entry_price - self.sl_price) * self.contract_size * self.position_size
            if initial_risk > 0:
                r_multiple = trade_pnl / initial_risk
                # Reward high R-multiples exponentially
                if r_multiple > 0:
                    r_bonus = min(r_multiple * 0.003, 0.01)  # FIXED: Balanced cap
                else:
                    r_bonus = max(r_multiple * 0.003, -0.01)  # FIXED: Same scaling as positive
                reward += r_bonus

        # 4C. Win rate tracking (encourage consistency)
        if self.num_trades > 5:
            win_rate = self.winning_trades / self.num_trades
            if win_rate > 0.5:
                consistency_bonus = (win_rate - 0.5) * 0.01
                reward += consistency_bonus

        # ================================================================
        # COMPONENT 5: POSITION MANAGEMENT (Phase 2 Only) - Weight: Bonus
        # ================================================================
        if pm_action:  # Only in Phase 2
            if pm_action == "move_to_be":
                reward += 0.005  # Strong reward for risk-free trades
            elif pm_action == "tighten_sl":
                reward += 0.003  # Reward locking in profits
            elif pm_action == "extend_tp":
                reward += 0.002  # Reward trend riding
            elif "trail" in pm_action:
                reward += 0.002  # Reward dynamic stops

        # ================================================================
        # COMPONENT 6: EPISODE LONGEVITY - Intrinsic Motivation (Fix #11)
        # ================================================================
        # RL FIX #11: Make longevity bonus conditional on portfolio growth
        # This prevents conflict with drawdown avoidance penalty
        # Only reward survival if actually making money
        # FIX #10: Increased profit requirement and reduced max bonus (0.008->0.005)
        steps_survived = self.current_step - self.window_size
        profit_threshold = self.initial_balance * 1.002  # Require 0.2% minimum profit
        if steps_survived > 50 and portfolio_value > profit_threshold:
            # Reward staying in the game (avoid early blowups) ONLY if profitable
            # This eliminates contradiction: we don't reward just "staying alive" during drawdowns
            survival_bonus = min(0.001 * (steps_survived / 100), 0.005)  # REDUCED max from 0.008
            reward += survival_bonus

        # ================================================================
        # COMPONENT 7: TIME DECAY PENALTY
        # ================================================================
        # Add penalty for holding positions too long
        if self.position != 0 and self.position_entry_step > 0:
            hold_time = self.current_step - self.position_entry_step
            if hold_time > self.max_hold_time:
                time_penalty = -0.001 * (hold_time - self.max_hold_time)
                reward += time_penalty

        # ================================================================
        # COMPONENT 8: ACTION DIVERSITY INCENTIVE - FIX for Sell-Only Bias
        # ================================================================
        # Encourage balanced use of Buy and Sell actions
        # This prevents model from getting stuck using only one action type
        if hasattr(self, 'buy_count') and hasattr(self, 'sell_count'):
            total_entries = self.buy_count + self.sell_count
            if total_entries >= 10:  # Only after sufficient samples
                buy_ratio = self.buy_count / total_entries
                sell_ratio = self.sell_count / total_entries

                # Reward balanced action usage (close to 50/50)
                # Penalize extreme imbalances (e.g., 90% sell, 10% buy)
                imbalance = abs(buy_ratio - sell_ratio)

                if imbalance > 0.6:  # More than 80/20 split
                    # Strong penalty for severe imbalance
                    diversity_penalty = -0.003 * imbalance
                    reward += diversity_penalty
                elif imbalance < 0.2:  # Close to 60/40 or better
                    # Small bonus for good diversity
                    diversity_bonus = 0.001
                    reward += diversity_bonus

        return reward

    def _calculate_sharpe_component(self) -> float:
        """
        Calculate Sharpe ratio component with anti-hacking protection.

        CRITICAL IMPROVEMENTS (RL Analysis Fix #1):
        1. Raised min_std from 0.01 -> 0.05 to prevent cash-holding strategies
        2. Added penalty for low volatility (agent not trading actively)
        3. Asymmetric scaling: favor positive Sharpe more than penalize negative
        4. Increased max reward from 0.05 -> 0.06 for better exploration

        This prevents the agent from gaming the system by holding cash with near-zero
        volatility to achieve artificially high Sharpe ratios.
        """
        if len(self.portfolio_values) < 2:
            return 0.0

        recent_values = self.portfolio_values[-20:]
        returns = np.diff(recent_values) / (np.array(recent_values[:-1]) + 1e-8)

        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        # CRITICAL FIX: Higher minimum volatility floor to prevent cash-holding
        # If agent isn't trading, volatility will be near zero -> we penalize this
        min_std = 0.05  # Raised from 0.01 to 0.05 (5% minimum volatility)

        # NEW: Penalize agents that aren't actively trading
        if std_return < min_std:
            # Low volatility = agent holding cash = gaming the system
            # Return negative reward proportional to how inactive the agent is
            penalty_factor = (min_std - std_return) / min_std
            return -0.01 * penalty_factor

        # Calculate Sharpe-like ratio (now we know agent is actually trading)
        sharpe = mean_return / std_return

        # Clip to reasonable range to prevent extreme values
        sharpe = np.clip(sharpe, -2.0, 2.0)

        # NEW: Asymmetric scaling - favor positive Sharpe more than penalize negative
        # This encourages risk-taking when profitable
        if sharpe > 0:
            sharpe_reward = sharpe * 0.12  # Increased from 0.1 (20% more reward)
        else:
            sharpe_reward = sharpe * 0.08  # Less penalty for negative Sharpe

        # NEW: Increased max reward ceiling from 0.05 -> 0.06
        return np.clip(sharpe_reward, -0.02, 0.06)

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
