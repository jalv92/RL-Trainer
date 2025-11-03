#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Model Evaluation
- Comprehensive backtest with position management analysis
- Sharpe ratio, max drawdown, win rate calculations
- Position management action usage statistics
- Equity curve and performance visualizations

Usage:
    python3 evaluate_phase2.py
    python3 evaluate_phase2.py --model models/phase2/best_model.zip
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO  # Phase 2 uses MaskablePPO, not standard PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment_phase2 import TradingEnvironmentPhase2
from apex_compliance_checker import ApexComplianceChecker
from feature_engineering import add_market_regime_features

# Set UTF-8 encoding for Windows compatibility
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except:
        pass


def safe_print(message=""):
    """Print with fallback for encoding errors on Windows."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: replace Unicode characters with ASCII equivalents
        replacements = {
            '✓': '[OK]',
            '✅': '[OK]',
            '✗': '[X]',
            '❌': '[X]',
            '→': '->',
            '⚠': '[WARN]',
            '⚠️': '[WARN]',
            '—': '-',
            '–': '-',
            '’': "'",
            '“': '"',
            '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        # Final safeguard: drop any remaining non-ASCII characters
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)


def find_data_file():
    """Find training data file."""
    # Get project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(script_dir, 'data')

    filenames = [
        'D1M.csv',  # 1-minute data
        'ES_D1M.csv',  # Instrument-prefixed 1-minute data
        'es_training_data_CORRECTED.csv',
        'databento_es_training_data_processed_cleaned.csv',
        'databento_es_training_data_processed.csv'
    ]

    for filename in filenames:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path

    # Fallback: look for instrument-prefixed files like ES_D1M.csv, NQ_D1M.csv, etc.
    pattern = os.path.join(data_dir, '*_D1M.csv')
    instrument_files = sorted(glob.glob(pattern))
    if instrument_files:
        return instrument_files[0]

    raise FileNotFoundError(
        f"Data not found in {data_dir}. "
        f"Expected one of: {filenames} or any '*_D1M.csv' file"
    )


def load_data(data_path):
    """Load and prepare data."""
    try:
        data = pd.read_csv(data_path, index_col='datetime', parse_dates=True)
    except Exception:
        df = pd.read_csv(data_path)
        time_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
        df[time_col] = pd.to_datetime(df[time_col])
        data = df.set_index(time_col)

    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')

    return data


def evaluate_phase2_model(
    model_path='models/phase2_position_mgmt_final.zip',
    vecnorm_path='models/phase2_vecnorm.pkl',
    max_steps=5000,
    verbose=False
):
    """
    Comprehensive Phase 2 evaluation.

    Args:
        model_path: Path to trained Phase 2 model
        vecnorm_path: Path to VecNormalize stats
        max_steps: Maximum evaluation steps (prevents infinite loops)

    Returns:
        Dict of evaluation metrics
    """
    safe_print("=" * 80)
    safe_print("PHASE 2 MODEL EVALUATION")
    safe_print("=" * 80)

    # Check if model exists
    if not os.path.exists(model_path):
        safe_print(f"[ERROR] Model not found: {model_path}")
        safe_print("[ERROR] Train Phase 2 first: python3 train_phase2.py")
        return None

    # Load data
    data_path = find_data_file()
    safe_print(f"[DATA] Loading minute-level data from {data_path}")
    data = load_data(data_path)
    safe_print(f"[DATA] Loaded {len(data):,} rows")

    # Add market regime features
    safe_print("[DATA] Adding market regime features...")
    data = add_market_regime_features(data)
    safe_print(f"[DATA] Enhanced feature count: {len(data.columns)}")

    # Load second-level data (optional, for real-time drawdown)
    second_data = None
    # Get project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    second_candidates = [
        os.path.join(script_dir, 'data', 'D1S.csv'),
        os.path.join(script_dir, 'data', 'ES_D1S.csv')
    ]
    if not any(os.path.exists(path) for path in second_candidates):
        pattern = os.path.join(script_dir, 'data', '*_D1S.csv')
        second_candidates.extend(sorted(glob.glob(pattern)))

    second_data_path = next((path for path in second_candidates if os.path.exists(path)), None)

    if second_data_path and os.path.exists(second_data_path):
        try:
            safe_print(f"[DATA] Loading second-level data from {second_data_path}")
            second_data = pd.read_csv(second_data_path, index_col='ts_event', parse_dates=True)
            # Ensure timezone is set
            if second_data.index.tz is None:
                second_data.index = second_data.index.tz_localize('UTC').tz_convert('America/New_York')
            elif str(second_data.index.tz) != 'America/New_York':
                second_data.index = second_data.index.tz_convert('America/New_York')
            safe_print(f"[DATA] Loaded {len(second_data):,} second-level bars")
        except Exception as e:
            safe_print(f"[DATA] Warning: Could not load second-level data: {e}")
            second_data = None
    else:
        safe_print("[DATA] Second-level data not found (D1S/ES_D1S/_D1S.csv). Optional feature.")

    # Create environment
    safe_print("[ENV] Creating Phase 2 evaluation environment...")
    env = TradingEnvironmentPhase2(
        data,
        window_size=20,
        initial_balance=50000,
        second_data=second_data,  # Pass second-level data
        position_size_contracts=1.0,
        trailing_drawdown_limit=2500
    )
    env = DummyVecEnv([lambda: env])

    # Load VecNormalize
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        safe_print(f"[ENV] Loaded VecNormalize from {vecnorm_path}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        safe_print("[ENV] Using fresh VecNormalize (stats not found)")

    env.training = False
    env.norm_reward = False

    # Load model
    safe_print(f"[MODEL] Loading from {model_path}")
    model = MaskablePPO.load(model_path)  # Use MaskablePPO for Phase 2

    # Evaluation loop
    obs = env.reset()
    done = False
    step = 0

    # Tracking
    equity_curve = []
    actions_taken = []
    pm_actions_taken = []
    rewards = []

    safe_print(f"\n[EVAL] Running evaluation (max {max_steps:,} steps)...")
    safe_print("[EVAL] Progress:")

    while not done and step < max_steps:
        # Get action mask from environment (Phase 2 uses action masking)
        action_masks = env.env_method('action_masks')
        # Predict with action masking
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        # ✅ FIX: Handle new Gymnasium API (5 returns: obs, reward, terminated, truncated, info)
        # VecEnv auto-converts to (obs, reward, done, info) where done = terminated | truncated
        obs, reward, done, info = env.step(action)

        # Verbose logging
        if verbose and step < 10:  # Log first 10 steps in detail
            safe_print(f"\n[VERBOSE] Step {step}:")
            safe_print(f"  Action: {action[0]} | Action Mask: {action_masks[0]}")
            safe_print(f"  Reward: {reward[0]:.4f} | Done: {done}")
            safe_print(f"  Position: {info[0].get('position', 0)}")
            safe_print(f"  Equity: ${info[0].get('portfolio_value', 0):,.2f}")
            if 'pm_action' in info[0]:
                safe_print(f"  PM Action: {info[0]['pm_action']}")

        # Track metrics
        equity_curve.append(info[0]['portfolio_value'])
        actions_taken.append(int(action[0]))
        rewards.append(reward[0])

        # Track position management actions
        if 'pm_action' in info[0] and info[0]['pm_action']:
            pm_actions_taken.append(info[0]['pm_action'])

        step += 1

        # Progress updates
        if step % 500 == 0:
            safe_print(f"  Step {step:,}/{max_steps:,} | Equity: ${equity_curve[-1]:,.2f}")

    # Diagnostic: Why did episode end?
    if done and step < max_steps:
        safe_print(f"\n[EVAL] [WARN] Episode ended early at step {step}/{max_steps}")
        safe_print(f"[EVAL] Final info: {info[0]}")

    safe_print(f"\n[EVAL] [OK] Evaluation complete ({step:,} steps)")

    # Get environment instance for trade history
    env_unwrapped = env.envs[0]

    # Calculate metrics
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]

    # Sharpe ratio (annualized)
    sharpe_ratio = 0
    if returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 390)  # 390 min/day

    # Max drawdown
    cummax = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - cummax) / cummax
    max_dd_pct = abs(drawdowns.min()) * 100

    # Returns
    final_equity = equity_array[-1]
    total_return = ((final_equity - 50000) / 50000) * 100

    # Action distribution
    action_counts = dict(zip(*np.unique(actions_taken, return_counts=True)))
    action_names = {
        0: 'Hold',
        1: 'Buy',
        2: 'Sell',
        3: 'Move to BE',      # Was 5
        4: 'Enable Trail',    # Was 7
        5: 'Disable Trail'    # Was 8
    }

    # PM action usage
    pm_usage = len(pm_actions_taken) / step * 100 if step > 0 else 0

    # Calculate comprehensive metrics
    comprehensive_metrics = calculate_comprehensive_metrics(
        equity_curve, env_unwrapped.trade_history
    )
    
    # Results
    results = {
        'total_steps': step,
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_dd_pct,
        'action_distribution': action_counts,
        'pm_actions_used': len(pm_actions_taken),
        'pm_usage_pct': pm_usage,
        'equity_curve': equity_curve,
        'returns': returns.tolist()
    }
    
    # Add comprehensive metrics
    results.update(comprehensive_metrics)

    # Print results
    safe_print("\n" + "=" * 80)
    safe_print("EVALUATION RESULTS")
    safe_print("=" * 80)
    safe_print(f"\n[PERFORMANCE]")
    safe_print(f"  Final Equity: ${final_equity:,.2f}")
    safe_print(f"  Total Return: {total_return:+.2f}%")
    safe_print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    safe_print(f"  Max Drawdown: {max_dd_pct:.2f}%")
    safe_print(f"  Profit Factor: {comprehensive_metrics['profit_factor']:.2f}")
    safe_print(f"  Recovery Factor: {comprehensive_metrics['recovery_factor']:.2f}")
    safe_print(f"  Consistency Score: {comprehensive_metrics['consistency_score']:.2f}")
    safe_print(f"  Apex Compliance Rate: {comprehensive_metrics['apex_compliance_rate']:.1f}%")
    safe_print()
    safe_print(f"[ACTIONS]")
    safe_print(f"  Total Steps: {step:,}")
    for action_id, count in sorted(action_counts.items()):
        pct = (count / step) * 100
        safe_print(f"  {action_names.get(action_id, f'Action {action_id}')}: {count:,} ({pct:.1f}%)")
    safe_print()
    safe_print(f"[POSITION MANAGEMENT]")
    safe_print(f"  PM Actions Used: {len(pm_actions_taken):,}")
    safe_print(f"  PM Usage Rate: {pm_usage:.2f}%")
    if pm_actions_taken:
        pm_breakdown = dict(zip(*np.unique(pm_actions_taken, return_counts=True)))
        for pm_action, count in sorted(pm_breakdown.items(), key=lambda x: -x[1]):
            safe_print(f"    {pm_action}: {count}")

    # Save plots
    safe_print(f"\n[PLOT] Generating evaluation charts...")
    save_evaluation_plots(results, action_names)

    # Save metrics
    save_metrics_json(results)

    # APEX COMPLIANCE CHECK
    safe_print("\n" + "=" * 80)
    safe_print("RUNNING APEX COMPLIANCE CHECK")
    safe_print("=" * 80)

    # env_unwrapped already defined above (line 186)

    # Create checker
    checker = ApexComplianceChecker(
        account_size=50000,
        trailing_dd_limit=2500,
        profit_target=3000,
        max_contracts=10
    )

    # Use actual trailing DD levels tracked during episode
    trailing_dd_levels = env_unwrapped.trailing_dd_levels

    # Run compliance check
    compliance_results = checker.check_episode(
        portfolio_values=equity_curve,
        trailing_dd_levels=trailing_dd_levels,
        trade_history=env_unwrapped.trade_history,
        timestamps=data.index[:len(equity_curve)],
        position_sizes=[env_unwrapped.position_size] * len(equity_curve)
    )

    # Print compliance report
    checker.print_report(compliance_results)

    # Add compliance results to return dict
    results['apex_compliance'] = compliance_results

    return results


def calculate_comprehensive_metrics(equity_curve, trade_history, initial_balance=50000):
    """Calculate comprehensive performance metrics."""
    equity_array = np.array(equity_curve)
    
    # Basic return metrics
    total_return = (equity_array[-1] - initial_balance) / initial_balance
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # Sharpe ratio (annualized)
    sharpe_ratio = 0
    if returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 390)
    
    # Drawdown metrics
    peak = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - peak) / peak
    max_drawdown = drawdown.min()
    
    # Trade metrics
    if trade_history:
        profits = [t['pnl'] for t in trade_history if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in trade_history if t.get('pnl', 0) < 0]
        
        # Profit Factor
        total_profits = sum(profits) if profits else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Win Rate
        win_rate = len(profits) / len(trade_history)
        
        # Average Win/Loss
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Recovery Factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Consistency Score (daily return stability)
        # Group returns by day and calculate consistency
        daily_returns = {}
        for i, ret in enumerate(returns):
            day = i // 390  # Assuming 390 minutes per trading day
            if day not in daily_returns:
                daily_returns[day] = []
            daily_returns[day].append(ret)
        
        day_returns = [sum(day_rets) for day_rets in daily_returns.values()]
        if day_returns:
            mean_daily = np.mean(day_returns)
            std_daily = np.std(day_returns)
            consistency_score = 1 - (std_daily / abs(mean_daily)) if mean_daily != 0 else 0
        else:
            consistency_score = 0
    else:
        profit_factor = win_rate = avg_win = avg_loss = recovery_factor = consistency_score = 0

    # Apex Compliance Rate (based on violations in trade history)
    apex_violations = 0
    apex_compliant_trades = 0
    for trade in trade_history:
        if trade.get('forced', False) or 'violation' in trade.get('exit_reason', '').lower():
            apex_violations += 1
        else:
            apex_compliant_trades += 1

    total_evaluable = apex_violations + apex_compliant_trades
    apex_compliance_rate = (apex_compliant_trades / total_evaluable * 100) if total_evaluable > 0 else 100.0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'recovery_factor': recovery_factor,
        'consistency_score': consistency_score,
        'total_trades': len(trade_history),
        'apex_compliance_rate': apex_compliance_rate,
        'apex_violations': apex_violations
    }


def save_evaluation_plots(results, action_names):
    """Generate and save evaluation plots."""
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 2 Model: Evaluation Results', fontsize=16, fontweight='bold')

    # 1. Equity curve
    axes[0, 0].plot(results['equity_curve'], linewidth=2, color='green')
    axes[0, 0].axhline(y=50000, color='red', linestyle='--', alpha=0.5, label='Starting Capital')
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(bottom=0)

    # 2. Drawdown
    equity_array = np.array(results['equity_curve'])
    cummax = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - cummax) / cummax * 100

    axes[0, 1].plot(drawdowns, linewidth=2, color='orange')
    axes[0, 1].fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='orange')
    axes[0, 1].axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='5% DD')
    axes[0, 1].set_title('Drawdown from Peak')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Action distribution
    action_ids = list(results['action_distribution'].keys())
    action_counts = list(results['action_distribution'].values())
    labels = [action_names.get(aid, f'Action {aid}') for aid in action_ids]

    axes[1, 0].bar(labels, action_counts, color=['gray', 'green', 'red', 'blue', 'purple', 'orange', 'cyan', 'magenta'][:len(labels)])
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Returns histogram
    axes[1, 1].hist(results['returns'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Return Distribution')
    axes[1, 1].set_xlabel('Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = 'results/phase2_evaluation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    safe_print(f"[PLOT] [OK] Saved to {plot_path}")
    plt.close()


def save_metrics_json(results):
    """Save metrics to JSON."""
    import json

    # Remove large arrays from JSON
    metrics = {
        'total_steps': results['total_steps'],
        'final_equity': float(results['final_equity']),
        'total_return_pct': float(results['total_return_pct']),
        'sharpe_ratio': float(results['sharpe_ratio']),
        'max_drawdown_pct': float(results['max_drawdown_pct']),
        'pm_actions_used': results['pm_actions_used'],
        'pm_usage_pct': float(results['pm_usage_pct']),
        'action_distribution': {int(k): int(v) for k, v in results['action_distribution'].items()}
    }

    metrics_path = 'results/phase2_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    safe_print(f"[METRICS] [OK] Saved to {metrics_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Phase 2 Model')
    parser.add_argument('--model', default='models/phase2_position_mgmt_final.zip',
                       help='Path to model')
    parser.add_argument('--vecnorm', default='models/phase2_vecnorm.pkl',
                       help='Path to VecNormalize stats')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum evaluation steps')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (first 10 steps)')

    args = parser.parse_args()

    results = evaluate_phase2_model(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        max_steps=args.max_steps,
        verbose=args.verbose
    )

    if results:
        safe_print("\n[SUCCESS] Evaluation complete!")
        safe_print("[OUTPUT] Charts: results/phase2_evaluation.png")
        safe_print("[OUTPUT] Metrics: results/phase2_metrics.json")
    else:
        safe_print("\n[FAILED] Evaluation failed")


if __name__ == '__main__':
    main()
