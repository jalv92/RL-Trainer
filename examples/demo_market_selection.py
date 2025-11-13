#!/usr/bin/env python3
"""
Demo script showing market detection and selection scenarios.
This demonstrates how the training scripts will behave with different data configurations.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_utils import detect_available_markets, select_market_for_training

def safe_print(msg):
    """Safe print function matching training scripts."""
    print(msg)

def demo_scenario_1():
    """Scenario 1: Multiple markets available (ES and NQ)"""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Multiple Markets Available")
    print("=" * 80)
    print("\nThis simulates what users see when multiple datasets exist.")
    print("In this case, ES and NQ data are both available.\n")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    markets = detect_available_markets(data_dir)

    print(f"Detected {len(markets)} market(s):")
    for i, market in enumerate(markets, 1):
        second_status = "[OK]" if market['has_second'] else "[MINUTE ONLY]"
        print(f"  {i}. {market['market']:<8} - {market['minute_file']:<20} {second_status}")

    print("\nBEHAVIOR:")
    print("  - User will be prompted to select market number")
    print("  - Can press 'q' to quit")
    print("  - Selected market data will be used for training")


def demo_scenario_2():
    """Scenario 2: Only one market available"""
    print("\n" + "=" * 80)
    print("SCENARIO 2: Single Market Auto-Selection")
    print("=" * 80)
    print("\nThis simulates what happens when only one dataset exists.")
    print("The system will auto-select it without prompting.\n")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    all_markets = detect_available_markets(data_dir)

    if all_markets:
        # Simulate having only ES
        single_market = [all_markets[0]]
        selected = select_market_for_training(single_market, safe_print)

        if selected:
            print(f"\nRESULT: Market '{selected['market']}' was automatically selected")
            print(f"        No user interaction required!")


def demo_scenario_3():
    """Scenario 3: No data available"""
    print("\n" + "=" * 80)
    print("SCENARIO 3: No Data Available")
    print("=" * 80)
    print("\nThis shows what happens when no market data exists.\n")

    # Simulate empty market list
    empty_markets = []
    selected = select_market_for_training(empty_markets, safe_print)

    print(f"\nRESULT: Training would be cancelled (selected = {selected})")


def demo_data_file_detection():
    """Show how find_data_file works with market parameter"""
    print("\n" + "=" * 80)
    print("DATA FILE DETECTION LOGIC")
    print("=" * 80)

    print("\nThe find_data_file() function now supports market selection:")
    print("\n  find_data_file(market='ES')  -> looks for ES_D1M.csv first")
    print("  find_data_file(market='NQ')  -> looks for NQ_D1M.csv first")
    print("  find_data_file(market=None)  -> falls back to old behavior")

    print("\n\nPriority order when market='ES':")
    print("  1. data/ES_D1M.csv          (market-specific)")
    print("  2. data/D1M.csv             (generic)")
    print("  3. data/es_training_*.csv   (legacy)")
    print("  4. data/*_D1M.csv           (wildcard)")

    print("\n\nSecond-level data detection:")
    print("  - Looks for {MARKET}_D1S.csv first")
    print("  - Falls back to D1S.csv (generic)")
    print("  - Shows warning if not found (second-level is optional)")


def main():
    print("=" * 80)
    print("MARKET DETECTION AND SELECTION - DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows how the training scripts now handle market selection.")
    print("The system intelligently detects available markets and prompts users")
    print("when multiple options exist.")

    demo_scenario_1()
    demo_scenario_2()
    demo_scenario_3()
    demo_data_file_detection()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe training scripts now support:")
    print("  [OK] Auto-detection of all available market data")
    print("  [OK] Interactive selection when multiple markets exist")
    print("  [OK] Auto-selection when only one market exists")
    print("  [OK] Clear error messages when no data is found")
    print("  [OK] Backward compatibility with existing data files")
    print("\nTo test interactively, run:")
    print("  python3 src/train_phase1.py --test")
    print("  python3 src/train_phase2.py --test")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
