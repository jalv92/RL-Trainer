#!/usr/bin/env python3
"""
Test script for market detection and selection functionality.
This simulates what happens when training scripts run.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_utils import detect_available_markets, select_market_for_training

def safe_print(msg):
    """Safe print function for testing."""
    print(msg)

def main():
    print("=" * 80)
    print("TESTING MARKET DETECTION AND SELECTION")
    print("=" * 80)

    # Test 1: Detect markets
    print("\nTest 1: Detecting available markets...")
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    markets = detect_available_markets(data_dir)

    print(f"\nDetected {len(markets)} market(s):")
    for market in markets:
        print(f"  Market: {market['market']}")
        print(f"    Minute file: {market['minute_file']}")
        print(f"    Second file: {market['second_file']}")
        print(f"    Has second: {market['has_second']}")
        print(f"    Path: {market['path']}")
        print()

    # Test 2: Simulate auto-selection (only one market)
    print("\nTest 2: Simulating auto-selection (single market scenario)...")
    single_market = [markets[0]] if markets else []
    selected = select_market_for_training(single_market, safe_print)
    if selected:
        print(f"\nAuto-selected: {selected['market']}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNOTE: Interactive market selection (multiple markets) requires user input.")
    print("      This can be tested by running train_phase1.py or train_phase2.py")
    print("      in test mode with --test flag.")

if __name__ == '__main__':
    main()
