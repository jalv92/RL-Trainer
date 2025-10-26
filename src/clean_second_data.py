#!/usr/bin/env python3
"""
Clean 1-second ES futures data for trailing drawdown calculation

Validates and removes:
- NaN values
- Duplicates
- Anomalous prices
- Invalid volumes
- Corrupted rows
"""

import pandas as pd
import numpy as np
import os
import sys


def load_processed_data(file_path):
    """Load processed but uncleaned 1-second data"""
    print(f"[1/5] Loading processed data...")

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"      Loaded {len(df):,} rows")
        return df
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def remove_nan_values(df):
    """Remove rows with NaN values"""
    print(f"\n[2/5] Removing NaN values...")

    rows_before = len(df)

    # Check for NaN in any column
    df = df.dropna()

    rows_after = len(df)
    removed = rows_before - rows_after

    print(f"      Before: {rows_before:,} rows")
    print(f"      After: {rows_after:,} rows")
    print(f"      Removed: {removed:,} rows ({(removed/rows_before*100):.2f}%)")

    return df


def remove_duplicates(df):
    """Remove duplicate timestamps"""
    print(f"\n[3/5] Removing duplicates...")

    rows_before = len(df)

    # Remove duplicate timestamps (keep first)
    df = df[~df.index.duplicated(keep='first')]

    rows_after = len(df)
    removed = rows_before - rows_after

    print(f"      Before: {rows_before:,} rows")
    print(f"      After: {rows_after:,} rows")
    print(f"      Removed: {removed:,} duplicates ({(removed/rows_before*100):.2f}%)")

    return df


def validate_prices(df):
    """Validate price values"""
    print(f"\n[4/5] Validating prices...")

    rows_before = len(df)

    # ES typically trades between 1000-10000
    min_price = 500
    max_price = 15000

    # Check OHLC in valid range
    valid_prices = (
        (df['open'] >= min_price) & (df['open'] <= max_price) &
        (df['high'] >= min_price) & (df['high'] <= max_price) &
        (df['low'] >= min_price) & (df['low'] <= max_price) &
        (df['close'] >= min_price) & (df['close'] <= max_price)
    )

    # Check high >= low
    valid_ohlc = (df['high'] >= df['low'])

    # Check volume is positive
    valid_volume = (df['volume'] > 0)

    # Combine all checks
    valid_rows = valid_prices & valid_ohlc & valid_volume

    # Report violations
    price_violations = (~valid_prices).sum()
    ohlc_violations = (~valid_ohlc).sum()
    volume_violations = (~valid_volume).sum()

    if price_violations > 0:
        print(f"      Price range violations: {price_violations:,}")
    if ohlc_violations > 0:
        print(f"      High < Low violations: {ohlc_violations:,}")
    if volume_violations > 0:
        print(f"      Invalid volume: {volume_violations:,}")

    df = df[valid_rows].copy()

    rows_after = len(df)
    removed = rows_before - rows_after

    print(f"      Before: {rows_before:,} rows")
    print(f"      After: {rows_after:,} rows")
    print(f"      Removed: {removed:,} rows ({(removed/rows_before*100):.2f}%)")

    return df


def detect_price_jumps(df, threshold_pct=5.0):
    """Detect anomalous price jumps"""
    print(f"\n[5/5] Detecting anomalous price jumps (>{threshold_pct}%)...")

    rows_before = len(df)

    # Calculate price changes
    price_change_pct = np.abs(df['close'].pct_change() * 100)

    # Find jumps exceeding threshold
    jumps = price_change_pct > threshold_pct

    # Don't remove first row (can't have change)
    jumps.iloc[0] = False

    n_jumps = jumps.sum()

    if n_jumps > 0:
        print(f"      Found {n_jumps:,} potential anomalies")
        print(f"      Max jump: {price_change_pct.max():.2f}%")

        # Remove jumped rows
        df = df[~jumps].copy()

    rows_after = len(df)
    removed = rows_before - rows_after

    print(f"      Before: {rows_before:,} rows")
    print(f"      After: {rows_after:,} rows")
    print(f"      Removed: {removed:,} rows ({(removed/rows_before*100):.2f}%)")

    return df


def save_cleaned_data(df, output_path):
    """Save cleaned data"""
    print(f"\nSaving cleaned data...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"      Saved to: {output_path}")
    print(f"      Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"      Rows: {len(df):,}")
    print(f"      Time range: {df.index.min()} to {df.index.max()}")

    return output_path


def main(input_path=None, output_path=None):
    """
    Main cleaning function

    Args:
        input_path: Path to input raw CSV file (optional)
        output_path: Path to output cleaned CSV file (optional)
    """
    print("=" * 80)
    print("ES FUTURES 1-SECOND DATA CLEANING")
    print("=" * 80)

    # Default paths (for backward compatibility)
    if input_path is None:
        input_path = "data/es_second_level_data_raw.csv"
    if output_path is None:
        output_path = "data/es_second_level_data.csv"

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print(f"Run process_second_data.py first")
        return False

    try:
        # Step 1: Load data
        df = load_processed_data(input_path)
        if df is None:
            return False

        # Step 2: Remove NaN
        df = remove_nan_values(df)

        # Step 3: Remove duplicates
        df = remove_duplicates(df)

        # Step 4: Validate prices
        df = validate_prices(df)

        # Step 5: Detect price jumps
        df = detect_price_jumps(df, threshold_pct=5.0)

        # Save cleaned data
        save_cleaned_data(df, output_path)

        print("\n" + "=" * 80)
        print("CLEANING COMPLETE")
        print("=" * 80)
        print(f"Cleaned output: {output_path}")
        print(f"Data ready for trailing drawdown calculation")

        # Delete raw intermediate file
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Removed intermediate file: {input_path}")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Clean 1-second ES futures data')
    parser.add_argument('--input', type=str, help='Input raw CSV file path')
    parser.add_argument('--output', type=str, help='Output cleaned CSV file path')
    args = parser.parse_args()

    success = main(input_path=args.input, output_path=args.output)
    sys.exit(0 if success else 1)
