#!/usr/bin/env python3
"""
Process 1-second ES futures data for real-time trailing drawdown calculation

Input: Raw 1-second OHLCV data from GLBX
Output: Clean 1-second data compatible with training environments
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
from zoneinfo import ZoneInfo


def load_raw_second_data(file_path):
    """Load raw 1-second OHLCV data"""
    print(f"[1/6] Loading raw 1-second data from {file_path}...")

    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"      Loaded {len(df):,} rows")
        print(f"      Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"      ERROR: {e}")
        return None


def filter_es_contracts(df):
    """Filter for ES futures contracts (ESU5, ESZ5)"""
    print(f"\n[2/6] Filtering for ES futures contracts...")

    es_contracts = ['ESU5', 'ESZ5']
    df_es = df[df['symbol'].isin(es_contracts)].copy()

    print(f"      ESU5 records: {len(df_es[df_es['symbol'] == 'ESU5']):,}")
    print(f"      ESZ5 records: {len(df_es[df_es['symbol'] == 'ESZ5']):,}")
    print(f"      Total ES records: {len(df_es):,}")

    return df_es


def create_continuous_contract(df_es):
    """Create continuous ES data by handling contract rollover"""
    print(f"\n[3/6] Creating continuous contract...")

    # Sort by timestamp
    df_es = df_es.sort_values('ts_event')

    # Find rollover point
    esu5_data = df_es[df_es['symbol'] == 'ESU5']
    esz5_data = df_es[df_es['symbol'] == 'ESZ5']

    if len(esu5_data) > 0 and len(esz5_data) > 0:
        esu5_end = esu5_data['ts_event'].max()
        esz5_start = esz5_data['ts_event'].min()
        print(f"      ESU5 ends: {esu5_end}")
        print(f"      ESZ5 starts: {esz5_start}")

    # Combine (ESU5 first, then ESZ5)
    df_continuous = pd.concat([esu5_data, esz5_data])
    df_continuous = df_continuous.sort_values('ts_event')

    print(f"      Continuous data: {len(df_continuous):,} records")

    return df_continuous


def convert_timestamp_and_timezone(df):
    """Convert timestamp to US/Eastern timezone"""
    print(f"\n[4/6] Converting timezone...")

    # Parse timestamp (utc=True handles the Z suffix and UTC timezone)
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)

    # Convert UTC to US/Eastern using zoneinfo
    eastern = ZoneInfo("America/New_York")
    df['ts_event'] = df['ts_event'].dt.tz_convert(eastern)
    df = df.set_index('ts_event')

    print(f"      Time range: {df.index.min()} to {df.index.max()}")
    print(f"      Timezone: {df.index.tz}")

    return df


def filter_trading_hours(df):
    """Filter for Apex trading hours (8:30 AM - 4:59 PM ET)"""
    print(f"\n[5/6] Filtering for Apex trading hours (8:30 AM - 4:59 PM ET)...")

    # Extract hour and minute
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    # Apex Trading Hours: 8:30 AM - 4:59 PM ET (08:30-16:59)
    mask_rth = (
        ((df['hour'] == 8) & (df['minute'] >= 30)) |
        ((df['hour'] >= 9) & (df['hour'] < 16)) |
        ((df['hour'] == 16) & (df['minute'] <= 59))
    )

    df_rth = df[mask_rth].copy()
    df_rth = df_rth.drop(['hour', 'minute'], axis=1)

    rows_before = len(df)
    rows_after = len(df_rth)

    print(f"      Before filtering: {rows_before:,} records")
    print(f"      After filtering: {rows_after:,} records")
    print(f"      Removed: {rows_before - rows_after:,} records ({((rows_before-rows_after)/rows_before*100):.1f}%)")

    return df_rth


def extract_ohlcv_only(df):
    """Extract only OHLCV columns"""
    print(f"\n[6/6] Extracting OHLCV data...")

    # Keep only OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']

    df_ohlcv = df[required_cols].copy()

    # Convert to numeric
    for col in required_cols:
        df_ohlcv[col] = pd.to_numeric(df_ohlcv[col], errors='coerce')

    print(f"      Final columns: {list(df_ohlcv.columns)}")
    print(f"      Records: {len(df_ohlcv):,}")

    return df_ohlcv


def save_processed_data(df, output_path):
    """Save processed data"""
    print(f"\nSaving processed data...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"      Saved to: {output_path}")
    print(f"      Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"      Rows: {len(df):,}")

    return output_path


def main(input_file=None, output_path=None):
    """
    Main processing function

    Args:
        input_file: Path to input 1s OHLCV CSV file (optional)
        output_path: Path to output processed CSV file (optional)
    """
    print("=" * 80)
    print("ES FUTURES 1-SECOND DATA PROCESSING")
    print("=" * 80)

    # Default paths (for backward compatibility)
    if input_file is None:
        input_dir = "data/GLBX-20251015-FHDV4GDHRX"
        input_file = os.path.join(input_dir, "glbx-mdp3-20250713-20251011.ohlcv-1s.csv")

    if output_path is None:
        output_path = "data/es_second_level_data_raw.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return False

    try:
        # Step 1: Load raw data
        df = load_raw_second_data(input_file)
        if df is None:
            return False

        # Step 2: Filter ES contracts
        df = filter_es_contracts(df)
        if len(df) == 0:
            print("ERROR: No ES contracts found")
            return False

        # Step 3: Create continuous contract
        df = create_continuous_contract(df)

        # Step 4: Convert timezone
        df = convert_timestamp_and_timezone(df)

        # Step 5: Filter trading hours
        df = filter_trading_hours(df)
        if len(df) == 0:
            print("ERROR: No data after filtering trading hours")
            return False

        # Step 6: Extract OHLCV only
        df = extract_ohlcv_only(df)

        # Save
        save_processed_data(df, output_path)

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Output file: {output_path}")
        print(f"Next step: python3 clean_second_data.py")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process 1-second ES futures data')
    parser.add_argument('--input', type=str, help='Input 1s OHLCV CSV file path')
    parser.add_argument('--output', type=str, help='Output processed CSV file path')
    args = parser.parse_args()

    success = main(input_file=args.input, output_path=args.output)
    sys.exit(0 if success else 1)
