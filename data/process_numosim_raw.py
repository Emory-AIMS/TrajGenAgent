import pandas as pd
import numpy as np
import os

INPUT_FILE = ""
OUTPUT_FILE = ""

ACT_TYPE_MAP = {
    0: "Transportation",
    1: "Home",
    2: "Work",
    3: "School",
    4: "ChildCare",
    5: "BuyGoods",
    6: "Services",
    7: "EatOut",
    8: "Errands",
    9: "Recreation",
    10: "Exercise",
    11: "Visit",
    12: "HealthCare",
    13: "Religious",
    14: "SomethingElse",
    15: "DropOff"
}

def preprocess_numosim_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    print(f"Reading data from: {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    if 'poi_id' in df.columns:
        df = df.drop(columns=['poi_id'])
    df = df.rename(columns={'new_poi_id': 'poi_id'})
    print("Column rename completed (new_poi_id -> poi_id).")

    print("Mapping activity types to semantic labels...")
    df['act_type'] = df['act_type'].map(ACT_TYPE_MAP)

    if df['act_type'].isnull().any():
        print("Warning: Some act_type values could not be mapped. Please check ACT_TYPE_MAP.")

    print("Checking temporal consistency...")
    df = df.sort_values(by=['agent_id', 'start_datetime']).reset_index(drop=True)

    next_start = df.groupby('agent_id')['start_datetime'].shift(-1)

    violation_mask = (df['end_datetime'] > next_start) & (next_start.notna())
    violation_count = int(violation_mask.sum())

    print(f"Done. Found {violation_count} overlapping-time violations.")

    if violation_count > 0:
        print("Fixing conflicts (random shift between 10 and 20 minutes)...")

        random_offsets_seconds = np.random.randint(600, 1201, size=violation_count)
        random_offsets_td = pd.to_timedelta(random_offsets_seconds, unit='s')

        df.loc[violation_mask, 'end_datetime'] = next_start[violation_mask] - random_offsets_td

        backwards_mask = df['end_datetime'] < df['start_datetime']
        if backwards_mask.any():
            print(f"Warning: {int(backwards_mask.sum())} cases resulted in negative duration. Forced to start + 1 minute.")
            df.loc[backwards_mask, 'end_datetime'] = df.loc[backwards_mask, 'start_datetime'] + pd.Timedelta(minutes=1)

    print(f"Saving processed data to: {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)

    print("Preprocessing completed!")
    print("-" * 30)
    print("Final columns:", df.columns.tolist())
    print("act_type unique samples:", df['act_type'].unique().tolist())
    print("Preview (first 5 rows):")
    print(df[['agent_id', 'start_datetime', 'end_datetime', 'act_type', 'poi_id']].head())


if __name__ == "__main__":
    preprocess_numosim_data()