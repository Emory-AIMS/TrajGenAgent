import pandas as pd
import numpy as np
import multiprocessing
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

INPUT_FILE = ""
OUTPUT_FILE = ""

OVERLAP_THRESHOLD_SECONDS = 120 
NUM_WORKERS = 64  
# ===========================================

def process_poi_group(group_data):

    if len(group_data) < 2:
        return {}

    df_sub = group_data[['agent_id', 'start_datetime', 'end_datetime', 'poi_id']]

    merged = pd.merge(df_sub, df_sub, on='poi_id')
    
    merged = merged[merged['agent_id_x'] < merged['agent_id_y']]
    
    if merged.empty:
        return {}

    overlap_start = merged[['start_datetime_x', 'start_datetime_y']].max(axis=1)
    overlap_end = merged[['end_datetime_x', 'end_datetime_y']].min(axis=1)
    
    overlap_duration = (overlap_end - overlap_start).dt.total_seconds()
    
    valid_matches = merged[overlap_duration >= OVERLAP_THRESHOLD_SECONDS]
    
    if valid_matches.empty:
        return {}

    pair_counts = valid_matches.groupby(['agent_id_x', 'agent_id_y']).size().to_dict()
    
    return pair_counts

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"cannot find input: {INPUT_FILE}")
        return

    print("loading...")
    df = pd.read_parquet(INPUT_FILE)
    
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])

    print(f"loading complete, total number: {len(df)}")
    
    print("partition tasks by POI...")
    poi_groups = [group for _, group in df.groupby('poi_id')]
    print(f"total {len(poi_groups)} POI to be processed")


    print(f"start parallel computing (Worker num: {NUM_WORKERS})...")
    coloc_pairs = defaultdict(int)
    
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_poi_group, poi_groups), total=len(poi_groups)))
    
    print("merge all results...")
    for res_dict in results:
        for (agent_a, agent_b), count in res_dict.items():

            coloc_pairs[(agent_a, agent_b)] += count

    print("make double ended indexing...")
    final_coloc_map = defaultdict(lambda: defaultdict(int))
    
    total_edges = 0
    for (agent_a, agent_b), count in coloc_pairs.items():

        final_coloc_map[agent_a][agent_b] = count
        final_coloc_map[agent_b][agent_a] = count
        total_edges += 1

    output_data = {k: dict(v) for k, v in final_coloc_map.items()}

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(output_data, f)
        

    print(f"{total_edges} num of agent colocation")
    print(f"saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()