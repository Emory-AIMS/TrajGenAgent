### stage2: get new continuous POI ID and activity types selection
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

np.random.seed(42)
random.seed(42)

def calculate_duration_mins(row):

    delta = row['end_datetime'] - row['start_datetime']
    return delta.total_seconds() / 60.0

def identify_anchors(traj_df, poi_dict):

    print("  > Identifying User Anchors (Home/Work) based on current file...")
    agent_anchors = {}
    
    traj_df = traj_df.copy()
    traj_df['duration_mins'] = (traj_df['end_datetime'] - traj_df['start_datetime']).dt.total_seconds() / 60.0
    traj_df['hour'] = traj_df['start_datetime'].dt.hour
    
    for agent_id, group in tqdm(traj_df.groupby('agent_id'), desc="  > Analyzing Agents"):
        anchors = {'home': None, 'work': None}
        
        night_mask = (group['hour'] >= 22) | (group['hour'] < 6)
        night_stays = group[night_mask]
        
        if not night_stays.empty:
            home_candidates = night_stays.groupby('poi_id')['duration_mins'].sum().sort_values(ascending=False)
            for pid, dur in home_candidates.items():
                candidates = poi_dict.get(pid, {}).get('act_types', [])
                if 1 in candidates:
                    anchors['home'] = pid
                    break

        day_mask = (group['start_datetime'].dt.weekday < 5) & (group['hour'] >= 9) & (group['hour'] < 17)
        day_stays = group[day_mask]
        
        if not day_stays.empty:
            work_candidates = day_stays.groupby('poi_id')['duration_mins'].sum().sort_values(ascending=False)
            for pid, dur in work_candidates.items():
                if pid == anchors['home']: continue
                candidates = poi_dict.get(pid, {}).get('act_types', [])
                if 2 in candidates or 3 in candidates:
                    anchors['work'] = pid
                    break
                    
        agent_anchors[agent_id] = anchors
        
    return agent_anchors

def infer_activity(row, anchors, poi_info):

    pid = row['poi_id']
    candidates = poi_info.get(pid, {}).get('act_types', [14]) 
    
    if len(candidates) == 1:
        return candidates[0]
    
    scores = {act: 1.0 for act in candidates}
    
    duration = row['duration_mins']
    hour = row['hour']
    
    if pid == anchors.get('home') and 1 in candidates:
        scores[1] = scores.get(1, 0) + 1000
    elif pid == anchors.get('work') and (2 in candidates or 3 in candidates):
        if duration > 30:
            if 2 in candidates: scores[2] = scores.get(2, 0) + 500
            if 3 in candidates: scores[3] = scores.get(3, 0) + 500
    
    if duration < 30:
        for act in [15, 8, 5]: 
            if act in scores: scores[act] += 50
        for act in [1, 2, 3, 9]: 
            if act in scores: scores[act] -= 0.5
    elif duration > 180:
        for act in [1, 2, 3]: 
            if act in scores: scores[act] += 50
        for act in [15, 8, 5, 7]: 
            if act in scores: scores[act] = max(0.1, scores[act] - 0.8)

    if (11 <= hour <= 14) or (17 <= hour <= 20):
        if 7 in scores: scores[7] += 30
    if (7 <= hour <= 9) or (16 <= hour <= 18):
        if 15 in scores: scores[15] += 20
    if (hour >= 22) or (hour < 6):
        if 1 in scores: scores[1] += 50

    acts = list(scores.keys())
    weights = [max(0.1, scores[a]) for a in acts]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    
    return np.random.choice(acts, p=probs)

def process_single_file(traj_path, poi_path, output_path):

    print("\n" + "="*60)
    print(f"PROCESSING FILE: {traj_path}")
    print("="*60)
    
    print("1. Loading Data...")
    traj_df = pd.read_parquet(traj_path)
    poi_df = pd.read_parquet(poi_path)
    
    print(f"   > Trajectory Rows: {len(traj_df)}")
    print(f"   > Total POIs in DB: {len(poi_df)}")

    print("   > Building POI Index (Optimized)...")
    
    poi_df_indexed = poi_df.set_index('poi_id')
    
    if 'act_types' in poi_df_indexed.columns:

        first_val = poi_df_indexed['act_types'].iloc[0]
        if isinstance(first_val, np.ndarray):
            poi_df_indexed['act_types'] = poi_df_indexed['act_types'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            
    cols_needed = ['act_types', 'name', 'latitude', 'longitude']
    poi_dict = poi_df_indexed[cols_needed].to_dict('index')

    traj_df['start_datetime'] = pd.to_datetime(traj_df['start_datetime'])
    traj_df['end_datetime'] = pd.to_datetime(traj_df['end_datetime'])
    
    anchors = identify_anchors(traj_df, poi_dict)
    
    print("2. Inferring Activities...")
    traj_df['duration_mins'] = (traj_df['end_datetime'] - traj_df['start_datetime']).dt.total_seconds() / 60.0
    traj_df['hour'] = traj_df['start_datetime'].dt.hour
    
    tqdm.pandas(desc="  > Calculating")
    traj_df['act_type'] = traj_df.progress_apply(
        lambda row: infer_activity(row, anchors.get(row['agent_id'], {}), poi_dict), axis=1
    )
    
    print("3. Enriching POI Metadata...")
    traj_df['poi_name'] = traj_df['poi_id'].map(lambda x: poi_dict.get(x, {}).get('name'))
    traj_df['latitude'] = traj_df['poi_id'].map(lambda x: poi_dict.get(x, {}).get('latitude'))
    traj_df['longitude'] = traj_df['poi_id'].map(lambda x: poi_dict.get(x, {}).get('longitude'))
    
    print("4. Generating Continuous New POI IDs (Independent System)...")
    
    unique_old_ids = sorted(traj_df['poi_id'].unique())
    total_unique_pois = len(unique_old_ids)
    
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_old_ids)}
    
    traj_df['new_poi_id'] = traj_df['poi_id'].map(id_map)
    
    max_new_id = traj_df['new_poi_id'].max()
    print(f"  > Verification: Total Unique POIs = {total_unique_pois}")
    print(f"  > Verification: Max New ID = {max_new_id}")
    
    if max_new_id == total_unique_pois - 1:
        print("  > SUCCESS: IDs are perfectly continuous [0, N-1].")
    else:
        print(f"  > ERROR: IDs are not continuous! Gap detected.")
        return
    
    cols_to_keep = [
        'agent_id', 'new_poi_id', 'poi_id',
        'start_datetime', 'end_datetime', 
        'act_type', 'poi_name', 'latitude', 'longitude'
    ]
    
    final_df = traj_df[cols_to_keep]
    
    print(f"5. Saving to {output_path}...")
    final_df.to_parquet(output_path, index=False)
    print("Done.\n")
    

if __name__ == "__main__":
    
    path_full_in = ""
    path_1week_in = ""
    
    path_poi = ""
    
    path_full_out = ""
    path_1week_out = ""
    
    if os.path.exists(path_full_in):
        process_single_file(path_full_in, path_poi, path_full_out)
    else:
        print(f"Skipping Full: File not found {path_full_in}")
    
    if os.path.exists(path_1week_in):
        process_single_file(path_1week_in, path_poi, path_1week_out)
    else:
        print(f"Skipping 1-Week: File not found {path_1week_in}")