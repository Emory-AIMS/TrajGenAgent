import os
import json
import pytz
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import concurrent.futures
from datetime import timedelta


CONFIG = {
    "generated_jsonl_path": "",
    "raw_parquet_path": "", 
    "output_parquet_path": "",
    
    "timezone_offset": 540, 
    "num_workers": 64,
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def extract_cross_day_stats():
    print("--- 1. Extracting Cross-Day Travel Stats from Raw Data ---")
    raw_df = pd.read_parquet(CONFIG['raw_parquet_path'])
    
    raw_df['start_datetime'] = pd.to_datetime(raw_df['start_datetime'])
    raw_df['end_datetime'] = pd.to_datetime(raw_df['end_datetime'])
    raw_df = raw_df.sort_values(['agent_id', 'start_datetime'])
    
    raw_df['date'] = raw_df['start_datetime'].dt.date
    
    raw_df['next_poi'] = raw_df.groupby('agent_id')['poi_id'].shift(-1)
    raw_df['next_act'] = raw_df.groupby('agent_id')['act_type'].shift(-1)
    raw_df['next_start'] = raw_df.groupby('agent_id')['start_datetime'].shift(-1)
    raw_df['next_date'] = raw_df.groupby('agent_id')['date'].shift(-1)
    
    cross_mask = (raw_df['date'] != raw_df['next_date']) & (~raw_df['next_start'].isna())
    cross_df = raw_df[cross_mask].copy()
    
    cross_df['travel_time_hr'] = (cross_df['next_start'] - cross_df['end_datetime']).dt.total_seconds() / 3600.0
    
    poi_lookup = raw_df.drop_duplicates('poi_id').set_index('poi_id')[['latitude', 'longitude']].to_dict('index')
    
    def calc_dist(row):
        if pd.isna(row['next_poi']): return np.nan
        lat1, lon1 = poi_lookup[row['poi_id']]['latitude'], poi_lookup[row['poi_id']]['longitude']
        lat2, lon2 = poi_lookup[row['next_poi']]['latitude'], poi_lookup[row['next_poi']]['longitude']
        return haversine_distance(lat1, lon1, lat2, lon2)
        
    cross_df['dist_km'] = cross_df.apply(calc_dist, axis=1)
    cross_df['speed_kmh'] = np.where(cross_df['travel_time_hr'] > 0, cross_df['dist_km'] / cross_df['travel_time_hr'], np.nan)
    

    avg_tt_poi = cross_df.groupby(['agent_id', 'poi_id', 'next_poi'])['travel_time_hr'].mean().to_dict()
    avg_speed_act = cross_df.groupby(['agent_id', 'act_type', 'next_act'])['speed_kmh'].mean().to_dict()
    
    print(f"Extracted {len(avg_tt_poi)} POI-pair TTs and {len(avg_speed_act)} Act-pair speeds.")
    return avg_tt_poi, avg_speed_act


def process_agent_trajectory(payload):
    agent_df, avg_tt_poi, avg_speed_act = payload
    
    agent_df = agent_df.sort_values('start_datetime').reset_index(drop=True)
    
    for i in range(len(agent_df) - 1):
        curr_row = agent_df.iloc[i]
        next_row = agent_df.iloc[i+1]
        
        if curr_row['start_datetime'].date() != next_row['start_datetime'].date():
            agent_id = curr_row['agent_id']
            poi_A, poi_B = curr_row['poi_id'], next_row['poi_id']
            act_A, act_B = curr_row['act_type'], next_row['act_type']
            
            dist_km = haversine_distance(curr_row['latitude'], curr_row['longitude'], next_row['latitude'], next_row['longitude'])
            
            tt_hr = avg_tt_poi.get((agent_id, poi_A, poi_B), None)
            
            if pd.isna(tt_hr) or tt_hr is None:
                speed = avg_speed_act.get((agent_id, act_A, act_B), None)
                
                if pd.isna(speed) or speed is None:
                    speed = 5.0 if dist_km <= 5.0 else 40.0
                    
                tt_hr = dist_km / speed if speed > 0 else 0
                
            tt_mins = tt_hr * 60
            tt_mins = max(5, min(tt_mins, 720)) 
            
            new_end_time = next_row['start_datetime'] - timedelta(minutes=tt_mins)
            
            if new_end_time <= curr_row['start_datetime']:
                new_end_time = curr_row['start_datetime'] + timedelta(minutes=60)
                
            agent_df.at[agent_df.index[i], 'end_datetime'] = new_end_time
            
    return agent_df


def main():
    avg_tt_poi, avg_speed_act = extract_cross_day_stats()
    
    print("--- 2. Loading Generated JSONL Trajectories ---")
    records = []
    with open(CONFIG['generated_jsonl_path'], 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            ag_id = data['agent_id']
            for t in data['trajectory']:
                records.append({
                    "agent_id": int(ag_id),
                    "new_poi_id": int(t['poi_id']),
                    "poi_id": int(t['poi_id']),
                    "start_datetime": pd.to_datetime(t['start_time']),
                    "end_datetime": pd.to_datetime(t['end_time']),
                    "act_type": str(t['activity']),
                    "poi_name": "", 
                    "latitude": float(t['lat']),
                    "longitude": float(t['lon'])
                })
                
    gen_df = pd.DataFrame(records)
    
    print("--- 3. Bridging End-of-Day Gaps in Parallel ---")
    agent_groups = [group for _, group in gen_df.groupby('agent_id')]
    payloads = [(grp, avg_tt_poi, avg_speed_act) for grp in agent_groups]
    
    processed_dfs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
        for res in tqdm(executor.map(process_agent_trajectory, payloads), total=len(payloads)):
            processed_dfs.append(res)
            
    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    print("--- 4. Formatting Schema and Timezones ---")

    final_df['agent_id'] = final_df['agent_id'].astype('int64')
    final_df['new_poi_id'] = final_df['new_poi_id'].astype('int64')
    final_df['poi_id'] = final_df['poi_id'].astype('int64')
    final_df['act_type'] = final_df['act_type'].astype('object')  
    final_df['poi_name'] = final_df['poi_name'].astype('object')  
    final_df['latitude'] = final_df['latitude'].astype('float64')
    final_df['longitude'] = final_df['longitude'].astype('float64')
    
    tz = pytz.FixedOffset(CONFIG['timezone_offset'])
    final_df['start_datetime'] = final_df['start_datetime'].dt.tz_localize(tz)
    final_df['end_datetime'] = final_df['end_datetime'].dt.tz_localize(tz)
    
    columns_order = [
        "agent_id", "new_poi_id", "poi_id", "start_datetime", 
        "end_datetime", "act_type", "poi_name", "latitude", "longitude"
    ]
    final_df = final_df[columns_order]
    
    print(f"--- 5. Saving to {CONFIG['output_parquet_path']} ---")
    os.makedirs(os.path.dirname(CONFIG['output_parquet_path']), exist_ok=True)
    final_df.to_parquet(CONFIG['output_parquet_path'], engine='pyarrow', index=False)
    
    print("✨ Conversion Complete! Schema and End-of-Day Gaps are perfectly aligned.")

if __name__ == "__main__":
    main()