import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import concurrent.futures


CONFIG = {

    "activity_chains_path": "/local/scratch/sli657/zllmagent/activity_chain/activity_chain_numosim_gen/generated_chains_numosim.jsonl",
    "trajectory_path": "/local/scratch/sli657/zllmagent/raw_data/numosim_stay_points_raw.parquet",
    "mapping_path": "/local/scratch/sli657/zllmagent/raw_data/numosim_raw_augmented_mapping.parquet",
    
    "output_path": "/local/scratch/sli657/zllmagent/gen_data/gen_traj_baseline_numosim.jsonl",
    
    "max_similar_agents": 10,
    
    "exploration_weight": 0.3, 
    
    "num_workers": 32,
    
    "seed": 42,
    "default_start_hour": 8, 
    "min_travel_minutes": 10,
    "max_travel_minutes": 45,
}


def load_data():
    print("--- 1. Loading Data ---")
    
    chains = []
    if os.path.exists(CONFIG['activity_chains_path']):
        with open(CONFIG['activity_chains_path'], 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chains.append(json.loads(line))
    print(f"Loaded {len(chains)} daily activity plans.")

    print(f"Loading trajectories from {CONFIG['trajectory_path']}...")
    traj_df = pd.read_parquet(CONFIG['trajectory_path'])
    
    traj_df['agent_id'] = traj_df['agent_id'].fillna(-1).astype(int)
    traj_df['poi_id'] = traj_df['poi_id'].fillna(-1).astype(int)
    traj_df['act_type'] = traj_df['act_type'].astype(str).str.strip()
    
    traj_df['start_datetime'] = pd.to_datetime(traj_df['start_datetime'])
    traj_df['end_datetime'] = pd.to_datetime(traj_df['end_datetime'])
    traj_df['duration_mins'] = (traj_df['end_datetime'] - traj_df['start_datetime']).dt.total_seconds() / 60.0
    
    print("Building POI lookup dictionary...")
    poi_lookup = traj_df[['poi_id', 'latitude', 'longitude']].drop_duplicates('poi_id').set_index('poi_id').to_dict('index')
    
    print(f"Loading similar mapping from {CONFIG['mapping_path']}...")
    mapping_df = pd.read_parquet(CONFIG['mapping_path'])
    
    similar_mapping = {}
    for target, group in mapping_df.groupby('target_agent_id'):

        sorted_sims = group.sort_values('score', ascending=False)['similar_agent_id'].tolist()

        sorted_sims = [x for x in sorted_sims if x != target]

        similar_mapping[target] = sorted_sims[:CONFIG['max_similar_agents']]

    return chains, traj_df, poi_lookup, similar_mapping

def get_transition_prob(df, current_poi, prev_poi):
    if prev_poi is None: return 0.0
    next_pois = df['poi_id'].shift(-1)
    mask = (df['poi_id'] == prev_poi)
    if not mask.any(): return 0.0
    total_transitions = mask.sum()
    targets = next_pois[mask]
    count = (targets == current_poi).sum()
    return count / total_transitions

def get_frequency_prob(df, activity, poi_id):
    act_subset = df[df['act_type'] == activity]
    if len(act_subset) == 0: return 0.0
    count = (act_subset['poi_id'] == poi_id).sum()
    return count / len(act_subset)


def process_single_task(task_data):
    task, personal_df, similarity_df, poi_lookup, cfg = task_data
    
    agent_id = task['agent_id']
    target_date_str = task['date']

    chain_list = [str(act).strip() for act in task['activity_chain']]
    
    result_record = {
        "agent_id": agent_id,
        "date": target_date_str,
        "trajectory": []
    }
    
    if not chain_list:
        return result_record
        
    base_date = pd.to_datetime(target_date_str)
    start_offset = random.randint(-30, 60)
    current_time = base_date + timedelta(hours=cfg['default_start_hour']) + timedelta(minutes=start_offset)
    
    prev_poi_id = None
    
    for seq, activity in enumerate(chain_list):
        is_last_activity = (seq == len(chain_list) - 1)
        
        candidates = set()
        if not personal_df.empty:
            candidates.update(personal_df[personal_df['act_type'] == activity]['poi_id'].unique())
        if not similarity_df.empty:
            candidates.update(similarity_df[similarity_df['act_type'] == activity]['poi_id'].unique())
            
        final_scores = {}
        
        if candidates:
            raw_p_scores = {} 
            raw_s_scores = {} 
            
            for poi in candidates:

                p_raw = 0.0
                if not personal_df.empty:
                    p_freq = get_frequency_prob(personal_df, activity, poi)
                    p_trans = get_transition_prob(personal_df, poi, prev_poi_id) if prev_poi_id is not None else 0.0
                    p_raw = p_freq if prev_poi_id is None else (0.5 * p_freq + 0.5 * p_trans)
                raw_p_scores[poi] = p_raw
                
                s_raw = 0.0
                if not similarity_df.empty:
                    s_raw = get_frequency_prob(similarity_df, activity, poi)
                raw_s_scores[poi] = s_raw

            max_p = max(raw_p_scores.values()) if raw_p_scores else 0
            max_s = max(raw_s_scores.values()) if raw_s_scores else 0
            
            alpha = cfg['exploration_weight']
            
            for poi in candidates:
                p_norm = raw_p_scores[poi] / max_p if max_p > 0 else 0.0
                s_norm = raw_s_scores[poi] / max_s if max_s > 0 else 0.0
                
                score = (1 - alpha) * p_norm + alpha * s_norm
                if score > 0:
                    final_scores[poi] = score
        
        loc_info = None
        if final_scores:
            pois = list(final_scores.keys())
            weights = list(final_scores.values())
            total_w = sum(weights)
            if total_w > 0:
                probs = [w/total_w for w in weights]
                selected_poi = np.random.choice(pois, p=probs)
                if selected_poi in poi_lookup:
                    loc_info = {
                        "poi_id": int(selected_poi),
                        "lat": poi_lookup[selected_poi]['latitude'],
                        "lon": poi_lookup[selected_poi]['longitude']
                    }
        
        if not loc_info and candidates:
            rand_poi = random.choice(list(candidates))
            if rand_poi in poi_lookup:
                loc_info = {
                    "poi_id": int(rand_poi), 
                    "lat": poi_lookup[rand_poi]['latitude'], 
                    "lon": poi_lookup[rand_poi]['longitude']
                }
            
        if not loc_info:
            loc_info = {"poi_id": -1, "lat": 34.0522, "lon": -118.2437} # LA default

        time_ref_df = personal_df if not personal_df.empty else similarity_df
        
        avg_dur = 60.0
        std_dur = 15.0
        if not time_ref_df.empty:
            act_subset = time_ref_df[time_ref_df['act_type'] == activity]
            if not act_subset.empty:
                avg_dur = act_subset['duration_mins'].mean()
                std_dur = act_subset['duration_mins'].std()
                if pd.isna(std_dur) or std_dur == 0: 
                    std_dur = 15.0
                if pd.isna(avg_dur):
                    avg_dur = 60.0
        
        travel_time = random.randint(cfg['min_travel_minutes'], cfg['max_travel_minutes'])
        start_time = current_time + timedelta(minutes=travel_time)
        
        if is_last_activity:
            end_time = base_date + timedelta(days=1)
            
            if start_time >= end_time:
                start_time = end_time - timedelta(minutes=30)
            duration = int((end_time - start_time).total_seconds() / 60)
        else:
            duration = max(10, int(np.random.normal(avg_dur, std_dur)))
            end_time = start_time + timedelta(minutes=duration)
        
        traj_point = {
            "seq": seq + 1,
            "activity": activity,
            "poi_id": loc_info['poi_id'],
            "lat": round(loc_info['lat'], 6),
            "lon": round(loc_info['lon'], 6),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": duration,
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        result_record['trajectory'].append(traj_point)
        
        current_time = end_time
        prev_poi_id = loc_info['poi_id']
        
    return result_record


def main():
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    chains, traj_df, poi_lookup, similar_mapping = load_data()
    
    print("--- 2. Preparing Data Slices for Parallel Workers ---")

    agent_groups = dict(tuple(traj_df.groupby('agent_id')))
    
    task_payloads = []
    
    for task in tqdm(chains, desc="Packaging Tasks"):
        ag_id = task['agent_id']
        
        personal_df = agent_groups.get(ag_id, pd.DataFrame())
        
        similarity_df = pd.DataFrame()
        sim_ids = similar_mapping.get(ag_id, [])
        
        if sim_ids:
            sim_dfs = [agent_groups[s_id] for s_id in sim_ids if s_id in agent_groups]
            if sim_dfs:
                similarity_df = pd.concat(sim_dfs, ignore_index=True)
                
        task_payloads.append((task, personal_df, similarity_df, poi_lookup, CONFIG))

    print(f"--- 3. Starting Parallel Generation ({CONFIG['num_workers']} Workers) ---")
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
        for res in tqdm(executor.map(process_single_task, task_payloads), total=len(task_payloads)):
            results.append(res)

    print(f"--- 4. Saving Results to {CONFIG['output_path']} ---")
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    with open(CONFIG['output_path'], 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
