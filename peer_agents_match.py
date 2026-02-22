import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, cosine
from scipy.stats import wasserstein_distance, entropy
from tqdm import tqdm
import multiprocessing
import warnings
import pickle 

warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG = {
    'input_file': '',
    'merged_colocation_path': '',
    'output_mapping_path': '',
    'output_data_path': '',
    
    'top_k': 40,  
    'num_workers': 64,
    
    'hard_filters': {
        'center_dist_km': 20.0,      
        'rog_tolerance_ratio': 5.0  
    },
    
    'weights': {
        'rog': 3.0,               
        'travel_dist': 1.0,       
        'anchor_stability': 1.5,  
        'time_dist': 2.5,         
        'spatial_entropy': 1.0,   
        'activity_dist': 3.0,     
        'weekly_pattern': 1.0,    
        'transition': 1.0,       
        'colocation': 2.0         
    }
}

def safe_entropy(counts):
    if len(counts) == 0: return 0.0
    return entropy(counts)

def get_distribution(series, bins, normalize=True):
    counts = series.value_counts(normalize=normalize).reindex(bins, fill_value=0).sort_index()
    return counts.values


def extract_features_worker(agent_data):
    """提取单个 Agent 的轨迹特征"""
    agent_id, sub_df = agent_data
    
    if len(sub_df) < 5: 
        return None

    sub_df['start_datetime'] = pd.to_datetime(sub_df['start_datetime'])
    sub_df['hour'] = sub_df['start_datetime'].dt.hour
    sub_df['weekday'] = sub_df['start_datetime'].dt.dayofweek
    sub_df['is_weekend'] = sub_df['weekday'] >= 5
    sub_df['date'] = sub_df['start_datetime'].dt.date

    lat_min, lat_max = sub_df['latitude'].min(), sub_df['latitude'].max()
    lon_min, lon_max = sub_df['longitude'].min(), sub_df['longitude'].max()
    lat_center, lon_center = (lat_min + lat_max)/2, (lon_min + lon_max)/2
    
    dists_sq = (sub_df['latitude'] - lat_center)**2 + (sub_df['longitude'] - lon_center)**2
    rog = np.sqrt(dists_sq.mean())
    
    diffs = np.sqrt(np.diff(sub_df['latitude'])**2 + np.diff(sub_df['longitude'])**2)
    travel_dist = np.sum(diffs)
    
    spatial_entropy = safe_entropy(sub_df['poi_id'].value_counts(normalize=True))
    
    daily_groups = sub_df.groupby('date')
    try:
        first_locs = daily_groups.first()['poi_id']
        last_locs = daily_groups.last()['poi_id']
        anchor_stability = (safe_entropy(first_locs.value_counts()) + safe_entropy(last_locs.value_counts())) / 2.0
    except:
        anchor_stability = 0.0

    time_dist_weekday = get_distribution(sub_df[~sub_df['is_weekend']]['hour'], range(24))
    time_dist_weekend = get_distribution(sub_df[sub_df['is_weekend']]['hour'], range(24))
    weekly_counts = get_distribution(sub_df['weekday'], range(7))

    active_days_count = sub_df['date'].nunique()
    daily_intensity = len(sub_df) / max(1, active_days_count)

    act_dist = sub_df['act_type'].value_counts(normalize=True).to_dict()
    
    transitions = []
    acts = sub_df['act_type'].values
    for i in range(len(acts)-1):
        transitions.append(f"{acts[i]}->{acts[i+1]}")
    trans_dist = pd.Series(transitions).value_counts(normalize=True).head(20).to_dict()

    return {
        'agent_id': agent_id,
        'lat_center': lat_center, 'lon_center': lon_center,
        'rog': rog,
        'travel_dist': travel_dist,
        'spatial_entropy': spatial_entropy,
        'anchor_stability': anchor_stability,
        'time_dist_weekday': time_dist_weekday,
        'time_dist_weekend': time_dist_weekend,
        'weekly_pattern': weekly_counts,
        'daily_intensity': daily_intensity,
        'act_dist': act_dist,
        'trans_dist': trans_dist
    }

def build_feature_pool(df):
    print("--- retrieving datset with all agents ---")
    groups = [(agent_id, group) for agent_id, group in df.groupby('agent_id')]
    
    all_features = []
    with multiprocessing.Pool(CONFIG['num_workers']) as pool:
        for res in tqdm(pool.imap_unordered(extract_features_worker, groups), total=len(groups)):
            if res is not None:
                all_features.append(res)
                
    feature_df = pd.DataFrame(all_features)
    print(f"successfully retireved {len(feature_df)} number of Agent's valid features")
    return feature_df


def calculate_similarity(target, candidate, colocation_map):

    dist_threshold = CONFIG['hard_filters']['center_dist_km'] / 111.0
    center_dist = np.sqrt((target['lat_center'] - candidate['lat_center'])**2 + 
                          (target['lon_center'] - candidate['lon_center'])**2)
    if center_dist > dist_threshold:
        return -1.0

    if target['rog'] > 0 and candidate['rog'] > 0:
        ratio = target['rog'] / candidate['rog']
        limit = CONFIG['hard_filters']['rog_tolerance_ratio']
        if ratio > limit or ratio < (1.0/limit): 
            return -1.0

    scores = {}
    w = CONFIG['weights']
    
    scores['rog'] = np.exp(-abs(np.log1p(target['rog']) - np.log1p(candidate['rog'])))
    scores['travel_dist'] = np.exp(-abs(np.log1p(target['travel_dist']) - np.log1p(candidate['travel_dist'])))
    scores['anchor_stability'] = np.exp(-abs(target['anchor_stability'] - candidate['anchor_stability']))
    scores['spatial_entropy'] = np.exp(-abs(target['spatial_entropy'] - candidate['spatial_entropy']))
    
    if np.sum(target['time_dist_weekday']) > 0 and np.sum(candidate['time_dist_weekday']) > 0:
        wd_score = np.exp(-wasserstein_distance(np.arange(24), np.arange(24), target['time_dist_weekday'], candidate['time_dist_weekday']))
    else: wd_score = 0.5
    if np.sum(target['time_dist_weekend']) > 0 and np.sum(candidate['time_dist_weekend']) > 0:
        we_score = np.exp(-wasserstein_distance(np.arange(24), np.arange(24), target['time_dist_weekend'], candidate['time_dist_weekend']))
    else: we_score = 0.5
    
    scores['time_dist'] = (wd_score + we_score) / 2.0
    scores['weekly_pattern'] = 1 - cosine(target['weekly_pattern'], candidate['weekly_pattern']) if np.any(target['weekly_pattern']) else 0

    all_keys = set(target['act_dist'].keys()) | set(candidate['act_dist'].keys())
    p = [target['act_dist'].get(k, 0) for k in all_keys]
    q = [candidate['act_dist'].get(k, 0) for k in all_keys]
    scores['activity_dist'] = 1 - jensenshannon(p, q) 
    
    trans_keys = set(target['trans_dist'].keys()) | set(candidate['trans_dist'].keys())
    if trans_keys:
        v1 = [target['trans_dist'].get(k, 0) for k in trans_keys]
        v2 = [candidate['trans_dist'].get(k, 0) for k in trans_keys]
        scores['transition'] = 1 - cosine(v1, v2)
    else: 
        scores['transition'] = 0

    coloc_val = 0
    if target['agent_id'] in colocation_map and candidate['agent_id'] in colocation_map[target['agent_id']]:
        coloc_val = colocation_map[target['agent_id']][candidate['agent_id']]
    scores['colocation'] = np.log1p(coloc_val) / 5.0 
    
    final_score = sum(val * w.get(k, 1.0) for k, val in scores.items() if not pd.isna(val))
    return final_score

def load_colocation_index():
    path = CONFIG['merged_colocation_path']
    print(f"--- Loading Colocation: {path} ---")
    if not os.path.exists(path): return {}
    try:
        with open(path, 'rb') as f: return pickle.load(f)
    except: return {}


def main():
    if not os.path.exists(CONFIG['input_file']):
        print(f"wrong cannot find {CONFIG['input_file']}")
        return

    print("loading raw data...")
    raw_df = pd.read_parquet(CONFIG['input_file'])
    
    feature_pool = build_feature_pool(raw_df)
    if feature_pool.empty: return
    
    coloc_map = load_colocation_index()
    
    print(f"--- retrieving {len(feature_pool)} num Agent from peers ---")
    retrieval_results = []
    
    candidates_df = feature_pool.copy()
    
    for _, target_feat in tqdm(feature_pool.iterrows(), total=len(feature_pool), desc="matching progress"):
        target_id = target_feat['agent_id']
        
        current_cands = candidates_df[candidates_df['agent_id'] != target_id]
        
        scores = current_cands.apply(
            lambda row: calculate_similarity(target_feat, row, coloc_map), axis=1
        )
        
        valid_indices = scores[scores >= 0].index
        if len(valid_indices) == 0: continue
            
        top_indices = scores.loc[valid_indices].nlargest(CONFIG['top_k']).index
        
        retrieval_results.append({
            'target_agent_id': target_id,
            'similar_agent_id': target_id,
            'score': 999.0
        })
        
        for cand_idx in top_indices:
            retrieval_results.append({
                'target_agent_id': target_id,
                'similar_agent_id': current_cands.loc[cand_idx, 'agent_id'],
                'score': scores.loc[cand_idx]
            })

    result_mapping = pd.DataFrame(retrieval_results)
    os.makedirs(os.path.dirname(CONFIG['output_mapping_path']), exist_ok=True)
    result_mapping.to_parquet(CONFIG['output_mapping_path'], index=False)
    print(f"save matching to: {CONFIG['output_mapping_path']}")

    print("merge extracted trajectories to source_agent_id...")
    
    final_data_df = pd.merge(
        raw_df,
        result_mapping[['target_agent_id', 'similar_agent_id', 'score']],
        left_on='agent_id',
        right_on='similar_agent_id'
    )
    
    final_data_df = final_data_df.drop(columns=['similar_agent_id'])
    final_data_df = final_data_df.rename(columns={'target_agent_id': 'source_agent_id'})
    
    final_data_df.to_parquet(CONFIG['output_data_path'], index=False)
    
    print(f"peer Agent data saved: {CONFIG['output_data_path']}")
    print(f"quantity after augmented {len(final_data_df)} lines")
    print("all done")

if __name__ == "__main__":
    main()
