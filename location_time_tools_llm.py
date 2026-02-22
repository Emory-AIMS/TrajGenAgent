import os
import json
import random
import math
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, TypedDict
from tqdm import tqdm
from datetime import datetime, timedelta
import concurrent.futures

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

CONFIG = {

    "chains_path": "",
    "traj_path": "",
    "mapping_path": "",
    "output_path": "",

    "explore_weight": 0.3,       
    "distance_weight": 0.6,      
    "frequency_weight": 0.4,     
    
    "llm_api_base": "http://localhost:8001/v1", 
    "llm_model_name": "Qwen/Qwen2.5-32B-Instruct",
    
    "num_workers": 64,
    "seed": 42
}

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_and_build_profiles():
    print("--- 1. Loading Raw Data & Cleaning Formats ---")
    tasks = []
    with open(CONFIG['chains_path'], 'r') as f:
        for line in f:
            if line.strip(): tasks.append(json.loads(line))
            
    traj_df = pd.read_parquet(CONFIG['traj_path'])
    
    traj_df['agent_id'] = traj_df['agent_id'].fillna(-1).astype(int)
    traj_df['poi_id'] = traj_df['poi_id'].fillna(-1).astype(int)
    traj_df['act_type'] = traj_df['act_type'].astype(str).str.strip()
    
    traj_df['start_datetime'] = pd.to_datetime(traj_df['start_datetime'])
    traj_df['end_datetime'] = pd.to_datetime(traj_df['end_datetime'])
    traj_df['duration_mins'] = (traj_df['end_datetime'] - traj_df['start_datetime']).dt.total_seconds() / 60.0
    traj_df['weekday'] = traj_df['start_datetime'].dt.dayofweek
    traj_df['hour'] = traj_df['start_datetime'].dt.hour
    
    poi_lookup = traj_df[['poi_id', 'latitude', 'longitude']].drop_duplicates('poi_id').set_index('poi_id').to_dict('index')
    
    mapping_df = pd.read_parquet(CONFIG['mapping_path'])
    mapping_df['target_agent_id'] = mapping_df['target_agent_id'].astype(int)
    mapping_df['similar_agent_id'] = mapping_df['similar_agent_id'].astype(int)
    similar_mapping = mapping_df.groupby('target_agent_id')['similar_agent_id'].apply(list).to_dict()
    
    print("--- 2. Building Statistical Profiles ---")
    agent_profiles = {}
    
    traj_df = traj_df.sort_values(['agent_id', 'start_datetime'])
    traj_df['next_poi'] = traj_df.groupby('agent_id')['poi_id'].shift(-1)
    traj_df['next_act'] = traj_df.groupby('agent_id')['act_type'].shift(-1)
    traj_df['next_start'] = traj_df.groupby('agent_id')['start_datetime'].shift(-1)
    
    def calc_dist(row):
        if pd.isna(row['next_poi']) or row['next_poi'] == -1 or row['poi_id'] == -1: 
            return np.nan
        lat1, lon1 = poi_lookup[row['poi_id']]['latitude'], poi_lookup[row['poi_id']]['longitude']
        lat2, lon2 = poi_lookup[row['next_poi']]['latitude'], poi_lookup[row['next_poi']]['longitude']
        return haversine_distance(lat1, lon1, lat2, lon2)
        
    traj_df['trans_dist_km'] = traj_df.apply(calc_dist, axis=1)
    traj_df['trans_time_hr'] = (traj_df['next_start'] - traj_df['end_datetime']).dt.total_seconds() / 3600.0
    traj_df['speed_kmh'] = np.where(traj_df['trans_time_hr'] > 0, traj_df['trans_dist_km'] / traj_df['trans_time_hr'], np.nan)
    
    grouped = traj_df.groupby('agent_id')
    for agent_id, group in tqdm(grouped, desc="Indexing Agents"):
        profile = {
            "act_freq": {}, "poi_by_act": {}, "avg_dur_by_act": {},
            "trans_dist_by_act_pair": {}, "trans_speed_by_act_pair": {}, "first_starts": {}
        }
        
        act_counts = group['act_type'].value_counts()
        if not act_counts.empty:
            profile['act_freq'] = (act_counts / act_counts.sum()).to_dict()
            
        profile['avg_dur_by_act'] = group.groupby('act_type')['duration_mins'].mean().to_dict()
        profile['trans_dist_by_act_pair'] = group.groupby(['act_type', 'next_act'])['trans_dist_km'].mean().to_dict()
        profile['trans_speed_by_act_pair'] = group.groupby(['act_type', 'next_act'])['speed_kmh'].mean().to_dict()
        profile['first_starts'] = group.groupby(['weekday', 'act_type'])['hour'].min().to_dict()
        
        poi_counts = group.groupby(['act_type', 'poi_id']).size()
        for (act, poi), count in poi_counts.items():
            if act not in profile["poi_by_act"]:
                profile["poi_by_act"][act] = {}
            profile["poi_by_act"][act][poi] = count
            
        for act in profile["poi_by_act"]:
            total = sum(profile["poi_by_act"][act].values())
            profile["poi_by_act"][act] = {p: float(v)/total for p, v in profile["poi_by_act"][act].items()}
            
        agent_profiles[int(agent_id)] = profile

    return tasks, agent_profiles, similar_mapping, poi_lookup


class AgentState(TypedDict):
    agent_id: int
    date: str
    chain: List[str]
    current_idx: int
    trajectory: List[Dict]
    current_time: datetime
    

def build_workflow(profiles_db, mapping_db, lookup_db):
    
    def location_node(state: AgentState) -> AgentState:
        idx = state['current_idx']
        activity = state['chain'][idx]
        agent_id = int(state['agent_id'])
        profile = profiles_db.get(agent_id, {})
        
        candidates = set()
        personal_pois = profile.get("poi_by_act", {}).get(activity, {})
        candidates.update(personal_pois.keys())
        
        similar_agents = mapping_db.get(agent_id, [])
        sim_pois = {}
        valid_sims = 0
        for sim_id in similar_agents[:10]:
            sim_prof = profiles_db.get(sim_id, {}).get("poi_by_act", {}).get(activity, {})
            if sim_prof:
                valid_sims += 1
                for p, prob in sim_prof.items():
                    sim_pois[p] = sim_pois.get(p, 0) + prob
                    candidates.add(p)
                    
        if valid_sims > 0:
            sim_pois = {p: prob / valid_sims for p, prob in sim_pois.items()}
                
        if not candidates:
            avail_acts = list(profile.get("poi_by_act", {}).keys())
            print(f"\n[FATAL FALLBACK] Agent {agent_id} -> '{activity}'.")
            print(f"   Reason: Not in personal history ({avail_acts}) AND not in similar agents' history!")
            state['trajectory'].append({"seq": idx + 1, "poi_id": -1, "lat": 0.0, "lon": 0.0, "activity": activity})
            return state

        final_scores = {}
        prev_poi_id = state['trajectory'][-1]['poi_id'] if idx > 0 else None
        prev_act = state['chain'][idx-1] if idx > 0 else None
        hist_avg_dist = profile.get("trans_dist_by_act_pair", {}).get((prev_act, activity), None)
        
        for poi in candidates:
            p_freq = personal_pois.get(poi, 0.0)
            s_freq = sim_pois.get(poi, 0.0)
            freq_score = (1 - CONFIG['explore_weight']) * p_freq + CONFIG['explore_weight'] * s_freq
            
            dist_score = 1.0 
            if prev_poi_id and prev_poi_id != -1 and poi != -1 and hist_avg_dist is not None:
                lat1, lon1 = lookup_db[prev_poi_id]['latitude'], lookup_db[prev_poi_id]['longitude']
                lat2, lon2 = lookup_db[poi]['latitude'], lookup_db[poi]['longitude']
                actual_dist = haversine_distance(lat1, lon1, lat2, lon2)
                dist_score = math.exp(-0.5 * abs(actual_dist - hist_avg_dist))
                
            if prev_poi_id:
                final_scores[poi] = CONFIG['frequency_weight'] * freq_score + CONFIG['distance_weight'] * dist_score
            else:
                final_scores[poi] = freq_score

        pois = list(final_scores.keys())
        weights = np.array(list(final_scores.values()))
        if weights.sum() == 0: weights = np.ones(len(weights))
        probs = weights / weights.sum()
        selected_poi = int(np.random.choice(pois, p=probs))
        
        state['trajectory'].append({
            "seq": idx + 1,
            "activity": activity,
            "poi_id": selected_poi,
            "lat": lookup_db[selected_poi]['latitude'],
            "lon": lookup_db[selected_poi]['longitude']
        })
        return state

    def time_travel_node(state: AgentState) -> AgentState:
        idx = state['current_idx']
        activity = state['chain'][idx]
        agent_id = int(state['agent_id'])
        profile = profiles_db.get(agent_id, {})
        current_date = pd.to_datetime(state['date'])
        weekday = current_date.dayofweek
        
        if idx == 0:
            start_hour = profile.get("first_starts", {}).get((weekday, activity), None)
            if start_hour is None: 
                starts = [h for (w, a), h in profile.get("first_starts", {}).items() if w == weekday]
                start_hour = starts[0] if starts else 8
                
            start_min = random.randint(0, 59)
            state['current_time'] = current_date + timedelta(hours=start_hour, minutes=start_min)
            state['trajectory'][idx]['start_time'] = state['current_time'].strftime("%Y-%m-%d %H:%M:%S")
        else:
            prev_act = state['chain'][idx-1]
            prev_poi = state['trajectory'][idx-1]['poi_id']
            curr_poi = state['trajectory'][idx]['poi_id']
            
            if prev_poi == -1 or curr_poi == -1:
                dist_km = 5.0
            else:
                lat1, lon1 = lookup_db[prev_poi]['latitude'], lookup_db[prev_poi]['longitude']
                lat2, lon2 = lookup_db[curr_poi]['latitude'], lookup_db[curr_poi]['longitude']
                dist_km = haversine_distance(lat1, lon1, lat2, lon2)
            
            speed_kmh = profile.get("trans_speed_by_act_pair", {}).get((prev_act, activity), None)
            if speed_kmh is None or pd.isna(speed_kmh):
                speed_kmh = 5.0 if dist_km <= 5.0 else 40.0
                
            travel_mins = int((dist_km / max(speed_kmh, 1.0)) * 60)
            travel_mins = max(5, min(travel_mins, 180))
            
            state['current_time'] = state['current_time'] + timedelta(minutes=travel_mins)
            state['trajectory'][idx]['start_time'] = state['current_time'].strftime("%Y-%m-%d %H:%M:%S")

        return state

    def duration_llm_node(state: AgentState) -> AgentState:
        idx = state['current_idx']
        activity = state['chain'][idx]
        agent_id = int(state['agent_id'])
        profile = profiles_db.get(agent_id, {})
        
        avg_dur = profile.get("avg_dur_by_act", {}).get(activity, 60.0)
        if pd.isna(avg_dur): avg_dur = 60.0
        
        remaining_acts = state['chain'][idx+1:]
        rem_avg_dur_total = sum([profile.get("avg_dur_by_act", {}).get(a, 60.0) for a in remaining_acts])
        
        current_time_str = state['current_time'].strftime("%H:%M")
        mins_since_midnight = state['current_time'].hour * 60 + state['current_time'].minute
        time_budget_left = max(0, 24 * 60 - mins_since_midnight)
        
        llm = ChatOpenAI(
            model=CONFIG["llm_model_name"], 
            api_key="EMPTY", 
            base_url=CONFIG["llm_api_base"],
            temperature=0.3,
            max_tokens=60
        )
        
        prompt_text = f"""
        You are scheduling time for an agent.
        
        Current Time: {current_time_str}
        Current Activity to schedule: {activity} (Historical average: {int(avg_dur)} mins)
        
        Schedule Pressure:
        - Remaining Activities today: {remaining_acts}
        - Estimated time needed for remaining: {int(rem_avg_dur_total)} mins
        - Time until midnight: {time_budget_left} mins
        
        Task: Decide a realistic duration in minutes for '{activity}'.
        Output MUST BE a raw JSON object exactly like this: {{"duration_minutes": 45}}
        """
        
        duration_mins = int(avg_dur)
        for attempt in range(2):
            try:
                res = llm.invoke(prompt_text)
                content = res.content.strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    parsed_json = json.loads(match.group(0))
                    llm_dur = parsed_json.get("duration_minutes")
                    if llm_dur is not None and 5 <= llm_dur <= min(time_budget_left, 720):
                        duration_mins = int(llm_dur)
                        break 
                raise ValueError("Invalid format")
            except Exception:
                if attempt == 1: pass 
                
        state['trajectory'][idx]['duration_min'] = duration_mins
        state['current_time'] = state['current_time'] + timedelta(minutes=duration_mins)
        state['trajectory'][idx]['end_time'] = state['current_time'].strftime("%Y-%m-%d %H:%M:%S")
        
        state['current_idx'] += 1
        return state

    def should_continue(state: AgentState):
        if state['current_idx'] < len(state['chain']):
            return "location_node"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("location_node", location_node)
    workflow.add_node("time_travel_node", time_travel_node)
    workflow.add_node("duration_llm_node", duration_llm_node)
    
    workflow.set_entry_point("location_node")
    workflow.add_edge("location_node", "time_travel_node")
    workflow.add_edge("time_travel_node", "duration_llm_node")
    workflow.add_conditional_edges("duration_llm_node", should_continue)
    
    return workflow.compile()


GLOBAL_APP = None

def process_task(task):
    global GLOBAL_APP
    
    cleaned_chain = [str(act).strip() for act in task['activity_chain']]
    
    initial_state = {
        "agent_id": int(task['agent_id']),
        "date": task['date'],
        "chain": cleaned_chain,
        "current_idx": 0,
        "trajectory": [],
        "current_time": pd.to_datetime(task['date'])
    }
    
    final_state = GLOBAL_APP.invoke(initial_state, config={"recursion_limit": 150})
    
    return {
        "agent_id": final_state['agent_id'],
        "date": final_state['date'],
        "trajectory": final_state['trajectory']
    }

def main():
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    global GLOBAL_APP
    tasks, profiles, similar_mapping, poi_lookup = load_and_build_profiles()
    
    GLOBAL_APP = build_workflow(profiles, similar_mapping, poi_lookup)
    
    print(f"--- 3. Executing LangGraph Workflow (Workers: {CONFIG['num_workers']}) ---")
    
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
        for res in tqdm(executor.map(process_task, tasks), total=len(tasks)):
            results.append(res)
            
    print(f"--- 4. Saving Results to {CONFIG['output_path']} ---")
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    with open(CONFIG['output_path'], 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Workflow Complete!")

if __name__ == "__main__":
    main()
