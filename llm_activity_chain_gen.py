import argparse
import json
import os
import re
import random
import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections import Counter, defaultdict


DEFAULT_CONFIG = {

    "traj_path": "",
    "output_dir": "", 
    
    "model_name_or_path": "Qwen/Qwen2.5-32B-Instruct",
    "gpu_memory_utilization": 0.95, 
    "max_model_len": 8192,
    "enforce_eager": True,
    
    "seed": 42,                   
    "temperature": 0.90,            
    "top_p": 0.95,
    "max_tokens": 1024,
    
    "history_sample_count": 4,      
}


PROMPT_TEMPLATE = """
You are a Stochastic Human Mobility Simulator. Generate a daily activity chain for a specific agent.

**Simulation Context:**
- **Target Date:** {target_date} ({weekday_str})
- **Day Type:** {day_type_str}
- **Random Seed:** {random_id}

**Agent Profile (derived from {day_type_str}s):**
- **Activity Likelihood:** {occurrence_probs}
- **Transition Patterns:** {transition_probs}

**Historical Patterns (STRONG REFERENCE):**
Use the following past examples as a strong template for the daily structure.
{history_examples}

**Mandatory Constraints:**
1. **Vocabulary**: You must ONLY use the following 15 activity types:
   ['Home', 'Work', 'School', 'ChildCare', 'BuyGoods', 'Services', 'EatOut', 'Errands', 'Recreation', 'Exercise', 'Visit', 'HealthCare', 'Religious', 'SomethingElse', 'DropOff']
2. **No Repetition**: Adjacent activities MUST NOT be the same (e.g., ['Home', 'Home'] is invalid).
3. **Home Logic**: 
   - 'Home' usually appears at the end of the day.
   - However, be flexible and adapt to the specific agent's historical patterns if they deviate from this general rule.

**Task:**
Generate a NEW activity chain for {target_date}.
Output ONLY a valid Python list of strings.

**Output:**
"""

def get_day_category(weekday_name: str) -> str:
    if weekday_name in ['Saturday', 'Sunday']:
        return 'Weekend'
    return 'Weekday'

def load_and_process_data(config):
    print("--- Step 1: Loading NumoSim Trajectory Data ---")
    traj_df = pd.read_parquet(config['traj_path'])
    
    traj_df['agent_id'] = pd.to_numeric(traj_df['agent_id'], errors='coerce').fillna(-1).astype(int)
    traj_df['start_datetime'] = pd.to_datetime(traj_df['start_datetime'])
    
    traj_df['date_str'] = traj_df['start_datetime'].dt.date.astype(str)
    traj_df['weekday'] = traj_df['start_datetime'].dt.day_name()
    traj_df['day_category'] = traj_df['weekday'].apply(get_day_category)
    
    return traj_df

def calculate_occurrence_probabilities(chains: List[List[str]]) -> str:
    if not chains: return "Insufficient Data"
    total_days = len(chains)
    act_counts = Counter()
    for chain in chains:
        unique_acts = set(chain)
        for act in unique_acts:
            act_counts[act] += 1
    
    probs = []

    for act, count in act_counts.most_common(8):
        prob = (count / total_days) * 100
        probs.append(f"{act}({prob:.0f}%)")
    return ", ".join(probs)

def calculate_transition_probabilities(chains: List[List[str]]) -> str:
    if not chains: return "  (Insufficient Data)"
    transitions = defaultdict(list)
    for chain in chains:
        for i in range(len(chain) - 1):
            curr_act = chain[i]
            next_act = chain[i+1]
            if curr_act != next_act: 
                transitions[curr_act].append(next_act)
    
    output_lines = []
    top_starts = sorted(transitions.keys(), key=lambda k: len(transitions[k]), reverse=True)
    for start_act in top_starts:
        next_acts = transitions[start_act]
        if not next_acts: continue
        total = len(next_acts)
        counts = Counter(next_acts)
        probs_str = []
        for next_act, count in counts.most_common(3):
            prob = (count / total) * 100
            probs_str.append(f"{next_act}({prob:.0f}%)")
        output_lines.append(f"  {start_act} -> [{', '.join(probs_str)}]")
    return "\n".join(output_lines)

def build_agent_history_index(traj_df):
    print("\n--- Step 2: Indexing Historical Trajectories ---")
    
    agent_index = {}
    
    cols = ['agent_id', 'date_str', 'act_type', 'start_datetime', 'weekday', 'day_category']
    grouped = traj_df[cols].sort_values('start_datetime').groupby(['agent_id', 'date_str'])
    
    for (agent_id, date_val), day_records in tqdm(grouped, desc="Building Index"):
        first_row = day_records.iloc[0]
        w_day = first_row['weekday']
        cat = first_row['day_category']
        
        acts = day_records['act_type'].tolist()
        cleaned_acts = [acts[0]]
        for k in range(1, len(acts)):
            if acts[k] != acts[k-1]: 
                cleaned_acts.append(acts[k])
        
        if agent_id not in agent_index:
            agent_index[agent_id] = {'daily_chains': {}, 'daily_metadata': {}}
            
        agent_index[agent_id]['daily_chains'][date_val] = cleaned_acts
        agent_index[agent_id]['daily_metadata'][date_val] = {'weekday': w_day, 'category': cat}

    print(f"Indexing Complete. Total Unique Agents: {len(agent_index)}")
    return agent_index

def calculate_single_layer_features(chains_list):
    if not chains_list: return "N/A", "N/A"
    occur_probs = calculate_occurrence_probabilities(chains_list)
    trans_probs = calculate_transition_probabilities(chains_list)
    return occur_probs, trans_probs

def get_chains_from_repo(repo, target_weekday, target_category):
    meta = repo['daily_metadata']
    data = repo['daily_chains']
    
    keys = [k for k, m in meta.items() if m['weekday'] == target_weekday]
    
    if len(keys) < 2:
        keys = [k for k, m in meta.items() if m['category'] == target_category]
        
    if not keys:
        keys = list(data.keys())
        
    chains = [data[k] for k in keys]
    return chains

def construct_prompts_for_all_tasks(traj_df, agent_index, config):
    print("\n--- Step 3: Constructing Prompts (From Unique Agent-Date Pairs) ---")
    prompts = []
    metadata_list = [] 
    
    unique_tasks = traj_df[['agent_id', 'date_str']].drop_duplicates()
    sample_count = config['history_sample_count']

    for _, row in tqdm(unique_tasks.iterrows(), total=len(unique_tasks), desc="Building Prompts"):
        agent_id = row['agent_id']
        target_date_str = row['date_str']
        
        target_date = pd.to_datetime(target_date_str).date()
        target_weekday = target_date.strftime("%A")
        target_category = get_day_category(target_weekday)

        repo = agent_index.get(agent_id)
        if not repo: continue 

        relevant_chains = get_chains_from_repo(repo, target_weekday, target_category)
        
        if not relevant_chains: 
            continue 

        occur_probs, trans_probs = calculate_single_layer_features(relevant_chains)

        if len(relevant_chains) >= sample_count:
            selected_indices = np.random.choice(len(relevant_chains), sample_count, replace=False)
            history_chains = [relevant_chains[i] for i in selected_indices]
        else:
            history_chains = relevant_chains

        history_str_list = []
        for chain in history_chains:
            chain_str = " -> ".join(chain)
            history_str_list.append(f"- [{target_weekday} Pattern]: {chain_str}")
        history_examples = "\n".join(history_str_list)

        random_noise_id = str(uuid.uuid4())[:6]

        prompt = PROMPT_TEMPLATE.format(
            weekday_str=target_weekday,
            day_type_str=target_category,
            occurrence_probs=occur_probs,
            transition_probs=trans_probs,
            history_examples=history_examples,
            target_date=target_date_str,
            random_id=random_noise_id
        )

        prompts.append(prompt)
        
        metadata_list.append({
            "agent_id": agent_id,
            "date": target_date_str
        })
    
    return prompts, metadata_list

def main():
    config = DEFAULT_CONFIG.copy()
    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
    
    traj_df = load_and_process_data(config)
    agent_index = build_agent_history_index(traj_df)
    prompts, metadata_list = construct_prompts_for_all_tasks(traj_df, agent_index, config)
    
    generated_results = []

    if len(prompts) > 0:
        print(f"\n--- Step 4: Batch Generating for {len(prompts)} unique agent-days ---")
        
        llm = LLM(
            model=config['model_name_or_path'],
            trust_remote_code=True,
            gpu_memory_utilization=config['gpu_memory_utilization'],
            max_model_len=config['max_model_len'],
            enforce_eager=config['enforce_eager'],
            seed=config['seed']
        )
        
        sampling_params = SamplingParams(
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
            stop=["\n\n"]
        )
        
        outputs = llm.generate(prompts, sampling_params)
        list_pattern = re.compile(r"\[.*?\]", re.DOTALL)
        
        print("\n--- Step 5: Parsing and Saving LLM Outputs ---")
        os.makedirs(config['output_dir'], exist_ok=True)
        output_file_path = os.path.join(config['output_dir'], "generated_chains_numosim.jsonl")
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for i, output in enumerate(outputs):
                meta = metadata_list[i]
                text = output.outputs[0].text.strip()
                final_chain = []
                
                try:
                    match = list_pattern.search(text)
                    if match:
                        chain = eval(match.group(0))
                        if isinstance(chain, list):
                            final_chain = [x for x in chain if x is not Ellipsis]
                        else:
                            final_chain = ["ERROR_FORMAT_NOT_LIST"]
                    else:
                        final_chain = ["ERROR_NO_LIST_FOUND"]
                except Exception:
                    final_chain = ["ERROR_PARSE_EXCEPTION"]
                
                record = {
                    "agent_id": int(meta['agent_id']),
                    "date": meta['date'],
                    "activity_chain": final_chain
                }
                f.write(json.dumps(record) + "\n")
                generated_results.append(record)
                
        print(f"Workflow Complete. Successfully generated and saved {len(generated_results)} records.")
        print(f"Output saved to: {output_file_path}")
    else:
        print("No valid tasks found for generation.")

if __name__ == "__main__":
    main()