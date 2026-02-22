import os
import json
import pandas as pd
from tqdm import tqdm

CONFIG = {
    "input_traj_path": "",
    "output_chain_path": "numosim_activity_chains.jsonl"
}

def extract_activity_chains():
    if not os.path.exists(CONFIG["input_traj_path"]):
        print(f"cannot find {CONFIG['input_traj_path']}")
        return

    print("--- Step 1: Loading Raw Trajectory Data ---")
    df = pd.read_parquet(CONFIG["input_traj_path"])
    
    print("Processing timestamps and dates...")
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])

    df['date_str'] = df['start_datetime'].dt.date.astype(str)
    
    print("Sorting data chronologically...")
    df.sort_values(by=['agent_id', 'start_datetime'], inplace=True)
    
    print(f"Loaded {len(df)} records. Grouping by Agent and Date...")
    

    grouped = df.groupby(['agent_id', 'date_str'])
    
    extracted_records = []
    

    for (agent_id, date_str), group_data in tqdm(grouped, desc="Extracting Chains"):

        raw_acts = group_data['act_type'].tolist()
        
        if not raw_acts:
            continue
            
        cleaned_acts = [raw_acts[0]]
        for i in range(1, len(raw_acts)):
            if raw_acts[i] != raw_acts[i-1]:
                cleaned_acts.append(raw_acts[i])
                
        record = {
            "agent_id": int(agent_id),  
            "date": date_str,
            "activity_chain": cleaned_acts
        }
        
        extracted_records.append(record)

    print("\n--- Step 2: Saving Extracted Chains ---")
    os.makedirs(os.path.dirname(CONFIG["output_chain_path"]), exist_ok=True)
    
    with open(CONFIG["output_chain_path"], 'w', encoding='utf-8') as f:
        for rec in extracted_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Extraction Complete. Saved {len(extracted_records)} daily chains to {CONFIG['output_chain_path']}")

if __name__ == "__main__":
    extract_activity_chains()

