### stage1: down-sample trajectories and basic processing
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

np.random.seed(42)
random.seed(42)

def process_trajectories(file_path, output_path_full):

    base, ext = os.path.splitext(output_path_full)
    output_path_subset = f"{base}_1week{ext}"

    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    
    print("Preprocessing dates...")
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])
    df['date_normalized'] = df['start_datetime'].dt.normalize()
    
    daily_stats = df.groupby(['agent_id', 'date_normalized']).size().reset_index(name='visit_count')
    daily_stats['weekday'] = daily_stats['date_normalized'].dt.weekday

    print("\n--- Identifying Agent Groups ---")
    
    def get_valid_agents(stats_df, min_v, max_v):
        valid_days = stats_df[(stats_df['visit_count'] > min_v) & (stats_df['visit_count'] < max_v)]

        coverage = valid_days.groupby('agent_id')['weekday'].nunique()

        return set(coverage[coverage == 7].index)

    group_a_ids = get_valid_agents(daily_stats, 5, 11)
    print(f"Group A (Strict: 5 < v < 11) count: {len(group_a_ids)}")

    group_b_ids = get_valid_agents(daily_stats, 4, 11)
    print(f"Group B (Relaxed: 4 < v < 11) count: {len(group_b_ids)}")

    target_total = 1200
    selected_ids = list(group_a_ids)
    needed = target_total - len(selected_ids)
    
    if needed > 0:
        candidates_b = list(group_b_ids - group_a_ids)
        if len(candidates_b) >= needed:
            sampled_b = np.random.choice(candidates_b, size=needed, replace=False)
            selected_ids.extend(sampled_b)
        else:
            print(f"Warning: Not enough agents in Group B. Taking all available.")
            selected_ids.extend(candidates_b)
    
    selected_ids_set = set(selected_ids)
    print(f"Total Selected Agents: {len(selected_ids_set)}")
    
    print("Extracting raw data for selected agents...")
    df_selected = df[df['agent_id'].isin(selected_ids_set)].copy()
    
    print("\n--- Processing Trajectories (Filling gaps & Replacing bad days) ---")
    
    final_records = []

    group_a_check = set(group_a_ids) 
    stats_selected = daily_stats[daily_stats['agent_id'].isin(selected_ids_set)].copy()
    df_selected_indexed = df_selected.set_index(['agent_id', 'date_normalized']).sort_index()

    for agent_id in tqdm(selected_ids, desc="Processing Agents"):

        if agent_id in group_a_check:
            min_limit, max_limit = 5, 11
        else:
            min_limit, max_limit = 4, 11
            
        agent_stats = stats_selected[stats_selected['agent_id'] == agent_id]
        valid_mask = (agent_stats['visit_count'] > min_limit) & (agent_stats['visit_count'] < max_limit)
        good_days_df = agent_stats[valid_mask]
        good_bank = good_days_df.groupby('weekday')['date_normalized'].apply(list).to_dict()
        
        min_date = agent_stats['date_normalized'].min()
        max_date = agent_stats['date_normalized'].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        for target_date in full_date_range:
            weekday = target_date.weekday()
            is_original_valid = False

            current_day_stat = agent_stats[agent_stats['date_normalized'] == target_date]
            if not current_day_stat.empty:
                visit_count = current_day_stat.iloc[0]['visit_count']
                if min_limit < visit_count < max_limit:
                    is_original_valid = True
            
            if is_original_valid:
                try:
                    rows = df_selected_indexed.loc[(agent_id, target_date)]
                    if isinstance(rows, pd.Series):
                        rows = rows.to_frame().T
                    rows = rows.reset_index()
                    final_records.append(rows)
                except KeyError:
                    pass
            else:
                available_dates = good_bank.get(weekday, [])
                if not available_dates:
                    continue 
                
                source_date = random.choice(available_dates)
                try:
                    source_rows = df_selected_indexed.loc[(agent_id, source_date)]
                    if isinstance(source_rows, pd.Series):
                        source_rows = source_rows.to_frame().T
                    
                    new_rows = source_rows.reset_index().copy()
                    
                    time_delta = target_date - source_date
                    new_rows['start_datetime'] = new_rows['start_datetime'] + time_delta
                    new_rows['end_datetime'] = new_rows['end_datetime'] + time_delta
                    new_rows['date_normalized'] = target_date 
                    
                    final_records.append(new_rows)
                except KeyError:
                    pass

    if not final_records:
        print("Error: No records generated.")
        return

    final_df = pd.concat(final_records, ignore_index=True)
    cols_to_keep = ['agent_id', 'poi_id', 'start_datetime', 'end_datetime']
    final_df = final_df[cols_to_keep]
    final_df = final_df.sort_values(by=['agent_id', 'start_datetime'])

    print("\n--- Running Validation ---")
    final_df['date_check'] = final_df['start_datetime'].dt.normalize()
    validation_stats = final_df.groupby(['agent_id', 'date_check']).size().reset_index(name='count')
    
    agent_date_counts = validation_stats.groupby('agent_id')['date_check'].nunique().rename('actual_days')
    agent_time_bounds = validation_stats.groupby('agent_id')['date_check'].agg(['min', 'max'])
    agent_time_bounds['expected_days'] = (agent_time_bounds['max'] - agent_time_bounds['min']).dt.days + 1
    
    comparison = pd.concat([agent_date_counts, agent_time_bounds['expected_days']], axis=1)

    comparison.columns = ['actual_days', 'expected_days']
    
    comparison['diff'] = comparison['expected_days'] - comparison['actual_days']
    
    if (comparison['diff'] != 0).sum() == 0:
        print("SUCCESS: All agents have perfectly continuous daily trajectories.")
    else:
        print(f"FAILURE: Continuity issues found in {(comparison['diff'] != 0).sum()} agents.")

    print(f"\n[1/2] Saving FULL Dataset to: {output_path_full}")
    final_df.drop(columns=['date_check'], inplace=True, errors='ignore')
    final_df.to_parquet(output_path_full, index=False)
    
    print("\n" + "="*50)
    print("Generating 1-Week Subset (Continuous Mon-Sun)")
    print("="*50)
    
    subset_records = []
    
    final_df['date_normalized'] = final_df['start_datetime'].dt.normalize()
    
    for agent_id, group in tqdm(final_df.groupby('agent_id'), desc="Extracting Weeks"):
        dates = sorted(group['date_normalized'].unique())
        total_days = len(dates)
        
        if total_days < 7:
            continue
            
        monday_indices = [i for i, d in enumerate(dates) if d.weekday() == 0]
        
        valid_start_indices = [i for i in monday_indices if (i + 6) < total_days]
        
        if valid_start_indices:
            middle_pos = len(valid_start_indices) // 2
            selected_start_idx = valid_start_indices[middle_pos]
        else:
            selected_start_idx = (total_days - 7) // 2
            
        start_date = dates[selected_start_idx]
        end_date = dates[selected_start_idx + 6]
        
        week_slice = group[(group['date_normalized'] >= start_date) & 
                           (group['date_normalized'] <= end_date)]
        
        subset_records.append(week_slice)
        
    if subset_records:
        df_1week = pd.concat(subset_records, ignore_index=True)
        df_1week = df_1week[cols_to_keep] 
        
        print(f"\n[2/2] Saving 1-WEEK Subset to: {output_path_subset}")
        print(f"      Subset Rows: {len(df_1week)}")
        df_1week.to_parquet(output_path_subset, index=False)
    else:
        print("Error: Could not generate subset.")
        
    print("\nAll processing done.")


if __name__ == "__main__":
    input_file = ""

    output_file_full = ""
    
    process_trajectories(input_file, output_file_full)
    