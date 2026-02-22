import pandas as pd
import scipy.stats
import numpy as np
from collections import Counter
import random
from tqdm import tqdm

class EvalUtils(object):

    @staticmethod
    def filter_zero(arr):
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min_val, max_val, bins):
        distribution, base = np.histogram(
            arr, np.arange(min_val, max_val, float(max_val - min_val) / bins)
        )
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        if len(arr) == 0: return np.zeros(bins), np.zeros(bins)
        arr_range = arr.max() - arr.min()
        if arr_range == 0: arr_range = 1e-9
        arr = (arr - arr.min()) / arr_range
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        return distribution, base[:-1]

    @staticmethod
    def get_js_divergence(p1, p2):
        # normalize
        p1 = p1 / (p1.sum()+1e-14) # avoid zero division
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
             0.5 * scipy.stats.entropy(p2, m)
        return js

class IndividualEval(object):
    
    def __init__(self, horizontal_n, seq_len, traj_file_path1, traj_file_path2, tz_name):

        print(f"Loading generated data from {traj_file_path1}...")
        self.df1 = pd.read_parquet(traj_file_path1)
        print(f"Loading ground truth data from {traj_file_path2}...")
        self.df2 = pd.read_parquet(traj_file_path2)
        
        self.seq_len = seq_len
        self.horizontal_n = horizontal_n
        
        print(f"Aligning timezones to {tz_name}...")
        self.df1 = self.apply_timezone(self.df1, tz_name)
        self.df2 = self.apply_timezone(self.df2, tz_name)
        
        print("Calculating global spatial boundaries...")
        all_lats = pd.concat([self.df1['latitude'], self.df2['latitude']]).dropna()
        all_lons = pd.concat([self.df1['longitude'], self.df2['longitude']]).dropna()
        
        self.min_lat, self.max_lat = all_lats.min(), all_lats.max()
        self.min_lon, self.max_lon = all_lons.min(), all_lons.max()
        
        print(f"Bounds: Lat[{self.min_lat:.4f}, {self.max_lat:.4f}], Lon[{self.min_lon:.4f}, {self.max_lon:.4f}]")

        ratio = (self.max_lat - self.min_lat) / (self.max_lon - self.min_lon) if (self.max_lon - self.min_lon) != 0 else 1
        self.vertical_n = max(int(ratio * horizontal_n), 1)
        self.max_locs = self.horizontal_n * self.vertical_n
        print(f"Grid System: {self.horizontal_n} x {self.vertical_n} = {self.max_locs} max locations")

        self.X, self.Y = self.grid2coor()
        self.max_distance = (self.max_lat - self.min_lat)**2 + (self.max_lon - self.min_lon)**2

        print("Processing Generated Trajectories...")
        self.processed_traj1, self.transition_matrix1 = self.build_sequences_for_df(self.df1)
        print("Processing Ground Truth Trajectories...")
        self.processed_traj2, self.transition_matrix2 = self.build_sequences_for_df(self.df2)
        
        self.formatted_traj1 = np.asarray(self.processed_traj1, dtype='int64')
        self.formatted_traj2 = np.asarray(self.processed_traj2, dtype='int64')

    def apply_timezone(self, df, tz_name):
        """Converts start and end times to the correct local timezone"""
        for col in ['start_datetime', 'end_datetime']:
            if df[col].dt.tz is None:
                # If naive, assume it's UTC, then convert to local
                df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(tz_name)
            else:
                # If aware (e.g., UTC or FixedOffset), convert directly
                df[col] = df[col].dt.tz_convert(tz_name)
        return df

    def coord2grid(self, coord_tuple):
        lat_min, lat_max = self.min_lat, self.max_lat
        lon_min, lon_max = self.min_lon, self.max_lon
        current_lat, current_long = coord_tuple
        
        if current_lat < lat_min or current_lat > lat_max or current_long < lon_min or current_long > lon_max:
            return 0 
            
        h_res = (lon_max - lon_min) / self.horizontal_n if self.horizontal_n > 0 else 1e-9
        v_res = (lat_max - lat_min) / self.vertical_n if self.vertical_n > 0 else 1e-9
        if h_res == 0: h_res = 1e-9
        if v_res == 0: v_res = 1e-9

        y = min(int((current_long - lon_min) / h_res), self.horizontal_n - 1)
        x = min(int((current_lat - lat_min) / v_res), self.vertical_n - 1)
        
        return x * self.horizontal_n + y + 1

    def grid2coor(self):
        """Get the center coordinates for each grid ID"""
        h_res = (self.max_lon - self.min_lon) / self.horizontal_n if self.horizontal_n > 0 else 1e-9
        v_res = (self.max_lat - self.min_lat) / self.vertical_n if self.vertical_n > 0 else 1e-9
        X, Y = [], []
            
        for grid in range(1, self.max_locs + 1):
            idx = grid - 1
            x_idx = idx // self.horizontal_n
            y_idx = idx % self.horizontal_n
            
            lat_center = self.min_lat + (x_idx + 0.5) * v_res
            lon_center = self.min_lon + (y_idx + 0.5) * h_res
            X.append(lat_center)
            Y.append(lon_center)
        
        return X, Y

    def build_sequences_for_df(self, df):

        all_seqs = []
        transition_matrix = np.zeros((self.max_locs, self.max_locs))
        
        def get_grid(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']): return 0
            return self.coord2grid((row['latitude'], row['longitude']))
            
        df['grid_id'] = df.apply(get_grid, axis=1)
        
        grouped = df.groupby('agent_id')
        for agent_id, group in tqdm(grouped, desc="Translating into 96-bins"):
            group = group.sort_values('start_datetime')
            
            grids = group[group['grid_id'] > 0]['grid_id'].values
            for i in range(len(grids) - 1):
                curr, nxt = grids[i], grids[i+1]

                if 0 < curr <= self.max_locs and 0 < nxt <= self.max_locs:
                    transition_matrix[curr-1, nxt-1] += 1
                    
            if len(group) == 0:
                continue
                
            min_date = group['start_datetime'].dt.date.min()
            max_date = group['end_datetime'].dt.date.max()
            
            num_days = (max_date - min_date).days + 1
            total_bins = num_days * self.seq_len
            continuous_seq = np.zeros(total_bins, dtype=int)
            
            for _, row in group.iterrows():
                grid_id = row['grid_id']
                if grid_id == 0: continue
                
                s_dt = row['start_datetime']
                e_dt = row['end_datetime']
                
                s_day_offset = (s_dt.date() - min_date).days
                s_bin = s_day_offset * self.seq_len + s_dt.hour * 4 + s_dt.minute // 15
                
                e_day_offset = (e_dt.date() - min_date).days
                e_bin = e_day_offset * self.seq_len + e_dt.hour * 4 + e_dt.minute // 15
                
                if e_bin <= s_bin: e_bin = s_bin + 1
                e_bin = min(e_bin, total_bins) 
                
                continuous_seq[s_bin:e_bin] = grid_id
                
            last_valid = random.randint(1, self.max_locs)
            
            for i in range(total_bins):
                if continuous_seq[i] == 0:
                    continuous_seq[i] = last_valid
                else:
                    last_valid = continuous_seq[i]
                    
            for day in range(num_days):
                daily_seq = continuous_seq[day * self.seq_len : (day+1) * self.seq_len]
                all_seqs.append(daily_seq)
                
        total_trans = transition_matrix.sum()
        if total_trans > 0:
            transition_matrix /= total_trans
            
        return all_seqs, transition_matrix

    def compute_matrix_norm(self, input_matrix):
        return np.linalg.norm(input_matrix, 'fro')

    def get_distances(self, trajs): 
        distances = []
        for traj in trajs:
            for i in range(self.seq_len - 1):
                idx_curr = traj[i] - 1
                idx_next = traj[i+1] - 1
                
                if 0 <= idx_curr < self.max_locs and 0 <= idx_next < self.max_locs:
                    dx = self.X[idx_curr] - self.X[idx_next]
                    dy = self.Y[idx_curr] - self.Y[idx_next]
                    distances.append(np.sqrt(dx**2 + dy**2))
        return np.array(distances, dtype=float)

    def get_gradius(self, trajs):
        gradius = []
        for traj in trajs:
            valid_indices = [t-1 for t in traj if 0 <= t-1 < self.max_locs]
            if not valid_indices:
                gradius.append(0)
                continue
                
            xs = np.array([self.X[idx] for idx in valid_indices])
            ys = np.array([self.Y[idx] for idx in valid_indices])
            
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = np.mean(dxs**2 + dys**2)
            gradius.append(np.sqrt(rad)) 
        return np.array(gradius, dtype=float)

    def get_durations(self, trajs):
        d = []
        for traj in trajs:
            if len(traj) == 0: continue
            current_loc = traj[0]
            count = 1
            for i in range(1, len(traj)):
                if traj[i] == current_loc:
                    count += 1
                else:
                    d.append(count * 15) 
                    current_loc = traj[i]
                    count = 1
            d.append(count * 15)
        return np.array(d) / (24*60) 

    def get_periodicity(self, trajs):
        reps = []
        for traj in trajs:
            reps.append(float(len(set(traj))) / self.seq_len)
        return np.array(reps, dtype=float)
        
    def get_overall_topk_visits_freq(self, trajs, k):
        all_visits = trajs.flatten()
        all_visits = all_visits[all_visits != 0]
        
        counter = Counter(all_visits)
        top_k_counts = [count for loc, count in counter.most_common(k)]
        
        if len(top_k_counts) < k:
            top_k_counts.extend([0] * (k - len(top_k_counts)))
            
        freqs = np.array(top_k_counts)
        return freqs / np.sum(freqs) if np.sum(freqs) > 0 else freqs

    def get_individual_jsds(self):
        print("\nCalculating Evaluation Metrics...")
        d1 = self.get_distances(self.formatted_traj1)
        d2 = self.get_distances(self.formatted_traj2)
        d1_dist, _ = EvalUtils.arr_to_distribution(d1, 0, np.sqrt(self.max_distance), 100)
        d2_dist, _ = EvalUtils.arr_to_distribution(d2, 0, np.sqrt(self.max_distance), 100)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)

        g1 = self.get_gradius(self.formatted_traj1)
        g2 = self.get_gradius(self.formatted_traj2)
        g1_dist, _ = EvalUtils.arr_to_distribution(g1, 0, np.sqrt(self.max_distance), 100)
        g2_dist, _ = EvalUtils.arr_to_distribution(g2, 0, np.sqrt(self.max_distance), 100)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        
        du1 = self.get_durations(self.formatted_traj1)
        du2 = self.get_durations(self.formatted_traj2)    
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, 48)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, 48)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)
        
        p1 = self.get_periodicity(self.formatted_traj1)
        p2 = self.get_periodicity(self.formatted_traj2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, 48)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, 48)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        l1 = CollectiveEval.get_visits(self.formatted_traj1, self.max_locs)
        l2 = CollectiveEval.get_visits(self.formatted_traj2, self.max_locs)
        l1_sorted = np.sort(l1)[::-1][:100] 
        l2_sorted = np.sort(l2)[::-1][:100]
        l1_dist = l1_sorted / (np.sum(l1_sorted) + 1e-9)
        l2_dist = l2_sorted / (np.sum(l2_sorted) + 1e-9)
        l_jsd = EvalUtils.get_js_divergence(l1_dist, l2_dist)

        f1 = self.get_overall_topk_visits_freq(self.formatted_traj1, 100)
        f2 = self.get_overall_topk_visits_freq(self.formatted_traj2, 100)
        f_jsd = EvalUtils.get_js_divergence(f1, f2)

        diff = self.transition_matrix1 - self.transition_matrix2
        diff_norm = self.compute_matrix_norm(diff)
        
        return d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd, diff_norm
    

class CollectiveEval(object):
    @staticmethod
    def get_visits(trajs, max_locs):
        flat_trajs = trajs.flatten()
        flat_trajs = flat_trajs[flat_trajs > 0] 
        counts = np.bincount(flat_trajs, minlength=max_locs+1)
        counts = counts[1:] 
        prob = counts / (np.sum(counts) + 1e-9)
        return prob


if __name__ == '__main__':


    horizontal_n = 100 
    seq_len = 96       
    
    # If NumoSim
    traj_file_path1 = '/local/scratch/sli657/zllmagent/deliverable_data/traj_gen_baseline_final_numosim.parquet'
    traj_file_path2 = '/local/scratch/sli657/zllmagent/raw_data/numosim_stay_points_raw.parquet'
    TIMEZONE_STR = 'America/Los_Angeles' 
    
    # If MobilitySyn
    # TIMEZONE_STR = 'Asia/Tokyo' 
    
    individualEval = IndividualEval(horizontal_n, seq_len, traj_file_path1, traj_file_path2, tz_name=TIMEZONE_STR)

    d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, transit_norm = individualEval.get_individual_jsds()
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f'Distance JSD:        {d_jsd:.4f}')
    print(f'Gyration Radius JSD: {g_jsd:.4f}')
    print(f'Duration JSD:        {du_jsd:.4f}')
    print(f'Periodicity JSD:     {p_jsd:.4f}')
    print(f'Location JSD:        {l_jsd:.4f}')
    print(f'Frequency JSD:       {f_jsd:.4f}')
    print(f'Transition Mat Norm: {transit_norm:.4f}')
    print("-" * 50)
    print(f'Latex Format: {d_jsd:.4f} & {g_jsd:.4f} & {du_jsd:.4f} & {p_jsd:.4f} & {f_jsd:.4f} & {l_jsd:.4f} & {transit_norm:.4f}')
    