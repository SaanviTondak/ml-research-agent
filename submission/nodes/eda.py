import argparse
import csv
import os
import numpy as np
from collections import defaultdict

def analyze_splits(data_dir):
    print("--- SPLIT STATISTICS ---")
    # We can read the files directly to get all columns
    train_file = os.path.join(data_dir, 'log_standard_4_08_to_4_21_pure.csv')
    valid_file = os.path.join(data_dir, 'log_standard_4_22_to_5_08_pure.csv')
    
    # Read train
    train_rows = []
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                train_rows.append(r)
    print(f"Train rows: {len(train_rows)}")
    
    # Read valid (only dates 20220422 to 20220428)
    valid_rows = []
    if os.path.exists(valid_file):
        with open(valid_file, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                date = int(r['date'])
                if 20220422 <= date <= 20220428:
                    valid_rows.append(r)
    print(f"Valid rows: {len(valid_rows)}")
    
    if not train_rows or not valid_rows:
        print("Data files not found or empty.")
        return train_rows, valid_rows

    # Unique users and videos
    train_users = {r['user_id'] for r in train_rows}
    train_videos = {r['video_id'] for r in train_rows}
    valid_users = {r['user_id'] for r in valid_rows}
    valid_videos = {r['video_id'] for r in valid_rows}
    
    print(f"Train unique users: {len(train_users)}, unique videos: {len(train_videos)}")
    print(f"Valid unique users: {len(valid_users)}, unique videos: {len(valid_videos)}")
    print(f"User overlap (valid users in train): {len(valid_users & train_users)} / {len(valid_users)} ({len(valid_users & train_users)/len(valid_users)*100:.2f}%)")
    print(f"Video overlap (valid videos in train): {len(valid_videos & train_videos)} / {len(valid_videos)} ({len(valid_videos & train_videos)/len(valid_videos)*100:.2f}%)")
    
    # Label rates
    train_long_views = sum(1 for r in train_rows if r['long_view'] == '1')
    valid_long_views = sum(1 for r in valid_rows if r['long_view'] == '1')
    print(f"Train long_view rate: {train_long_views / len(train_rows):.4f}")
    print(f"Valid long_view rate: {valid_long_views / len(valid_rows):.4f}")
    
    return train_rows, valid_rows

def analyze_users(valid_rows):
    print("\n--- USER-LEVEL STATISTICS IN VALID ---")
    user_labels = defaultdict(list)
    for r in valid_rows:
        user_labels[r['user_id']].append(int(r['long_view']))
        
    num_users = len(user_labels)
    all_zero_users = 0
    all_one_users = 0
    mixed_users = 0
    
    imp_counts = []
    for u, labs in user_labels.items():
        imp_counts.append(len(labs))
        s = sum(labs)
        if s == 0:
            all_zero_users += 1
        elif s == len(labs):
            all_one_users += 1
        else:
            mixed_users += 1
            
    print(f"Total users: {num_users}")
    print(f"All-zero users: {all_zero_users} ({all_zero_users/num_users*100:.2f}%)")
    print(f"All-one users: {all_one_users} ({all_one_users/num_users*100:.2f}%)")
    print(f"Mixed users (contribute to GAUC): {mixed_users} ({mixed_users/num_users*100:.2f}%)")
    print(f"Impression count per user: min={np.min(imp_counts)}, median={np.median(imp_counts)}, max={np.max(imp_counts)}, mean={np.mean(imp_counts):.2f}")

def analyze_long_view_definition(train_rows):
    print("\n--- LONG_VIEW DEFINITION ANALYSIS ---")
    # Let's see how play_time_ms and duration_ms relate to long_view
    # Sample some rows to avoid slow computations, say 100,000 rows
    sample = train_rows[:100000]
    
    correct_by_ratio_80 = 0
    correct_by_duration_3s = 0
    correct_by_either = 0
    
    for r in sample:
        play_time = float(r['play_time_ms'])
        duration = float(r['duration_ms'])
        long_view = int(r['long_view'])
        
        # Hypotheses:
        # 1. play_time / duration >= 0.8
        h_ratio = 1 if (duration > 0 and play_time / duration >= 0.8) else 0
        # 2. play_time >= 3000
        h_dur = 1 if play_time >= 3000 else 0
        # 3. play_time / duration >= 0.8 OR play_time >= 3000? Or maybe play_time / duration >= 1.0?
        # Let's check a few combinations
        h_either = 1 if (play_time >= 3000 or (duration > 0 and play_time / duration >= 0.8)) else 0
        
        if h_ratio == long_view:
            correct_by_ratio_80 += 1
        if h_dur == long_view:
            correct_by_duration_3s += 1
        if h_either == long_view:
            correct_by_either += 1
            
    n = len(sample)
    print(f"Accuracy of play_time/duration >= 0.8: {correct_by_ratio_80/n*100:.2f}%")
    print(f"Accuracy of play_time >= 3000ms: {correct_by_duration_3s/n*100:.2f}%")
    print(f"Accuracy of (play_time >= 3000ms OR play_time/duration >= 0.8): {correct_by_either/n*100:.2f}%")
    
    # Let's find the exact rule if possible
    # Let's print some examples where long_view is 1 but play_time is small, or vice versa
    mismatches = []
    for r in sample:
        play_time = float(r['play_time_ms'])
        duration = float(r['duration_ms'])
        long_view = int(r['long_view'])
        ratio = play_time / duration if duration > 0 else 0
        
        # Let's see if long_view == 1 is exactly (play_time >= 3000 or ratio >= 0.8) or something else
        # Let's check if there's any long_view == 1 with play_time < 3000 and ratio < 0.8
        if long_view == 1 and play_time < 3000 and ratio < 0.8:
            mismatches.append((play_time, duration, ratio, long_view))
        # Let's check if there's any long_view == 0 with play_time >= 3000 or ratio >= 0.8
        if long_view == 0 and (play_time >= 3000 or ratio >= 0.8):
            mismatches.append((play_time, duration, ratio, long_view))
            
    print(f"Number of mismatches with standard rule (play_time >= 3s or ratio >= 0.8): {len(mismatches)} out of {n}")
    if mismatches:
        print("Sample mismatches (play_time, duration, ratio, long_view):")
        for m in mismatches[:10]:
            print(f"  play_time={m[0]:.1f}, duration={m[1]:.1f}, ratio={m[2]:.3f}, long_view={m[3]}")

def analyze_other_targets(train_rows):
    print("\n--- CORRELATION WITH OTHER TARGETS ---")
    targets = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate', 'is_profile_enter']
    counts = defaultdict(int)
    joint_long_view = defaultdict(int)
    
    for r in train_rows:
        lv = int(r['long_view'])
        for t in targets:
            val = int(r[t])
            if val == 1:
                counts[t] += 1
                if lv == 1:
                    joint_long_view[t] += 1
                    
    n = len(train_rows)
    print(f"Total train rows: {n}")
    for t in targets:
        rate = counts[t] / n
        cond_lv = joint_long_view[t] / counts[t] if counts[t] > 0 else 0.0
        print(f"  {t:16s} | rate: {rate:.4f} | P(long_view=1 | {t}=1): {cond_lv:.4f}")

def analyze_features(data_dir):
    print("\n--- FEATURE CHARACTERISTICS ---")
    # Video basic features
    basic_file = os.path.join(data_dir, 'video_features_basic_pure.csv')
    if os.path.exists(basic_file):
        with open(basic_file, 'r') as f:
            reader = csv.DictReader(f)
            feats = list(reader)
        print(f"Video basic features count: {len(feats)}")
        keys = feats[0].keys()
        print(f"Columns: {list(keys)}")
        for k in keys:
            if k == 'video_id':
                continue
            vals = {r[k] for r in feats}
            print(f"  {k:20s} | unique values: {len(vals)}")
            
    # User features
    user_file = os.path.join(data_dir, 'user_features_pure.csv')
    if os.path.exists(user_file):
        with open(user_file, 'r') as f:
            reader = csv.DictReader(f)
            feats = list(reader)
        print(f"User features count: {len(feats)}")
        keys = list(feats[0].keys())
        print(f"Columns (first 15): {keys[:15]}")
        for k in keys[:15]:
            if k == 'user_id':
                continue
            vals = {r[k] for r in feats}
            print(f"  {k:25s} | unique values: {len(vals)}")

    # Video statistic features
    stat_file = os.path.join(data_dir, 'video_features_statistic_pure.csv')
    if os.path.exists(stat_file):
        with open(stat_file, 'r') as f:
            reader = csv.DictReader(f)
            feats = list(reader)
        print(f"Video statistic features count: {len(feats)}")
        keys = list(feats[0].keys())
        print(f"Columns (first 10): {keys[:10]}")
        print(f"Total statistic columns: {len(keys)}")

def analyze_user_history(train_rows, valid_rows):
    print("\n--- USER HISTORY ANALYSIS ---")
    # For each user, let's collect their training interactions in chronological order
    user_history = defaultdict(list)
    for r in train_rows:
        # We can sort by date, hourmin, or time_ms if available
        # Let's just append for now
        user_history[r['user_id']].append(r)
        
    history_lengths = []
    valid_users_with_history = 0
    for r in valid_rows:
        u = r['user_id']
        history_lengths.append(len(user_history[u]))
        if len(user_history[u]) > 0:
            valid_users_with_history += 1
            
    print(f"Valid impressions with ANY user history in train: {valid_users_with_history} / {len(valid_rows)} ({valid_users_with_history/len(valid_rows)*100:.2f}%)")
    print(f"User history length in train for valid impressions: min={np.min(history_lengths)}, median={np.median(history_lengths)}, max={np.max(history_lengths)}, mean={np.mean(history_lengths):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--split', default='valid')
    parser.add_argument('--out', default='dummy_out.csv')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    train_rows, valid_rows = analyze_splits(args.data_dir)
    if valid_rows:
        analyze_users(valid_rows)
    if train_rows:
        analyze_long_view_definition(train_rows)
        analyze_other_targets(train_rows)
    analyze_features(args.data_dir)
    if train_rows and valid_rows:
        analyze_user_history(train_rows, valid_rows)
        
    # Write a dummy file to satisfy the contract if run as a normal submission
    with open(args.out, 'w') as f:
        f.write("row_id,user_id,video_id,score\n")
        for i, r in enumerate(valid_rows):
            f.write(f"{i},{r['user_id']},{r['video_id']},0.5\n")

if __name__ == '__main__':
    main()