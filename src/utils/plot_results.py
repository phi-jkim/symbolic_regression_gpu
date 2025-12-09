import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_filename(filename):
    # Expected format: sr_results_<EvalName>_gen<GENS>_pop<POP>_dps<DPS>_<TIMESTAMP>.csv
    # Or without timestamp: sr_results_<EvalName>_gen<GENS>_pop<POP>_dps<DPS>.csv
    basename = os.path.basename(filename)
    # Remove extension
    basename = basename.replace('.csv', '')
    
    # Extract parts
    # Assumption: "sr_results_" prefix
    if not basename.startswith('sr_results_'):
        return None
    
    content = basename[11:] # remove "sr_results_"
    
    # Split by "_" but handle EvalName which might contain underscores
    # We know the suffixes start with gen, pop, dps.
    # Regex to find the start of the params
    match = re.search(r'(_gen\d+_pop\d+_dps\d+)', content)
    if not match:
        return None
    
    eval_name = content[:match.start()]
    # normalize names
    eval_name = eval_name.replace('_', ' ')
    
    return eval_name

import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot SR results.')
    parser.add_argument('--dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--filter', type=str, help='Filter evaluators by name (substring match)')
    parser.add_argument('--log', action='store_true', help='Use log scale for y-axis')
    args = parser.parse_args()

    # Try multiple directories
    possible_dirs = [
        '../../data/output/sr_results/plot-A2',
        '../../data/output/sr_results/plot-A1',
        '../../data/output/sr_results/plot-A',
        '../../data/output/sr_results/plot-A-bck',
        '../../data/output/sr_results'
    ]
    
    if args.dir:
        possible_dirs.insert(0, args.dir)
    
    data_dir = None
    csv_files = []
    
    for d in possible_dirs:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, 'sr_results_*.csv'))
            if files:
                data_dir = d
                csv_files = files
                print(f"Found data in: {d}")
                break
    
    if not csv_files:
        print(f"No CSV files found in any of the checked directories: {possible_dirs}")
        return

    # Group data by evaluator
    # content: { 'Eval Name': [df_run1, df_run2, ...] }
    grouped_data = {}
    
    for f in csv_files:
        eval_name = parse_filename(f)
        if not eval_name:
            continue
            
        if args.filter and args.filter.lower() not in eval_name.lower():
            continue
            
        try:
            # Check if file is empty
            if os.path.getsize(f) == 0:
                print(f"Skipping empty file: {f}")
                continue

            df = pd.read_csv(f)
            if df.empty or 'Gen' not in df.columns or 'TotalTimeMs' not in df.columns:
                print(f"Skipping {f}: missing columns or empty")
                continue
            
            if eval_name not in grouped_data:
                grouped_data[eval_name] = []
            grouped_data[eval_name].append(df)
            print(f"Loaded {f} as {eval_name}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Plot
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    # Filter out Gen 0 to avoid the initialization spike
    skip_gen_0 = True
    
    # Custom Legend Mapping
    # Custom Legend Mapping
    LEGEND_MAP = {
        'GPU Optimized Subtree': 'GPU with Subtree Caching',
        'GPU Simple': 'GPU Baseline',
        'CPU': 'CPU Baseline (96 threads)',
        'GPU NVRTC': 'GPU with NVRTC Compile',
        'GPU PTX': 'GPU with PTX Compile'
    }
    
    # Consistent Color Mapping
    COLOR_MAP = {
        'CPU': 'tab:purple',
        'GPU Simple': 'tab:blue',
        'GPU Optimized Subtree': 'tab:red',
        'GPU PTX': 'tab:green',
        'GPU NVRTC': 'tab:orange'
    }
    
    for i, (eval_name, dfs) in enumerate(grouped_data.items()):
        # Apply mapping if exists
        display_name = LEGEND_MAP.get(eval_name, eval_name)
        
        # Align dataframes
        # Get max generations
        max_gen = 0
        for df in dfs:
            max_gen = max(max_gen, df['Gen'].max())
        
        all_times = []
        common_gens = np.arange(max_gen + 1)
        
        for df in dfs:
            df_curr = df.set_index('Gen').reindex(common_gens)
            all_times.append(df_curr['TotalTimeMs'].values)
            
        all_times = np.array(all_times) # Shape: (NumRuns, NumGens)
        
        mean_times = np.nanmean(all_times, axis=0)
        std_times = np.nanstd(all_times, axis=0)
        
        mask = ~np.isnan(mean_times)
        valid_gens = common_gens[mask]
        valid_mean = mean_times[mask]
        valid_std = std_times[mask]
        
        # Skip Gen 0 if requested
        if skip_gen_0:
            mask_gen0 = valid_gens > 0
            valid_gens = valid_gens[mask_gen0]
            valid_mean = valid_mean[mask_gen0]
            valid_std = valid_std[mask_gen0]
        
        # Determine color
        if eval_name in COLOR_MAP:
            color = COLOR_MAP[eval_name]
        else:
            color = colors[i % len(colors)]
            
        marker = markers[i % len(markers)]
        
        # Use markevery=10 for better visibility but not too crowded (original was 20)
        plt.plot(valid_gens, valid_mean, label=display_name, color=color, linewidth=2)
        plt.fill_between(valid_gens, valid_mean - valid_std, valid_mean + valid_std, color=color, alpha=0.2)
        
        # Print average time (excluding gen 0)
        if len(valid_mean) > 0:
            avg_time = np.mean(valid_mean)
            print(f"{display_name} Average Time (Gen > 0): {avg_time:.2f} ms")

    plt.xlabel('Round Count')
    plt.ylabel('Time per Round (ms)')
    
    if args.log:
        plt.yscale('log')
        plt.ylabel('Time per Round (ms) [Log Scale]')
    else:
        plt.ylim(bottom=0)
        
    # title = 'Performance Comparison: Time vs Generation'
    # if skip_gen_0:
    #     title += ' (Gen 0 excluded)'
    # plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, which='both' if args.log else 'major')
    
    filename = 'plot_time_vs_gen.png'
    if args.filter:
        safe_filter = "".join([c for c in args.filter if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        filename = f'plot_time_vs_gen_{safe_filter}.png'
    
    if args.log:
        name_part, ext = os.path.splitext(filename)
        filename = f"{name_part}_log{ext}"
        
    output_path = os.path.join(data_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
