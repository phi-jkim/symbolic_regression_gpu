import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

def parse_filename(filename):
    basename = os.path.basename(filename)
    basename = basename.replace('.csv', '')
    
    if not basename.startswith('sr_results_'):
        return None, None
    
    content = basename[11:]
    
    # Extract params
    # Expected: ..._dps<DPS>_...
    match = re.search(r'(_gen\d+_pop\d+_dps(\d+))', content)
    if not match:
        return None, None
    
    eval_name = content[:match.start()]
    eval_name = eval_name.replace('_', ' ')
    
    dps_str = match.group(2)
    dps = int(dps_str)
    
    return eval_name, dps

def main():
    parser = argparse.ArgumentParser(description='Plot SR results (Data Points Scaling).')
    parser.add_argument('--dir', type=str, help='Directory containing CSV files')
    args = parser.parse_args()

    # Try multiple directories
    possible_dirs = [
        '../../data/output/sr_results/plot-C',
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

    # Structure: { EvalName: { DPS: [avg_time_run1, avg_time_run2, ...] } }
    results = {}
    
    for f in csv_files:
        eval_name, dps = parse_filename(f)
        if not eval_name:
            continue
            
        try:
            if os.path.getsize(f) == 0:
                continue
                
            df = pd.read_csv(f)
            if df.empty or 'Gen' not in df.columns:
                continue
            
            # Filter out Gen 0
            df = df[df['Gen'] > 0]
            if df.empty:
                continue
                
            # Use KernelTimeMs for PTX to show pure execution time (ignoring JIT)
            if 'PTX' in eval_name and 'KernelTimeMs' in df.columns:
                avg_time = df['KernelTimeMs'].mean()
            elif 'TotalTimeMs' in df.columns:
                avg_time = df['TotalTimeMs'].mean()
            else:
                 continue
            
            if eval_name not in results:
                results[eval_name] = {}
            if dps not in results[eval_name]:
                results[eval_name][dps] = []
            
            results[eval_name][dps].append(avg_time)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Plot
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
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
    
    for i, (eval_name, dps_data) in enumerate(results.items()):
        dps_list = list(x//1000 for x in sorted(dps_data.keys()))
        means = []
        stds = []
        
        for d in sorted(dps_data.keys()):
            runs = dps_data[d]
            means.append(np.mean(runs))
            stds.append(np.std(runs))
            
        means = np.array(means)
        stds = np.array(stds)
        
        display_name = LEGEND_MAP.get(eval_name, eval_name)
        
        # Determine color
        if eval_name in COLOR_MAP:
            color = COLOR_MAP[eval_name]
        else:
            color = colors[i % len(colors)]
            
        marker = markers[i % len(markers)]
        
        plt.plot(dps_list, means, label=display_name, color=color, marker=marker, linewidth=2)
        plt.fill_between(dps_list, means - stds, means + stds, color=color, alpha=0.2)
        
        print(f"{display_name} - DPS Stats: {list(zip(dps_list, np.round(means, 2)))}")

    plt.xlabel('Data Points (thousands)')
    plt.ylabel('Average Time per Round (ms)')
    # plt.title('Performance Scaling: Time vs Data Points')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    output_path = os.path.join(data_dir, 'plot_time_vs_dps.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
