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
        return None
    
    content = basename[11:]
    
    match = re.search(r'(_gen\d+_pop\d+_dps\d+)', content)
    if not match:
        return None
    
    eval_name = content[:match.start()]
    eval_name = eval_name.replace('_', ' ')
    
    return eval_name

def main():
    parser = argparse.ArgumentParser(description='Plot SR results (Kernel Only for JIT).')
    parser.add_argument('--dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--filter', type=str, help='Filter evaluators by name')
    args = parser.parse_args()

    # Try multiple directories
    possible_dirs = [
        '../../data/output/sr_results/plot-A1',
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
        print(f"No CSV files found.")
        return

    grouped_data = {}
    
    for f in csv_files:
        eval_name = parse_filename(f)
        if not eval_name:
            continue
            
        if args.filter and args.filter.lower() not in eval_name.lower():
            continue
            
        try:
            if os.path.getsize(f) == 0:
                continue
            df = pd.read_csv(f)
            if df.empty or 'Gen' not in df.columns:
                continue
            
            if eval_name not in grouped_data:
                grouped_data[eval_name] = []
            grouped_data[eval_name].append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Plot
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    LEGEND_MAP = {
        'GPU Optimized Subtree': 'GPU with Subtree Caching',
        'GPU Simple': 'GPU Baseline',
        'CPU': 'CPU Baseline (96 threads)',
        'GPU NVRTC': 'GPU with NVRTC Compile (Kernel Only)',
        'GPU PTX': 'GPU with PTX Compile (Kernel Only)'
    }
    
    COLOR_MAP = {
        'CPU': 'tab:purple',
        'GPU Simple': 'tab:blue',
        'GPU Optimized Subtree': 'tab:red',
        'GPU PTX': 'tab:green',
        'GPU NVRTC': 'tab:orange'
    }
    
    skip_gen_0 = True
    
    for i, (eval_name, dfs) in enumerate(grouped_data.items()):
        max_gen = 0
        for df in dfs:
            max_gen = max(max_gen, df['Gen'].max())
        
        all_times = []
        common_gens = np.arange(max_gen + 1)
        
        for df in dfs:
            df_curr = df.set_index('Gen').reindex(common_gens)
            
            # Metric Selection Logic
            # "One is plots with cpu, gpu simple, gpu subtree, and gpu ptx (kernel only), and gpu nvrtc (kernel only)"
            metric = 'TotalTimeMs'
            if 'PTX' in eval_name or 'NVRTC' in eval_name:
                if 'KernelTimeMs' in df.columns:
                    metric = 'KernelTimeMs'
            
            if metric not in df_curr.columns:
                # Fallback if preferred metric missing
                 metric = 'TotalTimeMs'
            
            all_times.append(df_curr[metric].values)
            
        all_times = np.array(all_times)
        
        mean_times = np.nanmean(all_times, axis=0)
        std_times = np.nanstd(all_times, axis=0)
        
        mask = ~np.isnan(mean_times)
        valid_gens = common_gens[mask]
        valid_mean = mean_times[mask]
        valid_std = std_times[mask]
        
        if skip_gen_0:
            mask_gen0 = valid_gens > 0
            valid_gens = valid_gens[mask_gen0]
            valid_mean = valid_mean[mask_gen0]
            valid_std = valid_std[mask_gen0]
        
        display_name = LEGEND_MAP.get(eval_name, eval_name)
        
        if eval_name in COLOR_MAP:
            color = COLOR_MAP[eval_name]
        else:
            color = colors[i % len(colors)]
        
        marker = markers[i % len(markers)]
        
        plt.plot(valid_gens, valid_mean, label=display_name, color=color, linewidth=2)
        plt.fill_between(valid_gens, valid_mean - valid_std, valid_mean + valid_std, color=color, alpha=0.2)
        
        avg_time = np.mean(valid_mean)
        print(f"{display_name} Average Time (Gen > 0): {avg_time:.2f} ms")

    plt.xlabel('Round Count')
    plt.ylabel('Time per Round (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    filename = 'plot_time_kernel_compare.png'
    output_path = os.path.join(data_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
