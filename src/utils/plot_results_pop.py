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
    match = re.search(r'(_gen\d+_pop(\d+)_dps\d+)', content)
    if not match:
        return None, None
    
    eval_name = content[:match.start()]
    eval_name = eval_name.replace('_', ' ')
    
    pop_str = match.group(2)
    pop = int(pop_str)
    
    return eval_name, pop

def main():
    parser = argparse.ArgumentParser(description='Plot SR results (Population Scaling).')
    parser.add_argument('--dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--filter', type=str, help='Filter evaluators by name (substring match)')
    args = parser.parse_args()

    # Try multiple directories
    possible_dirs = [
        '../../data/output/sr_results/plot-B',
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

    # Structure: { EvalName: { Pop: [avg_time_run1, avg_time_run2, ...] } }
    results = {}
    subtree_detect_results = {}
    
    for f in csv_files:
        eval_name, pop = parse_filename(f)
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
            if pop not in results[eval_name]:
                results[eval_name][pop] = []
            
            results[eval_name][pop].append(avg_time)
            
            # Collect DetectTimeMs for Optimized Subtree
            if 'Optimized' in eval_name and 'DetectTimeMs' in df.columns:
                avg_detect = df['DetectTimeMs'].mean()
                if eval_name not in subtree_detect_results:
                    subtree_detect_results[eval_name] = {}
                if pop not in subtree_detect_results[eval_name]:
                    subtree_detect_results[eval_name][pop] = []
                subtree_detect_results[eval_name][pop].append(avg_detect)
            
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
    
    for i, (eval_name, pop_data) in enumerate(results.items()):
        pops = sorted(pop_data.keys())
        means = []
        stds = []
        
        for p in pops:
            runs = pop_data[p]
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
        
        plt.plot(pops, means, label=display_name, color=color, marker=marker, linewidth=2)
        plt.fill_between(pops, means - stds, means + stds, color=color, alpha=0.2)
        
        print(f"{display_name} - Pop Stats: {list(zip(pops, np.round(means, 2)))}")
        
        # Plot Subtree Detection line if available for this evaluator
        if eval_name in subtree_detect_results:
            d_pop_data = subtree_detect_results[eval_name]
            d_pops = sorted(d_pop_data.keys())
            d_means = []
            for p in d_pops:
                d_means.append(np.mean(d_pop_data[p]))
            
            plt.plot(d_pops, d_means, label='Subtree Detection Time', color='gray', linestyle='--', linewidth=2)

    plt.xlabel('Population Size')
    plt.ylabel('Average Time per Generation (ms)')
    # plt.title('Performance Scaling: Time vs Population')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    filename = 'plot_time_vs_pop.png'
    if args.filter:
        safe_filter = "".join([c for c in args.filter if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        filename = f'plot_time_vs_pop_{safe_filter}.png'
        
    output_path = os.path.join(data_dir, filename)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
