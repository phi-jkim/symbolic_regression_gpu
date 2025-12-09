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
    parser = argparse.ArgumentParser(description='Plot SR results breakdown.')
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

    # Breakdown components
    # Map friendly name to CSV column
    # Order matters for stacking: typically from bottom to top
    COMPONENT_MAP = {
        'Memory Transfer': 'H2D_D2H_TimeMs',
        'Subtree Detection': 'DetectTimeMs',
        'JIT Compilation': 'JITTimeMs',
        'Kernel Execution': 'KernelTimeMs'
    }
    
    # Colors for components
    COMP_COLORS = {
        'Memory Transfer': 'tab:blue',
        'Subtree Detection': 'tab:green',
        'JIT Compilation': 'tab:red',
        'Kernel Execution': 'tab:orange'
    }

    LEGEND_MAP = {
        'GPU Optimized Subtree': 'GPU with Subtree Caching',
        'GPU Simple': 'GPU Baseline',
        'CPU': 'CPU Baseline (96 threads)',
        'GPU NVRTC': 'GPU with NVRTC Compile',
        'GPU PTX': 'GPU with PTX Compile'
    }

    skip_gen_0 = True
    
    for eval_name, dfs in grouped_data.items():
        # Only plot breakdown for GPUs typically, but script handles any with columns
        print(f"Generating breakdown for {eval_name}...")
        
        max_gen = 0
        for df in dfs:
            max_gen = max(max_gen, df['Gen'].max())
            
        common_gens = np.arange(max_gen + 1)
        
        # Accumulate means for each component
        comp_means = {}
        
        for comp_name, col_name in COMPONENT_MAP.items():
            all_vals = []
            has_col = False
            for df in dfs:
                if col_name in df.columns:
                    has_col = True
                    df_curr = df.set_index('Gen').reindex(common_gens).fillna(0)
                    all_vals.append(df_curr[col_name].values)
            
            if has_col and all_vals:
                comp_means[comp_name] = np.nanmean(np.array(all_vals), axis=0)
            else:
                comp_means[comp_name] = np.zeros_like(common_gens, dtype=float)

        # Accumulate Coverage if available
        coverage_mean = None
        if 'Optimized' in eval_name:
            all_cov = []
            for df in dfs:
                if 'Coverage' in df.columns:
                    df_curr = df.set_index('Gen').reindex(common_gens).fillna(0)
                    all_cov.append(df_curr['Coverage'].values)
            if all_cov:
                coverage_mean = np.nanmean(np.array(all_cov), axis=0)

        # Filter gens
        valid_gens = common_gens
        if skip_gen_0:
            mask = valid_gens > 0
            valid_gens = valid_gens[mask]
            for cn in comp_means:
                comp_means[cn] = comp_means[cn][mask]
            
            if coverage_mean is not None:
                coverage_mean = coverage_mean[mask]

        # Stackplot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        labels = list(comp_means.keys())
        values = [comp_means[k] for k in labels]
        colors = [COMP_COLORS.get(k, 'gray') for k in labels]
        
        ax1.stackplot(valid_gens, values, labels=labels, colors=colors, alpha=0.7)
        
        display_name = LEGEND_MAP.get(eval_name, eval_name)
        ax1.set_title(f'Latency Breakdown: {display_name}')
        ax1.set_xlabel('Round Count')
        ax1.set_ylabel('Time (ms)')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.set_xlim(left=min(valid_gens), right=max(valid_gens))
        ax1.set_ylim(bottom=0)
        
        # Plot Coverage on secondary axis
        if coverage_mean is not None:
            ax2 = ax1.twinx()
            ax2.plot(valid_gens, coverage_mean, color='black', linestyle='--', linewidth=2, label='Coverage Ratio')
            ax2.set_ylabel('Coverage Ratio')
            ax2.set_ylim(0, 1.05)
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        safe_name = eval_name.replace(' ', '_')
        filename = f'plot_breakdown_{safe_name}.png'
        output_path = os.path.join(data_dir, filename)
        plt.savefig(output_path, dpi=300)
        print(f"Saved {output_path}")
        plt.close()

if __name__ == "__main__":
    main()
