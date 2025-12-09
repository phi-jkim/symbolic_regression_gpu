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
    
    match = re.search(r'(_gen\d+_pop(\d+)_dps\d+)', content)
    if not match:
        return None, None
    
    eval_name = content[:match.start()]
    eval_name = eval_name.replace('_', ' ')
    
    pop_str = match.group(2)
    pop = int(pop_str)
    
    return eval_name, pop

def main():
    parser = argparse.ArgumentParser(description='Plot SR results breakdown (Population Scaling).')
    parser.add_argument('--dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--filter', type=str, help='Filter evaluators by name')
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
        print(f"No CSV files found.")
        return

    # Structure: { EvalName: { Pop: [df1, df2, ...] } }
    grouped_data = {}
    
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
                
            if eval_name not in grouped_data:
                grouped_data[eval_name] = {}
            if pop not in grouped_data[eval_name]:
                grouped_data[eval_name][pop] = []
                
            grouped_data[eval_name][pop].append(df)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Breakdown components
    COMPONENT_MAP = {
        'Memory Transfer': 'H2D_D2H_TimeMs',
        'Subtree Detection': 'DetectTimeMs',
        'JIT Compilation': 'JITTimeMs',
        'Kernel Execution': 'KernelTimeMs'
    }
    
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
    
    for eval_name, pop_dict in grouped_data.items():
        print(f"Generating breakdown for {eval_name}...")
        
        pops = sorted(pop_dict.keys())
        
        # Accumulate means for each component per population
        comp_means = {k: [] for k in COMPONENT_MAP.keys()}
        coverage_means = []
        
        for p in pops:
            dfs = pop_dict[p]
            
            # Helper to get mean of a component across runs for this pop
            for comp_name, col_name in COMPONENT_MAP.items():
                vals = []
                for df in dfs:
                    if col_name in df.columns:
                        vals.append(df[col_name].mean())
                    else:
                        vals.append(0.0)
                
                comp_means[comp_name].append(np.mean(vals))
            
            # Coverage
            if 'Optimized' in eval_name:
                cov_vals = []
                for df in dfs:
                    if 'Coverage' in df.columns:
                        cov_vals.append(df['Coverage'].mean())
                if cov_vals:
                    coverage_means.append(np.mean(cov_vals))
                else:
                    coverage_means.append(0)

        # Stackplot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        labels = list(comp_means.keys())
        values = [comp_means[k] for k in labels]
        colors = [COMP_COLORS.get(k, 'gray') for k in labels]
        
        ax1.stackplot(pops, values, labels=labels, colors=colors, alpha=0.7)
        
        display_name = LEGEND_MAP.get(eval_name, eval_name)
        ax1.set_title(f'Latency Breakdown vs Population: {display_name}')
        ax1.set_xlabel('Population Size')
        ax1.set_ylabel('Average Time per Round (ms)')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.set_xlim(left=min(pops), right=max(pops))
        ax1.set_ylim(bottom=0)
        
        # Plot Coverage
        if coverage_means and 'Optimized' in eval_name:
            ax2 = ax1.twinx()
            ax2.plot(pops, coverage_means, color='black', linestyle='--', linewidth=2, label='Coverage Ratio')
            ax2.set_ylabel('Coverage Ratio')
            ax2.set_ylim(0, 1.05)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
            
        safe_name = eval_name.replace(' ', '_')
        filename = f'plot_breakdown_pop_{safe_name}.png'
        output_path = os.path.join(data_dir, filename)
        plt.savefig(output_path, dpi=300)
        print(f"Saved {output_path}")
        plt.close()

if __name__ == "__main__":
    main()
