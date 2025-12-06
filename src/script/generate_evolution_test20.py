import os
import sys
import random
import argparse
import numpy as np
import sympy as sp
import pandas as pd

# Add the directory containing this script to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import preprocess_ai_feyn

def generate_data_for_equation(formula_str, var_names, num_dps, seed=42):
    """
    Generate random data for a given formula.
    """
    rng = np.random.RandomState(seed)
    
    # Generate random inputs
    # Avoid 0 to prevent division by zero errors, and keep range reasonable
    data = {}
    for var in var_names:
        # Uniform distribution between 0.1 and 5.0 (to be safe for divisions and logs)
        # Randomly negate some to have negative values too, but avoid 0
        vals = rng.uniform(0.1, 5.0, num_dps)
        signs = rng.choice([-1, 1], num_dps)
        data[var] = vals * signs
        
        # Special handling for variables that might be denominators or in logs
        # For test_20 (Klein-Nishina), variables are physical constants/quantities
        # omega, omega_0, m, c should probably be positive
        if var in ['omega', 'omega_0', 'm', 'c', 'h', 'alpha']:
             data[var] = np.abs(data[var])

    # Evaluate formula
    # We use SymPy's lambdify for efficient evaluation
    symbols = [sp.Symbol(v) for v in var_names]
    
    # Parse formula safely
    # Replace pi with numeric value for evaluation
    eval_formula_str = formula_str.replace("pi", "3.141592653589793")
    # Define local dictionary for parsing
    local_dict = {v: sp.Symbol(v) for v in var_names}
    local_dict['sin'] = sp.sin
    local_dict['cos'] = sp.cos
    local_dict['pi'] = sp.pi
    
    expr = sp.sympify(eval_formula_str, locals=local_dict)
    
    f = sp.lambdify(symbols, expr, modules="numpy")
    
    # Prepare args
    args = [data[v] for v in var_names]
    
    # Calculate output
    try:
        y = f(*args)
    except Exception as e:
        print(f"Error evaluating formula: {e}")
        return None, None

    # Combine into a matrix: [x1, x2, ..., xn, y]
    # Shape: (num_vars + 1, num_dps)
    dataset = []
    for var in var_names:
        dataset.append(data[var])
    dataset.append(y)
    
    return dataset, expr

def generate_evolution_dataset(
    output_dir="data/evolution_test20",
    gens=20,
    pop_size=1000,
    num_dps=500000,
    seed=42
):
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Generating evolution dataset in {output_dir}...")
    
    # test_20 equation details
    formula_str = "1/(4*pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-sin(beta)**2)"
    var_names = ["omega", "omega_0", "alpha", "h", "m", "c", "beta"]
    
    print(f"Target Equation: {formula_str}")
    print(f"Variables: {var_names}")
    
    # 1. Generate Shared Data
    print("Generating ground truth data...")
    dataset, sympy_expr = generate_data_for_equation(formula_str, var_names, num_dps, seed)
    
    if dataset is None:
        print("Failed to generate data.")
        return

    # Save shared data file
    data_file_path = os.path.join(output_dir, "shared_data.txt")
    with open(data_file_path, "w") as f:
        # Transpose to write row by row (datapoint by datapoint)
        # dataset is list of arrays (cols). zip(*dataset) gives rows.
        for row in zip(*dataset):
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    
    print(f"Saved shared data to {data_file_path}")
    
    # 2. Initialize Population
    # The population will consist of the target equation + mutations of it
    print("Initializing population...")
    
    # Parse and binarize the base expression for the evaluator
    # We use the helper from preprocess_ai_feyn
    binarized_expr = preprocess_ai_feyn.parse_and_binarize(formula_str, var_names)
    
    population = []
    
    # Add the exact solution as the first individual
    population.append(binarized_expr)
    
    # Fill the rest with mutations
    rng = random.Random(seed)
    
    # We want a mix of mutations
    while len(population) < pop_size:
        # Mutate the base expression to start with
        # We can use different aggression levels
        r = rng.random()
        if r < 0.3:
            mutated = preprocess_ai_feyn.mutate_expression_conservative(binarized_expr, var_names, rng)
        elif r < 0.7:
            mutated = preprocess_ai_feyn.mutate_expression_medium(binarized_expr, var_names, rng)
        else:
            mutated = preprocess_ai_feyn.mutate_expression_aggressive(binarized_expr, var_names, rng)
            
        # Binarize again to ensure structure
        mutated = preprocess_ai_feyn.binarize_tree(mutated)
        population.append(mutated)
        
    # 3. Simulate Evolution Loop
    # In this script, we are just generating the FILES that represent an evolution history.
    # We are NOT actually running a genetic algorithm to optimize fitness.
    # We are simulating a "trace" of populations for the benchmark to evaluate.
    # So we will just evolve the population randomly (mutate/crossover) to create subsequent generations.
    
    for gen in range(gens):
        filename = os.path.join(output_dir, f"gen_{gen}.txt")
        print(f"Generating Generation {gen} ({filename})...")
        
        with open(filename, "w") as f:
            f.write(f"{pop_size}\n")
            
            for expr in population:
                # Convert to tokens
                tokens, values = preprocess_ai_feyn.expr_to_tokens(expr, var_names)
                
                # Write to file
                # Format: num_vars num_dps num_tokens
                # tokens...
                # values...
                # data_file
                
                # Note: num_vars in the file format usually refers to INPUT variables
                f.write(f"{len(var_names)} {num_dps} {len(tokens)}\n")
                f.write(" ".join(map(str, tokens)) + "\n")
                f.write(" ".join(map(str, values)) + "\n")
                f.write(f"{data_file_path}\n")
        
        # Evolve for next generation
        if gen < gens - 1:
            new_pop = []
            
            # Elitism: Keep top 10% (assuming the first ones are "best" for simulation sake, 
            # or just keep random ones. Since we don't compute fitness here, we just keep some)
            # Let's keep the first one (target) and some others to maintain stability
            new_pop.append(population[0]) # Keep target
            
            # Randomly select others to keep or mutate
            while len(new_pop) < pop_size:
                parent = rng.choice(population)
                
                # Mutate
                r = rng.random()
                if r < 0.3:
                    child = preprocess_ai_feyn.mutate_expression_conservative(parent, var_names, rng)
                elif r < 0.7:
                    child = preprocess_ai_feyn.mutate_expression_medium(parent, var_names, rng)
                else:
                    child = preprocess_ai_feyn.mutate_expression_aggressive(parent, var_names, rng)
                
                child = preprocess_ai_feyn.binarize_tree(child)
                new_pop.append(child)
            
            population = new_pop

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evolution data for test_20 (Klein-Nishina)")
    parser.add_argument("--out", type=str, default="data/evolution_test20", help="Output directory")
    parser.add_argument("--gens", type=int, default=20, help="Number of generations")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--dps", type=int, default=100000, help="Number of data points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generate_evolution_dataset(
        output_dir=args.out,
        gens=args.gens,
        pop_size=args.pop,
        num_dps=args.dps,
        seed=args.seed
    )
