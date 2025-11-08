#!/usr/bin/env python3
"""
equation_evaluator.py - Load, parse, evaluate, and plot equations from Feynman/Bonus datasets

This script uses SymPy to parse mathematical formulas from the equation datasets,
evaluates them on the provided data, and creates visualizations.

Usage:
    python equation_evaluator.py --dataset bonus --equation test_1
    python equation_evaluator.py --dataset feynman --equation I.6.2a --samples 5000
    python equation_evaluator.py --dataset bonus --all
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import sympify, lambdify, symbols
from sympy import sin, cos, exp, sqrt, log, asin, acos, tanh, pi


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_equation_metadata(csv_path):
    """Load equation metadata from CSV file.

    Args:
        csv_path: Path to BonusEquations.csv or FeynmanEquations.csv

    Returns:
        pandas DataFrame with equation metadata
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def load_equation_data(filename, dataset_dir):
    """Load equation data file (space-separated, no header).

    Args:
        filename: Name of the data file (e.g., 'test_1', 'I.6.2a')
        dataset_dir: Directory containing data files

    Returns:
        numpy array of shape (n_samples, n_features + 1)
    """
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from {filepath}...")
    data = np.loadtxt(filepath)
    print(f"  Loaded {data.shape[0]} samples with {data.shape[1]} columns")
    return data


def get_equation_by_name(metadata, name):
    """Retrieve equation metadata by filename.

    Args:
        metadata: DataFrame from load_equation_metadata
        name: Filename to search for

    Returns:
        pandas Series with equation information
    """
    result = metadata[metadata['Filename'] == name]
    if len(result) == 0:
        raise ValueError(f"Equation '{name}' not found in metadata")
    return result.iloc[0]


def extract_variable_info(equation_row):
    """Extract variable names and ranges from equation metadata row.

    Args:
        equation_row: pandas Series from metadata

    Returns:
        List of tuples: [(var_name, low, high), ...]
    """
    n_vars = int(equation_row['# variables'])
    variables = []

    for i in range(1, n_vars + 1):
        var_name = equation_row[f'v{i}_name']
        var_low = float(equation_row[f'v{i}_low'])
        var_high = float(equation_row[f'v{i}_high'])
        variables.append((var_name, var_low, var_high))

    return variables


# ============================================================================
# Formula Parsing Functions
# ============================================================================

def parse_formula_sympy(formula_str, var_names):
    """Parse formula string into SymPy expression.

    Args:
        formula_str: Formula in Python syntax (e.g., 'x**2 + sin(y)')
        var_names: List of variable names used in formula

    Returns:
        SymPy expression
    """
    # Create SymPy symbols for all variables
    var_symbols = {name: symbols(name) for name in var_names}

    # Parse formula using sympify with local dict for variables
    local_dict = var_symbols.copy()
    # Add common mathematical functions
    local_dict.update({
        'sin': sin, 'cos': cos, 'exp': exp, 'sqrt': sqrt, 'log': log,
        'arcsin': asin, 'asin': asin,
        'arccos': acos, 'acos': acos,
        'tanh': tanh, 'pi': pi
    })

    try:
        expr = sympify(formula_str, locals=local_dict)
        return expr
    except Exception as e:
        raise ValueError(f"Failed to parse formula '{formula_str}': {e}")


def create_numpy_evaluator(sympy_expr, var_names):
    """Convert SymPy expression to fast numpy function.

    Args:
        sympy_expr: SymPy expression
        var_names: List of variable names in order

    Returns:
        Callable that accepts numpy arrays
    """
    var_symbols = [symbols(name) for name in var_names]
    numpy_func = lambdify(var_symbols, sympy_expr, modules=['numpy'])
    return numpy_func


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_equation(data, formula_str, var_names):
    """Evaluate formula on input data.

    Args:
        data: numpy array of shape (n_samples, n_features + 1)
              Last column is the ground truth output
        formula_str: Formula string
        var_names: List of variable names

    Returns:
        Tuple of (predicted, actual) numpy arrays
    """
    # Separate inputs and outputs
    X = data[:, :-1]  # All columns except last
    y_actual = data[:, -1]  # Last column

    # Parse formula and create evaluator
    expr = parse_formula_sympy(formula_str, var_names)
    evaluator = create_numpy_evaluator(expr, var_names)

    # Evaluate formula
    # Unpack columns as separate arguments
    y_predicted = evaluator(*[X[:, i] for i in range(len(var_names))])

    return y_predicted, y_actual


def compute_error_metrics(predicted, actual):
    """Calculate error metrics between predicted and actual values.

    Args:
        predicted: numpy array of predictions
        actual: numpy array of ground truth

    Returns:
        Dictionary with error metrics
    """
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))

    # R² score
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Relative error
    rel_error = np.mean(np.abs((predicted - actual) / (actual + 1e-10)))

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rel_error': rel_error
    }


# ============================================================================
# Display Functions
# ============================================================================

def display_sample_data(data, var_names, predicted, n_samples=10):
    """Display sample data in tabular format.

    Args:
        data: numpy array of shape (n_samples, n_features + 1)
        var_names: List of variable names
        predicted: Predicted values
        n_samples: Number of samples to display
    """
    print(f"\n{'='*80}")
    print(f"Sample Data (first {n_samples} rows)")
    print(f"{'='*80}")

    # Create header
    header = " | ".join([f"{name:>12}" for name in var_names] + ["Actual", "Predicted", "Error"])
    print(header)
    print("-" * len(header))

    # Display rows
    X = data[:, :-1]
    y_actual = data[:, -1]

    for i in range(min(n_samples, len(data))):
        row_values = [f"{X[i, j]:>12.6f}" for j in range(len(var_names))]
        row_values.append(f"{y_actual[i]:>12.6f}")
        row_values.append(f"{predicted[i]:>12.6f}")
        error = predicted[i] - y_actual[i]
        row_values.append(f"{error:>12.6f}")
        print(" | ".join(row_values))

    print(f"{'='*80}\n")


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_predicted_vs_actual(predicted, actual, equation_name, samples=1000):
    """Create scatter plot of predicted vs actual values.

    Args:
        predicted: Predicted values
        actual: Actual values
        equation_name: Name for plot title
        samples: Number of points to plot (randomly sampled)
    """
    # Sample data if too large
    if len(predicted) > samples:
        indices = np.random.choice(len(predicted), samples, replace=False)
        predicted_plot = predicted[indices]
        actual_plot = actual[indices]
    else:
        predicted_plot = predicted
        actual_plot = actual

    plt.figure(figsize=(8, 8))
    plt.scatter(actual_plot, predicted_plot, alpha=0.5, s=10)

    # Plot diagonal line
    min_val = min(actual_plot.min(), predicted_plot.min())
    max_val = max(actual_plot.max(), predicted_plot.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title(f'Predicted vs Actual: {equation_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_residuals(predicted, actual, equation_name, samples=10000):
    """Create histogram of residuals.

    Args:
        predicted: Predicted values
        actual: Actual values
        equation_name: Name for plot title
        samples: Number of points for histogram
    """
    residuals = predicted - actual

    # Sample if too large
    if len(residuals) > samples:
        residuals = np.random.choice(residuals, samples, replace=False)

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual (Predicted - Actual)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Residual Distribution: {equation_name}', fontsize=14)
    plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_data_overview(data, var_names, predicted, actual, equation_name, samples=1000):
    """Create comprehensive visualization of inputs, outputs, and predictions.

    Args:
        data: numpy array of shape (n_samples, n_features + 1)
        var_names: List of variable names
        predicted: Predicted values
        actual: Actual values
        equation_name: Name for plot title
        samples: Number of points to plot
    """
    n_vars = len(var_names)
    X = data[:, :-1]

    # Sample data if too large
    if len(data) > samples:
        indices = np.random.choice(len(data), samples, replace=False)
        X_plot = X[indices]
        predicted_plot = predicted[indices]
        actual_plot = actual[indices]
    else:
        X_plot = X
        predicted_plot = predicted
        actual_plot = actual

    # Create subplots: one for each input variable + output comparison
    n_cols = min(3, n_vars + 1)
    n_rows = (n_vars + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    fig.suptitle(f'Data Overview: {equation_name}', fontsize=16, y=1.00)

    # Plot each input variable vs actual output
    for i in range(n_vars):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        ax.scatter(X_plot[:, i], actual_plot, alpha=0.5, s=10, label='Actual', c='blue')
        ax.scatter(X_plot[:, i], predicted_plot, alpha=0.5, s=10, label='Predicted', c='red')
        ax.set_xlabel(var_names[i], fontsize=10)
        ax.set_ylabel('Output', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot predicted vs actual in the last subplot
    last_idx = n_vars
    row = last_idx // n_cols
    col = last_idx % n_cols
    ax = axes[row, col]

    ax.scatter(actual_plot, predicted_plot, alpha=0.5, s=10, c='green')
    min_val = min(actual_plot.min(), predicted_plot.min())
    max_val = max(actual_plot.max(), predicted_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Output', fontsize=10)
    ax.set_ylabel('Predicted Output', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_vars + 1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()


# ============================================================================
# Main Interface
# ============================================================================

def process_equation(dataset, equation_name, samples_to_plot=1000):
    """Process a single equation: load, evaluate, plot.

    Args:
        dataset: 'bonus' or 'feynman'
        equation_name: Filename of equation
        samples_to_plot: Number of samples for scatter plot
    """
    # Set up paths
    data_dir = Path(__file__).parent.parent.parent / 'data'

    if dataset == 'bonus':
        csv_path = data_dir / 'BonusEquations.csv'
        data_path = data_dir / 'bonus_without_units'
    elif dataset == 'feynman':
        csv_path = data_dir / 'FeynmanEquations.csv'
        data_path = data_dir / 'Feynman_without_units'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load metadata
    print(f"\n{'='*60}")
    print(f"Processing: {equation_name} from {dataset} dataset")
    print(f"{'='*60}")

    metadata = load_equation_metadata(csv_path)
    equation = get_equation_by_name(metadata, equation_name)

    # Extract variable info
    variables = extract_variable_info(equation)
    var_names = [v[0] for v in variables]

    print(f"\nFormula: {equation['Formula']}")
    print(f"Variables ({len(var_names)}):")
    for name, low, high in variables:
        print(f"  {name}: [{low}, {high}]")

    # Load data
    data = load_equation_data(equation_name, data_path)

    # Evaluate
    print("\nEvaluating formula...")
    predicted, actual = evaluate_equation(data, equation['Formula'], var_names)

    # Compute metrics
    metrics = compute_error_metrics(predicted, actual)
    print("\nError Metrics:")
    print(f"  MSE:            {metrics['mse']:.6e}")
    print(f"  MAE:            {metrics['mae']:.6e}")
    print(f"  R² Score:       {metrics['r2']:.6f}")
    print(f"  Relative Error: {metrics['rel_error']:.6f}")

    # Display sample data
    display_sample_data(data, var_names, predicted, n_samples=10)

    # Plot
    print("\nGenerating plots...")
    plot_data_overview(data, var_names, predicted, actual, equation_name, samples_to_plot)
    plot_predicted_vs_actual(predicted, actual, equation_name, samples_to_plot)
    plot_residuals(predicted, actual, equation_name)

    plt.show()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and visualize equations from Feynman/Bonus datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset bonus --equation test_1
  %(prog)s --dataset feynman --equation I.6.2a --samples 5000
        """
    )

    parser.add_argument('--dataset', type=str, required=True, choices=['bonus', 'feynman'],
                        help='Dataset to use (bonus or feynman)')
    parser.add_argument('--equation', type=str, required=True,
                        help='Equation filename (e.g., test_1, I.6.2a)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples for scatter plot (default: 1000)')

    args = parser.parse_args()

    try:
        process_equation(args.dataset, args.equation, args.samples)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
