# read csv, write to ai_feyn/digest/input_001.txt (in toml)
# File format:
"""
1           # num_exprs
2           # num_vars
1000000     # num_dps
12          # num_tokens (for formula)
2 -1 4 7 2 0 4 3 -1 -1 0 0                  # tokens
0 1 0 0 0 0 0 0 0 0 2 2.506628274631        # values
data/ai_feyn/Feynman_with_units/I.6.2a      # data filename
"""

import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import List, Tuple, Union
from collections import deque
import os


def binarize_tree(expr):
    """
    Convert n-ary operations (Add, Mul with >2 args) into binary operations.
    This is necessary because the C++ evaluator expects binary operations.
    Uses SymPy's as_two_terms() method where available.
    """
    if expr.is_Number or expr.is_Symbol:
        return expr

    if expr.is_Function:
        # Functions are already unary, just recurse on arguments
        new_args = [binarize_tree(arg) for arg in expr.args]
        return expr.func(*new_args)

    if expr.is_Pow:
        # Power is already binary
        base = binarize_tree(expr.base)
        exp = binarize_tree(expr.exp)
        return sp.Pow(base, exp, evaluate=False)

    if expr.is_Add or expr.is_Mul:
        # Recursively binarize all arguments first
        args = [binarize_tree(arg) for arg in expr.args]

        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            # Already binary
            return expr.func(args[0], args[1], evaluate=False)
        else:
            # Build left-associative binary tree: ((a op b) op c) op d
            result = expr.func(args[0], args[1], evaluate=False)
            for arg in args[2:]:
                result = expr.func(result, arg, evaluate=False)
            return result

    # For any other type, just return as-is
    return expr


def parse_and_binarize(formula_str: str, var_names: List[str]) -> sp.Expr:
    """
    Parse a formula string and binarize it.
    Returns the binarized SymPy expression.
    """
    from sympy.parsing.sympy_parser import parse_expr

    # Create symbols for variables
    symbols = {name: sp.Symbol(name) for name in var_names}

    try:
        # Replace 'pi' with its numeric value
        formula_str = formula_str.replace("pi", "3.141592653589793")
        expr = parse_expr(formula_str, local_dict=symbols)
    except Exception as e:
        print(f"Error parsing formula '{formula_str}': {e}")
        return None

    # Binarize the tree
    binary_expr = binarize_tree(expr)

    return binary_expr


def expr_to_tokens(expr, var_names: List[str]) -> Tuple[List[int], List[float]]:
    """
    Convert a SymPy expression to C++ token format (prefix notation).
    Returns (tokens, values) where:
    - tokens: List of operator/variable/constant indicators
      - 0 = constant
      - -1 = variable
      - 1-5 = binary ops (ADD, SUB, MUL, DIV, POW)
      - 10+ = unary ops (SIN, COS, etc.)
    - values: List of corresponding values or variable indices
    """
    from sympy import preorder_traversal

    var_to_idx = {name: i for i, name in enumerate(var_names)}
    tokens = []
    values = []

    # Operator mappings
    # Binary operators: 1-9
    # Unary operators: 10-27
    func_op_map = {
        "sin": 10,
        "cos": 11,
        "tan": 12,
        "sinh": 13,
        "cosh": 14,
        "tanh": 15,
        "exp": 16,
        "log": 17,
        "asin": 19,
        "acos": 20,
        "atan": 21,
        "arcsin": 19,  # Alias for asin
        "arccos": 20,  # Alias for acos
        "arctan": 21,  # Alias for atan
        "sqrt": 26,
        "abs": 24,
        "Abs": 24,
    }

    def get_node_info(node):
        if node.is_Number:
            return (0, float(node))
        elif node.is_Symbol:
            name = str(node)
            if name in var_to_idx:
                return (-1, float(var_to_idx[name]))
            else:
                return (0, 0.0)
        elif node.is_Function:
            func_name = type(node).__name__.lower()
            op = func_op_map.get(func_name)
            if op is not None:
                return (op, 0.0)
            else:
                print(f"Warning [Formula {expr}]: Unsupported function '{func_name}'")
                return (None, None)
        elif node.is_Add:
            return (1, 0.0)
        elif node.is_Mul:
            return (3, 0.0)
        elif node.is_Pow:
            return (5, 0.0)
        elif hasattr(node, "func") and node.func.__name__ == "Min":
            return (6, 0.0)
        elif hasattr(node, "func") and node.func.__name__ == "Max":
            return (7, 0.0)
        else:
            print(
                f"Warning [Formula {expr}]: Unsupported node type '{type(node).__name__}'"
            )
            return (None, None)

    nodes = []
    for _node in preorder_traversal(expr, keys=True):
        token, value = get_node_info(_node)
        is_leaf = token <= 0
        arg_count = 2 if 1 <= token <= 9 else 1 if 10 <= token else 0
        nodes.append(
            {
                "idx": len(nodes),
                "node": _node,
                "token": token,
                "value": value,
                "is_leaf": is_leaf,
                "arg_count": arg_count,
                "parent": -1,
                "children": [],
                "done_children": [],
            }
        )

    # build tree
    trans_queue = []
    call_stack = [0]
    for node in nodes[1:]:
        parent_idx = call_stack[-1]
        par_node = nodes[parent_idx]
        par_node["children"].append(node["idx"])
        node["parent"] = parent_idx
        if node["is_leaf"]:
            node["done_children"].append(node["idx"])
            while len(call_stack):
                lst_nd = nodes[call_stack[-1]]
                if len(lst_nd["children"]) == lst_nd["arg_count"]:
                    call_stack.pop()
                    if len(lst_nd["done_children"]) == lst_nd["arg_count"]:
                        trans_queue.append(lst_nd)
                else:
                    break
        else:
            call_stack.append(node["idx"])
        # print([nodes[idx] for idx in call_stack])

    assert len(call_stack) == 0

    # make concise order
    while len(trans_queue):
        node = trans_queue.pop()
        idx = node["idx"]
        if idx == 0:
            break
        par_node = nodes[node["parent"]]
        par_node["done_children"].append(idx)
        if len(par_node["done_children"]) == par_node["arg_count"]:
            trans_queue.append(par_node)

    # fill tokens and values
    def dfs(idx):
        node = nodes[idx]
        tokens.append(node["token"])
        values.append(node["value"])
        for cidx in node["done_children"]:
            dfs(cidx)

    dfs(0)

    return tokens, values


def tokens_to_string(
    tokens: List[int], values: List[float], var_names: List[str]
) -> Tuple[str, List[str]]:
    """
    Convert tokens and values (in PREFIX notation) back to a formula string for verification.
    Processes tokens from right to left to handle prefix notation.
    Returns (reconstructed_formula, list_of_warnings)
    """
    warnings = []

    # Sanity check: tokens and values must have same length
    if len(tokens) != len(values):
        warnings.append(
            f"Token/value length mismatch: {len(tokens)} tokens vs {len(values)} values"
        )
        return "ERROR", warnings

    # Operator mappings
    unary_ops = {
        10: "sin",
        11: "cos",
        12: "tan",
        13: "sinh",
        14: "cosh",
        15: "tanh",
        16: "exp",
        17: "log",
        18: "inv",
        19: "asin",
        20: "acos",
        21: "atan",
        22: "loose_log",
        23: "loose_inv",
        24: "abs",
        25: "-",
        26: "sqrt",
        27: "loose_sqrt",
    }

    binary_ops = {
        1: "+",
        2: "-",
        3: "*",
        4: "/",
        5: "**",
        6: "min",
        7: "max",
        8: "loose_div",
        9: "loose_pow",
    }

    # Process from right to left for prefix notation
    stack = []
    for token, value in reversed(list(zip(tokens, values))):
        if token == 0:  # Constant
            stack.append(str(value))
        elif token == -1:  # Variable
            idx = int(value)
            if idx >= len(var_names):
                warnings.append(
                    f"Variable index {idx} out of range (max {len(var_names)-1})"
                )
            var_name = var_names[idx] if idx < len(var_names) else f"var{idx}"
            stack.append(var_name)
        elif token in unary_ops:
            # Unary operator - pop one operand
            if not stack:
                warnings.append("Stack underflow for unary op")
                return "ERROR", warnings
            operand = stack.pop()
            op_name = unary_ops[token]
            if token == 25:  # NEG
                stack.append(f"(-{operand})")
            elif token == 18:  # INV
                stack.append(f"(1/{operand})")
            else:
                stack.append(f"{op_name}({operand})")
        elif token in binary_ops:
            # Binary operator - pop two operands
            # In prefix: OP left right, but we're processing right-to-left
            # So we pop in order: left, right
            if len(stack) < 2:
                warnings.append("Stack underflow for binary op")
                return "ERROR", warnings
            left = stack.pop()
            right = stack.pop()
            op_symbol = binary_ops[token]
            stack.append(f"({left} {op_symbol} {right})")
        else:
            warnings.append(f"Unknown operator token: {token}")
            stack.append(f"UNKNOWN_OP({token})")

    if len(stack) != 1:
        warnings.append(f"Stack has {len(stack)} elements, expected 1")
        return "ERROR", warnings

    return stack[0], warnings


def write_digest_file(
    output_path: str,
    data_filename: str,
    num_vars: int,
    num_dps: int,
    tokens: List[int],
    values: List[float],
    original_formula: str = "",
    reconstructed_formula: str = "",
) -> None:
    """
    Write a digest file in the required format.
    """
    with open(output_path, "w") as f:
        f.write("1\n")
        f.write(f"{num_vars}\n")
        f.write(f"{num_dps}\n")
        f.write(f"{len(tokens)}\n")

        # Write tokens (space-separated integers)
        tokens_str = " ".join(str(t) for t in tokens)
        f.write(f"{tokens_str}\n")

        # Write values (space-separated floats with double precision)
        values_str = " ".join(f"{v:.15g}" for v in values)
        f.write(f"{values_str}\n")

        # Write data filename
        f.write(f"data/ai_feyn/Feynman_with_units/{data_filename}\n")

        # Write original and reconstructed formulas for debugging
        f.write(f"{original_formula}\n")
        f.write(f"{reconstructed_formula}\n")


def create_multi_expression_file(
    digest_csv_filename: str, output_path: str, num_dps: int = 10000
) -> None:
    """
    Create a multi-expression input file with all formulas from the CSV.
    Each formula will use the specified num_dps.

    Args:
        digest_csv_filename: Path to the CSV file with formulas
        output_path: Path to write the multi-expression file
        num_dps: Number of data points per expression (default: 10000)
    """
    # Read CSV file
    df = pd.read_csv(digest_csv_filename)
    df = df[df["Filename"].notna()]  # Filter out rows with missing filenames

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    successful_exprs = []

    # Process each formula
    for i, row in df.iterrows():
        formula_str = row["Formula"]
        data_filename = row["Filename"]
        num_vars_input = int(row["# variables"])  # Input variables only
        num_vars = num_vars_input + 1  # +1 for output variable

        try:
            # Construct variable names properly (same as main())
            var_names = [
                row[f"v{i+1}_name"] for i in range(num_vars - 1) if row[f"v{i+1}_name"]
            ] + [row["Output"]]

            # Parse and binarize the formula (same as main())
            binary_expr = parse_and_binarize(formula_str, var_names)

            if binary_expr is None:
                raise Exception(f"Failed to parse formula")

            # Convert to tokens
            tokens, token_values = expr_to_tokens(binary_expr, var_names)

            # Reconstruct to verify
            reconstructed, _ = tokens_to_string(tokens, token_values, var_names)

            successful_exprs.append(
                {
                    "num_vars": num_vars_input,  # Store input vars only (like in singles)
                    "num_dps": num_dps,
                    "tokens": tokens,
                    "values": token_values,
                    "data_filename": data_filename,
                    "original_formula": formula_str,
                    "reconstructed_formula": reconstructed,
                }
            )

        except Exception as e:
            print(f"Warning: Skipping formula {i+1} due to error: {e}")
            import traceback

            print(traceback.format_exc())
            continue

    # Write multi-expression file
    with open(output_path, "w") as f:
        # Write num_exprs
        f.write(f"{len(successful_exprs)}\n")

        # Write each expression
        for expr_data in successful_exprs:
            f.write(f"{expr_data['num_vars']}\n")
            f.write(f"{expr_data['num_dps']}\n")
            f.write(f"{len(expr_data['tokens'])}\n")

            # Write tokens
            tokens_str = " ".join(str(t) for t in expr_data["tokens"])
            f.write(f"{tokens_str}\n")

            # Write values
            values_str = " ".join(f"{v:.15g}" for v in expr_data["values"])
            f.write(f"{values_str}\n")

            # Write data filename
            f.write(f"data/ai_feyn/Feynman_with_units/{expr_data['data_filename']}\n")

            # # Write formulas for debugging
            # f.write(f"{expr_data['original_formula']}\n")
            # f.write(f"{expr_data['reconstructed_formula']}\n")

    print(f"Created multi-expression file: {output_path}")
    print(f"Total expressions: {len(successful_exprs)}")
    print(f"Data points per expression: {num_dps}")


def main(
    digest_csv_filename: str, quiet: bool = False, write_output: bool = False
) -> None:
    # Read CSV file
    df = pd.read_csv(digest_csv_filename)
    df = df[df["Filename"].notna()]  # Filter out rows with missing filenames

    # Create output directory if writing output
    output_dir = "data/ai_feyn/singles"
    if write_output:
        os.makedirs(output_dir, exist_ok=True)
        if not quiet:
            print(f"Output directory: {output_dir}")

    # Statistics
    total_processed = 0
    total_warnings = 0
    total_tokens = 0
    failed_count = 0
    files_written = 0

    for i, row in df.iterrows():
        formula_str = row["Formula"]
        num_vars = int(row["# variables"]) + 1  # +1 for output variable
        var_names = [
            row[f"v{i+1}_name"] for i in range(num_vars - 1) if row[f"v{i+1}_name"]
        ] + [row["Output"]]

        if not quiet:
            print(f"\n{'='*50}")
            print(f"Processing formula {i+1}: {formula_str}")
            print(f"Variables: {var_names}")
            print(f"{'='*50}")

        # Parse and binarize the formula
        binary_expr = parse_and_binarize(formula_str, var_names)

        if binary_expr is None:
            if not quiet:
                print("Failed to parse formula")
            print(f"WARNING [Formula {i+1}]: Failed to parse formula '{formula_str}'")
            failed_count += 1
            continue

        if not quiet:
            print(f"\nBinarized: {binary_expr}")

        # Convert to C++ token format
        tokens, token_values = expr_to_tokens(binary_expr, var_names)

        # Sanity check: Check if all variables are used
        used_var_indices = set()
        for tok, val in zip(tokens, token_values):
            if tok == -1:
                used_var_indices.add(int(val))

        unused_vars = []
        for idx, var_name in enumerate(var_names[:-1]):
            if idx not in used_var_indices:
                unused_vars.append(var_name)

        if unused_vars:
            warning = (
                f"WARNING [Formula {i+1}]: Variables not used in formula: {unused_vars}"
            )
            print(warning)
            total_warnings += 1

        if not quiet:
            print(f"\nTokens ({len(tokens)} total):")
            print(f"  tokens: {tokens}")
            print(f"  values: {token_values}")

        # Reconstruct formula to verify correctness
        reconstructed, warnings = tokens_to_string(tokens, token_values, var_names)

        if warnings:
            for warning in warnings:
                print(f"WARNING [Formula {i+1}]: {warning}")
            total_warnings += len(warnings)

        if not quiet:
            print(f"\nReconstructed formula: {reconstructed}")
            print("=" * 50)

        total_processed += 1
        total_tokens += len(tokens)

        # Write digest file if requested
        if write_output:
            data_filename = row["Filename"]
            num_vars = int(row["# variables"])  # Exclude output variable
            num_dps = 1000000  # Always 1e6

            output_filename = f"input_{i+1:03d}.txt"
            output_path = os.path.join(output_dir, output_filename)

            try:
                write_digest_file(
                    output_path,
                    data_filename,
                    num_vars,
                    num_dps,
                    tokens,
                    token_values,
                    formula_str,
                    reconstructed,
                )
                files_written += 1
                if not quiet:
                    print(f"  Written to: {output_path}")
            except Exception as e:
                print(f"WARNING [Formula {i+1}]: Failed to write output file: {e}")
                total_warnings += 1

    # Print statistics
    print(f"\n{'='*50}")
    print("STATISTICS")
    print(f"{'='*50}")
    print(f"Total formulas processed: {total_processed}")
    print(f"Failed to parse: {failed_count}")
    print(f"Total warnings: {total_warnings}")
    if total_processed > 0:
        print(f"Average tokens per formula: {total_tokens / total_processed:.2f}")
    if write_output:
        print(f"Files written: {files_written} to {output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys

    digest_csv_filename = "data/ai_feyn/FeynmanEquations.csv"

    # Check if we should create multi-expression file
    if "--multi" in sys.argv or "-m" in sys.argv:
        num_dps = 10000
        output_path = None

        # Allow custom num_dps and output path
        for arg in sys.argv:
            if arg.startswith("--dps="):
                num_dps = int(arg.split("=")[1])
            elif arg.startswith("--output="):
                output_path = arg.split("=", 1)[1]

        # Auto-generate output path if not specified
        if output_path is None:
            # Format: input_100_10k.txt or input_100_100k.txt
            if num_dps >= 1000:
                dps_str = f"{num_dps // 1000}k"
            else:
                dps_str = str(num_dps)
            output_path = f"data/ai_feyn/multi/input_100_{dps_str}.txt"

        create_multi_expression_file(digest_csv_filename, output_path, num_dps)
    else:
        # Original single-expression mode
        quiet = "--quiet" in sys.argv or "-q" in sys.argv
        write_output = "--write" in sys.argv or "-w" in sys.argv

        main(digest_csv_filename, quiet=quiet, write_output=write_output)
