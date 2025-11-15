"""
go through a given data file (e.g. data/ai_feyn/multi/input_100_100k.txt), check if
 * (all data points are valid (i.e. no NaN, no Inf, no OutOfBounds),)
 * can go through the tokens and values with only stack size 4
 * correctly parsed with single token at last
 * tell the set of operations used
 * tell the max of num dp, num vars, num tokens, num exprs
"""

import argparse


def get_minmax(field, dicts):
    arr = [x[field] for x in dicts]
    return min(arr), max(arr)


def get_pop_count(token):
    if token <= 0:
        return 0
    if token <= 9:
        return 2
    if token <= 27:
        return 1
    raise ValueError(f"Invalid token: {token}")


def main():
    # read file
    lines = []
    with open(args.input, "r") as f:
        lines = f.readlines()

    num_exprs = int(lines[0].strip())
    expr_info = []
    cur_line_idx = 1
    for exp_idx in range(num_exprs):
        num_vars = int(lines[cur_line_idx].strip())
        cur_line_idx += 1
        num_dps = int(lines[cur_line_idx].strip())
        cur_line_idx += 1
        num_tokens = int(lines[cur_line_idx].strip())
        cur_line_idx += 1
        tokens = [int(x) for x in lines[cur_line_idx].strip().split()]
        cur_line_idx += 1
        values = [float(x) for x in lines[cur_line_idx].strip().split()]
        cur_line_idx += 1
        data_filename = lines[cur_line_idx].strip()
        cur_line_idx += 1

        expr_info.append(
            {
                "num_vars": num_vars,
                "num_dps": num_dps,
                "num_tokens": num_tokens,
                "tokens": tokens,
                "values": values,
                "data_filename": data_filename,
            }
        )

    # print range of nums
    print("\n" + "=" * 50)
    print("num_exprs:  ", num_exprs)
    print("num_vars:   ", *get_minmax("num_vars", expr_info))
    print("num_dps:    ", *get_minmax("num_dps", expr_info))
    print("num_tokens: ", *get_minmax("num_tokens", expr_info))
    print("=" * 50 + "\n")

    # check if can go through the tokens and values with only stack size 4
    for expr in expr_info:
        op_set = set()
        stack_size = 0
        max_size = 0
        tokens = expr["tokens"]
        for token in reversed(tokens):
            to_pop = get_pop_count(token)
            op_set.add(token)
            if stack_size < to_pop:
                raise ValueError(f"Invalid expression (stack underflow): {expr}")
            stack_size -= to_pop
            stack_size += 1
            max_size = max(max_size, stack_size)
        if stack_size != 1:
            raise ValueError(f"Invalid expression (stack size != 1): {expr}")
        expr["max_stack_size"] = max_size
        expr["op_set"] = op_set
        if max_size > 4:
            print(expr)

    print("max_stack_size: ", *get_minmax("max_stack_size", expr_info))
    print("op_set: ", set().union(*[x["op_set"] for x in expr_info]))
    print("=" * 50 + "\n\n")

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check data file")
    parser.add_argument(
        "--input",
        type=str,
        default="data/ai_feyn/multi/input_100_100k.txt",
        help="Input data file",
    )

    args = parser.parse_args()

    main()
