import os
import random
import shutil

# Token mappings based on src/utils/utils.cpp
# Binary (1-9)
OP_ADD = 1
OP_SUB = 2
OP_MUL = 3
OP_DIV = 4
OP_POW = 5
OP_MIN = 6
OP_MAX = 7
OP_LOOSE_DIV = 8
OP_LOOSE_POW = 9

# Unary (10-27)
OP_SIN = 10
OP_COS = 11
OP_TAN = 12
OP_SINH = 13
OP_COSH = 14
OP_TANH = 15
OP_EXP = 16
OP_LOG = 17
OP_INV = 18
OP_ASIN = 19
OP_ACOS = 20
OP_ATAN = 21
OP_LOOSE_LOG = 22
OP_LOOSE_INV = 23
OP_ABS = 24
OP_NEG = 25
OP_SQRT = 26
OP_LOOSE_SQRT = 27

# Special
CONST = 0
VAR = -1

BINARY_OPS = [OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS] # Reduced set for stability

class Node:
    def __init__(self, token, value=0.0):
        self.token = token
        self.value = value
        self.left = None
        self.right = None

    def size(self):
        s = 1
        if self.left: s += self.left.size()
        if self.right: s += self.right.size()
        return s

    def to_prefix(self, tokens, values):
        tokens.append(self.token)
        values.append(self.value)
        if self.left: self.left.to_prefix(tokens, values)
        if self.right: self.right.to_prefix(tokens, values)

    def copy(self):
        new_node = Node(self.token, self.value)
        if self.left: new_node.left = self.left.copy()
        if self.right: new_node.right = self.right.copy()
        return new_node

def random_tree(depth, max_depth, num_vars):
    if depth >= max_depth or (depth > 1 and random.random() < 0.3):
        # Terminal
        if random.random() < 0.5:
            return Node(VAR, float(random.randint(0, num_vars - 1)))
        else:
            return Node(CONST, random.uniform(-5, 5))
    
    op = random.choice(BINARY_OPS)
    node = Node(op)
    node.left = random_tree(depth + 1, max_depth, num_vars)
    if op < 10: # Binary
        node.right = random_tree(depth + 1, max_depth, num_vars)
    return node

def mutate(node, num_vars):
    if random.random() < 0.2:
        return random_tree(0, 3, num_vars)
    
    new_node = Node(node.token, node.value)
    if node.left:
        new_node.left = mutate(node.left, num_vars)
    if node.right:
        new_node.right = mutate(node.right, num_vars)
    return new_node

def generate_data(num_generations=5, pop_size=100, num_vars=5, num_dps=100000, output_dir="data/evolution"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Generate shared data file
    data_file = os.path.join(output_dir, "shared_data.txt")
    
    # Generate random data using standard random module
    data = []
    for _ in range(num_vars + 1):
        col = [random.uniform(-5, 5) for _ in range(num_dps)]
        data.append(col)
    
    # Save data file in format expected by loader
    # Space separated values
    with open(data_file, "w") as f:
        for j in range(num_dps):
            line = []
            for i in range(num_vars + 1):
                line.append(f"{data[i][j]:.6f}")
            f.write(" ".join(line) + "\n")

    # Initial population
    population = [random_tree(0, 5, num_vars) for _ in range(pop_size)]

    for gen in range(num_generations):
        filename = os.path.join(output_dir, f"gen_{gen}.txt")
        print(f"Generating {filename}...")
        
        with open(filename, "w") as f:
            f.write(f"{pop_size}\n")
            for tree in population:
                tokens = []
                values = []
                tree.to_prefix(tokens, values)
                
                f.write(f"{num_vars} {num_dps} {len(tokens)}\n")
                f.write(" ".join(map(str, tokens)) + "\n")
                f.write(" ".join(map(str, values)) + "\n")
                f.write(f"{data_file}\n")
        
        # Evolve
        new_pop = []
        # Elitism
        new_pop.extend([t.copy() for t in population[:10]])
        
        while len(new_pop) < pop_size:
            parent = random.choice(population)
            child = mutate(parent, num_vars)
            new_pop.append(child)
        
        population = new_pop

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evolution data for symbolic regression benchmark")
    parser.add_argument("--gens", type=int, default=5, help="Number of generations")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--vars", type=int, default=5, help="Number of variables")
    parser.add_argument("--dps", type=int, default=100000, help="Number of data points")
    parser.add_argument("--out", type=str, default="data/evolution", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    generate_data(num_generations=args.gens, pop_size=args.pop, num_vars=args.vars, num_dps=args.dps, output_dir=args.out)
