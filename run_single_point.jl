# Run a single-point evaluation using SymbolicRegression.eval_tree_array.
#
# How to run:
#   julia --project=/home/jinhakim/SymbolicRegression.jl /home/jinhakim/symbolic_regression_gpu/run_single_point.jl
#
# This example evaluates the expression "sin(x1) + cos(x2) + 3" on a single data point X = [0.5, 1.2].
using SymbolicRegression: Options, OperatorEnum, parse_expression, eval_tree_array

function main()
    operators = OperatorEnum(; binary_operators=(+, -, *, /), unary_operators=(sin, cos, exp, log))
    options = Options(; operators, define_helper_functions=false)
    expr = parse_expression("sin(x1) + cos(x2) + 3"; operators, variable_names=["x1", "x2"]) 
    X = reshape(Float32[0.5, 1.2], 2, 1)
    julia_out, complete = eval_tree_array(expr, X, options)
    println("sr_eval  = ", Float32(only(julia_out)))
    println("complete = ", complete)
    Y = reshape(Float32[0.5, 0.3], 2, 1)
    julia_out, complete = eval_tree_array(expr, Y, options)
    println("sr_eval  = ", Float32(only(julia_out)))
    println("complete = ", complete)
end

isinteractive() || main() 
