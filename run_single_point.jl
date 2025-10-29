# Run a single-point evaluation using SymbolicRegression.eval_tree_array.
#
# How to run:
#   julia --project=/home/jinhakim/SymbolicRegression.jl /home/jinhakim/symbolic_regression_gpu/run_single_point.jl
#
# This example evaluates the expression "sin(x1) + cos(x2) + 3" on a single data point X = [0.5, 1.2].
using SymbolicRegression: Options, OperatorEnum, parse_expression, eval_tree_array
using Printf: @sprintf

function main()
    operators = OperatorEnum(; binary_operators=(+, -, *, /), unary_operators=(sin, cos, exp, log))
    options = Options(; operators, define_helper_functions=false)
    expr = parse_expression("sin(x1) + cos(x2) + 3"; operators, variable_names=["x1", "x2"]) 
    X = reshape(Float32[0.5, 1.2], 2, 1)
    t1 = @elapsed begin
        julia_out, complete = eval_tree_array(expr, X, options)
    end
    println("sr_eval  = ", Float32(only(julia_out)))
    println("complete = ", complete)
    println(@sprintf("eval_time_ms = %.3f", t1 * 1e3))

    iters = 1000
    # Warm-up
    eval_tree_array(expr, X, options)
    t_avg1 = @elapsed begin
        for _ in 1:iters
            eval_tree_array(expr, X, options)
        end
    end
    println(@sprintf("avg_time_us (iters=%d) = %.3f", iters, (t_avg1/iters)*1e6))

    Y = reshape(Float32[0.5, 0.3], 2, 1)
    t2 = @elapsed begin
        julia_out, complete = eval_tree_array(expr, Y, options)
    end
    println("sr_eval  = ", Float32(only(julia_out)))
    println("complete = ", complete)
    println(@sprintf("eval_time_ms = %.3f", t2 * 1e3))
    # Warm-up
    eval_tree_array(expr, Y, options)
    t_avg2 = @elapsed begin
        for _ in 1:iters
            eval_tree_array(expr, Y, options)
        end
    end
    println(@sprintf("avg_time_us (iters=%d) = %.3f", iters, (t_avg2/iters)*1e6))
end

isinteractive() || main() 
