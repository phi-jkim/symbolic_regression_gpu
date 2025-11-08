
// Data input parsing
//  including toml parsing, token mapping, etc
//

#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
std::uniform_real_distribution<double> unif(0, 10);
unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
std::default_random_engine re(seed);

#include <string>
#include <sstream>
#include <iomanip>

std::string format_formula(int *tokens, double *values, int num_tokens)
{
    std::vector<std::string> stack;

    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int token = tokens[i];

        switch (token)
        {
        case 0: // Constant
            stack.push_back(std::to_string(values[i]));
            break;
        case -1: // Variable
            stack.push_back("x" + std::to_string(static_cast<int>(values[i])));
            break;
        case 1: // ADD
        case 2: // SUB
        case 3: // MUL
        case 4: // DIV
            if (stack.size() < 2)
                return "INVALID";
            {
                std::string a = stack.back();
                stack.pop_back();
                std::string b = stack.back();
                stack.pop_back();
                const char *op = (token == 1) ? " + " : (token == 2) ? " - "
                                                    : (token == 3)   ? " * "
                                                                     : " / ";
                stack.push_back("(" + a + op + b + ")");
            }
            break;
        case 5: // SIN
        case 6: // COS
        case 7: // EXP
        case 8: // LOG
            if (stack.empty())
                return "INVALID";
            {
                std::string a = stack.back();
                stack.pop_back();
                const char *func = (token == 5) ? "sin" : (token == 6) ? "cos"
                                                      : (token == 7)   ? "exp"
                                                                       : "log";
                stack.push_back(std::string(func) + "(" + a + ")");
            }
            break;
        default:
            return "?";
        }
    }

    if (stack.size() != 1)
    {
        std::cerr << "ERROR: Invalid stack state: ";
        for (const auto &item : stack)
        {
            std::cerr << item << ", ";
        }
        std::cerr << std::endl;
        return "INVALID";
    }
    return stack[0];
}
typedef struct
{

} InputInfo;

// InputInfo parse_input_info(std::string input_file)
// {
// }

double eval_op(int op, double val1, double val2)
{
    switch (op)
    {
    case 1:
        return val1 + val2;
    case 2:
        return val1 - val2;
    case 3:
        return val1 * val2;
    case 4:
        return val1 / val2;
    case 5:
        return sin(val1);
    case 6:
        return cos(val1);
    case 7:
        return exp(val1);
    case 8:
        return log(val1);
    default:
        return 0;
    }
}

double stk[1024];

double eval_tree_cpu(int *tokens, double *values, double *x, int num_tokens, int num_vars)
{
    int sp = 0;
    double tmp, val1, val2;
    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int tok = tokens[i];
        if (tok > 0) // operation
        {
            val1 = stk[sp - 1], sp--;
            if (tok <= 4)
                val2 = stk[sp - 1], sp--;

            tmp = eval_op(tok, val1, val2);
            stk[sp] = tmp, sp++;
        }
        else if (tok == 0) // constant
        {
            stk[sp] = values[i], sp++;
        }
        else if (tok == -1) // variable
        {
            stk[sp] = x[(int)values[i]], sp++;
        }
        else
        {
            // throw std::runtime_error("Invalid token");
            std::cerr << "Invalid token: " << tok << std::endl;
        }
    }
    return stk[0];
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    // @here time input parsing
    // InputInfo input_info = parse_input_info(input_file);

    // load any files from input info

    int num_vars;
    int num_dps;
    double **vars;
    // vars[i][j] means for variable x_i, its value in jth datapoint is vars[i][j].
    // i.e. i has small dim(=num_var < 10), j has large dim(=num_dps ~= 1M).

    // test with placeholder for x4 = 0.1 x1^2 + 4 x1 * x3 + sin(x2 * x3 + 3)
    num_vars = 4;
    num_dps = 1000;
    vars = (double **)malloc(num_vars * sizeof(double *));
    for (int i = 0; i < num_vars; i++)
    {
        vars[i] = (double *)malloc(num_dps * sizeof(double));
    }

    for (int j = 0; j < num_dps; j++)
    {
        // x1, x2, x3: random 0~10

        vars[0][j] = unif(re);
        vars[1][j] = unif(re);
        vars[2][j] = unif(re);
        vars[3][j] = 0.1 * vars[0][j] * vars[0][j] + 0.3 * vars[0][j] * vars[2][j] + 5 * sin(vars[1][j] * vars[2][j] + 3);
    }

    // write test text output boilerplate
    std::ofstream test_file("test.txt", std::ios::out);
    test_file << num_vars << " " << num_dps << "\n";
    for (int i = 0; i < num_vars; i++)
    {
        for (int j = 0; j < num_dps; j++)
        {
            test_file << vars[i][j] << " ";
        }
        test_file << "\n";
    }

    // types: 1 ~ P is operators (add, sin, etc)
    //        0 means constant
    //        -1 means variables
    // prefix notation of a formula +(OP1, OP2), ...
    // 1 = ADD, 2 = SUB, 3 = MUL, 4 = DIV, 5 = SIN, 6 = COS, 7 = EXP, 8 = LOG
    int num_tokens;
    int *tokens;
    double *values;

    /*
    ADD(
      ADD(
        MUL(0.1, MUL(VAR(0), VAR(0))), // 3, 0, 3, -1, -1 | 0, 0.1, 0, 0, 0
        MUL(0.3, MUL(VAR(0), VAR(2)))  // 3, 0, 3, -1, -1 | 0, 0.3, 0, 0, 2
        ),
      MUL(5, SIN(ADD(MUL(VAR(1), VAR(2)), 3))) // 3, 0, 5, 1, 3, -1, -1, 0 | 0, 5, 0, 0, 0, 1, 2, 3
    )
    */
    int tmp_tokens[] = {
        1,
        1, // ADD(ADD(
        3, // MUL (
        0,
        3,
        -1,
        -1, // ),
        3,  // MUL (
        0,
        3,
        -1,
        -1, // ) )
        3,  // MUL (
        0,
        5,
        1,
        3,
        -1,
        -1,
        0, // ) )
    };
    double tmp_values[] = {
        0, 0, 0, 0.1, 0, 0, 0, 0, 0.3, 0, 0, 2, 0, 5, 0, 0, 0, 1, 2, 3};

    num_tokens = sizeof(tmp_tokens) / sizeof(tmp_tokens[0]);
    tokens = tmp_tokens;
    values = tmp_values;

    double *pred = (double *)malloc(num_dps * sizeof(double));

    // run the single tree evaluation (with timing)
    for (int dp = 0; dp < num_dps; dp++)
    {
        double *x = (double *)malloc(num_vars * sizeof(double));
        for (int i = 0; i < num_vars; i++)
            x[i] = vars[i][dp];

        double y = eval_tree_cpu(tokens, values, x, num_tokens, num_vars);
        pred[dp] = y;
    }

    std::cout << "num_tokens: " << num_tokens << std::endl;
    std::cout << format_formula(tokens, values, num_tokens) << std::endl;

    // write the output info to output file
    std::ofstream result_file("output.txt", std::ios::out);
    result_file << num_dps << "\n";
    for (int i = 0; i < num_dps; i++)
    {
        result_file << pred[i] << "\n";
    }
    result_file.close();

    return 0;
}