#include "../utils/utils.h"
#include <iostream>
#include <string>
#include <cstdlib>

// Forward declaration from gpu_subtree_batch_state.cu
int run_evolution_benchmark_gpu_subtree_state(int start_gen,
                                              int end_gen,
                                              const std::string &data_dir);

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <start_gen> <end_gen> [data_dir]" << std::endl;
        std::cerr << "  start_gen, end_gen: inclusive generation indices" << std::endl;
        std::cerr << "  data_dir (optional): path to evolution data directory" << std::endl;
        std::cerr << "    default: data/evolution_short_small" << std::endl;
        return 1;
    }

    int start_gen = std::atoi(argv[1]);
    int end_gen   = std::atoi(argv[2]);
    std::string data_dir = (argc >= 4) ? argv[3]
                                       : std::string("data/evolution_short_small");

    return run_evolution_benchmark_gpu_subtree_state(start_gen, end_gen, data_dir);
}
