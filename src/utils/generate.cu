#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../utils/defs.h"
#include "../utils/gpu_kernel.h"

// Define MAX_STACK_SIZE if not already defined
#ifndef MAX_TREE_NODE_NUM
#define MAX_TREE_NODE_NUM 128 // number of nodes in tree 
#endif

// Random engine helper
struct RandomEngine {
    curandState state;
    __device__ RandomEngine(unsigned int seed, unsigned int id, unsigned int offset) {
        curand_init(seed, id, offset, &state);
    }
    __device__ float operator()() {
        return curand_uniform(&state);
    }
    __device__ unsigned int operator()(unsigned int range) {
        return curand(&state) % range;
    }
};

// Hash function for seeding
// __device__ unsigned int hash(unsigned int a, unsigned int b, unsigned int c) {
//     a ^= b; a *= 0x5bd1e995; a ^= c; a *= 0x5bd1e995;
//     return a;
// }

/**
 * @brief CUDA kernel to generate a population of genetic programming (GP) trees.
 *
 * @tparam multiOutput Flag indicating if multi-output nodes should be generated.
 * @param popSize Number of trees to generate.
 * @param gpLen Maximum length of each GP tree.
 * @param varLen Number of variables available for leaf nodes.
 * @param outLen Number of output nodes for multi-output mode.
 * @param constSamplesLen Number of constant samples available.
 * @param outProb Probability of generating an output node (for multi-output).
 * @param constProb Probability of generating a constant node.
 * @param value_res Output array for node values.
 * @param type_res Output array for node types.
 * @param subtree_size_res Output array for subtree sizes.
 * @param keys Random engine seed keys.
 * @param depth2leafProbs Array defining probability of generating leaf nodes based on depth.
 * @param rouletteFuncs Roulette wheel probabilities for selecting function nodes.
 * @param constSamples Array of constant values available for leaf nodes.
 */
template<bool multiOutput = false>
__global__ void treeGPGenerate(
    const unsigned int popSize,
    const unsigned int gpLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const unsigned int constSamplesLen,
    const float outProb,
    const float constProb,
    float* value_res, 
    int16_t* type_res, 
    int16_t* subtree_size_res, 
    const unsigned int* keys, 
    const float* depth2leafProbs, 
    const float* rouletteFuncs, 
    const float* constSamples
)
{   
    // Calculate tree index
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= popSize)
        return;
    // Initialize probabilities and random engine
    float leafProbs[MAX_FULL_DEPTH]{};
    float funcRoulette[Function::END]{};
    RandomEngine engine(keys[0], n, keys[1]);

    // Load probabilities into local memory
    #pragma unroll
    for (int i = 0; i < MAX_FULL_DEPTH; i++)
    {
        leafProbs[i] = depth2leafProbs[i];
    }
    #pragma unroll
    for (int i = 0; i < Function::END; i++)
    {
        funcRoulette[i] = rouletteFuncs[i];
    }

    // stack memory for node generation
    GPNode* gp = (GPNode*)alloca(MAX_TREE_NODE_NUM * sizeof(GPNode));  // result gp array
    NchildDepth* childsAndDepth = (NchildDepth*)alloca(MAX_TREE_NODE_NUM * sizeof(NchildDepth));  // stack
    childsAndDepth[0] = { 1, 0 };  // Start with the root node, {child, depth}
    int topGP = 0, top = 1;
    int pendingChildren = 1;           // root descriptor still to be expanded
    bool chk_first = true;

    // generate
    while (top > 0 && topGP < gpLen && topGP < MAX_TREE_NODE_NUM)
    {
        NchildDepth cd = childsAndDepth[--top];  // get one in stack
        cd.childs--;               // current descriptor satisfied

        NchildDepth cdNew{};  // new childDepth initialize to 0
        GPNode node{.0f, (int16_t)(0), (int16_t)(0)};  //new node, {value, type, subtree_size}

        // Determine whether to generate a leaf or function node
        int remainingSlots = gpLen - topGP - pendingChildren - 1; // slots left for *new* children after this node
        const float leafProb = (cd.depth < MAX_FULL_DEPTH) ? leafProbs[cd.depth] : 1.0f; // force leaf beyond max depth
        bool forceLeaf = (remainingSlots <= 0);
        if (!forceLeaf && engine() >= leafProb)
        {   
            // generate non-leaf (function) node
            float r = engine();
            int k = 0;  // function type
            #pragma unroll
            for (int i = Function::END - 1; i >= 0; i--)
            {
                if (r >= funcRoulette[i])
                {
                    k = i + 1;
                    break;
                }
            }
            // Determine node type from opcode arity
            int16_t type;
            if (k == Function::IF) {
                type = NodeType::TFUNC;
            } else if (op_arity(k) == 2) {
                type = NodeType::BFUNC;
            } else {
                type = NodeType::UFUNC;
            }
            if constexpr (multiOutput)
            {
                if (engine() <= outProb)
                {   
                    // output node
                    int16_t outType = type | NodeType::OUT_NODE;
                    // value(float32) contains the function(int16_t) and outIndex(int16_t) info when using multiOutput mode
                    union OutNodeValue {
                        float asFloat;
                        struct {
                            int16_t func;
                            int16_t outIdx;
                        } asInt;
                    };
                    OutNodeValue outNode;
                    outNode.asInt.func = (int16_t)k;
                    outNode.asInt.outIdx = static_cast<int16_t>(engine(outLen)); 
                    node = GPNode{ outNode.asFloat, outType, 1 };  // subtreesize temporarily set to 1
                }
            }
            // node.subtreeSize == 0 means not multiOutput node
            if (node.subtreeSize == 0)
            {
                node = GPNode{ float(k), type, 1 };
            }
            int16_t childNeed = (type == NodeType::TFUNC ? 3 : type == NodeType::BFUNC ? 2 : 1);
            // If not enough remaining slots for the children, downgrade to leaf
            if(childNeed > remainingSlots){
                node = GPNode{ constSamples[engine(constSamplesLen)], NodeType::CONST, 1 }; // force const leaf
                childNeed = 0;
            }
            cdNew = NchildDepth{ childNeed, int16_t(cd.depth + 1) };
        }
        else
        {   
            // generate leaf node
            float value{};
            int16_t type{};
            float r = engine();
            if (r < constProb || varLen == 0)
            {   
                // constant
                value = constSamples[engine(constSamplesLen)];
                type = NodeType::CONST;
            }
            else
            {   
                // variable
                value = static_cast<float>(engine(varLen));
                type = NodeType::VAR;
            }
            node = GPNode{ value, type, 1 };    // subtreesize of a leaf is 1
        }
        gp[topGP++] = node;  // add node in res_gp (safe due to loop guard)
        pendingChildren--;

        // ensure combined children of parent and new child fit before pushing either descriptor
        // int combinedNeed = cd.childs + cdNew.childs;
        // bool haveRoom = (topGP + combinedNeed) < gpLen;

        // add children of first parent and from then on only add children of new child 
        if (chk_first){
            pendingChildren += cd.childs;
            chk_first = false;
        }

        if (cd.childs > 0 && top < MAX_TREE_NODE_NUM) {
            childsAndDepth[top++] = cd;
        }          // push parent back first
        if (cdNew.childs > 0 && top < MAX_TREE_NODE_NUM) {
            childsAndDepth[top++] = cdNew;
            pendingChildren += cdNew.childs;
        }          // then push new child's descriptor
    }

    // Calculate subtree sizes
    int* nodeSize = (int*)childsAndDepth;  // reuse space
    top = 0;
    for (int i = topGP - 1; i >= 0; i--)
    {
        int16_t node_type = gp[i].nodeType;
        node_type &= NodeType::TYPE_MASK;
        if (node_type <= NodeType::CONST)  // VAR or CONST
        {
            nodeSize[top] = 1;
        }
        else if (node_type == NodeType::UFUNC)
        {
            int size1 = nodeSize[--top];
            nodeSize[top] = size1 + 1;
        }
        else if (node_type == NodeType::BFUNC)
        {
            int size1 = nodeSize[--top], size2 = nodeSize[--top];
            nodeSize[top] = size1 + size2 + 1;
        }
        else // if (node_type == NodeType::TFUNC)
        {
            int size1 = nodeSize[--top], size2 = nodeSize[--top], size3 = nodeSize[--top];
            nodeSize[top] = size1 + size2 + size3 + 1;
        }
        gp[i].subtreeSize = (int16_t)nodeSize[top];
        top++;
    }

    // Write result to global memory
    const int len = gp[0].subtreeSize;

    auto o_value = value_res + n * gpLen;
    auto o_type = type_res + n * gpLen;
    auto o_subtree_size = subtree_size_res + n * gpLen;

    for (int i = 0; i < len && i < gpLen && i < MAX_TREE_NODE_NUM; i++)
    {
        o_value[i] = gp[i].value;
        o_type[i] = gp[i].nodeType;
        o_subtree_size[i] = gp[i].subtreeSize;
    }
}

/**
 * @brief Launch the kernel to generate GP trees.
 */
template<bool MultiOutput = false>
inline void generateExecuteKernel(
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const unsigned int constSamplesLen,
    const float outProb,
    const float constProb,
    const unsigned int* keys, 
    const float* depth2leafProbs, 
    const float* rouletteFuncs, 
    const float* constSamples, 
    float* value_res, 
    int16_t* type_res, 
    int16_t* subtree_size_res
)
{   
    int gridSize = 0, blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, treeGPGenerate<MultiOutput>);
    if (gridSize * blockSize < popSize)
    {
        gridSize = (popSize - 1) / blockSize + 1;
    }
    treeGPGenerate<MultiOutput><<<gridSize, blockSize>>>(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb, value_res, type_res, subtree_size_res, keys, depth2leafProbs, rouletteFuncs, constSamples);
}

/**
 * @brief Public function to generate GP trees based on the provided descriptor. Handle data type selection for the GP generation process.
 */
void generate(
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const unsigned int constSamplesLen,
    const float outProb,
    const float constProb,
    const unsigned int* keys, 
    const float* depth2leafProbs, 
    const float* rouletteFuncs, 
    const float* constSamples, 
    float* value_res, 
    int16_t* type_res, 
    int16_t* subtree_size_res
    ){
    if (outLen > 1)
    {
        generateExecuteKernel<true>(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb, keys, depth2leafProbs, rouletteFuncs, constSamples, value_res, type_res, subtree_size_res);
    }
    else
    {
        generateExecuteKernel<false>(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb, keys, depth2leafProbs, rouletteFuncs, constSamples, value_res, type_res, subtree_size_res);
    }
}
