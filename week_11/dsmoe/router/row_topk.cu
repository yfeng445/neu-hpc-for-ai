// router/row_topk.cu
#include <cuda_runtime.h>

__global__ void row_topk_kernel(const float* __restrict__ in,
                                int* __restrict__ idx,
                                int T, int E, int K)
{
    int t = blockIdx.x;
    if (t >= T) return;
    extern __shared__ float srow[];
    for (int c = threadIdx.x; c < E; c += blockDim.x)
        srow[c] = in[(size_t)t * E + c];
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            int   arg = -1;
            float val = -1e30f;
            for (int c = 0; c < E; ++c) {
                float v = srow[c];
                if (v > val) { val = v; arg = c; }
            }
            if (arg < 0) { idx[(size_t)t * K + k] = 0; continue; }
            idx[(size_t)t * K + k] = arg;
            srow[arg] = -1e30f;
        }
    }
}

extern "C" void row_topk(const float* in,
                         int* idx,
                         int T, int E, int K,
                         cudaStream_t stream)
{
    dim3 grid(T);
    dim3 block(1);
    size_t smem = (size_t)E * sizeof(float);
    row_topk_kernel<<<grid, block, smem, stream>>>(in, idx, T, E, K);
}
