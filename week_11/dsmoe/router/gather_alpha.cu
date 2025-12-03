// router/gather_alpha.cu
#include <cuda_runtime.h>

__global__ void gather_alpha_kernel(const float* __restrict__ p,
                                    const int* __restrict__ idx,
                                    float* __restrict__ alpha,
                                    int T, int E, int K,
                                    int norm, float scale)
{
    int t = blockIdx.x;
    if (t >= T) return;
    float s = 0.f;
    for (int k = 0; k < K; ++k) {
        int j = idx[(size_t)t * K + k];
        float v = (j >= 0 && j < E) ? p[(size_t)t * E + j] : 0.f;
        alpha[(size_t)t * K + k] = v;
        s += v;
    }
    if (norm) {
        float d = s + 1e-20f;
        for (int k = 0; k < K; ++k)
            alpha[(size_t)t * K + k] = (alpha[(size_t)t * K + k] / d) * scale;
    } else {
        if (scale != 1.f)
            for (int k = 0; k < K; ++k)
                alpha[(size_t)t * K + k] *= scale;
    }
}

extern "C" void gather_alpha(const float* p,
                             const int* topk_idx,
                             float* alpha,
                             int T, int E, int K,
                             int norm_topk_prob,
                             float routed_scale,
                             cudaStream_t stream)
{
    dim3 grid(T);
    dim3 block(1);
    gather_alpha_kernel<<<grid, block, 0, stream>>>(
        p, topk_idx, alpha, T, E, K, norm_topk_prob, routed_scale);
}
