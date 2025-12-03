// router/group_top2_select.cu
#include <cuda_runtime.h>

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

__global__ void group_top2_select_kernel(
    const float* __restrict__ x,
    unsigned char* __restrict__ mask,
    int T, int E, int G, int K)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= T) return;
    int Eg = E / G;
    unsigned char* rowm = mask + (size_t)t * G;
    for (int g=0; g<G; ++g) rowm[g] = 0;

    for (int sel=0; sel<K; ++sel) {
        float best = -1e30f;
        int   bestg = -1;
        for (int g=0; g<G; ++g) {
            if (rowm[g]) continue;
            int off = t * E + g * Eg;
            float m1 = -1e30f, m2 = -1e30f;
            for (int i=0; i<Eg; ++i) {
                float v = x[off + i];
                if (v > m1) { m2 = m1; m1 = v; }
                else if (v > m2) { m2 = v; }
            }
            float s = m1 + m2;
            if (s > best) { best = s; bestg = g; }
        }
        if (bestg < 0) break;
        rowm[bestg] = 1;
    }
}

extern "C" void group_top2_select(
    const float* logits_choice,
    unsigned char* group_mask,
    int T, int E, int n_group, int topk_group,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = CEIL_DIV(T, threads);
    group_top2_select_kernel<<<blocks, threads, 0, stream>>>(
        logits_choice, group_mask, T, E, n_group, topk_group);
}
