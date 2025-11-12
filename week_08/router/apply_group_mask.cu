// router/apply_group_mask.cu
#include <cuda_runtime.h>
#include <stdint.h>

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

__global__ void apply_group_mask_kernel(
    float* __restrict__ x,
    const unsigned char* __restrict__ mask,
    int T, int E, int G)
{
    size_t N = (size_t)T * E;
    int Eg = E / G;
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += stride) {
        int t = i / E;
        int c = i - (size_t)t * E;
        int g = c / Eg;
        if (mask[(size_t)t * G + g] == 0) x[i] = 0.0f;
    }
}

extern "C" void apply_group_mask(
    float* logits_choice,
    const unsigned char* group_mask,
    int T, int E, int n_group,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = CEIL_DIV((int)((size_t)T*E), threads*4);
    apply_group_mask_kernel<<<blocks, threads, 0, stream>>>(
        logits_choice, group_mask, T, E, n_group);
}
