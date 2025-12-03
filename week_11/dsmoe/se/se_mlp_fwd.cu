// se/se_mlp_forward.cu
/*
nvcc -O3 -std=c++17 -arch=sm_86 utils\gemm.cu utils\sigmoid.cu utils\bias_add.cu router\group_top2_select.cu router\apply_group_mask.cu router\row_topk.cu router\gather_alpha.cu router\router_fwd.cpp dispatch\pack_routes.cu comm\ep_alltoallv_stub.cc experts\experts_mlp_fwd.cu combine\combine_and_add.cu se\se_mlp_fwd.cu moe_fwd.cpp main_moe.cpp -o test_moe.exe
*/
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void gemm(const half* x, const half* W, float* logits,
                     int T, int D, int E, cudaStream_t stream);
extern "C" void bias_add(float* inout, const float* bias,
                         int T, int E, cudaStream_t stream);

__device__ __forceinline__ float sg_(float x){
    if (x >= 0.f){ float z = __expf(-x); return 1.f/(1.f+z); }
    else { float z = __expf(x); return z/(1.f+z); }
}
__device__ __forceinline__ float silu_(float x){ return x * sg_(x); }
__device__ __forceinline__ float gelu_(float x){
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x3)));
}

__global__ void act_kernel_se(float* x, int n, int act){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        float v = x[i];
        x[i] = (act==0) ? silu_(v) : gelu_(v);
    }
}

__global__ void f2h_kernel_se(const float* __restrict__ in, half* __restrict__ out, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) out[i] = __float2half(in[i]);
}

extern "C" void se_mlp_forward(
    const half* x,
    half*       y_se,
    const half* W1_se, const float* B1_se,
    const half* W2_se, const float* B2_se,
    int T, int D, int H_se,
    int activation,
    cudaStream_t stream)
{
    const int threads = 256;
    const int nH = T * H_se;
    const int nD = T * D;

    float* H1_f = nullptr;
    half*  H1_h = nullptr;
    float* O2_f = nullptr;
    cudaMallocAsync(&H1_f, (size_t)nH * sizeof(float), stream);
    cudaMallocAsync(&H1_h, (size_t)nH * sizeof(half),  stream);
    cudaMallocAsync(&O2_f, (size_t)nD * sizeof(float), stream);

    gemm(x, W1_se, H1_f, T, D, H_se, stream);
    bias_add(H1_f, B1_se, T, H_se, stream);

    int blocksH = (nH + threads*4 - 1) / (threads*4);
    int blocksD = (nD + threads*4 - 1) / (threads*4);
    act_kernel_se<<<blocksH, threads, 0, stream>>>(H1_f, nH, activation);
    f2h_kernel_se<<<blocksH, threads, 0, stream>>>(H1_f, H1_h, nH);

    gemm(H1_h, W2_se, O2_f, T, H_se, D, stream);
    bias_add(O2_f, B2_se, T, D, stream);
    f2h_kernel_se<<<blocksD, threads, 0, stream>>>(O2_f, y_se, nD);

    cudaFreeAsync(H1_f, stream);
    cudaFreeAsync(H1_h, stream);
    cudaFreeAsync(O2_f, stream);
}
