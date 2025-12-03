// experts/experts_mlp_forward.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void gemm(const half* x, const half* W, float* logits,
                     int T, int D, int E, cudaStream_t stream);
extern "C" void bias_add(float* inout, const float* bias,
                         int T, int E, cudaStream_t stream);

__device__ __forceinline__ float sgm_(float x){
    if (x >= 0.f){ float z = __expf(-x); return 1.f/(1.f+z); }
    else { float z = __expf(x); return z/(1.f+z); }
}
__device__ __forceinline__ float silu_(float x){ return x * sgm_(x); }
__device__ __forceinline__ float gelu_(float x){
    const float k0 = 0.7978845608f, k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x3)));
}

__global__ void act_kernel_exp(float* x, int n, int act){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        float v = x[i];
        x[i] = (act==0) ? silu_(v) : gelu_(v);
    }
}

__global__ void f2h_kernel_exp(const float* __restrict__ in,
                               half* __restrict__ out, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) out[i] = __float2half(in[i]);
}

extern "C" void experts_mlp_forward(
    const half* inbuf,
    half*       outbuf,
    int D, int H,
    int n_local,
    const int*  local_offsets,     // host ptr, length n_local+1
    const half* const* W1_local,   // host array of device ptrs
    const float* const* B1_local,  // host array of device ptrs
    const half* const* W2_local,   // host array of device ptrs
    const float* const* B2_local,  // host array of device ptrs
    int activation,                // 0:SiLU, 1:GELU
    cudaStream_t stream)
{
    const int threads = 256;

    for (int i = 0; i < n_local; ++i){
        int row0 = local_offsets[i];
        int row1 = local_offsets[i+1];
        int N = row1 - row0;
        if (N <= 0) continue;

        const half* X_i  = inbuf  + (size_t)row0 * D;
        half*       Y_i  = outbuf + (size_t)row0 * D;

        float *H1_f = nullptr, *O2_f = nullptr;
        half  *H1_h = nullptr;

        if (cudaSuccess != cudaMalloc(&H1_f, (size_t)N * H * sizeof(float))) return;
        if (cudaSuccess != cudaMalloc(&H1_h, (size_t)N * H * sizeof(half)))  { cudaFree(H1_f); return; }
        if (cudaSuccess != cudaMalloc(&O2_f, (size_t)N * D * sizeof(float))) { cudaFree(H1_f); cudaFree(H1_h); return; }

        gemm(X_i, W1_local[i], H1_f, N, D, H, stream);
        bias_add(H1_f, B1_local[i], N, H, stream);

        int nH = N * H;
        int nD = N * D;

        if (nH > 0) {
            int blocksH = (nH + threads - 1) / threads;
            act_kernel_exp<<<blocksH, threads, 0, stream>>>(H1_f, nH, activation);
            f2h_kernel_exp<<<blocksH, threads, 0, stream>>>(H1_f, H1_h, nH);
        }

        gemm(H1_h, W2_local[i], O2_f, N, H, D, stream);
        bias_add(O2_f, B2_local[i], N, D, stream);

        if (nD > 0) {
            int blocksD = (nD + threads - 1) / threads;
            f2h_kernel_exp<<<blocksD, threads, 0, stream>>>(O2_f, Y_i, nD);
        }

        cudaFree(H1_f);
        cudaFree(H1_h);
        cudaFree(O2_f);
    }
}
