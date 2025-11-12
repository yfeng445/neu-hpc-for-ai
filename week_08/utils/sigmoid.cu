
#include <cuda_runtime.h>

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

__device__ __forceinline__ float sg(float x){
    if (x >= 0.f){ float z = __expf(-x); return 1.f/(1.f+z); }
    else{ float z = __expf(x); return z/(1.f+z); }
}

__global__ void sigmoid_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) out[i] = sg(in[i]);
}

extern "C" void sigmoid(const float* logits,
                        float* p,
                        int T, int E,
                        cudaStream_t stream){
    int n = T * E;
    int threads = 256;
    int blocks = CEIL_DIV(n, threads * 4);
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(logits, p, n);
}
