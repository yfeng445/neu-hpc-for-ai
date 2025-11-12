
#include <cuda_runtime.h>

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

__global__ void bias_add_kernel(float* __restrict__ inout,
                                const float* __restrict__ bias,
                                int T, int E){
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < T && c < E) inout[r*E + c] += bias[c];
}

extern "C" void bias_add(float* inout,
                         const float* bias,
                         int T, int E,
                         cudaStream_t stream){
    dim3 block(32, 8);
    dim3 grid(CEIL_DIV(E, block.x), CEIL_DIV(T, block.y));
    bias_add_kernel<<<grid, block, 0, stream>>>(inout, bias, T, E);
}
