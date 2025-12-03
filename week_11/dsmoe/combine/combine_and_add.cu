// combine/combine_and_add.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

__global__ void zero_half_kernel(half* x, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) x[i] = __float2half(0.f);
}

__global__ void combine_kernel(const half* __restrict__ contrib,
                               const int*  __restrict__ combine_idx,
                               int rows, int D,
                               half* __restrict__ y_re)
{
    int r = blockIdx.x;
    if (r >= rows) return;
    int t = combine_idx[r];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        half v = contrib[(size_t)r*D + d];
        atomicAdd(&y_re[(size_t)t*D + d], v);
    }
}

__global__ void add_kernel(const half* __restrict__ a,
                           const half* __restrict__ b,
                           half* __restrict__ out,
                           int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        float va = __half2float(a[i]);
        float vb = __half2float(b[i]);
        out[i] = __float2half(va + vb);
    }
}

extern "C" void combine_and_add(
    const half* contrib,
    const int*  combine_idx,
    int rows,
    half*       y_re,
    const half* y_se,
    half*       y,
    int T, int D,
    cudaStream_t stream)
{
    int nTD = T * D;
    int threads = 256;
    int blocksZ = CEIL_DIV(nTD, threads * 4);
    zero_half_kernel<<<blocksZ, threads, 0, stream>>>(y_re, nTD);

    dim3 gridC(rows);
    dim3 blockC(128);
    combine_kernel<<<gridC, blockC, 0, stream>>>(contrib, combine_idx, rows, D, y_re);

    int blocksA = CEIL_DIV(nTD, threads * 4);
    add_kernel<<<blocksA, threads, 0, stream>>>(y_re, y_se, y, nTD);
}
