#include <cuda_runtime.h>
#include <stdio.h>

// C(m,n) = alpha * A(m,k) * B(k,n) + beta * C(m,n)
__global__ void gemmKernel(const float* A, const float* B, float* C, int m, int n, int k, float alpha = 1, float beta = 1, bool ta = false, bool tb = false, unsigned Ads_sz, unsigned Bds_sz) {

    extern __shared__ float Ads_Bds[]; 

    float* Ads = (float* ) Ads_Bds;
    float* Bds = (float* ) Ads_Bds + Ads_sz;

    if(blockDim.x != blockDim.y){printf("WARNING: blockDim.x must be equal to blockDim.y\n"); return;};

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;     

    float Cvalue = 0;
    for(int ch = 0; ch<( (k + blockDim.x -1) / blockDim.x); ++ch){
        if(row < m && ch * blockDim.x + threadIdx.x < k)
            Ads[threadIdx.y * blockDim.x + threadIdx.x] = A[ta? (ch * blockDim.x + threadIdx.x) * m + row : row * k + ch * blockDim.x + threadIdx.x];
        else
            Ads[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;

        if(col < n && ch * blockDim.y + threadIdx.y < k)
            Bds[threadIdx.y * blockDim.x + threadIdx.x] = B[tb? col * k + ch * blockDim.y + threadIdx.y : (ch * blockDim.y + threadIdx.y) * n + col];
        else
            Bds[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;

        __syncthreads();

        for(int i = 0; i < blockDim.x; ++i)
            Cvalue += Ads[threadIdx.y * blockDim.x + i] * Bds[i * blockDim.x + threadIdx.x];

        __syncthreads();
    }
    if(row < m && col < n) C[row * n + col] = alpha * Cvalue + beta * C[row * n + col];
}


int main() {
    int m = 256, n = 256, k = 256;
    dim3 block(32, 32, 1);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    float *dA = NULL, *dB = NULL, *dC = NULL;
    float alpha = 1.0f, beta = 1.0f;
    unsigned Ads_sz = block.x * block.y;
    unsigned Bds_sz = block.x * block.y;
    size_t sharedMemSize = (Ads_sz + Bds_sz) * sizeof(float);

    gemmKernel<<<grid, block, sharedMemSize>>>(dA, dB, dC, m, n, k, alpha, beta, false, false, Ads_sz, Bds_sz);
    
    return 0;
}
