#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gemmKernel(const float* A, const float* B, const float* C, float* D,
                           int m, int n, int k, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int p = 0; p < k; ++p) {
        sum += A[row * k + p] * B[p * n + col];
    }
    D[row * n + col] = alpha * sum + beta * C[row * n + col];
}

int main() {
    int m = 256, n = 256, k = 256;
    dim3 block(32, 32, 1);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    float *dA = NULL, *dB = NULL, *dC = NULL, *dD = NULL;
    float alpha = 1.0f, beta = 1.0f;

    gemmKernel<<<grid, block>>>(dA, dB, dC, dD, m, n, k, alpha, beta);
    
    return 0;
}
