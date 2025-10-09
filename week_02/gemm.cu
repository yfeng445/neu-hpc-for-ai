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
    // Matrix sizes
    const int m = 2, n = 2, k = 2;
    const float alpha = 1.0f, beta = 1.0f;

    // Hard-coded matrices (row-major)
    float hA[m*k] = {1, 2,
                     3, 4};  // 2x2
    float hB[k*n] = {5, 6,
                     7, 8};  // 2x2
    float hC[m*n] = {1, 1,
                     1, 1};  // 2x2
    float hD[m*n] = {0};     // output

    // Device memory
    float *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC));
    cudaMalloc(&dD, sizeof(hD));

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(hC), cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 block(2, 2);
    dim3 grid(1, 1);

    // Kernel call
    gemmKernel<<<grid, block>>>(dA, dB, dC, dD, m, n, k, alpha, beta);
    cudaDeviceSynchronize();

    cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);

    // Print result
    printf("[host] Result D = alpha*A*B + beta*C:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%6.2f ", hD[i * n + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
    return 0;
}

