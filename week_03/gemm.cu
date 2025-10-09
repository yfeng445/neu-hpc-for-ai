#include <cuda_runtime.h>
#include <stdio.h>

// C(m,n) = alpha * A(m,k) * B(k,n) + beta * C(m,n)
__global__ void gemmKernel(const float* A, const float* B, float* C, int m, int n, int k, unsigned Ads_sz, unsigned Bds_sz, float alpha = 1, float beta = 1, bool ta = false, bool tb = false) {

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
    // Problem size (small so we can print everything)
    const int M = 4, N = 4, K = 4;
    const float alpha = 1.0f, beta = 1.0f;
    const bool ta = false, tb = false;   // test no-transpose first

    // Host matrices (row-major)
    float hA[M*K], hB[K*N], hC[M*N], hC_ref[M*N];

    // Simple deterministic init
    // A = [[1,2,3,4],
    //      [5,6,7,8],
    //      [9,10,11,12],
    //      [13,14,15,16]]
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            hA[i*K + j] = float(i*K + j + 1);

    // B = [[1,0,0,0],
    //      [0,1,0,0],
    //      [0,0,1,0],
    //      [0,0,0,1]]  (identity) so AB = A
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            hB[i*N + j] = (i == j) ? 1.0f : 0.0f;

    // C all ones (so we can see +beta*C)
    for (int i = 0; i < M*N; ++i) hC[i] = 1.0f;

    // CPU reference: D = alpha * (A or A^T) * (B or B^T) + beta * C
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a = ta ? hA[p*M + i] : hA[i*K + p];
                float b = tb ? hB[j*K + p] : hB[p*N + j];
                acc += a * b;
            }
            hC_ref[i*N + j] = alpha * acc + beta * hC[i*N + j];
        }
    }

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    cudaMalloc(&dA, M*K*sizeof(float));
    cudaMalloc(&dB, K*N*sizeof(float));
    cudaMalloc(&dC, M*N*sizeof(float));

    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, M*N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch config: square block (kernel requires x==y)
    dim3 block(4, 4, 1);                       // 16 threads, fits 4x4 nicely
    dim3 grid((N + block.x - 1)/block.x,
              (M + block.y - 1)/block.y, 1);

    // Shared-memory tiles are block.x * block.x each
    unsigned Ads_sz = block.x * block.x;
    unsigned Bds_sz = block.x * block.x;
    size_t shmem_bytes = (Ads_sz + Bds_sz) * sizeof(float);

    // Launch
    gemmKernel<<<grid, block, shmem_bytes>>>(dA, dB, dC,
        M, N, K, Ads_sz, Bds_sz, alpha, beta, ta, tb);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // Copy back
    float hD[M*N];
    cudaMemcpy(hD, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print GPU result
    printf("D = alpha*A*B + beta*C (M=%d,N=%d,K=%d):\n", M, N, K);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%7.2f ", hD[i*N + j]);
        printf("\n");
    }

    // Compare with reference
    double max_abs_err = 0.0;
    for (int i = 0; i < M*N; ++i)
        max_abs_err = fmax(max_abs_err, std::fabs(double(hD[i] - hC_ref[i])));

    printf("max_abs_err = %.3e\n", max_abs_err);

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
