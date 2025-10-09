# include <stdio.h>
# include <cuda_runtime.h>
# include <math_constants.h> 

__global__ void online_normalizer_softmax(float* X, float* Y, int rows, int cols){
    int row = blockIdx.x;
    if (row >= rows) return;

    float m_local = -CUDART_INF_F;  
    float d_local = 0.0f;

    const int base = row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float x = X[base + j];
        float m_new = fmaxf(m_local, x);                         
        d_local = d_local * __expf(m_local - m_new) + __expf(x - m_new);                           
        m_local = m_new;
    }

    extern __shared__ float smem[];
    float* smax = smem;                       
    float* ssum = smem + blockDim.x;         

    smax[threadIdx.x] = m_local;
    ssum[threadIdx.x] = d_local;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            float ma = smax[threadIdx.x];
            float da = ssum[threadIdx.x];
            float mb = smax[threadIdx.x + s];
            float db = ssum[threadIdx.x + s];
            float m   = fmaxf(ma, mb);                           
            float d   = da * __expf(ma - m) + db * __expf(mb - m);

            smax[threadIdx.x] = m;
            ssum[threadIdx.x] = d;
        }
        __syncthreads();
    }

    float mV = smax[0];
    float dV = ssum[0];
    if (threadIdx.x == 0) {
        if (!(dV > 0.0f)) dV = 1.0f; 
        smax[0] = mV;  ssum[0] = dV;
    }
    __syncthreads();
    mV = smax[0];  dV = ssum[0];

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float x = X[base + j];
        Y[base + j] = __expf(x - mV) / dV;
    }
}

int main() {
    // ---- Hard-coded test input (2 rows × 5 cols) ----
    const int rows = 2, cols = 5;
    float hX[rows * cols] = {
        1.0f,  2.0f,  0.0f, -1.0f,  3.0f,   // row 0
       -2.0f,  0.0f,  2.0f,  4.0f,  1.0f    // row 1
    };
    float hY[rows * cols] = {0};

    // ---- Device buffers ----
    float *dX = nullptr, *dY = nullptr;
    cudaMalloc(&dX, sizeof(hX));
    cudaMalloc(&dY, sizeof(hY));

    cudaMemcpy(dX, hX, sizeof(hX), cudaMemcpyHostToDevice);

    // ---- Launch config ----
    // One block per row; choose a power-of-two block size for the reduction.
    // For cols=5, next power of two is 8. (You could also just use 32.)
    const int bx = 8;                   // or 32/64/etc. if you prefer
    dim3 grid(rows);
    dim3 block(bx);

    // Dynamic shared memory: two float arrays of length block.x (smax + ssum)
    size_t shmem_bytes = 2 * block.x * sizeof(float);

    // ---- Kernel launch ----
    online_normalizer_softmax<<<grid, block, shmem_bytes>>>(dX, dY, rows, cols);

    // (Optional but recommended) check and sync
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // ---- Copy back & print ----
    cudaMemcpy(hY, dY, sizeof(hY), cudaMemcpyDeviceToHost);

    printf("Softmax per row (rows=%d, cols=%d):\n", rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%8.5f ", hY[r * cols + c]);
        }
        printf("\n");
    }

    // ---- Cleanup ----
    cudaFree(dX);
    cudaFree(dY);
    return 0;
}


