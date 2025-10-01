#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)


__device__ inline float dot_parallel(const float* __restrict__ qrow, const float* __restrict__ krow, int d, float* __restrict__ red){
    int tid = threadIdx.x;
    float part = 0.f;
    for (int t = tid; t < d; t += blockDim.x) {
        part += qrow[t] * krow[t];
    }
    red[tid] = part;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// O[N,dv] = softmax( Q[N,d] K[N,d]^T / sqrt(d) ) V[N,dv]
// Tiling over N: Br rows of Q per block; loop over Bc rows of K/V per tile.
__global__ void flashAttentionKernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O, int N, int d, int dv, float scale, int Br, int Bc){
    extern __shared__ float smem[];

    float* Qds    = smem;                          
    float* Kds    = Qds + (size_t)Br * d;         
    float* Vds    = Kds + (size_t)Bc * d;          

    float* m_row  = Vds + (size_t)Bc * dv;        
    float* l_row  = m_row + Br;                    
    float* p_row  = l_row + Br;                   

    float* m_tile = p_row + (size_t)Br * dv;       
    float* l_tile = m_tile + Br;                   
    float* p_tile = l_tile + Br;                  

    float* red    = p_tile + (size_t)Br * dv;     

    // parallel over q
    int row_start = blockIdx.y * Br;
    int row_end   = min((blockIdx.y + 1) * Br, N);
    int Br_eff    = row_end - row_start;

    // load q to smem in parallel
    for (int r = 0; r < Br_eff; ++r) {
        for (int c = threadIdx.x; c < d; c += blockDim.x) {
            Qds[r * d + c] = Q[(row_start + r) * d + c];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int r = 0; r < Br_eff; ++r) {
            m_row[r] = -CUDART_INF_F;
            l_row[r] = 0.f;
        }
        for (int r = 0; r < Br_eff; ++r)
            for (int v = 0; v < dv; ++v)
                p_row[r * dv + v] = 0.f;
    }
    __syncthreads();

    for (int col = 0; col < N; col += Bc) {
        int col_start = col;
        int col_end   = min(col + Bc, N);
        int Bc_eff    = col_end - col_start;

        // same trick for kv
        for (int r = 0; r < Bc_eff; ++r) {
            for (int c = threadIdx.x; c < d;  c += blockDim.x)
                Kds[r * d  + c]  = K[(col_start + r) * d  + c];
            for (int v = threadIdx.x; v < dv; v += blockDim.x)
                Vds[r * dv + v]  = V[(col_start + r) * dv + v];
        }
        __syncthreads();

        for (int r = 0; r < Br_eff; ++r) {
            const float* qrow = &Qds[r * d];
            float maxv = -CUDART_INF_F;

            for (int c = 0; c < Bc_eff; ++c) {
                const float* krow = &Kds[c * d];
                float s = dot_parallel(qrow, krow, d, red) * scale;
                if (threadIdx.x == 0) {
                    if (s > maxv) maxv = s;
                }
                __syncthreads(); 
            }
            if (threadIdx.x == 0) m_tile[r] = maxv;
            __syncthreads();
        }


        for (int r = 0; r < Br_eff; ++r) {
            const float* qrow = &Qds[r * d];

            for (int v = threadIdx.x; v < dv; v += blockDim.x) {
                p_tile[r * dv + v] = 0.f;
            }
            __syncthreads();

            float lsum = 0.f;
            for (int c = 0; c < Bc_eff; ++c) {
                const float* krow = &Kds[c * d];
                const float* vrow = &Vds[c * dv];

                float s = dot_parallel(qrow, krow, d, red) * scale;

                if (threadIdx.x == 0) {
                    float e = expf(s - m_tile[r]); // safe softmax
                    red[0] = e;
                    lsum  += e;
                }
                __syncthreads();
                float e = red[0];

                for (int v = threadIdx.x; v < dv; v += blockDim.x) {
                    p_tile[r * dv + v] += e * vrow[v];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) l_tile[r] = lsum;
            __syncthreads();
        }

        for (int r = 0; r < Br_eff; ++r) {
            if (threadIdx.x == 0) {
                float mnew  = fmaxf(m_row[r], m_tile[r]);
                float alpha = expf(m_row[r]  - mnew);
                float beta  = expf(m_tile[r] - mnew);
                m_row[r] = mnew;
                l_row[r] = alpha * l_row[r] + beta * l_tile[r];
                red[0] = alpha;
                red[1] = beta;
            }
            __syncthreads();
            float alpha = red[0];
            float beta  = red[1];

            for (int v = threadIdx.x; v < dv; v += blockDim.x) {
                p_row[r * dv + v] = alpha * p_row[r * dv + v] + beta  * p_tile[r * dv + v]; // cj​=cj−1​⋅exp(m−max(xj​,m))+exp(xj​−max(xj​,m)).
            }
            __syncthreads();
        }
    }

    // write back to O
    for (int r = 0; r < Br_eff; ++r) {
        float l = l_row[r];
        int grow = row_start + r;
        for (int v = threadIdx.x; v < dv; v += blockDim.x) {
            O[grow * dv + v] = p_row[r * dv + v] / l;
        }
    }
}

// ------------------------------
// Minimal test driver
// ------------------------------
int main() {
    // Parameters
    int N = 4;     // sequence length
    int d = 2;     // head_dim
    int dv = 2;    // value_dim
    int Br = 2;    // Q-tile rows per block
    int Bc = 2;    // K/V-tile rows
    float scale = 1.0f / std::sqrt((float)d);

    size_t sizeQ = (size_t)N * d  * sizeof(float);
    size_t sizeK = (size_t)N * d  * sizeof(float);
    size_t sizeV = (size_t)N * dv * sizeof(float);
    size_t sizeO = (size_t)N * dv * sizeof(float);

    // Host buffers
    std::vector<float> hQ(N*d), hK(N*d), hV(N*dv), hO(N*dv);

    // Simple deterministic init
    for (int i = 0; i < N*d; i++) {
        hQ[i] = (float)(i+1) * 0.1f;
        hK[i] = (float)(i+1) * 0.2f;
    }
    for (int i = 0; i < N*dv; i++) {
        hV[i] = (float)(i+1) * 0.3f;
    }

    // Device buffers
    float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
    CHECK_CUDA(cudaMalloc(&dQ, sizeQ));
    CHECK_CUDA(cudaMalloc(&dK, sizeK));
    CHECK_CUDA(cudaMalloc(&dV, sizeV));
    CHECK_CUDA(cudaMalloc(&dO, sizeO));

    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), sizeQ, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), sizeK, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), sizeV, cudaMemcpyHostToDevice));

    // Launch config: each block handles Br rows; use a small power-of-two block size
    dim3 grid(1, (N + Br - 1) / Br);
    dim3 block(32);  // can be 64/128 if d is large

    // Compute shared memory size (include +block.x for reduction buffer)
    size_t shmem_bytes =
        ( (size_t)Br*d + (size_t)Bc*d + (size_t)Bc*dv   // Qds, Kds, Vds
        + Br + Br + (size_t)Br*dv                       // m_row, l_row, p_row
        + Br + Br + (size_t)Br*dv                       // m_tile, l_tile, p_tile
        + block.x                                       // red buffer
        ) * sizeof(float);

    // Kernel
    flashAttentionKernel<<<grid, block, shmem_bytes>>>(
        dQ, dK, dV, dO, N, d, dv, scale, Br, Bc
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, sizeO, cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Output O:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "Row " << i << ": ";
        for (int v = 0; v < dv; v++) {
            std::cout << hO[i*dv + v] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    return 0;
}
