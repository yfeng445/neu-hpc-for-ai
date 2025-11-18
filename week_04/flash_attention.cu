#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <algorithm>

#define CHECK_CUDA(call) do {                          \
    cudaError_t err__ = (call);                        \
    if (err__ != cudaSuccess) {                        \
        std::cerr << "CUDA error: "                    \
                  << cudaGetErrorString(err__)         \
                  << " at " << __FILE__ << ":"         \
                  << __LINE__ << std::endl;            \
        std::exit(1);                                  \
    }                                                  \
} while (0)

// 单头 FlashAttention 核函数（流式 softmax，保持数值稳定）
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d,
    int M      // SRAM 容量（以 float 个数计）
) {
    extern __shared__ float smem[];

    int Bc = (M + 4*d - 1) / (4*d);
    if (Bc < 1) Bc = 1;
    int Br = (Bc < d) ? Bc : d;

    // shared memory 布局
    float* Qs     = smem;                    // [Br, d]
    float* Ks     = Qs     + Br * d;         // [Bc, d]
    float* Vs     = Ks     + Bc * d;         // [Bc, d]（论文里 V 也是 d 维）
    float* m_row  = Vs     + Bc * d;         // [Br]
    float* l_row  = m_row  + Br;             // [Br]
    float* p_row  = l_row  + Br;             // [Br, d]
    float* m_tile = p_row  + Br * d;         // [Br]
    float* l_tile = m_tile + Br;             // [Br]
    float* p_tile = l_tile + Br;             // [Br, d]

    // 每个 block 负责 Br 行 query
    int row_block = blockIdx.x;
    int row_start = row_block * Br;
    if (row_start >= N) return;
    int Br_eff = (row_start + Br <= N) ? Br : (N - row_start);

    // 载入 Q tile
    for (int r = 0; r < Br_eff; ++r) {
        for (int c = 0; c < d; ++c) {
            Qs[r * d + c] = Q[(row_start + r) * d + c];
        }
    }

    // 初始化行级累积状态 m_row, l_row, p_row
    for (int r = 0; r < Br_eff; ++r) {
        m_row[r] = -CUDART_INF_F;
        l_row[r] = 0.0f;
    }
    for (int r = 0; r < Br_eff; ++r) {
        for (int v = 0; v < d; ++v) {
            p_row[r * d + v] = 0.0f;
        }
    }

    // 按列块遍历 K, V
    for (int col_start = 0; col_start < N; col_start += Bc) {
        int Bc_eff = (col_start + Bc <= N) ? Bc : (N - col_start);

        // 载入当前 K, V tile
        for (int c = 0; c < Bc_eff; ++c) {
            for (int k = 0; k < d; ++k) {
                Ks[c * d + k] = K[(col_start + c) * d + k];
                Vs[c * d + k] = V[(col_start + c) * d + k];
            }
        }

        // 对当前 tile，逐行计算局部贡献
        for (int r = 0; r < Br_eff; ++r) {
            const float* q_row = &Qs[r * d];

            // 第一次遍历：找到该行在此 tile 内的局部最大值 m_tile[r]
            float mloc = -CUDART_INF_F;
            for (int c = 0; c < Bc_eff; ++c) {
                const float* k_row = &Ks[c * d];
                float s = 0.0f;
                for (int k = 0; k < d; ++k) {
                    s += q_row[k] * k_row[k];
                }
                if (s > mloc) mloc = s;
            }
            m_tile[r] = mloc;

            // 第二次遍历：计算局部未归一化 softmax 和对应的加权和
            float lloc = 0.0f;
            for (int v = 0; v < d; ++v) {
                p_tile[r * d + v] = 0.0f;
            }

            for (int c = 0; c < Bc_eff; ++c) {
                const float* k_row = &Ks[c * d];
                const float* v_row = &Vs[c * d];

                float s = 0.0f;
                for (int k = 0; k < d; ++k) {
                    s += q_row[k] * k_row[k];
                }
                float e = expf(s - m_tile[r]);  // P̃ = exp(s - m̃)
                lloc += e;
                for (int v = 0; v < d; ++v) {
                    p_tile[r * d + v] += e * v_row[v];
                }
            }
            l_tile[r] = lloc;

            // 将当前 tile 的统计量与之前累积的 (m_row, l_row, p_row) 合并
            float m_old = m_row[r];
            float l_old = l_row[r];

            float m_new;
            if (m_old == -CUDART_INF_F) {
                m_new = m_tile[r];
            } else {
                m_new = fmaxf(m_old, m_tile[r]);
            }

            float alpha = (m_old == -CUDART_INF_F) ? 0.0f : expf(m_old - m_new);
            float beta  = (l_tile[r] == 0.0f)      ? 0.0f : expf(m_tile[r] - m_new);

            float l_new = alpha * l_old + beta * l_tile[r];

            for (int v = 0; v < d; ++v) {
                float p_old_scaled  = alpha * p_row[r * d + v];
                float p_tile_scaled = beta  * p_tile[r * d + v];
                p_row[r * d + v] = p_old_scaled + p_tile_scaled;
            }

            m_row[r] = m_new;
            l_row[r] = l_new;
        }
    }

    // 最终归一化，得到输出 O
    for (int r = 0; r < Br_eff; ++r) {
        float inv_l = 1.0f / (l_row[r] + 1e-9f);
        for (int v = 0; v < d; ++v) {
            O[(row_start + r) * d + v] = p_row[r * d + v] * inv_l;
        }
    }
}


int main() {
    const int N = 32;
    const int d = 32;
    const int size = N * d;
    const int M = 512; 

    float* h_Q = new float[size];
    float* h_K = new float[size];
    float* h_V = new float[size];
    float* h_O = new float[size];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            int idx = i * d + j;
            float x = static_cast<float>(idx);

            h_Q[idx] = std::sinf(0.01f * x);           
            h_K[idx] = std::cosf(0.02f * x);          
            h_V[idx] = std::sinf(0.03f * x + 0.5f);  
        }
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, size * sizeof(float), cudaMemcpyHostToDevice));

    int Bc = (M + 4 * d - 1) / (4 * d);
    if (Bc < 1) Bc = 1;
    int Br = (Bc < d) ? Bc : d;

    int num_row_blocks = (N + Br - 1) / Br;

    dim3 grid(num_row_blocks);
    dim3 block(1); 

    size_t smem_floats = static_cast<size_t>(d) * (3 * Br + 2 * Bc)
                       + static_cast<size_t>(4 * Br);
    size_t smem_bytes = smem_floats * sizeof(float);

    flash_attention_kernel<<<grid, block, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, N, d, M
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_O, d_O, size * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("Output O (N=32, d=32):\n");
    for (int i = 0; i < N; ++i) {
        std::printf("Row %d:", i);
        for (int j = 0; j < d; ++j) {
            std::printf(" %.3f", h_O[i * d + j]);
        }
        std::printf("\n");
    }

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;

    return 0;
}
