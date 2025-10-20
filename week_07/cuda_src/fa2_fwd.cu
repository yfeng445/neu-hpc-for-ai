#ifndef FA2_FWD_CU_INCLUDED
#define FA2_FWD_CU_INCLUDED

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <algorithm>

using namespace nvcuda;

__host__ __device__ __forceinline__
int align16(int x) { return (x + 15) & ~15; }

// 动态共享内存需求（按 warp 数量放大）：
// 每 warp 需要：Ash(16x16 half) + Bsh(16x16 half col-major) + Ssh(16x16 float)
extern "C"
size_t fa2_forward_smem_bytes(int /*Br*/, int /*Bc*/, int /*d*/, int /*dv*/, int block_x){
    const size_t per_warp_bytes =
          16*16*sizeof(__half)        // Ash
        + 16*16*sizeof(__half)        // Bsh (col-major)
        + 16*16*sizeof(float);        // Ssh (row-major)
    int warps = std::max(1, block_x / 32);
    return per_warp_bytes * warps;
}

// --------------------------- FP16+WMMA + 多 warp/块 ---------------------------
// Q:[Nq_local,d], K:[Nk_full,d], V:[Nk_full,dv], O:[Nq_local,dv]
// 要求：d 为 16 的倍数（若更通用，可在 k0 维继续分块）
__global__ void
flashAttn2_wmma_kernel(
    const float* __restrict__ Q_f32,   // host 侧传 FP32，内核装入 smem 时转 half
    const float* __restrict__ K_f32,
    const float* __restrict__ V_f32,
    float* __restrict__ O_f32,
    int Nq_local, int Nk_full,
    int d, int dv, float scale, int Br, int /*Bc*/
){
    // ---- warp 切分 ----
    const int warp_id = threadIdx.x >> 5;   // 0..R-1
    const int lane    = threadIdx.x & 31;   // 0..31
    const int R       = max(1, blockDim.x / 32);

    // ---- 行 tile 归属：每个 warp 负责 16 行 ----
    const int rows_per_warp = 16;
    const int row_tile_start = blockIdx.y * (rows_per_warp * R) + warp_id * rows_per_warp;
    const int row_tile_end   = min(Nq_local, row_tile_start + rows_per_warp);
    const int valid_rows     = max(0, row_tile_end - row_tile_start);
    if (valid_rows <= 0) return;

    // ---- 动态共享内存（按 warp 分区）----
    extern __shared__ unsigned char gmem[];
    const size_t per_warp_bytes =
          16*16*sizeof(__half)        // Ash
        + 16*16*sizeof(__half)        // Bsh (col-major)
        + 16*16*sizeof(float);        // Ssh
    unsigned char* base = gmem + warp_id * per_warp_bytes;
    __half*  Ash = reinterpret_cast<__half*>(base);                 // [16,16] row-major
    __half*  Bsh = Ash + 16*16;                                     // [16,16] col-major
    float*   Ssh = reinterpret_cast<float*>(Bsh + 16*16);           // [16,16] row-major

    // ---- 行级在线 softmax 状态（每 lane 负责一行）----
    float m = -1e30f;   // 行最大
    float l = 0.f;      // 行的 exp 和

    // ---- 沿 K/V 的行方向（序列维）以 16 为步长做列 tile ----
    for (int j0 = 0; j0 < Nk_full; j0 += 16) {
        const int cols_this = min(16, Nk_full - j0);

        // ==== 用 WMMA 计算 S(16x16) = Q(16xd) * K(j0..j0+15)^T ====
        // 先把累加器清零
        wmma::fragment<wmma::accumulator, 16,16,16, float> C_acc;
        wmma::fill_fragment(C_acc, 0.0f);

        // k 维（head_dim）按 16 分块
        for (int k0 = 0; k0 < d; k0 += 16) {
            // 装 Q 到 Ash（row-major）
            for (int t = lane; t < 16*16; t += 32) {
                int r = t / 16, c = t % 16;
                const int gi = row_tile_start + r;
                __half val = __float2half(0.f);
                if (r < valid_rows && (k0 + c) < d) {
                    val = __float2half(Q_f32[(size_t)gi * d + (k0 + c)]);
                }
                Ash[r*16 + c] = val;
            }
            // 装 K 到 Bsh（col-major：B(c,r)=K(j0+c, k0+r)）
            for (int t = lane; t < 16*16; t += 32) {
                int c = t / 16, r = t % 16;
                const int gj = j0 + c;
                __half val = __float2half(0.f);
                if (c < cols_this && (k0 + r) < d) {
                    val = __float2half(K_f32[(size_t)gj * d + (k0 + r)]);
                }
                Bsh[c*16 + r] = val;
            }
            __syncwarp(); // 仅 warp 内同步

            // 载入 WMMA 片段并累加
            wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> A;
            wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> B;
            wmma::load_matrix_sync(A, Ash, 16);
            wmma::load_matrix_sync(B, Bsh, 16);
            wmma::mma_sync(C_acc, A, B, C_acc);
            __syncwarp();
        }

        // 把 S 存到共享内存，便于逐行 softmax
        wmma::store_matrix_sync(Ssh, C_acc, 16, wmma::mem_row_major);
        __syncwarp();

        // ==== 行级在线 softmax 融合 + 累加到 O ====
        if (lane < 16) {
            const int r = lane;
            if (r < valid_rows) {
                // 3.1 行内最大
                float m_tile = -1e30f;
                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    if (c < cols_this) {
                        float s = Ssh[r*16 + c] * scale;
                        m_tile = fmaxf(m_tile, s);
                    }
                }
                const float m_new = fmaxf(m, m_tile);
                const float alpha = __expf(m - m_new);

                // 3.2 行内 exp 和
                float sum_e = 0.f;
                float exp_buf[16];
                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    float e = 0.f;
                    if (c < cols_this) {
                        float s = Ssh[r*16 + c] * scale;
                        e = __expf(s - m_new);
                    }
                    exp_buf[c] = e;
                    sum_e += e;
                }
                const float l_new = l * alpha + sum_e + 1e-20f;
                const float coeff_old = (l * alpha) / l_new;
                const float inv_l_new = 1.0f / l_new;

                // 3.3 累加输出各列（逐 dv）
                const int gi = row_tile_start + r;
                for (int v = 0; v < dv; ++v) {
                    float prev = (j0 > 0) ? O_f32[(size_t)gi * dv + v] : 0.f;
                    float add  = 0.f;
                    #pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        if (c < cols_this) {
                            const int gj = j0 + c;
                            add += exp_buf[c] * V_f32[(size_t)gj * dv + v];
                        }
                    }
                    O_f32[(size_t)gi * dv + v] = prev * coeff_old + add * inv_l_new;
                }

                // 3.4 写回 softmax 行状态
                m = m_new;
                l = l_new;
            }
        }
        __syncwarp();
    }
    // 在线融合已完成归一化，无需尾部分母
}

// -------------------------- Host 侧包装（保持签名） --------------------------
extern "C" void
fa2_forward_wmma(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int Nq_local, int Nk_full,
    int d, int dv, float scale, int Br, int Bc,
    dim3 grid, dim3 block, size_t sharedBytes
){
    (void)Bc;
    // 约束：d 必须是 16 的倍数（当前实现要求）
    // 若 sharedBytes 未指定，则按 block.x 的 warp 数自动计算
    if (sharedBytes == 0) {
        sharedBytes = fa2_forward_smem_bytes(Br, Bc, d, dv, block.x);
    }
    flashAttn2_wmma_kernel<<<grid, block, sharedBytes>>>(
        d_Q, d_K, d_V, d_O,
        Nq_local, Nk_full, d, dv, scale, Br, Bc
    );
}

#endif // FA2_FWD_CU_INCLUDED
