// fa2_fwd.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cassert>
using namespace nvcuda;

static inline __host__ __device__ size_t align16(size_t x){ return (x + 15) & ~size_t(15); }

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v){
    #pragma unroll
    for (int off=16; off>0; off>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}

__host__ __device__ inline
void compute_block_sizes(int M, int d, int& Br, int& Bc) {
    int denom = 4 * d;
    int Bc_tmp = (M + denom - 1) / denom;  // ceil
    if (Bc_tmp < 1) Bc_tmp = 1;
    int Br_tmp = (Bc_tmp < d) ? Bc_tmp : d;
    Bc = Bc_tmp;
    Br = Br_tmp;
}

__global__ void flashAttn2_wmma_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, int N, int d, int M)
{
    int dv = d;
    int Br, Bc;
    compute_block_sizes(M, d, Br, Bc);

    const int W       = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const int i_tile  = blockIdx.y;
    const int i0      = i_tile * Br;

    extern __shared__ char smem_raw[]; // using char here because
    char* ptr = smem_raw;

    float* Kds = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Bc * d * sizeof(float));
    float* m_row = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Br * sizeof(float));
    float* l_row = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Br * sizeof(float));
    float* p_row = reinterpret_cast<float*>(ptr);
    ptr += align16((size_t)Br * dv * sizeof(float));
    __half* Vds = reinterpret_cast<__half*>(ptr);
    ptr += align16((size_t)Bc * dv * sizeof(__half));

    __half* Ash  = reinterpret_cast<__half*>(ptr);     (void)Ash;
    ptr += (size_t)W * (2 * 16 * 16 * sizeof(__half));
    float*  Swrp = reinterpret_cast<float*>(ptr);      (void)Swrp;

    __shared__ float sum_warp[32];

    // buffer init
    for (int r = threadIdx.x; r < Br; r += blockDim.x){ m_row[r] = -INFINITY; l_row[r] = 0.0f; }
    for (int idx = threadIdx.x; idx < Br*dv; idx += blockDim.x) p_row[idx] = 0.0f;
    __syncthreads();

    // CTA lvl
    // split kv
    for (int j0 = 0; j0 < N; j0 += Bc){
        const int cols = min(Bc, N - j0);

        // smsm <- kv
        for (int t = threadIdx.x; t < cols * d; t += blockDim.x)
            Kds[t] = K[(size_t)j0 * d + t];
        for (int t = threadIdx.x; t < cols * dv; t += blockDim.x)
            Vds[t] = __float2half(V[(size_t)j0 * dv + t]);
        __syncthreads();

        // CTA -> 16*16 tiles
        for (int i_off = 0; i_off < Br; i_off += 16){
            const int i_start = i0 + i_off;
            if (i_start >= N) break;
            const int mr_tail = min(16, N - i_start);

            // 列块分 16 列处理
            for (int c_off = 0; c_off < cols; c_off += 16){
                const int cc = min(16, cols - c_off);

                // === 用 WMMA 生成当前 16×cc 的 S 子块（warp0 计算，其余 warp 消费） ===
                float* Sblk = Swrp;         // 统一使用 warp0 的 staging
                if (warp_id == 0){
                    __half* As = Ash;                   // A(16x16) half
                    __half* Bs = As + 16*16;            // B(16x16) half（col-major）

                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> A;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> B;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> C;
                    wmma::fill_fragment(C, 0.0f);

                    for (int k0 = 0; k0 < d; k0 += 16){
                        // A: Q[i_start..i_start+15, k0..k0+15]（row-major）
                        for (int t = lane; t < 16*16; t += 32){
                            int r = t / 16, c = t % 16;
                            int gi = i_start + r;
                            __half a = (r < mr_tail && (k0 + c) < d) ? __float2half(Q[(size_t)gi * d + (k0 + c)]) : __float2half(0.0f);
                            As[r*16 + c] = a;
                        }
                        __syncwarp();

                        // B: Kds[(c_off..c_off+cc-1), k0..k0+15]（col-major视图：Bs[row + col*16]）
                        for (int t = lane; t < 16*16; t += 32){
                            int r = t / 16, c = t % 16;       // r≡k, c≡j
                            int gj = c_off + c;
                            __half b = (c < cc && (k0 + r) < d) ? __float2half(Kds[(size_t)gj * d + (k0 + r)]) : __float2half(0.0f);
                            Bs[r + c*16] = b;                 // col-major layout
                        }
                        __syncwarp();

                        wmma::load_matrix_sync(A, As, 16);
                        wmma::load_matrix_sync(B, Bs, 16);
                        wmma::mma_sync(C, A, B, C);
                        __syncwarp();
                    }

                    wmma::store_matrix_sync(Sblk, C, 16, wmma::mem_row_major);
                    for (int t = lane; t < 16*16; t += 32){
                        if ((t % 16) >= cc) Sblk[t] = 0.0f;
                    }
                }
                __syncthreads();

                // 逐行处理
                for (int r_loc = 0; r_loc < mr_tail; ++r_loc){
                    const int gi   = i_start + r_loc;
                    const int ridx = i_off + r_loc;

                    // warp/tile lvl：行最大
                    float mx_part = -INFINITY; // online update m init, from fa1
                    for (int c = warp_id*32 + lane; c < cc; c += W*32){
                        float s_ij = Sblk[r_loc*16 + c];
                        mx_part = fmaxf(mx_part, s_ij);
                    }
                    mx_part = warp_max(mx_part);
                    if (lane == 0) sum_warp[warp_id] = mx_part;
                    __syncthreads();
                    float mx_tile = -INFINITY;
                    if (warp_id == 0){
                        float val = (lane < W) ? sum_warp[lane] : -INFINITY;
                        val = warp_max(val);
                        if (lane == 0) sum_warp[0] = val;
                    }
                    __syncthreads();
                    mx_tile = sum_warp[0];

                    // 在线更新 (m,l)
                    const float m_old = m_row[ridx];
                    const float l_old = l_row[ridx];
                    const float m_new = fmaxf(m_old, mx_tile);
                    const float alpha = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);

                    // 分母 sum_e
                    float se_part = 0.f;
                    for (int c = warp_id*32 + lane; c < cc; c += W*32){
                        const float s_ij = Sblk[r_loc*16 + c];
                        se_part += __expf(s_ij - m_new);
                    }
                    se_part = warp_sum(se_part);
                    if (lane == 0) sum_warp[warp_id] = se_part;
                    __syncthreads();
                    float sum_e = 0.f;
                    if (warp_id == 0){
                        float val = (lane < W) ? sum_warp[lane] : 0.f;
                        val = warp_sum(val);
                        if (lane == 0) sum_warp[0] = val;
                    }
                    __syncthreads();
                    sum_e = sum_warp[0];

                    if (warp_id == 0 && lane == 0){
                        l_row[ridx] = alpha * l_old + sum_e;
                        m_row[ridx] = m_new;
                    }
                    __syncthreads();

                    // 分子 p_row（按 dv 条带并行）
                    for (int v = warp_id * 32 + lane; v < dv; v += W * 32){
                        float pv_acc = 0.f;
                        for (int c = 0; c < cc; ++c){
                            const float s_ij = Sblk[r_loc*16 + c];
                            const float e    = __expf(s_ij - m_new);
                            pv_acc += e * __half2float(Vds[(size_t)(c_off + c) * dv + v]);
                        }
                        const size_t poff = (size_t)ridx * dv + v;
                        const float prev  = p_row[poff];
                        p_row[poff] = alpha * prev + pv_acc;
                    }
                    __syncthreads();
                } // r_loc
            } // c_off

            // 16 行子块归一化并写回
            for (int r_loc = warp_id; r_loc < mr_tail; r_loc += W){
                const int gi2   = i_start + r_loc;
                const int ridx2 = i_off + r_loc;
                const float inv_l = 1.0f / fmaxf(l_row[ridx2], 1e-20f);
                for (int v = lane; v < dv; v += 32){
                    O[(size_t)gi2 * dv + v] = p_row[(size_t)ridx2 * dv + v] * inv_l;
                }
            }
            __syncthreads();
        } // i_off
    } // j0
}

//================= 动态共享内存大小 =================
extern "C" size_t fa2_forward_smem_bytes(int Br, int Bc, int d, int dv, int block_x){
    const int W = block_x / 32;
    size_t bytes = 0;
    bytes += align16((size_t)Bc * d  * sizeof(float)); // Kds
    bytes += align16((size_t)Br      * sizeof(float)); // m_row
    bytes += align16((size_t)Br      * sizeof(float)); // l_row
    bytes += align16((size_t)Br * dv * sizeof(float)); // p_row
    bytes += align16((size_t)Bc * dv * sizeof(__half));// Vds
    bytes += (size_t)W * (2 * 16 * 16 * sizeof(__half)); // Ash+Bsh（占位）
    bytes += align16((size_t)W * 16 * 16 * sizeof(float));// Swarp（占位）
    return bytes;
}

//================= 前向 launcher ====================
extern "C" void fa2_forward_wmma(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d, int M, dim3 grid, dim3 block, size_t sharedBytes)
{
    flashAttn2_wmma_kernel<<<grid, block, sharedBytes>>>(d_Q, d_K, d_V, d_O, N, d, M);
}
