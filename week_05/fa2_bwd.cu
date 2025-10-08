// fa2_bwd.cu
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

// 反向（工程版）：三遍扫描（同一 CTA 负责 Br 行）
//  Pass1: 重算 m_row, l_row（在线 softmax）；
//  Pass2: 计算 delta_row（每行标量）与 dV（atomicAdd）；
//  Pass3: 计算 dQ（行块唯一负责；此处为稳妥仍用 atomicAdd）与 dK（atomicAdd）。
__global__ void flashAttn2_bwd_wmma_kernel(
    const float* __restrict__ Q,   // [N,d]
    const float* __restrict__ K,   // [N,d]
    const float* __restrict__ V,   // [N,dv]
    const float* __restrict__ dO,  // [N,dv]
    float* __restrict__ dQ,        // [N,d]
    float* __restrict__ dK,        // [N,d]
    float* __restrict__ dV,        // [N,dv]
    int N, int d, int dv, float scale, int Br, int Bc)
{
    const int W       = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    const int i_tile  = blockIdx.y;   // 行 tile id
    const int i0      = i_tile * Br;

    extern __shared__ char smem_raw[]; // using char here because 
    char* ptr = smem_raw;

    // --- 共享内存布局（与前向保持一致的风格） ---
    float* Kds = reinterpret_cast<float*>(ptr);                   // [Bc, d]
    ptr += align16((size_t)Bc * d * sizeof(float));
    float* m_row = reinterpret_cast<float*>(ptr);                 // [Br]
    ptr += align16((size_t)Br * sizeof(float));
    float* l_row = reinterpret_cast<float*>(ptr);                 // [Br]
    ptr += align16((size_t)Br * sizeof(float));
    float* delta_row = reinterpret_cast<float*>(ptr);             // [Br] 第二遍累计 δ_i
    ptr += align16((size_t)Br * sizeof(float));
    __half* Vds = reinterpret_cast<__half*>(ptr);                 // [Bc, dv] 半精度缓存（按 j0 局部列索引排布）
    ptr += align16((size_t)Bc * dv * sizeof(__half));

    __half* Ash  = reinterpret_cast<__half*>(ptr);     (void)Ash; // warp0 的 A/B 瓦片 staging
    ptr += (size_t)W * (2 * 16 * 16 * sizeof(__half));
    float*  Sblk = reinterpret_cast<float*>(ptr);      (void)Sblk;// warp0 的 16×16 瓦片输出 staging
    // sum_warp 做 CTA 级规约
    __shared__ float sum_warp[32];

    // ====== 初始化 ======
    for (int r = threadIdx.x; r < Br; r += blockDim.x){
        m_row[r]     = -INFINITY;
        l_row[r]     = 0.0f;
        delta_row[r] = 0.0f;
    }
    // dQ 局部清零（仅清本 CTA 负责的行）
    for (int idx = threadIdx.x; idx < Br * d; idx += blockDim.x){
        int r = idx / d;
        int k = idx % d;
        int gi = i0 + r;
        if (gi < N) dQ[(size_t)gi * d + k] = 0.0f;
    }
    __syncthreads();

    // ========= Pass 1：重算 m_row, l_row =========
    for (int j0 = 0; j0 < N; j0 += Bc){
        const int cols = min(Bc, N - j0);

        // smsm <- K, V
        for (int t = threadIdx.x; t < cols * d; t += blockDim.x)
            Kds[t] = K[(size_t)j0 * d + t];
        for (int t = threadIdx.x; t < cols * dv; t += blockDim.x)
            Vds[t] = __float2half(V[(size_t)j0 * dv + t]);
        __syncthreads();

        // 行块分 16 行处理
        for (int i_off = 0; i_off < Br; i_off += 16){
            const int i_start = i0 + i_off;
            if (i_start >= N) break;
            const int mr_tail = min(16, N - i_start);

            // 列块分 16 列处理
            for (int c_off = 0; c_off < cols; c_off += 16){
                const int cc = min(16, cols - c_off);

                // 用 WMMA 生成当前 16×cc 的 S 子块（warp0 计算，其它 warp 消费）
                float* Sw = Sblk;  // 统一 staging
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
                        // B: Kds[(c_off..c_off+cc-1), k0..k0+15]（col-major：Bs[row + col*16]）
                        for (int t = lane; t < 16*16; t += 32){
                            int r = t / 16, c = t % 16;  // r≡k, c≡j
                            int gj = c_off + c;
                            __half b = (c < cc && (k0 + r) < d) ? __float2half(Kds[(size_t)gj * d + (k0 + r)]) : __float2half(0.0f);
                            Bs[r + c*16] = b;
                        }
                        __syncwarp();

                        wmma::load_matrix_sync(A, As, 16);
                        wmma::load_matrix_sync(B, Bs, 16);
                        wmma::mma_sync(C, A, B, C);
                        __syncwarp();
                    }
                    // 写出并乘 scale；超出 cc 的列清零
                    wmma::store_matrix_sync(Sw, C, 16, wmma::mem_row_major);
                    for (int t = lane; t < 16*16; t += 32){
                        if ((t % 16) < cc) Sw[t] *= scale;
                        else               Sw[t]  = 0.0f;
                    }
                }
                __syncthreads();

                // 在线 softmax 更新 (m,l)
                for (int r_loc = 0; r_loc < mr_tail; ++r_loc){
                    const int ridx = i_off + r_loc;

                    float mx_part = -INFINITY;
                    for (int c = warp_id*32 + lane; c < cc; c += W*32){
                        float s_ij = Sw[r_loc*16 + c];
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

                    const float m_old = m_row[ridx];
                    const float l_old = l_row[ridx];
                    const float m_new = fmaxf(m_old, mx_tile);
                    const float alpha = (m_old == -INFINITY) ? 0.f : __expf(m_old - m_new);

                    float se_part = 0.f;
                    for (int c = warp_id*32 + lane; c < cc; c += W*32){
                        const float s_ij = Sw[r_loc*16 + c];
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
                } // r_loc
            } // c_off
        } // i_off
    } // j0

    // ========= Pass 2：累计 delta_row 与 dV =========
    for (int j0 = 0; j0 < N; j0 += Bc){
        const int cols = min(Bc, N - j0);

        // smsm <- K, V
        for (int t = threadIdx.x; t < cols * d; t += blockDim.x)
            Kds[t] = K[(size_t)j0 * d + t];
        for (int t = threadIdx.x; t < cols * dv; t += blockDim.x)
            Vds[t] = __float2half(V[(size_t)j0 * dv + t]);
        __syncthreads();

        // 行块分 16 行处理
        for (int i_off = 0; i_off < Br; i_off += 16){
            const int i_start = i0 + i_off;
            if (i_start >= N) break;
            const int mr_tail = min(16, N - i_start);

            // 列块分 16 列处理
            for (int c_off = 0; c_off; c_off += 16){} // placate compiler
            for (int c_off = 0; c_off < cols; c_off += 16){
                const int cc = min(16, cols - c_off);

                // WMMA 生成 S 瓦片
                float* Sw = Sblk;
                if (warp_id == 0){
                    __half* As = Ash; __half* Bs = As + 16*16;
                    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> A;
                    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> B;
                    wmma::fragment<wmma::accumulator,16,16,16,float> C;
                    wmma::fill_fragment(C, 0.0f);
                    for (int k0 = 0; k0 < d; k0 += 16){
                        for (int t = lane; t < 16*16; t += 32){
                            int r=t/16,c=t%16; int gi=i_start+r;
                            __half a=(r<mr_tail && k0+c<d)?__float2half(Q[(size_t)gi*d+(k0+c)]):__float2half(0.f);
                            As[r*16+c]=a;
                        }
                        __syncwarp();
                        for (int t = lane; t < 16*16; t += 32){
                            int r=t/16,c=t%16; int gj=c_off+c;
                            __half b=(c<cc && k0+r<d)?__float2half(Kds[(size_t)gj*d+(k0+r)]):__float2half(0.f);
                            Bs[r + c*16]=b; // col-major
                        }
                        __syncwarp();
                        wmma::load_matrix_sync(A,As,16);
                        wmma::load_matrix_sync(B,Bs,16);
                        wmma::mma_sync(C,A,B,C);
                        __syncwarp();
                    }
                    wmma::store_matrix_sync(Sw, C, 16, wmma::mem_row_major);
                    for (int t = lane; t < 16*16; t += 32){
                        if ((t%16)<cc) Sw[t]*=scale; else Sw[t]=0.0f;
                    }
                }
                __syncthreads();

                // 累计 delta_row = Σ_j dP_ij * P_ij；同时累计 dV = Σ_i P_ij * dO_i
                for (int r_loc = 0; r_loc < mr_tail; ++r_loc){
                    const int gi   = i_start + r_loc;
                    const int ridx = i_off + r_loc;
                    const float m_i = m_row[ridx];
                    const float l_i = fmaxf(l_row[ridx], 1e-20f);
                    const float inv_l = 1.0f / l_i;

                    float delta_part = 0.f;
                    for (int c = warp_id*32 + lane; c < cc; c += W*32){
                        const int j  = j0 + c_off + c;  // 全局列
                        const int jl = c_off + c;       // <<< 局部列（相对 j0），用于 Vds/Kds

                        const float s_ij = Sw[r_loc*16 + c];
                        const float p_ij = __expf(s_ij - m_i) * inv_l;

                        // dP_ij = Σ_v dO[i,v] * V[j,v] —— 从 Vds 用“局部列”索引
                        float dP_ij = 0.f;
                        for (int v = 0; v < dv; ++v)
                            dP_ij += dO[(size_t)gi * dv + v] * __half2float(Vds[(size_t)jl * dv + v]);

                        delta_part += dP_ij * p_ij;

                        // dV[j,:] += p_ij * dO[i,:] —— 写全局 dV 用“全局列”
                        for (int v = 0; v < dv; ++v){
                            float contrib = p_ij * dO[(size_t)gi * dv + v];
                            atomicAdd(&dV[(size_t)j * dv + v], contrib);
                        }
                    }
                    // CTA 级规约 delta
                    delta_part = warp_sum(delta_part);
                    if (lane == 0) sum_warp[warp_id] = delta_part;
                    __syncthreads();
                    if (warp_id == 0){
                        float val = (lane < W) ? sum_warp[lane] : 0.f;
                        val = warp_sum(val);
                        if (lane == 0) atomicAdd(&delta_row[ridx], val);
                    }
                    __syncthreads();
                } // r_loc
            } // c_off
        } // i_off
    } // j0

    // ========= Pass 3：计算 dQ, dK =========
    for (int j0 = 0; j0 < N; j0 += Bc){
        const int cols = min(Bc, N - j0);

        // smsm <- K, V
        for (int t = threadIdx.x; t < cols * d; t += blockDim.x)
            Kds[t] = K[(size_t)j0 * d + t];
        for (int t = threadIdx.x; t < cols * dv; t += blockDim.x)
            Vds[t] = __float2half(V[(size_t)j0 * dv + t]);
        __syncthreads();

        // 行块分 16 行处理
        for (int i_off = 0; i_off < Br; i_off += 16){
            const int i_start = i0 + i_off;
            if (i_start >= N) break;
            const int mr_tail = min(16, N - i_start);

            // 列块分 16 列处理
            for (int c_off = 0; c_off < cols; c_off += 16){
                const int cc = min(16, cols - c_off);

                // WMMA 生成 S 瓦片
                float* Sw = Sblk;
                if (warp_id == 0){
                    __half* As = Ash; __half* Bs = As + 16*16;
                    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> A;
                    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> B;
                    wmma::fragment<wmma::accumulator,16,16,16,float> C;
                    wmma::fill_fragment(C, 0.0f);
                    for (int k0 = 0; k0 < d; k0 += 16){
                        for (int t = lane; t < 16*16; t += 32){
                            int r=t/16,c=t%16; int gi=i_start+r;
                            __half a=(r<mr_tail && k0+c<d)?__float2half(Q[(size_t)gi*d+(k0+c)]):__float2half(0.f);
                            As[r*16+c]=a;
                        }
                        __syncwarp();
                        for (int t = lane; t < 16*16; t += 32){
                            int r=t/16,c=t%16; int gj=c_off+c;
                            __half b=(c<cc && k0+r<d)?__float2half(Kds[(size_t)gj*d+(k0+r)]):__float2half(0.f);
                            Bs[r + c*16]=b;
                        }
                        __syncwarp();
                        wmma::load_matrix_sync(A,As,16);
                        wmma::load_matrix_sync(B,Bs,16);
                        wmma::mma_sync(C,A,B,C);
                        __syncwarp();
                    }
                    wmma::store_matrix_sync(Sw, C, 16, wmma::mem_row_major);
                    for (int t = lane; t < 16*16; t += 32){
                        if ((t%16)<cc) Sw[t]*=scale; else Sw[t]=0.0f;
                    }
                }
                __syncthreads();

                // 构造 dS 瓦片，并对 dQ,dK 做贡献
                for (int r_loc = 0; r_loc < mr_tail; ++r_loc){
                    const int gi   = i_start + r_loc;
                    const int ridx = i_off + r_loc;
                    const float m_i = m_row[ridx];
                    const float l_i = fmaxf(l_row[ridx], 1e-20f);
                    const float inv_l = 1.0f / l_i;
                    const float delta = delta_row[ridx];

                    for (int c = warp_id*32 + lane; c < cc; c += W*32){
                        const int j  = j0 + c_off + c;  // 全局列
                        const int jl = c_off + c;       // <<< 局部列（相对 j0），用于 Vds/Kds

                        const float s_ij = Sw[r_loc*16 + c];
                        const float p_ij = __expf(s_ij - m_i) * inv_l;

                        // dP_ij = Σ_v dO[i,v] * V[j,v] —— 从 Vds 用“局部列”索引
                        float dP_ij = 0.f;
                        for (int v = 0; v < dv; ++v)
                            dP_ij += dO[(size_t)gi * dv + v] * __half2float(Vds[(size_t)jl * dv + v]);

                        const float dS_ij = (dP_ij - delta) * p_ij;

                        // dQ 累加：对 k in [0..d)（同一 CTA 多线程并发，使用 atomicAdd）
                        for (int k = 0; k < d; ++k){
                            float contrib = dS_ij * Kds[(size_t)jl * d + k] * scale;
                            atomicAdd(&dQ[(size_t)gi * d + k], contrib);
                        }
                        // dK 累加：对 k in [0..d)（跨 CTA 汇总，使用 atomicAdd）
                        for (int k = 0; k < d; ++k){
                            float contrib = dS_ij * Q[(size_t)gi * d + k] * scale;
                            atomicAdd(&dK[(size_t)j * d + k], contrib);
                        }
                    }
                    __syncthreads();
                } // r_loc
            } // c_off
        } // i_off
    } // j0
}

//================= 动态共享内存大小 =================
extern "C" size_t fa2_backward_smem_bytes(int Br, int Bc, int d, int dv, int block_x){
    const int W = block_x / 32;
    size_t bytes = 0;
    bytes += align16((size_t)Bc * d  * sizeof(float)); // Kds
    bytes += align16((size_t)Br      * sizeof(float)); // m_row
    bytes += align16((size_t)Br      * sizeof(float)); // l_row
    bytes += align16((size_t)Br      * sizeof(float)); // delta_row
    bytes += align16((size_t)Bc * dv * sizeof(__half));// Vds
    bytes += (size_t)W * (2 * 16 * 16 * sizeof(__half)); // Ash+Bsh（占位）
    bytes += align16((size_t)W * 16 * 16 * sizeof(float));// Sblk（占位）
    return bytes;
}

//================= 反向 launcher ====================
extern "C" void fa2_backward_wmma(
    const float* dQ, const float* dK, const float* dV, const float* dO,
    float* dQ_out, float* dK_out, float* dV_out,
    int N, int d, int dv, float scale, int Br, int Bc,
    dim3 grid, dim3 block, size_t sharedBytes)
{
    assert((Br%16)==0 && (Bc%16)==0 && (d%16)==0);
    // dK_out/dV_out 需要在外部清零（atomicAdd 汇总），dQ_out 也可清零后使用 atomicAdd（或在本 kernel 内改为 +=）
    flashAttn2_bwd_wmma_kernel<<<grid, block, sharedBytes>>>(
        dQ, dK, dV, dO, dQ_out, dK_out, dV_out, N, d, dv, scale, Br, Bc);
}
