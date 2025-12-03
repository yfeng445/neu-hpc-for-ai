#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace nvcuda;

#define CHECK_CUDA(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

__host__ __device__ inline int ceil_div(int x, int y){ return (x + y - 1) / y; }

__device__ inline float atomicMaxF(float* addr, float val) {
  float old = *addr;
  while (val > old) {
    float assumed = old;
    unsigned int old_u = atomicCAS(
        reinterpret_cast<unsigned int*>(addr),
        __float_as_uint(assumed),
        __float_as_uint(val));
    old = __uint_as_float(old_u);
    if (old == assumed) break;
  }
  return old;
}

__global__ void fa2_forward_wmma(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, float* __restrict__ O, float* __restrict__ L, int N, int d, int Br, int Bc){
    
    constexpr int WM = 16, WN = 16, WK = 16;

    const int i0 = blockIdx.y * Br;
    if (i0 >= N) return;

    extern __shared__ char smem[];
    char* p = smem;

    __half* smA   = reinterpret_cast<__half*>(p);
    p += size_t(Br)*WK * sizeof(__half);
    __half* smB_k = reinterpret_cast<__half*>(p);
    p += WK*size_t(Bc) * sizeof(__half);
    float*  rowmx = reinterpret_cast<float*>(p);
    p += size_t(Br) * sizeof(float);
    float*  rowsm = reinterpret_cast<float*>(p);
    p += size_t(Br) * sizeof(float);
    float*  m_run = reinterpret_cast<float*>(p);
    p += size_t(Br) * sizeof(float);
    float*  l_run = reinterpret_cast<float*>(p);
    p += size_t(Br) * sizeof(float);
    float*  Pnum  = reinterpret_cast<float*>(p);
    p += size_t(Br)*size_t(d) * sizeof(float);

    const int tilesM = Br / WM;
    const int tilesN = Bc / WN;
    const int warps  = tilesM * tilesN;

    float*  Sbuf = reinterpret_cast<float*>(p);
    p += size_t(warps)*WM*WN * sizeof(float);
    __half* Wbuf = reinterpret_cast<__half*>(p);
    p += size_t(warps)*WM*WN * sizeof(__half);
    __half* Vbuf = reinterpret_cast<__half*>(p);

    for (int r = threadIdx.x; r < Br; r += blockDim.x) { m_run[r] = -CUDART_INF_F; l_run[r] = 0.f; }
    for (int t = threadIdx.x; t < Br*d; t += blockDim.x) Pnum[t] = 0.f;
    __syncthreads();

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= warps) return;

    const int tile_m = warp_id % tilesM;
    const int tile_n = warp_id / tilesM;

    float*  S_tile   = Sbuf + warp_id * WM * WN;
    __half* W_tile_h = Wbuf + warp_id * WM * WN;
    __half* V_tile_h = Vbuf + warp_id * WM * WN;

    const int j_blocks = ceil_div(N, Bc);
    for (int jb = 0; jb < j_blocks; ++jb) {

        for (int r = threadIdx.x; r < Br; r += blockDim.x) { rowmx[r] = -CUDART_INF_F; rowsm[r] = 0.f; }
        __syncthreads();

        wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < d; k0 += WK) {

            for (int t = threadIdx.x; t < Br*WK; t += blockDim.x) {
                int rr = t / WK, kk = t % WK;
                int gi = i0 + rr;
                __half v = (gi < N && (k0+kk) < d) ? Q[gi*d + (k0+kk)] : __float2half(0.f);
                smA[rr*WK + kk] = v;
            }

            const int j0 = jb * Bc;
            for (int e = threadIdx.x; e < WK*Bc; e += blockDim.x) {
                int kk = e / Bc, cc = e % Bc;
                int gk = j0 + cc;
                __half v = (gk < N && (k0+kk) < d) ? K[gk*d + (k0+kk)] : __float2half(0.f);
                smB_k[kk + cc*WK] = v;
            }
            __syncthreads();

            const __half* As = &smA[(tile_m * WM) * WK];
            const __half* Bs = &smB_k[(tile_n * WN) * WK];

            wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, As, WK);
            wmma::load_matrix_sync(b_frag, Bs, WK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }

        wmma::store_matrix_sync(S_tile, c_frag, WN, wmma::mem_row_major);

        for (int r = lane_id; r < WM; r += 32) {
            int rg = tile_m * WM + r;
            float mloc = -CUDART_INF_F;
            #pragma unroll
            for (int c = 0; c < WN; ++c) mloc = fmaxf(mloc, S_tile[r*WN + c]);
            atomicMaxF(&rowmx[rg], mloc);
        }
        __syncthreads();

        for (int r = lane_id; r < WM; r += 32) {
            int rg = tile_m * WM + r;
            float mglob = rowmx[rg];
            float ssum2 = 0.f;
            #pragma unroll
            for (int c = 0; c < WN; ++c)
                ssum2 += __expf(S_tile[r*WN + c] - mglob);
            atomicAdd(&rowsm[rg], ssum2);
        }
        __syncthreads();

        for (int r = threadIdx.x; r < Br; r += blockDim.x) {
            float m_old = m_run[r], l_old = l_run[r];
            float m_loc = rowmx[r], l_loc = rowsm[r];
            float m_new = fmaxf(m_old, m_loc);
            float s_prev = __expf(m_old - m_new);
            float s_cur  = __expf(m_loc - m_new);
            int base = r * d;
            for (int k = 0; k < d; ++k) Pnum[base + k] *= s_prev;
            m_run[r] = m_new;
            l_run[r] = l_old * s_prev + l_loc * s_cur;
        }
        __syncthreads();

        float rowmx_loc[WM], mnew_loc[WM];
        for (int r = 0; r < WM; ++r) {
            int rg = tile_m*WM + r;
            rowmx_loc[r] = rowmx[rg];
            mnew_loc[r]  = m_run[rg];
        }
        for (int r = 0; r < WM; ++r) {
            float scale = __expf(rowmx_loc[r] - mnew_loc[r]);
            #pragma unroll
            for (int c = 0; c < WN; ++c) {
                float w = __expf(S_tile[r*WN + c] - rowmx_loc[r]) * scale;
                W_tile_h[r*WN + c] = __float2half(w);
            }
        }

        for (int kd = 0; kd < d; kd += WK) {

            const int j0 = jb * Bc + tile_n * WN;
            for (int e = lane_id; e < WM*WN; e += 32) {
                int rr = e % WM;
                int cc = e / WM;
                int gr = j0 + rr;
                int gc = kd + cc;
                __half v = (gr < N && gc < d) ? V[gr*d + gc] : __float2half(0.f);
                V_tile_h[rr + cc*WM] = v;
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> A2;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> B2;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> C2;
            wmma::fill_fragment(C2, 0.0f);
            wmma::load_matrix_sync(A2, W_tile_h, WN);
            wmma::load_matrix_sync(B2, V_tile_h, WM);
            wmma::mma_sync(C2, A2, B2, C2);

            wmma::store_matrix_sync(S_tile, C2, WN, wmma::mem_row_major);

            const int r0 = tile_m * WM, c0 = kd;
            for (int r = 0; r < WM; ++r) {
                int br = r0 + r; if (br >= Br) continue;
                int base = br * d + c0;
                for (int c = lane_id; c < WN; c += 32) {
                    if (c0 + c < d) atomicAdd(&Pnum[base + c], S_tile[r*WN + c]);
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    for (int r = threadIdx.x; r < Br; r += blockDim.x) {
        int gi = i0 + r; if (gi >= N) continue;
        float lr = fmaxf(l_run[r], 1e-20f);
        L[gi] = m_run[r] + logf(lr);
        for (int k = 0; k < d; ++k) O[gi*d + k] = Pnum[r*d + k] / lr;
    }
}

inline size_t fa2_smem_bytes_rt(int Br, int Bc, int d) {
  constexpr int WM=16, WN=16, WK=16;
  const int tilesM = Br/WM, tilesN = Bc/WN;
  const int warps  = tilesM*tilesN;

  size_t bytes = 0;
  bytes += size_t(Br)*WK*sizeof(__half);
  bytes += WK*size_t(Bc)*sizeof(__half);
  bytes += 4*size_t(Br)*sizeof(float);
  bytes += size_t(Br)*size_t(d)*sizeof(float);
  bytes += size_t(warps)*WM*WN*sizeof(float);
  bytes += size_t(warps)*WM*WN*sizeof(__half);
  bytes += size_t(warps)*WM*WN*sizeof(__half);
  return bytes;
}

__host__ __device__ inline int threads_per_block_for(int Br, int Bc){
  return (Br/16) * (Bc/16) * 32;
}

inline void fa2_launch_rt(const __half* Q, const __half* K, const __half* V,
                          float* O, float* L,
                          int N, int d, int Br, int Bc)
{
  if (Br%16 || Bc%16) { throw std::runtime_error("Br/Bc must be multiples of 16."); }
  const int warps = (Br/16)*(Bc/16);
  if (warps <= 0 || warps > 32) { throw std::runtime_error("warps per CTA must be in (0,32]."); }

  dim3 grid(1, ceil_div(N, Br));
  dim3 block(threads_per_block_for(Br,Bc));
  size_t smem = fa2_smem_bytes_rt(Br, Bc, d);

  fa2_forward_wmma<<<grid, block, smem>>>(Q,K,V,O,L,N,d,Br,Bc);
  CHECK_CUDA(cudaGetLastError());
}

int main() {
    const int N = 32, d = 32, size = N*d;

    std::vector<float> h_Q(size), h_K(size), h_V(size), h_O(size), h_L(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            int idx = i * d + j;
            float x = static_cast<float>(idx);
            h_Q[idx] = std::sinf(0.01f * x);
            h_K[idx] = std::cosf(0.02f * x);
            h_V[idx] = std::sinf(0.03f * x + 0.5f);
        }
    }

    std::vector<__half> h_Qh(size), h_Kh(size), h_Vh(size);
    for (int i = 0; i < size; ++i) {
        h_Qh[i] = __float2half(h_Q[i]);
        h_Kh[i] = __float2half(h_K[i]);
        h_Vh[i] = __float2half(h_V[i]);
    }

    __half *d_Q=nullptr, *d_K=nullptr, *d_V=nullptr;
    float  *d_O=nullptr, *d_L=nullptr;
    CHECK_CUDA(cudaMalloc(&d_Q, size*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_K, size*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_V, size*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_O, size*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_L, N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Qh.data(), size*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_Kh.data(), size*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_Vh.data(), size*sizeof(__half), cudaMemcpyHostToDevice));

    const int Br = 32, Bc = 32;
    fa2_launch_rt(d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, size*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_L.data(), d_L, N*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout.setf(std::ios::fixed); std::cout<<std::setprecision(3);
    std::cout << "Output O:\n";
    for (int i = 0; i < N;  ++i) {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < d; ++j)
            std::cout << h_O[i*d + j] << (j+1<d ? ' ' : '\n');
    }
    std::cout << "L: ";
    for (int i = 0; i < N; ++i)
        std::cout << h_L[i] << (i+1<N ? ' ' : '\n');

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    return 0;
}
