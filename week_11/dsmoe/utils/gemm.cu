
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK =  32;
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WARPS = 16;
static constexpr int THREADS = WARPS * 32;

__global__ void gemm_kernel(const half* __restrict__ A,
                            const half* __restrict__ B,
                            float* __restrict__ C,
                            int T, int D, int E) {
    extern __shared__ unsigned char smem_raw[];
    half* As = reinterpret_cast<half*>(smem_raw);
    half* Bs = As + BM * BK;
    float* Cs = reinterpret_cast<float*>(Bs + BK * BN);

    int bm0 = blockIdx.y * BM;
    int bn0 = blockIdx.x * BN;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int warp_m = warp_id >> 2;
    int warp_n = warp_id &  3;
    int wm0 = warp_m * 32;
    int wn0 = warp_n * 32;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    #pragma unroll
    for (int i=0;i<2;i++) for (int j=0;j<2;j++) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0=0; k0<D; k0+=BK) {
        for (int idx = threadIdx.x; idx < BM*BK; idx += blockDim.x) {
            int r = idx / BK, c = idx % BK;
            int gr = bm0 + r, gc = k0 + c;
            As[idx] = (gr < T && gc < D) ? A[gr*D + gc] : __float2half(0.f);
        }
        for (int idx = threadIdx.x; idx < BK*BN; idx += blockDim.x) {
            int col = idx / BK, row = idx % BK;
            int gk = k0 + row, gn = bn0 + col;
            Bs[col*BK + row] = (gk < D && gn < E) ? B[gk*E + gn] : __float2half(0.f);
        }
        __syncthreads();

        #pragma unroll
        for (int kk=0; kk<BK; kk+=WMMA_K) {
            #pragma unroll
            for (int ti=0; ti<2; ++ti) {
                #pragma unroll
                for (int tj=0; tj<2; ++tj) {
                    int a_row = wm0 + ti*WMMA_M;
                    int b_col = wn0 + tj*WMMA_N;
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                    const half* ap = &As[a_row * BK + kk];
                    const half* bp = &Bs[b_col * BK + kk];
                    wmma::load_matrix_sync(a_frag, ap, BK);
                    wmma::load_matrix_sync(b_frag, bp, BK);
                    wmma::mma_sync(acc[ti][tj], a_frag, b_frag, acc[ti][tj]);
                }
            }
        }
        __syncthreads();
    }

    float* wCs = Cs + warp_id * (4 * WMMA_M * WMMA_N);
    #pragma unroll
    for (int ti=0; ti<2; ++ti) {
        #pragma unroll
        for (int tj=0; tj<2; ++tj) {
            int t = ti*2 + tj;
            float* cp = wCs + t * (WMMA_M * WMMA_N);
            wmma::store_matrix_sync(cp, acc[ti][tj], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            int cr0 = bm0 + wm0 + ti*WMMA_M;
            int cc0 = bn0 + wn0 + tj*WMMA_N;
            for (int e = lane_id; e < WMMA_M*WMMA_N; e += 32) {
                int r = e / WMMA_N, c = e % WMMA_N;
                int gr = cr0 + r, gc = cc0 + c;
                if (gr < T && gc < E) C[gr*E + gc] = cp[r*WMMA_N + c];
            }
            __syncwarp();
        }
    }
}

extern "C" void gemm(const half* x,
                     const half* W,
                     float* logits,
                     int T, int D, int E,
                     cudaStream_t stream) {
    dim3 grid(CEIL_DIV(E, BN), CEIL_DIV(T, BM), 1);
    dim3 block(THREADS, 1, 1);
    size_t smem =
        size_t(BM)*BK*sizeof(half) +
        size_t(BK)*BN*sizeof(half) +
        size_t(THREADS/32)*4*WMMA_M*WMMA_N*sizeof(float);
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemm_kernel<<<grid, block, smem, stream>>>(x, W, logits, T, D, E);
}
