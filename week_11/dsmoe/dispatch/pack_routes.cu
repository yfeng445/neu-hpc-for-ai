// dispatch/pack_routes.cu  (multi-GPU aware)
// - 单卡: 与旧版一致（乘 α、按 expert 打包、sendcounts[0]=T*K*D）
// - 多卡: 先对每个 tk 统计目的 rank 直方图 -> 前缀和 rank_offsets -> 每 block 仅一次 atomicAdd 到对应 rank 游标 -> pack 到 sendbuf 的对应 rank 段
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <assert.h>

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

__global__ void count_per_expert(const int* __restrict__ topk_idx,
                                 int* __restrict__ counts,
                                 int T, int K, int E) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int n = T * K;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        int e = topk_idx[i];
        assert(e >= 0 && e < E);
        atomicAdd(&counts[e], 1);
    }
}

__global__ void count_per_rank(const int* __restrict__ topk_idx,
                               const int* __restrict__ owner, // [E] → rank
                               int* __restrict__ counts_rank,  // [P]
                               int T, int K, int E, int P) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int n = T * K;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) {
        int e = topk_idx[i];
        assert(e >= 0 && e < E);
        int r = owner[e];
        assert(r >= 0 && r < P);
        atomicAdd(&counts_rank[r], 1);
    }
}

__global__ void exclusive_scan_int(const int* __restrict__ in,
                                   int* __restrict__ out,
                                   int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int acc = 0;
        out[0] = 0;
        for (int i = 0; i < n; ++i) {
            acc += in[i];
            out[i+1] = acc;
        }
    }
}

__global__ void zero_int(int* a, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) a[i] = 0;
}

__global__ void set_sendcount_single(const int* __restrict__ offsets_exp,
                                     int E, int D,
                                     unsigned long long* __restrict__ sendcounts){
    if (threadIdx.x==0 && blockIdx.x==0){
        sendcounts[0] = (unsigned long long)offsets_exp[E] * (unsigned long long)D;
    }
}

__global__ void set_sendcounts_from_rankcounts(const int* __restrict__ counts_rank,
                                               int64_t* __restrict__ sendcounts,
                                               int P, int D) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r < P) {
        long long rows = (long long)counts_rank[r];
        sendcounts[r] = rows * (long long)D; // 元素个数
    }
}

// 单卡 pack：与原版一致（每 tk 一个 block；thread0 分配游标，广播 row；D 维并行写）
__global__ void pack_kernel_single(const half* __restrict__ x,
                                   const int*  __restrict__ topk_idx,
                                   const float* __restrict__ alpha,
                                   const int*  __restrict__ offsets_exp,
                                   int*        __restrict__ cursors_exp,
                                   half*       __restrict__ sendbuf,
                                   int*        __restrict__ combine_idx,
                                   int*        __restrict__ expert_ids_packed,
                                   int T, int D, int K, int E) {
    int tk = blockIdx.x;
    if (tk >= T*K) return;
    int t = tk / K, k = tk % K;

    int e = topk_idx[(size_t)t*K + k];
    assert(e >= 0 && e < E);
    float a = alpha[(size_t)t*K + k];

    __shared__ int row_s;
    if (threadIdx.x == 0) {
        int pos = atomicAdd(&cursors_exp[e], 1);
        int lo  = offsets_exp[e];
        int hi  = offsets_exp[e+1];
        int row = lo + pos;
        assert(row >= lo && row < hi);
        row_s = row;
        combine_idx[row]       = t;
        expert_ids_packed[row] = e;
    }
    __syncthreads();

    int row = row_s;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xv = __half2float(x[(size_t)t*D + d]);
        sendbuf[(size_t)row*D + d] = __float2half(xv * a);
    }
}

// 多卡 pack：与单卡逻辑相同，但使用 rank_offsets / cursors_rank 到每个 rank 段中写入
__global__ void pack_kernel_multi(const half* __restrict__ x,
                                  const int*  __restrict__ topk_idx,
                                  const float* __restrict__ alpha,
                                  const int*  __restrict__ owner,        // [E] -> rank
                                  const int*  __restrict__ rank_offsets, // [P+1] 行前缀和
                                  int*        __restrict__ cursors_rank, // [P]
                                  half*       __restrict__ sendbuf,
                                  int*        __restrict__ combine_idx,
                                  int*        __restrict__ expert_ids_packed,
                                  int T, int D, int K, int E, int P) {
    int tk = blockIdx.x;
    if (tk >= T*K) return;
    int t = tk / K, k = tk % K;

    int e = topk_idx[(size_t)t*K + k];
    assert(e >= 0 && e < E);
    int r = owner[e];
    assert(r >= 0 && r < P);
    float a = alpha[(size_t)t*K + k];

    __shared__ int row_s;
    if (threadIdx.x == 0) {
        int pos = atomicAdd(&cursors_rank[r], 1);
        int lo  = rank_offsets[r];
        int hi  = rank_offsets[r+1];
        int row = lo + pos;
        assert(row >= lo && row < hi);
        row_s = row;
        combine_idx[row]       = t;
        expert_ids_packed[row] = e;
    }
    __syncthreads();

    int row = row_s;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xv = __half2float(x[(size_t)t*D + d]);
        sendbuf[(size_t)row*D + d] = __float2half(xv * a);
    }
}

extern "C" void pack_routes(
    const half*  x,
    const int*   topk_idx,
    const float* alpha,
    int T, int D, int K, int E,
    const int*   expert_owner,      // [E]，device
    float /*capacity_factor*/,
    int   ep_size,                  // P
    half* sendbuf,
    int64_t* sendcounts,            // [P]，元素数
    int*   combine_idx,             // [T*K]
    int*   expert_ids_packed,       // [T*K]
    int*   counts_per_expert,       // [E]
    int*   offsets_per_expert,      // [E+1]
    cudaStream_t stream)
{
    const int threads = 256;
    const int n_rows  = T * K;

    // 1) expert 直方图（单/多卡都保留，便于单卡 experts 前向）
    cudaMemsetAsync(counts_per_expert, 0, sizeof(int) * E, stream);
    {
        int blocks = CEIL_DIV(n_rows, threads);
        count_per_expert<<<blocks, threads, 0, stream>>>(
            topk_idx, counts_per_expert, T, K, E);
    }
    exclusive_scan_int<<<1,1,0,stream>>>(counts_per_expert, offsets_per_expert, E);

    if (ep_size <= 1) {
        // 单卡：沿用旧路径
        cudaMemsetAsync(sendcounts, 0, sizeof(int64_t), stream);

        // 游标为 expert 维度
        {
            int blocks_e = CEIL_DIV(E, threads);
            zero_int<<<blocks_e, threads, 0, stream>>>(counts_per_expert, E);
        }
        // sendcounts[0] = offsets[E] * D
        set_sendcount_single<<<1,1,0,stream>>>(
            offsets_per_expert, E, D,
            reinterpret_cast<unsigned long long*>(sendcounts));

        // pack
        int blocks_pack = n_rows;
        pack_kernel_single<<<blocks_pack, 128, 0, stream>>>(
            x, topk_idx, alpha,
            offsets_per_expert, counts_per_expert,
            sendbuf, combine_idx, expert_ids_packed,
            T, D, K, E);
        return;
    }

    // 多卡：按 rank 分桶
    // 2) rank 直方图（counts_rank[P]）
    int* counts_rank = nullptr;
    int* rank_offsets = nullptr;
    int* cursors_rank = nullptr;
    cudaMalloc(&counts_rank,  sizeof(int) * ep_size);
    cudaMalloc(&rank_offsets, sizeof(int) * (ep_size + 1));
    cudaMalloc(&cursors_rank, sizeof(int) * ep_size);
    cudaMemsetAsync(counts_rank,  0, sizeof(int) * ep_size, stream);
    cudaMemsetAsync(cursors_rank, 0, sizeof(int) * ep_size, stream);

    {
        int blocks = CEIL_DIV(n_rows, threads);
        count_per_rank<<<blocks, threads, 0, stream>>>(
            topk_idx, expert_owner, counts_rank, T, K, E, ep_size);
    }
    exclusive_scan_int<<<1,1,0,stream>>>(counts_rank, rank_offsets, ep_size);

    // 3) sendcounts[r] = counts_rank[r] * D（元素数）
    {
        int blocks = CEIL_DIV(ep_size, threads);
        set_sendcounts_from_rankcounts<<<blocks, threads, 0, stream>>>(
            counts_rank, sendcounts, ep_size, D);
    }

    // 4) pack 到每个 rank 的段
    {
        int blocks_pack = n_rows;
        pack_kernel_multi<<<blocks_pack, 128, 0, stream>>>(
            x, topk_idx, alpha,
            expert_owner, rank_offsets, cursors_rank,
            sendbuf, combine_idx, expert_ids_packed,
            T, D, K, E, ep_size);
    }

    cudaFree(counts_rank);
    cudaFree(rank_offsets);
    cudaFree(cursors_rank);
}
