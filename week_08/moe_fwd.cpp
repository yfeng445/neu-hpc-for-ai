// moe_fwd.cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

#include "comm/ep_comm.h"  // EpComm* & ep_* API（NCCL/MPI/stub）

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)

extern "C" void router_forward(
    const half*  x, const half*  W_gate, const float* e_score_bias,
    int T, int D, int E, int K, int n_group, int topk_group,
    int  norm_topk_prob, float routed_scale,
    int*   topk_idx, float* alpha,
    float* logits, float* p, unsigned char* group_mask,
    cudaStream_t stream);

extern "C" void pack_routes(
    const half*  x, const int*   topk_idx, const float* alpha,
    int T, int D, int K, int E,
    const int* expert_owner, float capacity_factor, int ep_size,
    half* sendbuf, int64_t* sendcounts,
    int*   combine_idx, int*   expert_ids_packed,
    int*   counts_per_expert, int* offsets_per_expert,
    cudaStream_t stream);

extern "C" void experts_mlp_forward(
    const half* inbuf, half* outbuf, int D, int H,
    int n_local, const int*  local_offsets_host, // host pointer
    const half* const* W1_local, const float* const* B1_local,
    const half* const* W2_local, const float* const* B2_local,
    int activation, cudaStream_t stream);

extern "C" void combine_and_add(
    const half* contrib, const int*  combine_idx, int rows,
    half* y_re, const half* y_se, half* y,
    int T, int D, cudaStream_t stream);

extern "C" void se_mlp_forward(
    const half* x, half* y_se,
    const half* W1_se, const float* B1_se,
    const half* W2_se, const float* B2_se,
    int T, int D, int H_se, int activation, cudaStream_t stream);

// --------------------------------------------
// 前向：workspace 分配 → 路由/打包 → EP 通信 → Experts → 回传 → 合并
// --------------------------------------------
extern "C" void moe_fwd(
    // 输入
    const half* x,                         // [T,D]
    // Router
    const half* W_gate,                    // [D,E]（按 (d*E+e)）
    const float* e_score_bias,             // [E]
    int T, int D, int E, int K,
    int n_group, int topk_group,
    int norm_topk_prob, float routed_scale,
    // Shared Expert (SE)
    const half* W1_se, const float* B1_se, // [D,H_se], [H_se]
    const half* W2_se, const float* B2_se, // [H_se,D], [D]
    int H_se,
    // Routed Experts
    const half* const* W1_local, const float* const* B1_local, // each: [D,H]
    const half* const* W2_local, const float* const* B2_local, // each: [H,D]
    int H,
    // Expert-Parallel
    const int* expert_owner,               // [E], expert -> rank
    float capacity_factor,
    int   ep_size,
    void* ep_comm,                         // EpComm*，可为 nullptr
    // 输出
    half* y,                               // [T,D]
    // 流
    cudaStream_t stream)
{
  // ---------- 分配工作区（发送端上界即可，接收端延后按 counts 精确分配） ----------
  float *logits=nullptr, *p=nullptr;
  unsigned char* group_mask=nullptr;
  int *topk_idx=nullptr; float *alpha=nullptr;
  half *y_se=nullptr, *y_re=nullptr;

  half *sendbuf=nullptr, *recvbuf=nullptr;     // recvbuf 延后按 counts 精确分配
  int64_t *sendcounts=nullptr, *recvcounts=nullptr;

  int *combine_idx=nullptr, *expert_ids_packed=nullptr;
  int *counts_per_expert=nullptr, *offsets_per_expert=nullptr;

  CUDA_CHECK(cudaMalloc(&logits, (size_t)T*E*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&p,      (size_t)T*E*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&group_mask, (size_t)T*n_group*sizeof(unsigned char)));
  CUDA_CHECK(cudaMalloc(&topk_idx, (size_t)T*K*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&alpha,    (size_t)T*K*sizeof(float)));

  CUDA_CHECK(cudaMalloc(&y_se, (size_t)T*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&y_re, (size_t)T*D*sizeof(half)));

  // 发送缓冲可用上界 T*K*D
  CUDA_CHECK(cudaMalloc(&sendbuf, (size_t)T*K*D*sizeof(half)));
  // 接收缓冲 recvbuf 先置空，待 allgather counts 后精确分配
  recvbuf = nullptr;

  CUDA_CHECK(cudaMalloc(&sendcounts, (size_t)ep_size * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&recvcounts, (size_t)ep_size * sizeof(int64_t)));

  CUDA_CHECK(cudaMalloc(&combine_idx,       (size_t)T*K*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&expert_ids_packed, (size_t)T*K*sizeof(int)));

  CUDA_CHECK(cudaMalloc(&counts_per_expert,  E*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&offsets_per_expert, (size_t)(E+1)*sizeof(int)));

  // ---------- 1) Router ----------
  router_forward(x, W_gate, e_score_bias,
                 T, D, E, K, n_group, topk_group,
                 norm_topk_prob, routed_scale,
                 topk_idx, alpha,
                 logits, p, group_mask, stream);
  CUDA_CHECK(cudaGetLastError());

  // ---------- 2) Shared Experts ----------
  se_mlp_forward(x, y_se, W1_se, B1_se, W2_se, B2_se, T, D, H_se, /*SiLU*/0, stream);
  CUDA_CHECK(cudaGetLastError());

  // ---------- 3) Pack routes（乘α + 构建 combine_idx / expert_ids） ----------
  CUDA_CHECK(cudaMemsetAsync(combine_idx,       0xFF, (size_t)T*K*sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(expert_ids_packed, 0xFF, (size_t)T*K*sizeof(int), stream));
  pack_routes(x, topk_idx, alpha, T, D, K, E,
              expert_owner, capacity_factor, ep_size,
              sendbuf, sendcounts, combine_idx, expert_ids_packed,
              counts_per_expert, offsets_per_expert, stream);
  CUDA_CHECK(cudaGetLastError());

  // ---------- 4) Expert-Parallel 通讯 + Experts 前向 ----------
  if (ep_size <= 1 || ep_comm == nullptr) {
    // 单卡：直接在本地 experts 前向
    std::vector<int> local_offsets_host(E+1, 0);
    CUDA_CHECK(cudaMemcpy(local_offsets_host.data(), offsets_per_expert,
                          sizeof(int)*(E+1), cudaMemcpyDeviceToHost));

    experts_mlp_forward(sendbuf, sendbuf, D, H,
                        /*n_local=*/E, local_offsets_host.data(),
                        W1_local, B1_local, W2_local, B2_local,
                        /*SiLU*/0, stream);
    CUDA_CHECK(cudaGetLastError());
  } else {
    // 多卡：Alltoallv（三路）
    EpComm* comm = reinterpret_cast<EpComm*>(ep_comm);

    // 4.1 交换激活元素计数（单位=half 元素 = 行数*D）
    ep_allgather_counts(sendcounts, recvcounts, comm, stream);

    // 4.2 host 侧统计 elem / row 总数，用于精确分配接收缓冲
    std::vector<int64_t> h_send_elems(ep_size,0), h_recv_elems(ep_size,0);
    CUDA_CHECK(cudaMemcpyAsync(h_send_elems.data(), sendcounts, sizeof(int64_t)*ep_size,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_recv_elems.data(), recvcounts, sizeof(int64_t)*ep_size,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // rows_count = elem_count / D
    std::vector<int64_t> h_send_rows(ep_size,0), h_recv_rows(ep_size,0);
    for (int r=0;r<ep_size;++r) {
      if (h_send_elems[r] % D != 0 || h_recv_elems[r] % D != 0) {
        std::cerr<<"[moe_fwd] counts not divisible by D, send["<<r<<"]="<<h_send_elems[r]
                 <<", recv["<<r<<"]="<<h_recv_elems[r]<<", D="<<D<<"\n";
        std::exit(1);
      }
      h_send_rows[r] = h_send_elems[r] / D;
      h_recv_rows[r] = h_recv_elems[r] / D;
    }

    // 精确分配激活接收缓冲（half）
    int64_t recv_elems_total = std::accumulate(h_recv_elems.begin(), h_recv_elems.end(), (int64_t)0);
    size_t  recv_bytes       = (size_t)recv_elems_total * sizeof(half);
    if (recvbuf) CUDA_CHECK(cudaFree(recvbuf));
    if (recv_bytes > 0) CUDA_CHECK(cudaMalloc(&recvbuf, recv_bytes));
    else                recvbuf = nullptr;

    // device 版 rows counts（给元数据 alltoallv 用）
    int64_t *d_send_rows=nullptr, *d_recv_rows=nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_rows, (size_t)ep_size*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_recv_rows, (size_t)ep_size*sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_send_rows, h_send_rows.data(), (size_t)ep_size*sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_recv_rows, h_recv_rows.data(), (size_t)ep_size*sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 4.3 三路 all-to-allv：activations / expert_ids / combine_idx
    // (1) 激活：单位=half 元素
    ep_alltoallv((const void*)sendbuf, sendcounts,
                 (void*)recvbuf, recvcounts,
                 (int)sizeof(half), comm, stream);

    // 元数据接收缓冲的精确分配（单位=行）
    int64_t recv_rows_total = std::accumulate(h_recv_rows.begin(), h_recv_rows.end(), (int64_t)0);
    int *expert_ids_recv   = nullptr;
    int *combine_idx_recv  = nullptr;
    if (recv_rows_total > 0) {
      CUDA_CHECK(cudaMalloc(&expert_ids_recv,  (size_t)recv_rows_total*sizeof(int)));
      CUDA_CHECK(cudaMalloc(&combine_idx_recv, (size_t)recv_rows_total*sizeof(int)));
    }

    // (2) expert_ids：单位=行
    ep_alltoallv((const void*)expert_ids_packed, d_send_rows,
                 (void*)expert_ids_recv, d_recv_rows,
                 (int)sizeof(int), comm, stream);

    // (3) combine_idx：单位=行
    ep_alltoallv((const void*)combine_idx, d_send_rows,
                 (void*)combine_idx_recv, d_recv_rows,
                 (int)sizeof(int), comm, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 4.4 在接收端构造本地 offsets（host 直方图 expert_ids）
    std::vector<int> h_eids(recv_rows_total, 0);
    if (recv_rows_total > 0) {
      CUDA_CHECK(cudaMemcpyAsync(h_eids.data(), expert_ids_recv,
                                 (size_t)recv_rows_total*sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<int> local_counts(E,0);
    for (int64_t i=0;i<recv_rows_total;++i) {
      int e = h_eids[i];
      if ((unsigned)e >= (unsigned)E) {
        std::cerr<<"[moe_fwd] expert_ids_recv["<<i<<"]="<<e<<" out of [0,"<<E<<")\n";
        std::exit(1);
      }
      ++local_counts[e];
    }
    std::vector<int> local_offsets_host(E+1,0);
    for (int e=0;e<E;++e) local_offsets_host[e+1] = local_offsets_host[e] + local_counts[e];

    // 4.5 Experts 前向（就地写 recvbuf）
    experts_mlp_forward(recvbuf, recvbuf, D, H,
                        /*n_local=*/E, local_offsets_host.data(),
                        W1_local, B1_local, W2_local, B2_local,
                        /*SiLU*/0, stream);
    CUDA_CHECK(cudaGetLastError());

    // 4.6 逆向 all-to-allv：把专家输出按原顺序发回源 rank
    // 计数对称：发回激活用 recvcounts / sendcounts 互换
    ep_alltoallv((const void*)recvbuf, recvcounts,
                 (void*)sendbuf, sendcounts,
                 (int)sizeof(half), comm, stream);

    // 清理临时
    if (d_send_rows) cudaFree(d_send_rows);
    if (d_recv_rows) cudaFree(d_recv_rows);
    if (expert_ids_recv)  cudaFree(expert_ids_recv);
    if (combine_idx_recv) cudaFree(combine_idx_recv);
    if (recvbuf)          cudaFree(recvbuf), recvbuf=nullptr; // 用完即可释放，回传后结果在 sendbuf
  }

  // ---------- 5) Combine & add SE ----------
  // 回传后，贡献向量已在 sendbuf，顺序对应本 rank 的原始 rows（T*K）
  const int rows_host = T*K; // 本实现无丢弃（capacity_factor=1）
  combine_and_add(sendbuf, combine_idx, rows_host, y_re, y_se, y, T, D, stream);
  CUDA_CHECK(cudaGetLastError());

  // ---------- 释放 ----------
  cudaFree(logits); cudaFree(p); cudaFree(group_mask);
  cudaFree(topk_idx); cudaFree(alpha);
  cudaFree(y_se); cudaFree(y_re);
  cudaFree(sendbuf);
  cudaFree(sendcounts); cudaFree(recvcounts);
  cudaFree(combine_idx); cudaFree(expert_ids_packed);
  cudaFree(counts_per_expert); cudaFree(offsets_per_expert);
}
