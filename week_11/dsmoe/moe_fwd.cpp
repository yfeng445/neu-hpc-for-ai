// dsmoe/moe_fwd.cpp
#include "./moe_fwd.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <algorithm>

// FlashMoE / Kleos headers (provided by flashmoe/csrc/include)
#include <kleos/bootstrap.cuh>
#include <kleos/moe/moe.cuh>

// ---------------- logging & checks ----------------
#ifndef DSMOE_VERBOSE
#define DSMOE_VERBOSE 0
#endif
#define LOG(...) do { if (DSMOE_VERBOSE) std::fprintf(stderr, __VA_ARGS__); } while (0)

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = static_cast<cudaError_t>(expr);                          \
    if (_err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "[moe_fwd] CUDA error %s @ %s:%d\n",                 \
                   cudaGetErrorString(_err), __FILE__, __LINE__);               \
      std::abort();                                                             \
    }                                                                           \
  } while (0)

// ---------------- optional helpers ----------------
namespace detail {

// 将 float bias 转为 Kleos 的 Element（通常是 __half）。若无法判断类型，则置零兜底。
template <typename Element>
__global__ void cast_bias_f32_to_elem(const float* __restrict__ src,
                                      Element* __restrict__ dst,
                                      int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
#if __cplusplus >= 201703L
  if constexpr (sizeof(Element) == 2) {
    // 假定 Element=__half 或 bf16；此处用 __half 变换作为常用路径
    dst[i] = __half2half_rn(__float2half(src[i]));
  } else {
    // 其他未知类型，置零兜底
    dst[i] = Element{};
  }
#else
  // C++14 及以下：直接按 __half 转换（常见配置）
  dst[i] = __half2half_rn(__float2half(src[i]));
#endif
}

template <typename Element>
void copy_or_cast_bias(const float* d_src_f32, Element* d_dst_elem,
                       int n, cudaStream_t stream) {
  if (!d_src_f32) {
    CUDA_CHECK(cudaMemsetAsync(d_dst_elem, 0, n * sizeof(Element), stream));
    return;
  }
  // 常见地 Kleos::ACC::Element 为 2 字节；尝试 kernel 转换
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  cast_bias_f32_to_elem<Element><<<blocks, threads, 0, stream>>>(d_src_f32, d_dst_elem, n);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace detail

// ---------------- DS-MoE → FlashMoE 前向桥接 ----------------
void moe_fwd(
    // 输入
    half*  x,                 // [T, D]
    // Router
    half*  W_gate, float* e_score_bias, // e_score_bias 未使用
    int    T, int D, int E, int K,
    int    n_group, int topk_group,
    int    norm_topk_prob, float routed_scale,
    // Shared Expert (SE) —— 本桥接未使用
    half*  W1_se, float* B1_se, half* W2_se, float* B2_se, int H_se,
    // Routed Experts（每个元素为 device 指针的数组；数组本体在 host 或 device 均可）
    half**  W1_local, float** B1_local,
    half**  W2_local, float** B2_local,
    int     H,
    // Expert-Parallel（保留签名，未使用）
    int*    expert_owner, float capacity_factor, int ep_size, void* ep_comm,
    // 输出
    half*   y,                // [T, D]
    cudaStream_t stream)
{
  using kleos::ACC;
  using Element = ACC::Element;     // Kleos 的标量类型（通常为 __half / bf16）
  (void)e_score_bias; (void)K; (void)n_group; (void)topk_group;
  (void)norm_topk_prob; (void)routed_scale;
  (void)W1_se; (void)B1_se; (void)W2_se; (void)B2_se; (void)H_se;
  (void)expert_owner; (void)capacity_factor; (void)ep_size; (void)ep_comm;

  // 1) 维度一致性（需与 Kleos 编译期 ACC::S/H/E/PX/P 定义一致）
  const int S_expected  = ACC::S::value;   // seq_len
  const int H_expected  = ACC::H::value;   // hidden (model dim)
  const int E_expected  = ACC::E::value;   // num experts (global)
  const int PX_expected = ACC::PX::value;  // gating projection dim（通常与 E 相等）
  const int P_expected  = ACC::P::value;   // inner (FFN) dim

  if (T != S_expected || D != H_expected || E != E_expected) {
    std::fprintf(stderr,
      "[moe_fwd] mismatched dims: got T=%d D=%d E=%d, expect S=%d H=%d E=%d (Kleos ACC)\n",
      T, D, E, S_expected, H_expected, E_expected);
    std::abort();
  }
  if (H != P_expected) {
    std::fprintf(stderr,
      "[moe_fwd] mismatched FFN inner dim: H=%d vs ACC::P=%d\n", H, P_expected);
    std::abort();
  }
  if (PX_expected != E_expected) {
    std::fprintf(stderr,
      "[moe_fwd] gating PX (%d) != experts E (%d) — layout mismatch.\n",
      PX_expected, E_expected);
    std::abort();
  }

  // 2) 初始化 Kleos / NVSHMEM（仅一次）
  static bool kleos_initialized = false;
  if (!kleos_initialized) {
    kleos::initialize();              // 内部做 NVSHMEM init 与 rank bookkeeping
    kleos_initialized = true;
  }
  CUDA_CHECK(cudaSetDevice(kleos::getRank()));

  // 本地专家数量（按 Kleos 分布）
  const uint32_t n_local = kleos::hostBookkeeping.nLx;

  // 3) 计算并分配连续输入/输出缓冲
  const size_t act_elems   = static_cast<size_t>(S_expected) * H_expected;       // [S,H]
  const size_t gate_elems  = static_cast<size_t>(PX_expected) * H_expected;      // [PX,H]
  const size_t up_elems    = static_cast<size_t>(n_local) * P_expected * H_expected; // [nL, P, H]
  const size_t down_elems  = static_cast<size_t>(n_local) * H_expected * P_expected; // [nL, H, P]
  const size_t bias_up     = static_cast<size_t>(n_local) * P_expected;          // [nL, P]
  const size_t bias_down   = static_cast<size_t>(n_local) * H_expected;          // [nL, H]
  const size_t input_elems = act_elems + gate_elems + up_elems + down_elems + bias_up + bias_down;

  const size_t out_elems   = static_cast<size_t>(S_expected) * (PX_expected + H_expected);
  // output_buf = [S, PX] (router logits) || [S, H] (final MoE output)

  Element* input_buf  = nullptr;
  Element* output_buf = nullptr;
  CUDA_CHECK(cudaMallocAsync(&input_buf,  input_elems * sizeof(Element), stream));
  CUDA_CHECK(cudaMallocAsync(&output_buf, out_elems   * sizeof(Element), stream));
  CUDA_CHECK(cudaMemsetAsync(output_buf, 0, out_elems * sizeof(Element), stream));

  // 4) 填充连续输入缓冲（layout 与 Kleos 统一）
  // 4.1 activations [S,H]
  CUDA_CHECK(cudaMemcpyAsync(
      input_buf,
      x,
      act_elems * sizeof(Element),
      cudaMemcpyDeviceToDevice, stream));

  // 4.2 gate weights [PX,H] —— 假设 W_gate layout 已与 Kleos 一致（PX major）
  Element* gate_ptr = input_buf + act_elems;
  CUDA_CHECK(cudaMemcpyAsync(
      gate_ptr,
      W_gate,
      gate_elems * sizeof(Element),
      cudaMemcpyDeviceToDevice, stream));

  // 4.3 experts: up / down / bias
  Element* up_ptr   = gate_ptr + gate_elems;
  Element* down_ptr = up_ptr   + up_elems;
  Element* bup_ptr  = down_ptr + down_elems;
  Element* bdn_ptr  = bup_ptr  + bias_up;

  const uint32_t experts_to_copy = std::min<uint32_t>(n_local, static_cast<uint32_t>(E_expected));
  const size_t up_stride   = static_cast<size_t>(P_expected) * H_expected;  // [P,H]
  const size_t down_stride = static_cast<size_t>(H_expected) * P_expected;  // [H,P]

  for (uint32_t e = 0; e < experts_to_copy; ++e) {
    half*  W1 = W1_local ? W1_local[e] : nullptr; // [P,H]（按桥接约定）
    half*  W2 = W2_local ? W2_local[e] : nullptr; // [H,P]
    float* B1 = B1_local ? B1_local[e] : nullptr; // [P]
    float* B2 = B2_local ? B2_local[e] : nullptr; // [H]

    // 权重按 Element 拷贝（通常 __half）
    if (W1) CUDA_CHECK(cudaMemcpyAsync(up_ptr   + e * up_stride,
                                       W1, up_stride   * sizeof(Element),
                                       cudaMemcpyDeviceToDevice, stream));
    else    CUDA_CHECK(cudaMemsetAsync(up_ptr   + e * up_stride,   0, up_stride   * sizeof(Element), stream));

    if (W2) CUDA_CHECK(cudaMemcpyAsync(down_ptr + e * down_stride,
                                       W2, down_stride * sizeof(Element),
                                       cudaMemcpyDeviceToDevice, stream));
    else    CUDA_CHECK(cudaMemsetAsync(down_ptr + e * down_stride, 0, down_stride * sizeof(Element), stream));

    // bias：float → Element（半精度等）；若无法转换则置零兜底
    detail::copy_or_cast_bias<Element>(B1, bup_ptr + e * P_expected,  P_expected,  stream);
    detail::copy_or_cast_bias<Element>(B2, bdn_ptr + e * H_expected,  H_expected,  stream);
  }

  // 5) 调用 Kleos 融合前向
  kleos::moe::forwardHost(input_buf, output_buf);
  CUDA_CHECK(cudaGetLastError());

  // 6) 提取 MoE 输出（output_buf 后半部分为 [S,H]）
  Element* moe_out = output_buf + static_cast<size_t>(S_expected) * PX_expected;
  CUDA_CHECK(cudaMemcpyAsync(
      y,
      moe_out,
      act_elems * sizeof(Element),
      cudaMemcpyDeviceToDevice, stream));

  // 7) 清理
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaFreeAsync(input_buf,  stream);
  cudaFreeAsync(output_buf, stream);
}
