// moe_fwd.cpp (FlashMoE-backed)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>

// FlashMoE headers
#include "../flashmoe/csrc/include/kleos/bootstrap.cuh"
#include "../flashmoe/csrc/include/kleos/moe/moe.cuh"

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)

// --------------------------------------------
// DS-MoE → FlashMoE 前向适配层
//  - 复用现有 moe_fwd 签名，内部打包为 FlashMoE 期望的连续缓冲
//  - 仅支持与 kleos::ACC 编译期配置一致的维度（SEQ_LEN、HIDDEN_SIZE、NUM_EXPERTS 等）
//  - Expert Parallel / 通信由 FlashMoE (NVSHMEM) 负责；ep_comm 参数被忽略
// --------------------------------------------
extern "C" void moe_fwd(
    // 输入
    const half* x,                         // [T,D]
    // Router
    const half* W_gate, const float* e_score_bias, // e_score_bias 未使用
    int T, int D, int E, int K,                     // K 未使用（FlashMoE 内部固定 top-k）
    int n_group, int topk_group,                    // 未使用
    int norm_topk_prob, float routed_scale,         // 未使用
    // Shared Expert (SE) - FlashMoE 不使用，保留形参以兼容签名
    const half* W1_se, const float* B1_se,
    const half* W2_se, const float* B2_se, int H_se,
    // Routed Experts
    const half* const* W1_local, const float* const* B1_local,
    const half* const* W2_local, const float* const* B2_local,
    int H,
    // Expert-Parallel
    const int* expert_owner, float capacity_factor,
    int ep_size, void* ep_comm,
    // 输出
    half* y,                               // [T,D]
    // 流
    cudaStream_t stream)
{
  using kleos::ACC;
  using Element = ACC::Element;

  // --------- 1) 参数合法性检查（仅支持与 ACC 编译期一致的维度） ---------
  const int S_expected  = ACC::S::value;
  const int H_expected  = ACC::H::value;
  const int E_expected  = ACC::E::value;
  const int PX_expected = ACC::PX::value;
  const int P_expected  = ACC::P::value; // inner dim (通常等于 D)

  if (T != S_expected || D != H_expected || E != E_expected) {
    std::cerr << "[moe_fwd] mismatched dims: got T="<<T<<" D="<<D<<" E="<<E
              <<", expect S="<<S_expected<<" H="<<H_expected<<" E="<<E_expected<<" (FlashMoE ACC)\n";
    std::exit(1);
  }

  // FlashMoE 仅在前向使用 half / bfloat16 等 ACC::Element；router bias 与 DS 的 top-k 相关参数被忽略。
  (void)e_score_bias; (void)K; (void)n_group; (void)topk_group; (void)norm_topk_prob; (void)routed_scale;
  (void)W1_se; (void)B1_se; (void)W2_se; (void)B2_se; (void)H_se;
  (void)expert_owner; (void)capacity_factor; (void)ep_size; (void)ep_comm;

  // --------- 2) 初始化 FlashMoE (NVSHMEM) ---------
  static bool kleos_initialized = false;
  if (!kleos_initialized) {
    kleos::initialize();
    kleos_initialized = true;
  }

  CUDA_CHECK(cudaSetDevice(kleos::getRank()));

  // 本地专家数量由 FlashMoE bootstrap 计算
  const uint32_t n_local = kleos::hostBookkeeping.nLx;

  // --------- 3) 计算缓冲大小并分配 ---------
  const size_t act_elems   = (size_t)S_expected * H_expected;
  const size_t gate_elems  = (size_t)PX_expected * H_expected;
  const size_t up_elems    = (size_t)n_local * P_expected * H_expected;
  const size_t down_elems  = (size_t)n_local * H_expected * P_expected;
  const size_t bias_up     = (size_t)n_local * P_expected;
  const size_t bias_down   = (size_t)n_local * H_expected;
  const size_t input_elems = act_elems + gate_elems + up_elems + down_elems + bias_up + bias_down;
  const size_t out_elems   = (size_t)S_expected * (PX_expected + H_expected); // gate + moe output

  Element* input_buf = nullptr;
  Element* output_buf = nullptr;
  CUDA_CHECK(cudaMallocAsync(&input_buf, input_elems * sizeof(Element), stream));
  CUDA_CHECK(cudaMallocAsync(&output_buf, out_elems * sizeof(Element), stream));
  CUDA_CHECK(cudaMemsetAsync(output_buf, 0, out_elems * sizeof(Element), stream));

  // --------- 4) 填充连续输入缓冲 ---------
  // 4.1 Activations [S,H]
  CUDA_CHECK(cudaMemcpyAsync(input_buf, x, act_elems * sizeof(Element), cudaMemcpyDeviceToDevice, stream));

  // 4.2 Gate weights [PX,H] （假设 PX == E，且 layout=PX major）
  Element* gate_ptr = input_buf + act_elems;
  CUDA_CHECK(cudaMemcpyAsync(gate_ptr, W_gate, gate_elems * sizeof(Element), cudaMemcpyDeviceToDevice, stream));

  // 4.3 Expert weights Up/Down & Bias
  Element* up_ptr   = gate_ptr + gate_elems;
  Element* down_ptr = up_ptr + up_elems;
  Element* bup_ptr  = down_ptr + down_elems;
  Element* bdn_ptr  = bup_ptr + bias_up;

  // 逐专家复制，按 FlashMoE 期望的 (local, P, H)/(local, H, P) layout 连续存放
  const uint32_t experts_to_copy = std::min<uint32_t>(n_local, static_cast<uint32_t>(E));
  const size_t up_stride   = (size_t)P_expected * H_expected;
  const size_t down_stride = (size_t)H_expected * P_expected;
  for (uint32_t e = 0; e < experts_to_copy; ++e) {
    const half* W1 = W1_local[e];
    const half* W2 = W2_local[e];
    const float* B1 = B1_local[e];
    const float* B2 = B2_local[e];

    // W1: [D,H] → [P,H]
    CUDA_CHECK(cudaMemcpyAsync(up_ptr + e * up_stride, W1,
                               up_stride * sizeof(Element), cudaMemcpyDeviceToDevice, stream));
    // W2: [H,D] → [H,P]
    CUDA_CHECK(cudaMemcpyAsync(down_ptr + e * down_stride, W2,
                               down_stride * sizeof(Element), cudaMemcpyDeviceToDevice, stream));

    // Bias up / down
    CUDA_CHECK(cudaMemcpyAsync(bup_ptr + e * P_expected, B1,
                               P_expected * sizeof(Element), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(bdn_ptr + e * H_expected, B2,
                               H_expected * sizeof(Element), cudaMemcpyDeviceToDevice, stream));
  }

  // --------- 5) 调用 FlashMoE fused kernel ---------
  kleos::moe::forwardHost(input_buf, output_buf);
  CUDA_CHECK(cudaGetLastError());

  // --------- 6) 提取 MoE 输出部分（output_buf 后半部分大小 S*H） ---------
  Element* moe_out = output_buf + (size_t)S_expected * PX_expected;
  CUDA_CHECK(cudaMemcpyAsync(y, moe_out, act_elems * sizeof(Element), cudaMemcpyDeviceToDevice, stream));

  // --------- 7) 资源回收 ---------
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaFreeAsync(input_buf, stream);
  cudaFreeAsync(output_buf, stream);
}
