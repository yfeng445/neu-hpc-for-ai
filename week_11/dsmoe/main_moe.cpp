// dsmoe/main_moe.cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// 编译期维度（来自 FlashMoE）
#include <kleos/bootstrap.cuh>

// 适配层入口
#include "moe_fwd.h"

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)

static int env_int(const char* k, int defv){
  const char* v = std::getenv(k);
  if(!v) return defv;
  try { return std::stoi(std::string(v)); } catch(...) { return defv; }
}

static void fill_half_from_float(const std::vector<float>& src, std::vector<half>& dst){
  dst.resize(src.size());
  for (size_t i=0;i<src.size();++i) dst[i] = __float2half(src[i]);
}

int main() {
  using kleos::ACC;

  // 编译期维度
  const int T  = ACC::S::value;   // sequence length
  const int D  = ACC::H::value;   // hidden size (model dim)
  const int E  = ACC::E::value;   // number of experts (global)
  const int P  = ACC::P::value;   // FFN inner dim
  const int PX = ACC::PX::value;  // gating projection (通常等于 E)

  // 设备选择：默认 LOCAL_RANK 或 0
  CUDA_CHECK(cudaSetDevice(env_int("LOCAL_RANK", 0)));

  // Host 侧随机初始化
  std::mt19937 rng(2025);
  std::uniform_real_distribution<float> dx(-1.f, 1.f);
  std::uniform_real_distribution<float> dw(-0.05f, 0.05f);

  // 输入 activations: [T, D]
  std::vector<float> hX_f((size_t)T * D);
  for (auto& v : hX_f) v = dx(rng);
  std::vector<half> hX_h; fill_half_from_float(hX_f, hX_h);

  // Router 权重: FlashMoE 期望 [PX, H]，此处 H==D，PX==E
  // 我们直接按 [E, D] 线性存放
  std::vector<float> hWgate_f((size_t)PX * D);
  for (auto& v : hWgate_f) v = dw(rng);
  std::vector<half> hWgate_h; fill_half_from_float(hWgate_f, hWgate_h);
  std::vector<float> hBias((size_t)E, 0.f);  // 未使用，但按接口保留

  // 专家参数：
  // W1: [P, H(=D)], W2: [H(=D), P], B1: [P], B2: [H(=D)]
  std::vector<float> W1_f((size_t)P * D), W2_f((size_t)D * P);
  for (auto& v : W1_f) v = dw(rng);
  for (auto& v : W2_f) v = dw(rng);
  std::vector<half>  W1_h, W2_h; fill_half_from_float(W1_f, W1_h); fill_half_from_float(W2_f, W2_h);
  std::vector<float> B1_f((size_t)P, 0.02f), B2_f((size_t)D, 0.05f);

  // Device 侧分配与拷贝
  half  *dX=nullptr, *dY=nullptr, *dWgate=nullptr, *dW1=nullptr, *dW2=nullptr;
  float *dBias=nullptr, *dB1=nullptr, *dB2=nullptr;
  CUDA_CHECK(cudaMalloc(&dX,     (size_t)T * D * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dY,     (size_t)T * D * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dWgate, (size_t)PX * D * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dBias,  (size_t)E * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dW1,    (size_t)P * D * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dW2,    (size_t)D * P * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB1,    (size_t)P * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB2,    (size_t)D * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dX,     hX_h.data(),     (size_t)T * D * sizeof(half),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWgate, hWgate_h.data(), (size_t)PX * D * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBias,  hBias.data(),    (size_t)E * sizeof(float),     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW1,    W1_h.data(),     (size_t)P * D * sizeof(half),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW2,    W2_h.data(),     (size_t)D * P * sizeof(half),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB1,    B1_f.data(),     (size_t)P * sizeof(float),     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB2,    B2_f.data(),     (size_t)D * sizeof(float),     cudaMemcpyHostToDevice));

  // 专家参数指针数组（长度 E；此处所有专家共享同一套参数，便于 smoke test）
  std::vector<half*>  hW1_local(E, dW1), hW2_local(E, dW2);
  std::vector<float*> hB1_local(E, dB1), hB2_local(E, dB2);

  // 其余接口参数（当前桥接未使用，可传 nullptr/占位）
  int* expert_owner = nullptr;
  const int  K              = 1;
  const int  n_group        = 1;
  const int  topk_group     = 1;
  const int  norm_topk_prob = 0;
  const float routed_scale  = 1.f;
  const int  H_se           = 0;
  const float capacity_factor = 1.f;
  const int  ep_size        = 1;
  void*      ep_comm        = nullptr;
  cudaStream_t stream       = nullptr;

  // 调用适配层（FlashMoE 前向）
  moe_fwd(
      /*x*/ dX,
      /*router*/ dWgate, dBias,
      /*shape*/ T, D, E, K,
      /*group route*/ n_group, topk_group,
      /*topk norm/scale*/ norm_topk_prob, routed_scale,
      /*SE*/ nullptr, nullptr, nullptr, nullptr, H_se,
      /*experts*/ hW1_local.data(), hB1_local.data(),
                  hW2_local.data(), hB2_local.data(), P,
      /*EP*/ expert_owner, capacity_factor,
      /*ep_size*/ ep_size, /*ep_comm*/ ep_comm,
      /*out*/ dY,
      /*stream*/ stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 打印首 8 个输出
  std::vector<half> hY((size_t)T * D);
  CUDA_CHECK(cudaMemcpy(hY.data(), dY, (size_t)T * D * sizeof(half), cudaMemcpyDeviceToHost));
  std::cout << "[FlashMoE] y[0,0:8]: ";
  for (int i = 0; i < std::min(8, D); ++i) std::cout << __half2float(hY[i]) << " ";
  std::cout << "\n";

  // 资源释放
  cudaFree(dX); cudaFree(dY); cudaFree(dWgate); cudaFree(dBias);
  cudaFree(dW1); cudaFree(dW2); cudaFree(dB1); cudaFree(dB2);

  return 0;
}
