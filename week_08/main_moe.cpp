// main_moe.cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <algorithm>

#include "comm/ep_comm.h"   // EpComm + ep_* API（NCCL/MPI/stub 的统一头）

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

// --------- 外部算子接口（与工程内已有签名保持一致） ---------
extern "C" void router_forward(
    const half*  x, const half*  W_gate, const float* e_score_bias,
    int T, int D, int E, int K, int n_group, int topk_group,
    int  norm_topk_prob, float routed_scale,
    int*   topk_idx, float* alpha,
    float* logits, float* p, unsigned char* group_mask,
    cudaStream_t stream);

extern "C" void moe_fwd(
    const half* x,
    const half* W_gate, const float* e_score_bias,
    int T, int D, int E, int K,
    int n_group, int topk_group,
    int norm_topk_prob, float routed_scale,
    const half* W1_se, const float* B1_se,
    const half* W2_se, const float* B2_se, int H_se,
    const half* const* W1_local, const float* const* B1_local,
    const half* const* W2_local, const float* const* B2_local,
    int H,
    const int* expert_owner, float capacity_factor,
    int ep_size, void* ep_comm,
    half* y,
    cudaStream_t stream);

int main(){
#ifdef _WIN32
  _putenv_s("CUDA_LAUNCH_BLOCKING","1");
#endif

  const int T_glob=32, D=64, E=16, K=3, n_group=4, topk_group=2;
  const int H=128, H_se=128;
  const int norm_topk_prob=1; const float routed_scale=1.f;
  const float capacity_factor=1.0f;


  int world_size = env_int("WORLD_SIZE", 1);
  int rank       = env_int("RANK", 0);
  int local_rank = env_int("LOCAL_RANK", rank);
  CUDA_CHECK(cudaSetDevice(local_rank));
  std::cerr << "[init] NCCL world="<<world_size<<" rank="<<rank<<" local_rank="<<local_rank<<"\n";


  auto split_T = [](int T, int ws, int rk){
    int base = T / ws, rem = T % ws;
    int tloc = base + (rk < rem ? 1 : 0);
    int off  = rk * base + std::min(rk, rem);
    return std::pair<int,int>(tloc, off);
  };
  auto [T_local, T_offset] = split_T(T_glob, world_size, rank);


  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dx(-1.f,1.f);
  std::uniform_real_distribution<float> dw(-0.05f,0.05f);


  std::vector<float> hX_glob((size_t)T_glob*D);
  for (auto& v: hX_glob) v = dx(rng);
  std::vector<float> hX_local((size_t)T_local*D);
  for (int t=0; t<T_local; ++t){
    std::copy_n(&hX_glob[(size_t)(T_offset + t)*D], D, &hX_local[(size_t)t*D]);
  }
  std::vector<half> hX_h; fill_half_from_float(hX_local, hX_h);

 
  std::vector<float> hWgate_py((size_t)E*D);
  for (auto& v: hWgate_py) v = dw(rng);
  std::vector<half> hWgate_cu((size_t)D*E);
  for (int e=0;e<E;++e) for (int d=0; d<D; ++d)
    hWgate_cu[(size_t)d*E + e] = __float2half(hWgate_py[(size_t)e*D + d]);
  std::vector<float> hBias(E, 0.f);


  std::vector<float> W1_se_f((size_t)D*H_se), W2_se_f((size_t)H_se*D);
  for (auto& v: W1_se_f) v = dw(rng);
  for (auto& v: W2_se_f) v = dw(rng);
  std::vector<half>  W1_se_h, W2_se_h;
  fill_half_from_float(W1_se_f, W1_se_h);
  fill_half_from_float(W2_se_f, W2_se_h);
  std::vector<float> B1_se_f(H_se, 0.01f), B2_se_f(D, 0.05f);

  std::vector<float> W1_e_f((size_t)D*H), W2_e_f((size_t)H*D);
  for (auto& v: W1_e_f) v = dw(rng);
  for (auto& v: W2_e_f) v = dw(rng);
  std::vector<half>  W1_e_h, W2_e_h;
  fill_half_from_float(W1_e_f, W1_e_h);
  fill_half_from_float(W2_e_f, W2_e_h);
  std::vector<float> B1_e_f(H, 0.02f), B2_e_f(D, 0.10f);


  std::vector<int> owner_h(E, 0);
  for (int e=0; e<E; ++e) owner_h[e] = (world_size>0) ? (e % world_size) : 0;

  // ----------------- Device 侧拷贝 -----------------
  half *dX=nullptr,*dY=nullptr,*dWgate=nullptr;
  float *dBias=nullptr;
  CUDA_CHECK(cudaMalloc(&dX, (size_t)T_local*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)T_local*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dWgate, (size_t)D*E*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dBias, E*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dX, hX_h.data(), (size_t)T_local*D*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWgate, hWgate_cu.data(), (size_t)D*E*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), E*sizeof(float), cudaMemcpyHostToDevice));

  half *dW1se=nullptr,*dW2se=nullptr; float *dB1se=nullptr,*dB2se=nullptr;
  CUDA_CHECK(cudaMalloc(&dW1se, (size_t)D*H_se*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dW2se, (size_t)H_se*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB1se, H_se*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB2se, D*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dW1se, W1_se_h.data(), (size_t)D*H_se*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW2se, W2_se_h.data(), (size_t)H_se*D*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB1se, B1_se_f.data(), H_se*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB2se, B2_se_f.data(), D*sizeof(float), cudaMemcpyHostToDevice));

  half *dW1e=nullptr,*dW2e=nullptr; float *dB1e=nullptr,*dB2e=nullptr;
  CUDA_CHECK(cudaMalloc(&dW1e,(size_t)D*H*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dW2e,(size_t)H*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB1e,H*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB2e,D*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dW1e, W1_e_h.data(), (size_t)D*H*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW2e, W2_e_h.data(), (size_t)H*D*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB1e, B1_e_f.data(), H*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB2e, B2_e_f.data(), D*sizeof(float), cudaMemcpyHostToDevice));

  // 指针数组（所有专家共用同一套参数）
  std::vector<const half*>  hW1_local(E, dW1e), hW2_local(E, dW2e);
  std::vector<const float*> hB1_local(E, dB1e), hB2_local(E, dB2e);

  int *dOwner=nullptr;
  CUDA_CHECK(cudaMalloc(&dOwner, E*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dOwner, owner_h.data(), E*sizeof(int), cudaMemcpyHostToDevice));

  // ----------------- 通信句柄 -----------------
  EpComm* ep = nullptr;
  if (world_size > 1) {
    ep_init_nccl(&ep, world_size, rank);
  } else {
    std::cerr << "[init] single-process (stub comm)\n";
  }

  cudaStream_t stream = 0;

  // ----------------- (可选) 路由单测 -----------------
  {
    float *logits=nullptr, *p=nullptr; unsigned char *gmask=nullptr;
    int *topk_idx=nullptr; float *alpha=nullptr;
    CUDA_CHECK(cudaMalloc(&logits, (size_t)T_local*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p,      (size_t)T_local*E*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gmask,  (size_t)T_local*n_group*sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&topk_idx,(size_t)T_local*K*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&alpha,   (size_t)T_local*K*sizeof(float)));

    router_forward(dX, dWgate, dBias,
                   T_local, D, E, K, n_group, topk_group,
                   norm_topk_prob, routed_scale,
                   topk_idx, alpha, logits, p, gmask, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int>   hIdx(T_local*K);
    std::vector<float> hAlp(T_local*K);
    CUDA_CHECK(cudaMemcpy(hIdx.data(), topk_idx, (size_t)T_local*K*sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hAlp.data(), alpha,    (size_t)T_local*K*sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "[OK] router_forward\n";
    std::cout << "  t0 idx: "; for(int k=0;k<K;++k) std::cout<<hIdx[k]<<" "; std::cout<<"\n";
    std::cout << "  t0 alp: "; for(int k=0;k<K;++k) std::cout<<hAlp[k]<<" "; std::cout<<"\n";

    cudaFree(logits); cudaFree(p); cudaFree(gmask); cudaFree(topk_idx); cudaFree(alpha);
  }

  // ----------------- MoE 前向（完整流水线） -----------------
  moe_fwd(
    /*x*/ dX,
    /*router*/ dWgate, dBias,
    /*shape*/ T_local, D, E, K,
    /*group route*/ n_group, topk_group,
    /*topk norm/scale*/ norm_topk_prob, routed_scale,
    /*SE*/ dW1se, dB1se, dW2se, dB2se, H_se,
    /*experts（共用一套参数）*/ hW1_local.data(), hB1_local.data(),
                                 hW2_local.data(), hB2_local.data(), H,
    /*EP*/ dOwner, capacity_factor,
    /*ep_size*/ world_size, /*ep_comm*/ (void*)ep,
    /*out*/ dY,
    /*stream*/ stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ----------------- 查看输出前 8 维 -----------------
  std::vector<half> hY((size_t)T_local*D);
  // **修正点**：从设备拷回主机 → DeviceToHost
  CUDA_CHECK(cudaMemcpy(hY.data(), dY, (size_t)T_local*D*sizeof(half), cudaMemcpyDeviceToHost));
  std::cout << "[CPP MoE] y[0,0:8]: ";
  for (int d0=0; d0<8 && d0<D; ++d0) std::cout << __half2float(hY[d0]) << " ";
  std::cout << "\n";

  // ----------------- 清理 -----------------
  if (ep) ep_destroy_nccl(ep);

  cudaFree(dX); cudaFree(dY); cudaFree(dWgate); cudaFree(dBias);
  cudaFree(dW1se); cudaFree(dW2se); cudaFree(dB1se); cudaFree(dB2se);
  cudaFree(dW1e); cudaFree(dW2e); cudaFree(dB1e); cudaFree(dB2e);
  cudaFree(dOwner);

  return 0;
}
