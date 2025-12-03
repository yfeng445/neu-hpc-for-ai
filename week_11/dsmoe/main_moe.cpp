// main_moe.cpp (FlashMoE driver)
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// FlashMoE ACC configuration (compile-time dims)
#include "../flashmoe/csrc/include/kleos/bootstrap.cuh"

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

// moe_fwd 接口保持与原 DS-MoE 一致，但内部调用 FlashMoE fused forward
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
  using kleos::ACC;

  // FlashMoE 编译期维度
  const int T = ACC::S::value;   // sequence length
  const int D = ACC::H::value;   // hidden size
  const int E = ACC::E::value;   // experts
  const int K = 1;               // unused by FlashMoE adapter
  const int n_group = 1, topk_group = 1; // unused
  const int H = ACC::P::value;   // expert hidden (P)
  const int H_se = 0;            // shared expert未使用
  const int norm_topk_prob = 0;  // unused
  const float routed_scale = 1.f;
  const float capacity_factor = 1.f;

  // 设备选择：默认使用 LOCAL_RANK（或 0）
  int local_rank = env_int("LOCAL_RANK", 0);
  CUDA_CHECK(cudaSetDevice(local_rank));
  std::cerr << "[init] device=" << local_rank
            << " FlashMoE dims S="<<T<<" H="<<D<<" E="<<E<<" P="<<H<<"\n";

  // Host 侧随机初始化
  std::mt19937 rng(2025);
  std::uniform_real_distribution<float> dx(-1.f,1.f);
  std::uniform_real_distribution<float> dw(-0.05f,0.05f);

  // 输入 activations [T,D]
  std::vector<float> hX_f((size_t)T*D);
  for (auto& v: hX_f) v = dx(rng);
  std::vector<half> hX_h; fill_half_from_float(hX_f, hX_h);

  // Router / gate weights [D,E] （FlashMoE 期望 layout: PX x H，默认 PX==E）
  std::vector<float> hWgate_f((size_t)D*E);
  for (auto& v: hWgate_f) v = dw(rng);
  std::vector<half> hWgate_h; fill_half_from_float(hWgate_f, hWgate_h);
  std::vector<float> hBias(E, 0.f); // 未使用，但保持接口

  // Expert weights/biases（所有专家共享一套参数以简化）
  std::vector<float> W1_f((size_t)D*H), W2_f((size_t)H*D);
  for (auto& v: W1_f) v = dw(rng);
  for (auto& v: W2_f) v = dw(rng);
  std::vector<half> W1_h, W2_h; fill_half_from_float(W1_f, W1_h); fill_half_from_float(W2_f, W2_h);
  std::vector<float> B1_f(H, 0.02f), B2_f(D, 0.05f);

  // 指针数组（长度 E；此示例让所有专家复用同一参数）
  std::vector<const half*>  hW1_local(E, nullptr), hW2_local(E, nullptr);
  std::vector<const float*> hB1_local(E, nullptr), hB2_local(E, nullptr);

  // Device 侧分配
  half *dX=nullptr,*dY=nullptr,*dWgate=nullptr,*dW1=nullptr,*dW2=nullptr;
  float *dBias=nullptr,*dB1=nullptr,*dB2=nullptr;
  CUDA_CHECK(cudaMalloc(&dX, (size_t)T*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)T*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dWgate, (size_t)D*E*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dBias, E*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dW1, (size_t)D*H*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dW2, (size_t)H*D*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB1, H*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB2, D*sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dX, hX_h.data(), (size_t)T*D*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWgate, hWgate_h.data(), (size_t)D*E*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), E*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW1, W1_h.data(), (size_t)D*H*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW2, W2_h.data(), (size_t)H*D*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB1, B1_f.data(), H*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB2, B2_f.data(), D*sizeof(float), cudaMemcpyHostToDevice));

  // 填充指针数组
  for (int e=0; e<E; ++e) {
    hW1_local[e] = dW1;
    hW2_local[e] = dW2;
    hB1_local[e] = dB1;
    hB2_local[e] = dB2;
  }

  // expert_owner（FlashMoE 路径未使用，但接口需要）
  std::vector<int> owner_h(E, 0);
  int *dOwner=nullptr;
  CUDA_CHECK(cudaMalloc(&dOwner, E*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dOwner, owner_h.data(), E*sizeof(int), cudaMemcpyHostToDevice));

  cudaStream_t stream = 0;

  // 调用 FlashMoE-backed moe_fwd
  moe_fwd(
    /*x*/ dX,
    /*router*/ dWgate, dBias,
    /*shape*/ T, D, E, K,
    /*group route*/ n_group, topk_group,
    /*topk norm/scale*/ norm_topk_prob, routed_scale,
    /*SE*/ nullptr, nullptr, nullptr, nullptr, H_se,
    /*experts*/ hW1_local.data(), hB1_local.data(), hW2_local.data(), hB2_local.data(), H,
    /*EP*/ dOwner, capacity_factor,
    /*ep_size*/ 1, /*ep_comm*/ nullptr,
    /*out*/ dY,
    /*stream*/ stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 拷回并打印前 8 个维度
  std::vector<half> hY((size_t)T*D);
  CUDA_CHECK(cudaMemcpy(hY.data(), dY, (size_t)T*D*sizeof(half), cudaMemcpyDeviceToHost));
  std::cout << "[FlashMoE] y[0,0:8]: ";
  for (int d0=0; d0<8 && d0<D; ++d0) std::cout << __half2float(hY[d0]) << " ";
  std::cout << "\n";

  // 资源释放
  cudaFree(dX); cudaFree(dY); cudaFree(dWgate); cudaFree(dBias);
  cudaFree(dW1); cudaFree(dW2); cudaFree(dB1); cudaFree(dB2);
  cudaFree(dOwner);

  return 0;
}
