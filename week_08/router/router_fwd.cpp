// router/router_forward.cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void gemm(const half* x, const half* W, float* logits,
                     int T, int D, int E, cudaStream_t stream);
extern "C" void sigmoid(const float* logits, float* p,
                        int T, int E, cudaStream_t stream);
extern "C" void bias_add(float* inout, const float* bias,
                         int T, int E, cudaStream_t stream);
extern "C" void group_top2_select(const float* logits_choice,
                                  unsigned char* group_mask,
                                  int T, int E, int n_group, int topk_group,
                                  cudaStream_t stream);
extern "C" void apply_group_mask(float* logits_choice,
                                 const unsigned char* group_mask,
                                 int T, int E, int n_group,
                                 cudaStream_t stream);
extern "C" void row_topk(const float* in, int* idx,
                         int T, int E, int K, cudaStream_t stream);
extern "C" void gather_alpha(const float* p, const int* topk_idx, float* alpha,
                             int T, int E, int K, int norm_topk_prob, float routed_scale,
                             cudaStream_t stream);

extern "C" void router_forward(
    const half*  x,
    const half*  W_gate,
    const float* e_score_bias,
    int T, int D, int E,
    int K,
    int n_group,
    int topk_group,
    int  norm_topk_prob,
    float routed_scale,
    int*   topk_idx,
    float* alpha,
    float* logits,
    float* p,
    unsigned char* group_mask,
    cudaStream_t stream)
{
    gemm(x, W_gate, logits, T, D, E, stream);
    sigmoid(logits, p, T, E, stream);
    bias_add(logits, e_score_bias, T, E, stream);
    group_top2_select(logits, group_mask, T, E, n_group, topk_group, stream);
    apply_group_mask(logits, group_mask, T, E, n_group, stream);
    row_topk(logits, topk_idx, T, E, K, stream);
    gather_alpha(p, topk_idx, alpha, T, E, K, norm_topk_prob, routed_scale, stream);
}
