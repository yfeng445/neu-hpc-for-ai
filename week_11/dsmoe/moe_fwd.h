// moe_fwd.h
#pragma once

#include <cuda_fp16.h>        // half
#include <cuda_runtime_api.h> // cudaStream_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FlashMoE 适配入口：以 DeepSeek-MoE 的前向接口形式暴露。
 *
 * 形状/类型约定（均为 device 上的数据，除非特别说明）：
 *   x           : half  [T, D]         —— 输入激活
 *   W_gate      : half  [D, E]         —— 路由/门控权重（适配层内部按 FlashMoE 期望布局打包）
 *   e_score_bias: float [E]            —— 当前实现未使用，可传 nullptr
 *
 *   W1_se/B1_se/W2_se/B2_se/H_se       —— SE 分支未使用，可传 nullptr/0
 *
 *   W1_local[e] : half*  指向第 e 个专家的 up 权重，视作 [P, H]
 *   B1_local[e] : float* 指向第 e 个专家的 up 偏置，视作 [P]
 *   W2_local[e] : half*  指向第 e 个专家的 down 权重，视作 [H, P]
 *   B2_local[e] : float* 指向第 e 个专家的 down 偏置，视作 [H]
 *   —— 上述四个为“指向 device 张量的指针”的数组；数组本体可位于 host 或 device，
 *      实现内部会在需要时拷贝指针表；但每个元素必须是有效的 device 指针。
 *
 *   expert_owner/capacity_factor/ep_size/ep_comm —— 当前实现未使用，保留兼容位
 *
 *   y           : half  [T, D]         —— 输出
 *
 * 维度一致性要求（运行时检查，不符将退出）：
 *   T == ACC::S::value, D == ACC::H::value, E == ACC::E::value, H == ACC::P::value
 *
 * 其余路由超参（K, n_group, topk_group, norm_topk_prob, routed_scale）当前实现忽略。
 */
void moe_fwd(
    half*  x,                 // [T, D]
    half*  W_gate,            // [D, E]
    float* e_score_bias,      // [E] (unused; may be nullptr)
    int    T, int D, int E, int K,
    int    n_group, int topk_group,
    int    norm_topk_prob, float routed_scale,
    // SE branch (unused; may be nullptr/0)
    half*  W1_se, float* B1_se, half* W2_se, float* B2_se, int H_se,
    // Per-expert weights/bias pointers (arrays of length E; each element is a device pointer)
    half**  W1_local, float** B1_local,
    half**  W2_local, float** B2_local,
    int     H,                // inner hidden dim (must equal ACC::P::value)
    // EP / comm placeholders (unused; keep for signature compatibility)
    int*    expert_owner, float capacity_factor, int ep_size, void* ep_comm,
    // Output
    half*   y,                // [T, D]
    cudaStream_t stream
);

#ifdef __cplusplus
} // extern "C"
#endif
