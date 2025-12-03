#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <algorithm>

static inline size_t IDX(size_t i, size_t j, size_t ld) { return i*ld + j; }

void bwd_cpu(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    const float* L,                 // length N
    int N, int d, float scale,      // scale=1.0f for this simplest variant
    float* dQ, float* dK, float* dV
){
    // zero outputs
    std::memset(dQ, 0, sizeof(float) * (size_t)N * d);
    std::memset(dK, 0, sizeof(float) * (size_t)N * d);
    std::memset(dV, 0, sizeof(float) * (size_t)N * d);

    // temps
    std::vector<float> S((size_t)N*N, 0.f);
    std::vector<float> P((size_t)N*N, 0.f);
    std::vector<float> dP_((size_t)N*N, 0.f);
    std::vector<float> dS_((size_t)N*N, 0.f);
    std::vector<float> D(N, 0.f);

    // D = rowsum(dO ⊙ O)
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        const float* dOi = dO + (size_t)i * d;
        const float* Oi  = O  + (size_t)i * d;
        for (int t = 0; t < d; ++t) acc += (double)dOi[t] * (double)Oi[t];
        D[i] = (float)acc;
    }

    // S = scale * Q K^T
    for (int i = 0; i < N; ++i) {
        const float* Qi = Q + (size_t)i * d;
        for (int j = 0; j < N; ++j) {
            const float* Kj = K + (size_t)j * d;
            double acc = 0.0;
            for (int t = 0; t < d; ++t) acc += (double)Qi[t] * (double)Kj[t];
            S[IDX(i,j,N)] = (float)(scale * acc);
        }
    }

    // P = exp(S - L)   (use provided L rowwise; no renormalization here)
    for (int i = 0; i < N; ++i) {
        const float Li = L[i];
        for (int j = 0; j < N; ++j) {
            P[IDX(i,j,N)] = std::exp((double)S[IDX(i,j,N)] - (double)Li);
        }
    }

    // dV = P^T dO
    for (int j = 0; j < N; ++j) {
        float* dVj = dV + (size_t)j * d;
        for (int t = 0; t < d; ++t) {
            double acc = 0.0;
            for (int i = 0; i < N; ++i) {
                acc += (double)P[IDX(i,j,N)] * (double)dO[IDX(i,t,d)];
            }
            dVj[t] = (float)acc;
        }
    }

    // dP = dO V^T
    for (int i = 0; i < N; ++i) {
        const float* dOi = dO + (size_t)i * d;
        for (int j = 0; j < N; ++j) {
            const float* Vj = V + (size_t)j * d;
            double acc = 0.0;
            for (int t = 0; t < d; ++t) acc += (double)dOi[t] * (double)Vj[t];
            dP_[IDX(i,j,N)] = (float)acc;
        }
    }

    // dS = P ⊙ (dP - D)   (broadcast D[i] over columns)
    for (int i = 0; i < N; ++i) {
        const float Di = D[i];
        for (int j = 0; j < N; ++j) {
            dS_[IDX(i,j,N)] = P[IDX(i,j,N)] * (dP_[IDX(i,j,N)] - Di);
        }
    }

    // dQ = scale * dS K
    for (int i = 0; i < N; ++i) {
        float* dQi = dQ + (size_t)i * d;
        for (int k_ = 0; k_ < d; ++k_) {
            double acc = 0.0;
            for (int j = 0; j < N; ++j) acc += (double)dS_[IDX(i,j,N)] * (double)K[IDX(j,k_,d)];
            dQi[k_] = (float)(acc * scale);
        }
    }

    // dK = scale * dS^T Q
    for (int j = 0; j < N; ++j) {
        float* dKj = dK + (size_t)j * d;
        for (int k_ = 0; k_ < d; ++k_) {
            double acc = 0.0;
            for (int i = 0; i < N; ++i) acc += (double)dS_[IDX(i,j,N)] * (double)Q[IDX(i,k_,d)];
            dKj[k_] = (float)(acc * scale);
        }
    }
}

int main() {
    const int N = 4, d = 3;
    const float scale = 1.0f;  // 最基础版本：不做 1/sqrt(d) 缩放

    float Q[N*d], K[N*d], V[N*d], O[N*d], dO[N*d], L[N];
    // 按你给的 .cu 初始化方式构造输入
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < d; ++k) {
            Q[i*d+k]  = 0.01f * (1 + i*d + k);
            K[i*d+k]  = 0.02f * (1 + i*d + k);
            V[i*d+k]  = 0.03f * (1 + i*d + k);
            O[i*d+k]  = 0.04f * (1 + i*d + k);
            dO[i*d+k] = 0.05f * (1 + i*d + k);
        }
        L[i] = 0.1f * (1 + i);
    }

    std::vector<float> dQ(N*d, 0.f), dK(N*d, 0.f), dV(N*d, 0.f);

    bwd_cpu(Q, K, V, O, dO, L, N, d, scale,
                     dQ.data(), dK.data(), dV.data());

    auto printMx = [](const char* name, const std::vector<float>& M, int R, int C){
        std::printf("%s:\n", name);
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) std::printf("%8.5f ", M[IDX(i,j,C)]);
            std::printf("\n");
        }
    };

    printMx("dQ:", dQ, N, d);
    printMx("dK:", dK, N, d);
    printMx("dV:", dV, N, d);
    return 0;
}
