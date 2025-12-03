#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>

static inline size_t idx2(size_t i, size_t j, size_t ld) { return i * ld + j; }

extern "C"
void fa2_backward_reference(const float* Q, const float* K, const float* V, const float* O, const float* dO, int N, int d, const float* L, int Bc, int Br, float* dQ, float* dK, float* dV){

    assert(N > 0 && d > 0);
    assert(Bc > 0 && Br > 0);

    std::memset(dQ, 0, sizeof(float) * (size_t)N * d);
    std::memset(dK, 0, sizeof(float) * (size_t)N * d);
    std::memset(dV, 0, sizeof(float) * (size_t)N * d);

    std::vector<float> D(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        const float* dOi = dO + (size_t)i * d;
        const float* Oi  = O  + (size_t)i * d;
        for (int k = 0; k < d; ++k) acc += (double)dOi[k] * (double)Oi[k];
        D[i] = (float)acc;
    }

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    std::vector<float> Qi, Oi, dOi, Li, Di; // sizes depend on Br
    std::vector<float> Kj, Vj;              // sizes depend on Bc
    std::vector<float> S, P, dP, dS;        // (Br x Bc)
    std::vector<float> dKj, dVj;            // (Bc x d)

    // Allocate to maximum possible tile sizes once
    Qi.resize((size_t)Br * d);
    Oi.resize((size_t)Br * d);
    dOi.resize((size_t)Br * d);
    Li.resize((size_t)Br);
    Di.resize((size_t)Br);

    Kj.resize((size_t)Bc * d);
    Vj.resize((size_t)Bc * d);

    S.resize((size_t)Br * Bc);
    P.resize((size_t)Br * Bc);
    dP.resize((size_t)Br * Bc);
    dS.resize((size_t)Br * Bc);

    dKj.resize((size_t)Bc * d);
    dVj.resize((size_t)Bc * d);

    for (int j_blk = 0; j_blk < Tc; ++j_blk) {
        const int j0   = j_blk * Bc;
        const int bc   = std::min(Bc, N - j0);
        if (bc <= 0) break;

        // Load K_j, V_j to "SRAM"
        for (int c = 0; c < bc; ++c) {
            const float* Krow = K + (size_t)(j0 + c) * d;
            const float* Vrow = V + (size_t)(j0 + c) * d;
            std::memcpy(&Kj[(size_t)c * d], Krow, sizeof(float) * d);
            std::memcpy(&Vj[(size_t)c * d], Vrow, sizeof(float) * d);
        }

        // Initialize dK_j, dV_j = 0 on "SRAM"
        std::memset(dKj.data(), 0, sizeof(float) * (size_t)bc * d);
        std::memset(dVj.data(), 0, sizeof(float) * (size_t)bc * d);

        for (int i_blk = 0; i_blk < Tr; ++i_blk) {
            const int i0 = i_blk * Br;
            const int br = std::min(Br, N - i0);
            if (br <= 0) break;

            for (int r = 0; r < br; ++r) {
                const float* Qrow  = Q  + (size_t)(i0 + r) * d;
                const float* Orow  = O  + (size_t)(i0 + r) * d;
                const float* dOrow = dO + (size_t)(i0 + r) * d;
                std::memcpy(&Qi[(size_t)r * d],  Qrow,  sizeof(float) * d);
                std::memcpy(&Oi[(size_t)r * d],  Orow,  sizeof(float) * d);
                std::memcpy(&dOi[(size_t)r * d], dOrow, sizeof(float) * d);
                Li[r] = L[i0 + r];
                Di[r] = D[i0 + r];
            }

            // S_i^(j) = Q_i K_j^T  (br x bc)
            for (int r = 0; r < br; ++r) {
                const float* qrow = &Qi[(size_t)r * d];
                for (int c = 0; c < bc; ++c) {
                    const float* krow = &Kj[(size_t)c * d];
                    double acc = 0.0;
                    for (int t = 0; t < d; ++t) acc += (double)qrow[t] * (double)krow[t];
                    S[idx2(r, c, bc)] = (float)acc;
                }
            }

            // P_i^(j) = exp(S - L_i)   (broadcast Li over columns)
            for (int r = 0; r < br; ++r) {
                const float lr = Li[r];
                for (int c = 0; c < bc; ++c) {
                    P[idx2(r, c, bc)] = std::exp(S[idx2(r, c, bc)] - lr);
                }
            }

            // dV_j += P^T * dO_i
            // For each column c and feature k: dVj[c,k] += sum_r P[r,c] * dOi[r,k]
            for (int c = 0; c < bc; ++c) {
                float* dVjc = &dVj[(size_t)c * d];
                for (int kf = 0; kf < d; ++kf) {
                    double acc = 0.0;
                    for (int r = 0; r < br; ++r) {
                        acc += (double)P[idx2(r, c, bc)] * (double)dOi[(size_t)r * d + kf];
                    }
                    dVjc[kf] += (float)acc;
                }
            }

            // dP_i^(j) = dO_i * V_j^T  (br x bc)
            for (int r = 0; r < br; ++r) {
                const float* dOrow = &dOi[(size_t)r * d];
                for (int c = 0; c < bc; ++c) {
                    const float* Vrow = &Vj[(size_t)c * d];
                    double acc = 0.0;
                    for (int kf = 0; kf < d; ++kf) acc += (double)dOrow[kf] * (double)Vrow[kf];
                    dP[idx2(r, c, bc)] = (float)acc;
                }
            }

            // dS_i^(j) = P ⊙ (dP - D_i)  (broadcast D_i over columns)
            for (int r = 0; r < br; ++r) {
                const float Dr = Di[r];
                for (int c = 0; c < bc; ++c) {
                    dS[idx2(r, c, bc)] = P[idx2(r, c, bc)] * (dP[idx2(r, c, bc)] - Dr);
                }
            }

            // Load Q_i again (Algorithm 2, line 15) and:
            // dQ_i += dS_i^(j) * K_j    (br x d)
            for (int r = 0; r < br; ++r) {
                float* dQrow = dQ + (size_t)(i0 + r) * d;
                for (int kf = 0; kf < d; ++kf) {
                    double acc = 0.0;
                    for (int c = 0; c < bc; ++c) {
                        acc += (double)dS[idx2(r, c, bc)] * (double)Kj[(size_t)c * d + kf];
                    }
                    dQrow[kf] += (float)acc;
                }
            }

            // dK_j += dS_i^(j)^T * Q_i   (bc x d)
            for (int c = 0; c < bc; ++c) {
                float* dKjc = &dKj[(size_t)c * d];
                for (int kf = 0; kf < d; ++kf) {
                    double acc = 0.0;
                    for (int r = 0; r < br; ++r) {
                        acc += (double)dS[idx2(r, c, bc)] * (double)Qi[(size_t)r * d + kf];
                    }
                    dKjc[kf] += (float)acc;
                }
            }
        } // end for i_blk

        // Write dK_j, dV_j back to HBM (global outputs)
        for (int c = 0; c < bc; ++c) {
            float* dKrow = dK + (size_t)(j0 + c) * d;
            float* dVrow = dV + (size_t)(j0 + c) * d;
            for (int kf = 0; kf < d; ++kf) {
                dKrow[kf] += dKj[(size_t)c * d + kf];
                dVrow[kf] += dVj[(size_t)c * d + kf];
            }
        }
    } // end for j_blk
}


int main() {
    const int N = 4, d = 3;
    float Q[N*d], K[N*d], V[N*d], O[N*d], dO[N*d], L[N];
    // Fill with small deterministic numbers
    for (int i=0;i<N;i++){
        for(int k=0;k<d;k++){
            Q[i*d+k] = 0.01f*(1+i*d+k);
            K[i*d+k] = 0.02f*(1+i*d+k);
            V[i*d+k] = 0.03f*(1+i*d+k);
            O[i*d+k] = 0.04f*(1+i*d+k);
            dO[i*d+k]= 0.05f*(1+i*d+k);
        }
        L[i] = 0.1f*(1+i);
    }
    std::vector<float> dQ(N*d,0.f), dK(N*d,0.f), dV(N*d,0.f);

    fa2_backward_reference(Q,K,V,O,dO,N,d,L, /*Bc=*/2, /*Br=*/2,
                           dQ.data(), dK.data(), dV.data());

    auto printMx=[&](const char* name, const std::vector<float>& M){
        printf("%s:\n", name);
        for(int i=0;i<N;i++){
            for(int k=0;k<d;k++) printf("%8.5f ", M[i*d+k]);
            printf("\n");
        }
    };
    printMx("dQ", dQ); printMx("dK", dK); printMx("dV", dV);
    return 0;
}

