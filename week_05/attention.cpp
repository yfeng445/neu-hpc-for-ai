#include <cstdio>
#include <cstdlib>
#include <cmath>

// 最基础的、带数值稳定 softmax 的 attention 实现
void attention(const float* Q, const float* K, const float* V, float* O, int N, int d) {

    float* scores  = (float*)std::malloc(sizeof(float) * N);
    float* weights = (float*)std::malloc(sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
        const float* qi = Q + i * d;

        // scores_ij = <qi, kj> * scale
        float max_s = -INFINITY;
        for (int j = 0; j < N; ++j) {
            const float* kj = K + j * d;
            float s = 0.0f;
            for (int k = 0; k < d; ++k) {
                s += qi[k] * kj[k];
            }
            scores[j] = s;
            if (s > max_s) max_s = s;
        }

        // safe softmax
        float sum_w = 0.0f;
        for (int j = 0; j < N; ++j) {
            float w = std::exp(scores[j] - max_s);
            weights[j] = w;
            sum_w += w;
        }
        float inv_sum_w = 1.0f / sum_w;
        for (int j = 0; j < N; ++j) {
            weights[j] *= inv_sum_w;
        }

        // O_i = sum_j weights_ij * V_j
        for (int v = 0; v < d; ++v) {
            float out = 0.0f;
            for (int j = 0; j < N; ++j) {
                const float* vj = V + j * d;
                out += weights[j] * vj[v];
            }
            O[i * d + v] = out;
        }
    }

    std::free(scores);
    std::free(weights);
}

int main() {

    const int N = 32;
    const int d = 32;
    const int size = N * d;

    float Q[size];
    float K[size];
    float V[size];
    float O[size];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            int idx = i * d + j;
            float x = (float)idx;

            Q[idx] = std::sinf(0.01f * x);            // [-1, 1]
            K[idx] = std::cosf(0.02f * x);            // [-1, 1]，和 Q 不同
            V[idx] = std::sinf(0.03f * x + 0.5f);     // 再一套不同的模式
        }
    }

    attention(Q, K, V, O, N, d);

    std::printf("Output O (N=32, d=32):\n");
    for (int i = 0; i < N; ++i) {
        std::printf("Row %d:", i);
        for (int j = 0; j < d; ++j) {
            std::printf(" %.3f", O[i * d + j]);
        }
        std::printf("\n");
    }

    return 0;
}
