# include <stdlib.h>
# include <math.h>
# include <stdlib.h>



void attention(const float* Q, const float* K, const float* V, float* O,int N,int d){
    // 论文里 implicit 的 scale = 1/sqrt(d)
    const float scale = 1.0f / sqrtf((float)d);
    const int dv = d;  // 论文默认 V/O 也用 d 这个维度

    float* scores  = (float*)malloc(sizeof(float) * N);
    float* weights = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
        const float* qi = Q + i * d;

        // 1) scores_ij = <qi, kj> * scale
        float max_s = -INFINITY; 
        for (int j = 0; j < N; ++j) {
            const float* kj = K + j * d;
            float s = 0.0f;
            for (int k = 0; k < d; ++k) {
                s += qi[k] * kj[k];
            }
            s *= scale;
            scores[j] = s;
            if (s > max_s) max_s = s;
        }

        // 2) 稳定 softmax
        float sum_w = 0.0f;
        for (int j = 0; j < N; ++j) {
            float w = expf(scores[j] - max_s);
            weights[j] = w;
            sum_w += w;
        }
        float inv_sum_w = 1.0f / sum_w;
        for (int j = 0; j < N; ++j) {
            weights[j] *= inv_sum_w;
        }

        // 3) O_i = sum_j weights_ij * V_j
        for (int v = 0; v < dv; ++v) {
            float out = 0.0f;
            for (int j = 0; j < N; ++j) {
                const float* vj = V + j * dv;
                out += weights[j] * vj[v];
            }
            O[i * dv + v] = out;
        }
    }

    free(scores);
    free(weights);
}
