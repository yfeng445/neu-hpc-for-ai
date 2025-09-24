# include <stdio.h>
# include <cuda_runtime.h>
# include <math_constants.h> 

__global__ void online_normalizer_softmax(float* X, float* Y, int rows, int cols){
    int row = blockIdx.x;
    if (row >= rows) return;

    float m_local = -CUDART_INF_F;  
    float d_local = 0.0f;

    const int base = row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float x = X[base + j];
        float m_new = fmaxf(m_local, x);                         
        d_local = d_local * __expf(m_local - m_new) + __expf(x - m_new);                           
        m_local = m_new;
    }

    extern __shared__ float smem[];
    float* smax = smem;                       
    float* ssum = smem + blockDim.x;         

    smax[threadIdx.x] = m_local;
    ssum[threadIdx.x] = d_local;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            float ma = smax[threadIdx.x];
            float da = ssum[threadIdx.x];
            float mb = smax[threadIdx.x + s];
            float db = ssum[threadIdx.x + s];
            float m   = fmaxf(ma, mb);                           
            float d   = da * __expf(ma - m) + db * __expf(mb - m);

            smax[threadIdx.x] = m;
            ssum[threadIdx.x] = d;
        }
        __syncthreads();
    }

    float mV = smax[0];
    float dV = ssum[0];
    if (threadIdx.x == 0) {
        if (!(dV > 0.0f)) dV = 1.0f; 
        smax[0] = mV;  ssum[0] = dV;
    }
    __syncthreads();
    mV = smax[0];  dV = ssum[0];

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float x = X[base + j];
        Y[base + j] = __expf(x - mV) / dV;
    }
}


