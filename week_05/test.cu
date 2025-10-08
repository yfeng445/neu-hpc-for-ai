// test.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>

// ---- 外部接口（来自 fa2_fwd.cu / fa2_bwd.cu）----
extern "C" size_t fa2_forward_smem_bytes(int Br, int Bc, int d, int dv, int block_x);
extern "C" void   fa2_forward_wmma(const float* dQ, const float* dK, const float* dV, float* dO,
                                   int N, int d, int dv, float scale, int Br, int Bc,
                                   dim3 grid, dim3 block, size_t sharedBytes);

extern "C" size_t fa2_backward_smem_bytes(int Br, int Bc, int d, int dv, int block_x);
extern "C" void   fa2_backward_wmma(const float* dQ, const float* dK, const float* dV, const float* dO,
                                    float* dQ_out, float* dK_out, float* dV_out,
                                    int N, int d, int dv, float scale, int Br, int Bc,
                                    dim3 grid, dim3 block, size_t sharedBytes);

// ---- 实用函数 ----
static double frob(const std::vector<float>& x){
    long double s=0; for(float v: x) s += (long double)v*v; return std::sqrt((double)s);
}
static std::pair<double,double> max_abs_rel(const std::vector<float>& a, const std::vector<float>& b){
    double ma=0.0, mr=0.0;
    for (size_t i=0;i<a.size();++i){
        double aa=a[i], bb=b[i];
        ma = std::max(ma, std::abs(aa-bb));
        double denom = std::abs(bb);
        if (denom>0) mr = std::max(mr, std::abs(aa-bb)/denom);
    }
    return {ma,mr};
}

#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA Error: %s @ %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1);} } while(0)

// ---- CPU 参考实现（与之前一致） ----
static void cpu_forward_ref(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                            int N, int d, int dv, float scale, std::vector<float>& O)
{
    std::vector<float> S((size_t)N*(size_t)N);
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            double acc=0.0;
            for (int k=0;k<d;k++) acc += (double)Q[(size_t)i*d+k]*(double)K[(size_t)j*d+k];
            S[(size_t)i*N+j] = (float)(acc*(double)scale);
        }
    }
    O.assign((size_t)N*(size_t)dv, 0.0f);
    for (int i=0;i<N;i++){
        float m=-INFINITY; for (int j=0;j<N;j++) m=fmaxf(m, S[(size_t)i*N+j]);
        double l=0.0; for(int j=0;j<N;j++) l += std::exp((double)S[(size_t)i*N+j] - (double)m);
        for (int v=0; v<dv; v++){
            double acc=0.0;
            for (int j=0;j<N;j++){
                double e = std::exp((double)S[(size_t)i*N+j] - (double)m) / l;
                acc += e * (double)V[(size_t)j*dv + v];
            }
            O[(size_t)i*dv + v] = (float)acc;
        }
    }
}

static void cpu_backward_ref(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                             const std::vector<float>& dO, int N, int d, int dv, float scale,
                             std::vector<float>& dQ, std::vector<float>& dK, std::vector<float>& dV)
{
    std::vector<float> S((size_t)N*(size_t)N), P((size_t)N*(size_t)N);
    // S, P
    for (int i=0;i<N;i++){
        float m=-INFINITY;
        for (int j=0;j<N;j++){
            double acc=0.0;
            for (int k=0;k<d;k++) acc += (double)Q[(size_t)i*d+k]*(double)K[(size_t)j*d+k];
            S[(size_t)i*N+j] = (float)(acc*(double)scale);
            m = fmaxf(m, S[(size_t)i*N+j]);
        }
        double l=0.0;
        for (int j=0;j<N;j++) l += std::exp((double)S[(size_t)i*N+j] - (double)m);
        for (int j=0;j<N;j++) P[(size_t)i*N+j] = (float)( std::exp((double)S[(size_t)i*N+j] - (double)m) / l );
    }
    // dV = P^T dO
    dV.assign((size_t)N*(size_t)dv, 0.0f);
    for (int j=0;j<N;j++) for (int v=0; v<dv; v++){
        double acc=0.0; for (int i=0;i<N;i++) acc += (double)P[(size_t)i*N + j]*(double)dO[(size_t)i*dv + v];
        dV[(size_t)j*dv + v] = (float)acc;
    }
    // dP = dO V^T
    std::vector<float> dP((size_t)N*(size_t)N, 0.0f);
    for (int i=0;i<N;i++) for (int j=0;j<N;j++){
        double acc=0.0; for (int v=0; v<dv; v++) acc += (double)dO[(size_t)i*dv + v]*(double)V[(size_t)j*dv + v];
        dP[(size_t)i*N + j] = (float)acc;
    }
    // dS
    std::vector<float> dS((size_t)N*(size_t)N);
    for (int i=0;i<N;i++){
        double dot=0.0; for (int j=0;j<N;j++) dot += (double)dP[(size_t)i*N+j]*(double)P[(size_t)i*N+j];
        for (int j=0;j<N;j++) dS[(size_t)i*N+j] = (float)( ((double)dP[(size_t)i*N+j] - dot) * (double)P[(size_t)i*N+j] );
    }
    // dQ, dK
    dQ.assign((size_t)N*(size_t)d, 0.0f); dK.assign((size_t)N*(size_t)d, 0.0f);
    for (int i=0;i<N;i++) for (int k=0;k<d;k++){
        double acc=0.0; for (int j=0;j<N;j++) acc += (double)dS[(size_t)i*N + j]*(double)K[(size_t)j*d + k];
        dQ[(size_t)i*d + k] = (float)( acc * (double)scale );
    }
    for (int j=0;j<N;j++) for (int k=0;k<d;k++){
        double acc=0.0; for (int i=0;i<N;i++) acc += (double)dS[(size_t)i*N + j]*(double)Q[(size_t)i*d + k];
        dK[(size_t)j*d + k] = (float)( acc * (double)scale );
    }
}

int main(){
    // ---- 配置参数（已验证数值正确的配置）----
    const int N=128, d=64, dv=64;
    const int Br=32, Bc=32;                     // 必须是 16 的倍数
    const int W = 4;                             // warp 数
    dim3 block(W*32,1,1);                        // 128 线程
    dim3 grid(1, (N + Br - 1) / Br, 1);          // 每个 CTA 负责 Br 行
    const float scale = 1.0f / std::sqrt((float)d);

    // ---- 随机初始化 ----
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> hQ((size_t)N*d), hK((size_t)N*d), hV((size_t)N*dv);
    for (auto& x: hQ) x = dist(rng);
    for (auto& x: hK) x = dist(rng);
    for (auto& x: hV) x = dist(rng);

    // ---- 设备内存 ----
    float *dQ,*dK,*dV,*dO,*dOut;
    CUDA_CHECK(cudaMalloc(&dQ, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, N*dv*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N*dv*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, N*dv*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), N*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), N*dv*sizeof(float), cudaMemcpyHostToDevice));

    // ---- 前向 ----
    size_t smem_fwd = fa2_forward_smem_bytes(Br, Bc, d, dv, block.x);
    fa2_forward_wmma(dQ,dK,dV,dOut,N,d,dv,scale,Br,Bc,grid,block,smem_fwd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> O_gpu(N*dv), O_ref;
    CUDA_CHECK(cudaMemcpy(O_gpu.data(), dOut, N*dv*sizeof(float), cudaMemcpyDeviceToHost));
    cpu_forward_ref(hQ,hK,hV,N,d,dv,scale,O_ref);

    auto [mae_f,mre_f] = max_abs_rel(O_gpu,O_ref);
    printf("FWD  Fro(GPU)=%.8e  Fro(REF)=%.8e  MaxAbs/MaxRel=%.3e/%.3e\n",
           frob(O_gpu), frob(O_ref), mae_f, mre_f);

    // ---- 反向：构造 dO，运行 GPU 反向（工程化 tiles+WMMA 版本）----
    std::vector<float> h_dO((size_t)N*dv);
    for (auto& x: h_dO) x = dist(rng);
    CUDA_CHECK(cudaMemcpy(dO, h_dO.data(), N*dv*sizeof(float), cudaMemcpyHostToDevice));

    float *d_dQ,*d_dK,*d_dV;
    CUDA_CHECK(cudaMalloc(&d_dQ, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dK, N*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dV, N*dv*sizeof(float)));
    // 反向中 dK/dV 需要 atomic 汇总，必须先清零
    CUDA_CHECK(cudaMemset(d_dQ, 0, N*d*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dK, 0, N*d*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dV, 0, N*dv*sizeof(float)));

    size_t smem_bwd = fa2_backward_smem_bytes(Br, Bc, d, dv, block.x);
    fa2_backward_wmma(dQ,dK,dV,dO,d_dQ,d_dK,d_dV,N,d,dv,scale,Br,Bc,grid,block,smem_bwd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 回读 GPU 反向结果 ----
    std::vector<float> dQ_gpu(N*d), dK_gpu(N*d), dV_gpu(N*dv);
    CUDA_CHECK(cudaMemcpy(dQ_gpu.data(), d_dQ, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dK_gpu.data(), d_dK, N*d*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dV_gpu.data(), d_dV, N*dv*sizeof(float), cudaMemcpyDeviceToHost));

    // ---- CPU 反向参考 ----
    std::vector<float> dQ_ref, dK_ref, dV_ref;
    cpu_backward_ref(hQ,hK,hV,h_dO,N,d,dv,scale,dQ_ref,dK_ref,dV_ref);

    auto [mae_q, mre_q] = max_abs_rel(dQ_gpu, dQ_ref);
    auto [mae_k, mre_k] = max_abs_rel(dK_gpu, dK_ref);
    auto [mae_v, mre_v] = max_abs_rel(dV_gpu, dV_ref);

    printf("BWD  ||dQ|| GPU/REF: %.8e / %.8e  MaxAbs/MaxRel=%.3e/%.3e\n",
           frob(dQ_gpu), frob(dQ_ref), mae_q, mre_q);
    printf("     ||dK|| GPU/REF: %.8e / %.8e  MaxAbs/MaxRel=%.3e/%.3e\n",
           frob(dK_gpu), frob(dK_ref), mae_k, mre_k);
    printf("     ||dV|| GPU/REF: %.8e / %.8e  MaxAbs/MaxRel=%.3e/%.3e\n",
           frob(dV_gpu), frob(dV_ref), mae_v, mre_v);

    // ---- 资源回收 ----
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFree(d_dQ));
    CUDA_CHECK(cudaFree(d_dK));
    CUDA_CHECK(cudaFree(d_dV));
    return 0;
}
