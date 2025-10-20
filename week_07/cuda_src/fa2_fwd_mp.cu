#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "fa2_fwd.cu"

#define CUDA_CHECK(cmd) do {                             \
  cudaError_t e = (cmd);                                 \
  if (e != cudaSuccess) {                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n",            \
            __FILE__, __LINE__, cudaGetErrorString(e));  \
    MPI_Abort(MPI_COMM_WORLD, -1);                       \
  }                                                      \
} while (0)

__global__ void smk_tst(
    const float* __restrict__ Q_local,  
    float* __restrict__ O_local,        
    int BH, int Nq_local, int d)
{
    int b = blockIdx.y;  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 行：序列维
    if (b >= BH || i >= Nq_local) return;

    // 让线程在列上stride循环，保证任意 d 都能覆盖
    for (int j = threadIdx.y; j < d; j += blockDim.y) {
        size_t off = (size_t)b * Nq_local * d + (size_t)i * d + j;
        O_local[off] = Q_local[off];
    }
}




int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // init MPI rank & nprocs
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int P; MPI_Comm_size(MPI_COMM_WORLD, &P);
   
    // bind MPI rank to GPU device
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0; MPI_Comm_rank(local_comm, &local_rank);
    int local_size = 0; MPI_Comm_size(local_comm, &local_size);
    int device_count = 0; cudaGetDeviceCount(&device_count);
    int dev = (device_count > 0) ? (local_rank % device_count) : 0; cudaSetDevice(dev);


    // build testing data
    const int N = 32;
    int q = N / P, r = N % P;
    int Lk   = q + (rank < r ? 1 : 0);
    size_t BH = 1;  // SHA
    int d = 16, dv = 16;

    std::vector<float> Q_full, K_full, V_full;
    
    if (rank == 0) {
        Q_full.resize(BH*N*d);
        K_full.resize(BH*N*d);
        V_full.resize(BH*N*dv);
        auto qidx = [&](int i,int j){ return i*d + j; };
        auto vidx = [&](int i,int j){ return i*dv + j; };
        for (int i=0;i<N;++i) for (int j=0;j<d;++j)  Q_full[qidx(i,j)] = 0.01f*i + 0.001f*j;
        for (int i=0;i<N;++i) for (int j=0;j<d;++j)  K_full[qidx(i,j)] = 0.02f*i + 0.002f*j;
        for (int i=0;i<N;++i) for (int j=0;j<dv;++j) V_full[vidx(i,j)] = 0.03f*i + 0.003f*j;
    }

    // gpu mem alloc
    float *dQ, *dK, *dV, *dK_full, *dV_full;
    cudaMalloc(&dQ,      BH*Lk*d   * sizeof(float));
    cudaMalloc(&dK,      BH*Lk*d   * sizeof(float));
    cudaMalloc(&dV,      BH*Lk*dv  * sizeof(float));
    cudaMalloc(&dK_full, BH*N*d    * sizeof(float));
    cudaMalloc(&dV_full, BH*N*dv   * sizeof(float));

    // root: prepare scatterv displacements & counts
    std::vector<int> scQ, sdQ, scK, sdK, scV, sdV;
    
    if (rank == 0) {
        scQ.resize(P); sdQ.resize(P);
        scK.resize(P); sdK.resize(P);
        scV.resize(P); sdV.resize(P);
        for (int k = 0; k < P; ++k) {
            int L   = q + (k < r ? 1 : 0);
            int off = k*q + (k < r ? k : r);
            scQ[k] = (int)(BH * L * d);    sdQ[k] = (int)(BH * off * d);
            scK[k] = (int)(BH * L * d);    sdK[k] = (int)(BH * off * d);
            scV[k] = (int)(BH * L * dv);   sdV[k] = (int)(BH * off * dv);
        }
    }

    // rank 0 host -> local rank scatterv
    std::vector<float> hQ_local(BH * Lk * d);
    std::vector<float> hK_local(BH * Lk * d);
    std::vector<float> hV_local(BH * Lk * dv);
    // root rank scatters to local ranks, so if not root, send nothing
    MPI_Scatterv(rank==0 ? Q_full.data() : nullptr, rank==0 ? scQ.data() : nullptr,rank==0 ? sdQ.data() : nullptr, 
                MPI_FLOAT, // send type
                hQ_local.data(), // receive buffer 
                (int)hQ_local.size(), // receive count 
                MPI_FLOAT, // receive type 
                0, // root 
                MPI_COMM_WORLD);
    MPI_Scatterv(rank==0 ? K_full.data() : nullptr, rank==0 ? scK.data() : nullptr, rank==0 ? sdK.data() : nullptr, 
                MPI_FLOAT, hK_local.data(), (int)hK_local.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank==0 ? V_full.data() : nullptr, rank==0 ? scV.data() : nullptr, rank==0 ? sdV.data() : nullptr,
                MPI_FLOAT, hV_local.data(), (int)hV_local.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // local rank host to device memcpy
    cudaMemcpy(dQ, hQ_local.data(), hQ_local.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK_local.data(), hK_local.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV_local.data(), hV_local.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    // allgatherv K, V to full K, V
    std::vector<int> rcK(P), dpK(P), rcV(P), dpV(P);
    for (int k = 0; k < P; ++k) {
        int L   = q + (k < r ? 1 : 0);
        int off = k*q + (k < r ? k : r);
        rcK[k] = (int)(BH * L * d);    dpK[k] = (int)(BH * off * d);
        rcV[k] = (int)(BH * L * dv);   dpV[k] = (int)(BH * off * dv);
    }

    std::vector<float> hK_full(BH * N * d);
    std::vector<float> hV_full(BH * N * dv);
    // allgather need each rank's local K, V, so there is no root check
    MPI_Allgatherv(hK_local.data(), (int)hK_local.size(), MPI_FLOAT, hK_full.data(), rcK.data(), dpK.data(), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(hV_local.data(), (int)hV_local.size(), MPI_FLOAT, hV_full.data(), rcV.data(), dpV.data(), MPI_FLOAT, MPI_COMM_WORLD);
    cudaMemcpy(dK_full, hK_full.data(), hK_full.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV_full, hV_full.data(), hV_full.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    
    // launch fa2_fwd kernel
    int R = 4;                          
    const int Br = 16 * R;            
    const int Bc = 16;                  
    dim3 block(32 * R, 1, 1);          
    dim3 grid(1, (Lk + Br - 1) / Br, 1);
    size_t shm_per_warp = fa2_forward_smem_bytes(16, Bc, d, dv, 32);
    size_t shm = R * shm_per_warp;

    float scale = 1.0f / sqrtf((float)d);
    float* dO = nullptr;
    cudaMalloc(&dO, (size_t)BH * Lk * dv * sizeof(float));

    fa2_forward_wmma(dQ, dK_full, dV_full, dO, Lk, N, d, dv, scale, Br, Bc, grid, block, shm);
    cudaDeviceSynchronize();


    // gather O as result
    std::vector<int> rcO(P), dpO(P);
    for (int k = 0; k < P; ++k) {
        int L    = q + (k < r ? 1 : 0);
        int off  = k*q + (k < r ? k : r);
        rcO[k]   = (int)(BH * L   * dv);
        dpO[k]   = (int)(BH * off * dv);
    }

    std::vector<float> hO_local(BH * Lk * dv);
    cudaMemcpy(hO_local.data(), dO, hO_local.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    std::vector<float> O_round;
    if (rank == 0) O_round.resize(BH * N * dv);

    MPI_Gatherv(hO_local.data(), (int)hO_local.size(), MPI_FLOAT,
                rank == 0 ? O_round.data() : nullptr,
                rcO.data(), dpO.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto odx = [&](int i, int j) { return i * dv + j; };
        int rows = std::min(N, N), cols = std::min(dv, dv);
        printf("[root] O (first %d rows × %d cols):\n", rows, cols);
        for (int i = 0; i < rows; ++i) {
            printf("row %2d:", i);
            for (int j = 0; j < cols; ++j) {
                printf(" %8.3f", O_round[odx(i, j)]);
            }
            printf("\n");
        }
        fflush(stdout);
    }

    // clean up
    cudaFree(dO);
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Comm_free(&local_comm);       
    MPI_Finalize();                   
    return 0;
}
