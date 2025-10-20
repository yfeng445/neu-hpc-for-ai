// fa2_fwd_mp.cu
//
// Purpose: Minimal MPI + CUDA smoke test (NOT FA2). Verifies that
// (1) MPI launches multiple ranks,
// (2) each rank binds to a distinct GPU,
// (3) a trivial CUDA kernel runs on that GPU,
// (4) cross-rank communication works (MPI_Allreduce).
//
// Build (A100 example):
//   nvcc -ccbin mpicxx -O3 -std=c++17 \
//        -gencode=arch=compute_80,code=sm_80 \
//        fa2_fwd_mp.cu -o fa2_fwd_mp -lmpi
//
// Fatbin (A100 + Ada L40/L40s; add/remove as needed):
//   nvcc -ccbin mpicxx -O3 -std=c++17 \
//        -gencode=arch=compute_80,code=sm_80 \
//        -gencode=arch=compute_89,code=sm_89 \
//        fa2_fwd_mp.cu -o fa2_fwd_mp -lmpi
//
// Run (single node, 2 ranks, 1 GPU per rank):
//   mpirun --allow-run-as-root -np 2 ./fa2_fwd_mp
// or Slurm:
//   srun -N1 -n2 --gpus-per-task=1 ./fa2_fwd_mp

#include <mpi.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ---------- CUDA error check ----------
#define CK_CUDA(CALL)                                                                
  do {                                                                               
    cudaError_t _err = (CALL);                                                       
    if (_err != cudaSuccess) {                                                       
      fprintf(stderr, "[cuda] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err));
      MPI_Abort(MPI_COMM_WORLD, 1);                                                  
    }                                                                                
  } while (0)

// ---------- trivial GPU kernel ----------
__global__ void write_val(float* out, float v) {
  if (threadIdx.x == 0) *out = v;  // one thread writes a scalar
}

int main(int argc, char** argv) {
  // ----- MPI init -----
  MPI_Init(&argc, &argv);
  int world_rank = -1, world_size = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Identify processor/node (portable, MPI-native)
  char hostname[MPI_MAX_PROCESSOR_NAME + 1];
  int name_len = 0;
  MPI_Get_processor_name(hostname, &name_len);
  hostname[name_len] = '\0';

  // Useful for debugging device mapping inside containers
  const char* vis = std::getenv("CUDA_VISIBLE_DEVICES");
  if (!vis) vis = "(unset)";

  // ----- CUDA device discovery -----
  int ndev = 0;
  CK_CUDA(cudaGetDeviceCount(&ndev));
  if (ndev == 0) {
    if (world_rank == 0) {
      fprintf(stderr, "[fatal] No CUDA devices visible. Check --gres or container GPU request.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (ndev < world_size && world_rank == 0) {
    fprintf(stderr,
            "[warn] Only %d GPU(s) visible but %d MPI ranks; mapping will wrap (rank %% ndev).\n",
            ndev, world_size);
  }

  // Map rank -> device (simple wraparound)
  int local_dev = world_rank % ndev;
  CK_CUDA(cudaSetDevice(local_dev));

  // Device info (model name + PCI bus id)
  cudaDeviceProp prop{};
  CK_CUDA(cudaGetDeviceProperties(&prop, local_dev));
  char busid[32];
  CK_CUDA(cudaDeviceGetPCIBusId(busid, sizeof(busid), local_dev));






  

  // ----- trivial CUDA work on this GPU -----
  float v_host = NAN, v_global = NAN;
  float* d_val = nullptr;
  CK_CUDA(cudaMalloc(&d_val, sizeof(float)));

  // Each rank writes a unique scalar (100.0, 200.0, 300.0, ...)
  float my_scalar = 100.0f * (world_rank + 1);
  write_val<<<1, 32>>>(d_val, my_scalar);
  CK_CUDA(cudaPeekAtLastError());
  CK_CUDA(cudaDeviceSynchronize());

  CK_CUDA(cudaMemcpy(&v_host, d_val, sizeof(float), cudaMemcpyDeviceToHost));
  CK_CUDA(cudaFree(d_val));

  // ----- cross-rank communication -----
  MPI_Allreduce(&v_host, &v_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // ----- structured log (one line per rank) -----
  // Example:
  // [rank 0/2 host=... gpu=0 name="NVIDIA A100-SXM4-80GB" bus=0000:17:00.0 vis=0,1] ndev=2 wrote=100.0 sum_all=300.0 ok
  printf("[rank %d/%d host=%s gpu=%d name=\"%s\" bus=%s vis=%s] "
         "ndev=%d wrote=%.1f sum_all=%.1f ok\n",
         world_rank, world_size, hostname, local_dev,
         prop.name, busid, vis, ndev, v_host, v_global);
  fflush(stdout);

  MPI_Finalize();
  return 0;
}
