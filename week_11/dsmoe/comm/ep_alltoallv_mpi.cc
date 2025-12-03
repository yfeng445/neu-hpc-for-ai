// comm/ep_alltoallv_mpi.cc
#include "ep_comm.h"

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <climits>

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::abort(); }}while(0)

static int env_int(const char* k, int defv){
  const char* v = std::getenv(k);
  if(!v) return defv;
  try { return std::stoi(v); } catch(...) { return defv; }
}

static void dbg_ptr(const char* tag, const void* p){
  cudaPointerAttributes attr{};
#if CUDART_VERSION >= 10000
  if (cudaPointerGetAttributes(&attr, p) == cudaSuccess) {
    if (attr.type == cudaMemoryTypeDevice) {
      std::cerr<<"[ep_mpi] "<<tag<<" ptr="<<p<<" type=Device devId="<<attr.device<<"\n";
    } else if (attr.type == cudaMemoryTypeHost) {
      std::cerr<<"[ep_mpi] "<<tag<<" ptr="<<p<<" type=Host\n";
    } else {
      std::cerr<<"[ep_mpi] "<<tag<<" ptr="<<p<<" type=Unknown\n";
    }
  } else {
    std::cerr<<"[ep_mpi] "<<tag<<" ptr="<<p<<" type=? (attr query failed)\n";
  }
#else
  std::cerr<<"[ep_mpi] "<<tag<<" ptr="<<p<<"\n";
#endif
}

void ep_init_nccl(EpComm** out, int /*world_size_hint*/, int /*rank_hint*/) {
  int inited = 0;
  MPI_Initialized(&inited);

  bool owns = false;
  if (!inited) {
    int argc = 0; char** argv = nullptr;
    MPI_Init(&argc, &argv);
    owns = true;
  }

  EpComm* ep = new EpComm();
  ep->owns_mpi = owns;

  MPI_Comm base = MPI_COMM_WORLD;
  MPI_Comm dup  = MPI_COMM_NULL;
  MPI_Comm_dup(base, &dup);

  ep->comm = dup;
  MPI_Comm_rank(ep->comm, &ep->rank);
  MPI_Comm_size(ep->comm, &ep->world);

  *out = ep;
}

void ep_destroy_nccl(EpComm* ep) {
  if (!ep) return;
  MPI_Barrier(ep->comm);
  MPI_Comm_free(&ep->comm);
  if (ep->owns_mpi) {
    MPI_Finalize();
  }
  delete ep;
}

void ep_allgather_counts(const int64_t* d_sendcounts,
                         int64_t*       d_recvcounts,
                         EpComm*        ep,
                         cudaStream_t   stream)
{
  const int P = ep->world;
  const int r = ep->rank;

  std::vector<int64_t> h_send(P, 0), h_recv(P, 0);
  CUDA_CHECK(cudaMemcpyAsync(h_send.data(), d_sendcounts, sizeof(int64_t)*P,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  static_assert(sizeof(long long) == 8, "MPI_LONG_LONG must be 8 bytes");
  // 交换“每对peer的计数”：sendcounts[i] -> peer i
  // 收到：recvcounts[i] <- peer i
  MPI_Alltoall(h_send.data(), 1, MPI_LONG_LONG,
               h_recv.data(), 1, MPI_LONG_LONG,
               ep->comm);

  CUDA_CHECK(cudaMemcpyAsync(d_recvcounts, h_recv.data(), sizeof(int64_t)*P,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int dbg = env_int("EP_MPI_DEBUG", 0);
  if (dbg) {
    int64_t ssum = 0, rsum = 0;
    for (int i=0;i<P;++i){ ssum += h_send[i]; rsum += h_recv[i]; }
    std::cerr << "[ep_mpi] rank "<<r<<" counts-exchange elems ssum="<<ssum
              << " rsum="<<rsum<<"\n";
  }
}

void ep_alltoallv(const void* d_sendbuf, const int64_t* d_sendcounts,
                  void*       d_recvbuf, const int64_t* d_recvcounts,
                  int         elem_bytes,
                  EpComm*     ep,
                  cudaStream_t stream)
{
  const int P = ep->world;
  const int r = ep->rank;
  const int dbg = env_int("EP_MPI_DEBUG", 0);

  // --- D2H counts ---
  std::vector<int64_t> h_sc(P,0), h_rc(P,0);
  CUDA_CHECK(cudaMemcpyAsync(h_sc.data(), d_sendcounts, sizeof(int64_t)*P,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_rc.data(), d_recvcounts, sizeof(int64_t)*P,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 字节级别的 counts / displs
  std::vector<size_t> sb(P,0), rb(P,0);
  size_t ssum=0, rsum=0;
  for (int i=0;i<P;++i) {
    sb[i] = (size_t)h_sc[i] * (size_t)elem_bytes;
    rb[i] = (size_t)h_rc[i] * (size_t)elem_bytes;
    ssum += sb[i]; rsum += rb[i];
  }

  if (dbg) {
    std::cerr << "[ep_mpi] rank "<<r<<" P="<<P<<" elem_bytes="<<elem_bytes
              << " ssum="<<ssum<<" rsum="<<rsum<<"\n";
    std::cerr << "[ep_mpi] rank "<<r<<" sendcounts_elems head: ";
    for (int i=0;i<std::min(P,3);++i) std::cerr<<h_sc[i]<<" ";
    std::cerr << "\n[ep_mpi] rank "<<r<<" recvcounts_elems head: ";
    for (int i=0;i<std::min(P,3);++i) std::cerr<<h_rc[i]<<" ";
    std::cerr << "\n";
  }

  // 0 字节可直接返回
  if (ssum==0 && rsum==0) return;

  // OpenMPI 的 Alltoallv 使用 int 计数/位移
  std::vector<int> sdis(P,0), rdis(P,0), scount(P,0), rcount(P,0);
  {
    size_t off=0;
    for (int i=0;i<P;++i){ 
      if (sb[i] > (size_t)INT_MAX) { std::cerr<<"[ep_mpi] send bytes too large\n"; std::abort(); }
      scount[i] = (int)sb[i];
      sdis[i]   = (int)off;
      off      += sb[i];
    }
    off=0;
    for (int i=0;i<P;++i){
      if (rb[i] > (size_t)INT_MAX) { std::cerr<<"[ep_mpi] recv bytes too large\n"; std::abort(); }
      rcount[i] = (int)rb[i];
      rdis[i]   = (int)off;
      off      += rb[i];
    }
  }

  // host 中转缓冲
  std::vector<unsigned char> h_send(ssum ? ssum : 1);
  std::vector<unsigned char> h_recv(rsum ? rsum : 1);

  // --- D2H 数据 ---
  if (ssum) {
    dbg_ptr("dev d_sendbuf", d_sendbuf);
    CUDA_CHECK(cudaMemcpyAsync(h_send.data(), d_sendbuf, ssum, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (dbg) std::cerr << "[ep_mpi] rank "<<r<<" memcpy D2H bytes="<<ssum<<"\n";
  }

  // --- MPI_Alltoallv（按字节交换） ---
  // 注意：这里的 datatype 用 MPI_BYTE，counts/displs 单位=字节
  MPI_Alltoallv(h_send.data(), scount.data(), sdis.data(), MPI_BYTE,
                h_recv.data(), rcount.data(), rdis.data(), MPI_BYTE,
                ep->comm);
  if (dbg) {
    std::cerr << "[ep_mpi] rank "<<r<<" MPI_Alltoallv send_b[0]="<<(P?scount[0]:0)
              <<" recv_b[0]="<<(P?rcount[0]:0)<<"\n";
  }

  // --- H2D 数据 ---
  if (rsum) {
    dbg_ptr("dev d_recvbuf", d_recvbuf);
    CUDA_CHECK(cudaMemcpyAsync(d_recvbuf, h_recv.data(), rsum, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (dbg) std::cerr << "[ep_mpi] rank "<<r<<" memcpy H2D bytes="<<rsum<<"\n";
  }
}
