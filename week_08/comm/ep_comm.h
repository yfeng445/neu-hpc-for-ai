// comm/ep_comm.h
#pragma once
#include <cstdint>
#include <mpi.h>
#include <cuda_runtime_api.h>

struct EpComm {
  MPI_Comm comm;
  int      world;
  int      rank;
  bool     owns_mpi;
};

// 统一接口：MPI 实现也沿用 ep_init_nccl / ep_destroy_nccl 命名
void ep_init_nccl(EpComm** out, int world_size_hint, int rank_hint);
void ep_destroy_nccl(EpComm* ep);

// 交换每对 peer 的元素计数（单位：元素，不是字节）
// d_sendcounts / d_recvcounts 为 device 内存，长度均为 ep->world
void ep_allgather_counts(const int64_t* d_sendcounts,
                         int64_t*       d_recvcounts,
                         EpComm*        ep,
                         cudaStream_t   stream);

// 变长 All-to-all：sendbuf/recvbuf 为 device 连续缓冲；计数单位为“元素”
// elem_bytes 为元素字节数；内部按字节构造 MPI_Alltoallv 参数
void ep_alltoallv(const void*     d_sendbuf,
                  const int64_t*  d_sendcounts,
                  void*           d_recvbuf,
                  const int64_t*  d_recvcounts,
                  int             elem_bytes,
                  EpComm*         ep,
                  cudaStream_t    stream);
