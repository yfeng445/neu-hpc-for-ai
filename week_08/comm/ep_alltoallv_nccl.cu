// comm/ep_alltoallv_nccl.cu
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#ifndef NCCL_UNIQUE_ID_BYTES
// ncclUniqueId 实际上是 128 字节
#define NCCL_UNIQUE_ID_BYTES 128
#endif

#define CUDA_CHECK(cmd) do {                                 \
  cudaError_t e_ = (cmd);                                    \
  if (e_ != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                \
            __FILE__, __LINE__, cudaGetErrorString(e_));     \
    std::exit(1);                                            \
  }                                                          \
} while(0)

#define NCCL_CHECK(cmd) do {                                 \
  ncclResult_t r_ = (cmd);                                   \
  if (r_ != ncclSuccess) {                                   \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                \
            __FILE__, __LINE__, ncclGetErrorString(r_));     \
    std::exit(1);                                            \
  }                                                          \
} while(0)

struct EpComm {
  ncclComm_t comm = nullptr;
  int world = 1;
  int rank  = 0;
};

// ---- helpers: parse ncclUniqueId from env/file ----
static bool parse_hex_byte(char hi, char lo, unsigned char& out) {
  auto cvt = [](char c)->int {
    if ('0'<=c && c<='9') return c - '0';
    if ('a'<=c && c<='f') return 10 + (c - 'a');
    if ('A'<=c && c<='F') return 10 + (c - 'A');
    return -1;
  };
  int h = cvt(hi), l = cvt(lo);
  if (h<0 || l<0) return false;
  out = (unsigned char)((h<<4) | l);
  return true;
}

static bool load_unique_id_from_hex_env(ncclUniqueId* uid) {
  const char* hex = std::getenv("EP_NCCL_UNIQUE_ID_HEX");
  if (!hex) return false;
  size_t n = std::strlen(hex);
  if (n != NCCL_UNIQUE_ID_BYTES*2) {
    fprintf(stderr, "[ep_init_nccl] EP_NCCL_UNIQUE_ID_HEX length=%zu, expect %d hex chars\n",
            n, NCCL_UNIQUE_ID_BYTES*2);
    return false;
  }
  unsigned char* bytes = reinterpret_cast<unsigned char*>(uid);
  for (size_t i=0;i<NCCL_UNIQUE_ID_BYTES;++i) {
    unsigned char b;
    if (!parse_hex_byte(hex[2*i], hex[2*i+1], b)) {
      fprintf(stderr, "[ep_init_nccl] invalid hex at pos %zu\n", 2*i);
      return false;
    }
    bytes[i] = b;
  }
  return true;
}

static bool load_unique_id_from_file_env(ncclUniqueId* uid) {
  const char* path = std::getenv("EP_NCCL_UNIQUE_ID_FILE");
  if (!path) return false;
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[ep_init_nccl] cannot open file: %s\n", path);
    return false;
  }
  size_t r = std::fread(uid, 1, NCCL_UNIQUE_ID_BYTES, f);
  std::fclose(f);
  if (r != NCCL_UNIQUE_ID_BYTES) {
    fprintf(stderr, "[ep_init_nccl] read %zu bytes, expect %d\n", r, NCCL_UNIQUE_ID_BYTES);
    return false;
  }
  return true;
}

// ---- public API ----
extern "C" void ep_init_nccl(EpComm** handle, int world, int rank) {
  if (!handle) {
    fprintf(stderr, "[ep_init_nccl] handle=null\n");
    std::exit(1);
  }
  EpComm* h = new EpComm();
  h->world = world;
  h->rank  = rank;

  ncclUniqueId id;
  bool ok = load_unique_id_from_hex_env(&id) || load_unique_id_from_file_env(&id);

  if (!ok) {
    if (world == 1) {
      // 单进程/单卡，允许本地生成 id
      NCCL_CHECK(ncclGetUniqueId(&id));
      fprintf(stderr, "[ep_init_nccl] world=1: using local ncclUniqueId\n");
    } else {
      fprintf(stderr, "[ep_init_nccl] world=%d>1 but no EP_NCCL_UNIQUE_ID_{HEX|FILE} found.\n", world);
      std::exit(1);
    }
  }

  NCCL_CHECK(ncclCommInitRank(&h->comm, world, id, rank));
  *handle = h;
}

extern "C" void ep_destroy_nccl(EpComm* handle) {
  if (!handle) return;
  if (handle->comm) {
    ncclCommDestroy(handle->comm);
    handle->comm = nullptr;
  }
  delete handle;
}

// 把每个 rank 的 sendcounts[world] allgather 成 world×world，提取本 rank 的列作为 recvcounts
extern "C" void ep_allgather_counts(const int64_t* d_sendcounts,  // [world]
                                    int64_t*       d_recvcounts,  // [world]
                                    EpComm*        h,
                                    cudaStream_t   stream) {
  if (!h || !h->comm) {
    fprintf(stderr, "[ep_allgather_counts] invalid EpComm\n");
    std::exit(1);
  }
  const int world = h->world;
  // 临时 device 缓冲：world * world 个 int64
  int64_t* d_all = nullptr;
  CUDA_CHECK(cudaMalloc(&d_all, sizeof(int64_t) * world * world));

  // 每 rank 提供 d_sendcounts[world]，收集到 d_all[world*world]
  NCCL_CHECK(ncclAllGather(
      (const void*)d_sendcounts, (void*)d_all,
      world, ncclInt64, h->comm, stream));

  // 拷回 host，提取“本 rank 列”
  std::vector<int64_t> h_all(world * world, 0);
  CUDA_CHECK(cudaMemcpyAsync(h_all.data(), d_all, sizeof(int64_t)*world*world,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int64_t> h_recv(world, 0);
  for (int r=0; r<world; ++r) {
    // 第 r 行是 rank r 的 sendcounts[0..world-1]
    // 我们要拿它的第 'rank' 个元素
    h_recv[r] = h_all[r*world + h->rank];
  }

  CUDA_CHECK(cudaMemcpyAsync(d_recvcounts, h_recv.data(),
                             sizeof(int64_t)*world, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(d_all));
}

// 变长 all-to-allv：按字节发送（ncclUint8），count = 元素数 * elem_bytes
extern "C" void ep_alltoallv(const void* sendbuf, const int64_t* d_sendcounts,
                             void*       recvbuf, const int64_t* d_recvcounts,
                             int elem_bytes, EpComm* h, cudaStream_t stream) {
  if (!h || !h->comm) {
    fprintf(stderr, "[ep_alltoallv] invalid EpComm\n");
    std::exit(1);
  }
  const int world = h->world;

  // 把 counts 拷到 host
  std::vector<int64_t> h_s(world, 0), h_r(world, 0);
  CUDA_CHECK(cudaMemcpyAsync(h_s.data(), d_sendcounts, sizeof(int64_t)*world,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_r.data(), d_recvcounts, sizeof(int64_t)*world,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 前缀和 → 元素数偏移
  std::vector<uint64_t> sdisp(world+1, 0), rdisp(world+1, 0);
  for (int i=0;i<world;i++) sdisp[i+1] = sdisp[i] + (uint64_t)h_s[i];
  for (int i=0;i<world;i++) rdisp[i+1] = rdisp[i] + (uint64_t)h_r[i];

  // 以字节为单位的偏移
  auto offs_bytes = [&](uint64_t elems)->uint64_t { return elems * (uint64_t)elem_bytes; };

  const char* sb = reinterpret_cast<const char*>(sendbuf);
  char*       rb = reinterpret_cast<char*>(recvbuf);

  NCCL_CHECK(ncclGroupStart());
  for (int p=0; p<world; ++p) {
    const uint64_t sc_elems = (uint64_t)h_s[p];
    const uint64_t rc_elems = (uint64_t)h_r[p];
    const uint64_t sc_bytes = offs_bytes(sc_elems);
    const uint64_t rc_bytes = offs_bytes(rc_elems);
    if (sc_bytes > 0) {
      NCCL_CHECK(ncclSend((const void*)(sb + offs_bytes(sdisp[p])),
                          (size_t)sc_bytes, ncclUint8, p, h->comm, stream));
    }
    if (rc_bytes > 0) {
      NCCL_CHECK(ncclRecv((void*)(rb + offs_bytes(rdisp[p])),
                          (size_t)rc_bytes, ncclUint8, p, h->comm, stream));
    }
  }
  NCCL_CHECK(ncclGroupEnd());
}
