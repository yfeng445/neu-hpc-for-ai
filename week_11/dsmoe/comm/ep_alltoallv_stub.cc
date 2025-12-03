// comm/ep_alltoallv_stub.cc
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" void ep_alltoallv(
    const void* sendbuf,
    const int64_t* sendcounts,
    void* recvbuf,
    int64_t* recvcounts,
    int elem_bytes,
    void* /*ep_comm*/,
    cudaStream_t /*stream*/)
{
    // 读取 device 侧计数到 host（同步；host 非 pinned）
    int64_t cnt = 0;
    cudaError_t e = cudaMemcpy(&cnt, sendcounts, sizeof(int64_t), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return;

    // 回写 recvcounts（单卡直通）
    if (recvcounts) {
        cudaMemcpy(recvcounts, sendcounts, sizeof(int64_t), cudaMemcpyDeviceToDevice);
    }

    // 直通数据拷贝（D2D 同步）
    size_t bytes = (size_t)cnt * (size_t)elem_bytes;
    if (bytes && recvbuf != sendbuf) {
        cudaMemcpy(recvbuf, sendbuf, bytes, cudaMemcpyDeviceToDevice);
    }
}
