#include <cuda_runtime.h>
#include <stdint.h>
#include <nccl.h>
#include <vector>

static inline void host_prefix_sum(const int64_t* in, int n, std::vector<size_t>& off) {
    off.resize(n+1);
    size_t acc = 0;
    for (int i=0;i<n;++i){ off[i] = acc; acc += (size_t)in[i]; }
    off[n] = acc;
}

extern "C" void ep_alltoallv(
    const void* sendbuf,
    const int64_t* sendcounts_dev,
    void* recvbuf,
    int64_t* recvcounts_dev,
    int elem_bytes,
    void* ep_comm,
    cudaStream_t stream)
{
    ncclComm_t comm = (ncclComm_t)ep_comm;
    int nranks = 1, myrank = 0;
    ncclCommCount(comm, &nranks);
    ncclCommUserRank(comm, &myrank);

    if (nranks == 1) {
        int64_t cnt = 0;
        cudaMemcpyAsync(&cnt, sendcounts_dev, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (recvcounts_dev) {
            cudaMemcpyAsync(recvcounts_dev, sendcounts_dev, sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
        }
        size_t bytes = (size_t)cnt * (size_t)elem_bytes;
        if (bytes && recvbuf != sendbuf) {
            cudaMemcpyAsync(recvbuf, sendbuf, bytes, cudaMemcpyDeviceToDevice, stream);
        }
        return;
    }

    size_t counts_bytes = (size_t)nranks * sizeof(int64_t);
    int64_t* counts_mat_dev = nullptr;
    cudaMallocAsync(&counts_mat_dev, (size_t)nranks * (size_t)nranks * sizeof(int64_t), stream);

    ncclAllGather((const void*)sendcounts_dev,
                  (void*)counts_mat_dev,
                  nranks,
                  ncclInt64,
                  comm,
                  stream);

    std::vector<int64_t> sendcounts_host(nranks);
    cudaMemcpyAsync(sendcounts_host.data(), sendcounts_dev, counts_bytes, cudaMemcpyDeviceToHost, stream);

    std::vector<int64_t> counts_mat_host((size_t)nranks * (size_t)nranks);
    cudaMemcpyAsync(counts_mat_host.data(), counts_mat_dev,
                    (size_t)nranks * (size_t)nranks * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<int64_t> recvcounts_host(nranks);
    for (int r=0; r<nranks; ++r) {
        recvcounts_host[r] = counts_mat_host[(size_t)r * nranks + myrank];
    }
    if (recvcounts_dev) {
        cudaMemcpyAsync(recvcounts_dev, recvcounts_host.data(), counts_bytes, cudaMemcpyHostToDevice, stream);
    }

    std::vector<size_t> send_off, recv_off;
    host_prefix_sum(sendcounts_host.data(), nranks, send_off);
    host_prefix_sum(recvcounts_host.data(), nranks, recv_off);

    ncclGroupStart();
    for (int r=0; r<nranks; ++r) {
        size_t sb = send_off[r] * (size_t)elem_bytes;
        size_t rb = recv_off[r] * (size_t)elem_bytes;
        size_t scb = (size_t)sendcounts_host[r] * (size_t)elem_bytes;
        size_t rcb = (size_t)recvcounts_host[r] * (size_t)elem_bytes;
        if (scb) ncclSend((const char*)sendbuf + sb, (size_t)scb, ncclUint8, r, comm, stream);
        if (rcb) ncclRecv((char*)recvbuf + rb, (size_t)rcb, ncclUint8, r, comm, stream);
    }
    ncclGroupEnd();

    cudaFreeAsync(counts_mat_dev, stream);
}
