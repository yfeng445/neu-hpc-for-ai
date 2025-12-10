# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# 基础依赖：
# - xz-utils: 支持 tar -xJf
# - python-is-python3: 提供 /usr/bin/python -> /usr/bin/python3 的别名，满足 Modal 的 "python -m pip ..."
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git wget curl ca-certificates \
    python3 python3-pip python-is-python3 \
    libnuma-dev libibverbs-dev \
    xz-utils \
    libboost-dev \
 && rm -rf /var/lib/apt/lists/*

# 安装 Kitware 官方仓库的新版 CMake（Ubuntu 22.04 默认 3.22 太旧）
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates gnupg wget && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /usr/share/keyrings/kitware.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
      > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y --no-install-recommends cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /workspace/src

# 归档文件放进构建环境（文件名需与本地一致）
COPY libnvshmem-linux-x86_64-3.3.20_cuda12-archive.tar.xz /workspace/src/
COPY nvidia-mathdx-25.06.1-cuda12.tar.gz                 /workspace/src/
# 把本地源码打进镜像（确保 Docker 构建上下文包含这些目录）
COPY flashmoe /workspace/src/flashmoe
COPY dsmoe    /workspace/src/dsmoe
COPY runner.py /root/runner.py

# 纯 POSIX 解压与部署（无 bash 语法）
RUN set -eux \
 && rm -rf /opt/nvshmem /opt/nvidia-mathdx /opt/nvidia-mathdx-unpacked \
 && mkdir -p /opt \
 && tar -xJf /workspace/src/libnvshmem-linux-x86_64-3.3.20_cuda12-archive.tar.xz -C /opt \
 # 查找 nvshmem 解压目录
 && found_dir="" \
 && for d in /opt/libnvshmem*cuda12*archive; do \
      [ -d "$d" ] || continue; found_dir="$d"; break; \
    done \
 && if [ -n "$found_dir" ]; then mkdir -p /opt/nvshmem && mv "$found_dir"/* /opt/nvshmem && rmdir "$found_dir"; fi \
 # 解压 MathDX 并创建稳定软链
 && mkdir -p /opt/nvidia-mathdx-unpacked \
 && tar -xzf /workspace/src/nvidia-mathdx-25.06.1-cuda12.tar.gz -C /opt/nvidia-mathdx-unpacked \
 && ln -sfn "/opt/nvidia-mathdx-unpacked/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06" "/opt/nvidia-mathdx"

# 供 CMake/运行期查找
ENV NVSHMEM_HOME=/opt/nvshmem
ENV MATHDX_HOME=/opt/nvidia-mathdx
ENV NVSHMEM_DIR=/opt/nvshmem/lib/cmake/nvshmem

# 绝对路径，不做自引用，避免 Modal 预解析失败
ENV CPATH=/opt/nvshmem/include:/opt/nvidia-mathdx/include
ENV LIBRARY_PATH=/opt/nvshmem/lib:/opt/nvidia-mathdx/lib64
ENV LD_LIBRARY_PATH=/opt/nvshmem/lib:/opt/nvidia-mathdx/lib64
ENV CMAKE_PREFIX_PATH=/opt/nvshmem:/opt/nvidia-mathdx




# （可选）缩小镜像体积
# RUN rm -f /workspace/src/libnvshmem-linux-x86_64-3.3.20_cuda12-archive.tar.xz \
#        /workspace/src/nvidia-mathdx-25.06.1-cuda12.tar.gz
