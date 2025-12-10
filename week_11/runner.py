# runner.py
# Build FlashMoE and the dsmoe demo in the same container image from the local Dockerfile.
# Usage:
#   modal run runner.py::build_and_run
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

# ---------------- Modal image & app ----------------
IMAGE = modal.Image.from_dockerfile(path="./Dockerfile")
APP = modal.App("runner")

# ---------------- helpers ----------------
def _run(cmd: str, cwd: str | os.PathLike | None = None, check: bool = True, env: dict | None = None):
    # Keep logs concise
    print("[runner] $ " + cmd)
    return subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=check, env=env)

def _ensure_dsmoe_cmakelists(ds_root: Path):
    cmake_txt = ds_root / "CMakeLists.txt"
    if cmake_txt.exists():
        return
    content = r"""
cmake_minimum_required(VERSION 3.27)
project(dsmoe_bridge LANGUAGES CXX CUDA)

# Standards & arch
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Reduce chatter
set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

# Bridge static library: only moe_fwd.cpp
add_library(dsmoe_bridge STATIC
  ${CMAKE_SOURCE_DIR}/moe_fwd.cpp
)

target_include_directories(dsmoe_bridge PUBLIC
  ${CMAKE_SOURCE_DIR}                      # for "moe_fwd.h"
  ${CMAKE_SOURCE_DIR}/../flashmoe/csrc/include
)

find_package(CUDAToolkit REQUIRED)

# NVSHMEM via config if available, fallback to explicit include/lib
find_package(nvshmem CONFIG PATHS /opt/nvshmem/lib/cmake/nvshmem NO_DEFAULT_PATH)
if(nvshmem_FOUND)
  target_link_libraries(dsmoe_bridge PUBLIC NVSHMEM::nvshmem NVSHMEM::nvshmem_host)
else()
  target_include_directories(dsmoe_bridge PUBLIC /opt/nvshmem/include)
  target_link_directories(dsmoe_bridge PUBLIC /opt/nvshmem/lib)
  target_link_libraries(dsmoe_bridge PUBLIC nvshmem nvshmem_host)
endif()

target_link_libraries(dsmoe_bridge PUBLIC CUDA::cudart CUDA::cudadevrt)
target_compile_definitions(dsmoe_bridge PUBLIC FMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0)

# Runtime search path consistent with the container layout
set_target_properties(dsmoe_bridge PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH   "/opt/nvidia-mathdx/lib;/opt/nvshmem/lib"
  INSTALL_RPATH "/opt/nvidia-mathdx/lib;/opt/nvshmem/lib"
)

# Demo executable (optional): compile dsmoe/main_moe.cpp and link the bridge
add_executable(dsmoe_demo ${CMAKE_SOURCE_DIR}/main_moe.cpp)
target_link_libraries(dsmoe_demo PRIVATE dsmoe_bridge)
set_target_properties(dsmoe_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "/workspace/out/bin"
)
"""
    cmake_txt.write_text(content, encoding="utf-8")

# ---------------- main build ----------------
@APP.function(image=IMAGE, timeout=60 * 60, gpu="A100-40GB:1")
def build_and_run():
    # paths
    src_root = Path("/workspace/src")
    fm_src_candidates = [
        src_root / "flashmoe" / "csrc",
        src_root / "flashmoe",
        src_root / "flashmoe" / "flashmoe",
    ]
    fm_src = next((p for p in fm_src_candidates if (p / "CMakeLists.txt").is_file()), None)
    if fm_src is None:
        _run("echo '[diag] flashmoe tree:' && ls -la /workspace/src/flashmoe || true", check=False)
        raise RuntimeError("flashmoe CMakeLists.txt not found under /workspace/src/flashmoe{,/csrc,/flashmoe}")

    # 1) Build flashmoe first (headers & libs available)
    fm_build = fm_src / "build"
    _run(f"rm -rf {shlex.quote(str(fm_build))} && mkdir -p {shlex.quote(str(fm_build))}")
    fm_cfg = (
        "cmake "
        f"-S {shlex.quote(str(fm_src))} "
        f"-B {shlex.quote(str(fm_build))} "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 "
        "-DCMAKE_CUDA_ARCHITECTURES=80 "
        "-DCMAKE_PREFIX_PATH=\"/opt/nvidia-mathdx;/opt/nvshmem\" "
        "-Dmathdx_ROOT=/opt/nvidia-mathdx "
        "-DNVSHMEM_DIR=/opt/nvshmem/lib/cmake/nvshmem "
        "-DCMAKE_INSTALL_PREFIX=/workspace/out "
        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/workspace/out/bin "
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/workspace/out/lib "
        "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=/workspace/out/lib "
        # keep quiet
        "-DCMAKE_RULE_MESSAGES=OFF -DCMAKE_MESSAGE_LOG_LEVEL=WARNING -Wno-dev "
        # safe compile defs
        "-DCMAKE_CUDA_FLAGS=\"-D__CUDA_ARCH_LIST__=800 -DKLEOS_ARCH=800 -DFMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0\" "
        "-DCMAKE_CXX_FLAGS=\"-DFMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0\" "
    )
    _run(fm_cfg)
    _run(f"cmake --build {shlex.quote(str(fm_build))} -j$(nproc)")
    _run("cmake --install " + shlex.quote(str(fm_build)) + " --prefix /workspace/out || true")
    print("[runner] flashmoe: build done.")

    # 2) Build dsmoe (bridge lib + demo)
    ds_root = src_root / "dsmoe"
    if not (ds_root / "moe_fwd.cpp").is_file():
        _run("echo '[diag] dsmoe tree:' && ls -la /workspace/src/dsmoe || true", check=False)
        raise RuntimeError("dsmoe/moe_fwd.cpp not found. Ensure your dsmoe folder is copied into /workspace/src.")
    _ensure_dsmoe_cmakelists(ds_root=ds_root)

    ds_build = ds_root / "build"
    _run(f"rm -rf {shlex.quote(str(ds_build))} && mkdir -p {shlex.quote(str(ds_build))}")
    ds_cfg = (
        "cmake "
        f"-S {shlex.quote(str(ds_root))} "
        f"-B {shlex.quote(str(ds_build))} "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 "
        "-DCMAKE_CUDA_ARCHITECTURES=80 "
        "-DCMAKE_PREFIX_PATH=\"/opt/nvshmem;/opt/nvidia-mathdx\" "
        "-DNVSHMEM_DIR=/opt/nvshmem/lib/cmake/nvshmem "
        "-DCMAKE_RULE_MESSAGES=OFF -DCMAKE_MESSAGE_LOG_LEVEL=WARNING -Wno-dev "
    )
    _run(ds_cfg)
    _run(f"cmake --build {shlex.quote(str(ds_build))} -j$(nproc)")
    print("[runner] dsmoe: build done.")

    # 3) Run demo quietly
    exe = Path("/workspace/out/bin/dsmoe_demo")
    if not exe.exists():
        # fallback: default cmake output dir
        exe = ds_build / "dsmoe_demo"
    _run(f"ls -l {shlex.quote(str(exe))}", check=False)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/nvidia-mathdx/lib:/opt/nvshmem/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["NVSHMEM_DISABLE_IB"] = "1"  # keep output clean in non-IB env
    nvshmrun = "/opt/nvshmem/bin/nvshmrun"
    if Path(nvshmrun).is_file():
        _run(f"{nvshmrun} -np 1 {shlex.quote(str(exe))}", env=env)
    else:
        _run(shlex.quote(str(exe)), env=env)

    print("[runner] done.")
