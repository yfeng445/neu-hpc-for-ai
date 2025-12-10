# runner.py
# Build FlashMoE inside the container image built from the local Dockerfile.
# Run: modal run runner.py::build_and_run

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
import re

import modal

# -------- Modal: Image and App --------
# 与当前 SDK/legacy builder 兼容：显式指向本地 Dockerfile
IMAGE = modal.Image.from_dockerfile(path="./Dockerfile")
APP = modal.App("runner")

# -------- Utils --------
def _run(cmd: str, cwd: str | os.PathLike | None = None, check: bool = True, env: dict | None = None):
    """Run a shell command with a log prefix."""
    print("[runner] $ " + cmd)
    return subprocess.run(["bash", "-lc", cmd], cwd=cwd, check=check, env=env)

def _patch_sm_detector(sm_cmake_path: Path, default_sms: int = 80):
    """
    Optional: patch SM.cmake (if it exists) to fall back to a fixed NUM_SMS
    when gpu-query fails. No-op if the file doesn't exist.
    """
    if not sm_cmake_path.is_file():
        return
    try:
        txt = sm_cmake_path.read_text(encoding="utf-8", errors="ignore")
        if "NUM_SMS" in txt and "FALLBACK" in txt:
            return  # already patched
        patched = txt + f"""

# ---- Auto-injected fallback (harmless if unused) ----
if(NOT DEFINED NUM_SMS)
  message(STATUS "SM.cmake: NUM_SMS not provided; using FALLBACK={default_sms}")
  set(NUM_SMS {default_sms})
endif()
"""
        sm_cmake_path.write_text(patched, encoding="utf-8")
        print(f"[runner] Patched {sm_cmake_path} with NUM_SMS fallback={default_sms}.")
    except Exception as e:
        print(f"[runner] Warn: failed to patch {sm_cmake_path}: {e}")

def _find_first(path: Path, pattern: str) -> Path | None:
    """Return first match under path.glob(pattern) or None."""
    for p in path.glob(pattern):
        return p
    return None

# -------- Build --------
@APP.function(image=IMAGE, timeout=60 * 60, gpu="A100-40GB:4")
def build_and_run():
    # 0) Quick diagnostics
    _run('echo "[diag] nvcc: $(command -v nvcc || echo missing)" && nvcc --version || true', check=False)
    _run('echo "[diag] cmake: $(command -v cmake || echo missing)" && cmake --version || true', check=False)
    _run('echo "[diag] python: $(command -v python || echo missing)" && python -V || true', check=False)
    _run('echo "[diag] ldconfig nvshmem/mathdx" && (ldconfig -p | egrep "nvshmem|mathdx" || true)', check=False)

    # 1) Locate flashmoe source root (must contain CMakeLists.txt) —— 优先 csrc
    candidates = [
        Path("/workspace/src/flashmoe/csrc"),
        Path("/workspace/src/flashmoe"),
        Path("/workspace/src/flashmoe/flashmoe"),
    ]
    fm_src: Path | None = None
    for p in candidates:
        if (p / "CMakeLists.txt").is_file():
            fm_src = p
            break
    if fm_src is None:
        _run("echo '[diag] list /workspace/src' && ls -la /workspace/src || true", check=False)
        _run("echo '[diag] list /workspace/src/flashmoe' && ls -la /workspace/src/flashmoe || true", check=False)
        raise RuntimeError(
            "CMakeLists.txt not found under /workspace/src/flashmoe{,/csrc,/flashmoe}. "
            "Ensure Dockerfile has `COPY flashmoe /workspace/src/flashmoe`."
        )

    # 2) (Optional) Patch SM.cmake fallbacks if present
    for sm_path in [
        fm_src / "cmake" / "SM.cmake",
        fm_src / "csrc" / "cmake" / "SM.cmake",
    ]:
        _patch_sm_detector(sm_path, default_sms=80)

    # 3) Configure & build
    build_dir = fm_src / "build"
    _run(f"rm -rf {shlex.quote(str(build_dir))} && mkdir -p {shlex.quote(str(build_dir))}")

    cmake_cfg = (
        "cmake "
        f"-S {shlex.quote(str(fm_src))} "
        f"-B {shlex.quote(str(build_dir))} "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 "
        "-DCMAKE_CUDA_ARCHITECTURES=80 "
        "-DCMAKE_PREFIX_PATH=\"/opt/nvidia-mathdx;/opt/nvshmem\" "
        "-Dmathdx_ROOT=/opt/nvidia-mathdx "
        "-DNVSHMEM_DIR=/opt/nvshmem/lib/cmake/nvshmem "
        "-DCMAKE_INSTALL_RPATH=\"/opt/nvidia-mathdx/lib;/opt/nvshmem/lib\" "
        "-DCMAKE_EXE_LINKER_FLAGS=\"-Wl,-rpath,/opt/nvidia-mathdx/lib:/opt/nvshmem/lib\" "
        # 仅保留与编译器/库兼容相关的宏，去掉 NUM_SMS / SEQ_LEN，避免 redefined
        "-DCMAKE_CUDA_FLAGS="
        "\"-D__CUDA_ARCH_LIST__=800 -DKLEOS_ARCH=800 -DFMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0\" "
        "-DCMAKE_CXX_FLAGS=\"-DFMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0\" "
        # 安装/产物目录
        "-DCMAKE_INSTALL_PREFIX=/workspace/out "
        "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/workspace/out/bin "
        "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/workspace/out/lib "
        "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=/workspace/out/lib "
    )
    _run(cmake_cfg)
    _run(f"cmake --build {shlex.quote(str(build_dir))} -j$(nproc)")
    _run("cmake --install " + shlex.quote(str(build_dir)) + " --prefix /workspace/out || true")

    print("[runner] Build completed successfully.")

    # ---- 手工用 nvcc 链接 main.cu 的目标并运行（方案 A：稳、最少假设） ----
    nvcc = "/usr/local/cuda/bin/nvcc"
    exe_dir = Path("/workspace/out/bin")
    exe_dir.mkdir(parents=True, exist_ok=True)
    exe_path = exe_dir / "flashmoe_demo"

    # 常见目标名为 csrc.dir；若不存在则自动探测 *.dir 里含 main.cu.o 的目录
    csrc_obj = build_dir / "CMakeFiles" / "csrc.dir" / "main.cu.o"
    devlink_obj = build_dir / "CMakeFiles" / "csrc.dir" / "cmake_device_link.o"
    if not csrc_obj.exists():
        # 回退：自动找 main.cu.o
        alt = _find_first(build_dir / "CMakeFiles", "*/*.dir/main.cu.o")
        if alt:
            csrc_obj = alt
            devlink_obj = alt.parent / "cmake_device_link.o"

    # 诊断与存在性检查
    _run(f"ls -l {shlex.quote(str(csrc_obj))} {shlex.quote(str(devlink_obj))}", check=False)
    if not csrc_obj.exists() or not devlink_obj.exists():
        _run(f"echo '[diag] no main.cu.o or cmake_device_link.o; list .o' && "
             f"find {shlex.quote(str(build_dir))} -name '*.o' -maxdepth 4 -print | sed -n '1,200p'", check=False)
        raise RuntimeError("Cannot locate main.cu.o / cmake_device_link.o; demo link step aborted.")

    # 用 nvcc 做最终链接（自动带上 cudart/cudadevrt），并链接 NVSHMEM
    link_cmd = (
        f"{nvcc} -std=c++20 -O3 "
        f"{shlex.quote(str(csrc_obj))} {shlex.quote(str(devlink_obj))} "
        "-L/opt/nvshmem/lib -lnvshmem -lnvshmem_host "
        # rpath 需要逐项通过 -Xlinker 传给链接器
        "-Xlinker -rpath -Xlinker /opt/nvshmem/lib "
        "-Xlinker -rpath -Xlinker /opt/nvidia-mathdx/lib "
        # 保险起见把 CUDA 运行时也显式带上（nvcc 通常会自动拉，但显式更稳）
        "-lcudadevrt -lcudart "
        f"-o {shlex.quote(str(exe_path))}"
    )

    print(f"[runner] nvcc link -> {exe_path}")
    # 关键：在 build_dir 下执行，确保相对包含路径等保持一致
    _run(link_cmd, cwd=str(build_dir))

    # 运行（优先 nvshmrun；否则直接运行）
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/nvidia-mathdx/lib:/opt/nvshmem/lib:" + env.get("LD_LIBRARY_PATH", "")
    _run(f"ls -l {shlex.quote(str(exe_path))}")
    nvshmrun = "/opt/nvshmem/bin/nvshmrun"
    if Path(nvshmrun).is_file():
        _run(f"{nvshmrun} -np 1 {shlex.quote(str(exe_path))}", env=env)
    else:
        _run(shlex.quote(str(exe_path)), env=env)
