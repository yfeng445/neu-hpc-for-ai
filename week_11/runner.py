# runner.py
# Build FlashMoE inside the container image built from the local Dockerfile.
# Run: modal run runner.py::build_and_run

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

# -------- Modal: Image and Stub --------
# Legacy builder 不支持 dockerfile=... 参数；仅传 path="."，默认使用 ./Dockerfile
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
    when gpu-query fails. No-op if file doesn't exist.
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

# -------- Build --------
@APP.function(image=IMAGE, timeout=60 * 60, gpu="A100-40GB:4")
def build_and_run():
    # 0) Quick diagnostics
    _run("echo \"[diag] nvcc: $(command -v nvcc || echo missing)\" && nvcc --version || true", check=False)
    _run("echo \"[diag] cmake: $(command -v cmake || echo missing)\" && cmake --version || true", check=False)
    _run("echo \"[diag] python: $(command -v python || echo missing)\" && python -V || true", check=False)
    _run("echo \"[diag] ldconfig nvshmem/mathdx\" && (ldconfig -p | egrep 'nvshmem|mathdx' || true)", check=False)

    # 1) Locate flashmoe source root (must contain CMakeLists.txt)
    candidates = [
        Path("/workspace/src/flashmoe"),            # expected after Dockerfile COPY
        Path("/workspace/src/flashmoe/csrc"),
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
        # 如果项目的 CMakeLists 确实消费这个变量，再开启；否则先不要强行干预
        # "-Dsequence_len=128 "
    )

    _run(cmake_cfg)
    _run(f"cmake --build {shlex.quote(str(build_dir))} -j$(nproc)")

    # 4) Artifacts snapshot
    _run(f"echo '[diag] build tree' && find {shlex.quote(str(build_dir))} -maxdepth 3 -type f | sort | sed -n '1,200p'")

    print("[runner] Build completed successfully.")
