# runner.py — build helper for DS-MoE + FlashMoE
# Usage:
#   modal run runner.py::build_and_run_mpi --backend flash --arch sm_80
# Optional args:
#   --backend flash|mpi   # flash = fused FlashMoE path; mpi = legacy DS-MoE path
#   --np <int>            # number of ranks; default = visible GPU count
#   --arch sm_80|...      # compute capability (A100=sm_80, RTX30=sm_86, etc.)

import os
import subprocess
from pathlib import Path
import modal

app = modal.App("deepseekv3-moe")

# Source lists
MPI_SOURCES = [
    "utils/gemm.cu",
    "utils/sigmoid.cu",
    "utils/bias_add.cu",
    "router/group_top2_select.cu",
    "router/apply_group_mask.cu",
    "router/row_topk.cu",
    "router/gather_alpha.cu",
    "router/router_fwd.cpp",
    "dispatch/pack_routes.cu",
    "comm/ep_alltoallv_mpi.cc",  # only MPI backend
    "experts/experts_mlp_fwd.cu",
    "combine/combine_and_add.cu",
    "se/se_mlp_fwd.cu",
    "moe_fwd.cpp",
    "main_moe.cpp",
]

# FlashMoE adapter build: only host sources; FlashMoE headers provide kernels.
FLASH_SOURCES = [
    "moe_fwd.cpp",
    "main_moe.cpp",
]

IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
    .apt_install("build-essential", "openmpi-bin", "libopenmpi-dev")
    .add_local_dir(".", "/workspace/src")
)

def _run(cmd: str, env=None, check=True):
    print("[runner] $", cmd)
    return subprocess.run(["bash", "-lc", cmd], env=env, check=check)

@app.function(image=IMAGE, timeout=60*60, gpu="A100-40GB:4")
def build_and_run_mpi(
    arch: str = "sm_80",
    out: str = "test_moe",
    np: int | None = None,
    backend: str = "flash",
):
    os.chdir("/workspace/src")
    _run("mkdir -p build")

    backend = backend.lower()
    if backend not in {"flash", "mpi"}:
        raise ValueError("backend must be 'flash' or 'mpi'")

    # Ensure ep_comm.h include path matches sources
    if backend == "mpi" and not Path("comm/ep_comm.h").exists():
        if Path("ep_comm.h").exists():
            _run("mkdir -p comm && cp ep_comm.h comm/ep_comm.h")
        else:
            raise FileNotFoundError("Missing header: neither comm/ep_comm.h nor ep_comm.h present.")

    if backend == "flash":
        flash_root = Path("../flashmoe/csrc").resolve()
        flash_inc = flash_root / "include"
        nvshmem_home = os.environ.get("NVSHMEM_HOME", "/usr/local/nvshmem")
        nvshmem_lib = Path(nvshmem_home) / "lib"

        # Build FlashMoE adapter; FlashMoE headers are header-only but require NVSHMEM/CUTLASS at link time.
        nvcc = [
            "nvcc",
            "-O3",
            "-lineinfo",
            "-std=c++20",
            f"-arch={arch}",
            "-Xcompiler",
            "-fPIC",
        ]
        nvcc += [f"-I{flash_inc}", f"-I{flash_root}"]
        nvcc += FLASH_SOURCES

        # Link NVSHMEM if available; otherwise rely on caller-provided rpath/LD_LIBRARY_PATH.
        if nvshmem_lib.exists():
            nvcc += [f"-L{nvshmem_lib}", "-lnvshmem", "-lnvshmem_host", "-lnvshmem_device"]
        nvcc += ["-lcublas", "-lcudart", "-lcusolver", "-o", f"build/{out}"]

        _run(" ".join(nvcc))
        _run(f"ls -lh build/{out}")

        # FlashMoE path uses a single-process host launcher by default.
        _run(f"./build/{out}")
        return

    # Compile (MPI backend). Use mpicxx as host compiler so mpi.h is found.
    nvcc = ["nvcc", "-O3", "-lineinfo", "-std=c++17", f"-arch={arch}", "-ccbin=mpicxx"]
    nvcc += MPI_SOURCES
    nvcc += ["-lmpi", "-o", f"build/{out}"]
    _run(" ".join(nvcc))
    _run(f"ls -lh build/{out}")

    # Determine rank count (default = visible GPUs)
    try:
        g = int(subprocess.run(
            ["bash","-lc","nvidia-smi -L | wc -l"],
            check=True, capture_output=True, text=True
        ).stdout.strip())
    except Exception:
        g = 1
    n = g if np is None else np
    print(f"[runner] GPUs={g}, np={n}")

    # Runtime env: avoid UCX; use ob1 + vader/self; host staging (non CUDA-aware)
    env = os.environ.copy()
    env.update({
        "CUDA_LAUNCH_BLOCKING": "1",
        "EP_MPI_CUDA_AWARE": "0",                 # force host staging in comm/ep_alltoallv_mpi.cc
        "OMPI_MCA_pml": "ob1",                    # do not use UCX
        "OMPI_MCA_btl_base_warn_component_unused": "0",
    })

    # mpirun: pick shared-memory transports, no network dependency
    mpicmd = (
        f"mpirun --allow-run-as-root -np {n} "
        f"--mca pml {env['OMPI_MCA_pml']} "
        f"--mca btl self,vader "
        f"--mca btl_base_warn_component_unused {env['OMPI_MCA_btl_base_warn_component_unused']} "
        f"-x WORLD_SIZE={n} "
        f"-x CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING "
        f"-x EP_MPI_CUDA_AWARE=$EP_MPI_CUDA_AWARE "
        f"-x EP_MPI_DEBUG=0 "
        "bash -lc 'export RANK=$OMPI_COMM_WORLD_RANK; "
        "export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; "
        "./build/{out}'"
    ).format(out=out)

    _run(mpicmd, env=env)
