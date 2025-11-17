# modal run ./runner.py::fa

import os
import subprocess
import modal
import time

app = modal.App("73755e1e-3e2f-4f6b-8f7b-5f3b3e3c4d2a")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
        .apt_install("build-essential")        # 单卡不再需要 openmpi
        .add_local_dir("./src", "/workspace/src")
)


def _gpu_count() -> int:
    """简单检查容器里可见 GPU 数量，用来做 sanity check。"""
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "nvidia-smi -L | wc -l"],
            text=True,
        )
        n = int(out.strip())
        return max(n, 0)
    except Exception:
        vis = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
        if vis and vis != "all":
            return len([x for x in vis.split(",") if x.strip() != ""])
        return 0


@app.function(image=image, gpu="A100-40GB", timeout=60 * 30)
def fa(src_name: str = "flash_attention.cu"):
    # 1. 编译：纯 nvcc，单进程 single-GPU 不再需要 mpicxx / -lmpi
    src_path = f"/workspace/src/{src_name}"
    subprocess.check_call(
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_80,code=sm_80",
            src_path,
            "-o",
            "/workspace/fa",
        ]
    )

    # 2. 检查 GPU
    avail = _gpu_count()
    print(f"[runner] visible GPUs = {avail}")
    if avail < 1:
        print("[runner] No GPUs visible in container; skipping run.")
        return

    # 3. 单卡运行：固定使用第 0 号 GPU
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 如果外部没限制，这里显式锁到 0 号卡
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print("\n" + "=" * 70)
    print("[runner] Launching single-GPU run on CUDA device 0")
    print("=" * 70)
    t0 = time.time()
    subprocess.check_call(["/workspace/fa"], env=env)
    print(f"[runner] Finished single-GPU run in {time.time() - t0:.3f} seconds.")


@app.function(image=image, timeout=600)
def sh(cmd: str = "python -V; nvcc --version; which python"):
    subprocess.check_call(["bash", "-lc", cmd])
