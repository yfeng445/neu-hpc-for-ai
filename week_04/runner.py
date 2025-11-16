# modal run ./runner.py::fa

import os, subprocess, modal, shlex, time

app = modal.App("wk4")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
        .apt_install("build-essential", "openmpi-bin", "libopenmpi-dev")
        .add_local_dir("./src", "/workspace/src")
)

def _gpu_count() -> int:
    try:
        out = subprocess.check_output(["bash", "-lc", "nvidia-smi -L | wc -l"], text=True)
        n = int(out.strip())
        return max(n, 0)
    except Exception:
        vis = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
        if vis and vis != "all":
            return len([x for x in vis.split(",") if x.strip() != ""])
        return 0

@app.function(image=image, gpu="A100-40GB", timeout=60*30)
def fa(src_name: str = "flash_attention.cu", np_max: int = 8):
    src_path = f"/workspace/src/{src_name}"
    subprocess.check_call([
        "nvcc", "-ccbin", "mpicxx", "-O3", "-std=c++17",
        "-gencode=arch=compute_80,code=sm_80",
        src_path,
        "-o", "/workspace/mpi", "-lmpi"
    ])

    avail = _gpu_count()
    print(f"[runner] visible GPUs = {avail}")

    lim = min(np_max, avail) if avail > 0 else 0
    if lim == 0:
        print("[runner] No GPUs visible in container; skipping run.")
        return

    base_env = os.environ.copy()
    base_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    for np in range(1, lim + 1):
        
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(np))

        print("\n" + "=" * 70)
        print(f"[runner] Launching mpirun with np={np} on devices {env['CUDA_VISIBLE_DEVICES']}")
        
        print("=" * 70)
        t0 = time.time()
        subprocess.check_call([
            "mpirun", "--allow-run-as-root",
            "-np", str(np),
            "--bind-to", "none",
            "/workspace/mpi"
        ], env=env)
        print(f"[runner] Finished nprocs={np} in {time.time() - t0:.3f} seconds.")

@app.function(image=image, timeout=600)
def sh(cmd: str = "python -V; nvcc --version; mpirun --version; which python"):
    subprocess.check_call(["bash", "-lc", cmd])
