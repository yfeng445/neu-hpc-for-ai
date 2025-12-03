# env_probe.py
# modal run env_probe.py::probe_default
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Any
import modal

app = modal.App("env-probe")

IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
    .apt_install("build-essential", "openmpi-bin", "libopenmpi-dev")
    .add_local_dir(".", "/workspace/src")
)

def _run(cmd: str) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True, text=True, check=False
        )
        out = (p.stdout or "") + (p.stderr or "")
        # 避免输出过长
        if len(out) > 4000:
            out = out[:4000] + "\n...[truncated]..."
        return {"cmd": cmd, "rc": p.returncode, "out": out}
    except Exception as e:
        return {"cmd": cmd, "rc": -999, "out": f"<EXCEPTION> {e}"}

@app.function(image=IMAGE, timeout=30*60, gpu="A100-40GB:4")
def probe_default(np: int = 2) -> Dict[str, Any]:
    """
    全默认路径的环境探针：
      1) 基础工具可用性
      2) 网络与 GPU 可见性
      3) 直接 mpirun -np 2 hostname（不加任何 env/mca）
    返回一个 dict，便于 `modal call` 查看。
    """
    os.chdir("/workspace/src")
    report: Dict[str, Any] = {"np": np, "steps": []}

    # 1) 基础工具
    report["steps"].append(_run("which mpirun || true"))
    report["steps"].append(_run("which mpicxx || true"))
    report["steps"].append(_run("mpirun --version || true"))
    report["steps"].append(_run("ompi_info --version || true"))

    # 简要列出可用 PML/BTL（若 ompi_info 不可用会显示错误）
    report["steps"].append(_run("ompi_info --parsable --all | grep -i 'mca:pml' | head -n 20 || true"))
    report["steps"].append(_run("ompi_info --parsable --all | grep -i 'mca:btl' | head -n 50 || true"))

    # 2) 系统资源可见性
    report["steps"].append(_run("test -d /sys/class/net && ls -l /sys/class/net || echo '/sys/class/net not present'"))
    report["steps"].append(_run("nvidia-smi -L || true"))

    # 3) 默认 mpirun smoke test（不设任何 env/mca）
    # 每个 rank 打印：RANK、LOCAL_RANK、HOSTNAME
    mpicmd = (
        f"mpirun --allow-run-as-root -np {np} "
        "bash -lc 'echo RANK=$OMPI_COMM_WORLD_RANK LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK HOST=$(hostname)'"
    )
    report["steps"].append(_run(mpicmd))

    # 返回整体报告
    print(report)
    
    return report
