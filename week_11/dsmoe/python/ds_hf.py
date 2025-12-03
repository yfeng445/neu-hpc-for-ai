# runner_hf_moe.py
import os, sys, types, importlib.util, torch, torch.nn as nn
import transformers

def try_import_from_transformers():
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3MoE, DeepseekV3TopkRouter, DeepseekV3MLP
        )
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
            DeepseekV3Config
        )
        return DeepseekV3MoE, DeepseekV3TopkRouter, DeepseekV3MLP, DeepseekV3Config, "transformers"
    except Exception as e:
        return None, None, None, None, str(e)

def fallback_import_local_modeling(path="modeling_deepseek_v3.py"):
    # 构造最小包结构，支持相对导入
    pkg_root = types.ModuleType("transformers")
    models_pkg = types.ModuleType("transformers.models")
    ds_pkg = types.ModuleType("transformers.models.deepseek_v3")
    cfg_mod = types.ModuleType("transformers.models.deepseek_v3.configuration_deepseek_v3")

    class DeepseekV3Config:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    setattr(cfg_mod, "DeepseekV3Config", DeepseekV3Config)
    sys.modules["transformers"] = pkg_root
    sys.modules["transformers.models"] = models_pkg
    sys.modules["transformers.models.deepseek_v3"] = ds_pkg
    sys.modules["transformers.models.deepseek_v3.configuration_deepseek_v3"] = cfg_mod

    here = os.path.dirname(os.path.abspath(__file__))
    modeling_path = os.path.join(here, path)
    spec = importlib.util.spec_from_file_location(
        "transformers.models.deepseek_v3.modeling_deepseek_v3", modeling_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.DeepseekV3MoE, mod.DeepseekV3TopkRouter, mod.DeepseekV3MLP, cfg_mod.DeepseekV3Config, "local"

# 1) 优先从 transformers 导入，失败则回退到本地文件
DeepseekV3MoE, DeepseekV3TopkRouter, DeepseekV3MLP, DeepseekV3Config, src = try_import_from_transformers()
if DeepseekV3MoE is None:
    DeepseekV3MoE, DeepseekV3TopkRouter, DeepseekV3MLP, DeepseekV3Config, src = fallback_import_local_modeling()
print(f"[runner] using DeepSeek-V3 MoE from: {src}")

# 2) 超参（与 C++ main_moe.cpp 保持一致）
T, D = 32, 64
E, K = 16, 3
n_group, topk_group = 4, 2
H, H_se = 128, 128
norm_topk_prob = True
routed_scale = 1.0
n_shared_experts = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2025)

cfg = DeepseekV3Config(
    hidden_size=D,
    hidden_act="silu",
    # routed experts
    n_routed_experts=E,
    num_local_experts=E,
    num_experts_per_tok=K,
    n_group=n_group,
    topk_group=topk_group,
    norm_topk_prob=norm_topk_prob,
    routed_scaling_factor=routed_scale,
    # mlp sizes
    moe_intermediate_size=H,
    n_shared_experts=n_shared_experts,
)

# 3) 构造 MoE 并初始化（小幅均匀随机，确保非零）
moe = DeepseekV3MoE(cfg).to(device).eval()

def init_linear_weight_uniform(w: torch.Tensor, low=-0.05, high=0.05):
    with torch.no_grad():
        w.uniform_(low, high)

# gate
if hasattr(moe.gate, "weight"):
    init_linear_weight_uniform(moe.gate.weight)

# shared MLP（DeepseekV3MLP：gate_proj/up_proj/down_proj）
for name in ["gate_proj", "up_proj", "down_proj"]:
    lin = getattr(moe.shared_experts, name, None)
    if isinstance(lin, nn.Linear):
        init_linear_weight_uniform(lin.weight)

# routed experts
# 注意：Hugging Face 的实现通常把每个专家封装在 ModuleList/ModuleDict 里
# 逐个 expert 赋初值（不同 expert 也用同一分布）
for ex in moe.experts:
    for name in ["gate_proj", "up_proj", "down_proj"]:
        lin = getattr(ex, name, None)
        if isinstance(lin, nn.Linear):
            init_linear_weight_uniform(lin.weight)

# 4) 输入（与 C++ 一致）
x = torch.empty((1, T, D), device=device).uniform_(-1.0, 1.0)

# 5) 打印路由 top-k（t0）
with torch.no_grad():
    router_logits = moe.gate(x)  # 形状 (1, T, E) 或 (T, E) 视实现而定
    if router_logits.dim() == 3:
        router_logits = router_logits[0]  # -> (T, E)
    topk_idx, topk_alpha = moe.route_tokens_to_experts(router_logits)
    print("[PY] router_forward")
    print("  t0 idx:", " ".join(str(int(v)) for v in topk_idx[0].tolist()))
    print("  t0 alp:", " ".join(f"{float(v):.6f}" for v in topk_alpha[0].tolist()))

# 6) 全前向并打印 y[0,0:8]
with torch.no_grad():
    y = moe(x)  # (1, T, D)
    y0 = y[0, 0, :8].tolist()
    print("[PY] y[0,0:8]:", " ".join(f"{float(v):.6f}" for v in y0))
