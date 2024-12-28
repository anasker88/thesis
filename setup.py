import os

import torch

llm_names = ["gpt2-small", "meta-llama/Llama-3.1-8B", "gemma-2-9b"]
sae_names = ["gpt2-small-res-jb", "llama_scope_lxm_32x", "google/gemma-scope-9b-pt-mlp"]
sae_ids = [
    "blocks.{}.hook_resid_pre",
    "l{}m_32x",
]

target_llm = 0
# 0:gpt-2-small, 1:llama-3.1-8B, 2:gemma-2-9b
k = 100
llm_name = llm_names[target_llm]
sae_release = sae_names[target_llm]
sae_id = sae_ids[target_llm]
output_dir = f"out/{sae_release}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/jp", exist_ok=True)
os.makedirs(f"{output_dir}/en", exist_ok=True)
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)
lang = ["en", "jp"]
strategy = ["top-k", "activation_score", "random"]
# initialize
torch.set_grad_enabled(False)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
print(f"Device: {device}")
