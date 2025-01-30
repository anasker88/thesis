import os

import torch

llm_names = ["gpt2-small", "meta-llama/Llama-3.1-8B", "google/gemma-2-9b"]
sae_names = ["gpt2-small-res-jb", "llama_scope_lxm_32x", "gemma-scope-9b-pt-mlp-canonical"]
sae_ids = [
    "blocks.{}.hook_resid_pre",
    "l{}m_32x",
    "layer_{}/width_131k/canonical",
]

target_llm = 2
# 0:gpt-2-small, 1:llama-3.1-8B, 2:gemma-2-9b
k = 100
k_list=[2**i for i in range(10)]
llm_name = llm_names[target_llm]
sae_release = sae_names[target_llm]
sae_id = sae_ids[target_llm]
output_dir = f"out/{sae_release}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/Ja", exist_ok=True)
os.makedirs(f"{output_dir}/En", exist_ok=True)
os.makedirs(f"{output_dir}/visualize", exist_ok=True)
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)
lang = ["En", "Ja"]
strategy = ["MD", "MS", "RANDOM"]
metrics=["f1","precision","recall","accuracy"]
