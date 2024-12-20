import os

import pandas as pd
import torch
from run_and_eval import *
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm

llm_names = ["gpt2-small", "meta-llama/Llama-3.1-8B", "gemma-2-9b"]
sae_names = ["gpt2-small-res-jb", "llama_scope_lxm_8x", "google/gemma-scope-9b-pt-mlp"]
sae_ids = [
    "blocks.{}.hook_resid_pre",
    "l{}m_8x",
]

target_llm = 0
# 0:gpt-2-small, 1:llama-3.1-8B, 2:gemma-2-9b
llm_name = llm_names[target_llm]
sae_release = sae_names[target_llm]
sae_id = sae_ids[target_llm]
output_dir = f"out/{sae_release}"
os.makedirs(output_dir, exist_ok=True)

# initialize
torch.set_grad_enabled(False)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
print(f"Device: {device}")

# load LLM
model = HookedSAETransformer.from_pretrained(llm_name, device=device)

# load dataset
en_data = pd.read_csv("data/en_data.csv")
jp_data = pd.read_csv("data/jp_data.csv")


layer_num = 32
bar = tqdm(range(9, 10))
with open(f"{output_dir}/output.txt", "w") as f:
    for layer in bar:
        bar.set_description(f"Loading SAE for layer {layer}")
        # load SAE
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=sae_release,  # <- Release name
            sae_id=sae_id.format(layer),  # <- SAE id (not always a hook point!)
            device=device,
        )
        f.write(f"Layer {layer}\n")
        for i in range(2):  # 0:en, 1:jp
            if i == 0:
                data = en_data
                bar.set_description(f"Processing en data for layer {layer}")
            else:
                data = jp_data
                bar.set_description(f"Processing jp data for layer {layer}")
            st_prompt = data["st"].tolist()
            anti_st_prompt = data["anti-st"].tolist()
            target = data["tgt"]
            # get activations
            st_act = get_activations(model, sae, st_prompt, target)
            anti_st_act = get_activations(model, sae, anti_st_prompt, target)
            # remove same pairs
            st_act, anti_st_act = remove_same_pairs(st_act, anti_st_act)
            print(f"sample size: {st_act.shape[0]}")
            # evaluate by diff
            diff_sum, vals, inds, avg_diff = evaluate_by_diff(st_act, anti_st_act)
            # 2-class classification by random forest
            accuracy_rf, feature_importance_rf = run_random_forest(st_act, anti_st_act)
            # 2-class classification by SVM
            accuracy_svm, normal_vector_svm = run_svm(st_act, anti_st_act)
            # save results
            f.write(f"Language: {'en' if i == 0 else 'jp'}\n")
            f.write(f"Diff sum: {diff_sum}\n")
            f.write(f"Top 5 features: {vals}\n")
            f.write(f"Top 5 feature indices: {inds}\n")
            f.write(f"Average of the absolute difference: {avg_diff}\n")
            f.write(f"Accuracy by random forest: {accuracy_rf}\n")
            vals, inds = torch.topk(feature_importance_rf, 10)
            f.write(f"Top 5 features by random forest: {vals}\n")
            f.write(f"Top 5 feature indices by random forest: {inds}\n")
            f.write(f"Accuracy by SVM: {accuracy_svm}\n")
            vals, inds = torch.topk(normal_vector_svm, 10)
            f.write(f"Top 5 features by SVM: {vals}\n")
            f.write(f"Top 5 feature indices by SVM: {inds}\n")
