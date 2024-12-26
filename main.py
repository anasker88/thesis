import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_and_eval import *
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm


def show_activation(st_act, anti_st_act, layer, feature, language):
    lang = ["en", "jp"]
    n_bins = 50
    st_act = st_act.cpu().numpy()
    anti_st_act = anti_st_act.cpu().numpy()
    max_val = max(st_act.max(), anti_st_act.max())
    min_val = min(st_act.min(), anti_st_act.min())
    y_max = 100
    bins = np.linspace(min_val, max_val, n_bins)
    plt.hist(st_act, bins=bins, alpha=0.5, label="stereotypical")
    plt.hist(anti_st_act, bins=bins, alpha=0.5, label="anti-stereotypical")
    # diff=st_act-anti_st_act
    plt.ylim(0, 100)
    plt.xlabel("Feature activation")
    plt.ylabel("Frequency")
    # plt.hist(diff, bins=n_bins, alpha=0.5, label="diff")
    plt.legend()
    plt.title(f"Layer {layer}, Feature {feature}, {lang[language]}")
    plt.savefig(f"{output_dir}/{lang[language]}/layer{layer}_feature{feature}.png")
    plt.close()


llm_names = ["gpt2-small", "meta-llama/Llama-3.1-8B", "gemma-2-9b"]
sae_names = ["gpt2-small-res-jb", "llama_scope_lxm_32x", "google/gemma-scope-9b-pt-mlp"]
sae_ids = [
    "blocks.{}.hook_resid_pre",
    "l{}m_32x",
]

target_llm = 1
# 0:gpt-2-small, 1:llama-3.1-8B, 2:gemma-2-9b
k = 10
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
data_num = len(en_data)
test_num= int(data_num * 0.1)
test_inds = np.random.choice(data_num, test_num, replace=False)
train_inds = np.array([i for i in range(data_num) if i not in test_inds])
test_inds = torch.tensor(test_inds)
train_inds = torch.tensor(train_inds)
layer_num = model.cfg.n_layers
print(f"Layer num: {layer_num}")
layers = range(0, layer_num)
bar = tqdm(layers)
diff_sum_list = [[], []]
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
            else:
                data = jp_data
            language = lang[i]
            bar.set_description(f"Processing {language} for layer {layer}")
            st_prompt = data["st"].tolist()
            anti_st_prompt = data["anti-st"].tolist()
            target = data["tgt"]
            # get activations
            st_act=get_activations(model, sae, st_prompt, target)
            anti_st_act=get_activations(model, sae, anti_st_prompt, target)
            # # remove same pairs
            # st_act, anti_st_act = remove_same_pairs(st_act, anti_st_act)
            f.write(f"Language: {language}\n")
            # f.write(f"samples: {len(st_act)}\n")
            # important_features = filter_out_features(st_act, anti_st_act,0)
            # f.write(f"Important features: {len(important_features)}\n")
            # small_st_act = st_act[:, important_features]
            # small_anti_st_act = anti_st_act[:, important_features]
            # k = min(k, len(important_features))
            # evaluate by diff
            diff_sum, vals, inds = evaluate_by_diff(st_act[train_inds,:], anti_st_act[train_inds,:],k)
            diff_sum_list[i].append(diff_sum)
            # # 2-class classification by random forest
            # accuracy_rf, feature_importance_rf = run_random_forest(
            #     small_st_act, small_anti_st_act
            # )
            # # 2-class classification by SVM
            # accuracy_svm, normal_vector_svm = run_svm(small_st_act, small_anti_st_act)
            # save results
            # f.write(f"Diff sum: {diff_sum}\n")
            avg_diff = (st_act[train_inds,:] - anti_st_act[train_inds,:]).abs().sum(dim=0).mean()
            # scores = activation_score(st_act, anti_st_act)
            f.write(f"Top k features: {vals}\n")
            f.write(f"Top k feature indices: {inds}\n")
            f.write(f"Average of the absolute difference: {avg_diff}\n")
            # save activations
            torch.save(st_act, f"{temp_dir}/{language}_st_act_{layer}.pt")
            torch.save(anti_st_act, f"{temp_dir}/{language}_anti_st_act_{layer}.pt")
            # f.write(f"Accuracy by random forest: {accuracy_rf}\n")
            # vals, inds = torch.topk(feature_importance_rf, k)
            # f.write(f"Top k features by random forest: {vals}\n")
            # f.write(
            #     f"Top k feature indices by random forest: {important_features[inds]}\n"
            # )
            # f.write(f"Accuracy by SVM: {accuracy_svm}\n")
            # vals, inds = torch.topk(normal_vector_svm.abs(), k)
            # f.write(f"Top k features activations by SVM: {diff_sum[inds]}\n")
            # f.write(f"Top k feature indices by SVM: {important_features[inds]}\n")
            # vals, inds = torch.topk(scores, k)
            # f.write(f"Top k features activations by score: {vals}\n")
            # f.write(f"Top k feature indices by score: {inds}\n")
            # f.write("\n")
        del sae, st_act, anti_st_act
        torch.cuda.empty_cache()
    for i in range(2):
        f.write(f"Language: {lang[i]}\n")
        vals, inds = torch.topk(torch.cat(diff_sum_list[i], dim=0), k)
        feature_per_layer = diff_sum_list[i][0].shape[0]
        top_k = (inds // feature_per_layer, inds % feature_per_layer)
        f.write(f"Top k features activations by diff: {vals}\n")
        for j in range(k):
            f.write(
                f"Top k feature indices by diff: Layer {layers[top_k[0][j]]}, Feature {top_k[1][j]}\n"
            )
            # show_activation(
            #     st_act_list[i][top_k[0]][:, top_k[1]],
            #     anti_st_act_list[i][top_k[0]][:, top_k[1]],
            #     layers[top_k[0]],
            #     inds_list[top_k[0]][top_k[1]],
            #     i,
            # ))
        f_1,recall,precision = k_sparse_probing(top_k,i,train_inds,test_inds)
        f.write(f"F1 score: {f_1}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}\n")
