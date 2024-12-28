import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_and_eval import *
from sae_lens import SAE, HookedSAETransformer
from setup import *
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


# caluculate scores for features
with open(f"{temp_dir}/{sae_release}/cfg_dict.json", "r") as f:
    data_num, layer_num = json.load(f)
layers = range(0, layer_num)
bar = tqdm(layers)
diff_sum_list = [[], []]
activation_score_list = [[], []]
for layer in bar:
    bar.set_description(f"Processing layer {layer}")
    for i in range(2):
        bar.set_description(f"Processing {lang[i]} for layer {layer}")
        st_act = torch.load(
            f"{temp_dir}/{sae_release}/{lang[i]}/st_act_{layer}.pt", weights_only=True
        )
        anti_st_act = torch.load(
            f"{temp_dir}/{sae_release}/{lang[i]}/anti_st_act_{layer}.pt",
            weights_only=True,
        )
        diff_sum, _, _ = evaluate_by_diff(st_act, anti_st_act, k)
        diff_sum_list[i].append(diff_sum)
        activation_score_list[i].append(activation_score(st_act, anti_st_act))
# 10-fold cross validation
rand_inds = torch.randperm(data_num)
test_inds_list = torch.split(rand_inds, data_num // 10)
with open(f"{output_dir}/scores.txt", "w") as f:
    scores = torch.zeros(10, 2, 3, 2, 3)  # 10-fold, language, strategy, score
    bar = tqdm(range(10))
    bar.set_description("10-fold cross validation")
    for i in bar:
        test_inds = test_inds_list[i]
        train_inds = torch.cat(test_inds_list[:i] + test_inds_list[i + 1 :], dim=0)
        for j in range(2):
            # top-k by diff
            vals, inds = torch.topk(torch.cat(diff_sum_list[j], dim=0), k)
            feature_per_layer = diff_sum_list[j][0].shape[0]
            top_k = (inds // feature_per_layer, inds % feature_per_layer)
            score = k_sparse_probing(top_k, j, train_inds, test_inds)
            scores[i, j, 0] = score
            vals, inds = torch.topk(
                torch.cat(activation_score_list[j], dim=0), k, largest=False
            )
            # top-k by activation score
            top_k = (inds // feature_per_layer, inds % feature_per_layer)
            score = k_sparse_probing(top_k, j, train_inds, test_inds)
            scores[i, j, 1] = score
            # random
            random_inds = torch.randperm(feature_per_layer)
            top_k = (
                random_inds[:k] // feature_per_layer,
                random_inds[:k] % feature_per_layer,
            )
            score = k_sparse_probing(top_k, j, train_inds, test_inds)
            scores[i, j, 2] = score
    # average
    scores = scores.mean(dim=0)
    for i in range(2):
        f.write(f"{lang[i]}\n")
        for j in range(3):
            f.write(f"strategy: {strategy[j]}\n")
            f.write(f"mono:\n")
            f.write(f"f1: {scores[i, j, 0, 0]}\n")
            f.write(f"precision: {scores[i, j, 0, 1]}\n")
            f.write(f"recall: {scores[i, j, 0, 2]}\n")
            f.write(f"cross:\n")
            f.write(f"f1: {scores[i, j, 1, 0]}\n")
            f.write(f"precision: {scores[i, j, 1, 1]}\n")
            f.write(f"recall: {scores[i, j, 1, 2]}\n")
