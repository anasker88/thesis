import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_and_eval import *
from setup import *
from tqdm import tqdm

# initialize
torch.set_grad_enabled(False)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
print(f"Device: {device}")
device1=device
device2=device
if device == "cuda":
    if torch.cuda.device_count() > 1:
        device1 = "cuda:0"
        device2 = "cuda:1"
        print(f"Device1: {device1}")
        print(f"Device2: {device2}")

# caluculate scores for features
with open(f"{temp_dir}/{sae_release}/cfg_dict.json", "r") as f:
    data_num, layer_num = json.load(f)
layers = range(0, layer_num)
bar = tqdm(layers)
diff_sum_list = [[], []]
activation_score_list = [[], []]
en_st=None
en_anti_st=None
jp_st=None
jp_anti_st=None
for layer in bar:
    bar.set_description(f"Processing layer {layer}")
    for i in range(2):
        bar.set_description(f"Processing {lang[i]} for layer {layer}")
        save_device = device1 if i == 0 else device2
        st = torch.load(
            f"{temp_dir}/{sae_release}/{lang[i]}/st_act_{layer}.pt", weights_only=True,map_location=save_device
        )
        anti_st = torch.load(
            f"{temp_dir}/{sae_release}/{lang[i]}/anti_st_act_{layer}.pt", weights_only=True,
            map_location=save_device
        )
        diff_sum, _, _ = evaluate_by_diff(st, anti_st, k)
        diff_sum_list[i].append(diff_sum)
        activation_score_list[i].append(activation_score(st, anti_st))
        if i==0:
            if layer==0:
                shape=st.shape
                en_st=torch.zeros(len(layers),shape[0],shape[1])
                en_anti_st=torch.zeros(len(layers),shape[0],shape[1])
            en_st[layer]=st
            en_anti_st[layer]=anti_st
        else:
            if layer==0:
                shape=st.shape
                jp_st=torch.zeros(len(layers),shape[0],shape[1])
                jp_anti_st=torch.zeros(len(layers),shape[0],shape[1])
            jp_st[layer]=st
            jp_anti_st[layer]=anti_st
        del st, anti_st, diff_sum

def spearmanr(idx, en_score, jp_score):
    en_device=en_score.device
    jp_device=jp_score.device
    en_score=en_score[idx.to(en_device)]
    jp_score=jp_score[idx.to(jp_device)]
    _, en_idx = torch.sort(en_score)
    _, jp_idx = torch.sort(jp_score)
    diff=en_idx-jp_idx.to(en_device)
    n=len(diff)
    return 1-6*(diff**2).sum()/(n*(n**2-1))

en_diff_sum=torch.cat(diff_sum_list[0],dim=0)
jp_diff_sum=torch.cat(diff_sum_list[1],dim=0)
en_activation_score=torch.cat(activation_score_list[0],dim=0)
jp_activation_score=torch.cat(activation_score_list[1],dim=0)
# calculate correlation
# k=1000
# # en, diff_sum
# _, en_top_k = torch.topk(en_diff_sum, k)
# correlation = spearmanr(en_top_k, en_diff_sum, jp_diff_sum)
# print(f"Correlation of diff_sum: {correlation:.4f}")
# # en, activation_score
# _, en_top_k = torch.topk(en_activation_score, k)
# correlation = spearmanr(en_top_k, en_activation_score, jp_activation_score)
# print(f"Correlation of activation_score: {correlation:.4f}")
# # jp, diff_sum
# _, jp_top_k = torch.topk(jp_diff_sum, k)
# correlation = spearmanr(jp_top_k, en_diff_sum, jp_diff_sum)
# print(f"Correlation of diff_sum: {correlation:.4f}")
# # jp, activation_score
# _, jp_top_k = torch.topk(jp_activation_score, k)
# correlation = spearmanr(jp_top_k, en_activation_score, jp_activation_score)
# print(f"Correlation of activation_score: {correlation:.4f}")

for k in [4,16,64,256]:
    print(f"k={k}")
    # diff_sum
    _, en_top_k = torch.topk(en_diff_sum, k)
    _, jp_top_k = torch.topk(jp_diff_sum, k)
    jp_top_k=jp_top_k.to(en_top_k.device)
    common=torch.isin(en_top_k,jp_top_k).sum().item()
    print(f"diff_sum common proportion: {common/k:.4f}")
    # activation_score
    _, en_top_k = torch.topk(en_activation_score, k)
    _, jp_top_k = torch.topk(jp_activation_score, k)
    jp_top_k=jp_top_k.to(en_top_k.device)
    common=torch.isin(en_top_k,jp_top_k).sum().item()
    print(f"activation_score common proportion: {common/k:.4f}")
