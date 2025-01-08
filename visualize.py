import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_and_eval import *
from sae_lens import SAE, HookedSAETransformer
from matplotlib.colors import Normalize
from matplotlib import font_manager,rc
from setup import *
from tqdm import tqdm


home_dir = os.environ["HOME"]
font_path = f"{home_dir}/fonts/Noto_Sans_JP/NotoSansJP-VariableFont_wght.ttf"
font_manager.fontManager.addfont(font_path)
def visualize_tokens_with_scores(
    st_tokens,
    st_scores,
    anti_st_tokens,
    anti_st_scores,
    ln=0,
    cmap="Reds",
    title="Token Visualization",
):
    """

    Args:
        st_tokens (list[list[str]]): List of tokens for stereotypical sentences.
        st_scores (list[list[float]]): List of scores for stereotypical sentences.
        anti_st_tokens (list[list[str]]): List of tokens for anti-stereotypical sentences.
        anti_st_scores (list[list[float]]): List of scores for anti-stereotypical sentences.
        cmap (str): Colormap name.
        title (str): Title of the plot.
    """
    #set font
    font="DejaVu Sans" if ln==0 else "Noto Sans JP"
    # Normalize
    norm = Normalize(vmin=0, vmax=40)
    num_sentences = len(st_tokens)
    # Plot
    fig, ax = plt.subplots(figsize=(12, 0.5 + num_sentences / 2))
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    # set background color
    fig.patch.set_facecolor("lightgray")
    y = 0
    y_step = 1 / (num_sentences + 1)
    for tokens, scores in zip(st_tokens, st_scores):
        y += y_step
        x = 0
        colors = plt.colormaps.get_cmap(cmap)(norm(scores))
        for token, color in zip(tokens, colors):
            # skip bos
            if token == "<|begin_of_text|>":
                continue
            text = ax.text(
                x,
                y,
                token,
                fontsize=8,
                font=font,
                ha="left",
                va="center",
                bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3"),
            )
            # get width of text
            text_bb = text.get_window_extent().transformed(ax.transData.inverted())
            print(text_bb.width)
            x += text_bb.width + 0.01

    y = 0
    for tokens, scores in zip(anti_st_tokens, anti_st_scores):
        y += y_step
        x = 0.5
        colors = plt.colormaps.get_cmap(cmap)(norm(scores))
        for token, color in zip(tokens, colors):
            # skip bos
            if token == "<|begin_of_text|>":
                continue
            text = ax.text(
                x,
                y,
                token,
                fontsize=8,
                font=font,
                ha="left",
                va="center",
                bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.3"),
            )
            # get width of text
            text_bb = text.get_window_extent().transformed(ax.transData.inverted())
            print(text_bb.width)
            x += text_bb.width + 0.01

    # カラーバー追加
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03)
    cbar.set_label("Score", fontsize=12)

    plt.savefig(f"{output_dir}/visualize/{lang[ln]}_{title}.png")


# initialize
torch.set_grad_enabled(False)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
print(f"Device: {device}")
device1=device #for en activations
device2=device #for jp activations
device3=device #for model
if device == "cuda":
    if torch.cuda.device_count() > 2:
        device1 = "cuda:0"
        device2 = "cuda:1"
        device3 = "cuda:2"
    else:
        print(f"Only {torch.cuda.device_count()} GPUs are available.")
print(f"Device1: {device1}")
print(f"Device2: {device2}")
print(f"Device3: {device3}")
model=HookedSAETransformer.from_pretrained(llm_name,device=device3)
en_data = pd.read_csv("data/en_data.csv")
jp_data = pd.read_csv("data/jp_data.csv")
# caluculate scores for features
with open(f"{temp_dir}/{sae_release}/cfg_dict.json", "r") as f:
    data_num, layer_num = json.load(f)
layers = range(0, layer_num)
bar = tqdm(layers)
diff_sum_list = [[], []]
activation_score_list = [[], []]
en_st=[]
en_anti_st=[]
jp_st=[]
jp_anti_st=[]
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
            en_st.append(st)
            en_anti_st.append(anti_st)
        else:
            jp_st.append(st)
            jp_anti_st.append(anti_st)
        del st, anti_st, diff_sum

en_st=torch.stack(en_st,dim=0)
en_anti_st=torch.stack(en_anti_st,dim=0)
jp_st=torch.stack(jp_st,dim=0)
jp_anti_st=torch.stack(jp_anti_st,dim=0)
k=10
for ln in range(2):
    data=None
    if ln==0:
        data=en_data
    else:
        data=jp_data
    print(f"Language {lang[ln]}")
    # diff_sum
    _, inds = torch.topk(torch.cat(diff_sum_list[ln], dim=0), k)
    feature_per_layer = diff_sum_list[ln][0].shape[0]
    top_k_diff_sum = (inds // feature_per_layer, inds % feature_per_layer)
    for i in range(k):
        layer = top_k_diff_sum[0][i]
        feature = top_k_diff_sum[1][i]
        print(f"Layer {layer}, Feature {feature}")
        sae, cfg_dict, _=SAE.from_pretrained(release=sae_release,sae_id=sae_id.format(layer),device=device3)
        st = None
        anti_st = None
        if ln==0:
            st = en_st[layer]
            anti_st = en_anti_st[layer]
        else:
            st = jp_st[layer]
            anti_st = jp_anti_st[layer]
        st = st[:, feature].clone()
        anti_st = anti_st[:, feature].clone()
        diff = st - anti_st
        top_diff,top_diff_pr=torch.topk(diff,10)
        # rand_diff_pr=torch.randint(0,diff.shape[0],(10,))
        # top_diff_pr=rand_diff_pr
        # print(f"Top diff: {top_diff.item()}")
        # print(f"st score: {st[top_diff_pr.item()].item()}")
        # print(f"anti-st score: {anti_st[top_diff_pr.item()].item()}")
        st_pr=data["st"].tolist()
        anti_st_pr=data["anti-st"].tolist()
        tgt=data["tgt"].tolist()
        st_pr=[st_pr[top_diff_pr[i]] for i in range(top_diff_pr.shape[0])]
        anti_st_pr=[anti_st_pr[top_diff_pr[i]] for i in range(top_diff_pr.shape[0])]
        # print(f"Prompt: {st_pr[top_diff_pr.item()]}")
        # print(f"Anti-prompt: {anti_st_pr[top_diff_pr.item()]}")
        # print(f"Target: {tgt[top_diff_pr.item()]}")
        _,cache=model.run_with_cache_with_saes(st_pr,saes=[sae],stop_at_layer=sae.cfg.hook_layer + 1,names_filter=[sae.cfg.hook_name + ".hook_sae_acts_post"],)
        cache=cache[sae.cfg.hook_name + ".hook_sae_acts_post"].clone()
        st_act=cache[:,:,feature].to(torch.float32).cpu().numpy()
        del cache
        st_tokens=model.to_str_tokens(st_pr)
        _,cache=model.run_with_cache_with_saes(anti_st_pr,saes=[sae],stop_at_layer=sae.cfg.hook_layer + 1,names_filter=[sae.cfg.hook_name + ".hook_sae_acts_post"],)
        cache=cache[sae.cfg.hook_name + ".hook_sae_acts_post"].clone()
        anti_st_act=cache[:,:,feature].to(torch.float32).cpu().numpy()
        anti_st_tokens=model.to_str_tokens(anti_st_pr)
        visualize_tokens_with_scores(st_tokens,st_act,anti_st_tokens,anti_st_act,ln=ln,title=f"Layer {layer} Feature {feature}")
        del st, anti_st, diff,cache,sae
        torch.cuda.empty_cache()
    # activation_score
    _, inds = torch.topk(torch.cat(activation_score_list[ln], dim=0), k)
    feature_per_layer = activation_score_list[ln][0].shape[0]
    top_k_activation_score = (inds // feature_per_layer, inds % feature_per_layer)
    for i in range(k):
        print(f"Layer {top_k_activation_score[0][i]}, Feature {top_k_activation_score[1][i]}")
del en_st, en_anti_st, jp_st, jp_anti_st, diff_sum_list, activation_score_list, top_k_diff_sum, top_k_activation_score
