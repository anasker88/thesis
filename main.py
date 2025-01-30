import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from run_and_eval import *
from setup import *
from tqdm import tqdm
import scipy.stats as stats


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


# with open(f"{output_dir}/scores.txt", "w") as f:
batch_num=10
bar = tqdm(range(batch_num))
total_scores = torch.zeros(batch_num,10,10, 2, 3,2, 4)  # batch,10-fold,k, language, strategy, mono/cross, score
for batch in bar:
    bar.set_description(f"Processing batch {batch}")
    # 10-fold cross validation
    rand_inds = torch.randperm(data_num)
    test_inds_list = torch.split(rand_inds, data_num // 10)
    scores = torch.zeros(10,10, 2, 3, 2, 4)  # 10-fold,k, language, strategy, mono/cross, score
    for i in range(10):
        test_inds = test_inds_list[i]
        train_inds = torch.cat(test_inds_list[:i] + test_inds_list[i + 1 :], dim=0)
        for log_k, k in enumerate(k_list):
            for j in range(2):
                # top-k by diff
                vals, inds = torch.topk(torch.cat(diff_sum_list[j], dim=0), k)
                feature_per_layer = diff_sum_list[j][0].shape[0]
                top_k = (inds // feature_per_layer, inds % feature_per_layer)
                score = k_sparse_probing(top_k, j, train_inds, test_inds,en_st,en_anti_st,jp_st,jp_anti_st)
                scores[log_k,i, j, 0] = score
                vals, inds = torch.topk(torch.cat(activation_score_list[j], dim=0), k)
                # top-k by activation score
                top_k = (inds // feature_per_layer, inds % feature_per_layer)
                score = k_sparse_probing(top_k, j, train_inds, test_inds,en_st,en_anti_st,jp_st,jp_anti_st)
                scores[log_k,i, j, 1] = score
                # random
                random_inds = torch.randperm(feature_per_layer * layer_num)
                rand_score=torch.zeros(10, 2, 4)
                for l in range(10):
                    inds = random_inds[l*k:(l+1)*k]
                    rand_top_k=(inds // feature_per_layer, inds % feature_per_layer)
                    rand_score[l] = k_sparse_probing(rand_top_k, j, train_inds, test_inds,en_st,en_anti_st,jp_st,jp_anti_st)
                scores[log_k,i, j, 2] = rand_score.mean(dim=0)
    total_scores[batch]=scores.clone()
    del scores

# t-test
# MD vs MS
# en,f1
en_diff_sum=total_scores[:, :, :, 0, 0, 0, 0]
en_activation_score=total_scores[:, :, :, 0, 1, 0, 0]
en_diff_sum=en_diff_sum.view(-1)
en_activation_score=en_activation_score.view(-1)
t, p = stats.ttest_rel(en_diff_sum, en_activation_score)
print(f"en_diff_sum vs en_activation_score: t={t}")
print(f"en_diff_sum vs en_activation_score: p={p}")
# jp,f1
jp_diff_sum=total_scores[:, :, :, 1, 0, 0, 0]
jp_activation_score=total_scores[:, :, :, 1, 1, 0, 0]
jp_diff_sum=jp_diff_sum.view(-1)
jp_activation_score=jp_activation_score.view(-1)
t, p = stats.ttest_rel(jp_diff_sum, jp_activation_score)
print(f"jp_diff_sum vs jp_activation_score: t={t}")
print(f"jp_diff_sum vs jp_activation_score: p={p}")

# average
scores = total_scores.mean(dim=1)
scores = scores.mean(dim=0)
    # for i in range(2):
    #     f.write(f"{lang[i]}\n")
    #     for j in range(3):
    #         f.write(f"strategy: {strategy[j]}\n")
    #         f.write(f"mono:\n")
    #         f.write(f"f1: {scores[i, j, 0, 0]}\n")
    #         f.write(f"precision: {scores[i, j, 0, 1]}\n")
    #         f.write(f"recall: {scores[i, j, 0, 2]}\n")
    #         f.write(f"accuracy: {scores[i, j, 0, 3]}\n")
    #         f.write(f"cross:\n")
    #         f.write(f"f1: {scores[i, j, 1, 0]}\n")
    #         f.write(f"precision: {scores[i, j, 1, 1]}\n")
    #         f.write(f"recall: {scores[i, j, 1, 2]}\n")
    #         f.write(f"accuracy: {scores[i, j, 1, 3]}\n")
    #         f.write("\n")
    #     f.write("\n")
# visulize scores

# mono
for i in range(2):
    for j in range(4):
        language=lang[i]
        met=metrics[j]
        plt.figure()
        plt.plot(k_list, scores[:, i,0,0,j], "-.",color="tab:green",label="MD")
        plt.plot(k_list, scores[:, i,1,0,j], "-",color="tab:red",label="MS(ours)")
        plt.plot(k_list, scores[:, i,2,0,j], ":",color="tab:gray",label="RANDOM(baseline)")
        plt.xlabel("k")
        plt.ylabel(met)
        plt.xscale("log",base=2)
        plt.ylim(0, 0.8)
        plt.legend()
        plt.title(f"sparse probing for {language}")
        plt.savefig(f"{output_dir}/{language}_mono_{met}.png")
        plt.close()

# cross
for i in range(2):
    for j in range(4):
        stra=strategy[i]
        met=metrics[j]
        plt.figure()
        plt.plot(k_list, scores[:, 0,i,0,j], "-v",color="tab:orange",label="En->En")
        plt.plot(k_list, scores[:, 1,i,0,j], "-s",color="tab:blue",label="Ja->Ja")
        plt.plot(k_list, scores[:, 0,i,1,j], "--v",color="tab:orange",label="En->Ja")
        plt.plot(k_list, scores[:, 1,i,1,j], "--s",color="tab:blue",label="Ja->En")
        plt.xlabel("k")
        plt.ylabel(met)
        plt.ylim(0, 0.8)
        plt.legend()
        plt.xscale("log",base=2)
        plt.title(f"crosslingual sparse probing ({stra})")
        plt.savefig(f"{output_dir}/cross_{stra}_{met}.png")
        plt.close()


# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,0],"-og", label="MD",)
# plt.plot(k_list, scores[:, 0,1,0,0], "-^m",label="MS(ours)")
# plt.plot(k_list, scores[:, 0,2,0,0],":k", label="RANDOM(baseline)")
# plt.xlabel("k")
# plt.ylabel("f1")
# plt.xscale("log",base=2)
# plt.ylim(0, 0.8)
# plt.legend()
# plt.title("sparse probing for en")
# plt.savefig(f"{output_dir}/en_mono_f1.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,1], "-og", label="MD")
# plt.plot(k_list, scores[:, 0,1,0,1], "-^m", label="MS")
# plt.plot(k_list, scores[:, 0,2,0,1], ":k", label="RANDOM")
# plt.xlabel("k")
# plt.ylabel("precision")
# plt.xscale("log",base=2)
# plt.ylim(0, 0.8)
# plt.legend()
# plt.title("sparse probing for en")
# plt.savefig(f"{output_dir}/en_mono_precision.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,2], "-og", label="MD")
# plt.plot(k_list, scores[:, 0,1,0,2], "-^m", label="MS")
# plt.plot(k_list, scores[:, 0,2,0,2], ":k", label="RANDOM")
# plt.xlabel("k")
# plt.ylabel("recall")
# plt.xscale("log",base=2)
# plt.ylim(0, 0.8)
# plt.legend()
# plt.title("sparse probing for en")
# plt.savefig(f"{output_dir}/en_mono_recall.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,3], "-og", label="MD")
# plt.plot(k_list, scores[:, 0,1,0,3], "-^m", label="MS")
# plt.plot(k_list, scores[:, 0,2,0,3], ":k", label="RANDOM")
# plt.xlabel("k")
# plt.ylabel("accuracy")
# plt.xscale("log",base=2)
# plt.ylim(0.4, 0.8)
# plt.legend()
# plt.title("sparse probing for en")
# plt.savefig(f"{output_dir}/en_mono_accuracy.png")
# plt.close()
# # jp_mono
# plt.figure()
# plt.plot(k_list, scores[:, 1,0,0,0], label="MD")
# plt.plot(k_list, scores[:, 1,1,0,0], label="MS")
# plt.plot(k_list, scores[:, 1,2,0,0], label="RANDOM")
# plt.xlabel("k")
# plt.ylabel("f1")
# plt.ylim(0, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("sparse probing for jp")
# plt.savefig(f"{output_dir}/jp_mono_f1.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 1,0,0,1], label="MD")
# plt.plot(k_list, scores[:, 1,1,0,1], label="MS")
# plt.plot(k_list, scores[:, 1,2,0,1], label="RANDOM")
# plt.xlabel("k")
# plt.ylabel("precision")
# plt.ylim(0.2, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("sparse probing for jp")
# plt.savefig(f"{output_dir}/jp_mono_precision.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 1,0,0,2], label="MD")
# plt.plot(k_list, scores[:, 1,1,0,2], label="MS")
# plt.plot(k_list, scores[:, 1,2,0,2], label="RANDOM")
# plt.xlabel("k")
# plt.ylabel("recall")
# plt.ylim(0.2, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("sparse probing for jp")
# plt.savefig(f"{output_dir}/jp_mono_recall.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 1,0,0,3], label="MD")
# plt.plot(k_list, scores[:, 1,1,0,3], label="MS")
# plt.plot(k_list, scores[:, 1,2,0,3], label="RANDOM")
# plt.xlabel("k")
# plt.ylabel("accuracy")
# plt.ylim(0.4, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("sparse probing for jp")
# plt.savefig(f"{output_dir}/jp_mono_accuracy.png")
# plt.close()

# # cross
# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,0], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,0,0,0], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,0,1,0], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,0,1,0], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("f1")
# plt.ylim(0, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MD)")
# plt.savefig(f"{output_dir}/cross_diff_sum_f1.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,3], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,0,0,3], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,0,1,3], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,0,1,3], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("accuracy")
# plt.ylim(0.4, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MD)")
# plt.savefig(f"{output_dir}/cross_diff_sum_accuracy.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,1], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,0,0,1], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,0,1,1], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,0,1,1], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("precision")
# plt.ylim(0, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MD)")
# plt.savefig(f"{output_dir}/cross_diff_sum_precision.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,0,0,2], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,0,0,2], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,0,1,2], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,0,1,2], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("recall")
# plt.ylim(0, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MD)")
# plt.savefig(f"{output_dir}/cross_diff_sum_recall.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,1,0,0], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,1,0,0], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,1,1,0], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,1,1,0], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("f1")
# plt.ylim(0, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MS)")
# plt.savefig(f"{output_dir}/cross_activation_score_f1.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,1,0,3], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,1,0,3], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,1,1,3], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,1,1,3], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("accuracy")
# plt.ylim(0.4, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MS)")
# plt.savefig(f"{output_dir}/cross_activation_score_accuracy.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,1,0,1], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,1,0,1], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,1,1,1], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,1,1,1], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("precision")
# plt.ylim(0, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MS)")
# plt.savefig(f"{output_dir}/cross_activation_score_precision.png")
# plt.close()

# plt.figure()
# plt.plot(k_list, scores[:, 0,1,0,2], label="En_Cls_En_Data")
# plt.plot(k_list, scores[:, 1,1,0,2], label="Jp_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 0,1,1,2], label="En_Cls_Jp_Data")
# plt.plot(k_list, scores[:, 1,1,1,2], label="Jp_Cls_En_Data")
# plt.xlabel("k")
# plt.ylabel("recall")
# plt.ylim(0, 0.8)
# plt.legend()
# plt.xscale("log",base=2)
# plt.title("crosslingual sparse probing (MS)")
# plt.savefig(f"{output_dir}/cross_activation_score_recall.png")
# plt.close()

# for i in range(10):
#     scores=total_scores[0,i,:,:,:,:,:]
#     #cross
#     plt.figure()
#     plt.plot(k_list, scores[:, 0,0,0,0], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,0,0,0], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,0,1,0], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,0,1,0], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("f1")
#     plt.ylim(0, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MD)")
#     plt.savefig(f"{output_dir}/case/f1/cross_diff_sum{i}.png")
#     plt.close()

#     plt.figure()
#     plt.plot(k_list, scores[:, 0,0,0,3], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,0,0,3], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,0,1,3], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,0,1,3], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("accuracy")
#     plt.ylim(0.4, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MD)")
#     plt.savefig(f"{output_dir}/case/accuracy/cross_diff_sum{i}.png")
#     plt.close()

#     plt.figure()
#     plt.plot(k_list, scores[:, 0,0,0,1], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,0,0,1], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,0,1,1], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,0,1,1], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("precision")
#     plt.ylim(0, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MD)")
#     plt.savefig(f"{output_dir}/case/precision/cross_diff_sum{i}.png")
#     plt.close()

#     plt.figure()
#     plt.plot(k_list, scores[:, 0,0,0,2], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,0,0,2], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,0,1,2], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,0,1,2], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("recall")
#     plt.ylim(0, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MD)")
#     plt.savefig(f"{output_dir}/case/recall/cross_diff_sum{i}.png")
#     plt.close()

#     plt.figure()
#     plt.plot(k_list, scores[:, 0,1,0,0], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,1,0,0], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,1,1,0], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,1,1,0], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("f1")
#     plt.ylim(0, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MS)")
#     plt.savefig(f"{output_dir}/case/f1/cross_activation_score{i}.png")
#     plt.close()

#     plt.figure()
#     plt.plot(k_list, scores[:, 0,1,0,3], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,1,0,3], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,1,1,3], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,1,1,3], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("accuracy")
#     plt.ylim(0.4, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MS)")
#     plt.savefig(f"{output_dir}/case/accuracy/cross_activation_score{i}.png")
#     plt.close()

#     plt.figure()
#     plt.plot(k_list, scores[:, 0,1,0,1], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,1,0,1], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,1,1,1], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,1,1,1], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("precision")
#     plt.ylim(0, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MS)")
#     plt.savefig(f"{output_dir}/case/precision/cross_activation_score{i}.png")
#     plt.close()

#     plt.figure()
#     plt.plot(k_list, scores[:, 0,1,0,2], label="En_Cls_En_Data")
#     plt.plot(k_list, scores[:, 1,1,0,2], label="Jp_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 0,1,1,2], label="En_Cls_Jp_Data")
#     plt.plot(k_list, scores[:, 1,1,1,2], label="Jp_Cls_En_Data")
#     plt.xlabel("k")
#     plt.ylabel("recall")
#     plt.ylim(0, 0.8)
#     plt.legend()
#     plt.xscale("log",base=2)
#     plt.title("crosslingual sparse probing (MS)")
#     plt.savefig(f"{output_dir}/case/recall/cross_activation_score{i}.png")
#     plt.close()
