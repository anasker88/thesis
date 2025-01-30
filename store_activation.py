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

# initialize
torch.set_grad_enabled(False)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
print(f"Device: {device}")
print(f"available GPUs: {torch.cuda.device_count()}")
# load LLM
model = HookedSAETransformer.from_pretrained(llm_name, device=device,n_devices=2)

# load dataset
en_data = pd.read_csv("data/en_data.csv")
jp_data = pd.read_csv("data/jp_data.csv")
data_num = len(en_data)
layer_num = model.cfg.n_layers
print(f"Layer num: {layer_num}")
layers = range(0, layer_num)
bar = tqdm(layers)
# bar = layers
os.makedirs(f"{temp_dir}/{sae_release}", exist_ok=True)
os.makedirs(f"{temp_dir}/{sae_release}/en", exist_ok=True)
os.makedirs(f"{temp_dir}/{sae_release}/jp", exist_ok=True)
for layer in bar:
    bar.set_description(f"Loading SAE for layer {layer}")
    # load SAE
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,  # <- Release name
        sae_id=sae_id.format(layer),  # <- SAE id (not always a hook point!)
        device=device,
    )
    # set device
    hook = sae.cfg.hook_name
    _, test_cache = model.run_with_cache("test", stop_at_layer=layer + 1)
    sae_device = test_cache[hook].device
    if sae_device != device:
        sae.to(sae_device)
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
        st_act = get_activations(model, sae, st_prompt, target)
        anti_st_act = get_activations(model, sae, anti_st_prompt, target)
        # save activations
        torch.save(st_act, f"{temp_dir}/{sae_release}/{language}/st_act_{layer}.pt")
        torch.save(
            anti_st_act, f"{temp_dir}/{sae_release}/{language}/anti_st_act_{layer}.pt"
        )
        del st_act, anti_st_act
    del sae, test_cache, cfg_dict, sparsity
    gc.collect()
    torch.cuda.empty_cache()
with open(f"{temp_dir}/{sae_release}/cfg_dict.json", "w") as f:
    json.dump((data_num, layer_num), f)
