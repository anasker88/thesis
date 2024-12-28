import gc
from typing import Tuple

import torch
from sae_lens import SAE, HookedSAETransformer
from setup import *
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import SVC


def find_target_index(
    sentence: str, target_phrase: str, model: HookedSAETransformer
) -> Tuple[int, int]:
    """
    Args:
        sentence: Sentence
        target_phrase: Target phrase
        model: Model
        Returns:
        target_index: Target index
    """
    # get target_phrase index in the sentence
    start_index = sentence.find(target_phrase) + 1
    end_index = start_index + len(target_phrase) - 1
    # print(f"Start index: {start_index}")
    # print(f"End index: {end_index}")
    # print(f"Target phrase: {target_phrase}")
    if start_index == -1:
        print(f"Target phrase '{target_phrase}' not found in the sentence.")
        print(f"Sentence: {sentence}")
        return (0, 0, 0)
    # get token indices
    tokens_indices = model.tokenizer(sentence, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    # print(f"Tokens: {tokens_indices}")
    target_index = (0, 0, 0)
    for i, token in enumerate(tokens_indices):
        if start_index < token[0]:
            break
        target_index = (i, 0, 0)
        if start_index == token[0]:
            break
    for i, token in enumerate(tokens_indices):
        if end_index < token[1]:
            break
        target_index = (target_index[0], i, 0)
    tokens = model.to_str_tokens(sentence)
    # prepend bos if not present
    if len(tokens) != len(tokens_indices):
        if len(tokens) == len(tokens_indices) + 1:
            target_index = (target_index[0] + 1, target_index[1] + 1)
        else:
            print(f"Token length mismatch: {len(tokens)} and {len(tokens_indices)}")
            return (0, 0, 0)
    target_index = (target_index[0], target_index[1], len(tokens))
    # print(f"Target index: {target_index}")
    # print(f"Tokens: {tokens}")
    # constructed_target = "".join(tokens[target_index[0] : target_index[1] + 1])
    # print(f"Constructed target: {constructed_target}")
    return target_index


def get_activations(
    model: HookedSAETransformer,
    sae: SAE,
    prompts: list[str],
    target: list[str],
    width: int = 20,
) -> torch.Tensor:
    """
    Args:
        model: Model
        sae: SAE
        prompts: Prompts
        target_index: Target index
    Returns:
        activations: Max activations of features at target index(shape: (batch_size, feature_size))
    """
    activations = torch.zeros(len(prompts), sae.cfg.d_sae, device=sae.device)
    for i in range((len(prompts) - 1) // width + 1):
        start_index = width * i
        end_index = min(width * (i + 1), len(prompts))
        _, cache = model.run_with_cache_with_saes(
            prompts[start_index:end_index],
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae.cfg.hook_name + ".hook_sae_acts_post"],
        )
        cache = cache[sae.cfg.hook_name + ".hook_sae_acts_post"]
        # get the max activations of features among target indices
        for i in range(start_index, end_index):
            # print(model.to_str_tokens(prompts[i]))
            target_index = find_target_index(prompts[i], target[i], model)
            for j in range(target_index[0], target_index[1] + 1):
                # activations[i] = torch.max(activations[i], cache[i - start_index, j, :])
                activations[i] = cache[i - start_index, j, :]
        del cache
        # if start_index > 10:
        #     exit()
        torch.cuda.empty_cache()
    return activations


# filter out features
def filter_out_features(
    st_act: torch.Tensor, anti_st_act: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """
    Args:
        st_act: Activations of stereotypical prompt(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt(shape: (batch_size, feature_size))
    Returns:
        important_features: Features that are important for classification
    """
    # remove if not activated in 50% of the samples
    diff = st_act - anti_st_act
    act_num = (diff != 0).sum(dim=0)
    important_features = torch.where(act_num > st_act.shape[0] * threshold)[0]
    return important_features


# evaluate by diff
def evaluate_by_diff(
    st_act: torch.Tensor, anti_st_act: torch.Tensor, k: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        st_act: Activations of stereotypical prompt(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt(shape: (batch_size, feature_size))
        k: Top k
    Returns:
        diff_sum: Difference of activations
        vals: Top k features
        inds: Top k feature indices
    """
    diff = st_act - anti_st_act
    # get sum
    diff_sum = diff.sum(dim=0)
    # top k features
    vals, inds = torch.topk(diff_sum, k)
    return diff_sum, vals, inds


# activation score
def activation_score(st_act: torch.Tensor, anti_st_act: torch.Tensor) -> torch.Tensor:
    """
    +1 if diff is positive and -1 if diff is negative.The activation score is sum/n_samples
    Args:
        st_act: Activations of stereotypical prompt(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt(shape: (batch_size, feature_size))
    Returns:
        activation_score: Activation score
    """
    diff = st_act - anti_st_act
    score = torch.where(
        diff > 0,
        torch.tensor(1.0),
        torch.tensor(0.0),
    )
    score += torch.where(
        diff < 0,
        torch.tensor(-1.0),
        torch.tensor(0.0),
    )
    activation_score = score.sum(dim=0)
    return activation_score


def k_sparse_probing(
    top_k: Tuple[torch.Tensor, torch.Tensor],
    language: int,
    train_inds: torch.Tensor,
    test_inds: torch.Tensor,
) -> torch.Tensor:
    lang = ["en", "jp"]
    st_act = []
    anti_st_act = []
    st_act_other_lang = []
    anti_st_act_other_lang = []
    layers, features = top_k
    # load activations
    for layer, feature in zip(layers, features):
        st = torch.load(
            f"temp/{sae_release}/{lang[language]}/st_act_{layer}.pt", weights_only=True
        )
        st = st[:, feature].clone()
        st_act.append(st)
        anti = torch.load(
            f"temp/{sae_release}/{lang[language]}/anti_st_act_{layer}.pt",
            weights_only=True,
        )
        anti = anti[:, feature].clone()
        anti_st_act.append(anti)
        st = torch.load(
            f"temp/{sae_release}/{lang[1 - language]}/st_act_{layer}.pt",
            weights_only=True,
        )
        st = st[:, feature].clone()
        st_act_other_lang.append(st)
        anti = torch.load(
            f"temp/{sae_release}/{lang[1 - language]}/anti_st_act_{layer}.pt",
            weights_only=True,
        )
        anti = anti[:, feature].clone()
        anti_st_act_other_lang.append(anti)
    st_act = torch.stack(st_act, dim=1)
    anti_st_act = torch.stack(anti_st_act, dim=1)
    st_act_other_lang = torch.stack(st_act_other_lang, dim=1)
    anti_st_act_other_lang = torch.stack(anti_st_act_other_lang, dim=1)
    train_act = torch.cat([st_act[train_inds, :], anti_st_act[train_inds, :]], dim=0)
    test_act = torch.cat([st_act[test_inds, :], anti_st_act[test_inds, :]], dim=0)
    test_act_other_lang = torch.cat(
        [st_act_other_lang[test_inds, :], anti_st_act_other_lang[test_inds, :]], dim=0
    )
    train_labels = torch.cat(
        [
            torch.ones(st_act[train_inds].shape[0]),
            torch.zeros(anti_st_act[train_inds].shape[0]),
        ]
    )
    test_labels = torch.cat(
        [
            torch.ones(st_act[test_inds].shape[0]),
            torch.zeros(anti_st_act[test_inds].shape[0]),
        ]
    )
    ret = torch.zeros(2, 3)
    # svm
    clf = SVC(kernel="linear", class_weight="balanced")
    clf.fit(train_act.cpu().numpy(), train_labels.cpu().numpy())
    test_pred = clf.predict(test_act.cpu().numpy())
    test_labels = test_labels.cpu().numpy()
    f_1 = f1_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred, zero_division=0)
    ret[0] = torch.tensor([f_1, recall, precision])
    test_pred = clf.predict(test_act_other_lang.cpu().numpy())
    f_1 = f1_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred, zero_division=0)
    ret[1] = torch.tensor([f_1, recall, precision])
    return ret
