from typing import Tuple

import torch
from sae_lens import SAE, HookedSAETransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def find_token_combinations(tokens, target_word):
    """Given a list of tokens and a target word,
    find combinations of consecutive tokens that form the target word.

    Args:
        tokens (list): List of tokens.
        target_word (str): Target word to find in the list of tokens.

    Returns:
        list of tuples: List of tuples where each tuple contains the start and end index of the tokens forming the target word.
    """
    results = []
    for start_idx in range(len(tokens)):
        # skip empty tokens
        if not tokens[start_idx]:
            continue
        combined_word = ""
        for end_idx in range(start_idx, len(tokens)):
            combined_word += tokens[end_idx]
            if end_idx == start_idx:
                combined_word = combined_word.lstrip(" ")
            # Check if the combined word matches the target word
            if combined_word == target_word:
                results.append((start_idx, end_idx))
            # Break if the combined word is longer than the target word
            if len(combined_word) > len(target_word):
                break
    return results


def normalize_and_find_target(tokens, target_word):
    """Normalize tokens and find the target word in the list of tokens.

    Args:
        tokens (list): List of tokens.
        target_word (str): Target word to find in the list of tokens.

    Returns:
        list of tuples: List of tuples where each tuple contains the start and end index of the tokens forming the target word.
    """
    # Normalize tokens
    # tokens = [token.replace(" ", "").lower() for token in tokens]
    # Find target word in the list of tokens
    result = find_token_combinations(tokens, target_word)
    if len(result) > 0:
        return result
    # If target word is not found, try with 's' appended
    result = find_token_combinations(tokens, target_word + "s")
    if len(result) > 0:
        return result
    # If target word is not found, try with 't' appended
    result = find_token_combinations(tokens, target_word + "t")
    if len(result) > 0:
        return result
    print(f"Target word '{target_word}' not found in the tokens.")
    print(f"Tokens: {tokens}")
    return [(0, 0)]


def get_activations(
    model: HookedSAETransformer,
    sae: SAE,
    prompts: list[str],
    target: list[str],
    width: int = 5,
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
    for i in range((len(prompts) + 4) // width):
        start_index = width * i
        end_index = min(width * (i + 1), len(prompts))
        _, cache = model.run_with_cache_with_saes(
            prompts[start_index:end_index],
            saes=[sae],
            # stop_at_layer=sae.cfg.hook_layer + 1,
            # names_filter=[sae.cfg.hook_name + ".hook_sae_acts_post"],
        )
        cache = cache[sae.cfg.hook_name + ".hook_sae_acts_post"]
        # get the max activations of features among target indices
        for i in range(start_index, end_index):
            # print(model.to_str_tokens(prompts[i]))
            target_index = normalize_and_find_target(
                model.to_str_tokens(prompts[i]), target[i]
            )
            for j in range(target_index[0][0], target_index[0][1] + 1):
                activations[i] = torch.max(activations[i], cache[i - start_index, j, :])
        del cache
        # if start_index > 10:
        #     exit()
        torch.cuda.empty_cache()
    return activations


def remove_same_pairs(
    st_act: torch.Tensor, anti_st_act: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        st_act: Activations of stereotypical prompt(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt(shape: (batch_size, feature_size))
    Returns:
        st_act: Activations of stereotypical prompt without same pairs(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt without same pairs(shape: (batch_size, feature_size))
    """
    # remove same pairs
    new_st_act = st_act[~torch.all(st_act == anti_st_act, dim=1)]
    new_anti_st_act = anti_st_act[~torch.all(st_act == anti_st_act, dim=1)]
    return new_st_act, new_anti_st_act


# evaluate by diff
def evaluate_by_diff(
    st_act: torch.Tensor, anti_st_act: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        st_act: Activations of stereotypical prompt(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt(shape: (batch_size, feature_size))
    Returns:
        diff_sum: Difference of activations
        vals: Top 5 features
        inds: Top 5 feature indices
        avg_diff: Average of the absolute difference
    """
    diff = st_act - anti_st_act
    # get sum
    diff_sum = diff.sum(dim=0)
    # top 5 features
    vals, inds = torch.topk(diff_sum, 5)
    # average of the absolute difference
    avg_diff = diff_sum.abs().mean()
    return diff_sum, vals, inds, avg_diff


# 2-class classification by random forest
def run_random_forest(
    st_act: torch.Tensor, anti_st_act: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """
    Args:
        st_act: Activations of stereotypical prompt(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt(shape: (batch_size, feature_size))
    Returns:
        accuracy: Accuracy
        feature_importance: Feature importance
    """
    # concatenate st_act and anti_st_act
    activations = torch.cat([st_act, anti_st_act], dim=0)
    # set labels
    labels = torch.cat([torch.ones(st_act.shape[0]), torch.zeros(anti_st_act.shape[0])])
    # random forest
    clf = RandomForestClassifier()
    clf.fit(activations.cpu().numpy(), labels.cpu().numpy())
    accuracy = clf.score(activations.cpu().numpy(), labels.cpu().numpy())
    feature_importance = torch.tensor(clf.feature_importances_)

    return accuracy, feature_importance


# 2-class classification by SVM
def run_svm(
    st_act: torch.Tensor, anti_st_act: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """
    Args:
        st_act: Activations of stereotypical prompt(shape: (batch_size, feature_size))
        anti_st_act: Activations of anti-stereotypical prompt(shape: (batch_size, feature_size))
    Returns:
        accuracy: Accuracy
        normal_vector: Normal vector
    """
    # concatenate st_act and anti_st_act
    activations = torch.cat([st_act, anti_st_act], dim=0)
    # set labels
    labels = torch.cat([torch.ones(st_act.shape[0]), torch.zeros(anti_st_act.shape[0])])
    # SVM
    clf = SVC(kernel="linear")
    clf.fit(activations.cpu().numpy(), labels.cpu().numpy())
    accuracy = clf.score(activations.cpu().numpy(), labels.cpu().numpy())
    normal_vector = torch.tensor(clf.coef_)

    return accuracy, normal_vector
