from __future__ import annotations

# import math
import torch
import numpy as np
from egg.core.util import find_lengths
# from scipy.optimize import minimize_scalar
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance  # , ratio
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from torch.utils.data import Dataset
from itertools import combinations
import pyitlib.discrete_random_variable as it
# from pyitlib.discrete_random_variable import entropy, entropy_joint
from nltk.lm import MLE#, NgramModel, LidstoneProbDist
from nltk.probability import LidstoneProbDist
from nltk.lm.models import Lidstone
from egg.zoo.language_bottleneck.intervention import entropy, mutual_info

from typing import Optional, Iterable, Tuple

from time import time


def timer(func):
    return func
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


class CustomDataset(Dataset):
    def __init__(self, messages, receiver_inputs):
        """
        Args:
            messages (torch.Tensor): Tensor of shape (N, 5), where N is the number of samples.
            receiver_inputs (torch.Tensor): Tensor of shape (N, 5, 8).
        """
        assert len(messages) == len(receiver_inputs), \
            "Messages and receiver_inputs must have the same number of samples."
        self.messages = messages
        self.receiver_inputs = receiver_inputs

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx], self.receiver_inputs[idx]


# Entropy, Mutual information
def binary_entropy(p: float):
    if p == 0. or p == 1.:
        return 0.
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def message_entropy(messages, return_length_probs=False, order=2):
    """
    work in progress...
    """
    if messages.dim() == 2:
        raise NotImplementedError

    min_real = torch.finfo(messages.dtype).min

    symbol_entropies = torch.zeros_like(messages[0, :, 0])
    eos_probs = torch.zeros_like(messages[0, :, 0])

    indices = [
        torch.empty(
            [messages.size(-1) - 1 for _ in range(order)],
            dtype=torch.long,
            device=messages.device)
        for _ in range(order)]
    for i, tensor in enumerate(indices):
        for j in range(messages.size(-1) - 1):
            tensor.index_fill_(i, torch.tensor(j), j + 1)
    indices = torch.stack([t.flatten() for t in indices], dim=-1)

    for symbol_i in range(messages.size(1)):
        symbol_probs = messages[:, symbol_i, :]
        if symbol_i == 0:
            eos_probs[0] = symbol_probs[:, 0].mean()
            probs = symbol_probs[:, 1:].mean(0)
            probs = probs / probs.sum(0, keepdim=True)
            log2_prob = torch.clamp(torch.log2(probs), min=min_real)
            symbol_entropies[symbol_i] = (-log2_prob * probs).sum()
            continue

        preceding_probs = messages[:, max(0, symbol_i - order):symbol_i]
        prefix_indices = torch.unique(
            indices[:, :min(order, symbol_i)],
            dim=0
        ).t()#.view(-1, preceding_probs.size(-2))
        # if preceding_probs.size(1) == 1:
        #    prefix_indices = prefix_indices.squeeze()
        print(symbol_i, prefix_indices.shape, preceding_probs.shape)
        # prefix_probs = torch.gather(preceding_probs, 1, prefix_indices)
        prefix_probs = preceding_probs[..., prefix_indices, torch.arange(preceding_probs.size(-1))]
        print(prefix_probs)
        print(prefix_probs.shape)
        # prefix_probs = torch.gather(
    #    if symbol_i < order:
    #        prefix_indices = indices[indices[0] < symbol_i]
    #    else:
    #        prefix_indices = indices.clone()
    #        prefix_indices[0] += indices[0].max() - (order - 1)

    #    print(prefix_indices)
    #    for idx in prefix_indices:
    #        prefix_probs = messages[torch.arange(len(messages)), idx]
    #        print(prefix_probs)
        
        # print("step", symbol_i)
        # prefix_probs = messages[:, symbol_i - 1, :] if symbol_i != 0 \
        #     else torch.ones_like(messages[:, symbol_i, :])

    #    print("prefix_probs", prefix_probs.shape)
    #    print("symbol_probs", symbol_probs.shape)

    #    probs = (prefix_probs * symbol_probs).sum(0)
    #    probs = probs / probs.sum(0, keepdim=True)

    #    print("probs shape", probs.shape)

    #    eos_probs[symbol_i] = probs[0]
    #    probs = probs[1:] / probs[1:].sum()  # exclude EOS prob

    #    log2_prob = torch.clamp(torch.log2(probs), min=min_real)
    #    symbol_entropies[symbol_i] = (-log2_prob * probs).sum()

        # prefix_probs[:, step_i  = prefix_probs * symbol_probs

        # shape = [probs.size(0)] + [1 for _ in range(prefix_probs.dim())]
        # prefix_probs = prefix_probs.unsqueeze(0).expand(probs.size(0), *prefix_probs.shape)
        # print(prefix_probs.shape, shape)
        # prefix_probs = prefix_probs * probs.view(shape)

    # prefix_probs = torch.ones_like(messages[:, 0, :])
    # prefix_probs = torch.ones_like(messages[0, 0, 0])
    # prefix_probs = torch.cat([
    #     torch.ones_like(messages[0, :1, 1:]),
    #     messages[0, 1:, 1:],
    # ], dim=1)
    #indices = torch.empty([messages.size(-1) - 1] * order, device=messages.device).long()

    print("smb entrs:", symbol_entropies)
    print("eos probs:", eos_probs)

    entropy = 0.
    for i in range(len(symbol_entropies) - 1):  # last symbol is always EOS
        symbol_entropy = symbol_entropies[i].item()
        # print(symbol_entropy, 'smb entropy', i)
        eos_prob = eos_probs[i].item()

        entropy += (
            binary_entropy(eos_prob)
            # + eos_prob * 0
            + (1 - eos_prob) * symbol_entropy
        )

    not_eosed_before = 1.
    length_probs = torch.zeros_like(eos_probs)
    for i in range(messages.size(1)):
        length_probs[i] = not_eosed_before * eos_probs[i]
        not_eosed_before *= 1 - eos_probs[i]
    # print('LEN PROBS', length_probs)
    # print('EOS PROBS', eos_probs)

    # length_log2_prob = torch.clamp(torch.log2(length_probs), min=min_real)
    # entropy += (-length_probs * length_log2_prob).sum(-1).item()

    if return_length_probs:
        return entropy, length_probs
    else:
        return entropy


# Redundancy
@timer
def compute_max_rep(messages: torch.Tensor) -> torch.Tensor:
    """
    Computes the number of occurrences of the most frequent symbol in each
    message (0 for messages that consist of EOS symbols only).
    """

    all_symbols = torch.unique(torch.flatten(messages), dim=0)
    non_eos_symbols = all_symbols[all_symbols != 0]

    output = torch.zeros(messages.size(0))
    for smb in non_eos_symbols:
        smb_tensor = smb.expand(messages.size(1))
        smb_tensor = smb_tensor.t().expand(*messages.size())

        match = messages.eq(smb_tensor).to(torch.int)
        for i in range(0, messages.size(1) - 1):
            # search for a repeating subsequence of length i + 1
            matching_msg = match.max(dim=1).values.to(torch.bool)
            length = torch.where(matching_msg, i + 1, 0)
            if torch.all(length == 0):  # if no message has any matches, continue
                break
            output = torch.where(length > output, length, output)
            match = torch.mul(match[:, :-1], match[:, 1:])

    return output


def sequence_entropy_old(entropy, categorical=None):
    print(entropy.shape, categorical.shape)
    assert len(entropy) == len(categorical)

    categories, indices = torch.unique(categorical, return_inverse=True)
    if len(categorical) == categorical.numel():
        categorical = categorical.reshape(-1)

    H_msg_y = entropy(categorical)
    for uni in torch.unique(indices):
        mask = indices[indices == uni]
        matches = entropy[mask]
        H_msg_y += matches.mean()
        print((indices == uni).int().sum(), len(matches), '/', len(categorical), matches.mean())

    return H_msg_y


def compute_mi(entropy_message, attributes, estimator='GOOD-TURING'):
    # TODO ...
    # attributes = categorical
    # entropy_msg = messages.sum(-1).mean().item()
    print("entropy mgs MI", entropy_message.shape)
    if attributes.size(1) == 1:
        # attributes = messages
        entropy_attr = tensor_entropy(attributes)
        if entropy_message is not None:
            entropy_msg_attr = sequence_entropy_old(entropy_message, attributes)
            mi_msg_attr = entropy_message + entropy_attr - entropy_msg_attr
            vi_msg_attr = 2 * entropy_msg_attr - entropy_message - entropy_attr
            vi_norm_msg_attr = 1. - mi_msg_attr / entropy_msg_attr
        else:
            entropy_msg_attr, mi_msg_attr, vi_msg_attr, vi_norm_msg_attr = \
                None, None, None, None

        output = {
            'entropy_msg': entropy_message,
            'entropy_attr': entropy_attr,
            'mi_msg_attr': mi_msg_attr,
            'vi_msg_attr': vi_msg_attr,
            'vi_norm_msg_attr': vi_norm_msg_attr,
            'is_msg_attr': 1 - vi_norm_msg_attr,
        }

    else:  # return values per attribute dimension instead
        # _, attributes = torch.unique(categorical, dim=0, return_inverse=True)
        # entropy_attr = sequence_entropy(attributes, estimator=estimator)
        entropy_attr_dim = [
            tensor_entropy(attributes[:, i], estimator=estimator)
            for i in range(attributes.size(-1))]
        entropy_msg_attr_dim = [sequence_entropy_old(messages, attributes[:,i]) 
                                for i in range(attributes.size(1))]
        mi_msg_attr_dim = [
            entropy_message + H_y - H_xy
            for H_y, H_xy in zip(entropy_attr_dim, entropy_msg_attr_dim)
        ]
        vi_msg_attr_dim = [
            2 * entropy_msg_attr - entropy_message - entropy_attr
            for entropy_attr, entropy_msg_attr
            in zip(entropy_attr_dim, entropy_msg_attr_dim)]
        vi_norm_msg_attr_dim = [
            1. - mi_msg_attr / entropy_msg_attr
            for mi_msg_attr, entropy_msg_attr
            in zip(mi_msg_attr_dim, entropy_msg_attr_dim)]
        is_msg_attr_dim = [
            mi_msg_attr / entropy_msg_attr
            for mi_msg_attr, entropy_msg_attr
            in zip(mi_msg_attr_dim, entropy_msg_attr_dim)]

        output = {
            'entropy_msg': entropy_message,
            # 'entropy_attr': entropy_attr,
            'entropy_attr_dim': entropy_attr_dim,
            'mi_msg_attr_dim': mi_msg_attr_dim,
            'vi_msg_attr_dim': vi_msg_attr_dim,
            'vi_norm_msg_attr_dim': vi_norm_msg_attr_dim,
            'is_msg_attr_dim': is_msg_attr_dim,
        }

    return {f'{k}_v2': v for k, v in output.items()}


def truncate_messages(messages, receiver_input, labels, mode):
    new_messages = []
    new_r_input = []
    new_labels = []
    for i, message in enumerate(messages):
        if mode == 'rf':
            truncated = remove_n_items(message, 1)
        else:
            truncated = remove_n_dims(message, 1)
        new_messages.extend(truncated)
        new_r_input.extend([receiver_input[i]] * len(truncated))
        new_labels.extend([labels[i]] * len(truncated))

    return new_messages, new_r_input, new_labels


def remove_n_items(tensor, n=1):
    """
    Removes all possible combinations of `n` items from the tensor,
    symbol 0 is never removed.
    Needed for "redundancy" measure if using rf.

    Args:
        tensor (torch.Tensor): The input tensor.
        n (int): The number of items to remove.

    Returns:
        list[torch.Tensor]: A list of tensors with `n` items removed.
    """
    # Get the indices of elements that can be removed (exclude 0)
    removable_indices = [idx for idx in range(len(tensor)) if tensor[idx] != 0]

    # Generate all combinations of `n` indices to remove
    combos = list(combinations(removable_indices, n))

    # Create new tensors with the combinations removed
    result = []
    for indices in combos:
        mask = torch.ones(len(tensor), dtype=torch.bool)
        mask[list(indices)] = True
        new = tensor[mask]
        new = new.to(torch.long)
        result.append(new)

    return result


def remove_n_dims(tensor, n=1):
    print(tensor.shape)
    # Get the number of rows (N)
    num_rows = tensor.shape[0]

    # Ensure there are enough rows to remove `n` and keep the last row
    if n >= num_rows:
        raise ValueError("Cannot remove more rows than available (excluding the last row).")
    if num_rows <= 1:
        raise ValueError("The input tensor must have more than one row.")

    # Get indices of rows that can be removed (exclude the last row)
    removable_indices = list(range(num_rows - 1))  # Exclude last row

    # Generate all combinations of `n` rows to remove
    combos = list(combinations(removable_indices, n))

    # Create new tensors with the selected rows removed
    result = []
    for combo in combos:
        mask = torch.ones(num_rows, dtype=torch.bool)
        mask[list(combo)] = False  # Set rows in the combo to False (remove them)
        new = tensor[mask].to(torch.float)
        result.append(new)

    return result


@timer
def compute_accuracy2(dump, receiver: torch.nn.Module, opts):

    messages, receiver_inputs, labels = truncate_messages(
        dump.messages, dump.receiver_inputs, dump.labels, opts.mode)

    dataset = CustomDataset(messages, receiver_inputs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        drop_last=True)

    predictions = []
    for batched_messages, batched_inputs in dataloader:
        with torch.no_grad():
            outputs = receiver(batched_messages, batched_inputs)

        if opts.mode == 'rf':
            outputs = outputs[0]
            predictions.append(outputs.detach().reshape(-1, 1))
        else:
            # TODO depenging on whether we take argmax on the train dataset,
            # the step/length might need to be adjusted
            lengths = find_lengths(batched_messages.argmax(-1))
            for i in range(batched_messages.size(0)):
                outputs_i = outputs[i, lengths[i] - 1].argmax(-1)
                predictions.append(outputs_i.detach().reshape(-1, 1))

    predictions = torch.cat(predictions, dim=0)
    labels = torch.stack(labels)[:len(predictions)]

    return (predictions == labels).float().mean().item()


# Compositionality
@timer
def compute_top_sim(attributes: torch.Tensor, messages: torch.Tensor) -> float:
    """
    Computes topographic rho.
    """
    # TODO switch to implementation from core.language_analysis?

    attributes = attributes.long()

    # if attribute tensor contains categorical variables,
    # apply one-hot encoding before computing cosine similarity
    one_hots = []
    for i in range(attributes.size(1)):
        if len(torch.unique(attributes[:, i])) <= 2:
            one_hots.append(attributes[:, i].reshape(-1, 1))
        else:
            one_hot = torch.nn.functional.one_hot(attributes[:, i])
            one_hots.append(one_hot)
    attributes = torch.cat(one_hots, dim=-1).numpy()

    # pairwise cosine similarity between object vectors
    cos_sims = cosine_similarity(attributes)

    messages = [
        [s.int().item() for s in msg if s > 0] + [0]
        for msg in messages]

    # pairwise Levenshtein distance between messages
    lev_dists = np.ones((len(messages), len(messages)), dtype='int')
    for i, msg_i in enumerate(messages):
        for j, msg_j in enumerate(messages):
            if i > j:
                continue
            elif i == j:
                lev_dists[i][j] = 1
            else:
                dist = distance(msg_i, msg_j) * -1
                # dist = ratio(msg_i, msg_j)  # normalized
                lev_dists[i][j] = dist
                lev_dists[j][i] = dist

    rho = spearmanr(cos_sims, lev_dists, axis=None).statistic
    return rho


@timer
def compute_posdis(
        sender_inputs: torch.Tensor,
        messages: torch.Tensor,
        receiver_vocab_size: Optional[int] = None) -> float:
    """
    Computes PosDis.
    """

    gaps = torch.zeros(messages.size(1))
    non_constant_positions = 0.0
    for j in range(messages.size(1)):
        symbol_mi = []

        # if receiver_vocab_size is not None:
        #     alphabet_x = torch.arange(receiver_vocab_size) \
        #         if j < messages.size(1) else torch.zeros(1, 1)
        # else:  # bosdis
        #     alphabet_x = torch.unique(messages)

        for i in range(sender_inputs.size(1)):
            x, y = messages[:, j], sender_inputs[:, i]
            y = sender_inputs[:, i]
            H_j = entropy(y)
            info = mutual_info(x, y)
            symbol_mi.append(info)

        symbol_mi.sort(reverse=True)

        if H_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / H_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def histogram(messages: torch.Tensor, vocab_size: int) -> torch.Tensor:

    # Handle messages with added noise
    if vocab_size in messages:
        vocab_size += 1

    # Create a histogram with size [batch_size, vocab_size] initialized with zeros
    histogram = torch.zeros(messages.size(0), vocab_size)

    if messages.dim() > 2:
        messages = messages.view(messages.size(0), -1)

    # Count occurrences of each value in strings and store them in histogram
    histogram.scatter_add_(1, messages.long(), torch.ones_like(messages, dtype=torch.float))

    return histogram


@timer
def compute_bosdis(
        sender_inputs: torch.Tensor,
        messages: torch.Tensor,
        vocab_size: int):
    """
    Computes BosDis.
    """
    histograms = histogram(messages, vocab_size)
    return compute_posdis(sender_inputs, histograms[:, 1:])
