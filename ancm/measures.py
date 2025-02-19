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
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs

from typing import Optional, Iterable, Tuple


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


def message_entropy(probs, order=4, split_size=1000):
    """
    work in progress...
    """
    # min_real = torch.finfo(probs.dtype).min

    size = probs.size()
    logits = probs_to_logits(
        probs.view(size[0] * size[1], size[2])).view(size)

    symbol_entropies = torch.zeros_like(probs[0, :, 0])
    eos_probs = torch.ones_like(probs[0, :, 0])

    for symbol_i in range(probs.size(1) - 1):
        symbol_logits = logits[:, symbol_i, :]
        if symbol_i == 0:
            log_prob = torch.logsumexp(symbol_logits, 0)
            log_prob -= log_prob.logsumexp(0)  # normalize
            log2_prob = log_prob / np.log(2)  # switch to base 2
            # c_log2_prob = torch.clamp(c_logits / np.log(2), min=min_real)
            prob = logits_to_probs(log_prob)
            eos_probs[0] = prob[0]
            symbol_entropies[0] = (-log2_prob * prob).sum()
            continue

        if order is not None:
            # prev_probs = non_eos_probs[:, max(0, symbol_i - order):symbol_i]
            # prev_probs = probs[:, max(0, symbol_i - order):symbol_i]
            # prev_logits = logits[:, max(0, symbol_i - order):symbol_i]
            prev_logits = logits[:, max(0, symbol_i - order):symbol_i, 1:]
        else:
            # prev_probs = non_eos_probs[:, :symbol_i]
            # prev_probs = probs[:, :symbol_i]
            # prev_logits = logits[:, :symbol_i]
            prev_logits = logits[:, :symbol_i, 1:]

        # oprint('prev pobs shape', prev_probs.shape)
        # print("prev_logits shape", prev_logits.shape)
        # print("symbol logits shape", symbol_logits.shape)
        prefix_indices = torch.cartesian_prod(
            *(torch.arange(prev_logits.size(-1))
              for i in range(prev_logits.size(1)))
        ).view(-1, prev_logits.size(1))
        prefix_logits, conditional_entropy = [], []
        eos_logits = []

        for p_indices in torch.split(prefix_indices, split_size):
            # indices of all possible non-EOS prefixes
            idx = (
                p_indices,
                torch.arange(prev_logits.size(1))
                .unsqueeze(0)
                .expand(p_indices.size(0), -1),
            )

            # prefix log-probabilities
            p_logits = prev_logits.transpose(0, -1)[idx].sum(1).t()

            # conditional entropy for each prefix
            c_logits = torch.logsumexp(
                symbol_logits.unsqueeze(1) + p_logits.unsqueeze(-1), 0)
            c_logits -= c_logits.logsumexp(-1, keepdim=True)  # normalize
            c_log2_prob = c_logits / np.log(2)  # switch to base 2
            c_prob = logits_to_probs(c_logits)
            c_entropy = (-c_prob * c_log2_prob).sum(-1)

            # print('c_logits', c_logits.shape)
            prefix_logits.append(p_logits.logsumexp(0))
            eos_logits.append(c_logits[:, 0])
            conditional_entropy.append(c_entropy)

        prefix_probs = logits_to_probs(torch.cat(prefix_logits))
        conditional_entropy = torch.cat(conditional_entropy)
        conditional_eos_prob = torch.exp(torch.cat(eos_logits))
        # print(
        #     "cond eos pr",
        #     conditional_eos_prob.shape,
        #     conditional_eos_prob.sum() / conditional_eos_prob.size(0))
        eos_probs[symbol_i] = torch.tensordot(
            prefix_probs,
            conditional_eos_prob,
            dims=([0], [0]))
        symbol_entropies[symbol_i] = torch.tensordot(
            prefix_probs,
            conditional_entropy,
            dims=([0], [0]))
        # print('conditional_entropy', conditional_entropy.shape, conditional_entropy[:5])

    # print(symbol_entropies)
    # print('eos_probs', eos_probs)
    # alt_eos_probs = logits_to_probs(logits.logsumexp(0))[:, 0]
    # print('alt_eos_probs', alt_eos_probs)
    entropy = symbol_entropies.sum()

    not_eosed_before = 1.
    length_probs = torch.zeros_like(eos_probs)
    for i in range(probs.size(1)):
        length_probs[i] = not_eosed_before * eos_probs[i]
        not_eosed_before *= 1 - eos_probs[i]

    return entropy, length_probs



# Redundancy
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


def joint_entropy(probs, y, split_size=512):
    """
    Given symbol probabilities representing realizations of a message M
    (M = M1, ..., Mn) and corresponding realizations of a non-compound RV Y,
    computes joint entropy H(M1, ..., Mn, Y) of message symbols and Y using
    the formula H(M1, ..., Mn, Y) = H(Y) + H(M1, ..., Mn | Y).
    """
    assert len(y) == y.numel()
    _, indices = torch.unique(y, dim=0, return_inverse=True)

    H_msg_y = entropy(indices)  # H(Y)
    for cat in range(indices.max() + 1):
        cat_mask = indices == cat
        prob_cat = (indices == cat).float().mean()
        entropy_cat, _ = message_entropy(probs[cat_mask], split_size)
        H_msg_y += prob_cat * entropy_cat  # P(Y = y) * H(M | Y = y)
        # print((indices == cat).int().sum(), len(probs_cat), '/', len(y), H_msg_y)

    return H_msg_y


def compute_mi(probs, attributes, entropy_message=None):
    if entropy_message is None:
        entropy_message, _ = message_entropy(probs)

    if attributes.size(1) == 1:
        entropy_attr = entropy(attributes)
        entropy_msg_attr = joint_entropy(probs, attributes)
        mi_msg_attr = entropy_message + entropy_attr - entropy_msg_attr
        vi_msg_attr = 2 * entropy_msg_attr - entropy_message - entropy_attr
        vi_norm_msg_attr = 1. - mi_msg_attr / entropy_msg_attr

        output = {
            'entropy_msg': entropy_message,
            'entropy_attr': entropy_attr,
            'MI_msg_attr': mi_msg_attr,
            'VI_msg_attr': vi_msg_attr,
            'VInorm_msg_attr': vi_norm_msg_attr,
            'IS_msg_attr': 1 - vi_norm_msg_attr,
        }

    else:  # return values per attribute dimension instead
        _, categorical = torch.unique(attributes, dim=0, return_inverse=True)
        entropy_attr = entropy(categorical)
        # entropy_attr = sequence_entropy(attributes, estimator=estimator)
        entropy_attr_dim = [
            entropy(attributes[:, i])
            for i in range(attributes.size(-1))]
        entropy_msg_attr_dim = [
            joint_entropy(probs, attributes[:, i])
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
            'entropy_attr': entropy_attr,
            'entropy_attr_dim': entropy_attr_dim,
            'MI_msg_attr_dim': mi_msg_attr_dim,
            'VI_msg_attr_dim': vi_msg_attr_dim,
            'VInorm_msg_attr_dim': vi_norm_msg_attr_dim,
            'IS_msg_attr_dim': is_msg_attr_dim,
        }

    return output


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


def compute_accuracy2(messages, receiver_inputs, labels, receiver: torch.nn.Module, opts):
    messages, receiver_inputs, labels = truncate_messages(
        messages, receiver_inputs, labels, opts.mode)

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
            lengths = find_lengths(batched_messages.argmax(-1))
            idx = (torch.arange(len(batched_messages)), lengths - 1)
            predictions.extend(outputs[idx].argmax(-1))

    predictions = torch.stack(predictions, dim=0)
    labels = torch.stack(labels)[:len(predictions)]

    return (predictions == labels).float().mean().item()


# Compositionality
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


def compute_bosdis(
        sender_inputs: torch.Tensor,
        messages: torch.Tensor,
        vocab_size: int):
    """
    Computes BosDis.
    """
    histograms = histogram(messages, vocab_size)
    return compute_posdis(sender_inputs, histograms[:, 1:])
