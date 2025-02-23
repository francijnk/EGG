from __future__ import annotations

# import math
import torch
import numpy as np
from egg.core.util import find_lengths
# from scipy.optimize import minimize_scalar
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance  # , ratio
from scipy.stats import spearmanr  # pearsonr
from torch.utils.data import Dataset
from itertools import combinations
# import pyitlib.discrete_random_variable as it
# from pyitlib.discrete_random_variable import entropy, entropy_joint
from egg.zoo.language_bottleneck import intervention
from torch.distributions.utils import logits_to_probs, probs_to_logits

from typing import Optional


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
# def binary_entropy(p: float):
#     if p == 0. or p == 1.:
#         return 0.
#     return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# @torch.compile
def get_prefix_indices(prev_symbols, split_size):
    n_prev = prev_symbols.size(1)
    prefix_indices = torch.cartesian_prod(
        *(torch.arange(prev_symbols.size(-1)) for i in range(n_prev))
    ).view(-1, n_prev)

    for indices in torch.split(prefix_indices, split_size):
        yield indices, torch.arange(n_prev).view(1, -1).expand(indices.size())


# @torch.compile
def aggregate_prefix_logits(prev_symbols, idx):
    return prev_symbols.transpose(0, -1)[idx].sum(1).t()


def min_message_entropy(receiver_inputs, labels, attributes=None):
    size = receiver_inputs.size()

    if attributes is None:
        # categorize VISA features and get target objects for each sample
        r_inputs = receiver_inputs.view(size[0] * size[1], size[2])
        _, r_inputs_cat = torch.unique(r_inputs, dim=0, return_inverse=True)
        targets = r_inputs_cat[labels]

        # sort objects in each sample, so that their order doesn't matter
        # and categorize object samples
        samples, _ = r_inputs_cat.view(size[:-1]).sort()

    else:  # instead of receiver_inputs, use the attribute provided
        # if more than one attribute per object is provided, categorize
        if attributes.dim() == 3:  # (n_messages, n_distractors, n_attributes)
            _, attributes = torch.unique(
                attributes.view(size[0] * size[1], -1),
                return_inverse=True,
                dim=0,
            )  # 40k x 5 x 2 -> 40k x 5
            attributes = attributes.view(size[:2])

        targets = attributes[:, 0]  # target attributes always come first
        samples, _ = attributes.sort()

    _, samples = torch.unique(samples, dim=0, return_inverse=True)
    unique_samples = torch.unique(samples)

    # H(label | receiver_inputs)
    entropy_min = 0
    for sample in unique_samples:
        targets_i = targets[samples == sample]
        p_i = len(targets_i) / len(targets)
        entropy_i = intervention.entropy(targets_i)
        entropy_min += p_i * entropy_i

    return entropy_min, len(unique_samples)


def message_entropy(probs, order=4, split_size=1000):
    min_real = torch.finfo(probs.dtype).min

    size = probs.size()
    logits = probs_to_logits(
        probs.view(size[0] * size[1], size[2])
    ).view(size)

    symbol_entropies = torch.zeros_like(probs[0, :, 0])
    eos_probs = torch.ones_like(probs[0, :, 0])
    p_not_eosed = 1

    # iterate over all symbols except for the appended EOS
    for step in range(size[1] - 1):
        symbol_logits = logits[:, step]

        if step == 0:
            log_prob = symbol_logits.logsumexp(0)
            log_prob -= log_prob.logsumexp(0)  # normalize
            log2_prob = torch.clamp(log_prob / np.log(2), min=min_real)
            prob = probs[:, step].sum(0) / probs.size(0)

            symbol_entropies[0] = torch.dot(-prob, log2_prob)
            eos_probs[0] = prob[0]
            p_not_eosed *= 1 - prob[0]
            continue

        first_symbol = 0 if order is None else max(0, step - order)
        prev_logits = logits[:, first_symbol:step, 1:]

        # iterate over all possible non-EOS prefixes of the symbol
        prefix_logits, cond_entropy, cond_p_eos = [], [], []
        for idx in get_prefix_indices(prev_logits, split_size):
            # log-probs of previous symbols being equal to a given prefix
            p_logits = aggregate_prefix_logits(prev_logits, idx)

            # conditional entropy for each prefix
            c_logits = torch.logsumexp(
                symbol_logits.unsqueeze(1) + p_logits.unsqueeze(-1), dim=0)
            c_logits -= c_logits.logsumexp(-1, keepdim=True)  # normalize
            c_log2_prob = torch.clamp(c_logits / np.log(2), min=min_real)
            c_probs = c_logits.exp()
            c_entropy = torch.linalg.vecdot(-c_probs, c_log2_prob)

            prefix_logits.append(p_logits.logsumexp(0))
            cond_entropy.append(c_entropy)
            cond_p_eos.append(c_probs[:, 0])

        prefix_probs = logits_to_probs(torch.cat(prefix_logits)) * p_not_eosed
        cond_entropy = torch.cat(cond_entropy)
        cond_p_eos = torch.cat(cond_p_eos)

        symbol_entropies[step] = torch.dot(prefix_probs, cond_entropy)
        p_eos = torch.dot(prefix_probs, cond_p_eos) / p_not_eosed
        eos_probs[step] = p_eos
        p_not_eosed *= 1 - p_eos

    entropy = symbol_entropies.sum()
    length_probs = torch.cat([
        eos_probs[:1],
        eos_probs[1:] * torch.cumprod(1 - eos_probs[:-1], dim=0),
    ])
    return entropy, length_probs


def relative_message_entropy(probs_p, probs_q, order=4, split_size=1000):
    """
    Computes conditional KLD: D( Mi | M1, ..., Mi-1 || M'i | M'1, ..., M'i-1)
    """
    # assert probs_p.size() == probs_q.size()
    min_real = torch.finfo(probs_p.dtype).min

    size_p, size_q = probs_p.size(), probs_q.size()
    logits_p = probs_to_logits(
        probs_p.view(size_p[0] * size_p[1], size_p[2])
    ).view(size_p)
    logits_q = probs_to_logits(
        probs_q.view(size_q[0] * size_q[1], size_q[2])
    ).view(size_q)

    p_not_eosed = 1
    symbol_kld = torch.zeros_like(probs_p[0, :, 0])

    # iterate over all symbols except for the appended EOS
    for step in range(size_p[1] - 1):
        step_logits_p, step_logits_q = logits_p[:, step], logits_q[:, step]

        if step == 0:
            log_q = step_logits_q.logsumexp(0)
            log_q -= log_q.logsumexp(0)  # normalize
            log2_q = torch.clamp(log_q / np.log(2), min=min_real)
            p = probs_p[:, step].sum(0) / size_p[0]

            symbol_kld[0] = torch.dot(-p, log2_q)
            p_not_eosed *= 1 - p[0]
            continue

        first_symbol = 0 if order is None else max(0, step - order)
        prev_logits_p = logits_p[:, first_symbol:step, 1:]
        prev_logits_q = logits_q[:, first_symbol:step, 1:]

        # iterate over all possible non-EOS prefixes of the symbol
        cond_kld, prefix_logits, cond_p_eos = [], [], []
        for idx in get_prefix_indices(prev_logits_p, split_size):
            # log-probs of previous symbols being equal to a given prefix
            prefix_logits_p = aggregate_prefix_logits(prev_logits_p, idx)
            prefix_logits_q = aggregate_prefix_logits(prev_logits_q, idx)

            # conditional log-probs of each symbol at position i for each prefix
            c_logits_p = torch.logsumexp(
                step_logits_p.unsqueeze(1) + prefix_logits_p.unsqueeze(-1), dim=0)
            c_logits_p -= c_logits_p.logsumexp(-1, keepdim=True)  # normalize
            c_log2_p = c_logits_p / np.log(2)
            c_probs_p = c_logits_p.exp()

            c_logits_q = torch.logsumexp(
                step_logits_q.unsqueeze(1) + prefix_logits_q.unsqueeze(-1), dim=0)
            c_logits_q -= c_logits_q.logsumexp(-1, keepdim=True)  # normalize
            c_log2_q = c_logits_q / np.log(2)

            # conditional KLD: D( Mi | M1, ..., Mi-1 || M'i | M'1, ..., M'i-1)
            c_log2_pq = torch.clamp(c_log2_p - c_log2_q, min=min_real)
            c_kld = torch.linalg.vecdot(c_probs_p, c_log2_pq)

            cond_kld.append(c_kld)
            prefix_logits.append(prefix_logits_p.logsumexp(0))
            cond_p_eos.append(c_probs_p[:, 0])

        cond_kld = torch.cat(cond_kld)
        prefix_probs = logits_to_probs(torch.cat(prefix_logits)) * p_not_eosed
        cond_p_eos = torch.cat(cond_p_eos)

        symbol_kld[step] = torch.dot(prefix_probs, cond_kld)
        p_eos = torch.dot(prefix_probs, cond_p_eos) / p_not_eosed
        p_not_eosed *= 1 - p_eos

    return symbol_kld.sum()


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


def joint_entropy(probs, y, split_size=1000):
    """
    Given symbol probabilities representing realizations of a message M
    (M = M1, ..., Mn) and corresponding realizations of a non-compound RV Y,
    computes joint entropy H(M1, ..., Mn, Y) of message symbols and Y using
    the formula H(M1, ..., Mn, Y) = H(Y) + H(M1, ..., Mn | Y).
    """
    assert len(y) == y.numel()
    _, indices = torch.unique(y, dim=0, return_inverse=True)

    H_msg_y = intervention.entropy(indices)  # H(Y)
    for cat in range(indices.max() + 1):
        cat_mask = indices == cat
        prob_cat = (indices == cat).float().mean()
        entropy_cat, _ = message_entropy(probs[cat_mask], split_size)
        H_msg_y += prob_cat * entropy_cat  # P(Y = y) * H(M | Y = y)

    return H_msg_y


def compute_mi(probs, attributes, entropy_message=None, split_size=1000):
    if entropy_message is None:
        entropy_message, _ = message_entropy(probs)

    if attributes.size(1) == 1:
        entropy_attr = intervention.entropy(attributes)
        entropy_msg_attr = joint_entropy(probs, attributes, split_size)
        mi_msg_attr = entropy_message + entropy_attr - entropy_msg_attr
        vi_msg_attr = 2 * entropy_msg_attr - entropy_message - entropy_attr
        is_norm_msg_attr = mi_msg_attr / entropy_msg_attr

        output = {
            'entropy_msg': entropy_message,
            'entropy_attr': entropy_attr,
            'MI_msg_attr': mi_msg_attr,
            'VI_msg_attr': vi_msg_attr,
            'IS_msg_attr': is_norm_msg_attr,
        }

    else:  # return values per attribute dimension instead
        _, categorical = torch.unique(attributes, dim=0, return_inverse=True)
        entropy_attr = intervention.entropy(categorical)
        entropy_attr_dim = [
            intervention.entropy(attributes[:, i])
            for i in range(attributes.size(-1))]
        entropy_msg_attr_dim = [
            joint_entropy(probs, attributes[:, i], split_size)
            for i in range(attributes.size(1))]
        mi_msg_attr_dim = [
            entropy_message + H_y - H_xy
            for H_y, H_xy in zip(entropy_attr_dim, entropy_msg_attr_dim)
        ]
        vi_msg_attr_dim = [
            2 * entropy_msg_attr - entropy_message - entropy_attr
            for entropy_attr, entropy_msg_attr
            in zip(entropy_attr_dim, entropy_msg_attr_dim)]
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


def compute_accuracy2(messages, receiver_inputs, labels, receiver, opts):
    messages, receiver_inputs, labels = truncate_messages(
        messages, receiver_inputs, labels, opts.mode)

    dataset = CustomDataset(messages, receiver_inputs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        drop_last=False)

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

        for i in range(sender_inputs.size(1)):
            x, y = messages[:, j], sender_inputs[:, i]
            y = sender_inputs[:, i]
            H_j = intervention.entropy(y)
            info = intervention.mutual_info(x, y)
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
