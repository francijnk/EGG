from __future__ import annotations

import torch
import argparse
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

from typing import Optional, List, Tuple, Dict


# Entropy, Mutual information
def get_prefix_indices(
    prev_symbols: torch.Tensor, split_size: int = 1000
) -> torch.Tensor:
    n_prev = prev_symbols.size(1)
    prefix_indices = torch.cartesian_prod(
        *(torch.arange(prev_symbols.size(-1)) for i in range(n_prev))
    ).view(-1, n_prev)

    for indices in torch.split(prefix_indices, split_size):
        yield indices, torch.arange(n_prev).view(1, -1).expand(indices.size())


def aggregate_prefix_logits(
        prev_symbols: torch.Tensor, idx: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    min_real = torch.finfo(prev_symbols.dtype).min
    aggregated = prev_symbols.transpose(0, -1)[idx].sum(1).t()
    return aggregated.clamp(min=min_real)


def min_message_entropy(
    receiver_inputs: torch.Tensor,
    labels: torch.Tensor,
    attributes: Optional[torch.Tensor] = None
) -> Tuple[float, int]:
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


def message_entropy(
    probs: torch.Tensor, order: int = 4, split_size: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_real = torch.finfo(probs.dtype).min

    size = probs.size()
    logits = probs_to_logits(
        probs.view(size[0] * size[1], size[2])
    ).view(size)

    symbol_entropies = torch.zeros_like(probs[0, :, 0])
    eos_probs = torch.ones_like(probs[0, :, 0])
    eos_logits = torch.ones_like(probs[0, :, 0])
    p_not_eosed = 1

    # iterate over all symbols except for the appended EOS
    for step in range(size[1] - 1):
        symbol_logits = logits[:, step]

        if step == 0:
            log_prob = symbol_logits.logsumexp(0)
            log_prob -= log_prob.logsumexp(0)  # normalize
            log2_prob = torch.clamp(log_prob / np.log(2), min=min_real)
            prob = logits_to_probs(log_prob)
            # prob = probs[:, step].sum(0) / probs.size(0)
            symbol_entropies[0] = torch.dot(-prob, log2_prob)
            eos_probs[0] = prob[0]
            p_not_eosed *= 1 - prob[0]
            # print('ZERO', p_not_eosed, torch.allclose(p_not_eosed, torch.zeros_like(p_not_eosed)))
            # print('ZERO:0', prob)
            continue

        check = False
        # print("STEP", step, "not eosed:", p_not_eosed)
        first_symbol = 0 if order is None else max(0, step - order)
        prev_logits = logits[:, first_symbol:step, 1:]

        # iterate over all possible non-EOS prefixes of the symbol
        prefix_logits, cond_entropy, cond_p_eos = [], [], []
        cond_eos_logits = []
        for idx in get_prefix_indices(prev_logits, split_size):
            # log-probs of previous symbols being equal to a given prefix
            p_logits = aggregate_prefix_logits(prev_logits, idx)
            if torch.any(torch.isnan(p_logits)):
                print('P_LOGITS', p_logits.shape, probs.shape)

            # conditional entropy for each prefix
            c_logits = torch.logsumexp(
                symbol_logits.unsqueeze(1) + p_logits.unsqueeze(-1), dim=0)
            c_logits -= c_logits.logsumexp(-1, keepdim=True)  # normalize
            c_log2_prob = torch.clamp(c_logits / np.log(2), min=min_real)
            c_probs = logits_to_probs(c_logits)
            c_entropy = torch.linalg.vecdot(-c_probs, c_log2_prob)
            threshold = 1e-6
            if torch.any(p_logits.logsumexp(0).isnan()):  # probs < threshold):
                # mask = c_probs < threshold
                # mask = p_logits.logsumexp(0).isnan()
                mask = p_logits.isinf()
                print('')
                print('prob', c_probs[mask])
                print('log2', c_log2_prob[mask])
                # print('entr', c_entropy[mask])
                c_probs = torch.where(c_probs > threshold, c_probs, 0.)
                c_entropy = torch.linalg.vecdot(-c_probs, c_log2_prob)
                print('prob', c_probs[mask])
                # print('log2', c_log2_prob[mask])
                # print('entr', c_entropy[mask])

            prefix_logits.append(p_logits.logsumexp(0))
            cond_entropy.append(c_entropy)
            cond_eos_logits.append(c_logits[:, 0])
            cond_p_eos.append(c_probs[:, 0])

        prefix_probs = logits_to_probs(torch.cat(prefix_logits)) * p_not_eosed
        cond_entropy = torch.cat(cond_entropy)
        cond_p_eos = torch.cat(cond_p_eos)

        symbol_entropies[step] = torch.dot(prefix_probs, cond_entropy)
        if symbol_entropies[step].isnan():
            mult = prefix_probs * cond_entropy
            mask = (mult).isnan()
            prefix_logits = torch.cat(prefix_logits)
            print('prefix pro', prefix_probs[mask])
            mask = torch.logical_or(prefix_logits.isnan(), prefix_logits.isinf())
            print('prefix log', prefix_logits[mask])
            print('p_not_eosed', p_not_eosed)
            raise ValueError
        p_eos = torch.dot(prefix_probs, cond_p_eos) / p_not_eosed
        if False:
            mask = torch.logical_or(
                (prefix_probs * cond_p_eos).isnan(),
                (prefix_probs * cond_p_eos).isinf())
            print('cond p eos', cond_p_eos[mask])
            print('pref', prefix_probs[mask])
            print("P eos", p_eos)

        if not p_eos.isnan():  # prevents nan values for eos probs close to 1
            eos_probs[step] = p_eos
        else:
            print('EOS REPLACED')
            check = True

        p_not_eosed *= 1 - p_eos

    entropy = symbol_entropies.sum()
    # not_eosed_logits = torch.logcumsumexp(
    #     k
    #     , dim=0,
    # )
    # length_logits = torch.cat([
    #     eos_logits[:1],
    #     eos_logits[1:] * torch.logcumsumexp(torch.logsumexp(torch.e, ) , dim=0)
    # ])
    length_probs = torch.cat([
        eos_probs[:1],
        eos_probs[1:] * torch.cumprod(1 - eos_probs[:-1], dim=0),
    ])
    if torch.any(length_probs.isnan()):
        print('EOS PROBS', eos_probs)

    if check:
        print('check', eos_probs, length_probs)
    return entropy, length_probs


def relative_message_entropy(
    probs_p: torch.Tensor,
    probs_q: torch.Tensor,
    order: int = 4,
    split_size: int = 1000,
) -> torch.Tensor:
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

            log_p = step_logits_p.logsumexp(0)
            log_p -= log_p.logsumexp(0)  # normalize
            p = logits_to_probs(log_p.clamp(min=min_real))
            # p = probs_p[:, step].sum(0) / size_p[0]

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
            c_probs_p = logits_to_probs(c_logits_p)

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
        if symbol_kld[step].isnan():
            print(
                'step',
                step, torch.any(prefix_probs.isnan()), torch.any(cond_kld.isnan()),
                torch.any(cond_p_eos.isnan()), p_not_eosed.isnan(),
            )
        p_eos = torch.dot(prefix_probs, cond_p_eos) / p_not_eosed
        p_not_eosed *= 1 - p_eos
        if p_not_eosed.isnan():
            print('eos step', p_eos)

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


def joint_entropy(
    probs: torch.Tensor,
    y: torch.Tensor,
    split_size: int = 1000,
) -> torch.Tensor:
    """
    Given symbol probabilities representing realizations of a message M
    (M = M1, ..., Mn) and corresponding realizations of a non-compound RV Y,
    computes joint entropy H(M1, ..., Mn, Y) of message symbols and Y using
    the formula H(M1, ..., Mn, Y) = H(Y) + H(M1, ..., Mn | Y).
    """
    assert len(y) == y.numel()
    _, indices = torch.unique(y, dim=0, return_inverse=True)

    # exclude lowest values to improve numerical stability
    probs = torch.where(probs > 1e-8, probs, 0)
    probs = probs / probs.sum(-1, keepdim=True)

    entropy_msg_y = intervention.entropy(indices)  # H(Y)
    for cat in range(indices.max() + 1):
        cat_mask = indices == cat
        prob_cat = (cat_mask).float().mean()
        entropy_cat, _ = message_entropy(probs[cat_mask], split_size)
        entropy_msg_y += prob_cat * entropy_cat  # P(Y = y) * H(M | Y = y)

    return entropy_msg_y


def compute_mi(
    probs: torch.Tensor,
    attributes: torch.Tensor,
    entropy_message: Optional[torch.Tensor] = None,
    split_size: int = 1000,
) -> Dict[str, float]:
    if entropy_message is None:
        entropy_message, _ = message_entropy(probs)

    if attributes.size(1) == 1:
        entropy_attr = intervention.entropy(attributes)
        if entropy_message is None or entropy_message > 1e-2:
            entropy_msg_attr = joint_entropy(probs, attributes, split_size)
        else:
            entropy_msg_attr = 0
        mi_msg_attr = entropy_message + entropy_attr - entropy_msg_attr
        vi_msg_attr = 2 * entropy_msg_attr - entropy_message - entropy_attr
        proficiency_msg_attr = mi_msg_attr / entropy_attr
        redundancy_msg_attr = mi_msg_attr / (entropy_message + entropy_attr)

        output = {
            'entropy_msg': entropy_message,
            'entropy_attr': entropy_attr,
            'mutual_info_msg_attr': mi_msg_attr,
            'variation_of_info_msg_attr': vi_msg_attr,
            'proficiency_msg_attr': proficiency_msg_attr,
            'redundancy_msg_attr': redundancy_msg_attr,
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
            entropy_message + entropy_attr - entropy_msg_attr
            for entropy_attr, entropy_msg_attr
            in zip(entropy_attr_dim, entropy_msg_attr_dim)
        ]
        vi_msg_attr_dim = [
            2 * entropy_msg_attr - entropy_message - entropy_attr
            for entropy_attr, entropy_msg_attr
            in zip(entropy_attr_dim, entropy_msg_attr_dim)]
        redundancy_msg_attr_dim = [
            mi_msg_attr / (entropy_message + entropy_attr)
            for mi_msg_attr, entropy_attr
            in zip(mi_msg_attr_dim, entropy_attr_dim)]
        proficiency_msg_attr_dim = [
            mi_msg_attr / entropy_attr
            for mi_msg_attr, entropy_attr
            in zip(mi_msg_attr_dim, entropy_attr_dim)]

        output = {
            'entropy_msg': entropy_message,
            'entropy_attr': entropy_attr,
            'entropy_attr_dim': entropy_attr_dim,
            'mutual_info_msg_attr_dim': mi_msg_attr_dim,
            'variation_of_info_msg_attr_dim': vi_msg_attr_dim,
            'proficiency_msg_attr_dim': proficiency_msg_attr_dim,
            'redundancy_msg_attr_dim': redundancy_msg_attr_dim,
        }

    return output


def truncate_messages(
    messages: torch.Tensor,
    receiver_input: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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


def remove_n_items(tensor: torch.Tensor, n: int = 1) -> List[torch.Tensor]:
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


def remove_n_dims(tensor: torch.Tensor, n: int = 1) -> List[torch.Tensor]:
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


class MessageDataset(Dataset):
    def __init__(self, messages: torch.Tensor, receiver_inputs: torch.Tensor):
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


def compute_symbol_removal_accuracy(
    messages: torch.Tensor,
    receiver_inputs: torch.Tensor,
    labels: torch.Tensor,
    receiver: torch.nn.Module,
    opts: argparse.Namespace,
) -> float:
    messages, receiver_inputs, labels = truncate_messages(
        messages, receiver_inputs, labels, opts.mode)

    dataset = MessageDataset(messages, receiver_inputs)
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

    messages = [[s.int().item() for s in m if s > 0] + [0] for m in messages]

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

    return spearmanr(cos_sims, lev_dists, axis=None).statistic
