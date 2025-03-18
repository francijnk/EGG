from __future__ import annotations

import torch
import argparse
import numpy as np
from egg.core.util import find_lengths
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance  # , ratio
from scipy.stats import spearmanr  # pearsonr
from torch.utils.data import Dataset
from itertools import combinations
from egg.zoo.language_bottleneck import intervention
from torch.distributions.utils import logits_to_probs
from torch.distributions.categorical import Categorical
import pyitlib.discrete_random_variable as drv

from typing import Optional, List, Tuple, Dict


min_real = torch.finfo(torch.get_default_dtype()).min


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
    aggregated = prev_symbols.transpose(0, -1)[idx].sum(1).t()
    return aggregated.clamp(min=min_real)


def unique_samples(
    receiver_inputs: torch.Tensor,
    attributes: Optional[torch.Tensor] = None
) -> Tuple[float, int]:
    size = receiver_inputs.size()

    if attributes is None:
        # categorize VISA features and get target objects for each sample
        r_inputs = receiver_inputs.view(size[0] * size[1], size[2])
        _, r_inputs_cat = torch.unique(r_inputs, dim=0, return_inverse=True)

        # sort objects in each sample, so that their order doesn't matter
        # then categorize object samples
        samples, _ = r_inputs_cat.view(size[:-1]).sort()

    else:  # instead of receiver_inputs, use the attribute provided
        # if more than one attribute per object is provided, categorize
        if attributes.dim() == 3:  # (n_messages, n_distractors, n_attributes)
            _, attributes = torch.unique(
                attributes.view(size[0] * size[1], -1),
                return_inverse=True,
                dim=0,
            )
            attributes = attributes.view(size[:2])

        samples, _ = attributes.sort()

    return len(torch.unique(samples, dim=0))


def tensor_binary_entropy(p: torch.Tensor):
    q = 1 - p
    log2_p = torch.clamp(torch.log2(p), min=min_real)
    log2_q = torch.clamp(torch.log2(q), min=min_real)
    return -p * log2_p - q * log2_q


def message_entropy(
    logits: torch.Tensor,
    order: int = 4,
    split_size: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:

    size = logits.size()

    symbol_entropies = torch.zeros_like(logits[0, :, 0])
    logp_not_eosed = 0
    eos_logits = torch.zeros_like(logits[0, :, 0])

    # iterate over all symbols except for the appended EOS
    for step in range(size[1] - 1):
        symbol_logits = logits[:, step]
        if step == 0:
            log_prob = symbol_logits.logsumexp(0).log_softmax(0)
            prob = logits_to_probs(log_prob)

            log2_prob = torch.clamp(log_prob / np.log(2), min=min_real)
            symbol_entropies[0] = torch.dot(-prob, log2_prob)

            eos_logits[0] = torch.log(prob[0]).clamp(min=min_real)
            logp_not_eosed = (
                logp_not_eosed + torch.log(1 - prob[0])
            ).clamp(min_real, 0)
            continue

        first_symbol = 0 if order is None else max(0, step - order)
        prev_logits = logits[:, first_symbol:step, 1:]

        # iterate over all possible non-EOS prefixes of the symbol
        prefix_logits, cond_eos_logits, cond_entropy = [], [], []
        for idx in get_prefix_indices(prev_logits, split_size):
            # log-probs of previous symbols being equal to a given prefix
            p_logits = aggregate_prefix_logits(prev_logits, idx)

            # conditional entropy for each prefix
            c_log_prob = (
                symbol_logits.unsqueeze(1) + p_logits.unsqueeze(-1)
            ).logsumexp(0).log_softmax(-1)
            c_log2_prob = torch.clamp(c_log_prob / np.log(2), min=min_real)
            c_prob = logits_to_probs(c_log_prob)

            c_entropy = torch.matmul(
                -c_prob.unsqueeze(1), c_log2_prob.unsqueeze(-1)
            ).view(-1)

            prefix_logits.append(p_logits.logsumexp(0))
            cond_eos_logits.append(c_log_prob[:, 0])
            cond_entropy.append(c_entropy)

        prefix_logits = torch.cat(prefix_logits).log_softmax(0)
        cond_eos_logits = torch.cat(cond_eos_logits)
        cond_entropy = torch.cat(cond_entropy)

        logp_eos = (prefix_logits + cond_eos_logits).logsumexp(0).clamp(min_real, 0)

        symbol_entropies[step] = torch.dot(
            (prefix_logits + logp_not_eosed).exp(), cond_entropy
        )
        eos_logits[step] = logp_eos
        logp_not_eosed = (
            logp_not_eosed + torch.log(1 - logp_eos.exp())
        ).clamp(min_real, 0)

    entropy = symbol_entropies.sum()
    length_probs = torch.cat([
        eos_logits[:1].exp(),
        torch.exp(
            eos_logits[1:]
            + torch.cumsum(torch.log(1 - eos_logits[:-1].exp()), dim=0)
        ),
    ])

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
    assert probs_p.numel() > 0 and probs_q.numel() > 0

    logits_p = probs_p
    logits_q = probs_q

    logp_not_eosed = 0
    symbol_kld = torch.zeros_like(probs_p[0, :, 0])

    # iterate over all symbols except for the appended EOS
    for step in range(logits_p.size(1) - 1):
        step_logits_p, step_logits_q = logits_p[:, step], logits_q[:, step]

        if step == 0:
            log_p = step_logits_p.logsumexp(0).log_softmax(0)
            p = logits_to_probs(log_p.clamp(min=min_real))
            log_q = step_logits_q.logsumexp(0).log_softmax(0)
            log2_q = torch.clamp(log_q / np.log(2), min=min_real)

            symbol_kld[0] = torch.dot(-p, log2_q)
            logp_not_eosed = (
                logp_not_eosed + torch.log(1 - p[0].clamp(0, 1))
            ).clamp(min_real, 0)
            continue

        first_symbol = 0 if order is None else max(0, step - order)
        prev_logits_p = logits_p[:, first_symbol:step, 1:]
        prev_logits_q = logits_q[:, first_symbol:step, 1:]

        # iterate over all possible non-EOS prefixes of the symbol
        cond_kld, prefix_logits, cond_eos_logits = [], [], []
        for idx in get_prefix_indices(prev_logits_p, split_size):
            # log-probs of previous symbols being equal to a given prefix
            prefix_logits_p = aggregate_prefix_logits(prev_logits_p, idx)
            prefix_logits_q = aggregate_prefix_logits(prev_logits_q, idx)

            # conditional log-probs of each symbol at position i for each prefix
            c_log_p = (
                step_logits_p.unsqueeze(1) + prefix_logits_p.unsqueeze(-1)
            ).logsumexp(0).log_softmax(-1)
            c_log2_p = (c_log_p / np.log(2)).clamp(min=min_real)
            c_p = logits_to_probs(c_log_p)

            c_log_q = (
                step_logits_q.unsqueeze(1) + prefix_logits_q.unsqueeze(-1)
            ).logsumexp(0).log_softmax(-1)
            c_log2_q = (c_log_q / np.log(2)).clamp(min=min_real)

            # conditional KLD: D( Mi | M1, ..., Mi-1 || M'i | M'1, ..., M'i-1)
            c_log2_pq = torch.clamp(c_log2_p - c_log2_q, min=min_real)
            c_kld = torch.matmul(
                c_p.unsqueeze(1),
                c_log2_pq.unsqueeze(-1),
            ).view(-1)
            if torch.any(c_kld.isnan()):
                print('prob', c_p[:20])
                print('logp', c_log2_p[:20])
                print('logq', c_log2_q[:20])

            cond_kld.append(c_kld)
            prefix_logits.append(prefix_logits_p.logsumexp(0))
            cond_eos_logits.append(c_log_p[:, 0])

        cond_kld = torch.cat(cond_kld)
        prefix_logits = torch.cat(prefix_logits).log_softmax(0)
        cond_eos_logits = torch.cat(cond_eos_logits)

        logp_eos = (prefix_logits + cond_eos_logits).logsumexp(0)

        symbol_kld[step] = torch.dot(
            logits_to_probs(prefix_logits), cond_kld
        ) * logp_not_eosed.exp()
        logp_not_eosed = (
            logp_not_eosed + torch.log(1 - logp_eos.exp().clamp(0, 1))
        ).clamp(min_real, 0)

    return symbol_kld.sum()


# Redundancy
def compute_max_rep(messages: torch.Tensor) -> torch.Tensor:
    """
    Computes the number of occurrences of the most frequent symbol in each
    message (0 for messages that consist of EOS symbols only).
    """

    all_symbols = torch.unique(torch.flatten(messages), dim=0)
    non_eos_symbols = all_symbols[all_symbols != 0]

    output = torch.zeros(messages.size(0)).to(messages.device)
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

    entropy_msg_y = intervention.entropy(indices)  # H(Y)
    for cat in range(indices.max() + 1):
        cat_mask = indices == cat
        prob_cat = (cat_mask).float().mean()
        entropy_cat, _ = message_entropy(
            probs[cat_mask],
            split_size=split_size,
        )
        entropy_msg_y += prob_cat * entropy_cat  # P(Y = y) * H(M | Y = y)

    return entropy_msg_y


def crop(sample):
    lengths = find_lengths(sample)
    not_eosed = (
        torch.unsqueeze(
            torch.arange(0, sample.size(1)),
            dim=0,
        ).expand(sample.size()[:2]).to(sample.device)
        < torch.unsqueeze(lengths - 1, dim=-1).expand(sample.size()[:2])
    )
    return torch.where(not_eosed, sample, 0)



def mutual_info_sent_received(
    logits_sent: torch.Tensor,
    logits_received: torch.Tensor,
    max_len: int,
    vocab_size: int,
    erasure_channel: bool = False,
    n_samples: int = 100,
):
    size = logits_sent.size()
    sample_sent = Categorical(logits=logits_sent).sample((n_samples,))
    sample_received = Categorical(logits=logits_received).sample((n_samples,))
    sample_sent = sample_sent.reshape(size[0] * n_samples, size[1])
    sample_received = sample_received.reshape(size[0] * n_samples, size[1])

    sample_sent, sample_received = crop(sample_sent), crop(sample_received)
    _, sent = torch.unique(sample_sent, return_inverse=True, dim=0)
    _, received = torch.unique(sample_received, return_inverse=True, dim=0)

    n_messages_sent = ((vocab_size - 1) ** np.arange(max_len + 1)).sum()
    n_messages_received = ((vocab_size) ** np.arange(max_len + 1)).sum() \
        if erasure_channel else n_messages_sent

    return drv.information_mutual(
        sent.cpu().numpy(), received.cpu().numpy(),
        estimator='PERKS',
        Alphabet_X=np.arange(n_messages_sent.sum()),
        Alphabet_Y=np.arange(n_messages_received.sum()),
    ).item()


def entropy_message_as_a_whole(
    # messages: torch.Tensor,
    logits: torch.Tensor,
    max_len: int,
    vocab_size: int,
    n_samples: int = 100,
    erasure_channel: bool = False,
):
    # sample = messages
    sample = Categorical(logits=logits).sample((n_samples,))
    size = sample.size()
    sample = crop(sample.reshape(size[0] * size[1], size[2]))
    n_messages = np.sum(
        (vocab_size if erasure_channel else vocab_size - 1)
        ** np.arange(max_len + 1))
    _, sample = torch.unique(sample, return_inverse=True, dim=0)
    return drv.entropy(
        sample.cpu().numpy(),
        estimator='PERKS',
        Alphabet_X=np.arange(n_messages),
    ).item()


def compute_mi(
    probs: torch.Tensor,
    attributes: torch.Tensor,
    entropy_message: Optional[torch.Tensor] = None,
    split_size: int = 1000,
) -> Dict[str, float]:

    if entropy_message is None:
        entropy_message, _ = message_entropy(probs, split_size)

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
    remove_n: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Removes all possible combinations of `n` symbols from the tensor,
    symbol 0 is never removed.

    Args:
        tensor (torch.Tensor): The input tensor.
        n (int): The number of items to remove.

    Returns:
        list[torch.Tensor]: A list of tensors with `n` items removed.
    """
    new_messages, new_r_input, new_labels = [], [], []
    for i, message in enumerate(messages):
        num_rows = message.shape[0]  # Get the number of rows (N)

        # Ensure there are enough rows to remove `n` and keep the last row
        if remove_n >= num_rows:
            raise ValueError("Cannot remove more rows than available (excluding the last row).")
        if num_rows <= 1:
            raise ValueError("The input tensor must have more than one row.")

        # Get indices of rows that can be removed (exclude the last row)
        removable_indices = list(range(num_rows - 1))  # Exclude last row

        # Generate all combinations of `n` rows to remove
        combos = list(combinations(removable_indices, remove_n))

        truncated = []  # Create new tensors with the selected rows removed
        for combo in combos:
            mask = torch.ones(num_rows, dtype=torch.bool)
            mask[list(combo)] = False  # Set rows in the combo to False (remove them)
            new = message[mask].to(torch.float)
            truncated.append(new)

        new_messages.extend(truncated)
        new_r_input.extend([receiver_input[i]] * len(truncated))
        new_labels.extend([labels[i]] * len(truncated))

    return new_messages, new_r_input, new_labels


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
    messages, receiver_inputs, labels = \
        truncate_messages(messages, receiver_inputs, labels)

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
    attributes = torch.cat(one_hots, dim=-1).cpu().numpy()

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
