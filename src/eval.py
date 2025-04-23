from __future__ import annotations

import torch
import argparse
import numpy as np
import torch.nn.functional as F
from egg.core.util import find_lengths
# from sklearn import preprocessing
from scipy.stats import spearmanr
# from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from torch.utils.data import Dataset
from operator import attrgetter
from itertools import combinations
# from torch.nn.functional import one_hot
from torch.distributions.utils import logits_to_probs
from torch.distributions.categorical import Categorical
import pyitlib.discrete_random_variable as drv
from rapidfuzz.distance import Levenshtein, DamerauLevenshtein

from src.channels import Channel

from typing import Optional, Union, List, Tuple, Dict

from functools import wraps
from time import perf_counter


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result
    return wrap


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


def get_alphabets(
    max_len: int,
    vocab_size: int,
    erasure_channel: bool,
) -> Tuple[np.array, np.array]:

    n_messages_sent = ((vocab_size - 1) ** np.arange(max_len + 1)).sum()
    n_messages_received = ((vocab_size) ** np.arange(max_len + 1)).sum() \
        if erasure_channel else n_messages_sent

    return np.arange(n_messages_sent), np.arange(n_messages_received)


def relative_message_entropy(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    order: int = 4,
    split_size: int = 1000,
) -> torch.Tensor:
    """
    Computes KLD between messages based on two symbol distributions.
    """
    assert logits_p.numel() > 0 and logits_q.numel() > 0

    logp_not_eosed = 0
    symbol_kld = torch.zeros_like(logits_p[0, :, 0])

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
    logits: torch.Tensor,
    y: torch.Tensor,
    split_size: int = 1000,
) -> float:
    pass


def crop_messages(
    messages: torch.Tensor,
    lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Removes non EOS symbols after the first EOS from the message tensor.
    Message tensor may have 2 or 3 dimensions (if messages are one-hot
    encoded).
    """
    symbols = messages if messages.dim() == 2 else messages.argmax(-1)
    if lengths is None:
        lengths = find_lengths(
            symbols if torch.all(symbols[:, -1] == 0)
            else torch.cat([symbols, torch.zeros_like(symbols[:, :1])], dim=-1)
        )

    not_eosed = (
        torch.unsqueeze(
            torch.arange(0, symbols.size(1)),
            dim=0,
        ).expand(symbols.size()[:2]).to(symbols.device)
        < torch.unsqueeze(lengths - 1, dim=-1).expand(symbols.size()[:2])
    )

    cropped_symbols = torch.where(not_eosed, symbols, 0)
    if messages.dim() == 2:
        return cropped_symbols
    else:  # convert to one-hot vectors
        cropped_probs = torch.zeros_like(messages).view(-1, messages.size(2))
        cropped_probs.scatter_(1, cropped_symbols.view(-1, 1), 1)
        return cropped_probs.view(messages.size())


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
    channel: Channel,
    entropy_sent: float,
    entropy_received: float,
    max_len: int,
    vocab_size: int,
    erasure_channel: bool = False,
    n_samples: int = 100,
    estimator: str = 'PERKS',
) -> float:
    size = logits_sent.size()
    n_messages_sent = ((vocab_size - 1) ** np.arange(max_len + 1)).sum()
    n_messages_received = ((vocab_size) ** np.arange(max_len + 1)).sum() \
        if erasure_channel else n_messages_sent

    sample_sent = Categorical(logits=logits_sent).sample((n_samples,))
    sample_sent = crop_messages(
        sample_sent.reshape(size[0] * n_samples, size[1])
    )

    one_hots_sent = torch.zeros(
        size[0] * n_samples, *size[1:]
    ).to(logits_sent).view(-1, size[2])
    one_hots_sent.scatter_(1, sample_sent.view(-1, 1), 1)
    one_hots_sent = one_hots_sent.view(size[0] * n_samples, *size[1:])

    one_hots_received, _ = channel.process(
        one_hots_sent[:, :-1],
        logits_sent[:, :-1],
        True)
    sample_received = one_hots_received.argmax(-1)

    _, sent = torch.unique(sample_sent, return_inverse=True, dim=0)
    _, received = torch.unique(sample_received, return_inverse=True, dim=0)

    alphabet_received = np.arange(n_messages_received)
    alphabet_sent = alphabet_received.copy()
    alphabet_sent[alphabet_sent >= n_messages_sent] = -1
    alphabet = np.hstack([alphabet_sent, alphabet_received])

    entropy_joint = drv.entropy_joint(
        torch.cat([sent, received]).cpu().numpy(),
        Alphabet_X=alphabet,
        estimator=estimator,
    ).item()

    return entropy_sent + entropy_received - entropy_joint
    return drv.information_mutual(
        sent.cpu().numpy(), received.cpu().numpy(),
        estimator=estimator,
        Alphabet_X=np.arange(n_messages_sent),
        Alphabet_Y=np.arange(n_messages_received),
    ).item()


def mutual_info_message_attributes(
    logits: torch.Tensor,
    attributes: torch.Tensor,
    max_len: int,
    vocab_size: int,
    erasure_channel: bool,
    n_samples: int = 100,
    estimator: str = 'PERKS',
) -> float:

    n_messages = np.sum(
        (vocab_size if erasure_channel else vocab_size - 1)
        ** np.arange(max_len + 1))

    sample = Categorical(logits=logits).sample((n_samples,))
    size = sample.size()
    sample = crop(sample.reshape(size[0] * size[1], size[2]))

    _, sample = torch.unique(sample, return_inverse=True, dim=0)
    _, attributes = torch.unique(attributes, return_inverse=True, dim=0)
    attributes = attributes.expand(n_samples, *attributes.size()).reshape(size[0] * size[1])

    return drv.information_mutual(
        sample.cpu().numpy(),
        attributes.cpu().numpy(),
        Alphabet_X=np.arange(n_messages),
        estimator=estimator,
    ).item()


def message_entropy_mc(
    logits: torch.Tensor,
    max_len: int,
    vocab_size: int,
    n_samples: int = 100,
    erasure_channel: bool = False,
    estimator: str = 'PERKS',
) -> float:

    n_messages = np.sum(
        (vocab_size if erasure_channel else vocab_size - 1)
        ** np.arange(max_len + 1))
    sample = Categorical(logits=logits).sample((n_samples,))
    size = sample.size()
    sample = crop(sample.reshape(size[0] * size[1], size[2]))
    _, sample = torch.unique(sample, return_inverse=True, dim=0)
    return drv.entropy(
        sample.cpu().numpy(),
        estimator=estimator,
        Alphabet_X=np.arange(n_messages),
    ).item()


def compute_mi(
    logits: torch.Tensor,
    attributes: torch.Tensor,
    max_len: int,
    vocab_size: int,
    erasure_channel: bool,
    entropy_message: Optional[torch.Tensor] = None,
    n_samples: int = 100,
    estimator: str = 'PERKS',
) -> Dict[str, Union[float, List[float]]]:

    if entropy_message is None:
        entropy_message = message_entropy_mc(
            logits,
            max_len=max_len, vocab_size=vocab_size,
            n_samples=n_samples, erasure_channel=erasure_channel,
        )

    _, attr = torch.unique(attributes, return_inverse=True, dim=0)
    entropy_attr = drv.entropy(
        attr.cpu().numpy(),
        estimator=estimator,
        fill_value=-9,
    ).item()
    mi_msg_attr = mutual_info_message_attributes(
        logits=logits, attributes=attributes, max_len=max_len,
        vocab_size=vocab_size, n_samples=n_samples,
        erasure_channel=erasure_channel)
    proficiency_msg_attr = mi_msg_attr / entropy_attr
    redundancy_msg_attr = mi_msg_attr / (entropy_message + entropy_attr)

    output = {
        'entropy_msg': entropy_message,
        'entropy_attr': entropy_attr,
        'mutual_info_msg_attr': mi_msg_attr,
        'proficiency_msg_attr': proficiency_msg_attr,
        'redundancy_msg_attr': redundancy_msg_attr,
    }

    if attributes.size(1) > 1:
        entropy_attr_dim = [
            drv.entropy(
                attributes[:, i].cpu().numpy(),
                estimator=estimator,
                fill_value=-10,
            ).item()
            for i in range(attributes.size(-1))]
        mi_msg_attr_dim = [
            mutual_info_message_attributes(
                logits, attributes[:, i],
                max_len=max_len,
                vocab_size=vocab_size,
                n_samples=n_samples,
                erasure_channel=erasure_channel)
            for i in range(attributes.size(1))
        ]
        redundancy_msg_attr_dim = [
            mi_msg_attr / (entropy_message + entropy_attr)
            for mi_msg_attr, entropy_attr
            in zip(mi_msg_attr_dim, entropy_attr_dim)]
        proficiency_msg_attr_dim = [
            mi_msg_attr / entropy_attr
            for mi_msg_attr, entropy_attr
            in zip(mi_msg_attr_dim, entropy_attr_dim)]

        output.update({
            'entropy_attr_dim': entropy_attr_dim,
            'mutual_info_msg_attr_dim': mi_msg_attr_dim,
            'proficiency_msg_attr_dim': proficiency_msg_attr_dim,
            'redundancy_msg_attr_dim': redundancy_msg_attr_dim,
        })

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
    def __init__(
        self,
        messages: torch.Tensor,
        receiver_inputs: torch.Tensor,
    ):
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


class MessageDataset2(Dataset):
    def __init__(
        self,
        messages: torch.Tensor,
        receiver_inputs: torch.Tensor,
        labels: torch.Tensor,
        disruption_fn: callable,
        n_symbols: int,
        lengths: Optional[torch.Tensor] = None,
        n_samples: int = 5,
    ):

        rng = np.random.default_rng()
        if lengths is None:
            lengths = find_lengths(messages.argmax(-1))

        assert (
            len(messages) == len(receiver_inputs) == len(lengths) == len(labels)
        )

        count = 0
        self.messages, self.r_inputs, self.labels = [], [], []
        for message, r_input, label, length in zip(
            messages, receiver_inputs, labels, lengths
        ):
            # get all possible non-EOS symbol combinations
            available_indices = range(length.item() - 1)
            r = n_symbols if n_symbols == 1 \
                else max(min(n_symbols, length.item() - 1), 2)
            all_combos = list(combinations(available_indices, r))

            # sample: if message is to short, save undisrupted message
            combos = rng.choice(all_combos, n_samples, replace=True).tolist() \
                if len(all_combos) > 0 else [None for _ in range(n_samples)]

            for combo in combos:
                disrupted = disruption_fn(message, combo) \
                    if combo is not None else message
                self.messages.append(disrupted)
                self.r_inputs.append(r_input)
                self.labels.append(label)
                if len(disrupted) == len(message):
                    count += torch.any(disrupted != message).int().item()

        print('different message after disruption:', count , '/', len(self.messages), disruption_fn, n_symbols)

        self.labels = torch.stack(self.labels)

        assert (
            len(self.messages)
            == len(self.r_inputs)
            == len(self.labels)
        ), "Messages and receiver_inputs must have the same number of samples."

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx], self.r_inputs[idx]


def compute_disruption_accuracy(
    messages: torch.Tensor,
    receiver_inputs: torch.Tensor,
    labels: torch.Tensor,
    receiver: torch.nn.Module,
    opts: argparse.Namespace,
):

    rng = np.random.default_rng()
    eos = torch.zeros_like(messages[0, :1])
    eos[..., 0] = 1

    # def erase(message, combo):
    #     message = message.clone()
    #     message[list(combo)] = 0
    #     message[list(combo), opts.vocab_size] = 1
    #     return message

    def delete(message, combo):
        mask = torch.ones(len(message), device=message.device).bool()
        mask[list(combo)] = False
        return torch.cat([message[mask], eos])

    def replace(message, combo):
        actual_symbol = message[combo[0]].argmax(-1).item()
        candidates = np.arange(1, opts.vocab_size)
        candidates = candidates[candidates != actual_symbol]
        message = message.clone()
        replacement = torch.zeros_like(message[0])
        replacement[rng.choice(candidates)] = 1
        message[combo[0]] = replacement
        return message

    def permute(message, combo):
        message = message.clone()
        perm = np.arange(len(combo), dtype=np.int32)
        while np.any(perm == np.arange(len(combo))):
            perm = rng.permutation(perm)

        message[combo] = message[[combo[p] for p in perm]]
        return message

    output = {}
    disruptions = {  # (callable, number of symbols in a combination)
        'deletion': (delete, 1),
        'replacement': (replace, 1),
    }
    for n in range(2, opts.max_len + 1):
        disruptions[f'permutation_{n}'] = (permute, n)

    lengths = find_lengths(messages.argmax(-1))
    for key, val in disruptions.items():
        dataset = MessageDataset2(messages, receiver_inputs, labels, *val, lengths)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            drop_last=False)

        predictions = []
        check = False
        for b_messages, b_inputs in dataloader:
            with torch.no_grad():
                outputs = receiver(b_messages, b_inputs)

            b_lengths = find_lengths(b_messages.argmax(-1))
            idx = (torch.arange(len(b_messages)), b_lengths - 1)
            if check:
                print('b msg shape', b_messages.shape, key)
                print('b msg idx', b_messages[idx].argmax(-1), b_lengths)
                print(b_messages[:3].argmax(-1))
                print('')
                check = False
                print(outputs[idx].shape, outputs[idx][0])
            predictions.extend(outputs[idx].argmax(-1))

        predictions = torch.stack(predictions)
        output[f'accuracy_{key}'] = torch.mean(
            (predictions == dataset.labels).float()
        ).item()

    return output


# Compositionality
def compute_topsim(
    meanings: torch.Tensor,
    messages: torch.Tensor,
    meaning_distance: str = 'hamming',
    message_distance: str = 'levenshtein',
    norm: Optional[str] = None,
) -> float:
    """
    Computes topographic similarity.
    Optionally uses Damerau-Levenshtein edit distance for messages.
    """
    assert meaning_distance in ('cosine', 'hamming', 'euclidean')
    assert message_distance in ('levenshtein', 'damerau_levenshtein')
    assert norm in ('mean', 'max', None)

    i, j = torch.combinations(
        torch.arange(len(messages), device=messages.device)
    ).unbind(-1)
    messages = [
        ''.join([chr(s.item() + 65) for s in m[:l].int()])
        for m, l in zip(messages.int(), find_lengths(messages))
    ]

    def cosine_distance(x):
        normalized = F.normalize(x.double(), dim=1)
        dists = 1 - (normalized @ normalized.t())
        return dists[i, j].cpu().numpy()

    def hamming_distance(x):
        return (F.pdist(x.double(), p=0) / meanings.size(1)).cpu().numpy()

    def euclidean_distance(x):
        return F.pdist(x.double(), p=2).cpu().numpy()

    # def jaccard_distance(x):
    #     return pdist(x.cpu().numpy(), metric='jaccard')
    #     # return 1 - binary_jaccard_index(x[i], x[j]).cpu().numpy()

    meth = attrgetter('normalized_distance' if norm == 'max' else 'distance')
    distances = {
        'cosine': cosine_distance,
        'hamming': lambda x: (F.pdist(x.double(), p=0) / len(meanings)).cpu().numpy(),
        'euclidean': lambda x: F.pdist(x.double(), p=2).cpu().numpy(),
        # 'jaccard': jaccard_distance,
        'levenshtein': meth(Levenshtein),
        'damerau_levenshtein': meth(DamerauLevenshtein),
    }

    meaning_dist_fn = distances[meaning_distance]
    message_dist_fn = distances[message_distance] if norm != 'mean' else \
        lambda x, y: distances[message_distance](x, y) / (len(x) + len(y)) / 2

    meaning_dists = meaning_dist_fn(meanings)
    message_dists = [
        message_dist_fn(messages[i], messages[j])
        for i, j in combinations(range(len(messages)), 2)
    ]

    topsim = spearmanr(meaning_dists, message_dists).statistic
    return topsim if not np.isnan(topsim) else 0.
