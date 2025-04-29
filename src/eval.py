from __future__ import annotations

import torch
import argparse
import numpy as np
import torch.nn.functional as F
from egg.core.util import find_lengths
from scipy.stats import spearmanr
from torch.utils.data import Dataset
from operator import attrgetter
from itertools import combinations
from torch.distributions.utils import logits_to_probs
from torch.distributions.categorical import Categorical
import pyitlib.discrete_random_variable as drv
from rapidfuzz.distance import Levenshtein, DamerauLevenshtein
from typing import Optional, Union, List, Tuple, Dict

from src.channels import Channel


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


def sample_messages(
    logits: torch.Tensor,
    *attributes,
    n_samples: int = 100,
):

    sample = Categorical(logits=logits).sample((n_samples,))
    size = sample.size()
    sample = torch.unique(
        crop_messages(sample.reshape(size[0] * size[1], size[2])),
        return_inverse=True, dim=0)[1]
    attributes = [
        torch.unique(
            attr, return_inverse=True, dim=0
        )[1].expand(n_samples, attr.size(0)).reshape(size[0] * size[1])
        for attr in attributes
    ]
    return (sample, *attributes)


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
    sample, attributes = sample_messages(logits, attributes, n_samples=n_samples)

    return drv.information_mutual(
        sample.cpu().numpy(),
        attributes.cpu().numpy(),
        Alphabet_X=np.arange(n_messages),
        estimator=estimator,
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

    n_messages = np.sum(
        (vocab_size if erasure_channel else vocab_size - 1)
        ** np.arange(max_len + 1))
    if entropy_message is None:
        sample = crop_messages(sample_messages(logits, n_samples=n_samples)[0])
        entropy_message = drv.entropy(
            sample.cpu().numpy(),
            estimator=estimator,
            Alphabet_X=np.arange(n_messages),
        ).item()

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

    output = {
        'entropy_msg': entropy_message,
        'entropy_attr': entropy_attr,
        'mutual_info_msg_attr': mi_msg_attr,
        'proficiency_msg_attr': proficiency_msg_attr,
    }

    if attributes.size(1) > 1:
        n_messages = np.sum(
            (vocab_size if erasure_channel else vocab_size - 1)
            ** np.arange(max_len + 1))
        n_attr = attributes.size(1)

        # to compute H and MI between the message and attribute conditioned
        # select remaining attributes
        idx = torch.arange(n_attr).to(attributes.device).expand(n_attr, -1)
        idx = idx[idx != idx[0].unsqueeze(-1)].view(n_attr, n_attr - 1)
        context_dim = attributes[:, idx]

        samples = sample_messages(
            logits,
            *attributes.unbind(1),
            *context_dim.unbind(1),
            n_samples=n_samples,
        )
        split = 1 + attributes.size(1)
        msg, attr_dim, ctx_dim = samples[0], samples[1:split], samples[split:]

        msg_attr_dim = [
            torch.unique(
                torch.stack([msg, attr], dim=-1),
                return_inverse=True, dim=0,
            )[1].cpu().numpy() for attr in attr_dim
        ]
        attr_dim = [attr.cpu().numpy() for attr in attr_dim]
        ctx_dim = [ctx.cpu().numpy() for ctx in ctx_dim]
        msg, alphabet_msg = msg.cpu().numpy(), np.arange(n_messages)
        n_unique_tuples_dim = [n_messages * len(np.unique(attr)) for attr in attr_dim]

        cond_entropy_msg_attr_dim = [
            drv.entropy_conditional(
                msg_attr, ctx,
                Alphabet_X=np.arange(n_uniq),
                estimator=estimator,
            ).item() for msg_attr, ctx, n_uniq
            in zip(msg_attr_dim, ctx_dim, n_unique_tuples_dim)
        ]  # H(messages, attribute_i | attributes_!=i)
        cond_mi_msg_attr_dim = [
            drv.information_mutual_conditional(
                msg, attr, ctx,
                Alphabet_X=alphabet_msg,
                Alphabet_Y=np.arange(n_uniq_msg_attr),
                estimator=estimator,
            ).item() for attr, ctx, n_uniq_msg_attr
            in zip(attr_dim, ctx_dim, n_unique_tuples_dim)
        ]  # I(messages, attribute_i | attributes_!=i)
        cond_proficiency_msg_attr_dim = [
            cond_mi / cond_entropy
            for cond_mi, cond_entropy in zip(cond_mi_msg_attr_dim, cond_entropy_msg_attr_dim)
        ]
        output.update(
            mutual_info_msg_attr_dim=cond_mi_msg_attr_dim,
            proficiency_msg_attr_dim=cond_proficiency_msg_attr_dim,
        )

    return output


class MessageDataset(Dataset):
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
        dataset = MessageDataset(messages, receiver_inputs, labels, *val, lengths)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            drop_last=False)

        predictions = []
        for b_messages, b_inputs in dataloader:
            with torch.no_grad():
                outputs = receiver(b_messages, b_inputs)

            b_lengths = find_lengths(b_messages.argmax(-1))
            idx = (torch.arange(len(b_messages)), b_lengths - 1)
            predictions.extend(outputs[idx].argmax(-1))

        predictions = torch.stack(predictions)
        output[f'accuracy_{key}'] = torch.mean(
            (predictions == dataset.labels).float()
        ).item()

    return output


def compute_topsim(
    meanings: torch.Tensor,
    messages: torch.Tensor,
    meaning_distance: str = 'hamming',
    message_distance: str = 'levenshtein',
    normalize: bool = False,
) -> float:
    """
    Computes topographic similarity.
    Optionally uses Damerau-Levenshtein edit distance for messages.
    """
    assert meaning_distance in ('cosine', 'hamming', 'euclidean')
    assert message_distance in ('levenshtein', 'damerau_levenshtein')

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

    meth = attrgetter('normalized_distance' if normalize else 'distance')
    distances = {
        'cosine': cosine_distance,
        'hamming': lambda x: (F.pdist(x.double(), p=0) / len(meanings)).cpu().numpy(),
        'euclidean': lambda x: F.pdist(x.double(), p=2).cpu().numpy(),
        'levenshtein': meth(Levenshtein),
        'damerau_levenshtein': meth(DamerauLevenshtein),
    }

    meaning_dist_fn = distances[meaning_distance]
    message_dist_fn = distances[message_distance]

    meaning_dists = meaning_dist_fn(meanings)
    message_dists = [
        message_dist_fn(messages[i], messages[j])
        for i, j in combinations(range(len(messages)), 2)
    ]

    topsim = spearmanr(meaning_dists, message_dists).statistic
    return topsim if not np.isnan(topsim) else 0.
