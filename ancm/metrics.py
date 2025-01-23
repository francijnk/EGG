import math
import torch
import numpy as np
from egg.core.util import find_lengths
from scipy.optimize import minimize_scalar
from pyitlib.discrete_random_variable import entropy, entropy_joint
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance  # , ratio
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

from typing import List, Optional, Union


# Entropy, Mutual information
def binary_entropy(p: float):
    if p == 0. or p == 1.:
        return 0.
    return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)


def tensor_entropy(tensor: torch.Tensor):
    """
    Computes entropy using a James-Stein estimator.
    If the tensor has more than one dimension, each tensors indexed by the 1st
    dimension is treated as one of the elements to apply the operation upon.
    """
    if tensor.dim() == 2:
        values = torch.empty(len(tensor), dtype=torch.int)
        for i, val in enumerate(torch.unique(tensor, dim=0)):
            for j in range(len(tensor)):
                if torch.equal(tensor[j], val):
                    values[j] = i
                    continue
    else:
        values = tensor

    alphabet = np.unique(values.numpy())
    # print(alphabet.shape)
    # print(values.t().numpy().shape)
    H = entropy(values.t().numpy(), estimator='JAMES-STEIN', Alphabet_X=alphabet)
    return H.item()


def sequence_entropy(
        sequences: Union[torch.Tensor, List[torch.Tensor]],
        vocab_size: Optional[int] = None,
        length: Optional[int] = None,
        alphabet: Optional[List[int]] = None):
    """
    Computes entropy of the sequences, where each symbol is treated as a
    distinct random variable. The entropy is approximated from the sample of
    using the James-Stein formula. If any of the vocab_size, length or alphabet
    parameters is specified, the function assumes that the sequences are
    messages. In case length is provided, it is assumed that all messages have
    the requested length (excluding EOS).
    """
    # handle arbitrary sequences
    if vocab_size is None and alphabet is None:
        alphabet = torch.nn.utils.rnn.pad_sequence([
            torch.unique(sequences[..., i])
            for i in range(sequences.shape[-1])], padding_value=-1).t()
        return entropy_joint(
            sequences.t().numpy(), estimator='JAMES-STEIN', Alphabet_X=alphabet.numpy())

    # handle messages
    if alphabet is None:
        non_eos_alphabet = [i + 1 for i in range(vocab_size - 1)]
    else:
        non_eos_alphabet = [s for s in alphabet if s > 0] + [-1] * (vocab_size - len(alphabet) - 1)

    # compute entropy assuming the message length provided (excluding EOS)
    if length is not None:
        alphabet_smb = (
            [[-1] + non_eos_alphabet] * length
            + [[0] + [-1] * len(non_eos_alphabet)] * (sequences.size(1) - length))
        entropy = entropy_joint(
            sequences.t().numpy(), estimator='JAMES-STEIN', Alphabet_X=alphabet_smb)

    # handle messages of any permissible length
    else:
        max_len = find_lengths(sequences).max() - 1
        alphabet_smb = (
            [[0] + non_eos_alphabet] * max_len
            + [[0] + [-1] * (vocab_size - 1)] * (sequences.size(1) - max_len))
        entropy = entropy_joint(
            sequences.t().numpy(), estimator='JAMES-STEIN', Alphabet_X=alphabet_smb)

    return entropy


def mutual_info(x: torch.Tensor, y: torch.Tensor):
    """
    Given a two tensors of equal length, representing two random variables,
    computes mutual information between them. In case x is two dimensional, its
    entropy will be computed as joint entropy of the 2nd dimension, and the
    value returned will be an estimate of I(X1, X2, ..., Xm; Y).

    Otherwise, if x is 1 dimensional, an estimate of I(X; Y) is returned.

    In case a multi-dimensional tensor is passed as y, all dimensions except
    for the first one will be categorized.
    """
    assert len(x) == len(y)

    x = x if x.dim() > 1 else x.unsqueeze(dim=1)
    y = y if y.dim() > 1 else y.unsqueeze(dim=1)
    xy = torch.cat([x, y], dim=-1)

    H_x = sequence_entropy(x) if x.dim() == 2 else tensor_entropy(x)
    H_y = tensor_entropy(y)
    H_xy = sequence_entropy(xy)

    mi = (H_x + H_y - H_xy).item()
    mi = max(0., min(1., mi))  # estimated entropy is biased and in come cases could result in negative MI
    return mi


def compute_mi_input_msgs(sender_inputs: List[torch.Tensor], messages: List[torch.Tensor]):
    """
    Computes multiple information-theoretic metrics: message entropy, input entropy,
    mutual information between messages and target objects, entropy of each input
    dimension and mutual information between each input dimension and messages.
    """
    # transform messages to a tensor, append 0 to each message
    messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)
    messages = torch.cat([messages, torch.zeros_like(messages[:, :1])], dim=-1)

    # for the context game, only consider target objects
    sender_inputs = torch.stack(sender_inputs)
    target_objs = sender_inputs \
        if sender_inputs.dim() == 2 \
        else sender_inputs[:, 0]

    to_values = torch.empty(len(target_objs), dtype=torch.int)
    for i, val in enumerate(torch.unique(target_objs, dim=0)):
        for j in range(len(target_objs)):
            if torch.equal(target_objs[j], val):
                to_values[j] = i
                continue

    return {
        'entropy_msg': sequence_entropy(messages),
        'entropy_inp': tensor_entropy(to_values),
        'mi_msg_inp': mutual_info(messages, to_values),
        'entropy_inp_dim': [
            tensor_entropy(target_objs[:, i])
            for i in range(target_objs.size(-1))],
        'mi_msg_inp_dim': [
            round(mutual_info(messages, target_objs[:, i]), 4)
            for i in range(target_objs.size(-1))],
    }


def compute_conceptual_alignment(
        dataloader: torch.utils.data.DataLoader,
        sender: torch.nn.Module,
        receiver: torch.nn.Module,
        device: torch.device,
        bs: int):
    """
    Computes speaker-listener alignment.
    """
    all_features = dataloader.dataset.obj_sets
    targets = dataloader.dataset.labels
    obj_features = np.unique(all_features[:, targets[0], :], axis=0)
    obj_features = torch.tensor(obj_features, dtype=torch.float).to(device)

    n_batches = math.ceil(obj_features.size()[0] / bs)
    sender_embeddings, receiver_embeddings = None, None

    for batch in [obj_features[bs * y:bs * (y + 1), :] for y in range(n_batches)]:
        with torch.no_grad():
            b_sender_embeddings = sender.fc1(batch).tanh().cpu().numpy()
            b_receiver_embeddings = receiver.fc1(batch).tanh().cpu().numpy()
            if sender_embeddings is None:
                sender_embeddings = b_sender_embeddings
                receiver_embeddings = b_receiver_embeddings
            else:
                sender_embeddings = np.concatenate((sender_embeddings, b_sender_embeddings))
                receiver_embeddings = np.concatenate((receiver_embeddings, b_receiver_embeddings))

    sender_sims = cosine_similarity(sender_embeddings)
    receiver_sims = cosine_similarity(receiver_embeddings)
    r = pearsonr(sender_sims.ravel(), receiver_sims.ravel())
    return r.statistic


# Redundancy
def compute_max_rep(messages: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Computes the number of occurrences of the most frequent symbol in each
    message (0 for messages that consist of EOS symbols only).
    """

    if isinstance(messages, list):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

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


def compute_redundancy_msg(messages: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Computes redundancy at the message level.
    """
    if isinstance(messages, list):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)
    H = tensor_entropy(messages)
    H_max = math.log(len(messages), 2)
    return 1 - H / H_max


def maximize_sequence_entropy(max_len, vocab_size, channel=None, error_prob=None, maxiter=5000):
    """
    Redursively approximates the highest achievable entropy for a given maximum
    length (excluding EOS) and vocabulary size (including EOS).

    Returns the optimized entropy value and a list of probabilities of
    generating the EOS symbol at each position (provided that no EOS symbol was
    generated yet.
    """

    if max_len == 1 and (channel is None or error_prob == 0. or channel == 'symmetric'):
        max_entropy = math.log(vocab_size, 2)
        eos_prob = 1. / vocab_size
        return max_entropy, [eos_prob]
    elif max_len == 1 and channel == 'erasure':

        def _entropy(eos_prob):
            erased_prob = (1 - eos_prob) * error_prob
            if vocab_size == 1:
                return 0
            elif vocab_size == 2:
                return (
                    - eos_prob * math.log(eos_prob, 2)
                    - erased_prob * math.log(erased_prob, 2)
                    - (1 - eos_prob) * (1 - erased_prob) * (
                        math.log(1 - eos_prob, 2)
                        + math.log(1 - erased_prob, 2)
                    )
                )
            else:
                return (
                    - eos_prob * math.log(eos_prob, 2)
                    - erased_prob * math.log(erased_prob, 2)
                    - (1 - eos_prob) * (1 - erased_prob) * (
                        math.log(1 - eos_prob, 2)
                        + math.log(1 - erased_prob, 2)
                        - math.log(vocab_size - 1, 2)
                    )
                )

        optimal_eos_prob = minimize_scalar(
            lambda p: -1 * _entropy(p),
            method='bounded', bounds=(0., 1.),
            options={'maxiter': maxiter})
        return _entropy(optimal_eos_prob.x), [optimal_eos_prob.x]

    elif max_len == 1 and channel == 'deletion':
        if error_prob == 1.:
            return 0, [1.]

        def _entropy(eos_prob):
            if vocab_size == 1:
                return 0
            elif vocab_size == 2:
                return (
                    - (eos_prob + error_prob) * math.log(eos_prob + error_prob, 2)
                    - (1 - eos_prob - error_prob) * (
                        math.log(1 - eos_prob - error_prob, 2)
                    )
                )
            else:
                return (
                    - (eos_prob + error_prob) * math.log(eos_prob + error_prob, 2)
                    - (1 - eos_prob - error_prob) * (
                        math.log(1 - eos_prob - error_prob, 2)
                        - math.log(vocab_size - 1, 2)
                    )
                )

        optimal_eos_prob = minimize_scalar(
            lambda p: -1 * _entropy(p),
            method='bounded', bounds=(0., 1.),
            options={'maxiter': maxiter})
        return _entropy(optimal_eos_prob.x), [optimal_eos_prob.x]

    max_suffix_entropy, eos_probs = maximize_sequence_entropy(
        max_len - 1, vocab_size, channel, error_prob, maxiter)

    def _sequence_entropy(eos_prob):
        if vocab_size == 1:
            return 0
        elif channel == 'erasure' and error_prob > 0:
            return (
                binary_entropy(eos_prob)
                # + eos_prob * 0
                + (1 - eos_prob) * (
                    binary_entropy(error_prob)
                    + error_prob * max_suffix_entropy
                    + (1 - error_prob) * (
                        math.log(vocab_size - 1, 2)
                        + max_suffix_entropy
                    )
                )
            )

        elif channel == 'deletion' and error_prob > 0:
            entropy_nodel = (
                binary_entropy(eos_prob)
                # + eos_prob * 0
                + (1 - eos_prob) * (
                    math.log(vocab_size - 1, 2)
                    + max_suffix_entropy
                )
            )
            entropy_del = max_suffix_entropy
            return (
                binary_entropy(error_prob)
                + error_prob * entropy_del
                + (1 - error_prob) * entropy_nodel
            )

        else:
            # no channel, error_prob == 0, or channel == 'symmetric'
            return (
                binary_entropy(eos_prob)
                # + eos_prob * 0
                + (1 - eos_prob) * (
                    math.log(vocab_size - 1, 2)
                    + max_suffix_entropy
                )
            )

    optimal_eos_prob = minimize_scalar(
        lambda p: -1 * _sequence_entropy(p),
        method='bounded', bounds=(0., 1.),
        options={'maxiter': maxiter})
    eos_probs = [optimal_eos_prob.x] + eos_probs

    return _sequence_entropy(optimal_eos_prob.x), eos_probs


def compute_redundancy_smb(messages, max_len, vocab_size, channel, error_prob, alphabet=None, maxiter=1000):
    """
    Computes a redundancy based on the symbol-level message entropy.
    The value returned is multiplied by a factor dependent on the maximum
    message length, so that the range of possible values is [0, 1].
    """
    if vocab_size == 1 or (alphabet is not None and len(alphabet) == 1):
        return 1.

    if not isinstance(messages, torch.Tensor):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)
        messages = torch.cat([messages, torch.zeros_like(messages[:, 0]).unsqueeze(-1)], dim=-1)

    if channel == 'erasure' and error_prob > 0.:
        vocab_size += 1
        if alphabet is not None:
            alphabet = list(alphabet)
            if vocab_size not in alphabet:
                alphabet.append(vocab_size)

    if alphabet is not None:
        vocab_size = len(alphabet)

    H = sequence_entropy(messages, vocab_size, alphabet=alphabet)
    H_max, _ = maximize_sequence_entropy(max_len, vocab_size, channel, error_prob, maxiter)
    H_max = max(H_max, H)  # the value of H is biased, and could exceed H_max in some cases
    return 1 - H / H_max


def compute_redundancy_smb_adjusted(messages, channel, error_prob, alphabet, erased_symbol=None, maxiter=1000):
    """
    Computes a redundancy based on the symbol-level message entropy, adjusted
    not to depend on message length.
    """

    if not isinstance(messages, torch.Tensor):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

    lengths = find_lengths(messages) - 1
    max_len = lengths.max().item()

    len_probs = defaultdict(float)
    len_probs.update({
        l.item(): (lengths == l).int().sum().item() / messages.size(0)
        for l in torch.unique(lengths)})

    vocab_size = len(alphabet)
    if erased_symbol is not None and channel == 'erasure' and error_prob > 0. \
            and erased_symbol not in alphabet:
        vocab_size += 1  # make sure erased_symbol is included in vocab_size

    # compute the maximum entropies for the erasure channel
    if channel == 'erasure' and error_prob > 0.:
        _, eos_probs = maximize_sequence_entropy(
            max_len, vocab_size - 1, channel, error_prob, maxiter)
        max_entropies = [0]
        for eos_prob in eos_probs[::-1]:
            max_entropy = (
                binary_entropy(error_prob)
                + error_prob * max_entropies[-1]
                + (
                    binary_entropy(eos_prob)
                    # + eos_prob * 0
                    + (1 - eos_prob) * (
                        math.log(vocab_size - 2, 2)
                        + max_entropies[-1]
                    )
                )
            )
            max_entropies.append(max_entropy)
        max_entropies = max_entropies[1:]

    # compute redundancy
    H_msg = H_max = sum(-p * math.log(p, 2) for p in len_probs.values())
    for i in range(1, max_len + 1):
        _messages = messages[(lengths == i), ...]
        if _messages.size(0) == 0:
            continue

        ent_msg = sequence_entropy(_messages, vocab_size, i, alphabet)

        if channel == 'erasure' and error_prob > 0.:
            ent_max = max_entropies[i - 1]
        else:
            ent_max = math.log(vocab_size - 1, 2) * i if vocab_size > 2 else 0.

        # due to the bias of entropy estimators, the value of ent_msg could
        # exceed the value of the maximum entropy value
        ent_max = max(ent_max, ent_msg)

        H_msg += len_probs[i] * ent_msg
        H_max += len_probs[i] * ent_max

    return 1. - H_msg / H_max if H_max != 0. else 1.


# Compositionality
def compute_top_sim(
        sender_inputs: Union[torch.Tensor, List[torch.Tensor]],
        messages: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Computes topographic rho.
    """

    sender_inputs = torch.stack(sender_inputs) \
        if isinstance(sender_inputs, list) else sender_inputs

    # handling the compare variant
    if sender_inputs.dim() == 4:
        n_samples = sender_inputs.size(0)
        sender_inputs = sender_inputs.reshape(n_samples, -1)

    messages = [
        [s.int().item() for s in msg if s > 0] + [0]
        for msg in messages]

    # pairwise cosine similarity between object vectors
    cos_sims = cosine_similarity(sender_inputs)

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
        sender_inputs: Union[torch.Tensor, List[torch.Tensor]],
        messages: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Computes PosDis.
    """
    if isinstance(messages, list):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

    attributes = torch.stack(sender_inputs) \
        if isinstance(sender_inputs, list) else sender_inputs

    gaps = torch.zeros(messages.size(1))
    non_constant_positions = 0.0
    for j in range(messages.size(1)):
        symbol_mi = []
        H_j = None
        for i in range(attributes.size(1)):
            x, y = messages[:, j], attributes[:, i]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if H_j is None:
                H_j = tensor_entropy(y)

        symbol_mi.sort(reverse=True)

        if H_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / H_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def histogram(
        messages: Union[torch.Tensor, List[torch.Tensor]],
        vocab_size: int):
    if isinstance(messages, list):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

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
        sender_inputs: Union[torch.Tensor, List[torch.Tensor]],
        messages: Union[torch.Tensor, List[torch.Tensor]],
        vocab_size: int):
    """
    Computes BosDis.
    """
    histograms = histogram(messages, vocab_size)
    return compute_posdis(sender_inputs, histograms[:, 1:])


##########################################################
# for testing - remove later or change into proper tests #
##########################################################

import random


def generate_messages(n, max_len, vocab_size, repeat_prob, var_len=False):
    if var_len:
        lengths = {
            i: 1. / len(list(range(max_len // 2, max_len + 1)))
            for i in range(max_len // 2, max_len + 1)
        }
    else:
        lengths = {max_len: 1.}

    messages = []
    for i in range(n):
        symbols = []
        msg_len = np.random.choice(
            np.arange(
                min(list(lengths.keys())),
                max(list(lengths.keys())) + 1),
            p=list(lengths.values()))
        for j in range(msg_len):
            if j == 0:
                smb = random.randint(1, vocab_size - 1)
            else:
                repeat = random.random() < repeat_prob
                if repeat:
                    smb = symbols[-1]
                else:
                    options = np.array(
                        [i for i in range(1, vocab_size) if i != symbols[-1]])
                    probs = [1. / len(options) for _ in range(len(options))]
                    smb = np.random.choice(options, p=probs)
            symbols.append(smb)
        symbols.append(0)
        msg = torch.tensor(symbols)
        messages.append(msg)
    messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

    return messages


#messages = generate_messages(1000, 10, 3, repeat_prob=1., var_len=False)
#alphabet = torch.unique(torch.flatten(messages), dim=0)
# print(alphabet)
#print(compute_redundancy_smb(messages, 10, 3, None, 0.0))
#print(compute_redundancy_smb_adjusted(messages, 10, 3, None, 0.0))
# print(compute_redundancy_smb(messages, 10, 8, None, 0.0, alphabet=alphabet))
# print(compute_redundancy_smb_adjusted(messages, 10, 8, None, 0.0, alphabet=alphabet))
# print(compute_redundancy_smb(messages, 10, 8, None, 0.0, alphabet=None))
# print(compute_redundancy_smb_adjusted(messages, 10, 8, None, 0.0, alphabet=None))
