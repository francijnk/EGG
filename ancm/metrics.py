from __future__ import annotations

import math
import torch
import numpy as np
from egg.core.util import find_lengths
from scipy.optimize import minimize_scalar
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance  # , ratio
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from torch.utils.data import Dataset
from itertools import combinations
from pyitlib.discrete_random_variable import entropy, entropy_joint

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


def tensor_entropy(
        x: torch.Tensor,
        alphabet: Optional[torch.Tensor] = None,
        estimator: str = 'JAMES-STEIN'):
    """
    Estimates entropy of the RV X represented by the tensor x.

    If the tensor has more than one dimension, each tensors indexed by the 1st
    dimension is treated as one of the elements to apply the operation upon.
    """
    if x.dim() == 0:
        return 0.
    elif x.dim() > 1:
        _, x = torch.unique(x, return_inverse=True, dim=0)

    if alphabet is not None:
        alphabet = alphabet.numpy()

    H = entropy(x.numpy(), Alphabet_X=alphabet, estimator=estimator)

    return H.item()


def build_alphabet(
        x: Optional[torch.Tensor] = None,
        non_message_sequences: bool = False,
        symbols: Optional[Iterable[int]] = None,
        vocab_size: Optional[int] = None,
        length: [int] = None):
    """
    Builds a relevant alphabet for the tensor x.
    In case length is specified, it is assumed that all messages have the
    provided length (excluding EOS).
    """
    assert not non_message_sequences or length is None

    # handle arbitrary sequences: identify unique symbols per each dimension
    if non_message_sequences:
        all_symbols = torch.unique(x).unsqueeze(0)
        alphabet = all_symbols.expand(x.size(1), -1)
        for i in range(x.size(1)):
            unique_symbols = torch.unique(x[:, i])
            mask = torch.isin(alphabet[:, i], unique_symbols)
            mask = torch.logical_not(mask)
            alphabet[mask, i] = -1

        return alphabet#.to(torch.float)

    # handle messages
    if symbols is not None:
        non_eos_symbols = [s for s in symbols if s > 0]
    elif vocab_size is not None:
        non_eos_symbols = [i + 1 for i in range(vocab_size - 1)]
    else:
        non_eos_symbols = [s for s in torch.unique(x) if s > 0]

    # compute entropy assuming the message length provided (excluding EOS)
    if length is not None:
        alphabet = (
            [[-1] + non_eos_symbols] * length
            + ([[0] + [-1] * len(non_eos_symbols)]
               * (x.size(1) - length)))
    else:  # handle messages of any permissible length <= max actual length
        max_len = find_lengths(x).max() - 1
        alphabet = (
            [[0] + non_eos_symbols] * max_len
            + [[0] + [-1] * len(non_eos_symbols)] * (x.size(1) - max_len))

    return torch.tensor(alphabet)#.to(torch.float)


def sequence_entropy(
        x: torch.Tensor,
        alphabet: Optional[torch.Tensor] = None,
        estimator: str = 'JAMES-STEIN') -> float:
    """
    Estimates the entropy of the RV X represented by the tensor x, assuming
    that X if a compound RV if x has 2 dimensions, i.e. X = X1, ..., Xm.
    The entropy is approximated from the sample using the James-Stein formula.

    """
    if x.dim() == 1:
        return tensor_entropy(x)

    if alphabet is not None:
        alphabet = alphabet.numpy()

    try:
        H = entropy_joint(
            x.transpose(0, -1).numpy(),
            Alphabet_X=alphabet,
            estimator=estimator)
    except ValueError:
        print(x)
        print(alphabet)
        print(x.shape, alphabet.shape)
        return 0.

    return H.item()


def mutual_info(
        x: torch.Tensor,
        y: torch.Tensor,
        alphabet_x: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """
    Given a two tensors x, y of equal length, representing realizations of RVs
    X and Y, estimates I(X; Y) using the James-Stein estimator by approximating
    H(X), H(Y) and H(X, Y). Returns a tuple of I(X; Y), H(X, Y).

    If the first tensor has more than one dimension, i.e. it represents a
    compound RV (X = X1, ..., Xn), computations are based on the joint entropy
    of X1, ..., Xn and the function approximates I(X1, ..., Xn; Y).

    If Y has multiple dimensions, its values are first categorized along the
    first axis.
    """
    assert len(x) == len(y), "x and y must be of equal length"
    assert len(x) == len(y.view(-1)), "y may only represent a single RV"

    # single_dim_x = len(x) == len(x.view(-1))

    x = x if x.dim() == 2 else x.unsqueeze(-1)
    y = y if y.dim() == 2 else y.unsqueeze(-1)
    xy = torch.cat([x, y], dim=-1)

    alphabet_x = build_alphabet(x) if alphabet_x is None else alphabet_x
    alphabet_y = build_alphabet(y, True)

    # all_symbols = torch.unique(
    #     torch.cat([torch.unique(alphabet_x), torch.unique(alphabet_y)]))
    # symbols_x = all_symbols.expand(alphabet_x.size(0), -1)
    # symbols_y = all_symbols.clone()

    # for i in range(symbols_x.size(0)):
    #     mask = torch.isin(all_symbols, alphabet_x[i])
    #     mask = torch.logical_not(mask)
    #     symbols_x[i, mask] = -1
    # mask = torch.isin(symbols_y, alphabet_y)
    # mask = torch.logical_not(mask)
    # symbols_y[mask] = -1
    # alphabet_xy = torch.cat([symbols_x, symbols_y.unsqueeze(0)])

    # ensure symbol sets are disjoint
    alphabet_y += alphabet_x.max() + 1.
    y += alphabet_x.max() + 1.
    xy[:, -1] += alphabet_x.max() + 1.

    # pad both alphabets with the fill value
    padded_alphabet_x = torch.cat([
        alphabet_x,
        torch.ones(alphabet_x.size(0), alphabet_y.size(1)) * -1], dim=1)
    padded_alphabet_y = torch.cat([
        torch.ones(alphabet_y.size(0), alphabet_x.size(1)) * -1,
        alphabet_y], dim=1)
    alphabet_xy = torch.cat((padded_alphabet_x, padded_alphabet_y))

    H_x = sequence_entropy(x, alphabet_x)
    H_y = tensor_entropy(y)
    H_xy = sequence_entropy(xy, alphabet_xy)
    I_xy = H_x + H_y - H_xy

    return I_xy, H_xy


def compute_mi(messages: torch.Tensor, attributes: torch.Tensor) -> dict:
    """
    Computes multiple information-theoretic metrics: message entropy, input entropy,
    mutual information between messages and target objects, entropy of each input
    dimension and mutual information between each input dimension and messages.
    In case x has two dimensions, the function assumes it represents a
    compound RV, i.e. a sequence of realizations x1, ..., xm of RVs
    X1, ..., Xm, and returns an estimate of I(X1, ..., Xm; Y)

    Alphabet - applied to the message
    """

    alphabet = build_alphabet(messages)

    if attributes.size(1) == 1:
        entropy_msg = sequence_entropy(messages, alphabet)
        entropy_attr = tensor_entropy(attributes)
        mi_msg_attr, entropy_msg_attr = mutual_info(messages, attributes, alphabet)
        vi_msg_attr = 2 * entropy_msg_attr - entropy_msg - entropy_attr
        vi_norm_msg_attr = 1. - mi_msg_attr / entropy_msg_attr

        return {
            'entropy_msg': entropy_msg,
            'entropy_attr': entropy_attr,
            'mi_msg_attr': mi_msg_attr,
            'vi_msg_attr': vi_msg_attr,
            'vi_norm_msg_attr': vi_norm_msg_attr,
            'is_msg_attr': 1 - vi_norm_msg_attr,
        }

    else:  # return values per attribute dimension instead
        entropy_msg = sequence_entropy(messages, alphabet)
        entropy_attr = sequence_entropy(attributes)
        entropy_attr_dim = [
            tensor_entropy(attributes[:, i])
            for i in range(attributes.size(-1))]
        mi_msg_attr_dim, entropy_msg_attr_dim = list(zip(*[
            mutual_info(messages, attributes[:, i], alphabet)
            for i in range(attributes.size(-1))]))
        vi_msg_attr_dim = [
            2 * entropy_msg_attr - entropy_msg - entropy_attr
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

        return {
            'entropy_msg': entropy_msg,
            'entropy_attr': entropy_attr,
            'entropy_attr_dim': entropy_attr_dim,
            'mi_msg_attr_dim': mi_msg_attr_dim,
            'vi_msg_attr_dim': vi_msg_attr_dim,
            'vi_norm_msg_attr_dim': vi_norm_msg_attr_dim,
            'is_msg_attr_dim': is_msg_attr_dim,
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

    n_batches = np.ceil(obj_features.size()[0] / bs)
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


def compute_redundancy_msg(messages: torch.Tensor) -> float:
    """
    Computes redundancy at the message level.
    """
    H = tensor_entropy(messages)
    H_max = np.log2(len(messages))
    return 1 - H / H_max


def maximize_sequence_entropy(
        max_len: int,
        vocab_size: int,
        channel: str = None,
        error_prob: float = None,
        maxiter: int = 5000):
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
        new = tensor[mask]
        new = new.to(torch.float)
        result.append(new)

    return result


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
        outputs = receiver(batched_messages, batched_inputs)
        if opts.mode == 'rf':
            outputs = outputs[0]
            predictions.append(outputs.detach().reshape(-1, 1))
        else:
            lengths = find_lengths(batched_messages.argmax(-1))
            for i in range(batched_messages.size(0)):
                outputs_i = outputs[i, lengths[i] - 1].argmax(-1)
                predictions.append(outputs_i.detach().reshape(-1, 1))

    predictions = torch.cat(predictions, dim=0)
    labels = torch.stack(labels)[:len(predictions)]

    return (predictions == labels).float().mean().item()


def compute_redundancy(
        messages: torch.Tensor,
        vocab_size: int,
        max_len: int,
        channel: Optional[str],
        error_prob: float,
        alphabet: Optional[torch.Tensor] = None,
        maxiter: int = 1000) -> float:
    """
    Computes a redundancy based on the symbol-level message entropy.
    The value returned is multiplied by a factor dependent on the maximum
    message length, so that the range of possible values is [0, 1].
    """
    if vocab_size == 1 or (alphabet is not None and len(alphabet) == 1):
        return 1.

    # if channel == 'erasure' and error_prob > 0.:
    #     vocab_size += 1
    #     if alphabet is not None:
    #         alphabet = list(alphabet)
    #         if vocab_size not in alphabet:
    #            alphabet.append(vocab_size)

    # if alphabet is not None:
    #     vocab_size = len(alphabet)
    alphabet = build_alphabet(messages)
    H = sequence_entropy(messages, alphabet)
    H_max, _ = maximize_sequence_entropy(max_len, vocab_size, channel, error_prob, maxiter)
    H_max = max(H_max, H)  # the value of H is biased, and could exceed H_max in some cases
    return 1 - H / H_max


def compute_adjusted_redundancy(
        messages: torch.Tensor,
        channel: Optional[str],
        error_prob: float,
        symbols: Optional[Iterable[float]] = None,
        erased_symbol: Optional[float] = None,
        maxiter: int = 1000) -> float:
    """
    Computes a redundancy based on the symbol-level message entropy, adjusted
    not to depend on message length.
    """

    lengths = find_lengths(messages) - 1
    max_len = lengths.max().item()

    len_probs = defaultdict(float)
    len_probs.update({
        l.item(): (lengths == l).int().sum().item() / messages.size(0)
        for l in torch.unique(lengths)})

    vocab_size = len(symbols)
    if erased_symbol is not None and channel == 'erasure' and error_prob > 0. \
            and erased_symbol not in symbols:
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
    H_msg = H_max = sum(-p * np.log2(p) for p in len_probs.values())
    for i in range(1, max_len + 1):
        _messages = messages[(lengths == i), ...]
        if _messages.size(0) == 0:
            continue

        alphabet = build_alphabet(_messages, symbols=symbols, length=i)
        ent_msg = sequence_entropy(_messages, alphabet)

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


def compute_posdis(sender_inputs: torch.Tensor, messages: torch.Tensor) -> float:
    """
    Computes PosDis.
    """

    gaps = torch.zeros(messages.size(1))
    non_constant_positions = 0.0
    for j in range(messages.size(1)):
        symbol_mi = []
        H_j = None
        for i in range(sender_inputs.size(1)):
            x, y = messages[:, j], sender_inputs[:, i]
            alphabet_x = build_alphabet(x.unsqueeze(1), True)
            info, _ = mutual_info(x, y, alphabet_x)
            symbol_mi.append(info)

            if H_j is None:
                H_j = tensor_entropy(y)

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


# messages = generate_messages(1000, 10, 3, repeat_prob=1., var_len=False)
# alphabet = torch.unique(torch.flatten(messages), dim=0)
# print(alphabet)
# print(compute_redundancy_smb(messages, 10, 3, None, 0.0))
# print(compute_redundancy_smb_adjusted(messages, 10, 3, None, 0.0))
# print(compute_redundancy_smb(messages, 10, 8, None, 0.0, alphabet=alphabet))
# print(compute_redundancy_smb_adjusted(messages, 10, 8, None, 0.0, alphabet=alphabet))
# print(compute_redundancy_smb(messages, 10, 8, None, 0.0, alphabet=None))
# print(compute_redundancy_smb_adjusted(messages, 10, 8, None, 0.0, alphabet=None))
