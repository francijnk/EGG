import math
import torch
import numpy as np
from egg.core.util import find_lengths
from scipy.optimize import minimize_scalar
from pyitlib.discrete_random_variable import entropy_joint
from collections import defaultdict
import random  # for testing only


def binary_entropy(p):
    if p == 0. or p == 1.:
        return 0
    return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)


def sequence_entropy(messages, vocab_size, length=None):
    """
    Computes entropy of the messages, where each symbol is treated as a
    distinct random variable. The entropy is approximated from the sample of
    messages using the James-Stein formula. If the length parameter is
    specified, the function returns entropy assuming that each message has the
    requested length (excluding EOS).
    """

    if length is not None:
        # compute entropy assuming the message length provided (excluding EOS)
        alphabet = (
            [[-1] + [i + 1 for i in range(vocab_size - 1)]] * length
            + [[0] + [-1] * (vocab_size - 1)] * (messages.size(1) - length))
        entropy = entropy_joint(
            messages.t().numpy(), estimator='JAMES-STEIN', Alphabet_X=alphabet)

    else:
        # handle sequences of any permissible length
        max_len = find_lengths(messages).max() - 1
        alphabet = (
            [[i for i in range(vocab_size)]] * max_len
            + [[0] + [-1] * (vocab_size - 1)] * (messages.size(1) - max_len))
        entropy = entropy_joint(
            messages.t().numpy(), estimator='JAMES-STEIN', Alphabet_X=alphabet)

    return entropy


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
        if channel == 'erasure' and error_prob > 0:
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


def compute_redundancy_smb(messages, max_len, vocab_size, channel, error_prob, maxiter=1000):
    """
    Computes a redundancy based on the symbol-level message entropy.
    The value returned is multiplied by a factor dependent on the maximum
    message length, so that the range of possible values is [0, 1].
    """
    if not isinstance(messages, torch.Tensor):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

    if channel == 'erasure':
        vocab_size += 1

    H = sequence_entropy(messages, vocab_size)
    H_max, _ = maximize_sequence_entropy(max_len, vocab_size, channel, error_prob, maxiter)
    H_max = max(H_max, H)  # the value of H is biased, and could exceed H_max
    return 1 - H / H_max


def compute_redundancy_smb_adjusted(messages, max_len, vocab_size, channel, error_prob, maxiter=1000):
    """
    Computes a redundancy based on the symbol-level message entropy, adjusted
    not to depend on message length and to have values in range [0, 1].
    """

    if not isinstance(messages, torch.Tensor):
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

    lengths = find_lengths(messages) - 1
    n = messages.size(0)

    len_probs_msg = defaultdict(int)
    len_probs_msg.update({
        l.item(): (lengths == l).to(torch.int).sum().item() / n
        for l in torch.unique(lengths)})

    # compute the maximum entropies for the erasure channel
    if channel == 'erasure' and error_prob > 0.:
        _, eos_probs = maximize_sequence_entropy(
            max_len, vocab_size, channel, error_prob, maxiter)
        max_entropies = [0]
        for eos_prob in eos_probs[::-1]:
            max_entropy = (
                binary_entropy(error_prob)
                + error_prob * max_entropies[0]
                + (
                    binary_entropy(eos_prob)
                    # + eos_prob * 0
                    + (1 - eos_prob) * (
                        math.log(vocab_size - 1, 2)
                        + max_entropies[0]
                    )
                )
            )
            max_entropies = [max_entropy] + max_entropies
        max_entropies = max_entropies[:-1][::-1]

    # compute redundancy
    redundancy = len_probs_msg[0]
    for i in range(1, max_len + 1):
        _messages = messages[(lengths == i), ...]

        if _messages.size(0) == 0:
            continue

        if channel == 'deletion':
            ent_msg = sequence_entropy(_messages, vocab_size)
        elif channel == 'erasure':
            ent_msg = sequence_entropy(_messages, vocab_size + 1, i)
        else:
            ent_msg = sequence_entropy(_messages, vocab_size, i)

        if channel == 'erasure' and error_prob != 0.:
            ent_max = max_entropies[i - 1]
        else:
            ent_max = math.log(vocab_size - 1, 2) * i

        # due to the bias of entropy estimators, the value of ent_msg could
        # exceed the value of the maximum entropy value
        ent_max = max(ent_max, ent_msg)

        prob_msg = len_probs_msg[i]
        redundancy_msg = 1 - ent_msg / ent_max

        if redundancy:
            redundancy += prob_msg * redundancy_msg
        else:
            redundancy = prob_msg * redundancy_msg

    return redundancy if redundancy is not None else 1.


# for testing - remove later or change into proper tests
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


# messages = generate_messages(1000, 5, 2, repeat_prob=1., var_len=True)
# print(compute_redundancy_smb(messages, 10, 9, None, 0.0))
# print(compute_redundancy_smb_adjusted(messages, 10, 9, None, 0.0))
