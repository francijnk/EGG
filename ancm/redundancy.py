import random
import torch
import math
from egg.zoo.objects_game.util import mutual_info, entropy
from scipy.optimize import minimize_scalar

import json

erasure_prob = 0.15
length = 5

with open(f'runs/erasure_pr_{erasure_prob}/{length}_4-results.json') as fp:
    data = json.load(fp)
    messages = data['messages']

try:
    messages = [x['message_no_noise'] for x in messages]
except:
    messages = [x['message'] for x in messages]
messages = [torch.Tensor([int(smb) for smb in x.split(',')]) for x in messages]

messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)
print(messages[0])

actual_vs = torch.unique(torch.flatten(messages)).size(0)
if erasure_prob != 0.:
    actual_vs = actual_vs - 1
print('actual vs:', actual_vs)
actual_vs = 31

messages = []
message = []
while len(messages) < 10000:
    for _ in range(length):
        smb = random.randint(1, 30)
        message.append(smb)
    messages.append(torch.Tensor(message))
    message = []
messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)


def sequence_entropy(messages):
    # directly compute entropy for a sample of single symbols
    if messages.size(1) == 1:
        return entropy(torch.squeeze(messages, dim=-1))  # entropy(torch.squeeze(messages))

    unique_first_smb = torch.unique(messages[:, 0])

    # drop the 1st symbol if it's the same for all messages
    if unique_first_smb.size(0) == 1:
        return sequence_entropy(messages[:, 1:])

    H = entropy(messages[:,0])

    n = messages.size(0)
    for prefix_symbol in torch.unique(messages[:, 0]):
        prob = (messages[:, 0] == prefix_symbol).to(torch.int).sum().item() / n

        suffixes = messages[messages[:, 0] == prefix_symbol] 
        suffixes = suffixes[:, 1:]

        # print(suffixes.size())
        H += prob * sequence_entropy(suffixes)

    return H


def sequence_entropy2(messages, level=0):
    messages, length = pad_messages(messages)

    if length == 1:
        return entropy(messages)

    if len(set(m[0] for m in messages)) == 1:
        messages = [m[1:] for m in messages]
        return recursive_entropy2(messages, level=level+1)

    # otherwise, the message consists of at least 2 symbols and
    # the 1st symbol is not always the same

    # decompose the entropy for the 1st symbol of the 1st message
    symbol = messages[0][0]

    msg_dict = defaultdict(list)
    for msg in messages:
        if msg[0] == symbol:
            msg_dict[symbol].append(tuple(msg[1:]))  # drop the 1st symbol
        else:
            msg_dict[-1].append(msg)

    H = entropy(msg_dict.keys())
    n = len(messages)

    for key, msg_list in msg_dict.items():
        p = len(msg_list) / len(messages)
        subset_entropy = recursive_entropy2(msg_list, level + 1)
        if key >= 0:
            print('--' * level, key, p, subset_entropy, len(msg_list))
        H += p * subset_entropy  #(msg_list)  # recursive_entropy_2(msg_list, level + 1)

    return H


def binary_entropy(p):
    if p == 0. or p == 1.:
        return 0
    return -p * math.log(p, 2) - (1-p) * math.log(1-p, 2)


def maximum_sequence_entropy(vocab_size, max_len, erasure_prob=0.):
    if max_len == 1:
        if erasure_prob == 0.:
            print(math.log(vocab_size, 2))
            return math.log(vocab_size, 2)
        else:
            def _entropy(eos_prob):
                return (
                    binary_entropy(eos_prob)
                    + (1-eos_prob) * (
                        binary_entropy(erasure_prob)
                        # + erasure_prob * 0
                        + (1-erasure_prob) * math.log(vocab_size-1, 2)
                    )
                )
            optimal_eos_prob = minimize_scalar(lambda p: -1 * _entropy(p), method='bounded', bounds=(0., 1.), options={'maxiter': 1000})
            return _entropy(optimal_eos_prob.x)

    max_suffix_ent = maximum_sequence_entropy(vocab_size, max_len-1, erasure_prob)
    # print(max_suffix_ent, vocab_size, max_len)

    def _sequence_entropy(eos_prob):
        return (
            binary_entropy(eos_prob) 
            # + eos_prob * 0 = eos_prob * h(suffix | eos)
            + (1-eos_prob) * (
                binary_entropy(erasure_prob)
                + erasure_prob * max_suffix_ent
                + (1-erasure_prob) * (math.log(vocab_size-1, 2) + max_suffix_ent)
            )
        )

    optimal_eos_prob = minimize_scalar(lambda p: -1 * _sequence_entropy(p), method='bounded', bounds=(0., 1.), options={'maxiter': 1000})
    if optimal_eos_prob.x < 5.97e-06:
        optimal_eos_prob.x = 0
    # print(optimal_eos_prob.x, optimal_eos_prob.success, optimal_eos_prob.status, optimal_eos_prob.message)
   
    return _sequence_entropy(optimal_eos_prob.x)
    # H = binary_entropy(erasure_prob) + erasure_prob * max_suffix_ent + (1-erasure_prob) *  

# print(messages[:100])

def compute_redundancy(messages, vocab_size, max_len, erasure_prob):
    return 1 - sequence_entropy(messages) / maximum_sequence_entropy(vocab_size, max_len, erasure_prob)

print('no noise')
#print('<>', maximum_sequence_entropy(2, 2, 0.))
print('seq ent', sequence_entropy(messages))
print('max ent', maximum_sequence_entropy(actual_vs, length, 0.0))
print('redund', compute_redundancy(messages, actual_vs, length, 0))

print('\n\nwith noise')
messages = [x['message'] for x in data['messages']]
messages = [torch.Tensor([int(smb) for smb in x.split(',')]) for x in messages]

messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)
print('seq ent', sequence_entropy(messages))
print('max ent', maximum_sequence_entropy(actual_vs, length, erasure_prob))
print('redund', compute_redundancy(messages, actual_vs, length, erasure_prob))
