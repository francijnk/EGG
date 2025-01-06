import json
import torch
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from Levenshtein import distance
from scipy.stats import pearsonr, spearmanr

from ancm.archs import ErasureChannel

from typing import Optional
from collections import defaultdict

from egg.core.util import move_to
from egg.zoo.objects_game.util import mutual_info, entropy

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def compute_mi_input_msgs(sender_inputs, messages):
    num_dimensions = len(sender_inputs[0])
    each_dim = [[] for _ in range(num_dimensions)]
    result = []
    for i, _ in enumerate(each_dim):
        for vector in sender_inputs:
            each_dim[i].append(vector[i])  # only works for 1-D sender inputs

    for i, dim_list in enumerate(each_dim):
        result.append(round(mutual_info(messages, dim_list), 4))

    return {
        'entropy_msg': entropy(messages),
        'entropy_inp': entropy(sender_inputs),
        'mi': mutual_info(messages, sender_inputs),
        'entropy_inp_dim': [entropy(elem) for elem in each_dim],
        'mi_dim': result,
    }


def compute_alignment(dataloader, sender, receiver, device, bs):
    """
    Computes speaker-listener alignment.
    """
    all_features = dataloader.dataset.list_of_tuples
    targets = dataloader.dataset.target_idxs
    obj_features = np.unique(all_features[:,targets[0],:], axis=0)
    obj_features = torch.tensor(obj_features, dtype=torch.float).to(device)

    n_batches = math.ceil(obj_features.size()[0]/bs)
    sender_embeddings, receiver_embeddings = None, None

    for batch in [obj_features[bs*y:bs*(y+1),:] for y in range(n_batches)]:
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


def compute_max_rep(messages):
    """
    Computes the number of occurrences of the most frequent symbol in each
    message (0 for messages that consist of EOS symbols only).
    """
    if isinstance(messages, list):
        messages = [msg.argmax(dim=1) if msg.dim() == 2
                    else msg for msg in messages]
        messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)
        messages = torch.stack(messages)

    messages = messages.to(torch.float16)
    messages[messages == 0] = float('nan')
    mode = torch.mode(messages, dim=1).values
    is_rep = (messages == torch.unsqueeze(mode, dim=-1).expand(*messages.size()))
    max_rep = (is_rep.to(torch.int16).sum(dim=1))
    return max_rep


def compute_redundancy_smb(messages, max_len, vocab_size):
    """
    Computes redundancy at the symbol level, treating messages as sequences.
    """

    # To be able to calculate rendundancy
    n_possible_messages = 0
    vocab_size_without_eos = vocab_size - 1
    for i in range(max_len):
        n_possible_messages += vocab_size_without_eos**i

    max_entropy = math.log(n_possible_messages) / math.log(2)

    freq_table = []
    for i in range(max_len + 1): # +1 because of EOS
        freq_table.append({})

    # print(freq_table)
    messages = [msg.argmax(dim=1) if msg.dim() == 2
                else msg for msg in messages]
    for message in messages:
        # im not sure if it is a string or not/how the messages look rn
        message = ','.join([str(int(x)) for x in message.tolist()])
        string = ''
        # print(f' the message is {message} and the max_len is {max_len}')
        index = 0
        for symbol in message:
            if symbol != ',':
                string = string + symbol
            else:
                if string in freq_table[index].keys():
                    freq_table[index][string] += 1
                else:
                    freq_table[index][string] = 1
                index += 1
                string = ''
    entropies = []

    for message in messages:
        message = ','.join([str(int(x)) for x in message.tolist()])
        H = 0
        string = ''
        index = 0
        for symbol in message:
            if symbol != ',':
                string = string + symbol
            else:
                # rn probability is based on frequency of the partial string and frequency of all other options of the same length
                # prbably want to change this to have actual conditional probabilities
                p = freq_table[index][string]/sum(freq_table[index].values())
                H += p * math.log(1/p)
                index += 1
                string = ''
        H = H / math.log(2)

        entropies.append(H)

        redundancies = []
        for entropy in entropies:
            redundancy = 1 - entropy/max_entropy
            redundancies.append(redundancy)

    return redundancies


def compute_redundancy_msg(messages,  max_len):
    """
    Computes redundancy at the message level.
    """
    messages = [msg.argmax(dim=1) if msg.dim() == 2
                else msg for msg in messages]

    actual_entropy = entropy(messages)
    maximal_entropy = math.log(len(messages), 2)

    return 1 - actual_entropy / maximal_entropy


def compute_top_sim(sender_inputs, messages, dimensions=None):
    """
    Computes topographic rho.
    """
    obj_tensor = torch.stack(sender_inputs) \
        if isinstance(sender_inputs, list) else sender_inputs

    if dimensions is None:
        dimensions = []
        for d in range(obj_tensor.size(1)):
            dim = len(torch.unique(obj_tensor[:,d]))
            dimensions.append(dim)

    onehot = []
    for i, dim in enumerate(dimensions):
        if dim == 4:
            # one-hot encode categorical dimensions
            n1 = (np.logical_or(obj_tensor[:,i].int() == 1, obj_tensor[:,i].int() == 2)).int().reshape(obj_tensor.size(0), 1)
            n2 = (np.logical_or(obj_tensor[:,i].int() == 1, obj_tensor[:,i].int() == 3)).int().reshape(obj_tensor.size(0), 1)
            onehot.append(n1)
            onehot.append(n2)
        else:
            # binary dimensions need not be transformed
            onehot.append(obj_tensor[:,i:i+1])
    onehot = np.concatenate(onehot, axis=1)

    messages = [msg.argmax(dim=1).tolist() if msg.dim() == 2
                else msg.tolist() for msg in messages]

    # pairwise cosine similarity between object vectors
    cos_sims = cosine_similarity(onehot)

    # pairwise Levenshtein distance between messages
    lev_dists = np.ones((len(messages), len(messages)), dtype='int')
    for i, msg_i in enumerate(messages):
        for j, msg_j in enumerate(messages):
            if i > j:
                continue
            elif i == j:
                lev_dists[i][j] = 1
            else:
                m1 = [str(int(x)) for x in msg_i]
                m2 = [str(int(x)) for x in msg_j]
                dist = distance(m1, m2)
                lev_dists[i][j] = dist
                lev_dists[j][i] = dist

    rho = spearmanr(cos_sims, lev_dists, axis=None).statistic * -1
    return rho


def compute_posdis(sender_inputs, messages):
    """
    Computes PosDis.
    """
    messages = [msg.argmax(dim=1) if msg.dim() == 2
                else msg for msg in messages]
    strings = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

    attributes = torch.stack(sender_inputs) \
        if isinstance(sender_inputs, list) else sender_inputs

    gaps = torch.zeros(strings.size(1))
    non_constant_positions = 0.0
    for j in range(strings.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(attributes.size(1)):
            x, y = attributes[:, i], strings[:, j]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def histogram(messages, vocab_size):
    messages = [msg.argmax(dim=1) if msg.dim() == 2
                else msg for msg in messages]
    messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)
    messages = torch.stack(messages) \
        if isinstance(messages, list) else messages

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


def compute_bosdis(sender_inputs, messages, vocab_size):
    """
    Computes BosDis.
    """
    histograms = histogram(messages, vocab_size)
    return compute_posdis(sender_inputs, histograms[:, 1:])


def dump_sender_receiver(
    game: torch.nn.Module,
    dataset: "torch.utils.data.DataLoader",
    gs: bool,
    apply_noise: bool,
    variable_length: bool,
    max_len: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
):
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether Gumbel-Softmax relaxation was used during training
    :param variable_length: whether variable-length communication is used
    :param max_entropy: needed to calculate redundancy of the message
    :param max_len: max message length
    :param device: device (e.g. 'cuda') to be used
    :return:
    """
    train_state = game.training  # persist so we restore it back
    game.eval()

    device = device if device is not None else common_opts.device

    sender_inputs, messages, receiver_inputs, receiver_outputs = [], [], [], []
    labels = []
    redundancy = []

    with torch.no_grad():
        for batch in dataset:
            # by agreement, each batch is (sender_input, labels) plus optional (receiver_input)
            sender_input = move_to(batch[0], device)
            receiver_input = None if len(batch) == 2 else move_to(batch[2], device)

            message = game.sender(sender_input)

            # Under GS, the only output is a message; under Reinforce, two additional tensors are returned.
            # We don't need them.
            if gs:
                log_prob, entropy = None, None
            else:
                message, log_prob, entropy = message

            # Add noise to the message
            if game.channel:
                message = game.channel(message, apply_noise=apply_noise)

            output = game.receiver(message, receiver_input)
            if not gs:
                output = output[0]

            if batch[1] is not None:
                labels.extend(batch[1])

            if isinstance(sender_input, list) or isinstance(sender_input, tuple):
                sender_inputs.extend(zip(*sender_input))
            else:
                sender_inputs.extend(sender_input)

            if receiver_input is not None:
                receiver_inputs.extend(receiver_input)

            if gs:
                message = message.argmax(
                    dim=-1
                )  # actual symbols instead of one-hot encoded

            if not variable_length:
                messages.extend(message)
                receiver_outputs.extend(output)
            else:
                # A trickier part is to handle EOS in the messages. It also might happen that not every message has EOS.
                # We cut messages at EOS if it is present or return the entire message otherwise. Note, EOS id is always
                # set to 0.

                for i in range(message.size(0)):
                    eos_positions = (message[i, :] == 0).nonzero()
                    message_end = (
                        eos_positions[0].item() if eos_positions.size(0) > 0 else -1
                    )
                    assert message_end == -1 or message[i, message_end] == 0
                    if message_end < 0:
                        messages.append(message[i, :])
                    else:
                        messages.append(message[i, : message_end + 1])

                    if gs:
                        receiver_outputs.append(output[i, message_end, ...])
                    else:
                        receiver_outputs.append(output[i, ...])

        redundancies = compute_redundancy_smb(messages, max_len, vocab_size)

    game.train(mode=train_state)

    return sender_inputs, messages, receiver_inputs, receiver_outputs, labels, redundancies  # log_prob, entropy


def crop_messages(interaction):
    """
    Given an Interaction object, removes non EOS symbols after the first EOS.
    """
    assert interaction.message_length is not None
    for i in range(interaction.size):
        length = interaction.message_length[i].long().item()
        interaction.message[i, length:] = 0  # 0 is always EOS
