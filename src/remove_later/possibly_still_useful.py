            # conditional_probs = torch.tensordot(prefix_probs, symbol_probs, dims=([1], [0]))
            # print('conditional probs', conditional_probs.shape, conditional_probs[:5])
            # prefix_probs = prev_probs.transpose(0, -1)[idx].prod(1).t()
            # prev_logits = probs_to_logits(prev_probs)
            # prefix_probs = logits_to_probs(
            #     prev_logits.transpose(0, -1)[idx].sum(1)#.t()
            # )#.unsqueeze(-1)
            # print("prefix probs shapw", prefix_probs.shape)
            # avg_prefix_logits = prefix_logits.logsumexp(0)
            # avg_prefix_probs = logits_to_probs(prefix_logits.logsumexp(0))
            # print('avg prefix probs', avg_prefix_probs.shape, avg_prefix_probs[:5])
            # print('avg prefix logits', avg_prefix_logits.shape, avg_prefix_logits[:5])
            # avg_prefix_probs = prefix_probs.sum(-1) / prefix_probs.sum()
            # print('avg_prefix_probs', avg_prefix_probs.shape, avg_prefix_probs[:5])
            # print('symbol probs', symbol_probs.shape)
            # prefix_logits = prefix_logits.unsqueeze(1)
            # symbol_logits = symbol_logits.unsqueeze(-1)

        # avg_eos_probs[symbol_i] = symbol_probs[:, 0].sum() / symbol_probs.sum()
        # avg_prefix_probs = logits_to_probs(prefix_logits.logsumexp(0))
            # avg_eos_probs[0] = symbol_probs[:, 0].sum() / symbol_probs.size(0)
            # prob = symbol_probs[:, 1:].sum(0)  # non-EOS only
            # prob = symbol_probs.sum(0) / symbol_probs.sum() # prob = prob / prob.sum(0, keepdim=True)
            # log2_prob = torch.clamp(torch.log2(prob), min=min_real)


            # conditional_log2_prob = torch.clamp(torch.log2(conditional_probs), min=min_real)
            # print('conditional_log2_prob', conditional_log2_prob.shape)
    # print('eos probs', avg_eos_probs)
    # print('smb entropies', symbol_entropies)
    # print('total entropy', symbol_entropies.sum())
    # return symbol_entropies.sum()
    # conditional_probs = symbol_probs.unsqueeze(1).expand(*prefix_probs.shape[:-1], -1)
    # print('conditional_entropy', symbol_i, conditional_entropy[:5])
    # print(prefix_probs.size(), prefix_probs[0])
    # indices = torch.arange(prev_probs.size(-1)) 
    # prefix_indices = torch.unique(
    #     indices[:, :min(order, symbol_i)],
    #     dim=0
    #).t()#.view(-1, preceding_probs.size(-2))
    # if prev_probs.size(1) == 1:
    #     prefix_indices = prefix_indices.squeeze()
    # print(symbol_i, prefix_indices.shape, prev_probs.shape)
    # prefix_probs = torch.gathejjr(prev_probs, 1, prefix_indices)
    # print(prefix_probs)
    #print(prev_probs[:, prefix_indices].shape)
    #idx = [torch.arange(len(probs))] \        #+ []
    #prefix_probs = prev_probs[:, prefix_indices, torch.arange(prev_probs.size(-1))]
    #print(prefix_probs)
    #print(prefix_probs.shape)
    # prefix_probs = torch.gather(
    #    if symbol_i < order:
    #        prefix_indices = indices[indices[0] < symbol_i]
    #    else:
    #        prefix_indices = indices.clone()
    #        prefix_indices[0] += indices[0].max() - (order - 1)
    #    print(prefix_indices)
    #    for idx in prefix_indices:
    #        prefix_probs = messages[torch.arange(len(messages)), idx]
    #        print(prefix_probs)
    # print("step", symbol_i)
    # prefix_probs = messages[:, symbol_i - 1, :] if symbol_i != 0 \
    #     else torch.ones_like(messages[:, symbol_i, :])
    #    print("prefix_probs", prefix_probs.shape)
    #    print("symbol_probs", symbol_probs.shape)
    #    probs = (prefix_probs * symbol_probs).sum(0)
    #    probs = probs / probs.sum(0, keepdim=True)
    #    print("probs shape", probs.shape)
    #    eos_probs[symbol_i] = probs[0]
    #    probs = probs[1:] / probs[1:].sum()  # exclude EOS prob
    #    log2_prob = torch.clamp(torch.log2(probs), min=min_real)
    #    symbol_entropies[symbol_i] = (-log2_prob * probs).sum()
    # prefix_probs[:, step_i  = prefix_probs * symbol_probs
    # shape = [probs.size(0)] + [1 for _ in range(prefix_probs.dim())]
    # prefix_probs = prefix_probs.unsqueeze(0).expand(probs.size(0), *prefix_probs.shape)
    # print(prefix_probs.shape, shape)
    # prefix_probs = prefix_probs * probs.view(shape)
    # prefix_probs = torch.ones_like(messages[:, 0, :])
    # prefix_probs = torch.ones_like(messages[0, 0, 0])
    # prefix_probs = torch.cat([
    #     torch.ones_like(messages[0, :1, 1:]),
    #     messages[0, 1:, 1:],
    # ], dim=1)
    #indices = torch.empty([messages.size(-1) - 1] * order, device=messages.device).long()
    # print("smb entrs:", symbol_entropies)
    # print("eos probs:", eos_probs)
    # entropy = 0.
    # for i in range(len(symbol_entropies) - 1):  # last symbol is always EOS
    #     symbol_entropy = symbol_entropies[i].item()
    #    # print(symbol_entropy, 'smb entropy', i)
    #    eos_prob = eos_probs[i].item()
    #    entropy += (
    #        binary_entropy(eos_prob)
    #        # + eos_prob * 0
    #        + (1 - eos_prob) * symbol_entropy
    #    )

    # not_eosed_before = 1.
    # length_probs = torch.zeros_like(avg_eos_probs)
    # for i in range(probs.size(1)):
    #     length_probs[i] = not_eosed_before * avg_eos_probs[i]
    #     not_eosed_before *= 1 - avg_eos_probs[i]

    # print('LEN PROBS', length_probs)
    # print('EOS PROBS', eos_probs)

    # length_log2_prob = torch.clamp(torch.log2(length_probs), min=min_real)
    # entropy += (-length_probs * length_log2_prob).sum(-1).item()

    if return_length_probs:
        return entropy, length_probs
    else:
        return entropy


###################
# ancm/metrics.py #
###################


def tensor_entropy(
        x: torch.Tensor,
        alphabet: Optional[torch.Tensor] = None,
        estimator: str = 'GOOD-TURING'):
    """
    Estimates entropy of the RV X represented by the tensor x.

    If the tensor has more than one dimension, each tensors indexed by the 1st
    dimension is treated as one of the elements to apply the operation upon.
    """
    if x.dim() == 0:
        return 0.
    elif x.dim() > 1:
        _, x = torch.unique(x, return_inverse=True, dim=0)

    H = it.entropy(x.numpy(), Alphabet_X=alphabet, estimator=estimator)

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
        all_symbols = torch.unique(x)
        alphabet = all_symbols.unsqueeze(0).expand(x.size(1), -1)
        for i in range(x.size(1)):
            unique_symbols = torch.unique(x[:, i])
            mask = torch.isin(all_symbols, unique_symbols)
            mask = torch.logical_not(mask)
            alphabet[i, mask] = -1

        return alphabet

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

    return torch.tensor(alphabet)


def sequence_entropy(
        x: torch.Tensor,
        alphabet: Optional[torch.Tensor] = None,
        estimator: str = 'GOOD-TURING') -> float:
    """
    Estimates the entropy of the RV X represented by the tensor x, assuming
    that X if a compound RV if x has 2 dimensions, i.e. X = X1, ..., Xm.
    The entropy is approximated from the sample using the James-Stein formula.
    """
    if x.dim() == 1 or len(x) == len(x.view(-1)):
        return tensor_entropy(x, estimator=estimator)

    if alphabet is not None:
        alphabet = alphabet.numpy()

    try:
        H = entropy_joint(
            x.t().numpy(),
            Alphabet_X=alphabet,
            estimator=estimator)
    except:
        print(alphabet)
        print(x.argmax(-1))
        print(x)

    return H.item()



def compute_mi_disabled(messages: torch.Tensor, attributes: torch.Tensor, vocab_size: int, estimator='GOOD-TURING') -> dict:
    """
    Computes multiple information-theoretic metrics: message entropy, input entropy,
    mutual information between messages and target objects, entropy of each input
    dimension and mutual information between each input dimension and messages.
    In case x has two dimensions, the function assumes it represents a
    compound RV, i.e. a sequence of realizations x1, ..., xm of RVs
    X1, ..., Xm, and returns an estimate of I(X1, ..., Xm; Y)

    Alphabet - applied to the message
    """

    alphabet = build_alphabet(messages, vocab_size=vocab_size)

    if attributes.size(1) == 1:
        entropy_msg = sequence_entropy(messages, alphabet, estimator)
        entropy_attr = tensor_entropy(attributes, estimator=estimator)
        mi_msg_attr, entropy_msg_attr = mutual_info(messages, attributes, alphabet, estimator)
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
        entropy_msg = sequence_entropy(messages, alphabet, estimator)
        entropy_attr = sequence_entropy(attributes, estimator=estimator)
        entropy_attr_dim = [
            tensor_entropy(attributes[:, i], estimator=estimator)
            for i in range(attributes.size(-1))]
        mi_msg_attr_dim, entropy_msg_attr_dim = list(zip(*[
            mutual_info(messages, attributes[:, i], alphabet, estimator)
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

def mutual_information(
        x: torch.Tensor,
        y: torch.Tensor,
        alphabet_x: Optional[torch.Tensor] = None,
        estimator: Optional[str] = 'GOOD-TURING') -> Tuple[float, float]:
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

    x = x if x.dim() == 2 else x.unsqueeze(-1)
    y = y if y.dim() == 2 else y.unsqueeze(-1)
    xy = torch.cat([x, y], dim=-1)

    alphabet_x = build_alphabet(x) if alphabet_x is None else alphabet_x
    alphabet_y = build_alphabet(y, True)
    if alphabet_x.dim() == 1:
        alphabet_x = alphabet_x.unsqueeze(0)

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

    H_x = sequence_entropy(x, alphabet_x, estimator)
    H_y = tensor_entropy(y, estimator=estimator)
    H_xy = sequence_entropy(xy, alphabet_xy, estimator)
    I_xy = H_x + H_y - H_xy

    return I_xy, H_xy


def compute_redundancy_disabled(
        messages: torch.Tensor,
        vocab_size: int,
        channel: Optional[str],
        error_prob: float,
        maxiter: int = 1000) -> float:
    """
    Computes a redundancy based on the symbol-level message entropy.
    The value returned is multiplied by a factor dependent on the maximum
    message length, so that the range of possible values is [0, 1].
    """
    if vocab_size == 1:
        return 1.

    # if channel == 'erasure' and error_prob > 0.:
    #     vocab_size += 1
    #     if alphabet is not None:
    #         alphabet = list(alphabet)
    #         if vocab_size not in alphabet:
    #            alphabet.append(vocab_size)

    # if alphabet is not None:
    #     vocab_size = len(alphabet)
    symbols = torch.arange(vocab_size)
    alphabet = build_alphabet(messages, vocab_size=vocab_size)  # symbols=torch.arange(vocab_size))

    max_len = messages.size(1) - 1
    vocab_size = (symbols != vocab_size).int().sum().item()

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

###########
# util.py #
###########

    # entropy_msg = channel_dict['entropy_msg']
    # entropy_smb = channel_dict['entropy_smb'][:, :-1]

    # if isinstance(entropy_msg, torch.Tensor):
    #     max_entropy_msg, _ = maximize_sequence_entropy(
    #         max_len=opts.max_len,
    #         vocab_size=receiver_vocab_size,
    #         channel=channel,
    #         error_prob=error_prob)
    #     max_entropy_smb = maxent_smb(channel, error_prob, opts.vocab_size)
    #     results['redund_msg'] = (1 - entropy_msg / max_entropy_msg).mean().item()
    #     results['redund_smb'] = (1 - entropy_smb / max_entropy_smb).mean().item()
    # else:
    #     results['redund_msg'] = None
    #     results['redund_smb'] = None
    #     # mi = MI(entropy_smb, attr)
     # update_dict_names(mi, 'v2'))

    # entropy_msg_nn = channel_dict['message_entropy_nn']
    # entropy_smb_nn = channel_dict['symbol_entropy_nn']
    # max_entropy_msg_nn, _ = maximize_sequence_entropy(
    #     max_len=opts.max_len,
    #     vocab_size=receiver_vocab_size,
    #     channel=None,
    #     error_prob=0.0)
    # max_entropy_smb_nn = maxent_smb(None, 0.0, opts.vocab_size)
    # results['redund_msg_v2'] = (1 - entropy_msg_nn / max_entropy_msg_nn).mean().item()
    # results['redund_smb_v2'] = (1 - entropy_smb_nn / max_entropy_smb_nn).mean().item()
    # mi_nn = MI(entropy_smb_nn, attr)
    # results.update(update_dict_names(mi_nn, 'before_noise'))

    # else:
    #    entropy_msg = channel_dict['message_entropy']
    #    entropy_smb = channel_dict['symbol_entropy']
    #    if entropy_msg is not None:
    #        max_entropy_msg, _ = maximize_sequence_entropy(
    #            max_len=opts.max_len,
    #            vocab_size=receiver_vocab_size,
    #            channel=None,
    #            error_prob=0.0)
    #        max_entropy_smb = maxent_smb(None, 0., receiver_vocab_size)
    #        results['redund_msg'] = (1 - entropy_msg / max_entropy_msg).mean().item()
    #        results['redund_smb'] = (1 - entropy_msg / max_entropy_smb).mean().item()
    #        mi = MI(entropy_smb, attr)
    #        results.update(update_dict_names(mi, 'no_noise'))


    # if opts.image_input:
    #    mi_attr_msg = compute_mi(msg, attr, receiver_vocab_size)
    #    results['entropy_msg'] = mi_attr_msg['entropy_msg']
    #    results['entropy_attr'] = mi_attr_msg['entropy_attr']
    #    results['entropy_attr_dim'] = {
    #        name: value for name, value
    #        in zip(attr_names, mi_attr_msg['entropy_attr_dim'])}
    #    results['mi_msg_attr_dim'] = {
    #        name: value for name, value
    #        in zip(attr_names, mi_attr_msg['mi_msg_attr_dim'])}
    #    results['vi_msg_attr_dim'] = {
    #        name: value for name, value
    #        in zip(attr_names, mi_attr_msg['vi_msg_attr_dim'])}
    #    results['vi_norm_msg_attr_dim'] = {
    #        name: value for name, value
    #        in zip(attr_names, mi_attr_msg['vi_norm_msg_attr_dim'])}
    #    results['is_msg_attr_dim'] = {
    #        name: value for name, value
    #        in zip(attr_names, mi_attr_msg['is_msg_attr_dim'])}
    # else:
    #    unique_objects, categorized_input = torch.unique(
    #        s_inp, return_inverse=True, dim=0)
    #    if len(unique_objects) < 200:  # test
    #        categorized_input = categorized_input.unsqueeze(-1).to(torch.float)
    #        mi_inp_msg = compute_mi(msg, categorized_input, receiver_vocab_size)
    #        results['entropy_msg'] = mi_inp_msg['entropy_msg']
    #        results['entropy_inp'] = mi_inp_msg['entropy_attr']
    #        results['mi_msg_inp'] = mi_inp_msg['mi_msg_attr']
    #        results['vi_msg_inp'] = mi_inp_msg['vi_msg_attr']
    #        results['vi_norm_msg_inp'] = mi_inp_msg['vi_norm_msg_attr']
    #        results['is_msg_inp'] = mi_inp_msg['vi_msg_attr']
    #    else:  # train
    #        results['entropy_msg'] = None  # mi_inp_msg['entropy_msg']
    #        results['entropy_inp'] = None  # mi_inp_msg['entropy_attr']
    #        results['mi_msg_inp'] = None  # mi_inp_msg['mi_msg_attr']
    #        results['vi_msg_inp'] = None  # mi_inp_msg['vi_msg_attr']
    #        results['vi_norm_msg_inp'] = None  # mi_inp_msg['vi_norm_msg_attr']
    #        results['is_msg_inp'] = None  # mi_inp_msg['vi_msg_attr']
    #    mi_cat_msg = compute_mi(msg, attr, receiver_vocab_size)
    #    results['entropy_cat'] = mi_cat_msg['entropy_attr']
    #    results['mi_msg_cat'] = mi_cat_msg['mi_msg_attr']
    #    results['vi_msg_cat'] = mi_cat_msg['vi_msg_attr']
    #    results['vi_norm_msg_cat'] = mi_cat_msg['vi_norm_msg_attr']
    #    results['is_msg_cat'] = mi_cat_msg['vi_msg_attr']

########
# loss #
########

    # if _sender_input.dim() == 2:  # VISA
    #     size = _receiver_input.shape
    #     all_ids = torch.arange(_receiver_input.size(0))
    #     # priors = _receiver_input.sum(1) / size[1]
    #     # print(priors, priors.shape)
    #     target_object = _receiver_input[all_ids, _labels]
    #     selected_object = torch.zeros_like(target_object)
    #     for i in range(size[1]):
    #        prob_i = logits_to_probs(receiver_output[all_ids, i])
    #        selected_object += _receiver_input[all_ids, i] * prob_i.unsqueeze(-1)
    #    min_positive = torch.finfo(receiver_output.dtype).tiny
    #    target_object = torch.clamp(target_object, min=min_positive)
    #    selected_object = torch.clamp(selected_object, min=min_positive)
    #    log_ratio = torch.log(target_object) - torch.log(selected_object)
    #    kld = (target_object * log_ratio).sum(-1)
    #    # print(log_ratio, kld)
    # else:  # Obverter
    #

def compute_bigram_entropy(train_messages, test_messages, key1='message', key2='message'):
    """
    Previous attempts, written in a rush. Doesn't work for now.
    """
    #train_lengths = find_lengths(train_messages)
    #test_lengths = find_lengths(test_messages)
    try:
        train_strings = [m[key1].split(',') for m in train_messages]
        test_strings = [m[key2].split(',') for m in test_messages]
    except:
        return None, None
    # train_strings, test_strings = [], []
    # for message, length in zip(train_messages, train_lengths):
    #    message = message.tolist()[:length]
    #    message = [str(int(x)) for x in message]
    #    train_strings.append(message)
    #for message, length in zip(test_messages, train_lengths):
    #    message = message.tolist()[:length]
    #    message = [str(int(x)) for x in message]
    #    test_strings.append(message)
    # text_bigrams = [ngrams(sent, 2) for sent in strings]
    train_corpus = [smb for string in train_strings for smb in string]
    test_corpus = [smb for string in test_strings for smb in string]
    vocab_train = Vocabulary(train_corpus, unk_cutoff=0)
    #print(train_messages[:10])
    # print(train_strings[0])
    #est = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
    #lm = NgramModel(2, train_corpus, estimator=est)
    #lm = MLE(2,
    #lm = Lidstone(gamma=0.2, 2)
    #lm.fit(test_strings, vocab_train)
    #cross_entropy = lm.entropy(test_strings, vocab_train)
    #perplexity = lm.perplexity(test_strings, vocab_train)

    #return cross_entropy, perplexity

    # text_unigrams = [ngrams(sent, 1) for sent in strings]



    pass


########
# dump #
########
            # symbols_nn = crop_messages(message_nn.argmax(-1))
            # message_nn = torch.zeros_like(message).view(size[0] * size[1], size[2])
            # message_nn.scatter_(1, symbols_nn.view(size[0] * size[1], size[2]), 1)
            # message_nn = message_nn.view(size)
            # symbols = message.argmax(-1)
            # lengths = find_lengths(symbols)
            #    symbols_nn = message_nn.argmax(-1)
            #    lengths_nn = find_lengths(symbols_nn)

             
            # indexes = sample.argmax(dim=-1)
            # one_hot = torch.zeros_like(logits).view(-1, logits.size(-1))
            # one_hot.scatter_(1, indexes.view(-1, 1), 1)
            # one_hot = one_hot.view(*logits.size())

            # message_nn = torch.zeros_like(message_nn)
            
            # if True:  # TODO unindent? game.training:
            #     lengths = torch.zeros_like(message[:, 0, 0])
            #     eosed_mask = torch.zeros_like(message[:, 0, 0]).bool()
            #     lengths_nn = lengths.clone()
            #     eosed_mask_nn = eosed_mask.clone()

            #     symbols, symbols_nn = [], []
            #     for step in range(message.size(1)):
            #         # TODO add_mask_nn etc: remove if we dont bring back Deletion
            #         eos_mask = message[:, step, 0] == 1
            #        eos_mask_nn = message_nn[:, step, 0] == 1
            #        add_mask = torch.logical_and(
            #            eos_mask, torch.logical_not(eosed_mask))
            #        add_mask_nn = torch.logical_and(
            #            eos_mask_nn, torch.logical_not(eosed_mask_nn))

            #        lengths[add_mask] = step
            #        lengths_nn[add_mask_nn] = step

            #        step_symbols = message[:, step]
            #        step_symbols[eosed_mask, 1:] = 0
            #        step_symbols[eosed_mask, 0] = 1

            #        step_symbols_nn = message_nn[:, step]
            #        step_symbols_nn[eosed_mask_nn, 1:] = 0
            #        step_symbols_nn[eosed_mask_nn, 0] = 1

            #        symbols.append(step_symbols)
            #        symbols_nn.append(step_symbols_nn)

            #        eosed_mask = torch.logical_or(eosed_mask, eos_mask)
            #        eosed_mask_nn = torch.logical_or(eosed_mask_nn, eos_mask_nn)
            #        # not_eosed_before = torch.logical_and(not_eosed_before, torch.logical_not(eos_mask))
            #        # not_eosed_before = torch.logical_and(not_eosed_before_nn, torch.logical_not(eos_mask_nn))

            #        # symbols = symbol_probs.argmax(-1)
            #        # TODO we might want to take argmax here for during training
            #        # distr = OneHotCategorical(probs=symbol_probs)  # 32x10 = 1 smb
            #        # symbols.append(distr.sample())

            #        # channel_dict['length_probs'][:, step] = add_mask.detach()
            #        # aggregate message entropy
            #        # channel_dict['entropy_msg'] = channel_dict['entropy_msg'] \
            #        #     + torch.where(
                    #         add_mask > 1e-5,
                    #         add_mask.detach() * prefix_entropy,
                    #         0)
                    # entropy of the symbol, assuming it is not eos
                    # (the furmula exploits decomposability of entropy)
                    # prefix_entropy += (
                    #     channel_dict['entropy_smb'][:, step]
                    #     - game.channel.tensor_binary_entropy(eos_mask.detach())
                    # ) / (1 - eos_mask.detach())
                    # not_eosed_before = not_eosed_before * (1.0 - eos_mask)
                    # prefix_entropy_nn += (
                    #     channel_dict['entropy_smb_nn'][:, step]
                    #     - game.channel.tensor_binary_entropy(message_nn[:, step, 0])
                    # ).detach() / (1 - eos_mask.detach())
                    # not_eosed_before_nn = (
                    #     not_eosed_before_nn * (1.0 - message_nn[:, step, 0])
                    # ).detach()

                # adjust message entropy to cover message length variability
                # exclude appended EOS from symbol entropy and compute redundancy
                # game.channel.update_values(channel_dict)
                # for k in (
                #     'entropy_msg',
                #     'entropy_msg_nn',
                #     'entropy_smb',
                #     'entropy_smb_nn',
                #     'redundancy_msg',
                #     'redundancy_smb',
                #     'max_entropy',
                # ):
                #     channel_output[k].append(channel_dict[k])
                #message = torch.stack(symbols, dim=1)

            # else:
            #     for k in (
            #        'entropy_msg',
            #        'entropy_msg_nn',
            #        'entropy_smb',
            #        'entropy_smb_nn',
            #        'redundancy_msg',
            #        'redundancy_smb',
            #        'max_entropy',
            #    ):
            #        channel_output[k].append(None)


########
# crop #
########

    # if messages.dim() == 3:
    #     msg = messages.argmax(-1)
    #     lengths = find_lengths(msg).unsqueeze(-1).expand(msg.size())
    #    positions = torch.arange(msg.size(1)).unsqueeze(0).expand(msg.size())
    #    nonzero_ids = msg.nonzero()
    #    nonzero_chunks = nonzero_ids.t().chunk(chunks=2)
    #    targets = (positions[nonzero_chunks] > lengths[nonzero_chunks] - 1)
    #    targets = targets.squeeze()
    #    target_ids = nonzero_ids[targets]
    #    # if no targets are found, the dimension is 3
    #    if target_ids.dim() == 2:
    #        target_chunks = target_ids.t().chunk(chunks=3)
    #        replacement_probs = torch.zeros_like(messages[0, 0])
    #        replacement_probs[0] = 1.
    #        messages[target_chunks] = replacement_probs
    #    # if not torch.all(torch.eq(messages.argmax(-1), msg)):
    #    #     check = [(messages[i], messages[i].argmax(-1), msg[i]) for i in range(len(msg))
    #    # if not torch.all(torch.eq(messages[i].argmax(-1), msg[i]))]
    #    #    for item in check:
    #    #        print(item[0])
    #    #        print(item[1])
    #    #        print(item[2])
    #    #        print('----')
    #return messages
