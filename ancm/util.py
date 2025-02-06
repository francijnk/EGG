import math
import json
import torch
import numpy as np
from tabulate import tabulate
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.distributions import OneHotCategorical

from typing import Optional

from egg.core.util import find_lengths, move_to, get_opts

from ancm.archs import tensor_binary_entropy
from ancm.metrics import (
    # compute_conceptual_alignment,
    compute_max_rep,
    compute_redundancy,
    compute_adjusted_redundancy,
    compute_accuracy2,
    compute_top_sim,
    compute_mi,
    compute_posdis,
    compute_bosdis,
    maximize_sequence_entropy,
    binary_entropy,
    MI,
)

common_opts = get_opts()


class ObjectDataset(Dataset):
    def __init__(self, obj_sets, labels, attributes=None, attribute_names=None, n_permutations=None):
        self.obj_sets = obj_sets
        self.labels = labels
        self.attributes = attributes
        self.attribute_names = attribute_names

        n_objects = obj_sets.shape[1]
        self.n_distractors = n_objects - 1
        if n_permutations is not None:
            self.permutations = [
                np.random.permutation(np.arange(n_objects))
                for _ in range(len(obj_sets) * (n_permutations - 1))]
        else:
            self.permutations = []

    def __len__(self):
        return len(self.obj_sets) + len(self.permutations)

    def __getitem__(self, idx):
        if idx < len(self.obj_sets):
            return self.obj_sets[idx], self.labels[idx], self.attributes[idx]
        else:
            _idx = idx % len(self.obj_sets)
            permutation = self.permutations[_idx]
            obj_set = self.obj_sets[_idx, permutation, ...]
            label = permutation[self.labels[_idx]]

            n_attributes = len(self.attribute_names) // (self.n_distractors + 1)
            distr_attr_names = [
                self.attribute_names[n_attributes * (permutation[i] - 1) + j]
                for i in range(self.n_distractors + 1)
                for j in range(n_attributes)]

            attribute_tuple = tuple(
                self.attributes[_idx][attr_name]
                for attr_name in distr_attr_names)

            dtype = [(item, np.int64) for item in distr_attr_names if item.startswith('target')]
            num_attr_per_distractor = len(dtype)
            for i, name in enumerate(distr_attr_names):
                d_num = i // num_attr_per_distractor
                if d_num == label:
                    continue
                if d_num < label:
                    prefix = f'distr_{d_num}_'
                else:
                    prefix = f'distr_{d_num - 1}_'
                suffix = name.split('_')[-1]
                dtype.append((prefix + suffix, np.int64))

            attributes = np.array([attribute_tuple], dtype=dtype)

            assert not np.isnan(obj_set).any()
            return obj_set, label, attributes


class DataHandler:
    def __init__(self, opts):
        self.data_path = opts.data_path
        self.batch_size = opts.batch_size
        self.shuffle_train_data = opts.no_shuffle

        self._n_features = None
        self.n_distractors = None

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    def load_data(self, opts):
        data = np.load(self.data_path)
        train = data["train"], data["train_labels"], data["train_attributes"]
        val = data["valid"], data["valid_labels"], data["valid_attributes"]
        test = data["test"], data["test_labels"], data["test_attributes"]
        attribute_names = data["train_attributes"].dtype.names

        self._n_features = train[0].shape[-1]
        self.train_samples = train[0].shape[0]
        self.validation_samples = val[0].shape[0]
        self.test_samples = test[0].shape[0]
        self.n_distractors = train[0].shape[1] - 1

        opts.train_samples = self.train_samples
        opts.validation_samples = self.validation_samples
        opts.test_samples = self.test_samples
        opts.n_distractors = self.n_distractors
        opts.n_features = self.n_features

        train_dataset = ObjectDataset(
            *train, attribute_names, n_permutations=opts.n_permutations_train)
        val_dataset = ObjectDataset(*val, attribute_names)
        test_dataset = ObjectDataset(*test, attribute_names)

        def _collate(batch):
            obj_sets, target_ids, obj_attributes = zip(*batch)
            bs = self.batch_size

            r_inputs, labels = np.vstack(np.expand_dims(obj_sets, 0)), np.array(target_ids)
            targets = r_inputs[np.arange(bs), labels]
            obj_attributes = np.vstack(obj_attributes)

            aux = {}
            for attr_name in obj_attributes.dtype.names:
                aux[attr_name] = torch.from_numpy(obj_attributes[attr_name]).float()

            return (
                torch.from_numpy(targets).float(),
                torch.from_numpy(labels).long(),
                torch.from_numpy(r_inputs).float(),
                aux,
            )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=_collate,
            drop_last=True,
            shuffle=self.shuffle_train_data)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=_collate,
            drop_last=True,
            shuffle=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=_collate,
            drop_last=True,
            shuffle=False)

        if "train_eval" in data:
            if "train_eval_attributes" in data:
                train_eval = (
                    data["train_eval"],
                    data["train_eval_labels"],
                    data["train_eval_attributes"])
            else:
                train_eval = data["train_eval"], data["train_eval_labels"]
            train_eval_dataset = ObjectDataset(*train_eval)
            train_eval_dataloader = DataLoader(
                train_eval_dataset,
                batch_size=self.batch_size,
                collate_fn=_collate,
                drop_last=True,
                shuffle=False)
        else:
            train_eval_dataloader = None

        return train_dataloader, val_dataloader, test_dataloader, train_eval_dataloader


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def build_optimizer(game, opts):
    if opts.optimizer.lower() == 'RMSprop'.lower():
        optimizer = torch.optim.RMSprop
    elif opts.optimizer.lower() == 'Adam'.lower():
        optimizer = torch.optim.Adam
    else:
        raise ValueError('Optimizer must be either RMSprop or Adam')
    return optimizer([
        {"params": game.sender.parameters(), "lr": opts.sender_lr},
        {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
    ])


class Dump:
    def __init__(self, sender_inputs, messages, receiver_inputs, receiver_outputs, labels, attributes):
        self.sender_inputs = torch.stack(sender_inputs)
        self.messages = crop_messages(torch.stack(messages))
        print(self.messages.shape)
        self.receiver_inputs = torch.stack(receiver_inputs)
        self.labels = torch.stack(labels)

        attributes = {k: torch.cat(v, dim=0) for k, v in attributes.items()}

        self.target_attributes = {
            k.replace('target_', ''): v for k, v in attributes.items()
            if not k.startswith('distr')}
        self.distractor_attributes = {
            k: v for k, v in attributes.items()
            if k.startswith('distr')}

        if self.messages.dim() == 3:
            self.lengths = find_lengths(self.messages.argmax(-1))
            self.receiver_outputs = torch.stack(receiver_outputs)
            self.receiver_outputs = torch.stack([
                receiver_output[self.lengths[i] - 1].argmax(-1)
                for i, receiver_output in enumerate(receiver_outputs)
            ], dim=0)
            self.strings = self.messages.argmax(-1)
        else:
            self.lengths = find_lengths(self.messages)
            self.receiver_outputs = torch.stack(receiver_outputs)
            self.strings = self.messages

    def __len__(self):
        return len(self.sender_inputs)

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self.sender_inputs[i],
                self.strings[i, :self.lengths[i]],
                self.receiver_inputs[i],
                self.receiver_outputs[i].item(),
                self.labels[i].int().item(),
                {
                    k: self.target_attributes[k][i].int().item()
                    for k in self.target_attributes
                },
                [
                    {k: self.distractor_attributes[k][i].int().item()}
                    for k in self.distractor_attributes
                ],
            )

    def get_tensors(self):
        if self.messages.dim() == 3:
            messages = self.messages.argmax(-1)
        target_attributes = torch.cat(list(self.target_attributes.values()), dim=-1)
        distractor_attributes = torch.cat(list(self.distractor_attributes.values()), dim=-1)
        # distractor_attributes = [
        #     [torch.cat(a_tensors, dim=0) for a_tensors in self.distractor_attributes.values()]
        #     for attributes in self.distractor_attributes]
        attribute_names = list(self.target_attributes.keys())

        return self.sender_inputs, messages, self.receiver_inputs, \
            self.receiver_outputs, self.labels, target_attributes, \
            distractor_attributes, attribute_names


def dump_sender_receiver(
        game: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        apply_noise: bool,
        max_len: int,
        vocab_size: int,
        mode: str,
        device: Optional[torch.device] = None):
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param apply_noise: whether noise should be applied
    :param max_entropy: needed to calculate redundancy of the message
    :param max_len: max message length
    :param device: device (e.g. 'cuda') to be used
    :return:
    """

    device = device if device is not None else common_opts.device

    sender_inputs, messages = [], []
    receiver_inputs, receiver_outputs, labels = [], [], []
    attributes, channel_dict = defaultdict(list), defaultdict(list) 

    for i, batch in enumerate(dataset):
        sender_input = move_to(batch[0], device)
        sender_inputs.extend(sender_input)
        labels.extend(batch[1])
        receiver_input = move_to(batch[2], device)
        receiver_inputs.extend(receiver_input)
        for key, val in batch[3].items():  #  ok
            attributes[key].append(val)
        with torch.no_grad():
            sender_output = game.sender(sender_input)
        message_nn = sender_output[0]
        entropy_nn = sender_output[2]

        if mode == 'rf':  # Add noise to the message
            message = crop_messages(message_nn)
            message = game.channel(message_nn, entropy_nn, apply_noise=apply_noise)

            # TODO 
        elif mode == 'gs':
            message, channel_output, = game.channel(
                message_nn, entropy=entropy_nn, apply_noise=apply_noise)
            #for key, val in channel_output.items():  # 32 messages appends is ok
            #    channel_dict[key].append(val)
            if game.training:
                not_eosed_before = torch.ones(message.size(0)).to(device)
                not_eosed_before_nn = not_eosed_before.clone().detach()

                symbols = []
                for step in range(message.size(1)):
                    eos_mask = message[:, step, 0]
                    # add_mask = eos_mask * not_eosed_before
                    symbol_probs = message[:, step]
                    symbol_probs[:, 1:] *= not_eosed_before.unsqueeze(-1)
                    eos = torch.zeros_like(symbol_probs)
                    eos[:, 0] = 1
                    symbol_probs = torch.where(
                        symbol_probs.sum(-1, keepdim=True) > 0,
                        symbol_probs / symbol_probs.sum(-1, keepdim=True),
                        eos)
                    distr = OneHotCategorical(probs=symbol_probs)  # 32x10 = 1 smb
                    symbols.append(distr.sample())

                    h_not_eosed = tensor_binary_entropy(not_eosed_before)
                    channel_output['message_entropy'] += h_not_eosed \
                        + not_eosed_before * channel_output['symbol_entropy'][:, step]
                        # + (1 - prob_not_eosed) * 0
                    not_eosed_before = not_eosed_before * (1.0 - eos_mask)

                    channel_output['message_entropy_nn'] += h_not_eosed \
                        + not_eosed_before_nn * channel_output['symbol_entropy_nn'][:, step]
                        # + (1 - prob_not_eosed_nn * 0
                    not_eosed_before_nn *= 1.0 - message_nn[:, step, 0]

                for k, v in channel_output.items():
                    channel_dict[k].append(v)
                message = torch.stack(symbols).permute(1, 0, 2)

            else:
                for k, v in channel_output.items():
                    channel_dict[k].append(v)

        messages.extend(message)
        with torch.no_grad():
            output = game.receiver(message, receiver_input)
        receiver_outputs.extend(output)
    channel_dict = {k: torch.cat(v, dim=0) for k, v in channel_dict.items()}


    return Dump(sender_inputs, messages, receiver_inputs, receiver_outputs, labels, attributes), channel_dict


def get_results_dict(dump, receiver, opts, unique_dict, channel_dict, noise=True):
    s_inp, msg, r_inp, r_out, labels, attr, distr_attr, attr_names \
        = dump.get_tensors()

    if noise and opts.channel == 'erasure' and opts.error_prob > 0.:
        receiver_vocab_size = opts.vocab_size + 1
    else:
        receiver_vocab_size = opts.vocab_size
    actual_vocab = set(int(s) for m in msg for s in m.tolist())

    # receiver_outputs = move_to(receiver_outputs, device)
    # labels = move_to(labels, device)

    channel, error_prob = (None, 0.) if not noise \
        else (opts.channel, opts.error_prob)

    results = {
        'accuracy': (r_out == labels).float().mean().item(),
        'accuracy2': compute_accuracy2(dump, receiver, opts),
        'unique_messages': len(torch.unique(msg, dim=0)),
        'unique_target_objects': len(unique_dict.keys()),
        'actual_vocab_size': len(actual_vocab),
        'redundancy': compute_redundancy(msg, receiver_vocab_size, channel, error_prob),
        'redundancy_adj': compute_adjusted_redundancy(
            msg, channel, error_prob, torch.arange(receiver_vocab_size)),
        'redundancy_adj_voc': compute_adjusted_redundancy(
            msg, channel, error_prob, actual_vocab),
        'max_rep': compute_max_rep(msg).mean().item(),
        # alignment = compute_conceptual_alignment(
        #     test_data, _receiver, _sender, device, opts.batch_size)
    }

    def maxent_smb(channel, p, vocab_size):
        if channel is None or channel == 'symmetric':
            return np.log2(vocab_size)
        elif channel == 'erasure':
            return binary_entropy(p) + (1 - p) * math.log2(vocab_size)

    update_dict_names = lambda dct, sffx: {f'{k}_{sffx}': v for k, v in dct.items()}
    if noise:
        entropy_msg = channel_dict['message_entropy']
        entropy_smb = channel_dict['symbol_entropy']
        max_entropy_msg, _ = maximize_sequence_entropy(
            max_len=opts.max_len,
            vocab_size=receiver_vocab_size,
            channel=channel,
            error_prob=error_prob)
        max_entropy_smb = maxent_smb(channel, error_prob, opts.vocab_size)
        results['redund_smb_v2'] = (1 - entropy_msg / max_entropy_smb).mean().item()
        results['redund_smb_v2'] = (1 - entropy_smb / max_entropy_smb).mean().item()
        mi = MI(entropy_smb, attr)
        results.update(update_dict_names(mi, 'v2'))

        entropy_msg_nn = channel_dict['message_entropy_nn']
        entropy_smb_nn = channel_dict['symbol_entropy_nn']
        max_entropy_msg_nn, _ = maximize_sequence_entropy(
            max_len=opts.max_len,
            vocab_size=receiver_vocab_size,
            channel=None,
            error_prob=0.0)
        max_entropy_smb_nn = maxent_smb(None, 0.0, opts.vocab_size)
        results['redund_msg_v2_before_noise'] = (1 - entropy_msg_nn / max_entropy_msg_nn).mean().item()
        results['redund_smb_v2_before_noise'] = (1 - entropy_smb_nn / max_entropy_smb_nn).mean().item()
        mi_nn = MI(entropy_smb_nn, attr)
        results.update(update_dict_names(mi_nn, 'before_noise'))

    else:
        entropy_msg = channel_dict['message_entropy']
        entropy_smb = channel_dict['symbol_entropy']
        max_entropy_msg, _ = maximize_sequence_entropy(
            max_len=opts.max_len,
            vocab_size=receiver_vocab_size,
            channel=None,
            error_prob=0.0)
        max_entropy_smb = maxent_smb(None, 0., receiver_vocab_size)
        results['redund_msg_v2_no_noise'] = (1 - entropy_msg / max_entropy_msg).mean().item()
        results['redund_smb_v2_no_noise'] = (1 - entropy_msg / max_entropy_smb).mean().item()
        mi = MI(entropy_smb, attr)
        results.update(update_dict_names(mi, 'no_noise'))

    if opts.image_input:
        results['topographic_rho'] = compute_top_sim(attr, msg)
        results['posdis'] = compute_posdis(attr, msg)
        results['bosdis'] = compute_bosdis(attr, msg, receiver_vocab_size)
    else:
        results['topographic_rho'] = compute_top_sim(s_inp, msg)
        results['topographic_rho_category'] = compute_top_sim(attr, msg)

    if opts.image_input:
        mi_attr_msg = compute_mi(msg, attr, receiver_vocab_size)
        results['entropy_msg'] = mi_attr_msg['entropy_msg']
        results['entropy_attr'] = mi_attr_msg['entropy_attr']
        results['entropy_attr_dim'] = {
            name: value for name, value
            in zip(attr_names, mi_attr_msg['entropy_attr_dim'])}
        results['mi_msg_attr_dim'] = {
            name: value for name, value
            in zip(attr_names, mi_attr_msg['mi_msg_attr_dim'])}
        results['vi_msg_attr_dim'] = {
            name: value for name, value
            in zip(attr_names, mi_attr_msg['vi_msg_attr_dim'])}
        results['vi_norm_msg_attr_dim'] = {
            name: value for name, value
            in zip(attr_names, mi_attr_msg['vi_norm_msg_attr_dim'])}
        results['is_msg_attr_dim'] = {
            name: value for name, value
            in zip(attr_names, mi_attr_msg['is_msg_attr_dim'])}
    else:
        unique_objects, categorized_input = torch.unique(
            s_inp, return_inverse=True, dim=0)
        if len(unique_objects) < 200:  # test
            categorized_input = categorized_input.unsqueeze(-1).to(torch.float)
            mi_inp_msg = compute_mi(msg, categorized_input, receiver_vocab_size)
            results['entropy_msg'] = mi_inp_msg['entropy_msg']
            results['entropy_inp'] = mi_inp_msg['entropy_attr']
            results['mi_msg_inp'] = mi_inp_msg['mi_msg_attr']
            results['vi_msg_inp'] = mi_inp_msg['vi_msg_attr']
            results['vi_norm_msg_inp'] = mi_inp_msg['vi_norm_msg_attr']
            results['is_msg_inp'] = mi_inp_msg['vi_msg_attr']
        else:  # train
            results['entropy_msg'] = None  # mi_inp_msg['entropy_msg']
            results['entropy_inp'] = None  # mi_inp_msg['entropy_attr']
            results['mi_msg_inp'] = None  # mi_inp_msg['mi_msg_attr']
            results['vi_msg_inp'] = None  # mi_inp_msg['vi_msg_attr']
            results['vi_norm_msg_inp'] = None  # mi_inp_msg['vi_norm_msg_attr']
            results['is_msg_inp'] = None  # mi_inp_msg['vi_msg_attr']
        mi_cat_msg = compute_mi(msg, attr, receiver_vocab_size)
        results['entropy_cat'] = mi_cat_msg['entropy_attr']
        results['mi_msg_cat'] = mi_cat_msg['mi_msg_attr']
        results['vi_msg_cat'] = mi_cat_msg['vi_msg_attr']
        results['vi_norm_msg_cat'] = mi_cat_msg['vi_norm_msg_attr']
        results['is_msg_cat'] = mi_cat_msg['vi_msg_attr']

    return results


def print_training_results(output_dict):
    def _format(value):
        if value is None:
            return np.nan
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            return f'{value:.2f}'
        elif isinstance(value, dict):
            values = [f'{k}: {_format(v)}' for k, v in value.items()]
            return '\n'.join(values)
        else:
            return value

    flattened_dict = {
        ' '.join((key1, key3)): value3
        for key1, value1 in output_dict.items()
        for key2, value2 in value1.items()
        if key2 == 'results'
        for key3, value3 in value2.items()
    }
    header = ['metric'] + list(flattened_dict.keys())
    metrics = list(list(flattened_dict.values())[0].keys())
    values = [metrics] + [
        [_format(v) for v in col.values()]
        for col in flattened_dict.values()]
    table_dict = {h: v for h, v in zip(header, values)}
    print(tabulate(
        table_dict,
        headers='keys',
        tablefmt='rst',
        maxcolwidths=[24] * len(header),
        # numalign='center',
        # stralign='left',
        disable_numparse=True,
    ))


def crop_messages(messages: torch.Tensor, message_length: Optional[torch.Tensor] = None):
    """
    Given an Interaction object, removes non EOS symbols after the first EOS.
    Used to trim EOSed symbols on validation.
    """
    if messages.dim() == 2:
        if message_length is None:
            message_length = find_lengths(messages)
        size = messages.shape
        not_eosed = (
            torch.unsqueeze(
                torch.arange(0, size[1]), dim=0
            ).expand(size[:2]).to(messages.device)
            < torch.unsqueeze(
                message_length - 1, dim=-1
            ).expand(size[:2])
        )
        messages = torch.where(not_eosed, messages, 0.)

    if messages.dim() == 3:
        msg = messages.argmax(-1)
        lengths = find_lengths(msg).unsqueeze(-1).expand(msg.size())
        positions = torch.arange(msg.size(1)).unsqueeze(0).expand(msg.size())

        nonzero_ids = msg.nonzero()
        nonzero_chunks = nonzero_ids.t().chunk(chunks=2)
        targets = (positions[nonzero_chunks] > lengths[nonzero_chunks] - 1)
        targets = targets.squeeze()
        target_ids = nonzero_ids[targets]

        # if no targets are found, the dimension is 3
        if target_ids.dim() == 2:
            target_chunks = target_ids.t().chunk(chunks=3)
            replacement_probs = torch.zeros_like(messages[0, 0])
            replacement_probs[0] = 1.
            messages[target_chunks] = replacement_probs

        # if not torch.all(torch.eq(messages.argmax(-1), msg)):
        #     check = [(messages[i], messages[i].argmax(-1), msg[i]) for i in range(len(msg)) if not torch.all(torch.eq(messages[i].argmax(-1), msg[i]))]
        #    for item in check:
        #        print(item[0])
        #        print(item[1])
        #        print(item[2])
        #        print('----')

    return messages
