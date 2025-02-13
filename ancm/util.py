import math
import json
import torch
import numpy as np
from tabulate import tabulate
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
# from torch.distributions import OneHotCategorical

from typing import Optional

from egg.core.util import find_lengths, move_to, get_opts

# from ancm.archs import tensor_binary_entropy
from ancm.metrics import (
    # compute_conceptual_alignment,
    compute_max_rep,
    # compute_redundancy,
    # compute_adjusted_redundancy,
    compute_accuracy2,
    compute_top_sim,
    compute_mi,
    compute_posdis,
    compute_bosdis,
    # maximize_sequence_entropy,
    binary_entropy,
)
from ancm.archs import NoChannel

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
    def __init__(
            self, sender_inputs, messages, messages_nn, receiver_inputs,
            receiver_outputs, receiver_outputs_nn, labels, attributes, channel_output):
        self.sender_inputs = torch.stack(sender_inputs)
        self.messages = crop_messages(torch.stack(messages))
        self.messages_nn = crop_messages(torch.stack(messages_nn)) \
            if isinstance(messages_nn[0], torch.Tensor) else self.messages
        self.receiver_inputs = torch.stack(receiver_inputs)
        self.labels = torch.stack(labels)
        self.channel_output = channel_output

        attributes = {k: torch.cat(v, dim=0) for k, v in attributes.items()}

        self.target_attributes = {
            k.replace('target_', ''): v for k, v in attributes.items()
            if not k.startswith('distr')}
        self.distractor_attributes = {
            k: v for k, v in attributes.items()
            if k.startswith('distr')}

        if self.messages.dim() == 3:
            self.lengths = find_lengths(self.messages.argmax(-1))
            self.lengths_nn = find_lengths(self.messages_nn.argmax(-1))
            # self.receiver_outputs = torch.stack(receiver_outputs)
            self.receiver_outputs = torch.stack([
                receiver_output[self.lengths[i] - 1].argmax(-1)
                for i, receiver_output in enumerate(receiver_outputs)
            ], dim=0)
            self.receiver_outputs_nn = torch.stack([
                receiver_output[self.lengths[i] - 1].argmax(-1)
                for i, receiver_output in enumerate(receiver_outputs_nn)
            ], dim=0) \
                if isinstance(receiver_outputs_nn[0], torch.Tensor) \
                else self.receiver_outputs
            self.strings = self.messages.argmax(-1)
            self.strings_nn = self.messages_nn.argmax(-1)
        else:
            self.lengths = find_lengths(self.messages)
            self.receiver_outputs = torch.stack(receiver_outputs)
            self.receiver_outputs_nn = torch.stack(receiver_outputs_nn)
            self.strings = self.messages
            self.strings = self.messages_nn

    def __len__(self):
        return len(self.sender_inputs)

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self.sender_inputs[i],
                self.strings[i, :self.lengths[i]],
                self.strings_nn[i, :self.lengths_nn[i]],
                self.receiver_inputs[i],
                self.receiver_outputs[i].item(),
                self.receiver_outputs_nn[i].item(),
                self.labels[i].int().item(),
                {
                    k: self.target_attributes[k][i].int().item()
                    for k in self.target_attributes
                },
                [
                    {k: self.distractor_attributes[k][i].int().item()}
                    for k in self.distractor_attributes
                ],
                {
                    k: self.channel_output[k][i].item()
                    if self.channel_output[k][i].numel() == 1
                    else self.channel_output[k][i]
                    for k in self.channel_output
                },
            )

    def get_tensors(self):
        if self.messages.dim() == 3:
            messages = self.messages.argmax(-1)
            messages_nn = self.messages_nn.argmax(-1)
        else:
            messages, messages_nn = self.messages, self.messages_nn
        target_attributes = torch.cat(list(self.target_attributes.values()), dim=-1)
        distractor_attributes = torch.cat(list(self.distractor_attributes.values()), dim=-1)
        # distractor_attributes = [
        #     [torch.cat(a_tensors, dim=0) for a_tensors in self.distractor_attributes.values()]
        #     for attributes in self.distractor_attributes]
        attribute_names = list(self.target_attributes.keys())

        return self.sender_inputs, messages, messages_nn, \
            self.receiver_inputs, self.receiver_outputs, \
            self.receiver_outputs_nn, self.labels, target_attributes, \
            distractor_attributes, attribute_names, self.channel_output


def dump_sender_receiver(
        game: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        max_len: int,
        vocab_size: int,
        mode: str,
        device: Optional[torch.device] = None):
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param max_len: max message length
    :param vocab_size: vocab size, without channel-specific additional symbols
    :param device: device (e.g. 'cuda') to be used
    :return:
    """
    train_state = game.training
    game.eval()

    device = device if device is not None else common_opts.device

    sender_inputs, receiver_inputs, labels = [], [], []
    messages, messages_nn = [], []
    receiver_outputs, receiver_outputs_nn = [], []
    attributes, channel_output = defaultdict(list), defaultdict(list)

    for i, batch in enumerate(dataset):
        sender_input = move_to(batch[0], device)
        sender_inputs.extend(sender_input)
        labels.extend(batch[1])
        receiver_input = move_to(batch[2], device)
        receiver_inputs.extend(receiver_input)
        for key, val in batch[3].items():
            attributes[key].append(val)
        with torch.no_grad():
            sender_output = game.sender(sender_input)
        message = sender_output[0]
        symbol_entropy = sender_output[2]

        if mode == 'rf':  # Add noise to the message
            raise NotImplementedError
            # message = crop_messages(message)
            # message = game.channel(message, entropy_nn, apply_noise=apply_noise)

        elif mode == 'gs':
            message, message_nn, channel_dict = game.channel(
                message, entropy=symbol_entropy)
            if True:  # game.training:
                not_eosed_before = torch.ones(message.size(0)).to(device)
                not_eosed_before_nn = not_eosed_before.clone().detach()
                prefix_entropy = torch.zeros_like(not_eosed_before)
                prefix_entropy_nn = torch.zeros_like(not_eosed_before)

                symbols = []
                for step in range(message.size(1)):
                    eos_mask = message[:, step, 0]
                    add_mask = eos_mask * not_eosed_before
                    add_mask_nn = message_nn[:, step, 0].detach() \
                        * not_eosed_before_nn
                    symbol_probs = message[:, step]
                    symbol_probs[:, 1:] *= not_eosed_before.unsqueeze(-1)
                    eos = torch.zeros_like(symbol_probs)
                    eos[:, 0] = 1
                    symbol_probs = torch.where(
                        symbol_probs.sum(-1, keepdim=True) > 0,
                        symbol_probs / symbol_probs.sum(-1, keepdim=True),
                        eos)
                    symbols.append(symbol_probs)

                    channel_dict['length_probs'][:, step] = add_mask.detach()

                    # symbols = symbol_probs.argmax(-1)
                    # TODO we might want to take argmax here for during training
                    # distr = OneHotCategorical(probs=symbol_probs)  # 32x10 = 1 smb
                    # symbols.append(distr.sample())

                    channel_dict['length_probs'][:, step] = add_mask.detach()
                    # aggregate message entropy
                    channel_dict['entropy_msg'] = channel_dict['entropy_msg'] \
                        + torch.where(
                            add_mask > 1e-5,
                            add_mask.detach() * prefix_entropy,
                            0)

                    # entropy of the symbol, assuming it is not eos
                    # (the furmula exploits decomposability of entropy)
                    prefix_entropy += (
                        channel_dict['entropy_smb'][:, step]
                        - game.channel.tensor_binary_entropy(eos_mask.detach())
                    ) / (1 - eos_mask.detach())

                    not_eosed_before = not_eosed_before * (1.0 - eos_mask)

                    # TODO make sure this part is the same as in archs.py if something changes there
                    channel_dict['entropy_msg_nn'] = channel_dict['entropy_msg_nn'] \
                        + torch.where(
                            add_mask_nn > 1e-5,
                            add_mask_nn.detach() * prefix_entropy_nn,
                            0)
                    prefix_entropy_nn += (
                        channel_dict['entropy_smb_nn'][:, step]
                        - game.channel.tensor_binary_entropy(message_nn[:, step, 0])
                    ).detach() / (1 - eos_mask.detach())

                    not_eosed_before_nn = (
                        not_eosed_before_nn * (1.0 - message_nn[:, step, 0])
                    ).detach()

                # adjust message entropy to cover message length variability
                # exclude appended EOS from symbol entropy and compute redundancy
                game.channel.update_values(channel_dict)
                for k in (
                    'entropy_msg',
                    'entropy_msg_nn',
                    'entropy_smb',
                    'entropy_smb_nn',
                    'redundancy_msg',
                    'redundancy_smb',
                    'max_entropy',
                ):
                    channel_output[k].append(channel_dict[k])
                message = torch.stack(symbols).permute(1, 0, 2)

            else:
                for k in (
                    'entropy_msg',
                    'entropy_msg_nn',
                    'entropy_smb',
                    'entropy_smb_nn',
                    'redundancy_msg',
                    'redundancy_smb',
                    'max_entropy',
                ):
                    channel_output[k].append(None)

        if isinstance(game.channel, NoChannel):
            receiver_output = game.receiver(message, receiver_input)

            messages.extend(message)
            messages_nn.extend([None for _ in range(len(messages))])
            receiver_outputs.extend(receiver_output)
            receiver_outputs_nn.extend([None for _ in range(len(messages))])
        else:
            # compute receiver outputs for messages without noise
            message_joined = torch.cat([message, message_nn], dim=0)
            receiver_input_joined = torch.cat([receiver_input, receiver_input], dim=0)
            with torch.no_grad():
                receiver_output_joined = game.receiver(
                    message_joined,
                    receiver_input_joined)
            receiver_output = receiver_output_joined[:len(message)]
            receiver_output_nn = receiver_output_joined[len(message):]

            messages.extend(message)
            messages_nn.extend(message_nn)
            receiver_outputs.extend(receiver_output)
            receiver_outputs_nn.extend(receiver_output_nn)

    channel_output = {
        k: torch.cat(v, dim=0) if v[0] is not None
        else [None for _ in range(len(v))]
        for k, v in channel_output.items()
    }
    # channel_dict = game.channel.update_values(channel_dict)
    # channel_dict = {k: v if game.training else None for k, v in channel_dict.items()}

    game.train(mode=train_state)

    return Dump(
        sender_inputs,
        messages,
        messages_nn,
        receiver_inputs,
        receiver_outputs,
        receiver_outputs_nn,
        labels,
        attributes,
        channel_output
    )


def get_results_dict(dump, receiver, opts, unique_dict):
    s_inp, msg, msg_nn, r_inp, r_out, r_out_nn, labels, attr, distr_attr, \
        attr_names, ch_out = dump.get_tensors()

    receiver_vocab_size = opts.vocab_size + 1 if opts.channel == 'erasure' \
        else opts.vocab_size
    actual_vocab = set(int(s) for m in msg for s in m.tolist())

    # receiver_outputs = move_to(receiver_outputs, device)
    # labels = move_to(labels, device)

    results = {
        'samples': len(labels),
        'samples_per_target_obj': len(labels) / len(unique_dict.keys()),
        'accuracy': (r_out == labels).float().mean().item(),
        'accuracy_nn': (r_out_nn == labels).float().mean().item(),
        'accuracy2': compute_accuracy2(dump, receiver, opts),
        'unique_messages': len(torch.unique(msg, dim=0)),
        'unique_target_objects': len(unique_dict.keys()),
        'actual_vocab_size': len(actual_vocab),
        'entropy_msg': ch_out['entropy_msg'].mean().item(),
        'redundancy_msg': ch_out['redundancy_msg'].mean().item(),
        'redundancy_msg_v2': (
            ch_out['entropy_msg'].mean()
            / ch_out['max_entropy'].mean()
        ).item(),
        'entropy_smb': ch_out['entropy_smb'].mean().item(),
        'redundancy_smb': ch_out['redundancy_smb'].mean().item(),
        # 'redundancy': compute_redundancy(msg, receiver_vocab_size, channel, error_prob),
        # 'redundancy_adj': compute_adjusted_redundancy(
        #     msg, channel, error_prob, torch.arange(receiver_vocab_size)),
        # 'redundancy_adj_voc': compute_adjusted_redundancy(
        #     msg, channel, error_prob, actual_vocab),
        'max_rep': compute_max_rep(msg).mean().item(),
    }

    def maxent_smb(channel, p, vocab_size):
        if channel is None or channel == 'symmetric':
            return np.log2(vocab_size)
        elif channel == 'erasure':
            return binary_entropy(p) + (1 - p) * math.log2(vocab_size)

    # update_dict_names = lambda dct, sffx: {f'{k}_{sffx}': v for k, v in dct.items()}
    results['redund_msg'] = ch_out['redundancy_msg'].mean().item()
    results['redund_smb'] = ch_out['redundancy_smb'].mean().item()
    # results.update(compute_mi(channel_dict['entropy_msg'].mean(), attr))
    if opts.image_input:
        results['topographic_rho'] = compute_top_sim(attr, msg)
        results['posdis'] = compute_posdis(attr, msg)
        results['bosdis'] = compute_bosdis(attr, msg, receiver_vocab_size)
    else:
        results['topographic_rho'] = compute_top_sim(s_inp, msg)
        results['topographic_rho_category'] = compute_top_sim(attr, msg)
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
        dataset_key: results
        for dataset_key, dataset_dict in output_dict.items()
        for key, results in dataset_dict.items()
        if key == 'results'
    }

    header = ['metric'] + list(flattened_dict.keys())
    header = ['metric'] + list(flattened_dict.keys())
    metrics = list(list(flattened_dict.values())[0].keys())
    values = [metrics] + [
        [_format(v) for v in col.values()]
        for col in flattened_dict.values()]
    table_dict = {h: v for h, v in zip(header, values)}
    # for h, v in table_dict.items():
    #     print('--')
    #     print(h)
    #     for val in v:
    #         print(v)
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
        #     check = [(messages[i], messages[i].argmax(-1), msg[i]) for i in range(len(msg))
        # if not torch.all(torch.eq(messages[i].argmax(-1), msg[i]))]
        #    for item in check:
        #        print(item[0])
        #        print(item[1])
        #        print(item[2])
        #        print('----')

    return messages
