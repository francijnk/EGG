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
from ancm.eval import (
    message_entropy,
    compute_max_rep,
    compute_accuracy2,
    compute_top_sim,
    compute_mi,
    compute_posdis,
    compute_bosdis,
    # binary_entropy,
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
    if opts.optim.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop
    elif opts.optim.lower() == 'adam':
        optimizer = torch.optim.Adam
    else:
        raise ValueError('Optimizer must be either RMSprop or Adam')

    return optimizer([
        {"params": game.sender.parameters(), "lr": opts.sender_lr},
        {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
    ])


# def get_one_hots(self, symbols, vocab_size):
#     size = (*symbols.size(), vocab_size)
#     one_hots = torch.zeros(size).to(symbols.device).view(-1, vocab_size)
#     one_hots.scatter_(1, symbols.view(-1, 1), 1)
#     return one_hots.view(size)


def crop_messages(messages: torch.Tensor, lengths: Optional[torch.Tensor] = None):
    """
    Given an Interaction object, removes non EOS symbols after the first EOS.
    Used to trim EOSed symbols on validation.
    """
    symbols = messages if messages.dim() == 2 else messages.argmax(-1)
    lengths = lengths if lengths is not None else find_lengths(symbols)

    not_eosed = (
        torch.unsqueeze(
            torch.arange(0, symbols.size(1)),
            dim=0).expand(symbols.size()[:2]).to(symbols.device)
        < torch.unsqueeze(lengths - 1, dim=-1).expand(symbols.size()[:2])
    )

    cropped_symbols = torch.where(not_eosed, symbols, 0)

    if messages.dim() == 2:  # Reinforce
        return cropped_symbols

    else:  # GS
        cropped_probs = torch.zeros_like(messages).view(-1, messages.size(2))
        cropped_probs.scatter_(1, cropped_symbols.view(-1, 1), 1)
        return cropped_probs.view(messages.size())


class Dump:
    def __init__(
            self,
            game: torch.nn.Module,
            dataset: torch.utils.data.DataLoader,
            device: Optional[torch.device] = None):

        train_state = game.training
        game.eval()
        device = device if device is not None else common_opts.device

        sender_inputs, receiver_inputs, labels = [], [], []
        messages, messages_nn = [], []
        message_inputs, message_inputs_nn = [], []
        symbol_probs, symbol_probs_nn = [], []
        receiver_outputs, receiver_outputs_nn = [], []
        message_lengths, message_lengths_nn = [], []
        attributes = defaultdict(list)
        # attributes, channel_output = defaultdict(list), defaultdict(list)

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

            _message = sender_output[0]
            _probs = sender_output[-1]

            # pass messages and symbol probabilities through the channel
            message, message_nn, probs, probs_nn, channel_aux = \
                game.channel(_message, _probs)

            # append EOS to each message & symbol probability tensor
            eos = torch.zeros_like(message[:, :1, :])
            eos[:, 0, 0] = 1
            probs = torch.cat([probs, eos], dim=1)
            probs_nn = torch.cat([probs_nn, eos], dim=1)
            eos = eos if message.dim() == 3 else torch.zeros_like(message[:, 0])
            message = torch.cat([message, eos], dim=1)
            message_nn = torch.cat([message_nn, eos], dim=1)

            # remove symbols preceded by EOS
            symbols = message if message.dim() == 2 else message.argmax(-1)
            lengths = find_lengths(symbols)
            symbols = crop_messages(symbols, lengths)
            message = crop_messages(message, lengths)

            symbols_nn = message_nn.int() if message_nn.dim() == 2 \
                else message_nn.argmax(-1)
            lengths_nn = find_lengths(symbols_nn)
            symbols_nn = crop_messages(symbols_nn, lengths_nn)
            message_nn = crop_messages(message_nn, lengths_nn)

            messages.extend(symbols)  # always 2 dimensions
            message_inputs.extend(message)  # 3 dimensions for GS
            symbol_probs.extend(probs)
            message_lengths.extend(lengths)

            # save receiver outputs at first EOS
            if not isinstance(game.channel, NoChannel):
                messages_nn.extend(symbols_nn)
                message_inputs_nn.extend(message_nn)
                symbol_probs_nn.extend(probs_nn)
                message_lengths_nn.extend(lengths_nn)

                _message = torch.cat([message, message_nn], dim=0)
                _receiver_input = torch.cat(
                    [receiver_input, receiver_input], dim=0)
                with torch.no_grad():
                    r_output = game.receiver(_message, _receiver_input)
                if r_output.numel() != len(r_output):
                    r_output = r_output.argmax(-1)
                idx = (torch.arange(0, len(message)), lengths - 1)
                idx_nn = (idx[0] + len(message), lengths_nn - 1)
                receiver_outputs.extend(r_output[idx])
                receiver_outputs_nn.extend(r_output[idx_nn])
            else:
                with torch.no_grad():
                    r_output = game.receiver(message, receiver_input)
                if r_output.numel() != len(r_output):
                    r_output = r_output.argmax(-1)
                idx = (torch.arange(0, len(message)), lengths - 1)
                receiver_outputs.extend(r_output[idx])  # output at 1st EOS

        # channel_output = {
        #     k: torch.cat(v, dim=0) if v[0] is not None
        #     else [None for _ in range(len(v))]
        #     for k, v in channel_output.items()
        # }  # TODO remove if not used for DeletionChannel or max. entropy

        game.train(mode=train_state)

        self.sender_inputs = torch.stack(sender_inputs)
        self.receiver_inputs = torch.stack(receiver_inputs)
        self.labels = torch.stack(labels)
        self.messages = torch.stack(messages)
        self.probs = torch.stack(symbol_probs)
        self.message_inputs = torch.stack(message_inputs)
        self.lengths = torch.stack(message_lengths)
        self.receiver_outputs = torch.stack(receiver_outputs)
        # self.channel_output = channel_output
        if messages_nn:
            self.messages_nn = torch.stack(messages_nn)
            self.probs_nn = torch.stack(symbol_probs_nn)
            self.message_inputs_nn = torch.stack(message_inputs_nn)
            self.lengths_nn = torch.stack(message_lengths_nn)
            self.receiver_outputs_nn = torch.stack(receiver_outputs_nn)
        else:  # no noise
            self.messages_nn = self.messages
            self.probs_nn = self.probs
            self.message_inputs_nn = self.message_inputs
            self.lengths_nn = self.lengths
            self.receiver_outputs_nn = self.receiver_outputs

        attributes = {k: torch.cat(v, dim=0) for k, v in attributes.items()}
        distr_prefixes = {
            k[:k.index('_')]: None for k in attributes.keys() if k.startswith('distr')
        }.keys()

        self.attribute_names = [
            k.replace('target_', '') for k in attributes.keys()
            if not k.startswith('distr')]
        self.target_attributes = torch.cat([
            tensor for k, tensor in attributes.items()
            if not k.startswith('distr')
        ], dim=-1)
        self.distractor_attributes = [
            torch.cat([
                tensor for k, tensor in attributes.items()
                if k.startswith(prefix)
            ], dim=-1)
            for prefix in distr_prefixes
        ]

    def __len__(self):
        return len(self.sender_inputs)

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self.sender_inputs[i].int(),
                self.messages[i, :self.lengths[i]].int(),
                self.messages_nn[i, :self.lengths_nn[i]].int(),
                self.receiver_inputs[i].int(),
                self.receiver_outputs[i].int().item(),
                self.receiver_outputs_nn[i].int().item(),
                self.labels[i].int().item(),
                {
                    key: self.target_attributes[i, j].int().item()
                    for j, key in enumerate(self.attribute_names)
                },
                [
                    {
                        key: attribute_tensor[i, j].int().item()
                        for j, key in enumerate(self.attribute_names)
                    }
                    for attribute_tensor in self.distractor_attributes
                ],
                # {
                #     k: self.channel_output[k][i].item()
                #     if self.channel_output[k][i].numel() == 1
                #     else self.channel_output[k][i]
                #     for k in self.channel_output
                # },
            )

    def get_tensors(self, noise=True):
        if noise:
            return (
                self.sender_inputs, self.messages, self.probs,
                self.message_inputs, self.receiver_inputs,
                self.receiver_outputs, self.labels,
                self.target_attributes, self.distractor_attributes,
                self.attribute_names
            )

        else:
            return (
                self.sender_inputs, self.messages_nn, self.probs_nn,
                self.message_inputs_nn, self.receiver_inputs,
                self.receiver_outputs_nn, self.labels,
                self.target_attributes, self.distractor_attributes,
                self.attribute_names
            )

    def get_results_dict(self, game, opts, target_counts):
        results = {}

        keys = ('noise', 'no noise') if opts.channel != 'none' else ('baseline',)
        for key in keys:
            (sender_inputs, messages, probs, message_inputs, receiver_inputs,
             receiver_outputs, labels, t_attributes, d_attributes, attr_names) \
                = self.get_tensors(key == 'noise')

            # whether to include additional symbols
            vocab_size = self.probs.size(-1) if key == 'noise' else opts.vocab_size

            entropy, length_probs = message_entropy(probs)
            max_entropy = game.channel.max_message_entropy(
                length_probs, key == 'noise')

            results[key] = {
                'samples': sum(target_counts.values()),
                'samples_per_target_obj':
                    sum(target_counts.values()) / len(target_counts),
                'unique_messages': len(torch.unique(messages, dim=0)),
                'unique_target_objects': len(target_counts),
                'actual_vocab_size': torch.unique(messages).numel(),
                'accuracy': (receiver_outputs == labels).float().mean(),
                'accuracy2': compute_accuracy2(
                    message_inputs, receiver_inputs, labels, game.receiver, opts),
                'max_rep': compute_max_rep(messages).mean().item(),
                'entropy_msg': entropy,
                'entropy_max': max_entropy,
                'redundancy': 1 - entropy / max_entropy,
            }

            if opts.image_input:
                results[key].update({
                    'topsim': compute_top_sim(t_attributes, messages),
                    'posdis': compute_posdis(t_attributes, messages),
                    'bosdis': compute_bosdis(t_attributes, messages, vocab_size),
                })
                mi_attr = compute_mi(probs, t_attributes, entropy)
                results[key]['entropy_attr'] = mi_attr['entropy_attr']
                for i, name in enumerate(attr_names):
                    results[key].update({
                        k.replace('attr_dim', name): v[i]
                        for k, v in mi_attr.items() if 'attr_dim' in k
                    })
            else:
                results[key].update({
                    'topsim': compute_top_sim(sender_inputs, messages),
                    'topsim_cat': compute_top_sim(t_attributes, messages),
                })

                # assign a different number to every input vector
                _, input_cat = torch.unique(
                    sender_inputs, return_inverse=True, dim=0)
                input_cat = input_cat.unsqueeze(-1).to(torch.float)
                results[key].update({
                    k.replace('attr', 'input'): v for k, v
                    in compute_mi(probs, input_cat, entropy).items()
                })
                results[key].update({
                    k.replace('attr', 'category'): v
                    for k, v in compute_mi(probs, t_attributes, entropy).items()
                })

                # TODO cross entropy / KLD between the training and test set?

            # convert tensors to numeric
            results[key] = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in results[key].items()
            }

        return results


def print_training_results(dump_dict):
    def format_values(value):
        # if value is None:
        #     return np.nan
        # if isinstance(value, int):
        #     return value
        if isinstance(value, float):
            return f'{value:.2f}'
        # elif isinstance(value, dict):
        #     values = [f'{k}: {_format(v)}' for k, v in value.items()]
        #     return '\n'.join(values)
        else:
            return value

    flattened_dict = {
        ' '.join((key1, key2)) if key2 != 'baseline' else key1: values
        for key1, dataset_dict in dump_dict.items()
        if key1 in ('train', 'test')
        for key2, values in dataset_dict['evaluation'].items()
    }

    header = ['measure'] + list(flattened_dict.keys())
    measures = list(list(flattened_dict.values())[0].keys())
    values = [
        [m] + [format_values(col[m]) for col in flattened_dict.values()]
        for m in measures
    ]
    print(tabulate(
        values,
        header,
        tablefmt='rounded_outline',
        numalign='right',
        stralign='left',
    ))
