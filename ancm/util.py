import math
import json
import torch
import argparse
import numpy as np
from tabulate import tabulate, SEPARATING_LINE
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
# from torch.distributions import OneHotCategorical

from typing import Optional

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
from ancm.interaction import Interaction

from egg.core.util import find_lengths, get_opts
from egg.core.batch import Batch


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
            obj_sets, target_ids, aux_attributes = zip(*batch)

            r_inputs, labels = np.vstack(np.expand_dims(obj_sets, 0)), np.array(target_ids)
            targets = r_inputs[np.arange(labels.shape[0]), labels]
            aux_attributes = np.vstack(aux_attributes)

            aux = {}
            for attr_name in aux_attributes.dtype.names:
                aux[attr_name] = torch.from_numpy(aux_attributes[attr_name]).float()

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
            drop_last=False,
            shuffle=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=_collate,
            drop_last=False,
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
                drop_last=False,
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
            opts: argparse.Namespace,
            device: Optional[torch.device] = None):

        train_state = game.training
        game.eval()
        device = device if device is not None else common_opts.device

        logs = []
        for i, batch in enumerate(dataset):
            batch = Batch(*batch).to(device)
            with torch.no_grad():
                _, interaction = game(*batch)
                logs.append(interaction)
        logs = Interaction.from_iterable(logs)

        game.train(mode=train_state)

        self.receiver = game.receiver
        self.channel = game.channel
        self.opts = opts

        self.sender_inputs = logs.sender_input
        self.receiver_inputs = logs.receiver_input
        self.labels = logs.labels

        messages = logs.message.int() \
            if opts.mode == 'rf' \
            else logs.message.argmax(-1)
        lengths = logs.message_length.long()  # always works on evaluation
        self.messages = crop_messages(messages, lengths)  # 2 dimensional
        self.message_inputs = crop_messages(logs.message, lengths)
        self.probs = logs.probs
        self.lengths = lengths

        # select receiver output at 1st EOS
        idx = (torch.arange(len(messages)), lengths - 1)
        self.receiver_outputs = logs.receiver_output[idx] \
            if opts.mode == 'rf' \
            else logs.receiver_output.argmax(-1)[idx]

        if opts.channel != 'none':
            messages = logs.message_nn.int() \
                if opts.mode == 'rf' \
                else logs.message_nn.argmax(-1)
            lengths = logs.message_length_nn.long()
            self.messages_nn = crop_messages(messages, lengths)
            self.message_inputs_nn = crop_messages(logs.message_nn, lengths)
            self.probs_nn = logs.probs_nn
            self.lengths_nn = lengths

            idx = (torch.arange(len(messages)), lengths - 1)
            self.receiver_outputs_nn = logs.receiver_output_nn[idx] \
                if opts.mode == 'rf' \
                else logs.receiver_output_nn.argmax(-1)[idx]

        distr_prefixes = {
            k[:k.index('_')]: None
            for k in logs.aux_input if k.startswith('distr')
        }.keys()

        self.attribute_names = [
            k.replace('target_', '') for k in logs.aux_input.keys()
            if not k.startswith('distr')]
        self.target_attributes = torch.cat([
            tensor for k, tensor in logs.aux_input.items()
            if not k.startswith('distr')
        ], dim=-1)
        self.distractor_attributes = [
            torch.cat([
                tensor for k, tensor in logs.aux_input.items()
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

    def get_message_logs(self):
        message_logs = []
        for s_input, msg, msg_nn, r_input, r_output, r_output_nn, label, \
                target_attr, distr_attr in self:

            if self.opts.image_input:
                # For the Obverter dataset, we save object features rather than
                # images (color, shape, position, rotation)
                def attr_repr(attr_dict):
                    attr_strings = [f'{k}: {v}' for k, v in attr_dict.items()]
                    return ', '.join(attr_strings)

                target_vec = attr_repr(target_attr)
                distr_vex = [attr_repr(attr_dict) for attr_dict in distr_attr]
            else:
                # as VISA concepts are sparse binary tensors, we represent each
                # object as indices of features that it does have
                def input_repr(input_tensor):
                    indices = input_tensor.nonzero().squeeze().tolist()
                    return ','.join([str(x) for x in indices])

                target_vec = input_repr(s_input)
                distr_vex = [
                    input_repr(candidate)
                    for i, candidate in enumerate(r_input) if i != label
                ]

            message_log = {
                'target_obj': target_vec,
                'distractor_objs': distr_vex,
                'label': label,
                'message': ','.join([str(x) for x in msg.tolist()]),
                'prediction': r_output,
            }

            if self.opts.channel != 'none':
                message_nn = ','.join([str(x) for x in msg_nn.tolist()])
                message_log.update({
                    'message_no_noise': message_nn,
                    'prediction_no_noise': r_output_nn,
                })

            if not self.opts.image_input:
                message_log.update({
                    'target_cat': target_attr['category'],
                    'distractor_cats': [x['category'] for x in distr_attr],
                })

            message_logs.append(message_log)

        return message_logs

    def get_eval_dict(self):
        opts = self.opts

        results = {}
        keys = ('noise', 'no noise') if opts.channel != 'none' else ('baseline',)
        for key in keys:
            (s_inputs, messages, probs, m_inputs, r_inputs,
             r_outputs, labels, t_attributes, d_attributes, attr_names) \
                = self.get_tensors(key == 'noise')

            # whether to include additional symbols
            vocab_size = self.probs.size(-1) if key == 'noise' \
                else opts.vocab_size

            entropy, length_probs = message_entropy(probs)
            max_entropy = self.channel.max_message_entropy(
                length_probs, key == 'noise')
            unique_targets = len(torch.unique(s_inputs, dim=0))

            # TODO average length add 1?
            results[key] = {
                'samples': len(messages),
                'samples_per_target_obj': len(messages) / unique_targets,
                'unique_target_objects': unique_targets,
                'unique_messages': len(torch.unique(messages, dim=0)),
                'average_length': (
                    torch.arange(probs.size(1)) * length_probs).sum(),
                'actual_vocab_size': torch.unique(messages).numel(),
                'accuracy': (r_outputs == labels).float().mean(),
                'accuracy2': compute_accuracy2(
                    m_inputs, r_inputs, labels, self.receiver, opts),
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
                    'topsim': compute_top_sim(s_inputs, messages),
                })

                # assign a different number to every input vector
                _, input_cat = torch.unique(
                    s_inputs, return_inverse=True, dim=0)
                input_cat = input_cat.unsqueeze(-1).to(torch.float)
                results[key].update({
                    k.replace('attr', 'input'): v for k, v
                    in compute_mi(probs, input_cat, entropy).items()
                })
                results[key].update({
                    k.replace('attr', 'category'): v
                    for k, v in compute_mi(probs, t_attributes, entropy).items()
                })

            # convert tensors to numeric
            results[key] = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in results[key].items()
            }

        return results


def print_training_results(dump_dict):
    def format_values(value):
        return round(value, 2) if isinstance(value, float) else value

    flattened_dict = {
        '/'.join((key1, key2)) if key2 != 'baseline' else key1: values
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
    sep_list = ['accuracy'] + [m for m in measures if 'entropy' in m][:1]
    while sep_list:
        m = sep_list.pop(-1)
        if m not in ('entropy_msg', 'entropy_max'):
            values.insert(measures.index(m), SEPARATING_LINE)
    print(tabulate(
        values,
        header,
        tablefmt='rst',
        numalign='right',
        stralign='left',
    ))
