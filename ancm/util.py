import json
import torch
import argparse
import numpy as np
from tabulate import tabulate, SEPARATING_LINE
from torch.utils.data import Dataset, DataLoader
from egg.zoo.language_bottleneck import intervention

from typing import Optional

from ancm.eval import (
    message_entropy,
    min_message_entropy,
    compute_max_rep,
    compute_accuracy2,
    compute_top_sim,
    compute_mi,
    compute_posdis,
    compute_bosdis,
)
from ancm.interaction import Interaction

from egg.core.util import find_lengths, get_opts
from egg.core.batch import Batch


common_opts = get_opts()


class ObjectDataset(Dataset):
    def __init__(
            self, obj_sets, targets, attributes, n_additional_targets=2, seed=42):
        n_objects = obj_sets.shape[1]
        self.n_targets = 1 if n_additional_targets is None \
            else 1 + n_additional_targets \

        # save object sets
        self.obj_sets = obj_sets

        # get additional targets & save them as labels
        if n_additional_targets is not None:
            rng = np.random.default_rng(seed=seed)
            candidates = np.tile(np.arange(n_objects), (len(obj_sets), 1))
            candidates = candidates[candidates != targets.reshape(-1, 1)]
            candidates = candidates.reshape(len(obj_sets), -1)
            candidates = rng.permuted(candidates, axis=1)
            additional_targets = candidates[:, :n_additional_targets]
            target_pos = np.hstack([targets.reshape(-1, 1), additional_targets])
        else:
            target_pos = targets.reshape(-1, 1)

        # save additional labels
        self.labels = target_pos.reshape(-1, order='F')

        # save attributes
        obj_set = np.tile(np.arange(len(obj_sets)), self.n_targets)
        distr_pos = np.tile(
            np.arange(n_objects),
            (self.n_targets * len(obj_sets), 1))
        distr_pos = distr_pos[distr_pos != target_pos.reshape(-1, 1)]
        distr_pos = distr_pos.reshape(len(target_pos) * self.n_targets, -1)
        col = np.hstack([target_pos.reshape(-1, 1), distr_pos])
        row = np.tile(np.arange(len(col)), (col.shape[1], 1)).T
        attr_array = np.hstack([
            attributes[name][obj_set][row, col]
            for name in attributes.dtype.names
        ])
        dtype = [
            item
            for name in attributes.dtype.names
            for item in [(f'target_{name}', np.int64)]
            + [(f'distr{i}_{name}', np.int64) for i in range(n_objects - 1)]
        ]
        self.attributes = np.array(list(map(tuple, attr_array)), dtype=dtype)

        # idx = len(obj_sets)
        # x, y = 0, 10
        # print('ATTRIBUTES CHECK')
        # print(self.attributes[x:y])
        # print(self.attributes[idx + x:idx + y])
        # print(self.attributes[2 * idx + x : 2 * idx + y])
        # print('LABELS CHECK')
        # print(self.labels[x:y])
        # print(self.labels[idx + x:idx + y])
        # print(self.labels[2 * idx + x : 2 * idx + y])

    def __len__(self):
        return len(self.obj_sets) * self.n_targets

    def __getitem__(self, idx):
        obj_set = self.obj_sets[idx % len(self.obj_sets)]
        label = self.labels[idx]
        attributes = self.attributes[idx]
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
        train = data["train"], data["train_targets"], data["train_attributes"]
        eval_train = data["eval_train"], \
            data["eval_train_targets"], data["eval_train_attributes"]
        eval_test = data["eval_test"], \
            data["eval_test_targets"], data["eval_test_attributes"]

        self.eval_train_mapping = data["eval_train_attribute_mapping"]
        self.eval_test_mapping = data["eval_test_attribute_mapping"]

        self._n_features = train[0].shape[-1]
        self.train_samples = len(train[0])
        self.eval_train_samples = len(eval_train[0])
        self.eval_test_samples = len(eval_test[0])
        self.n_distractors = train[0].shape[1] - 1

        opts.train_samples = self.train_samples
        opts.eval_train_samples = self.eval_train_samples
        opts.eval_test_samples = self.eval_test_samples
        opts.n_distractors = self.n_distractors
        opts.n_features = self.n_features

        train_dataset = ObjectDataset(*train, opts.n_targets)
        eval_train_dataset = ObjectDataset(*eval_train, opts.n_targets)
        eval_test_dataset = ObjectDataset(*eval_test, opts.n_targets)

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
        eval_train_dataloader = DataLoader(
            eval_train_dataset,
            batch_size=self.batch_size,
            collate_fn=_collate,
            drop_last=False,
            shuffle=False)
        eval_test_dataloader = DataLoader(
            eval_test_dataset,
            batch_size=self.batch_size,
            collate_fn=_collate,
            drop_last=False,
            shuffle=False)

        return train_dataloader, eval_train_dataloader, eval_test_dataloader


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

    if opts.mode != 'gs' or opts.temperature_lr is None:
        return optimizer([
            {"params": game.sender.parameters(), "lr": opts.sender_lr},
            {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
        ])
    else:
        sender_params = [
            param for param in game.sender.parameters()
            if param is not game.sender.temperature
        ]

        return optimizer([
            {"params": game.sender.temperature, "lr": opts.temperature_lr},
            {"params": sender_params, "lr": opts.sender_lr},
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

    def get_message_logs(self, mapping):
        message_logs = []

        def map(key, value):
            mapped = mapping[key][value]
            if isinstance(mapped, np.bytes_):
                return np.str_(mapped.astype('U'))
            return mapped

        for s_input, msg, msg_nn, r_input, r_output, \
                r_output_nn, label, target_attr, distr_attr in self:

            if self.opts.image_input:
                # For the Obverter dataset, we save object features rather than
                # images (color, shape, position, rotation)
                def attr_repr(attr_dict):
                    attr_mapped = {k: map(k, v) for k, v in attr_dict.items()}
                    attr_str = [f'{k}: {v}' for k, v in attr_mapped.items()]
                    return ', '.join(attr_str)

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
                    'target_cat': map('category', target_attr['category']),
                    'distractor_cats':
                        [map('category', x['category']) for x in distr_attr],
                })

            message_logs.append(message_log)

        return message_logs

    @staticmethod
    def get_attribute(attribute_dict, key):
        keys = [
            k.replace('target_', '') for k in attribute_dict
            if k.startswith('target_')]
        n_distractors = len(attribute_dict) // len(keys) - 1
        prefixes = ['target_'] + [f'distr{i}_' for i in range(n_distractors)]
        values = [attribute_dict[prefix + key] for prefix in prefixes]
        return torch.cat(values, dim=1)

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

            if opts.image_input:
                idx = torch.tensor(
                    [attr_names.index('shape'), attr_names.index('color')]
                )
                obj_repr = torch.stack(
                    [t_attributes[:, idx]] + [a[:, idx] for a in d_attributes],
                    dim=1,
                )
                print('repr', obj_repr.shape)
                min_entropy, n_uniq_samples = min_message_entropy(
                    r_inputs, labels, obj_repr)
            else:
                min_entropy, n_uniq_samples = \
                    min_message_entropy(r_inputs, labels)

            unique_messages, categorized_messages = \
                torch.unique(messages, dim=0, return_inverse=True)
            n_uniq_targets = len(torch.unique(s_inputs, dim=0))
            n_uniq_messages = len(unique_messages)

            results[key] = {
                'samples': len(messages),
                'unique_msg': n_uniq_messages,
                'unique_samples': n_uniq_samples,
                'unique_target_objs': n_uniq_targets,
                'samples_per_target_obj': len(messages) / n_uniq_targets,
                'unique_target_objs_per_msg': n_uniq_targets / n_uniq_messages,
                'unique_samples_per_target_obj':
                    n_uniq_samples / n_uniq_targets,
                'unique_samples_cat': None,
                'unique_cat': None,
                'unique_samples_per_target_cat': None,
                'unique_cats_per_msg': None,
                'samples_per_cat': None,
                'average_length': (
                    torch.arange(probs.size(1)) * length_probs
                ).sum(),  # does not include additional EOS
                'actual_vocab_size': torch.unique(messages).numel(),
                'accuracy': (r_outputs == labels).float().mean(),
                'accuracy_symbol_removal': compute_accuracy2(
                    m_inputs, r_inputs, labels, self.receiver, opts),
                'max_rep': compute_max_rep(messages).mean().item(),
                'redundancy': 1 - entropy / max_entropy,
                'topsim': None,
                'posdis': None,
                'bosdis': None,
                'entropy_msg': entropy,
                'entropy_msg_as_a_whole':
                    intervention.entropy(categorized_messages),
                'entropy_max': max_entropy,
                'entropy_min': min_entropy,
            }

            if opts.image_input:
                for k in results[key]:
                    if 'cat' in k:
                        del results[key]

                results[key].update({
                    'topsim': compute_top_sim(t_attributes, messages),
                    'posdis': compute_posdis(t_attributes, messages),
                    'bosdis': compute_bosdis(
                        t_attributes, messages, vocab_size),
                })
                mi_attr = compute_mi(probs, t_attributes, entropy)
                results[key]['entropy_attr'] = mi_attr['entropy_attr']
                for i, name in enumerate(attr_names):
                    results[key].update({
                        k.replace('attr_dim', name): v[i]
                        for k, v in mi_attr.items() if 'attr_dim' in k
                    })
            else:
                del results[key]['posdis'], results[key]['bosdis']
                # print(t_attributes.shape)
                # print(d_attributes[0].shape)
                category = torch.cat([t_attributes] + d_attributes, dim=-1)
                # category = torch.cat([
                #     item['category'] for item in [t_attributes] + d_attributes
                # ], dim=-1)
                min_entropy_cat, n_uniq_samples_cat = min_message_entropy(
                    r_inputs, labels, category)
                n_uniq_cat = len(torch.unique(category))
                results[key].update({
                    'unique_samples_cat': n_uniq_samples_cat,
                    'unique_cat': n_uniq_cat,
                    'samples_per_cat': len(messages) / n_uniq_cat,
                    'unique_cats_per_msg': n_uniq_cat / n_uniq_messages,
                    'unique_samples_per_target_cat':
                        n_uniq_samples_cat / n_uniq_cat,
                    'entropy_min_cat': min_entropy_cat,
                    'topsim': compute_top_sim(s_inputs, messages),
                })

                # assign a different number to every input vector
                _, inp = torch.unique(
                    s_inputs, return_inverse=True, dim=0)
                inp = inp.unsqueeze(-1).to(torch.float)
                results[key].update({
                    k.replace('attr', 'input'): v for k, v
                    in compute_mi(probs, inp, entropy).items()
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
        if m not in ('entropy_msg', 'entropy_max', 'entropy_whole_msg'):
            values.insert(measures.index(m), SEPARATING_LINE)
    print(tabulate(
        values,
        header,
        tablefmt='rst',
        numalign='right',
        stralign='left',
    ))
