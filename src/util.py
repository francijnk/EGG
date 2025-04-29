import json
import torch
import argparse
import numpy as np
from tabulate import tabulate, SEPARATING_LINE
from torch.utils.data import Dataset, DataLoader

from typing import Optional

from src.eval import (
    message_entropy,
    unique_samples,
    compute_disruption_accuracy,
    compute_topsim,
    compute_mi,
    mutual_info_sent_received,
)
from src.interaction import Interaction
from src.channels import ErasureChannel

from egg.core.util import find_lengths, get_opts
from egg.core.batch import Batch


common_opts = get_opts()


class ObjectDataset(Dataset):
    def __init__(
        self, obj_sets: np.ndarray, targets: np.ndarray, attributes: np.ndarray
    ):
        self.receiver_input = obj_sets
        self.labels = targets
        self.sender_input = obj_sets[np.arange(targets.shape[0]), targets]
        self.attributes = attributes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        return (
            self.sender_input[key], self.receiver_input[key],
            self.labels[key], self.attributes[key],
        )


class DataHandler:
    def __init__(self, opts):
        self.batch_size = opts.batch_size
        self.shuffle_train_data = not opts.no_shuffle
        self.float_dtype = torch.get_default_dtype()

        self.n_distractors = None
        self._n_features = None
        self._n_attributes = None

    @property
    def n_features(self):
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        self._n_features = n_features

    @property
    def n_attributes(self):
        return self._n_attributes

    @n_attributes.setter
    def n_attributes(self, n_attributes):
        self._n_attributes = n_attributes

    def load_data(self, opts):
        data = np.load(opts.data_path)

        train = data["train"], data["train_targets"], data["train_attributes"]
        eval_train = data["eval_train"], \
            data["eval_train_targets"], data["eval_train_attributes"]
        eval_test = data["eval_test"], \
            data["eval_test_targets"], data["eval_test_attributes"]

        self.eval_train_mapping = data["eval_train_attribute_mapping"]
        self.eval_test_mapping = data["eval_test_attribute_mapping"]

        if opts.image_input:
            self.eval_train_sample_types = data["eval_train_sample_modes"]
            self.eval_test_sample_types = data["eval_test_sample_modes"]
        else:
            self.eval_train_sample_types = None
            self.eval_test_sample_types = None

        self.train_samples = len(train[0])
        self.eval_train_samples = len(eval_train[0])
        self.eval_test_samples = len(eval_test[0])
        self.n_distractors = train[0].shape[1] - 1

        self.n_features = train[0].shape[-1]
        self.n_attributes = [
            dtype[0].shape[0] for dtype
            in self.eval_train_mapping.dtype.fields.values()
        ]

        opts.train_samples = self.train_samples
        opts.eval_train_samples = self.eval_train_samples
        opts.eval_test_samples = self.eval_test_samples
        opts.n_distractors = self.n_distractors
        opts.n_features = self.n_features
        opts.n_attributes = self.n_attributes

        train_dataset = ObjectDataset(*train)
        eval_train_dataset = ObjectDataset(*eval_train)
        eval_test_dataset = ObjectDataset(*eval_test)

        def collate(batch):
            s_input, r_input, labels, attributes = zip(*batch)
            aux_input = np.stack(attributes)
            return (
                torch.from_numpy(np.stack(s_input)).to(self.float_dtype),
                torch.from_numpy(np.array(labels)).long(),
                torch.from_numpy(np.stack(r_input)).to(self.float_dtype),
                {
                    name: torch.from_numpy(aux_input[name]).long()
                    for name in aux_input.dtype.names
                },
            )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate,
            drop_last=True,
            shuffle=self.shuffle_train_data,
        )
        eval_train_dataloader = DataLoader(
            eval_train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate,
            drop_last=False,
            shuffle=False,
        )
        eval_test_dataloader = DataLoader(
            eval_test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate,
            drop_last=False,
            shuffle=False,
        )

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
    elif opts.optim.lower() == 'adamw':
        optimizer = torch.optim.AdamW
    else:
        raise ValueError('Optimizer must be RMSprop, Adamor AdamW')

    wd = opts.weight_decay
    s_params = list(game.sender.named_parameters())
    r_params = list(game.receiver.named_parameters())
    s_decay = [p for n, p in s_params if 'convnet' not in n]
    s_no_decay = [p for n, p in s_params if 'convnet' in n]
    r_decay = [p for n, p in r_params if 'convnet' not in n]
    r_no_decay = [p for n, p in r_params if 'convnet' in n]

    return optimizer([
        {'params': s_decay, 'lr': opts.sender_lr, 'weight_decay': wd},
        {'params': s_no_decay, 'lr': opts.sender_lr, 'weight_decay': 0},
        {'params': r_decay, 'lr': opts.receiver_lr, 'weight_decay': wd},
        {'params': r_no_decay, 'lr': opts.receiver_lr, 'weight_decay': 0}
    ])


def crop_messages(
    messages: torch.Tensor,
    lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Given an Interaction object, removes non EOS symbols after the first EOS.
    Used to trim EOSed symbols on validation.
    """
    symbols = messages if messages.dim() == 2 else messages.argmax(-1)
    lengths = lengths if lengths is not None else find_lengths(symbols)

    not_eosed = (
        torch.unsqueeze(
            torch.arange(0, symbols.size(1)),
            dim=0,
        ).expand(symbols.size()[:2]).to(symbols.device)
        < torch.unsqueeze(lengths - 1, dim=-1).expand(symbols.size()[:2])
    )

    cropped_symbols = torch.where(not_eosed, symbols, 0)
    if messages.dim() == 2:
        return cropped_symbols
    else:
        cropped_probs = torch.zeros_like(messages).view(-1, messages.size(2))
        cropped_probs.scatter_(1, cropped_symbols.view(-1, 1), 1)
        return cropped_probs.view(messages.size())


class Dump:
    def __init__(
        self,
        game: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        sample_types: Optional[np.ndarray],
        opts: argparse.Namespace,
        # device: Optional[torch.device] = None
    ):
        train_state = game.training
        game.eval()

        logs = []
        for i, batch in enumerate(dataset):
            batch = Batch(*batch).to(opts.device)
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

        messages = logs.message.argmax(-1)
        lengths = logs.message_length.long()  # always accurate on evaluation
        self.messages = crop_messages(messages, lengths)  # 2 dimensional
        self.message_inputs = crop_messages(logs.message, lengths)
        self.logits = logs.logits
        self.lengths = lengths
        self.sample_types = sample_types if sample_types is not None else []

        # select receiver output at 1st EOS
        idx = (torch.arange(len(messages), device=opts.device), lengths - 1)
        # self.receiver_outputs = logs.receiver_output[idx]
        self.receiver_outputs = logs.receiver_output.argmax(-1)[idx]

        if opts.channel != 'none':
            messages = logs.message_nn.argmax(-1)
            lengths = logs.message_length_nn.long()
            self.messages_nn = crop_messages(messages, lengths)
            self.message_inputs_nn = crop_messages(logs.message_nn, lengths)
            self.logits_nn = logs.logits_nn
            self.lengths_nn = lengths

            idx = (torch.arange(len(messages)), lengths - 1)
            # self.receiver_outputs_nn = logs.receiver_output_nn[idx]
            self.receiver_outputs_nn = logs.receiver_output_nn.argmax(-1)[idx]
        else:
            self.messages_nn = self.messages
            self.logits_nn = self.logits
            self.lengths_nn = self.lengths
            self.receiver_outputs_nn = self.receiver_outputs

        self.attribute_names = list(logs.aux_input.keys())
        idx = torch.arange(len(messages)).to(messages.device)

        self.target_attributes = torch.stack(
            [attr[idx, logs.labels] for attr in logs.aux_input.values()],
            dim=-1,
        )
        self.attributes = torch.stack(list(logs.aux_input.values()), dim=1)

        n_distr = self.receiver_inputs.size(1) - 1
        distr_pos = torch.arange(n_distr + 1).to(messages.device).expand(len(messages), -1)
        distr_pos = distr_pos[distr_pos != logs.labels.view(-1, 1)].view(-1, 4)
        self.distractor_attributes = [
            torch.stack([
                attr[idx, distr_pos[:, i]] for attr in logs.aux_input.values()
            ], dim=-1)
            for i in range(n_distr)
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
                # self.receiver_outputs[i].argmax(-1).item(),
                # self.receiver_outputs_nn[i].argmax(-1).item(),
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
                self.sample_types[i] if i < len(self.sample_types) else None,
            )

    def get_tensors(self, noise: bool):
        if noise or self.opts.channel == 'none':
            return (
                self.sender_inputs, self.messages, self.logits,
                self.lengths, self.message_inputs, self.receiver_inputs,
                self.receiver_outputs, self.labels,
                self.target_attributes, self.attributes,
                self.attribute_names
            )

        else:
            return (
                self.sender_inputs, self.messages_nn, self.logits_nn,
                self.lengths_nn, self.message_inputs_nn, self.receiver_inputs,
                self.receiver_outputs_nn, self.labels,
                self.target_attributes, self.attribute,
                self.attribute_names
            )

    def get_message_logs(self, mapping):
        message_logs = []

        def map(key, value):
            mapped = mapping[key][value]
            if isinstance(mapped, np.bytes_):
                return np.str_(mapped.astype('U'))
            return mapped

        for s_input, msg, msg_nn, r_input, r_output, r_output_nn, label, \
                target_attr, distr_attr, sample_type in self:

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

            if self.opts.channel is not None:
                message_nn = ','.join([str(x) for x in msg_nn.tolist()])
                message_log.update({
                    'message_no_noise': message_nn,
                    'prediction_no_noise': r_output_nn,
                })

            if self.opts.image_input:
                message_log['sample_type'] = sample_type
            else:
                message_log.update({
                    'target_category': map('category', target_attr['category']),
                    'distractor_cats':
                        [map('category', x['category']) for x in distr_attr],
                })

            message_logs.append(message_log)

        return message_logs

    def get_eval_dict(self):
        opts = self.opts

        results = {}
        keys = ('baseline',) if opts.channel is None else ('received', 'sent')

        entropy_sent, length_probs_sent = message_entropy(self.logits)
        (entropy_received, length_probs_received) = \
            message_entropy(self.logits_nn) if opts.channel is not None \
            else (entropy_sent, length_probs_sent)

        max_entropy_sent = self.channel.max_message_entropy(
            length_probs_sent, noise=False)
        max_entropy_received = self.channel.max_message_entropy(
            length_probs_received, noise=True)

        for key in keys:
            (s_inputs, messages, logits, lengths, m_inputs, r_inputs,
             r_outputs, labels, t_attributes, attributes, attr_names) \
                = self.get_tensors(key != 'sent')

            idx = torch.arange(len(messages), device=messages.device).long()
            s_attributes = attributes[idx, :, r_outputs]

            entropy, max_entropy = (entropy_received, max_entropy_received) \
                if key != 'sent' else (entropy_sent, max_entropy_sent)
            if opts.image_input:
                idx = torch.arange(len(attr_names)).to(t_attributes.device)
                n_uniq_samples = unique_samples(r_inputs, attributes)
            else:
                n_uniq_samples = unique_samples(r_inputs)

            unique_messages, categorized_messages = \
                torch.unique(messages, dim=0, return_inverse=True)
            if opts.image_input:
                n_uniq_targets = len(torch.unique(t_attributes, dim=0))
            else:
                n_uniq_targets = len(torch.unique(s_inputs, dim=0))
            n_uniq_messages = len(unique_messages)

            results[key] = {
                'samples': len(messages),
                'samples_per_target_obj': len(messages) / n_uniq_targets,
                'samples_per_cat': None,
                'unique_msg': n_uniq_messages,
                'unique_samples': n_uniq_samples,
                'unique_target_objs': n_uniq_targets,
                'unique_target_objs_per_msg': n_uniq_targets / n_uniq_messages,
                'unique_samples_per_target_obj':
                    n_uniq_samples / n_uniq_targets,
                'unique_samples_cat': None,
                'unique_cat': None,
                'unique_samples_per_target_cat': None,
                'unique_cats_per_msg': None,
                'average_length': lengths.float().mean() - 1,
                'actual_vocab_size': torch.unique(messages).numel(),
                'accuracy': (r_outputs == labels).float().mean(),
            }
            results[key].update(compute_disruption_accuracy(
                m_inputs, r_inputs, labels, self.receiver, opts
            ))

            topsim_args = (t_attributes, messages) if opts.image_input \
                else (s_inputs, messages)
            mi_kwargs = {
                'max_len': opts.max_len,
                'vocab_size': opts.vocab_size,
                'erasure_channel': (
                    isinstance(self.channel, ErasureChannel)
                    and key == 'received'),
                'n_samples': 400 if opts.image_input else 100,
            }

            results[key].update({
                'redundancy': 1 - entropy / max_entropy,
                'topsim': compute_topsim(*topsim_args, norm=None),
                'topsim_norm_max': compute_topsim(*topsim_args, norm='max'),
                'topsim_norm_mean': compute_topsim(*topsim_args, norm='mean'),
                'topsim_cosine': None,
                'topsim_cosine_norm_max': None,
                'topsim_cosine_norm_mean': None,
                'entropy_msg': entropy,
                'entropy_max': max_entropy,
                'mutual_info_sent_received': mutual_info_sent_received(
                    logits_sent=self.logits_nn,
                    channel=self.channel,
                    entropy_sent=entropy_sent,
                    entropy_received=entropy_received,
                    **mi_kwargs),
            })
            mi_kwargs['entropy_message'] = entropy

            if opts.image_input:
                for k in list(results[key]):
                    if 'cat' in k or 'cosine' in k:
                        del results[key][k]

                mi_target = compute_mi(logits, t_attributes, **mi_kwargs)
                results[key].update(
                    entropy_target=mi_target['entropy_attr'],
                    mutual_info_msg_target=mi_target['mutual_info_msg_attr'],
                    proficiency_msg_target=mi_target['proficiency_msg_attr'],
                    redundancy_msg_target=mi_target['proficiency_msg_attr'],
                )
                for i, name in enumerate(attr_names):
                    results[key].update({
                        k.replace('attr_dim', f'target_{name}'): v[i]
                        for k, v in mi_target.items() if 'attr_dim' in k
                    })

                mi_selected = compute_mi(logits, s_attributes, **mi_kwargs)
                results[key].update(
                    entropy_selected=mi_selected['entropy_attr'],
                    mutual_info_msg_selected=mi_selected['mutual_info_msg_attr'],
                    proficiency_msg_selected=mi_selected['proficiency_msg_attr'],
                    redundancy_msg_selected=mi_selected['proficiency_msg_attr'],
                )
                for i, name in enumerate(attr_names):
                    results[key].update({
                        k.replace('attr_dim', f'selected_{name}'): v[i]
                        for k, v in mi_selected.items() if 'attr_dim' in k
                    })

                for sample_type in np.unique(self.sample_types):
                    idx = torch.tensor(self.sample_types == sample_type)
                    results[key][f'accuracy_{sample_type}'] = (
                        r_outputs[idx] == labels[idx]
                    ).float().mean()
            else:
                # category = torch.cat([t_attributes] + d_attributes, dim=-1)
                # category = attributes
                n_uniq_samples_cat = unique_samples(r_inputs, attributes)
                n_uniq_cat = len(torch.unique(attributes))
                results[key].update({
                    'unique_samples_cat': n_uniq_samples_cat,
                    'unique_cat': n_uniq_cat,
                    'samples_per_cat': len(messages) / n_uniq_cat,
                    'unique_cats_per_msg': n_uniq_cat / n_uniq_messages,
                    'unique_samples_per_target_cat':
                        n_uniq_samples_cat / n_uniq_cat,
                    'topsim_cosine': compute_topsim(
                        *topsim_args, meaning_distance='cosine', norm='mean'),
                    'topsim_cosine_norm_max': compute_topsim(
                        *topsim_args, meaning_distance='cosine', norm='mean'),
                    'topsim_cosine_norm_mean': compute_topsim(
                        *topsim_args, meaning_distance='cosine', norm='mean'),
                })

                # assign a different number to every input vector
                _, inp = torch.unique(
                    s_inputs, return_inverse=True, dim=0)
                inp = inp.unsqueeze(-1).to(torch.float)
                results[key].update({
                    k.replace('attr', 'target'): v for k, v
                    in compute_mi(logits, inp, **mi_kwargs).items()
                })
                _, selected = torch.unique(
                    r_inputs[idx, labels], return_inverse=True, dim=0)
                selected = selected.unsqueeze(-1).to(torch.float)
                results[key].update({
                    k.replace('attr', 'selected'): v for k, v
                    in compute_mi(logits, selected, **mi_kwargs).items()
                })
                results[key].update({
                    k.replace('attr', 'target_category'): v for k, v
                    in compute_mi(logits, t_attributes, **mi_kwargs).items()
                })

            # convert tensors to numeric
            results[key] = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in results[key].items()
            }

        return results


def print_training_results(dump_dict: dict):
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
