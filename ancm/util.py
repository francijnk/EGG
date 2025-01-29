import json
import torch
import numpy as np
from tabulate import tabulate
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from typing import Optional

from egg.core.util import find_lengths, move_to, get_opts

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
)

common_opts = get_opts()


class ObjectDataset(Dataset):
    def __init__(self, obj_sets, labels, attributes=None, attribute_names=None):
        self.obj_sets = obj_sets
        self.labels = labels
        self.attributes = attributes
        self.attribute_names = attribute_names

    def __len__(self):
        return len(self.obj_sets)

    def __getitem__(self, idx):
        return self.obj_sets[idx], self.labels[idx], self.attributes[idx]


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

        train_dataset = ObjectDataset(*train, attribute_names)
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
                aux[attr_name] = torch.from_numpy(obj_attributes[attr_name]).long()

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
                shuffle=False,
            )
        else:
            train_eval_dataloader = None

        return train_dataloader, val_dataloader, test_dataloader, train_eval_dataloader


class Dump:
    def __init__(self, sender_inputs, messages, receiver_inputs, receiver_outputs, labels, attributes):
        self.sender_inputs = sender_inputs
        self.messages = messages
        self.receiver_inputs = receiver_inputs
        self.receiver_outputs = receiver_outputs
        self.labels = labels

        n_distractors = receiver_inputs[0].size(0) - 1
        self.target_attributes = {
            k.strip('target_category'): v for k, v in attributes.items()
            if not k.startswith('distr')}
        self.distractor_attributes = [{k: v} for k, v in list(attributes.items())[1:]]
        # self.distractor_attributes = [
        #     {k: v for k, v in attributes.items()
        #      if k.startswith('distr') and str(i) in k}
        #     for i in range(n_distractors)]

        print(self.distractor_attributes)
        print(list(attributes.keys()))

    def __len__(self):
        return len(self.sender_inputs)

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self.sender_inputs[i],
                self.messages[i],
                self.receiver_inputs[i],
                self.receiver_outputs[i].item(),
                self.labels[i].int().item(),
                {
                    k: self.target_attributes[k][i].int().item()
                    for k in self.target_attributes
                },
                [
                    {k: attributes[k][i].int().item() for k in attributes}
                    for attributes in self.distractor_attributes
                ],
            )

    def get_tensors(self):
        sender_inputs = torch.stack(self.sender_inputs)
        messages = torch.nn.utils.rnn.pad_sequence(
            self.messages, batch_first=True)
        messages = torch.cat(
            [messages, torch.zeros_like(messages[:, :1])], dim=-1)
        receiver_inputs = torch.stack(
            self.receiver_inputs)
        receiver_outputs = torch.stack(
            self.receiver_outputs)
        labels = torch.stack(self.labels)
        target_attributes = [
            torch.stack(a_tensors)
            for a_tensors in self.target_attributes.values()]
        target_attributes = torch.cat(target_attributes, dim=-1)

        distractor_attributes = [
            [torch.stack(a_tensors) for a_tensors in attributes.values()]
            for attributes in self.distractor_attributes]
        distractor_attributes = [
            torch.cat(attributes, dim=-1)
            for attributes in distractor_attributes]

        attribute_names = list(self.target_attributes.keys())

        return sender_inputs, messages, receiver_inputs, \
            receiver_outputs, labels, target_attributes, \
            distractor_attributes, attribute_names


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


def dump_sender_receiver(
        game: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        apply_noise: bool,
        max_len: int,
        vocab_size: int,
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
    train_state = game.training
    game.eval()

    device = device if device is not None else common_opts.device

    sender_inputs, messages = [], []
    receiver_inputs, receiver_outputs = [], []
    labels, attributes = [], defaultdict(list)

    with torch.no_grad():
        for batch in dataset:
            sender_input = move_to(batch[0], device)
            sender_inputs.extend(sender_input)
            receiver_input = move_to(batch[2], device)
            receiver_inputs.append(receiver_input)
            labels.extend(batch[1])
            for key, val in batch[3].items():
                attributes[key].extend(val)

            message = game.sender(sender_input)
            if isinstance(message, tuple):
                message, log_prob, entropy = message

            # Add noise to the message
            if game.channel:
                message = game.channel(message, apply_noise=apply_noise)

            output = game.receiver(message, receiver_input)
            output = output[0] if isinstance(output, tuple) else output

            if message.dim() == 3:
                message = message.argmax(-1)

            # if isinstance(sender_input, list) or isinstance(sender_input, tuple):
            #     if sender_input[0].dim() == 3:
            #         sender_input = [item[:, 0] for item in sender_input]
            #     sender_inputs.extend(zip(*sender_input))
            # else:
            #     if sender_input.dim() == 3:
            #         sender_input = sender_input[:, 0]
            #     sender_inputs.extend(sender_input)

            # Each message ends with EOS
            lengths = find_lengths(message)
            for i in range(message.size(0)):
                messages.append(message[i, :lengths[i]])
                receiver_outputs.append(output[i, ...])

    game.train(mode=train_state)

    return Dump(sender_inputs, messages, receiver_inputs, receiver_outputs, labels, attributes)


def get_results_dict(dump, receiver, opts, unique_dict, noise=True, aux_dump=None):
    if aux_dump is None:
        aux_dump = dump

    (s_inp, msg, r_inp, r_out, labels, attr,
     distr_attr, attr_names) = dump.get_tensors()
    (aux_s_inp, aux_msg, aux_r_inp, aux_r_out,
     aux_labels, aux_attr, aux_distr_attr, _) = aux_dump.get_tensors()

    receiver_vocab_size = opts.vocab_size if opts.channel != 'erasure' \
        or opts.error_prob == 0 else opts.vocab_size + 1
    actual_vocab = set(int(s) for m in dump.messages for s in m.tolist())

    # receiver_outputs = move_to(receiver_outputs, device)
    # labels = move_to(labels, device)

    channel, error_prob = (None, 0.) if not noise \
        else (opts.channel, opts.error_prob)

    results = {
        'accuracy': (r_out == labels).float().mean().item(),
        'accuracy2': compute_accuracy2(
            aux_msg, aux_r_inp,
            aux_labels, receiver, opts.batch_size),
        'unique_messages': len(torch.unique(msg, dim=0)),
        'unique_target_objects': len(unique_dict.keys()),
        'actual_vocab_size': len(actual_vocab),
        'redundancy': compute_redundancy(
            msg, opts.max_len, opts.vocab_size, channel, error_prob),
        'redundancy_adj': compute_adjusted_redundancy(
            msg, channel, error_prob, torch.arange(receiver_vocab_size)),
        'redundancy_adj_voc': compute_adjusted_redundancy(
            msg, channel, error_prob, actual_vocab),
        'max_rep': compute_max_rep(msg).mean().item(),
        # alignment = compute_conceptual_alignment(
        #     test_data, _receiver, _sender, device, opts.batch_size)
    }

    if opts.image_input:
        results['topographic_rho'] = compute_top_sim(aux_attr, aux_msg)
        results['posdis'] = compute_posdis(aux_s_inp, aux_msg)
        results['bosdis'] = compute_bosdis(aux_s_inp, aux_msg, receiver_vocab_size)
    else:
        results['topographic_rho'] = compute_top_sim(aux_s_inp, aux_msg)
        results['topographic_rho_category'] = compute_top_sim(aux_attr, aux_msg)

    if opts.image_input:
        mi_attr_msg = compute_mi(msg, attr)
        results['entropy_msg'] = mi_attr_msg['entropy_msg']
        results['entropy_attr'] = mi_attr_msg['entropy_attr']
        results['mi_msg_attr'] = mi_attr_msg['mi_msg_attr']
        results['entropy_attr_dim'] = {
            name: value for name, value
            in zip(attr_names, mi_attr_msg['entropy_attr_dim'])}
        results['mi_msg_attr_dim'] = {
            name: value for name, value
            in zip(attr_names, mi_attr_msg['mi_msg_attr_dim'])}
    else:
        mi_inp_msg = compute_mi(msg, s_inp)
        results['entropy_msg'] = mi_inp_msg['entropy_msg']
        results['entropy_inp'] = mi_inp_msg['entropy_attr']
        results['mi_msg_inp'] = mi_inp_msg['mi_msg_attr']
    mi_cat_msg = compute_mi(msg, attr)
    results['entropy_cat'] = mi_cat_msg['entropy_attr']
    results['mi_msg_cat'] = mi_cat_msg['mi_msg_attr']

    return results


def print_training_results(output_dict):
    def _format(value):
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            return f'{value:.2f}'
        elif isinstance(value, dict):
            output = [f'{k}: {_format(v)}' for k, v in value.items()]
            return '\n'.join(output)
        else:
            return value

    flattened_dict = {
        ' '.join((key1, key3)).center(24): value3
        for key1, value1 in output_dict.items()
        for key2, value2 in value1.items()
        if key2 == 'results'
        for key3, value3 in value2.items()
    }
    header = ['metric'] + list(flattened_dict.keys())
    metrics = list(list(flattened_dict.values())[0].keys())
    values = [metrics] + [
        [v for v in col.values()]
        for col in flattened_dict.values()]
    table_dict = {h: v for h, v in zip(header, values)}
    print(tabulate(
        table_dict,
        headers='keys',
        tablefmt="rounded_outline",
        maxcolwidths=24,
    ))


def crop_messages(interaction):
    """
    Given an Interaction object, removes non EOS symbols after the first EOS.
    Used to trim EOSed symbols on validation.
    Only works with Reinforce.
    """
    if interaction.message.dim() == 2:
        assert interaction.message_length is not None
        size = interaction.message.shape
        not_eosed = (
            torch.unsqueeze(
                torch.arange(0, size[1]), dim=0
            ).expand(size[:2]).to(interaction.message.device)
            < torch.unsqueeze(
                interaction.message_length - 1, dim=-1
            ).expand(size[:2])
        )
        interaction.message = torch.where(not_eosed, interaction.message, 0.)

    if interaction.message.dim() == 3:
        message = interaction.message.argmax(-1)
        lengths = find_lengths(message).unsqueeze(-1).expand(message.size())
        positions = torch.arange(
            message.size(1)).unsqueeze(0).expand(message.size())

        nonzero_ids = message.nonzero()
        nonzero_chunks = nonzero_ids.t().chunk(chunks=2)
        # nonzero_chunks = message.nonzero(as_tuple=True)
        targets = (positions[nonzero_chunks] > lengths[nonzero_chunks] - 1)
        targets = targets.squeeze()
        target_ids = nonzero_ids[targets]

        # if no targets are found, the dimension is 3
        if target_ids.dim() == 2:
            target_chunks = target_ids.t().chunk(chunks=3)
            replacement_probs = torch.zeros_like(interaction.message[0, 0])
            replacement_probs[0] = 1.
            interaction.message[target_chunks] = replacement_probs
