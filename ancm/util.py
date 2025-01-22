import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

from typing import Optional

from egg.core.util import move_to, get_opts


common_opts = get_opts()


class ObjectDataset(Dataset):
    def __init__(self, obj_sets, labels):
        self.obj_sets = obj_sets
        self.labels = labels

    def __len__(self):
        return len(self.obj_sets)

    def __getitem__(self, idx):
        return self.obj_sets[idx], self.labels[idx]


class CustomDataset(Dataset):
    def __init__(self, messages, receiver_inputs):
        """
        Args:
            messages (torch.Tensor): Tensor of shape (N, 5), where N is the number of samples.
            receiver_inputs (torch.Tensor): Tensor of shape (N, 5, 8).
        """
        assert len(messages) == len(receiver_inputs), "Messages and receiver_inputs must have the same number of samples."
        self.messages = messages
        self.receiver_inputs = receiver_inputs

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.messages[idx], self.receiver_inputs[idx]


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
        train = data["train"], data["train_labels"]
        val = data["valid"], data["valid_labels"]
        test = data["test"], data["test_labels"]

        self._n_features = train[0].shape[-1]
        self.perceptual_dimensions = [-1] * self._n_features
        self.train_samples = train[0].shape[0]
        self.validation_samples = val[0].shape[0]
        self.test_samples = test[0].shape[0]
        self.n_distractors = train[0].shape[1] - 1

        opts.perceptual_dimensions = self.perceptual_dimensions
        opts.train_samples = self.train_samples
        opts.validation_samples = self.validation_samples
        opts.test_samples = self.test_samples
        opts.n_distractors = self.n_distractors
        opts.n_features = self.n_features

        train_dataset = ObjectDataset(*train)
        val_dataset = ObjectDataset(*val)
        test_dataset = ObjectDataset(*test)

        def _collate(batch):
            obj_sets, target_ids = zip(*batch)
            bs = self.batch_size

            r_inputs, labels = np.vstack(np.expand_dims(obj_sets, 0)), np.array(target_ids)
            targets = r_inputs[np.arange(bs), labels]
            return (
                torch.from_numpy(targets).float(),
                torch.from_numpy(labels).long(),
                torch.from_numpy(r_inputs).float(),
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
            drop_last=True)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=_collate,
            drop_last=True)

        return train_dataloader, val_dataloader, test_dataloader


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def build_optimizer(game, opts):
    return torch.optim.RMSprop([
        {"params": game.sender.parameters(), "lr": opts.sender_lr},
        {"params": game.receiver.parameters(), "lr": opts.receiver_lr},
    ])


def crop_messages(interaction):
    """
    Given an Interaction object, removes non EOS symbols after the first EOS.
    """
    assert interaction.message_length is not None
    for i in range(interaction.size):
        length = interaction.message_length[i].long().item()
        interaction.message[i, length:] = 0


def dump_sender_receiver(
    game: torch.nn.Module,
    dataset: "torch.utils.data.DataLoader",
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

    with torch.no_grad():
        for batch in dataset:
            # by agreement, each batch is (sender_input, labels) plus optional (receiver_input)
            sender_input = move_to(batch[0], device)
            receiver_input = None if len(batch) == 2 else move_to(batch[2], device)

            message = game.sender(sender_input)

            message, log_prob, entropy = message

            # Add noise to the message
            if game.channel:
                message = game.channel(message, apply_noise=apply_noise)

            output = game.receiver(message, receiver_input)
            output = output[0]

            if batch[1] is not None:
                labels.extend(batch[1])

            if isinstance(sender_input, list) or isinstance(sender_input, tuple):
                if sender_input[0].dim() == 3:
                    sender_input = [item[:, 0] for item in sender_input]
                sender_inputs.extend(zip(*sender_input))
            else:
                if sender_input.dim() == 3:
                    sender_input = sender_input[:, 0]
                sender_inputs.extend(sender_input)

            if receiver_input is not None:
                receiver_inputs.extend(receiver_input)

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

                    receiver_outputs.append(output[i, ...])

    game.train(mode=train_state)

    return sender_inputs, messages, receiver_inputs, receiver_outputs, labels


def truncate_messages(messages, receiver_input, labels):
    new_messages = []
    new_r_input = []
    new_labels = []
    for i, message in enumerate(messages):
        truncated = remove_n_items(message, 1)
        new_messages.extend(truncated)
        new_r_input.extend([receiver_input[i]] * len(truncated))
        new_labels.extend([labels[i]] * len(truncated))

    return new_messages, new_r_input, new_labels


def remove_n_items(tensor, n=1):
    """
    Removes all possible combinations of `n` items from the tensor,
    symbol 0 is never removed.
    Needed for "redundancy" measure if using rf.

    Args:
        tensor (torch.Tensor): The input tensor.
        n (int): The number of items to remove.

    Returns:
        list[torch.Tensor]: A list of tensors with `n` items removed.
    """
    # Ensure the input is a PyTorch tensor
    tensor = torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor

    # Get the indices of elements that can be removed (exclude 0)
    removable_indices = [idx for idx in range(len(tensor)) if tensor[idx] != 0]

    # Generate all combinations of `n` indices to remove
    combos = list(combinations(removable_indices, n))

    # Create new tensors with the combinations removed
    result = []
    for indices in combos:
        mask = torch.ones(len(tensor), dtype=torch.bool)
        mask[list(indices)] = True
        new = tensor[mask]
        new = new.to(torch.long)
        result.append(new)

    return result
