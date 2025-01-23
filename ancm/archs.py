import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from typing import Callable, Optional
from collections import defaultdict

from egg.core.reinforce_wrappers import RnnEncoder
from egg.core.baselines import Baseline, BuiltInBaseline, NoBaseline, MeanBaseline
from egg.core.interaction import LoggingStrategy
from egg.core.util import find_lengths


class SeeingConvNet(nn.Module):
    def __init__(self):
        super(SeeingConvNet, self).__init__()
        
        # Define the sequence of convolutional layers, same as Lazaridou paper 2018
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        return x


class ModMeanBaseline(Baseline):
    """Running mean baseline; all loss batches have equal importance/weight,
    hence it is better if they are equally-sized.
    """

    def __init__(self):
        super().__init__()

        self.mean_baseline = torch.zeros(1, requires_grad=False)
        self.n_points = 0.0

    def update(self, loss: torch.Tensor) -> None:
        self.n_points += 1
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)

        self.mean_baseline += (
            loss.detach().mean().item() - self.mean_baseline
        ) / (self.n_points ** 0.5)

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)
        return self.mean_baseline


class BoundedMeanBaseline(Baseline):
    """Running mean baseline; all loss batches have equal importance/weight,
    hence it is better if they are equally-sized.
    """

    def __init__(self):
        super().__init__()

        self.mean_baseline = torch.zeros(1, requires_grad=False)
        self.n_points = 0.0

    def update(self, loss: torch.Tensor) -> None:
        if self.n_points < 5000:
            self.n_points += 1
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)

        self.mean_baseline += (
            loss.detach().mean().item() - self.mean_baseline
        ) / self.n_points

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)
        return self.mean_baseline


class MovingAverageBaseline(Baseline):
    """Running mean baseline; all loss batches have equal importance/weight,
    hence it is better if they are equally-sized.
    """

    def __init__(self):
        super().__init__()
        self.history = torch.zeros(1, requires_grad=False)
        self.stride = 1000

    def update(self, loss: torch.Tensor) -> None:
        if self.history.device != loss.device:
            self.history = self.history.to(loss.device)
        if len(self.history) == self.stride:
            self.history = self.history[1:]
        self.history = torch.cat(
            [self.history, loss.detach().mean().unsqueeze(0)],
            dim=0)

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.history.device != loss.device:
            self.history = self.history.to(loss.device)
        return self.history.detach().mean().unsqueeze(0)


class SenderReinforce(nn.Module):
    def __init__(self, n_features, n_hidden, image=False):
        super(SenderReinforce, self).__init__()
        self.image = image

        # Vision module for image-based inputs
        if self.image:
            self.vision_module = SeeingConvNet()
            # Update input features for the fully connected layer
            n_features = 2048  # Adjust this based on the output channels of SeeingConvNet

        self.fc1 = nn.Linear(n_features, n_hidden)
            
    def forward(self, x, _aux_input=None):
        if self.image:
            x = self.vision_module(x)
            x = x.flatten(start_dim=1)
        return self.fc1(x).tanh()


class ReceiverReinforce(nn.Module):
    def __init__(self, n_features, linear_units, image=False):
        super(ReceiverReinforce, self).__init__()
        self.image = image

        # Vision module for image-based inputs
        if self.image:
            self.vision_module = SeeingConvNet()
            # Update input features for the fully connected layer
            n_features = 2048 # Adjust this based on the output channels of SeeingConvNet
        self.fc1 = nn.Linear(n_features, linear_units)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x, _input, _aux_input=None):
        if self.image:
            # Pass input through the vision module

            batch_size, n_distractors, channels, height, width = _input.shape
            _input = _input.view(batch_size * n_distractors, channels, height, width)

            _input = self.vision_module(_input)
            _input = _input.flatten(start_dim=1)  # Flatten spatial dimensions
            _input = _input.view(batch_size, n_distractors, -1) # reshape back to 5D

        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        energies = energies.squeeze()
        return self.logsoft(energies)


def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output == _labels).detach().float()
    return -acc, {'accuracy': acc * 100}


class SenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce
    the variance of the gradient estimate.

    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(3, 10)
    ...     def forward(self, rnn_output, _input=None, _aux_input=None):
    ...         return self.fc(rnn_output)
    >>> sender = Sender()
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input=None, _aux_input=None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    ...     loss = F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1)
    ...     aux = {'aux': torch.ones(sender_input.size(0))}
    ...     return loss, aux
    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((5, 3)).normal_()
    >>> optimized_loss, interaction = game(input, labels=None, aux_input=None)
    >>> sorted(list(interaction.aux.keys()))  # returns debug info such as entropies of the agents, message length etc
    ['aux', 'length', 'receiver_entropy', 'sender_entropy']
    >>> interaction.aux['aux'], interaction.aux['aux'].sum()
    (tensor([1., 1., 1., 1., 1.]), tensor(5.))
    """

    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        vocab_size: int,
        channel_type: Optional[str],
        error_prob: float = 0.0,
        length_cost: float = 0.0,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        device: torch.device = torch.device("cpu"),
        baseline_type: Baseline = BoundedMeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
        seed: int = 42,
    ):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks
        """
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.mechanics = CommunicationRnnReinforce(
            sender_entropy_coeff,
            receiver_entropy_coeff,
            vocab_size,
            channel_type,
            error_prob,
            length_cost,
            device,
            baseline_type,
            train_logging_strategy,
            test_logging_strategy,
            seed)
        self.channel = self.mechanics.channel

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None, apply_noise=True):
        return self.mechanics(
            self.sender,
            self.receiver,
            self.loss,
            sender_input,
            labels,
            receiver_input,
            aux_input,
            apply_noise=apply_noise,
        )


class CommunicationRnnReinforce(nn.Module):
    def __init__(
        self,
        sender_entropy_coeff: float,
        receiver_entropy_coeff: float,
        vocab_size: int,
        channel_type: Optional[str],
        error_prob: float = 0.0,
        length_cost: float = 0.0,
        device: torch.device = torch.device('cpu'),
        baseline_type: Baseline = BoundedMeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
        seed: int = 42,
    ):
        """
        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks
        """
        super().__init__()

        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.length_cost = length_cost

        if channel_type == 'erasure':
            self.channel = ErasureChannel(error_prob, vocab_size, device, seed)
        elif channel_type == 'symmetric':
            self.channel = SymmetricChannel(error_prob, vocab_size, device, seed)
        elif channel_type == 'deletion':
            self.channel = DeletionChannel(error_prob, vocab_size, device, seed)
        else:
            self.channel = None

        self.baselines = defaultdict(baseline_type)
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(
        self,
        sender,
        receiver,
        loss_fn,
        sender_input,
        labels,
        receiver_input=None,
        aux_input=None,
        apply_noise=True,
    ):
        message, log_prob_s, entropy_s = sender(sender_input, aux_input)
        message_length_nn = find_lengths(message)
        if self.channel:
            message = self.channel(message, message_length_nn, apply_noise)
            message_length = find_lengths(message)
        else:
            message_length = message_length_nn

        receiver_output, log_prob_r, entropy_r = receiver(
            message, receiver_input, aux_input, message_length
        )

        loss, aux_info = loss_fn(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_length).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_length_nn.float()

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff
            + entropy_r.mean() * self.receiver_entropy_coeff
        )

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_length_nn.float() * self.length_cost
        policy_length_loss = (length_loss * effective_log_prob_s).mean()

        baseline = self.baselines['loss'].predict(loss.detach())
        policy_loss = ((loss.detach() - baseline) * log_prob).mean()

        aux_info['reinf_sg'] = loss.detach()
        aux_info['baseline'] = baseline

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        if self.training:
            self.baselines["loss"].update(loss)
            self.baselines["length"].update(length_loss)

        aux_info["sender_entropy"] = entropy_s.detach()
        aux_info["receiver_entropy"] = entropy_r.detach()
        aux_info["length"] = message_length.float()  # will be averaged

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )

        return optimized_loss, interaction


# Noisy channel classes
class Channel(nn.Module):
    def __init__(self, error_prob, vocab_size, device, seed=42):
        super().__init__()
        self.p = error_prob
        self.vocab_size = vocab_size
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.device = device


class ErasureChannel(Channel):
    """
    Erases a symbol from a message with a given probability
    """

    def forward(self, message, message_length=None, apply_noise=False):
        if self.p != 0. and apply_noise:
            msg = message if message.dim() == 2 else message.argmax(dim=-1)

            # sample symbol indices to be erased
            target_ids = (torch.rand(*msg.size(), generator=self.generator) < self.p).to(self.device)

            # if message length is not provided, compute it
            if message_length is None:
                message_length = find_lengths(message)

            # True for all message symbols before the 1st EOS symbol
            not_eosed = (
                torch.unsqueeze(
                    torch.arange(0, message.size(1)), dim=0
                ).expand(message.size(0), message.size(1)).to(self.device)
                < torch.unsqueeze(
                    message_length - 1, dim=-1
                ).expand(message.size(0), message.size(1))
            )

            # erase
            message = torch.where(
                torch.logical_and(target_ids, not_eosed),
                torch.tensor(self.vocab_size),  # i.e. erased symbol
                message)

        return message


class DeletionChannel(Channel):
    """
    Deletes a symbol with a given probability.
    """

    def forward(self, message, message_length=None, apply_noise=False):
        if self.p != 0. and apply_noise:
            msg = message if message.dim() == 2 else message.argmax(dim=-1)
            msg = msg.detach()

            if message_length is None:
                message_length = find_lengths(msg)

            not_eosed = (
                torch.unsqueeze(
                    torch.arange(0, message.size(1)), dim=0).expand(
                        message.size(0), message.size(1)).to(self.device)
                < torch.unsqueeze(message_length - 1, dim=-1).expand(
                    message.size(0), message.size(1))
            )

            # sample symbol indices to be erased
            target_ids = (torch.rand(*msg.size(), generator=self.generator) < self.p).to(self.device)
            delete_ids = torch.logical_and(target_ids, not_eosed)
            keep_ids = torch.logical_not(delete_ids)
            num_deleted = torch.sum(delete_ids.int(), dim=1)

            message = torch.stack([
                torch.cat(
                    [message[i][keep_ids[i]], torch.zeros(num_deleted[i], dtype=torch.int)])
                for i in range(message.size(0))
            ])

        return message


class SymmetricChannel(Channel):
    """
    Replaces each symbol with a different symbol with a given probability.
    The replacement symbol is randomly sampled from a uniform distribution.
    """

    def forward(self, message, message_length=None, apply_noise=False):
        if self.p != 0. and apply_noise:
            msg = message if message.dim() == 2 else message.argmax(dim=-1)

            # which symbols should be erased
            target_ids = (torch.rand(*msg.size(), generator=self.generator) < self.p).to(self.device)

            # possible replacements of each symbol (excl. EOS)
            candidate_symbols = torch.arange(1, self.vocab_size).reshape(1, -1)
            candidate_symbols = candidate_symbols.expand(msg.size(1), -1)
            candidate_symbols = candidate_symbols.reshape(1, msg.size(1), -1)
            candidate_symbols = candidate_symbols.expand(msg.size(0), msg.size(1), -1)

            # each symbol of the message has vocab_size - 2 possible replacements
            # we replace 0s with 1s to ensure that exactly one symbol is excluded
            # this works as only non-EOS symbols may be replaced
            msg_exp = torch.where(msg != 0, msg, 1)
            msg_exp = msg_exp.expand(self.vocab_size - 1, *msg_exp.size()).permute(1, 2, 0)
            keep_ids = (candidate_symbols != msg_exp)
            candidate_symbols = candidate_symbols[keep_ids].reshape(*msg.size(), self.vocab_size - 2)

            # sample the replacement symbol to be used
            replacement_indices = torch.randint(
                high=self.vocab_size - 2,
                size=(msg.size()),
                generator=self.generator,
                device=self.device)

            # select replacement symbols
            replacement_ids = torch.stack([
                torch.tensor([i, j, replacement_indices[i, j]], dtype=torch.int)
                for i in range(msg.size(0))
                for j in range(msg.size(1))
            ])
            replacement_ids_chunked = replacement_ids.t().chunk(chunks=3)
            replacement_symbols = candidate_symbols[replacement_ids_chunked]
            replacement_symbols = replacement_symbols.reshape(*msg.size())

            # compute message length if it is not provided
            if message_length is None:
                message_length = find_lengths(message)

            # True for all message symbols before the 1st EOS symbol
            not_eosed = (
                torch.unsqueeze(
                    torch.arange(0, message.size(1)), dim=0
                ).expand(message.size(0), message.size(1)).to(self.device)
                < torch.unsqueeze(
                    message_length - 1, dim=-1
                ).expand(message.size(0), message.size(1))
            )

            message = torch.where(
                torch.logical_and(target_ids, not_eosed),
                replacement_symbols,
                message)

        return message
