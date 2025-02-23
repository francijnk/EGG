import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.utils import logits_to_probs, probs_to_logits
from torch.distributions import RelaxedOneHotCategorical, Categorical, OneHotCategorical

# from ancm.util import crop_messages

from typing import Callable, Optional
from collections import defaultdict

from ancm.channels import (
    NoChannel,
    ErasureChannel,
    DeletionChannel,
    SymmetricChannel,
)

# import egg.core as core
# from egg.core.reinforce_wrappers import RnnEncoder
from egg.core.baselines import Baseline  # NoBaseline, MeanBaseline, BuiltInBaseline
from ancm.interaction import LoggingStrategy
from egg.core.util import find_lengths


update_bound = None


# Baselines
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
        global update_bound
        if self.n_points < update_bound:
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


# CNN module
class SeeingConvNet(nn.Module):
    def __init__(self, n_hidden):
        super(SeeingConvNet, self).__init__()

        # TODO decide on one of these architectures, remove
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
        self.conv_layers = None
        # params = sum(p.numel() for p in self.conv_layers.parameters())
        # print('Lazaridou', (params), 'parameters')

        # Denamganaï, Missaoui and Walker, 2023 (https://arxiv.org/pdf/2304.14511)
        # https://github.com/Near32/ReferentialGym/blob/develop/ReferentialGym/networks/networks.py

        input_shape = 64
        n_filters = 32
        k = 3
        s = 2
        p = 1

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=k, stride=s, padding=p, bias=False, dilation=p),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),

            nn.Conv2d(n_filters, n_filters, kernel_size=k, stride=s, padding=p, bias=False, dilation=p),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),

            nn.Conv2d(n_filters, n_filters * 2, kernel_size=k, stride=s, padding=p, bias=False, dilation=p),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),

            nn.Conv2d(n_filters * 2, n_filters * 2, kernel_size=k, stride=s, padding=p, bias=False, dilation=p),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),
        )
        # params = sum(p.numel() for p in self.conv_net.parameters())
        # print('ReferentialGym', (params), 'parameters')

        self.out_features = ((input_shape - k + 2 * p) // s + 1) ** 2

        self.fc = nn.Sequential(
            nn.Linear(self.out_features, n_hidden),
            nn.ReLU(),
        )
        # Choi & Lazaridou (2018) use one layer of 256 hidden units? (https://arxiv.org/pdf/1804.02341)
        # Denamganaï, Missaoui and Walker use 2 x 128 instead
        # self.fc = nn.Linear(out_features, n_hidden)

    def forward(self, x):
        if x.dim() == 5:
            batch_size, n_candidates, *image_dims = x.shape
            x = self.conv_net(x.view(batch_size * n_candidates, *image_dims))
            return self.fc(x.view(batch_size, n_candidates, -1))
        else:
            x = self.conv_net(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)


# Basic Sender and Receiver classes
class Sender(nn.Module):
    def __init__(self, n_features, n_hidden, image_input=False):
        super(Sender, self).__init__()
        if image_input:
            self.encoder = SeeingConvNet(n_hidden)
        else:
            self.encoder = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.encoder(x).tanh()


class Receiver(nn.Module):
    def __init__(self, n_features, linear_units, image_input=False):
        super(Receiver, self).__init__()
        self.image_input = image_input

        if self.image_input:
            self.encoder = SeeingConvNet(linear_units)
        else:
            self.encoder = nn.Linear(n_features, linear_units)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, encoded_message, _input, _aux_input=None):
        embedded_input = self.encoder(_input).tanh()
        energies = torch.matmul(embedded_input, encoded_message.unsqueeze(-1))
        energies = energies.squeeze()
        return self.logsoft(energies)


# REINFORCE
def loss_rf(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output == _labels).detach().float()
    return -acc, {'accuracy': acc * 100}


class RnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(10, 3)
    ...     def forward(self, x, _input=None, _aux_input=None):
    ...         return self.fc(x)
    >>> agent = Agent()
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm')
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()  # batch size x max_len+1
    torch.Size([16, 11])
    >>> (entropy[:, -1] > 0).all().item()  # EOS symbol will have 0 entropy
    False
    """

    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
        cell="rnn",
    ):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        """
        super(RnnSenderReinforce, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList(
            [
                cell_type(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else cell_type(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )  # noqa: E502

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = [self.agent(x, aux_input)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        prev_c = [
            torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        e_t = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []
        # probs = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(e_t, (prev_hidden[i], prev_c[i]))
                    prev_c[i] = c_t
                else:
                    h_t = layer(e_t, prev_hidden[i])
                prev_hidden[i] = h_t
                e_t = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            # probs.append(distr.probs())

            if True:  # self.training: TODO test
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            e_t = self.embedding(symbols)
            sequence.append(symbols)

        sequence = torch.stack(sequence, dim=1)
        logits = torch.stack(logits, dim=1)
        entropy = torch.stack(entropy, dim=1)
        # probs = torch.stack(probs, dim=1)

        zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
        # eos = torch.zeros_like(probs[:, :1, :])
        # eos[:, 0, :] = 1

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)
        # probs = torch.cat([probs, eos], dim=1)

        return sequence, logits, entropy


class SenderReceiverRnnReinforce(nn.Module):
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
        TODO
        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
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
            # train_logging_strategy,
            # test_logging_strategy,
            seed)
        self.channel = self.mechanics.channel

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        return self.mechanics(
            self.sender,
            self.receiver,
            self.loss,
            sender_input,
            labels,
            receiver_input,
            aux_input,
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
        seed: int = 42,
    ):
        """
        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        """
        super().__init__()

        channel_types = {
            'none': NoChannel,
            'erasure': ErasureChannel,
            'deletion': DeletionChannel,
            'symmetric': SymmetricChannel,
        }

        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.length_cost = length_cost
        self.channel = channel_types[channel_type](error_prob, vocab_size, device, seed)
        self.baselines = defaultdict(baseline_type)

    def forward(
        self,
        sender,
        receiver,
        loss_fn,
        sender_input,
        labels,
        receiver_input=None,
        aux_input=None,
    ):
        global update_bound

        if update_bound is None:
            update_bound = 2 ** 16 / sender_input.size(0)

        message_nn, probs_nn, log_prob_s, entropy_s = sender(sender_input, aux_input)
        message_length_nn = find_lengths(message_nn)

        message, message_nn, probs = self.channel(
            message=message_nn,
            lengths=message_length_nn)

        if isinstance(self.channel, NoChannel):
            message_length = message_length_nn.detach()
            receiver_output, log_prob_r, entropy_r = \
                self.receiver(message, receiver_input, aux_input)
            receiver_output_nn = receiver_output.detach()
        else:
            # compute receiver outputs for messages without noise
            message_length = find_lengths(message).detach()
            message_joined = torch.cat([
                message,
                message_nn.detach(),
            ], dim=0)
            receiver_input_joined = torch.cat([
                receiver_input,
                receiver_input.detach(),
            ], dim=0)
            aux_input_joined = {
                key: torch.cat([
                    vals,
                    vals.detach(),
                ], dim=0)
                for key, vals in aux_input.items()
            }

            receiver_output_joined = self.receiver(
                message_joined,
                receiver_input_joined,
                aux_input_joined)

            receiver_output, log_prob_r, entropy_r = \
                (item[:len(message)] for item in receiver_output_joined)
            # receiver_output_nn = receiver_output_joined[0][len(message):]

        loss, aux_info = loss_fn(
            sender_input, message, receiver_input,
            receiver_output, labels, aux_input)
        # aux_info['accuracy_nn'] = \
        #     (receiver_output_nn == labels).detach().float() * 100

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

        logging_strategy = LoggingStrategy()

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


# Gumbel-Softmax
def loss_gs(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    cross_entropy = F.cross_entropy(receiver_output, _labels, reduction="none")
    accuracy = (receiver_output.detach().argmax(dim=-1) == _labels).float()

    return cross_entropy, {'accuracy': accuracy * 100}

    min_real = torch.finfo(receiver_output.dtype).min
    r_output = receiver_output.detach()
    r_probs = logits_to_probs(r_output)
    r_log2_prob = torch.clamp(r_output / np.log(2), min=min_real)
    r_entropy = torch.linalg.vecdot(-r_probs, r_log2_prob)

    aux_info = {
        'accuracy': accuracy * 100,
        'entropy_receiver': r_entropy,
    }

    return cross_entropy, aux_info


class RnnSenderGS(nn.Module):
    """
    Gumbel Softmax wrapper for Sender that outputs variable-length sequence of symbols.
    The user-defined `agent` takes an input and outputs an initial hidden state vector for the RNN cell;
    `RnnSenderGS` then unrolls this RNN for the `max_len` symbols. The end-of-sequence logic
    is supposed to be handled by the game implementation. Supports vanilla RNN ('rnn'), GRU ('gru'), and LSTM ('lstm')
    cells.

    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc_out = nn.Linear(10, 5) #  input size 10, the RNN's hidden size is 5
    ...     def forward(self, x, _aux_input=None):
    ...         return self.fc_out(x)
    >>> agent = Sender()
    >>> agent = RnnSenderGS(agent, vocab_size=2, embed_dim=10, hidden_size=5, max_len=3, temperature=1.0, cell='lstm')
    >>> output = agent(torch.ones((1, 10)))
    >>> output.size()  # batch size x max_len+1 x vocab_size
    torch.Size([1, 4, 2])
    """

    def __init__(
            self,
            agent,
            vocab_size,
            embed_dim,
            hidden_size,
            max_len,
            temperature,
            cell="rnn",
            temperature_lr=None,
            straight_through=False):

        super(RnnSenderGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if temperature_lr is None:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]),
                requires_grad=True,
            )

        self.straight_through = straight_through
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.reset_parameters()

    def gumbel_softmax_sample(self, logits: torch.Tensor):
        probs = logits_to_probs(logits.detach() / self.temperature)

        if self.training:
            distr = RelaxedOneHotCategorical(
                logits=logits,
                temperature=self.temperature)
            sample = distr.rsample()
            return sample, probs
        else:
            # argmax of GS sample is equivalent to sampling from the original
            # distribution: Softmax(logits, temperature)
            distr = OneHotCategorical(logits=(logits / self.temperature))
            sample = distr.sample()
            return sample, probs

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = self.agent(x, aux_input)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))

        sequence, probs = [], []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            symbols, symbol_probs = self.gumbel_softmax_sample(step_logits)

            prev_hidden = h_t
            e_t = self.embedding(symbols)

            sequence.append(symbols)
            probs.append(symbol_probs)

        sequence = torch.stack(sequence, dim=1)
        probs = torch.stack(probs, dim=1)

        return sequence, probs


class SenderReceiverRnnGS(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        vocab_size: int,
        channel_type: Optional[str],
        error_prob: float = 0.0,
        length_cost: int = 0.0,
        device: torch.device = torch.device("cpu"),
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
        :param length_cost: the penalty applied to Sender for each symbol produced
        """

        channel_types = {
            'none': NoChannel,
            'erasure': ErasureChannel,
            'deletion': DeletionChannel,
            'symmetric': SymmetricChannel,
        }

        super(SenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost
        self.channel = channel_types[channel_type](
            error_prob, sender.max_len, vocab_size, device, seed)

    def forward(self, sender_input, labels, receiver_input, aux_input):
        sender_output = self.sender(sender_input, aux_input)

        # pass messages and symbol probabilities through the channel
        (message, probs), (message_nn, probs_nn) = self.channel(*sender_output)

        # append EOS to each message
        eos = torch.zeros_like(message[:, :1, :])
        eos[:, 0, 0] = 1
        message = torch.cat([message, eos], dim=1)
        message_nn = torch.cat([message_nn, eos], dim=1)
        probs = torch.cat([probs, eos], dim=1)
        probs_nn = torch.cat([probs_nn, eos], dim=1)

        if isinstance(self.channel, NoChannel):
            receiver_output = self.receiver(message, receiver_input, aux_input)
            receiver_output_nn = receiver_output.detach()
        else:
            # compute receiver outputs for messages without noise as well
            message_joined = torch.cat([
                message,
                message_nn.detach(),
            ], dim=0)
            receiver_input_joined = torch.cat([
                receiver_input,
                receiver_input.detach(),
            ], dim=0)
            aux_input_joined = {
                key: torch.cat([
                    vals,
                    vals.detach(),
                ], dim=0)
                for key, vals in aux_input.items()
            }

            receiver_output_joined = self.receiver(
                message_joined,
                receiver_input_joined,
                aux_input_joined)

            receiver_output = receiver_output_joined[:len(message)]
            receiver_output_nn = receiver_output_joined[len(message):]

        loss, z, length, length_nn = 0, 0, 0, 0
        # length_probs = torch.zeros_like(message[:, :, 0])
        # length_probs_nn = length_probs.clone()
        not_eosed_before = torch.ones(message.size(0)).to(message.device)
        not_eosed_before_nn = not_eosed_before.detach().clone()
        aux_info = {}

        # compute aux info values without noise
        accuracy_nn = (receiver_output_nn.argmax(-1) == labels.unsqueeze(-1)).float() * 100
        # min_real = torch.finfo(receiver_output.dtype).min
        # r_probs = logits_to_probs(receiver_output_nn)
        # r_log2_prob = torch.clamp(receiver_output_nn / np.log(2), min=min_real)
        # assert torch.allclose(r_probs.sum(-1), torch.ones_like(receiver_output_nn[..., 0]))
        # r_entropy_nn = torch.linalg.vecdot(-r_probs, r_log2_prob)

        for step in range(receiver_output.size(1)):
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
                aux_input)

            add_mask = message[:, step, 0] * not_eosed_before
            add_mask_nn = message_nn[:, step, 0] * not_eosed_before_nn

            z += add_mask
            loss += step_loss * add_mask
            loss += self.length_cost * (1 + step) * add_mask
            length += add_mask.detach() * (1 + step)
            length_nn += add_mask_nn * (1 + step)

            # aggregate aux info
            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0)
            aux_info['accuracy_nn'] = accuracy_nn[:, step] * add_mask_nn \
                + aux_info.get('accuracy_nn', 0)
            # aux_info['entropy_receiver_nn'] = \
            #     r_entropy_nn[:, step] * add_mask_nn \
            #     + aux_info.get('entropy_receiver_nn', 0)

            not_eosed_before = not_eosed_before * (1 - message[:, step, 0])
            not_eosed_before_nn *= 1 - message_nn[:, step, 0]

        # the remainder of the probability mass
        z += not_eosed_before
        loss += step_loss * not_eosed_before
        loss += self.length_cost * (step + 1) * not_eosed_before
        length += (step + 1) * not_eosed_before
        length_nn += (step + 1) * not_eosed_before_nn

        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

        aux_info["length"] = length - 1
        aux_info["length_nn"] = length_nn - 1
        if isinstance(self.sender.temperature, torch.nn.Parameter):
            aux_info["temperature"] = \
                self.sender.temperature.detach().clone()

        # compute expected_length on probs
        # TODO remove? this is more accurate but there's barely any difference
        # length_probs = torch.cat([
        #     probs[:, :1, 0],
        #     probs[:, 1:, 0] * torch.cumprod(1 - probs[:, :-1, 0], dim=1),
        # ], dim=1)
        # length_probs_nn = torch.cat([
        #     probs_nn[:, :1, 0],
        #     probs_nn[:, 1:, 0] * torch.cumprod(1 - probs_nn[:, :-1, 0], dim=1),
        # ], dim=1)
        # exp_length = (torch.arange(step + 1) * length_probs).sum(-1) + 1
        # exp_length_nn = (torch.arange(step + 1) * length_probs_nn).sum(-1) + 1
        # aux_info["length_probs"] = length_probs
        # aux_info["length_probs_nn"] = length_probs_nn
        # aux_info["exp_length"] = exp_length
        # aux_info["exp_length_nn"] = exp_length_nn

        logging_strategy = LoggingStrategy()

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            message=message.detach(),
            probs=probs.detach(),
            message_length=length.detach(),
            receiver_output=receiver_output.detach(),
            message_nn=message_nn.detach(),
            probs_nn=probs_nn.detach(),
            message_length_nn=length_nn.detach(),
            receiver_output_nn=receiver_output_nn.detach(),
            aux=aux_info)

        return loss.mean(), interaction
