import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
from torch.distributions.utils import logits_to_probs  #, probs_to_logits

# from ancm.util import crop_messages

from typing import Callable, Optional
from collections import defaultdict

import egg.core as core
from egg.core.reinforce_wrappers import RnnEncoder
from egg.core.baselines import Baseline, BuiltInBaseline, NoBaseline, MeanBaseline
from egg.core.interaction import LoggingStrategy
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

        channel_types = {
            'erasure': ErasureChannel,
            'deletion': DeletionChannel,
            'symmetric': SymmetricChannel,
        }
        
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.length_cost = length_cost
        self.channel = channel_types[channel_type](error_prob, seed)
        self.baselines = defaultdict(baseline_type)
        self.train_logging_strategy = LoggingStrategy() \
            if train_logging_strategy is None \
            else train_logging_strategy
        self.test_logging_strategy = LoggingStrategy() \
            if test_logging_strategy is None \
            else test_logging_strategy

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
        global update_bound

        if update_bound is None:
            update_bound = 2 ** 16 / sender_input.size(0)

        message, log_prob_s, entropy_s = sender(sender_input, aux_input)
        message_length_nn = find_lengths(message)
        if self.channel and error_prob != 0:
            message = self.channel(
                message=message,
                vocab_size=self.vocab_size,
                lengths=message_length_nn,
                apply_noise=apply_noise)
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


# Gumbel-Softmax
def loss_gs(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=-1) == _labels).detach().float()
    cross_entropy = F.cross_entropy(receiver_output, _labels, reduction="none")
    # if _sender_input.dim() == 2:  # VISA
    #     size = _receiver_input.shape
    #     all_ids = torch.arange(_receiver_input.size(0))
    #     # priors = _receiver_input.sum(1) / size[1]
    #     # print(priors, priors.shape)
    #     target_object = _receiver_input[all_ids, _labels]
    #     selected_object = torch.zeros_like(target_object)
    #     for i in range(size[1]):
    #        prob_i = logits_to_probs(receiver_output[all_ids, i])
    #        selected_object += _receiver_input[all_ids, i] * prob_i.unsqueeze(-1)
    #    min_positive = torch.finfo(receiver_output.dtype).tiny
    #    target_object = torch.clamp(target_object, min=min_positive)
    #    selected_object = torch.clamp(selected_object, min=min_positive)
    #    log_ratio = torch.log(target_object) - torch.log(selected_object)
    #    kld = (target_object * log_ratio).sum(-1)
    #    # print(log_ratio, kld)
    # else:  # Obverter
    #    pass
    return cross_entropy, {"accuracy": acc.mean() * 100}


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float = 1.0):

    distr = RelaxedOneHotCategorical(logits=logits, temperature=temperature)
    sample = distr.rsample()
    
    min_real = torch.finfo(logits.dtype).min
    min_positive = torch.finfo(logits.dtype).tiny
    probs = torch.clamp(distr.probs, min=min_positive)
    log2_prob = torch.clamp(torch.log2(probs), min=min_real)
    entropy = (-log2_prob * distr.probs).sum(-1)
    # log2_prob = torch.clamp(log2_prob, min=min_real)
    # entropy = distr.base_dist._categorical.entropy()

    return sample, distr.logits, entropy


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
            temperature_learning,
            cell="rnn",
            trainable_temperature=False,
            straight_through=False,):

        super(RnnSenderGS, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if not trainable_temperature:
            self.temperature = temperature_learning
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature_learning]),
                requires_grad=True)

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

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = self.agent(x, aux_input)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))

        sequence, logits, entropy = [], [], []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            #print(h_t)
            step_logits = self.hidden_to_output(h_t)
            #print(step_logits)
            symbols, symbol_logits, symbol_entropy = \
                gumbel_softmax_sample(step_logits, self.temperature)

            prev_hidden = h_t
            e_t = self.embedding(symbols)

            sequence.append(symbols)
            logits.append(symbol_logits)
            entropy.append(symbol_entropy)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)
        logits = torch.stack(logits).permute(1, 0, 2)
        entropy = torch.stack(entropy, dim=1)
        entropy = torch.cat([entropy, torch.zeros_like(entropy[:, :1])], dim=-1)

        return sequence, logits, entropy


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
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
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
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in the callbacks.

        """

        channel_types = {
            'erasure': ErasureChannel,
            'deletion': DeletionChannel,
            'symmetric': SymmetricChannel,
        }

        super(SenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost
        self.channel = channel_types[channel_type](error_prob, device, seed)
        self.train_logging_strategy = LoggingStrategy() \
            if train_logging_strategy is None \
            else train_logging_strategy
        self.test_logging_strategy = LoggingStrategy() \
            if test_logging_strategy is None \
            else test_logging_strategy

    def forward(self, sender_input, labels, receiver_input, aux_input, apply_noise=True):
        message, logits, entropy = self.sender(sender_input, aux_input)

        message, entropy, channel_aux = self.channel(
            message,
            entropy=entropy,
            apply_noise=apply_noise)

        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, expected_length, z = 0.0, 0.0, 0.0
        not_eosed_before = torch.ones(
            receiver_output.size(0)).to(receiver_output.device)
        aux_info = {}

        for step in range(receiver_output.size(1)):
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
                aux_input)

            # additional accumulated EOS prob
            #if isinstance(self.channel, DeletionChannel) and self.training:
            #    eos_mask = message[:, step, 0] + channel_aux.sum(-1).detach()
            #    # not_eosed_before -= channel_aux[:, step]
            #elif isinstance(self.channel, DeletionChannel) and not self.training:
            if isinstance(self.channel, DeletionChannel):
                eos_mask = channel_aux + step < not_eosed_before.detach
            else:
                eos_mask = message[:, step, 0]

            add_mask = eos_mask * not_eosed_before  # s per messagetep_eos_prob
            z += add_mask
            expected_length += add_mask.detach() * (1.0 + step)
            loss += step_loss * add_mask
            loss += self.length_cost * (1.0 + step) * add_mask

            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)

            # binary entropy of prob. of ending the sequence at this step
            # h_eos_step = binary_entropy(add_mask)

            h_not_eosed = tensor_binary_entropy(not_eosed_before).detach()
            entropy['message'] = h_not_eosed \
                + not_eosed_before.detach() * entropy['symbol'][:, step]
                # + (1 - prob_not_eosed) * 0

            # if apply_noise:
            #     entropy['message_entropy_nn'] += h_not_eosed \
            #         + prob_not_eosed_nn * entropy['symbol_entropy_nn'][:, step]
            #         # + (1 - prob_not_eosed_nn * 0

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += step_loss * not_eosed_before
        loss += self.length_cost * (step + 1.0) * not_eosed_before

        expected_length += (step + 1) * not_eosed_before
        z += not_eosed_before

        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

        aux_info["length"] = expected_length
        for key, value in entropy.items():
            aux_info[key] = value.detach()

        logging_strategy = self.train_logging_strategy \
            if self.training else self.test_logging_strategy

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=expected_length.detach(),
            aux=aux_info)

        return loss.mean(), interaction


def tensor_binary_entropy(p: torch.Tensor):
    q = 1 - p
    min_real = torch.finfo(p.dtype).min
    log2_p = torch.clamp(torch.log2(p), min=min_real)
    log2_q = torch.clamp(torch.log2(q), min=min_real)
    return -p * log2_p - q * log2_q


class Channel(nn.Module):
    def __init__(self, error_prob, device, seed=42):
        super().__init__()
        self.p = torch.tensor(error_prob, requires_grad=False)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.device = device

    def forward(self, messages, apply_noise, aux=None, *args, **kwargs):
        # GS
        if messages.dim() == 3:
            entropies = kwargs['entropy']
            symbols, symbol_entropies = [], []

            for i in range(messages.size(1)):
                symbol_i, entropy_i = messages[:, i], entropies[:, i]
                symbol_i, entropy_i, aux = self.gs(
                    symbol_i, entropy_i, aux, apply_noise)
                symbols.append(symbol_i)
                symbol_entropies.append(entropy_i)

            messages = torch.stack(symbols).permute(1, 0, 2)
            symbol_entropies = torch.stack(symbol_entropies).t()

            entropy_dict = {
                'message': symbol_entropies.sum(-1),
                'symbol': symbol_entropies,
                'message_nn': entropies.sum(-1),
                'symbol_nn': entropies,
            }

            return messages, entropy_dict, aux

        # Reinforce
        else:
            return self.rf(messages, apply_noise, *args, **kwargs)


class ErasureChannel(Channel):
    """
    Erases a symbol from a message with a given probability
    """

    def gs(self, probs, entropy, aux=None, apply_noise=True):
        if not apply_noise:
            # if aux == False, append a column with 0 probabilities
            placeholder_probs = torch.zeros_like(probs[:, :1])
            probs = torch.cat([probs, placeholder_probs], dim=-1)
            return probs, entropy, None

        elif self.training:
            # append an empty column to the original tensor; each non-EOS
            # symbol is swapped with a special symbol with probability p
            non_eos_prob = probs[:, 1:].sum(-1, keepdim=True)
            erased_probs = torch.ones_like(probs[:, :1])
            erased_probs = erased_probs * non_eos_prob * self.p
            probs = torch.cat([probs, erased_probs], dim=-1)

            # adjust the prob of successful transmission (not for EOS)
            probs[:, 1:] = probs[:, 1:] * (1 - self.p)
            probs[:, -1] = erased_probs.squeeze(1)

            # we don't adjust for (future) message length, as here we're
            # computing entropy for the specific symbols transmitted/generated
            entropy += tensor_binary_entropy(self.p)

            assert torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]))
            return probs, entropy, None

        else:
            # apply argmax, exclude EOS, sample batch rows to be replaced
            discrete_symbols = probs.argmax(-1)
            non_eos_mask = discrete_symbols != 0
            non_eos_ids = torch.arange(len(probs))[non_eos_mask]
            target_mask = torch.rand(
                non_eos_mask.sum(),
                generator=self.generator,
                device=self.device
            ) < self.p
            target_ids = non_eos_ids.flatten()[target_mask]

            # append a column for erased symbols
            erased_probs = torch.zeros_like(probs[:, :1])
            probs = torch.cat([probs, erased_probs], dim=-1)

            if target_ids.numel() == 0:
                return probs, entropy, None

            # create a replacement probability array and replace
            erased_probs = torch.zeros(
                target_ids.size(0),
                probs.size(1),
                dtype=probs.dtype,
                requires_grad=False)
            erased_probs[:, 0] = probs[target_ids, 0]  # EOS prob
            erased_probs[:, -1] = 1 - probs[target_ids, 0]  # erased prob
            index = target_ids.expand(probs.size(1), -1).t()

            # replace & adjust the entropy
            probs.scatter_(0, index, erased_probs)
            entropy[target_ids] += tensor_binary_entropy(erased_probs[:, -1])

            return probs, entropy, aux

    def reinforce(self, messages, vocab_size=None, lengths=None, apply_noise=False):
        if not apply_noise:
            return messages

        # sample symbol indices to be erased, make sure EOS is not erased
        non_eos_ids = messages.nonzero()
        non_eos_target_ids = torch.rand(
            non_eos_ids.size(0),
            generator=self.generator,
            device=self.device
        ) < self.p

        target_ids = non_eos_ids[non_eos_target_ids.nonzero(), torch.arange(messages.dim())]
        target_chunks = target_ids.t().chunk(chunks=2)
        messages[target_chunks] = vocab_size

        return messages


class DeletionChannel(Channel):
    """
    Deletes a symbol with a given probability.
    """

    def gs(self, probs, entropy, aux, apply_noise=True):
        if not apply_noise:
            return probs, entropy, aux

        if self.training or True:
            pass
            # apply argmax, exclude EOS, sample batch rows to be replaced
            # target_mask = torch.rand(
            #     torch.arange(len(probs)),
            #     generator=self.generator,
            #     device=self.device
            # ) < self.p
            # target_ids = probs[target_mask]
            # probs = torch.where(target_mask, -1, probs)
            # non_eos_ids[target_mask] = -1
            #print(non_eos_ids[target_mask])
            #print(target_ids.shape)

            # sample non-eos symbols to be replaced
            # target_ids = torch.rand(  # (32, 9)
            #     probs[:, 1:].size(),
            #     generator=self.generator,
            #     device=self.device
            # ) < self.p
            # target_mask = target_ids.nonzero(as_tuple=True)
            # non_eos_ids = torch.arange(1, probs.size(-1)).expand(
            #     target_mask[-1].size(0), -1)  # (60, 9)


            # aux: accumulated additional EOS probability
            #eos_prob = probs[:, 0]
            #if aux is None:
            #    accumulated_eos_prob = torch.zeros_like(probs[:, 0])
            #    aux = ((1 - eos_prob) * self.p).unsqueeze(-1)
            #else:
            #    accumulated_eos_prob = aux.sum(-1)
            #    aux = torch.cat([
            #        aux,
            #        self.p * (1 - accumulated_eos_prob - eos_prob).unsqueeze(-1),
            #    ], dim=-1)

            #print('before', probs)

            # adjust the prob of successful transmission (not for EOS)
            #adjusted_eos = probs[:, 0]
            #adjusted_non_eos = probs[:, 1:]
            #adjusted_non_eos = adjusted_non_eos * (
            #    1 - accumulated_eos_prob - eos_prob).unsqueeze(-1)
            #adjusted_non_eos = adjusted_non_eos / (1 - eos_prob).unsqueeze(-1)
            #adjusted_eos = adjusted_eos + accumulated_eos_prob
            #adjusted_eos = adjusted_eos + probs[:, 1:].sum(-1) * self.p
            #adjusted_non_eos = adjusted_non_eos * (1 - self.p)
            #adjusted = torch.cat(
            #    [adjusted_eos.unsqueeze(-1), adjusted_non_eos], dim=-1)

            #print('after', probs)
            #entropy = entropy + tensor_binary_entropy(self.p)

            #assert torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]))
            #return adjusted, entropy, aux.detach()
        else:
            # aux: number of removed symbols per message
            if aux is None:
                aux = torch.zeros(probs.size(0), dtype=torch.int)

            # apply argmax, exclude EOS, sample batch rows to be replaced
            discrete_symbols = probs.argmax(-1)
            non_eos_mask = discrete_symbols != 0
            non_eos_ids = torch.arange(len(probs))[non_eos_mask]
            target_mask = torch.rand(
                non_eos_mask.sum(),
                generator=self.generator,
                device=self.device
            ) < self.p
            target_ids = non_eos_ids.flatten()[target_mask]

            #print(target_ids.size(), target_ids.numel(), target_ids)
            if target_ids.numel() == 0:
                return probs, entropy, aux

            # create a replacement prob array
            source = torch.zeros(
                target_ids.numel(), probs.size(1),
                dtype=probs.dtype,
                requires_grad=False)
            #print(source)
            source[target_ids, 0] = -1
            index = target_ids.expand(probs.size(1), -1).t()

            #print('index', index, index.shape)
            #print('source', source)
            #print('target_ids', target_ids[:8])
            #print('actual smb', discrete_symbols[target_ids[:8]])
            #print('before', probs)

            probs.scatter_(0, index, source)

            #print('after', probs)

            aux[target_ids] += 1

            # recompute entropy on the updated distribution
            min_real = torch.finfo(logits.dtype).min
            min_positive = torch.finfo(logits.dtype).tiny
            log2_prob = torch.clamp(torch.log2(probs), min=min_real)
            entropy = (-log2_prob * probs).sum(-1)

            # TODO check if it works

            return probs, entropy, aux

    def delete_symbols(self, messages, aux):
        for i in messages.size(1):
            symbol_i = messages[:, i, :]
            mask = symbol_i == -1
            if symbol_i[mask].numel() == 0:
                continue

    def reinforce(self, messages, vocab_size=None, lengths=None, apply_noise=False):
        if not apply_noise:
            return messages

        if lengths is None:
            lengths = find_lengths(messages)

        # sample symbol indices to be erased
        non_eos_ids = messages.nonzero()
        non_eos_target_ids = torch.rand(
            non_eos_ids.size(0),
            generator=self.generator,
            device=self.device
        ) < self.p

        target_ids = non_eos_ids[non_eos_target_ids.nonzero(), torch.arange(messages.dim())]
        target_chunks = target_ids.t().chunk(chunks=2)

        delete_ids = torch.zeros_like(messages).bool()
        delete_ids[target_chunks] = True
        keep_ids = torch.logical_not(delete_ids)

        num_deleted = torch.sum(delete_ids.int(), dim=1)

        messages = torch.stack([
            torch.cat(
                [messages[i][keep_ids[i]], torch.zeros(num_deleted[i], dtype=torch.int)])
            for i in range(messages.size(0))
        ])

        return messages


class SymmetricChannel(Channel):
    """
    Replaces each symbol with a different symbol with a given probability.
    The replacement symbol is randomly sampled from a uniform distribution.
    """

    def gs(self, probs, entropy, aux=None, apply_noise=True):
        if self.training:
            # sample non-eos symbols to be replaced
            target_ids = torch.rand(  # (32, 9)
                probs[:, 1:].size(),
                generator=self.generator,
                device=self.device
            ) < self.p
            target_mask = target_ids.nonzero(as_tuple=True)
            non_eos_ids = torch.arange(1, probs.size(-1)).expand(
                target_mask[-1].size(0), -1)  # (60, 9)

            # select remaining non-EOS symbols
            candidate_mask = torch.logical_not(
                non_eos_ids - 1 == target_mask[-1].unsqueeze(-1))  # 60, 9
            candidate_symbols = non_eos_ids[candidate_mask].view(
                candidate_mask.size(0), -1)

            # get original probabilities
            original_target_probs = probs[target_mask[0], target_mask[1] - 1]
            original_non_target_probs = torch.gather(
                probs[target_mask[0]], 1, candidate_symbols)

            # the probability of sampling the target symbol decreases to 0
            # and is uniformy divided among remaining non-EOS symbols
            adjustment = torch.zeros_like(probs)
            adjustment[target_mask[0], target_mask[1] - 1] -= original_target_probs
            adjustment[target_mask[0]].scatter_(
                1, candidate_symbols,
                original_non_target_probs
                + original_target_probs.unsqueeze(-1) / (probs.size(1) - 2))
            
            probs = probs + adjustment
            entropy += tensor_binary_entropy(self.p.expand(probs.size(0)))
            entropy += torch.log2(torch.tensor(probs.size(1) - 2))

            return probs, entropy, None

        else:
            # apply argmax, exclude EOS, sample batch rows to be replaced
            discrete_symbols = probs.argmax(-1)
            non_eos_mask = discrete_symbols != 0
            non_eos_ids = torch.arange(len(probs))[non_eos_mask]
            target_mask = torch.rand(
                non_eos_mask.sum(),
                generator=self.generator,
                device=self.device
            ) < self.p
            target_ids = non_eos_ids.flatten()[target_mask]

            if target_ids.numel() == 0:
                return probs, entropy, None

            size = target_ids.size(0)

            # find and select possible non-EOS replacements
            candidates = torch.arange(1, probs.size(-1)).expand(size, -1)
            candidate_mask = (
                candidates != discrete_symbols[target_ids].unsqueeze(-1))  # 4, 9
            candidate_symbols = candidates[candidate_mask].view(
                candidate_mask.size(0), -1)  # (4, 8), ready to mask
            replacement_ids = torch.randint(
                size=(size,),
                high=candidate_symbols.size(-1) - 1,
                generator=self.generator,
                device=self.device)
            replaced_symbols = torch.gather(
                candidate_symbols, -1, replacement_ids.unsqueeze(-1)
            ).squeeze().to(torch.int64)

            # create a replacement prob array
            original_non_eos_probs = probs[target_ids, 1:].sum(-1)
            source = torch.zeros(
                size, probs.size(1) - 1,
                dtype=probs.dtype, requires_grad=False)
            mask = (torch.arange(size), replaced_symbols)
            source[mask] = original_non_eos_probs
            source = torch.cat([probs[target_ids, :1], source], dim=-1)
            index = target_ids.expand(probs.size(1), -1).t()

            # print('index', index, index.shape)
            # print('source', source)
            # print('target_ids', target_ids[:8])
            # print('actual smb', discrete_symbols[target_ids[:8]])
            # print('candidate_symbols', candidate_symbols[:8])
            # print('replacement', replaced_symbols[:8])
            # print('before', probs)

            # replace
            probs.scatter_(0, index, source)

            # the resulting entropy for all messages increases
            entropy += tensor_binary_entropy(self.p.expand(probs.size(0)))
            entropy += torch.log2(torch.tensor(probs.size(1) - 2))

            # TODO test, test

            return probs, entropy, None

    def reinforce(self, messages, vocab_size=None, lengths=None, apply_noise=True):
        if not apply_noise:
            return messages

        non_eos_ids = messages.nonzero()
        non_eos_target_ids = torch.rand(
            non_eos_ids.size(0),
            generator=self.generator,
            device=self.device) < self.p
        target_ids = non_eos_ids[non_eos_target_ids.nonzero(), [0, 1]]
        target_chunks = target_ids.t().chunk(chunks=2)

        non_eos_symbols = torch.arange(1, vocab_size).expand(
            target_ids.size(0), -1)

        discrete_symbols = messages[target_chunks]
        discrete_symbols = discrete_symbols.expand(vocab_size - 1, -1).t()

        # exclude actual messages symbols from the set of candidates
        # each symbol of the messages has vocab_size - 2 possible replacements
        candidate_ids = (non_eos_symbols != discrete_symbols)
        candidate_symbols = non_eos_symbols[candidate_ids].view(-1, vocab_size - 2)

        # sample the replacement symbol to be used
        replacement_ids = torch.randint(
            high=vocab_size - 2,
            size=(target_ids.size(0),),
            generator=self.generator,
            device=self.device)

        replacement_chunks = (torch.arange(target_ids.size(0)), replacement_ids)
        replacement_symbols = candidate_symbols[replacement_chunks]

        messages[target_chunks] = replacement_symbols

