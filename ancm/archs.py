import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import Categorical
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical, Categorical
from torch.distributions.utils import logits_to_probs, probs_to_logits
from ancm.util import crop_messages

from typing import Callable, Optional
from collections import defaultdict

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

    def forward(self, encoded_msg, _input, _aux_input=None):
        embedded_input = self.encoder(_input).tanh()
        energies = torch.matmul(embedded_input, encoded_msg.unsqueeze(-1))
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

        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.length_cost = length_cost

        channels = {
            'erasure': ErasureChannel,
            'deletion': DeletionChannel,
            'symmetric': SymmetricChannel,
        }

        if channel_type in channels.keys():
            self.channel = channels[channel_type](error_prob, vocab_size, device, seed)
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
        global update_bound

        if update_bound is None:
            update_bound = 2 ** 16 / sender_input.size(0)

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


# Gumbel-Softmax
def loss_gs(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=-1) == _labels).detach().float()

    cross_entropy = F.cross_entropy(receiver_output, _labels, reduction="none")

    if _sender_input.dim() == 2:  # VISA
        print(_labels)
        size = _receiver_input.shape
        feature_agreement = torch.empty(size[0], size[-1], dtype=torch.float)

        for i in range(size[-1]):
            feature_i = _receiver_input[:, :, i]
            unique_values = torch.unique(feature_i, dim=-1)
            print("feature", feature_i.shape)
            unique_features = torch.unique(_receiver_input[:,:, i], dim=-1)
            print(_receiver_input[:,:, i].shape, unique_features.shape, unique_features, i)

        features_values = torch.unique(_receiver_input)
        #print(relevant_features.shape)
        selected_object = _receiver_input[torch.arange(_labels.size(0)), _labels]
        # log_ratio = torch.log()
        # kld = 
    else:  # Obverter
        pass
    
    return cross_entropy, {"accuracy": acc.mean() * 100}


def gumbel_softmax_sample(
        logits: torch.Tensor,
        # channel: Optional[nn.Module] = None,
        temperature: float = 1.0,
        training: bool = True):

    if training:
        # learning based on a concrete distribution (original implementation)
        distr_concrete = RelaxedOneHotCategorical(logits=logits, temperature=temperature)
        sample_concrete = distr_concrete.rsample()

        # one hot symbol from a discrete distribution for evaluation
        logits_relaxed = F.gumbel_softmax(logits=logits.detach(), tau=temperature)
        distr_one_hot = OneHotCategorical(logits=logits_relaxed)
        sample_one_hot = distr_one_hot.sample()

    else:
        logits_relaxed = F.gumbel_softmax(logits=logits, tau=temperature)
        distr_one_hot = OneHotCategorical(logits=logits_relaxed)
        sample_concrete = sample_one_hot = distr_one_hot.sample()

    log_prob = distr_one_hot.logits
    min_real = torch.finfo(log_prob.dtype).min
    log_prob = torch.clamp(log_prob, min=min_real)
    p_log_p = log_prob * logits_to_probs(log_prob)
    entropy = -p_log_p.sum(-1)

    return sample_concrete, sample_one_hot, log_prob, entropy


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
        temperature_inference,
        cell="rnn",
        trainable_temperature=False,
        straight_through=False,
    ):
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
                torch.tensor([temperature_learning]), requires_grad=True
            )
        self.temperature_inference = temperature_inference

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

    def forward(self, x, aux_input=None, channel=None, apply_noise=False):
        prev_hidden = self.agent(x, aux_input)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))

        if self.training:
            temperature = self.temperature
        else:
            temperature = self.temperature_inference

        accumulated_eos_prob = torch.zeros(x.size(0))
        sequence, one_hots, log_probs, entropy = [], [], [], []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            smb_concrete, smb_one_hot, smb_logits, smb_entropy = \
                gumbel_softmax_sample(step_logits, temperature, self.training)

            prev_hidden = h_t
            e_t = self.embedding(smb_concrete)

            # apply noise after the speaker the sender
            # embeds the previous symbol
            if channel is not None:
                smb_concrete, accumulated_eos_prob = \
                    channel.concrete(smb_concrete, accumulated_eos_prob)

            sequence.append(smb_concrete)
            one_hots.append(smb_one_hot)
            log_probs.append(smb_logits)
            entropy.append(smb_entropy)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        one_hots = torch.stack(one_hots).permute(1, 0, 2)
        log_probs = torch.stack(log_probs).permute(1, 0, 2)
        entropy = torch.stack(entropy)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1

        sequence = torch.cat([sequence, eos], dim=1)
        if channel is not None and channel.p > 0.:
            one_hots = torch.cat([one_hots, eos[:, :, :-1]], dim=1)
        else:
            one_hots = torch.cat([one_hots, eos], dim=1)

        return sequence, one_hots.detach(), log_probs.detach(), entropy.detach()


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
        super(SenderReceiverRnnGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost

        channels = {
            'erasure': ErasureChannel,
            'deletion': DeletionChannel,
            'symmetric': SymmetricChannel,
        }

        if channel_type in channels.keys():
            self.channel = channels[channel_type](error_prob, vocab_size, device, seed)
        else:
            self.channel = None

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

    def forward(self, sender_input, labels, receiver_input, aux_input, apply_noise=True):
        # conrcete message representation includes noise
        message_concrete, message_one_hot, log_probs, entropy = \
            self.sender(sender_input, aux_input, self.channel, apply_noise)

        # get noise for the one hot message
        if self.channel:
            message_one_hot = self.channel(
                message_one_hot,
                message_length=None,
                apply_noise=apply_noise)

        message = message_concrete if self.training \
            else crop_messages(message_one_hot)

        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss = 0
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0

        aux_info = {}
        z = 0.0

        for step in range(receiver_output.size(1)):
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
                aux_input,
            )
            eos_mask = message_concrete[:, step, 0]  # always eos == 0

            add_mask = eos_mask * not_eosed_before
            z += add_mask
            loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
            expected_length += add_mask.detach() * (1.0 + step)

            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)

            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (
            step_loss * not_eosed_before
            + self.length_cost * (step + 1.0) * not_eosed_before
        )
        expected_length += (step + 1) * not_eosed_before

        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = value * not_eosed_before + aux_info.get(name, 0.0)

        aux_info["length"] = expected_length
        aux_info["entropy"] = entropy

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message_one_hot.detach(),
            message_length=expected_length.detach(),
            aux=aux_info,
        )

        return loss.mean(), interaction


# Noisy channel classes
class Channel(nn.Module):
    def __init__(self, error_prob, vocab_size, device, seed=42):
        super().__init__()
        self.p = torch.tensor([error_prob], requires_grad=False)
        self.vocab_size = vocab_size
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.device = device


class ErasureChannel(Channel):
    """
    Erases a symbol from a message with a given probability
    """

    def concrete(self, probs, aux=None):
        if self.p == 0:
            return probs, None

        # add an empty row to the original tensor
        non_eos_prob = probs[:, 1:].sum(-1, keepdim=True)  # has gradient
        erased_probs = torch.ones_like(probs[:, :1])  # doesnt have
        erased_probs = erased_probs * non_eos_prob * self.p  # has gradient
        probs = torch.cat([probs, erased_probs], dim=-1)

        # adjust the prob of successful transmission (not for EOS)
        probs[:, 1:] = probs[:, 1:] * (1 - self.p)  # has gradient
        probs[:, -1] = erased_probs.squeeze(1)

        assert torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]))
        return probs, None

    def forward(self, message, message_length=None, apply_noise=False, concrete=False):
        if self.p == 0. or not apply_noise:
            return message

        msg = message.detach() if message.dim() == 2 \
            else message.detach().argmax(dim=-1)

        # sample symbol indices to be erased, make sure EOS is not erased
        non_eos_ids = msg.nonzero()
        non_eos_target_ids = (
            torch.rand(non_eos_ids.size(0), generator=self.generator)
            < self.p
        ).int().to(self.device)

        target_ids = non_eos_ids[non_eos_target_ids.nonzero(), torch.arange(msg.dim())]

        if message.dim() == 2:  # Reinforce
            target_chunks = target_ids.t().chunk(chunks=2)
            message[target_chunks] = self.vocab_size

        else:  # GS (one hot)
            message = torch.cat([message, torch.zeros_like(message[:, :, :1])], dim=-1)

            # erased symbol vector
            erased_probs = torch.zeros_like(message[0, 0])
            erased_probs[-1] = 1.

            # erase
            target_chunks = target_ids.t().chunk(chunks=3)
            message[target_chunks] = erased_probs

        return message


class DeletionChannel(Channel):
    """
    Deletes a symbol with a given probability.
    """

    def concrete(self, probs, accumulated_eos_prob=None):
        if self.p == 0:
            return probs, None

        eos_prob = probs[:, 0]
        accumulated_eos_prob = accumulated_eos_prob + (1 - eos_prob) * self.p

        # adjust the prob of successful transmission (not for EOS)
        probs[:, 0] = probs[:, 0] + (1 - eos_prob) * self.p
        probs[:, 1:] = probs[:, 1:] * (1 - accumulated_eos_prob)  # has gradient

        assert torch.allclose(probs.sum(-1), torch.ones_like(probs[:, 0]))
        return probs

    def forward(self, message, message_length=None, apply_noise=False):
        if self.p == 0. or not apply_noise:
            return message

        msg = message.detach() if message.dim() == 2 \
            else message.detach().argmax(dim=-1)

        if message_length is None:
            message_length = find_lengths(msg)

        # sample symbol indices to be erased
        non_eos_ids = msg.nonzero()
        non_eos_target_ids = (
            torch.rand(non_eos_ids.size(0), generator=self.generator)
            < self.p
        ).int().to(self.device)

        target_ids = non_eos_ids[non_eos_target_ids.nonzero(), torch.arange(msg.dim())]
        target_chunks = target_ids.t().chunk(chunks=2)

        delete_ids = torch.zeros_like(msg).bool()
        delete_ids[target_chunks] = True
        keep_ids = torch.logical_not(delete_ids)

        num_deleted = torch.sum(delete_ids.int(), dim=1)

        if message.dim() == 2:  # Reinforce
            message = torch.stack([
                torch.cat(
                    [message[i][keep_ids[i]], torch.zeros(num_deleted[i], dtype=torch.int)])
                for i in range(message.size(0))
            ])
        else:  # GS
            # EOS vector
            eos_probs = torch.zeros_like(message[0, 0])
            eos_probs[0] = 1.

            message = torch.stack([
                torch.stack(
                    [
                        message[i, j]
                        for j in range(message.size(1)) if keep_ids[i, j]
                    ]
                    + [eos_probs] * num_deleted[i]
                ) for i in range(message.size(0))
            ])

        return message


class SymmetricChannel(Channel):
    """
    Replaces each symbol with a different symbol with a given probability.
    The replacement symbol is randomly sampled from a uniform distribution.
    """

    def forward(self, message, message_length=None, apply_noise=False):
        if self.p == 0. or not apply_noise:
            return message

        msg = message.detach() if message.dim() == 2 \
            else message.detach().argmax(dim=-1)

        non_eos_ids = msg.nonzero()
        non_eos_target_ids = (
            torch.rand(non_eos_ids.size(0), generator=self.generator)
            < self.p
        ).int().to(self.device)
        target_ids = non_eos_ids[non_eos_target_ids.nonzero(), torch.arange(msg.dim())]
        target_chunks = target_ids.t().chunk(chunks=2)

        non_eos_symbols = torch.arange(1, self.vocab_size).expand(
            target_ids.size(0), -1)

        actual_symbols = msg[target_chunks]
        actual_symbols = actual_symbols.expand(self.vocab_size - 1, -1).t()

        # exclude actual message symbols from the set of candidates
        # each symbol of the message has vocab_size - 2 possible replacements
        candidate_ids = (non_eos_symbols != actual_symbols)
        candidate_symbols = non_eos_symbols[candidate_ids].view(-1, self.vocab_size - 2)

        # sample the replacement symbol to be used
        replacement_ids = torch.randint(
            high=self.vocab_size - 2,
            size=(target_ids.size(0),),
            generator=self.generator,
            device=self.device)

        replacement_chunks = (torch.arange(target_ids.size(0)), replacement_ids)
        replacement_symbols = candidate_symbols[replacement_chunks]

        if message.dim() == 2:  # Reinforce
            message[target_chunks] = replacement_symbols
        else:  # GS
            def replacement_probs(ind):
                row = [0] * self.vocab_size
                row[ind] = 1
                return torch.tensor(row)

            if len(replacement_symbols) > 0:
                replaced_probs = torch.stack([
                    replacement_probs(s) for s in replacement_symbols
                ]).to(message)
                target_chunks = target_ids.t().chunk(chunks=3)
                message[target_chunks] = replaced_probs

        return message
