import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
from argparse import Namespace

from src.channels import (
    NoChannel,
    ErasureChannel,
    DeletionChannel,
)
from src.interaction import LoggingStrategy


class SeeingConvNet(nn.Module):
    def __init__(self, input_shape: int, n_hidden: int):
        super(SeeingConvNet, self).__init__()

        # adapted from Denamganaï, Missaoui and Walker, 2023
        n_filters = 32
        kwargs = dict(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )

        # Define the sequence of convolutional layers, same as Lazaridou paper 2018
        self.convnet = nn.Sequential(
            nn.Conv2d(3, n_filters, **kwargs),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),

            nn.Conv2d(n_filters, n_filters, **kwargs),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),

            nn.Conv2d(n_filters, n_filters * 2, **kwargs),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),

            nn.Conv2d(n_filters * 2, n_filters * 2, **kwargs),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(),
        )

        self.out_features = (
            (
                input_shape + 2 * kwargs['padding'] - 1
                - kwargs['dilation'] * (kwargs['kernel_size'] - 1)
            ) // kwargs['stride'] + 1
        ) ** 2

        self.fc = nn.Sequential(
            nn.Linear(self.out_features, n_hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.dim() == 5:
            batch_size, n_candidates, *image_dims = x.shape
            x = self.convnet(x.view(batch_size * n_candidates, *image_dims))
            return self.fc(x.view(batch_size, n_candidates, -1))
        else:
            x = self.convnet(x).view(x.size(0), -1)
            return self.fc(x)


class RnnSenderGS(nn.Module):
    """
    Gumbel Softmax wrapper for Sender that outputs variable-length sequence of
    symbols. The user-defined `agent` takes an input and outputs an initial
    hidden state vector for the RNN cell; `RnnSenderGS` then unrolls this RNN
    for the `max_len` symbols. The end-of-sequence logic is supposed to be
    handled by the game implementation. Supports vanilla RNN ('rnn'), GRU
    ('gru'), and LSTM ('lstm') cells.

    Based on `egg/core/gs_wrappers.py`.
    """

    def __init__(self, opts: Namespace):
        assert opts.max_len >= 1, "Cannot have a max_len below 1"
        super(RnnSenderGS, self).__init__()
        self.max_len = opts.max_len

        input_encoder = SeeingConvNet if opts.image_input else nn.Linear
        self.encoder = input_encoder(opts.n_features, opts.sender_hidden)
        self.hidden_to_output = nn.Linear(opts.sender_hidden, opts.vocab_size)
        self.embedding = nn.Linear(opts.vocab_size, opts.embedding)
        self.sos_embedding = nn.Parameter(torch.zeros(opts.embedding))
        self.vocab_size = opts.vocab_size

        self.temperature_eval = opts.evaluation_temperature
        self.temperature_max = opts.temperature_max
        self.hidden_to_inv_temperature = nn.Sequential(
            nn.Linear(opts.sender_hidden, 1, bias=False),
            nn.Softplus(1),
        )

        cells = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
            'lstm': nn.LSTMCell,
        }
        if opts.sender_cell not in cells:
            raise ValueError(f"Unknown RNN Cell: {opts.sender_cell}")

        self.cell = cells[opts.sender_cell](
            input_size=opts.embedding,
            hidden_size=opts.sender_hidden,
        )
        self.reset_parameters()

    def step_temperature(self, hidden):
        if self.training or self.temperature_eval is None:
            # predict inverse temperature and scale it
            tau_0 = self.temperature_max ** -1
            return (self.hidden_to_inv_temperature(hidden) + tau_0) ** -1
        else:
            return self.temperature_eval

    def gumbel_softmax_sample(self, logits, temperature):
        if self.training:
            sample = RelaxedOneHotCategorical(
                logits=logits,
                temperature=temperature,
            ).rsample()
        elif temperature is None or (not isinstance(temperature, torch.Tensor) and temperature == 0):
            # apply argmax and return a one hot tensor and original log-probs
            # log-probs do not reflect the distribution of discrete symbols
            # and hence cannot be used to compute entropy of resulting messages
            size = logits.size()
            indexes = logits.argmax(-1)
            one_hots = torch.zeros_like(logits).view(-1, size[-1])
            one_hots.scatter_(1, indexes.view(-1, 1), 1)
            one_hots = one_hots.view(size)
            return one_hots, logits
            # elif temperature is not None and temperature > 0:
        else:
            # argmax of GS sample is equivalent to sampling from the original
            # distribution: Softmax(logits, temperature)
            sample = OneHotCategorical(logits=logits / temperature).sample()

        if isinstance(temperature, torch.Tensor):
            logits = (logits.detach() / temperature.detach()).log_softmax(-1)
        else:
            logits = (logits.detach() / temperature).log_softmax(-1)

        return sample, logits

        # logits = (logits.detach() / temperature.detach()).log_softmax(-1) \
        #     if isinstance(temperature, torch.Tensor) \
        #     else (logits.detach() / temperature).log_softmax(-1)
        # return sample, logits

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = self.encoder(x).tanh()
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))

        sequence, probs, temperature = [], [], []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            step_temperature = self.step_temperature(h_t)

            symbols, symbol_probs = self.gumbel_softmax_sample(
                step_logits,
                step_temperature,
            )

            prev_hidden = h_t
            e_t = self.embedding(symbols)

            sequence.append(symbols)
            probs.append(symbol_probs)
            temperature.append(step_temperature)

        sequence = torch.stack(sequence, dim=1)
        probs = torch.stack(probs, dim=1)
        temperature = torch.cat(temperature, dim=-1)

        return sequence, probs, temperature


class RnnReceiverGS(nn.Module):
    def __init__(self, opts):
        super(RnnReceiverGS, self).__init__()

        cells = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
            'lstm': nn.LSTMCell,
        }

        if opts.receiver_cell not in cells:
            raise ValueError(f"Unknown RNN Cell: {opts.receiver_cell}")

        input_encoder = SeeingConvNet if opts.image_input else nn.Linear
        vocab_size = opts.vocab_size + 1 \
            if opts.channel == 'erasure' else opts.vocab_size

        self.cell = cells[opts.receiver_cell](
            input_size=opts.embedding,
            hidden_size=opts.receiver_hidden)
        self.message_encoder = nn.Linear(vocab_size, opts.embedding)
        self.input_encoder = input_encoder(
            opts.n_features,
            opts.receiver_hidden)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, message, _input=None, aux_input=None):
        embedded_input = self.input_encoder(_input).tanh()
        embedded_message = self.message_encoder(message)

        prev_hidden = None
        prev_c = None

        outputs = []
        for step in range(message.size(1)):
            e_t = embedded_message[:, step]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            # h_t: message embedding
            energies = torch.matmul(embedded_input, h_t.unsqueeze(-1))
            outputs.append(self.logsoft(energies.squeeze()))
            prev_hidden = h_t

        return torch.stack(outputs, dim=1)


class SenderReceiverRnnGS(nn.Module):
    def __init__(
        self,
        opts: Namespace,
        device: torch.device = torch.device("cpu"),
    ):
        super(SenderReceiverRnnGS, self).__init__()
        channel_types = {
            'none': NoChannel,
            'erasure': ErasureChannel,
            'deletion': DeletionChannel,
        }

        self.sender = RnnSenderGS(opts)
        self.receiver = RnnReceiverGS(opts)
        self.channel = channel_types[opts.channel](opts, device)

        self.length_cost = opts.length_cost
        self.warmup = iter(True for _ in range(opts.warmup_steps))

    @staticmethod
    def loss(s_input, message, r_input, r_output, labels, aux_input):
        cross_entropy = F.cross_entropy(r_output, labels, reduction="none")
        accuracy = (r_output.detach().argmax(dim=-1) == labels).float()
        return cross_entropy, {'accuracy': accuracy * 100}

    def forward(self, sender_input, labels, receiver_input, aux_input):
        warmup = next(self.warmup, False)

        sender_output = self.sender(sender_input, aux_input)
        temperatures = sender_output[-1]

        # pass messages and symbol probabilities through the channel
        message, logits, message_nn, logits_nn = self.channel(*sender_output)

        # append EOS to each message
        eos = torch.zeros_like(message[:, :1, :])
        eos[:, 0, 0] = 1
        # log_eos = torch.log(eos)  # .log_softmax(0)#clamp(min=torch.finfo(eos.dtype).min)
        message = torch.cat([message, eos], dim=1)
        logits = torch.cat([logits, torch.log(eos)], dim=1)
        message_nn = torch.cat([message_nn, eos], dim=1)  # no noise
        logits_nn = torch.cat([logits_nn, torch.log(eos)], dim=1)

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
        not_eosed_before = torch.ones(message.size(0)).to(message.device)
        not_eosed_before_nn = not_eosed_before.detach().clone()
        aux_info = {}

        # compute aux info values without noise
        accuracy_nn = (
            receiver_output_nn.argmax(-1) == labels.unsqueeze(-1)
        ).float() * 100

        for step in range(receiver_output.size(1)):
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
                aux_input,
            )

            add_mask = message[:, step, 0] * not_eosed_before
            add_mask_nn = message_nn[:, step, 0] * not_eosed_before_nn

            z += add_mask
            loss += step_loss * add_mask
            if not warmup:
                loss += self.length_cost * step * add_mask
            length += add_mask.detach() * (1 + step)
            length_nn += add_mask_nn * (1 + step)

            # aggregate aux info
            for name, value in step_aux.items():
                aux_info[name] = (
                    value * add_mask.detach() + aux_info.get(name, 0)
                )
            aux_info['accuracy_nn'] = (
                accuracy_nn[:, step] * add_mask_nn.detach()
                + aux_info.get('accuracy_nn', 0)
            )

            # aggregate temperature weighted by eos probs without noise
            # (this only matters for the deletion channel)
            step_temperature = (
                temperatures[:, step] if step < temperatures.size(1)
                else temperatures[:, -1]
            ).detach()
            aux_info['temperature'] = (
                step_temperature * add_mask_nn.detach()
                + aux_info.get('temperature', 0)
            )

            not_eosed_before = not_eosed_before * (1 - message[:, step, 0])
            not_eosed_before_nn *= 1 - message_nn[:, step, 0].detach()

        # the remainder of the probability mass
        z += not_eosed_before
        loss += step_loss * not_eosed_before
        if not warmup:
            loss += self.length_cost * step * not_eosed_before
        length += (step + 1) * not_eosed_before.detach()
        length_nn += (step + 1) * not_eosed_before_nn

        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        for name, value in step_aux.items():
            aux_info[name] = (
                value * not_eosed_before.detach() + aux_info.get(name, 0.0)
            )
        aux_info["temperature"] += step_temperature * not_eosed_before_nn

        aux_info["length"] = (length - 1).detach()
        aux_info["length_nn"] = (length_nn - 1).detach()

        interaction = LoggingStrategy().filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            message=message.detach(),
            logits=logits.detach(),
            message_length=length.detach(),
            receiver_output=receiver_output.detach(),
            message_nn=message_nn.detach(),
            logits_nn=logits_nn.detach(),
            message_length_nn=length_nn.detach(),
            receiver_output_nn=receiver_output_nn.detach(),
            aux=aux_info,
        )

        return loss.mean(), interaction
