import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
from argparse import Namespace
from collections import defaultdict, OrderedDict

from src.channels import (
    NoChannel,
    ErasureChannel,
    DeletionChannel,
)
from src.interaction import LoggingStrategy


channels = {
    None: NoChannel,
    'erasure': ErasureChannel,
    'deletion': DeletionChannel,
}

rnn_cells = {
    'rnn': nn.RNNCell,
    'gru': nn.GRUCell,
    'lstm': nn.LSTMCell,
}


class SeeingConvNet(nn.Module):
    def __init__(self, input_shape: int, n_hidden: int):
        super(SeeingConvNet, self).__init__()

        n_filters = 32
        out_features = 1024
        kwargs = dict(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )

        # based on Denamganaï, Missaoui and Walker, 2023
        self.convnet = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, n_filters, **kwargs)),
            ('norm1', nn.BatchNorm2d(n_filters)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(n_filters, n_filters, **kwargs)),
            ('norm2', nn.BatchNorm2d(n_filters)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(n_filters, n_filters * 2, **kwargs)),
            ('norm3', nn.BatchNorm2d(n_filters * 2)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(n_filters * 2, n_filters * 2, **kwargs)),
            ('norm4', nn.BatchNorm2d(n_filters * 2)),
            ('relu4', nn.ReLU()),
        ]))
        self.fc = nn.Sequential(
            nn.Linear(out_features, n_hidden),
            nn.ReLU(),
        )

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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
        self.temperature = opts.temperature

        if opts.sender_cell not in rnn_cells:
            raise ValueError(f"Unknown RNN cell: {opts.sender_cell}")

        input_encoder = SeeingConvNet if opts.image_input else nn.Linear
        self.encoder = input_encoder(opts.n_features, opts.sender_hidden)
        self.hidden_to_output = nn.Linear(opts.sender_hidden, opts.vocab_size)
        self.embedding = nn.Linear(opts.vocab_size, opts.embedding)
        self.sos_embedding = nn.Parameter(torch.zeros(opts.embedding))
        self.vocab_size = opts.vocab_size
        self.cell = rnn_cells[opts.sender_cell](
            input_size=opts.embedding,
            hidden_size=opts.sender_hidden,
        )

        self.reset_parameters()

    def gs_sample(self, logits):
        if self.training:
            sample = RelaxedOneHotCategorical(
                logits=logits,
                temperature=self.temperature,
            ).rsample()
        else:
            # argmax of GS sample is equivalent to sampling from the original
            # distribution: Softmax(logits, temperature)
            sample = OneHotCategorical(
                logits=(logits / self.temperature)
            ).sample()

        return sample, (logits.detach() / self.temperature).log_softmax(-1)

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, aux_input=None):
        prev_hidden = self.encoder(x).tanh()
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))

        sequence, logits = [], []
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            symbol_sample, symbol_logits = self.gs_sample(step_logits)

            prev_hidden = h_t
            e_t = self.embedding(symbol_sample)
            sequence.append(symbol_sample)
            logits.append(symbol_logits)

        return torch.stack(sequence, dim=1), torch.stack(logits, dim=1)


class RnnReceiverGS(nn.Module):
    def __init__(self, opts):
        super(RnnReceiverGS, self).__init__()

        if opts.receiver_cell not in rnn_cells:
            raise ValueError(f"Unknown RNN Cell: {opts.receiver_cell}")

        input_encoder = SeeingConvNet if opts.image_input else nn.Linear
        vocab_size = opts.vocab_size + 1 \
            if opts.channel == 'erasure' else opts.vocab_size

        self.message_encoder = nn.Linear(vocab_size, opts.embedding)
        self.input_encoder = input_encoder(
            opts.n_features, opts.receiver_hidden,
        )
        self.cell = rnn_cells[opts.receiver_cell](
            input_size=opts.embedding,
            hidden_size=opts.receiver_hidden,
        )
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

            energies = torch.matmul(embedded_input, h_t.unsqueeze(-1))
            outputs.append(self.logsoft(energies.squeeze()))
            prev_hidden = h_t

        return torch.stack(outputs, dim=1)


class SenderReceiverRnnGS(nn.Module):
    def __init__(self, opts: Namespace):
        super(SenderReceiverRnnGS, self).__init__()

        self.sender = RnnSenderGS(opts)
        self.receiver = RnnReceiverGS(opts)
        self.channel = channels[opts.channel](opts)
        self.image_input = opts.image_input

        self.length_cost = opts.length_cost
        self.warmup = iter(True for _ in range(opts.warmup_steps))
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        if opts.image_input:
            device = torch.device("cuda" if opts.cuda else "cpu")
            n_candidates = opts.n_distractors + 1
            n_attributes = torch.tensor(opts.n_attributes + [n_candidates]).to(device)
            self.loss = self.loss_obverter
            self.positions = torch.arange(
                n_candidates, dtype=torch.long, device=device
            )
            self.loss_weights = n_attributes.log().pow(-1).unsqueeze(-1)
            self.loss_weights[:] = 1 / len(n_attributes)

            # adjust loss weights so that the computed value equals
            # label_coeff * H(r_output, labels)
            # + features_coeff * Sum [H(target_attr, selected_attr)]
            self.loss_weights[:-1] *= opts.features_coeff
            self.loss_weights[-1] *= (
                opts.label_coeff * (len(self.loss_weights) - 1)
            )
        else:
            self.loss = self.loss_visa
            self.label_weight = opts.label_coeff / np.log(2)
            self.features_weight = opts.features_coeff / np.log(opts.n_distractors + 1)

        self.logging_strategy_train = LoggingStrategy(
            store_sender_input=(not opts.image_input),
            store_receiver_input=(not opts.image_input),
        )
        self.logging_strategy_eval = LoggingStrategy()
        self.min_real = torch.finfo(torch.get_default_dtype()).min

    def loss_visa(self, s_input, r_input, symbol, r_output, labels, aux_input):
        loss = torch.zeros_like(symbol[:, 0])
        if self.label_weight > 0:
            ce_labels = F.cross_entropy(r_output, labels, reduction='none')
            loss += self.label_weight * ce_labels
        if self.features_weight > 0:
            ce_features = F.binary_cross_entropy(
                torch.matmul(
                    r_output.softmax(-1).unsqueeze(1), r_input
                ).squeeze(1).clamp(0, 1),
                r_input[torch.arange(r_output.size(0)).to(labels.device), labels],
                reduction='none',
            ).sum(-1)
            loss += self.features_weight * ce_features
        accuracy = (r_output.detach().argmax(dim=-1) == labels).float() * 100
        return loss, {'accuracy': accuracy}

    def loss_obverter(
        self, s_input, r_input, symbol, r_output, labels, aux_input
    ):
        features = torch.stack(
            list(aux_input.values())
            + [self.positions.expand(len(s_input), -1)],
            dim=1,
        )
        idx = torch.arange(r_output.size(0)).to(labels.device)
        targets = features[idx, :, labels]
        idx = (
            idx.unsqueeze(-1).expand(targets.size()),
            torch.arange(targets.size(1)).to(idx.device).expand(targets.shape),
            targets,
        )
        one_hots = F.one_hot(features).to(r_output.dtype)
        selected = (
            r_output.unsqueeze(2).unsqueeze(3) + one_hots.transpose(1, 2).log()
        ).clamp(min=self.min_real).logsumexp(1)

        ce = torch.matmul(-selected[idx], self.loss_weights).squeeze()
        accuracy = (r_output.detach().argmax(dim=-1) == labels).float() * 100
        return ce, {'accuracy': accuracy}

    def forward(self, sender_input, labels, receiver_input, aux_input):
        warmup = next(self.warmup, False)

        sender_output = self.sender(sender_input, aux_input)
        message, logits, message_nn, logits_nn = self.channel(*sender_output)

        # append EOS to each message
        eos = torch.zeros_like(message[:, :1, :])
        eos[:, 0, 0] = 1
        log_eos = torch.log(eos)

        message = torch.cat([message, eos], dim=1)
        logits = torch.cat([logits, log_eos], dim=1)
        message_nn = torch.cat([message_nn, eos], dim=1)  # no noise
        logits_nn = torch.cat([logits_nn, log_eos], dim=1)

        if isinstance(self.channel, NoChannel):
            receiver_output = self.receiver(message, receiver_input, aux_input)
            receiver_output_nn = receiver_output.detach()
        else:  # compute receiver outputs also for messages without noise
            aux_input_joined = {
                key: torch.cat([vals, vals]) for key, vals in aux_input.items()
            }

            receiver_output_joined = self.receiver(
                torch.cat([message, message_nn.detach()]),
                torch.cat([receiver_input, receiver_input]),
                aux_input_joined,
            )
            section = len(message)
            receiver_output = receiver_output_joined[:section]
            receiver_output_nn = receiver_output_joined[section:].detach()

        loss, z, length, length_nn = 0, 0, 0, 0
        not_eosed_before = torch.ones(message.size(0)).to(message.device)
        not_eosed_before_nn = not_eosed_before.clone()
        aux_info = defaultdict(float)

        # compute aux info values without noise
        accuracy_nn = (
            receiver_output_nn.argmax(-1) == labels.unsqueeze(-1)
        ).float() * 100

        for step in range(receiver_output.size(1)):
            symbol, symbol_nn = message[:, step], message_nn[:, step]
            step_loss, step_aux = self.loss(
                sender_input,
                receiver_input,
                symbol,
                receiver_output[:, step],
                labels,
                aux_input,
            )

            add_mask = symbol[:, 0] * not_eosed_before
            add_mask_nn = symbol_nn[:, 0].detach() * not_eosed_before_nn.detach()
            z += add_mask
            loss += step_loss * add_mask
            if not warmup:
                loss += self.length_cost * (step + 1) * add_mask

            # aggregate aux info
            length += add_mask.detach() * (step + 1)
            length_nn += add_mask_nn * (step + 1)

            for name, value in step_aux.items():
                aux_info[name] += value * add_mask.detach()
            aux_info['accuracy_nn'] += accuracy_nn[:, step] * add_mask_nn

            not_eosed_before = not_eosed_before * (1 - symbol[:, 0])
            not_eosed_before_nn = not_eosed_before_nn * (1 - symbol_nn[:, 0])

        # the remainder of the probability mass
        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"

        loss += step_loss * not_eosed_before
        if not warmup:
            loss += self.length_cost * (step + 1) * not_eosed_before

        aux_info['accuracy_nn'] += accuracy_nn[:, step] * not_eosed_before_nn
        for name, value in step_aux.items():
            aux_info[name] += value * not_eosed_before.detach()

        length += (step + 1) * not_eosed_before.detach()
        length_nn += (step + 1) * not_eosed_before_nn.detach()
        aux_info['temperature'] = torch.tensor([self.sender.temperature])
        aux_info['length'] = length
        aux_info['length_nn'] = length_nn

        logging_strategy = (
            self.logging_strategy_train if self.training
            else self.logging_strategy_eval
        )

        interaction = logging_strategy.filtered_interaction(
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
