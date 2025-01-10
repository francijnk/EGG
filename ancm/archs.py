import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional
from collections import defaultdict

from egg.core.reinforce_wrappers import RnnEncoder
from egg.core.baselines import Baseline, MeanBaseline
from egg.core.interaction import LoggingStrategy
from egg.core.util import find_lengths


# GS classes
class SenderGS(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(SenderGS, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.fc1(x).tanh()


class ReceiverGS(nn.Module):
    def __init__(self, n_features, linear_units):
        super(ReceiverGS, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)

    def forward(self, x, _input, _aux_input=None):
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()


class SenderReceiverRnnGS(nn.Module):
    """
    This class implements the Sender/Receiver game mechanics for the Sender/Receiver game with variable-length
    communication messages and Gumber-Softmax relaxation of the channel. The vocabulary term with id `0` is assumed
    to the end-of-sequence symbol. It is assumed that communication is stopped either after all the message is processed
    or when the end-of-sequence symbol is met.

    >>> class Sender(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(10, 5)
    ...     def forward(self, x, _input=None, aux_input=None):
    ...         return self.fc(x)
    >>> sender = Sender()
    >>> sender = RnnSenderGS(sender, vocab_size=2, embed_dim=3, hidden_size=5, max_len=3, temperature=5.0, cell='gru')
    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(7, 10)
    ...     def forward(self, x, _input=None, aux_input=None):
    ...         return self.fc(x)
    >>> receiver = RnnReceiverGS(Receiver(), vocab_size=2, embed_dim=4, hidden_size=7, cell='rnn')
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, labels, aux_input):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {'aux': torch.zeros(sender_input.size(0))}
    >>> game = SenderReceiverRnnGS(sender, receiver, loss)
    >>> loss, interaction = game(torch.ones((3, 10)), None, None)  # batch of 3 10d vectors
    >>> interaction.aux['aux'].detach()
    tensor([0., 0., 0.])
    >>> loss.item() > 0
    True
    """

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
            'truncation': TruncationChannel,
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

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None, apply_noise=True):
        message = self.sender(sender_input, aux_input)
        if self.channel:
            message = self.channel(message, message_length=None, apply_noise=apply_noise)
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
            eos_mask = message[:, step, 0]  # always eos == 0

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

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=expected_length.detach(),
            aux=aux_info,
        )

        return loss.mean(), interaction


def loss_gs(_sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
    acc = (receiver_output.argmax(dim=-1) == _labels).detach().float()
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {"accuracy": acc * 100}


# REINFORCE
class SenderReinforce(nn.Module):
    def __init__(self, n_features, n_hidden):
        super(SenderReinforce, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        return self.fc1(x).tanh()


class ReceiverReinforce(nn.Module):
    def __init__(self, n_features, linear_units):
        super(ReceiverReinforce, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x, _input, _aux_input=None):
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        energies = energies.squeeze()
        return self.logsoft(energies)


class RnnReceiverReinforce(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game. The wrapper logic feeds the message into the cell
    and calls the wrapped agent on the hidden state vector for the step that either corresponds to the EOS input to the
    input that reaches the maximal length of the sequence.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """

    def __init__(
        self, agent, vocab_size, embed_dim, hidden_size, cell="rnn", num_layers=1
    ):
        super(RnnReceiverReinforce, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size+1, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, aux_input=None, lengths=None):
        encoded = self.encoder(message, lengths)
        sample, logits, entropy = self.agent(encoded, input, aux_input)

        return sample, logits, entropy


def loss_reinforce(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input):
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
        baseline_type: Baseline = MeanBaseline,
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
        baseline_type: Baseline = MeanBaseline,
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
        loss,
        sender_input,
        labels,
        receiver_input=None,
        aux_input=None,
        apply_noise=True,
    ):
        message, log_prob_s, entropy_s = sender(sender_input, aux_input)
        message_length = find_lengths(message)
        if self.channel:
            message = self.channel(message, message_length, apply_noise)

        receiver_output, log_prob_r, entropy_r = receiver(
            message, receiver_input, aux_input, message_length
        )

        loss, aux_info = loss(
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
        effective_entropy_s = effective_entropy_s / message_length.float()

        weighted_entropy = (
            effective_entropy_s.mean() * self.sender_entropy_coeff
            + entropy_r.mean() * self.receiver_entropy_coeff
        )

        log_prob = effective_log_prob_s + log_prob_r

        length_loss = message_length.float() * self.length_cost

        policy_length_loss = (
            (length_loss - self.baselines["length"].predict(length_loss))
            * effective_log_prob_s
        ).mean()
        policy_loss = (
            (loss.detach() - self.baselines["loss"].predict(loss.detach())) * log_prob
        ).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

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
        # GS: append 0 vector representing Pr[erased_symbol] for all messages
        if message.dim() == 3 and self.p != 0:
            message = torch.cat([message, torch.zeros((message.size(0), message.size(1), 1)).to(self.device)], dim=2)

        if self.p != 0. and apply_noise:
            msg = message if message.dim() == 2 else message.argmax(dim=-1)


            # sample symbol indices to be erased
            target_ids = (torch.rand(*msg.size(), generator=self.generator) < self.p).to(self.device)

            if message.dim() == 2: # REINFORCE
                # if message length is not provided, compute it
                if message_length is None:
                    message_length = find_lengths(message)

                # True for all message symbols before the 1st EOS symbol
                not_eosed = (
                    torch.unsqueeze(torch.arange(0, message.size(1)), dim=0).expand(message.size(0), message.size(1)).to(self.device)
                    < torch.unsqueeze(message_length-1, dim=-1).expand(message.size(0), message.size(1))
                )

                # erase
                message = torch.where(
                    torch.logical_and(target_ids, not_eosed),
                    torch.tensor(self.vocab_size),  # i.e. erased symbol
                    message)

            else:  # GS
                # make sure EOS is not erased
                not_eos = (msg != 0)
                combined = torch.logical_and(target_ids, not_eos)
                combined = combined[:,:,None]
                combined = combined.expand(*message.size())

                # for erased symbols – where should we put 0/1?
                erased_probs = torch.tensor([0]*(message.size(2)-1) + [1], device=self.device)
                erased_probs = erased_probs.expand(1, message.size(1), message.size(2))
                erased_probs = erased_probs.expand(*message.size())

                # erase
                message = torch.where(combined, erased_probs, message)

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
            
            # True for all message symbols before the 1st EOS symbol
            #not_eosed = (
            #    torch.stack(
            #        [torch.arange(0, msg.size(1)).to(self.device)])
            #    < torch.cat(
            #        [torch.unsqueeze(message_length-1, dim=-1).to(self.device) for _ in range(msg.size(1))],
            #        dim=1))

            not_eosed = (
                torch.unsqueeze(torch.arange(0, message.size(1)), dim=0).expand(message.size(0), message.size(1)).to(self.device)
                < torch.unsqueeze(message_length-1, dim=-1).expand(message.size(0), message.size(1))
            )

            # sample symbol indices to be erased
            target_ids = (torch.rand(*msg.size(), generator=self.generator) < self.p).to(self.device)
            delete_ids = torch.logical_and(target_ids, not_eosed)
            keep_ids = torch.logical_not(delete_ids)
            num_deleted = torch.sum(delete_ids.int(), dim=1)

            if message.dim() == 2:  # REINFORCE 
                message = torch.stack([
                    torch.cat(
                        [message[i][keep_ids[i]], torch.zeros(num_deleted[i], dtype=torch.int)])
                    for i in range(message.size(0))
                ])

            else:  # GS
                keep_ids = torch.logical_not(delete_ids)
                eos_probs = torch.tensor([1] + [0] * (message.size(2)-1), device=self.device)
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
            msg_exp = msg_exp.expand(self.vocab_size-1, *msg_exp.size()).permute(1, 2, 0)
            keep_ids = (candidate_symbols != msg_exp)  # torch.where(candidate_symbols != msg_exp, True, False)
            candidate_symbols = candidate_symbols[keep_ids].reshape(*msg.size(), self.vocab_size-2)

            # sample the replacement symbol to be used
            replacement_indices = torch.randint(
                high=self.vocab_size-2,
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
 
            if message.dim() == 2:  # REINFORCE
                # compute message length if it is not provided
                if message_length is None:
                    message_length = find_lengths(message)
                
                # True for all message symbols before the 1st EOS symbol
                #not_eosed = (
                #    torch.stack(
                #        [torch.arange(0, message.size(1)).to(self.device)])  # FIX! 
                #    < torch.cat(
                #        [torch.unsqueeze(message_length-1, dim=-1).to(self.device) for _ in range(message.size(1))],
                #        dim=1))
                not_eosed = (
                    torch.unsqueeze(torch.arange(0, message.size(1)), dim=0).expand(message.size(0), message.size(1)).to(self.device)
                    < torch.unsqueeze(message_length-1, dim=-1).expand(message.size(0), message.size(1))
                )
                #not_eosed = (
                #    torch.arange(0, message.size(1)).expand(message.size(0), -1).to(self.device)
                #    < (message_length-1).expand(message.size(1), -1).permute(1, 0))

                message = torch.where(
                    torch.logical_and(target_ids, not_eosed),
                    replacement_symbols,
                    message)

            else:  # GS
                # make sure EOS is not erased
                not_eos = (msg != 0)

                combined = torch.logical_and(target_ids, not_eos)
                combined = combined[:,:,None]
                combined = combined.expand(*message.size())

                def replacement_probs(ind):
                    row = [0] * self.vocab_size
                    row[ind] = 1
                    return torch.tensor(row, dtype=torch.int)

                replaced_probs = torch.stack([
                    torch.stack([
                        replacement_probs(replacement_symbols[i, j])
                        for j in range(message.size(1))
                    ])
                    for i in range(message.size(0))
                ])

                # replace
                message = torch.where(combined, replaced_probs, message)

        return message


class TruncationChannel(Channel):

    # def __init__(self, error_prob, vocab_size, device, is_relative_detach=True, seed=42):
    #     super().__init__(error_prob, vocab_size, device, is_relative_detach, seed)
    #     self.p = error_prob

    def forward(self, message, message_length=None, apply_noise=False):
        pass
        return message

        if self.p != 0. and apply_noise: 
            msg = message if message.dim() == 2 else message.argmax(dim=-1)
            if message_length is None:
                message_length = find_lengths(msg)
            message_length = message_length.detach()

            seq_error_probs = 2 * self.p / message_length / (message_length-1)
            print("lengths", message_length)
            print("seq error", seq_error_probs)

            target_ids = (torch.rand(msg.size(0), generator=self.generator) < seq_error_probs).to(self.device)
            target_ids = torch.where(seq_error_probs <= 1, target_ids, False)
            target_ids = target_ids.expand(msg.size(1), msg.size(0)).permute(1, 0)
            print("target ids", target_ids)

            print([message_length[i].item() for i in range(message_length.size(0))])
            num_truncated = torch.cat([
                torch.randint(low=1, high=message_length[i].item(),
                              size=(1,), generator=self.generator)
                if message_length[i].item() != 1
                else torch.ones((1,), dtype=torch.int)
                for i in range(message_length.size(0))
            ])
            print(num_truncated)
            # truncated = torch.stack([
            #    message[]
            #])
        return message


