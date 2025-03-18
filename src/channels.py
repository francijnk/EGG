import torch
from argparse import Namespace
import numpy as np
import torch.nn as nn
from scipy.stats import binom
from abc import ABCMeta, abstractmethod
from torch.distributions.utils import logits_to_probs, probs_to_logits


class Channel(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        opts: Namespace,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.p = opts.error_prob
        self.vocab_size = opts.vocab_size
        self.max_len = opts.max_len
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(opts.random_seed)
        self.device = device
        self.min_real = torch.finfo(torch.get_default_dtype()).min

        # compute max. entropy for every message length (incl. additional EOS)
        # each length is mapped to a tuple of values with and without noise
        self._max_message_entropy = {
            i: (
                i * self.max_non_eos_entropy(noise=True),
                i * self.max_non_eos_entropy(noise=False),
            ) for i in range(self.max_len + 1)
        }

    @abstractmethod
    def process(
        messages: torch.Tensor,
        probs: torch.Tensor,
        apply_noise: bool,
    ):
        return

    def sample_targets(self, *size):
        return torch.rand(
            *size,
            generator=self.generator,
            device=self.device,
        ) < self.p

    def max_non_eos_entropy(self, noise: bool):
        """
        Returns maximum achievable entropy of a single symbol passing through
        the channel, assuming it is not EOS.
        """
        return np.log2(self.vocab_size - 1)

    def max_message_entropy(self, length_probs: torch.Tensor, noise: bool):
        """
        Given a tensor L of length probabilities of a messages M, returns
        maximum achievable message entropy, computed according to the formula
        H_max(M) = H(L) + H_max(M | L).
        """

        # clamping log probabilities to (0, 1) prevents nan entropy values for
        # negative prob. values that may occasionally happen, for symbols that
        # are almost certainly eosed
        length_log2_prob = torch.log2(length_probs.clamp(0, 1)).clamp(min=self.min_real)

        max_entropy = (-length_probs.clamp(0, 1) * length_log2_prob).sum()  # H(L)
        for i in range(len(length_probs)):
            idx = 1 - int(noise)  # 0/1 for max. entropy with/without noise
            max_entropy_i = self._max_message_entropy[i][idx]
            max_entropy += length_probs[i].clamp(0, 1) * max_entropy_i
            # P(L = i) * H_max(M | L = i)

        return max_entropy

    def forward(self, messages: torch.Tensor, logits: torch.Tensor, *args):
        return (
            *self.process(messages, logits, apply_noise=True),
            *self.process(messages, logits, apply_noise=False),
        )


class NoChannel(Channel):
    def process(self, messages, logits, apply_noise):
        return messages, logits


class ErasureChannel(Channel):
    """
    Erases a symbol from a message with a given probability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.placeholder_probs = torch.tensor([])
        self.placeholder_logits = torch.tensor([])

    @staticmethod
    def binary_entropy(p: float):
        if p == 0. or p == 1.:
            return 0.
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def max_non_eos_entropy(self, noise: bool):
        if noise:
            return (
                self.binary_entropy(self.p)
                # + self.p * 0
                + (1 - self.p) * np.log2(self.vocab_size - 1)
            )
        else:
            return np.log2(self.vocab_size - 1)

    def process(self, messages, logits, apply_noise):
        if len(messages) != len(self.placeholder_probs):
            self.placeholder_probs = torch.zeros_like(messages[..., :1])
            self.placeholder_logits = torch.ones_like(messages[..., :1])
            self.placeholder_logits[:] = self.min_real

        if not apply_noise:
            messages = torch.cat([messages, self.placeholder_probs], dim=-1)
            logits = torch.cat([logits, self.placeholder_logits], dim=-1)
            return messages, logits

        elif self.training:
            # append a column for erased symbols (creates new tensors)
            messages = torch.cat([messages, self.placeholder_probs], dim=-1)
            logits = torch.cat([logits, self.placeholder_logits], dim=-1)

            # sample targets
            target_mask = self.sample_targets(messages.size()[:-1])

            # create a replacement probability array and replace
            erased_messages = torch.zeros_like(messages)
            erased_messages[:, :, 0] = messages[:, :, 0]
            erased_messages[:, :, -1] = 1 - messages[:, :, 0]

            target_messages = torch.zeros_like(messages).to(torch.bool)
            target_messages[target_mask] = 1
            messages = torch.where(target_messages, erased_messages, messages)

            # adjust symbol log-probs
            logits[..., 1:-1] += np.log(1 - self.p)
            logits[..., -1] = np.log(self.p) + torch.log(1 - logits[..., 0].exp())
            return messages, logits

        else:
            # append a column for erased symbols
            messages = torch.cat([messages, self.placeholder_probs], dim=-1)
            logits = torch.cat([logits, self.placeholder_logits], dim=-1)

            # apply argmax, exclude EOS, sample batch rows to be replaced
            discrete_symbols = messages.argmax(-1)
            non_eos_mask = discrete_symbols != 0
            target_mask = self.sample_targets(non_eos_mask.sum())
            # n_targets = target_mask.sum()

            # prepare the index and source of replacement
            target_messages = torch.zeros_like(messages).bool()
            target_messages[non_eos_mask] = torch.where(
                target_mask.unsqueeze(-1),
                torch.ones(target_mask.size(0), messages.size(-1)).to(self.device).bool(),
                False)
            erased_messages = torch.zeros_like(messages)
            erased_messages[:, :, -1] = 1

            # replace
            messages = torch.where(target_messages, erased_messages, messages)

            # adjust symbol log-probs
            logits = logits.clone()
            logits[..., 1:-1] += np.log(1 - self.p)
            logits[..., -1] = np.log(self.p) + torch.log(1 - logits[..., 0].exp())
            return messages, logits


class DeletionChannel(Channel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        binary = torch.tensor(
            [[0, 1]],
            dtype=torch.bool,
            device=self.device,
            requires_grad=False)
        self.all_combos = torch.cartesian_prod(*binary.expand(self.max_len, -1))

        n_deleted = self.all_combos.sum(-1)
        # self.combo_probs = self.p ** n_deleted * (1 - self.p) ** (self.max_len - n_deleted)
        self.combo_logits = (
            np.log(self.p) * n_deleted
            + np.log(1 - self.p) * (self.max_len - n_deleted)
        )

        self.n_deleted_logits = torch.zeros(
            (self.max_len + 1, self.max_len + 1),
            requires_grad=False,
            device=self.device)
        self.n_deleted_logits[:] = torch.finfo(self.n_deleted_logits.dtype).min
        for n in range(self.max_len + 1):
            distr = binom(n, self.p)
            k = np.arange(n + 1)
            rev_k = k[::-1].copy()
            self.n_deleted_logits[n, rev_k] = \
                torch.tensor(distr.logpmf(k)).to(self.n_deleted_logits)

        not_deleted_mask = (
            torch.arange(self.max_len - 1, -1, step=-1, device=self.device)
            .expand(len(self.all_combos), -1)
            .t()
        ) >= n_deleted.unsqueeze(0)
        symbol_combo_logits = torch.where(
            not_deleted_mask,
            self.combo_logits.unsqueeze(0).expand(not_deleted_mask.size()),
            torch.finfo(self.combo_logits.dtype).min,
        ).log_softmax(-1)
        assert torch.allclose(
            symbol_combo_logits.logsumexp(-1).exp(),
            torch.ones_like(symbol_combo_logits[:, 0]),
        )
        self.symbol_combo_logits = symbol_combo_logits.unsqueeze(0).unsqueeze(-2)

    def shift(self, tensor, target_mask):
        """
        Shifts symbols to be deleted to the end of the message.
        """
        for i in range(1, target_mask.size(1)):
            idx = -i - 1
            targets = target_mask[:, idx]
            tensor[targets, idx:] = torch.roll(
                tensor[targets, idx:],
                shifts=-1,
                dims=1
            )

    def shift_combos(self, symbol_combos):
        """
        Shifts probabilities to be deleted to the end of the message, for each
        combination of symbols to be deleted.
        """
        for i in range(1, self.max_len):
            idx = -i - 1
            targets = self.all_combos[:, idx]
            symbol_combos[targets, :, idx:] = torch.roll(
                symbol_combos[targets, :, idx:],
                shifts=-1,
                dims=2,
            )

    def adjust_logits(self, logits):
        non_eos_logits = logits.clone()[..., 1:].log_softmax(-1)
        symbol_combos = non_eos_logits.expand(
            len(self.all_combos),
            *logits.size()[:2],
            logits.size(2) - 1,
        ).clone()
        self.shift_combos(symbol_combos)

        expected_logits = (
            symbol_combos.permute(1, 2, 3, 0)  # (bs, len, symbol probs, combos)
            + self.symbol_combo_logits  # (1, len, 1, combos)
        ).clamp(min=self.min_real).logsumexp(-1)  # (bs, len, symbol probs, combos)

        length_logits = torch.cat([
            logits[:, :1, 0],
            logits[:, 1:, 0] + torch.cumsum(
                torch.log(1 - logits[:, :-1, 0].exp()),
                dim=-1,
            ).clamp(self.min_real, 0),
            torch.sum(
                torch.log(1 - logits[..., 0].exp()), dim=-1
            ).clamp(self.min_real, 0).unsqueeze(-1),
        ], dim=-1)
        adjusted_length_logits = (
            length_logits.unsqueeze(-1) + self.n_deleted_logits.unsqueeze(0)
        ).logsumexp(-2)

        adjusted_logits = torch.ones_like(logits)
        adjusted_logits[:] = torch.finfo(logits.dtype).min
        adjusted_logits[..., 0] = adjusted_length_logits[:, :-1]
        for i in range(1, self.max_len):
            adjusted_logits[:, i, 0] = (
                adjusted_logits[:, i, 0]
                - torch.log(1 - adjusted_logits[:, :i, 0].exp()).sum(-1)
            ).clamp(self.min_real, 0)
        adjusted_logits[..., 1:] = (
            expected_logits.log_softmax(-1)
            + torch.log(1 - adjusted_logits[..., :1].exp())
        )

        return adjusted_logits

    def adjust_probs(self, probs):
        # compute expected probabilities for non-EOS symbols at each position
        non_eos_probs = probs.clone()
        non_eos_probs[..., 0] = 0
        non_eos_probs /= non_eos_probs.sum(-1, keepdim=True)
        symbol_combos = non_eos_probs.expand(
            len(self.all_combos), *probs.size()
        ).clone()
        self.shift_combos(symbol_combos)
        expected_probs = torch.matmul(
            symbol_combos.permute(1, 2, 3, 0),
            self.symbol_combo_probs,
        ).squeeze()

        # compute adjusted EOS probabilities at each position
        length_probs = torch.cat([
            probs[:, :1, 0],
            probs[:, 1:, 0] * torch.cumprod(1 - probs[:, :-1, 0], dim=-1),
            torch.prod(1 - probs[:, :, 0], dim=-1).unsqueeze(-1),
        ], dim=-1)
        adjusted_length_probs = torch.matmul(length_probs, self.n_deleted_probs)

        adjusted_probs = torch.zeros_like(probs)
        adjusted_probs[..., 0] = adjusted_length_probs[:, :-1]
        for i in range(1, self.max_len):
            adjusted_probs[:, i, 0] /= \
                torch.prod(1 - adjusted_probs[:, :i, 0], dim=-1)

        adjusted_probs[..., 1:] = (
            expected_probs[..., 1:]
            * (1 - adjusted_probs[..., :1])
            / expected_probs[..., 1:].sum(-1, keepdim=True)
        )

        return adjusted_probs

    def process(self, messages, logits, apply_noise):
        assert messages.size(1) == self.max_len
        if messages.is_cuda:
            self.shift = torch.compile(self.shift)
            self.shift_combos = torch.compile(self.shift_combos)

        if not apply_noise:
            return messages, logits.clone()

        elif self.training:
            # sample target symbols
            size = messages.size()
            messages = messages.clone()
            target_mask = self.sample_targets(size[:-1])
            # n_targets = target_mask.sum()

            # shift target symbols to the end of the message
            self.shift(messages, target_mask)

            # get new target positions
            n_deleted = target_mask.sum(1, keepdim=True)
            target_mask = (
                torch.arange(size[1] - 1, -1, step=-1)
                .to(self.device)
                .unsqueeze(0)
                .expand(size[:-1])
            ) < n_deleted

            # delete
            replacement_probs = torch.zeros_like(messages[target_mask])
            replacement_probs[..., 0] = 1
            messages[target_mask] = replacement_probs

            return messages, self.adjust_logits(logits)
        else:
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[2])
            discrete_symbols = messages.argmax(-1)
            non_eos_ids = torch.arange(len(messages)).to(self.device)[discrete_symbols != 0]
            target_mask = self.sample_targets(non_eos_ids.numel())
            # n_targets = target_mask.sum()

            # get target positions & symbols
            target_rows = non_eos_ids[target_mask]
            target_mask = torch.zeros_like(messages[:, 0]).bool()
            target_mask[target_rows] = True
            messages = messages.view(size)
            target_mask = target_mask.view(size[:-1])

            # shift target symbols to the end of the message
            self.shift(messages, target_mask)

            # get new target positions
            n_deleted = target_mask.sum(1, keepdim=True)
            target_mask = (
                torch.arange(size[1] - 1, -1, step=-1)
                .to(self.device)
                .unsqueeze(0)
                .expand(size[:-1])
            ) < n_deleted

            # delete
            replacement_probs = torch.zeros_like(messages[target_mask])
            replacement_probs[..., 0] = 1
            messages[target_mask] = replacement_probs

            return messages, self.adjust_logits(logits)


class SymmetricChannel(Channel):
    """
    Replaces each symbol with a different symbol with a given probability.
    The replacement symbol is randomly sampled from a uniform distribution.
    """

    def process(self, messages, logits, apply_noise):
        if not apply_noise:
            return messages, logits.clone()

        elif self.training:
            # reshape & sample targets
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[-1])
            target_mask = self.sample_targets(messages[:, 1:].size())
            n_targets = target_mask.sum()

            # get target positions & highest symbol probs
            rows = torch.arange(len(target_mask)).to(self.device)
            cols = torch.arange(size[-1] - 1).to(self.device)
            target_rows = rows.expand(size[-1] - 1, -1).t()[target_mask]
            target_cols = cols.expand(target_mask.size())[target_mask]

            # compute probability adjustment for each symbol with a target
            targets = messages.clone()[target_rows, target_cols]
            adjustment = targets.expand(size[-1] - 1, -1).t() / (size[-1] - 2)
            idx = (torch.arange(len(adjustment)).to(self.device), target_cols)
            adjustment[torch.arange(len(adjustment)), target_cols] = -targets

            # adjust relaxed symbols
            messages[target_rows, 1:] += adjustment
            messages = messages.view(size)

            # adjust symbol log-probs
            logits = logits.clone()
            logp_replacement = torch.log(
                (1 - logits[..., 1:].exp() - logits[..., :1].exp()).clamp(0, 1)
            ) + np.log(self.p / (size[-1] - 2))
            # logp_replacement += np.log(self.p / (size[-1] - 2))
            logits[..., 1:] += np.log(1 - self.p)
            logits[..., 1:] = torch.logaddexp(logits[..., 1:], logp_replacement)
            return messages, logits
            assert not torch.any(logits.isnan())
            assert torch.allclose(logits.exp().sum(-1), torch.ones_like(logits[..., 0]))

        else:
            # reshape, apply argmax, exclude EOS, sample symbols to be replaced
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[-1])
            discrete_symbols = messages.argmax(-1)
            non_eos_ids = torch.arange(len(messages)).to(self.device)[discrete_symbols != 0]
            target_mask = self.sample_targets(non_eos_ids.numel())
            n_targets = target_mask.sum()

            # get target positions & symbols
            target_rows = non_eos_ids[target_mask]
            target_symbols = discrete_symbols[target_rows]

            # find candidate symbols different from target symbols
            n_targets = target_symbols.numel()
            symbols = torch.arange(1, messages.size(-1)).to(self.device).expand(n_targets, -1)
            candidate_mask = symbols != target_symbols.unsqueeze(-1)
            candidate_symbols = symbols[candidate_mask].view(-1, size[-1] - 2)

            # sample replacement symbols
            replacement_ids = torch.randint(
                size=(n_targets,),
                high=messages.size(-1) - 2,
                generator=self.generator,
                device=self.device,
            )
            idx = (torch.arange(n_targets).to(self.device), replacement_ids)
            replacement_symbols = candidate_symbols[idx]

            # replace probabilities
            messages[target_rows] = 0
            messages[target_rows, replacement_symbols] = 1
            messages = messages.view(size)

            # adjust symbol log-probs
            logits = logits.clone()
            logp_replacement = torch.log(
                (1 - logits[..., 1:].exp() - logits[..., :1].exp()).clamp(0, 1)
            ) + np.log(self.p / (size[-1] - 2))
            # logp_replacement += np.log(self.p / (size[-1] - 2))
            logits[..., 1:] += np.log(1 - self.p)
            logits[..., 1:] = torch.logaddexp(logits[..., 1:], logp_replacement)
            assert torch.allclose(logits.exp().sum(-1), torch.ones_like(logits[..., 0]))
            assert not torch.any(logits.isnan())
            return messages, logits
