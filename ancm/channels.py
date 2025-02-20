import torch
import numpy as np
import torch.nn as nn
from egg.core.util import find_lengths
from scipy.optimize import minimize_scalar
from abc import ABCMeta, abstractmethod
from typing import Optional

from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs


# TODO instead of entropies, adjust messages


class Channel(nn.Module, metaclass=ABCMeta):
    # __metaclass__ = ABCMeta

    def __init__(self, error_prob, max_len, vocab_size, device, seed=42):
        super().__init__()
        self.p = torch.tensor(error_prob, requires_grad=False)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.device = device

        # compute max. entropy for every message length (incl. additional EOS)
        # each length is mapped to a tuple of values with and without noise
        self._max_message_entropy = {
            i: (
                i * self.max_non_eos_entropy(noise=True),
                i * self.max_non_eos_entropy(noise=False),
            ) for i in range(self.max_len + 1)
        }

    @abstractmethod
    def gs(
            messages: torch.Tensor,
            probs: torch.Tensor,
            apply_noise: bool):
        return

    @abstractmethod
    def reinforce(
            messages: torch.Tensor,
            probs: torch.Tensor,
            apply_noise: bool, **kwargs):
        return

    # @staticmethod
    # def tensor_binary_entropy(p: torch.Tensor):
    #     q = 1 - p
    #     min_real = torch.finfo(p.dtype).min
    #     log2_p = torch.clamp(torch.log2(p), min=min_real)
    #     log2_q = torch.clamp(torch.log2(q), min=min_real)
    #     return -p * log2_p - q * log2_q

    @staticmethod
    def binary_entropy(p: float):  # TODO move to ErasureChannel?
        if p == 0. or p == 1.:
            return 0.
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

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
        H_max(M) = H(L) + H_max(M | L)
        """
        min_real = torch.finfo(length_probs.dtype).min
        length_log2_prob = torch.clamp(torch.log2(length_probs), min=min_real)

        max_entropy = (-length_probs * length_log2_prob).sum()  # H(L)
        for i in range(len(length_probs)):
            idx = 1 - int(noise)  # 0/1 for max. entropy with/without noise
            max_entropy_i = self._max_message_entropy[i][idx]
            max_entropy += length_probs[i] * max_entropy_i
            # P(L = i) * H_max(M | L = i)

        return max_entropy

    def forward(self, messages, probs, **kwargs):
        # GS
        if messages.dim() == 3:
            _messages, _probs = self.gs(messages, probs, True)
            _messages_nn, _probs_nn = self.gs(messages, probs, False)

            output_dict = {
                'accumulated_eos_prob': torch.zeros_like(messages[:,  :, 0])
            }  # TODO remove if we dont bring deletion back

            return _messages, _messages_nn, _probs, _probs_nn, output_dict

        # Reinforce
        else:
            raise NotImplementedError


class NoChannel(Channel):
    def gs(self, messages, probs, apply_noise):
        return messages, probs

    def reinforce(self, messages, probs, apply_noise, **kwargs):
        return messages, probs


class ErasureChannel(Channel):
    """
    Erases a symbol from a message with a given probability
    """

    def max_non_eos_entropy(self, noise: bool):
        if noise:
            return (
                self.binary_entropy(self.p.item())
                # + self.p * 0
                + (1 - self.p.item()) * np.log2(self.vocab_size - 1)
            )
        else:
            return np.log2(self.vocab_size - 1)

    def gs(self, messages, probs, apply_noise):
        if not apply_noise:
            placeholder_probs = torch.zeros_like(messages[:, :, :1])
            messages = torch.cat([messages, placeholder_probs], dim=-1)
            probs = torch.cat([probs, placeholder_probs], dim=-1)
            return messages, probs

        elif self.training:
            target_mask = torch.rand(
                messages.size()[:-1],
                generator=self.generator,
                device=self.device,
            ) < self.p

            # append a column for erased symbols
            placeholder_probs = torch.zeros_like(messages[:, :, :1])
            messages = torch.cat([messages, placeholder_probs], dim=-1)
            probs = torch.cat([probs, placeholder_probs], dim=-1)

            if target_mask.sum() == 0:
                return messages, probs

            # create a replacement probability array and replace
            erased_messages = torch.zeros_like(messages)
            erased_messages[:, :, 0] = messages[:, :, 0]
            erased_messages[:, :, -1] = 1 - messages[:, :, 0]

            target_messages = torch.zeros_like(messages).to(torch.bool)
            target_messages[target_mask] = 1
            messages = torch.where(target_messages, erased_messages, messages)

            # adjust symbol probs
            probs[:, :, 1:-1] *= (1 - self.p)
            probs[:, :, -1] = self.p * (1 - probs[:, :, 0])
            assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)))

            return messages, probs

        else:
            # append a column for erased symbols
            placeholder_probs = torch.zeros_like(messages[:, :, :1])
            messages = torch.cat([messages, placeholder_probs], dim=-1)
            probs = torch.cat([probs, placeholder_probs], dim=-1)

            # apply argmax, exclude EOS, sample batch rows to be replaced
            discrete_symbols = messages.argmax(-1)
            non_eos_mask = discrete_symbols != 0
            # non_eos_symbols = discrete_symbols[non_eos_mask]
            target_mask = torch.rand(
                non_eos_mask.sum(),
                generator=self.generator,
                device=self.device
            ) < self.p
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages, probs

            # prepare the index and source of replacement
            target_messages = torch.zeros_like(messages).bool()
            target_messages[non_eos_mask] = torch.where(
                target_mask.unsqueeze(-1),
                torch.ones(target_mask.size(0), messages.size(-1)).bool(),
                False)
            erased_messages = torch.zeros_like(messages)
            erased_messages[:, :, -1] = 1

            # replace
            messages = torch.where(target_messages, erased_messages, messages)

            # adjust symbol probs
            probs[:, :, 1:-1] *= (1 - self.p)
            probs[:, :, -1] = self.p * (1 - probs[:, :, 0])
            assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)))

            return messages, probs

    def reinforce(self, messages, entropies, apply_noise, **kwargs):
        if not apply_noise:
            return messages, entropies

        # sample symbol indices to be erased, make sure EOS is not erased
        size = messages.size()
        non_eos_mask = messages != 0
        target_mask = torch.rand(
            non_eos_mask.sum(),
            generator=self.generator,
            device=self.device
        ) < self.p
        n_targets = target_mask.sum()

        if n_targets == 0:
            return messages, entropies

        non_eos_symbols = torch.arange(1, size[-1]).expand(size[0], -1)
        target_symbols = non_eos_symbols[target_mask]
        target_rows = torch.arange(size[0]).unsqueeze(1)
        target_rows = target_rows.expand(-1, size[-1])[target_mask]

        messages[target_rows, target_symbols] = self.vocab_size
        return messages  # TODO check


class DeletionChannel(Channel):
    def gs(self, messages, entropies, apply_noise):
        if not apply_noise:
            return messages, entropies

        elif self.training:
            raise NotImplementedError
            # reshape & sample targets
            size = messages.size()
            messages = messages.clone()
            target_mask = torch.rand(
                size[:-1],
                generator=self.generator,
                device=self.device,
            ) < self.p
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages, entropies

            # get target positions
            positions = torch.arange(1, size[1]).expand(len(messages), -1)
            target_symbols = positions[target_mask]
            target_rows = torch.arange(len(messages)).unsqueeze(1)
            target_rows = target_rows.expand(-1, size[-1] - 1)[target_mask]
        else:
            pass

    def reinforce(self, messages, entropies, apply_noise, lengths=None):
        pass


class SymmetricChannel(Channel):
    """
    Replaces each symbol with a different symbol with a given probability.
    The replacement symbol is randomly sampled from a uniform distribution.
    """

    def gs(self, messages, probs, apply_noise):
        if not apply_noise:
            return messages, probs

        if self.training:
            # reshape & sample targets
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[-1])
            target_mask = torch.rand(
                messages[:, 1:].size(),
                generator=self.generator,
                device=self.device,
            ) < self.p
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages.view(size), probs

            # get target positions & messages
            non_eos_symbols = torch.arange(1, size[-1]).expand(len(messages), -1)
            target_symbols = non_eos_symbols[target_mask]
            target_rows = torch.arange(len(messages)).unsqueeze(1)
            target_rows = target_rows.expand(-1, size[-1] - 1)[target_mask]
            target_messages = messages[target_rows, target_symbols]

            # find candidate symbols different from target symbols
            non_eos_symbols = non_eos_symbols[0].expand(n_targets, -1)
            candidate_mask = non_eos_symbols != target_symbols.unsqueeze(-1)
            candidate_symbols = non_eos_symbols[candidate_mask]
            candidate_symbols = candidate_symbols.view(-1, size[-1] - 2)

            # sample replacement symbols
            replacement_ids = torch.randint(
                size=(n_targets,),
                high=messages.size(-1) - 2,
                generator=self.generator,
                device=self.device)
            replacement_symbols = candidate_symbols[torch.arange(n_targets), replacement_ids]

            # adjust message tensors
            adjustment = torch.zeros_like(messages)
            adjustment[target_rows, target_symbols] -= target_messages
            adjustment[target_rows, replacement_symbols] += target_messages

            messages = (messages + adjustment).view(size)

            # adjust symbol probabilities
            probs = probs.clone()
            p_replacement = (1 - probs[:, :, 1:] - probs[:, :, :1])
            p_replacement *= self.p / (size[-1] - 2)
            probs[:, :, 1:] *= 1 - self.p
            probs[:, :, 1:] += p_replacement
            assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)))

            return messages, probs

        else:
            # reshape, apply argmax, exclude EOS, sample symbols to be replaced
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[-1])
            discrete_symbols = messages.argmax(-1)
            non_eos_ids = torch.arange(len(messages))[discrete_symbols != 0]
            target_mask = torch.rand(
                non_eos_ids.numel(),
                generator=self.generator,
                device=self.device,
            ) < self.p
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages.view(size), probs

            # get target positions & symbols
            target_rows = non_eos_ids[target_mask]
            target_symbols = discrete_symbols[target_rows]

            # find candidate symbols different from target symbols
            candidate_symbols = torch.arange(1, size[-1]).unsqueeze(0).expand(n_targets, -1)
            candidate_mask = candidate_symbols != target_symbols.unsqueeze(-1)
            candidate_symbols = candidate_symbols[candidate_mask]
            candidate_symbols = candidate_symbols.view(-1, size[-1] - 2)

            # sample replacement symbols
            replacement_ids = torch.randint(
                size=(n_targets,),
                high=messages.size(-1) - 2,
                generator=self.generator,
                device=self.device)
            replacement_symbols = candidate_symbols[torch.arange(n_targets), replacement_ids]

            # replace probabilities
            messages[target_rows] = 0
            messages[target_rows, replacement_symbols] = 1
            messages = messages.view(size)

            # adjust symbol probabilities
            probs = probs.clone()
            p_replacement = (1 - probs[:, :, 1:] - probs[:, :, :1])
            p_replacement *= self.p / (size[-1] - 2)
            probs[:, :, 1:] *= 1 - self.p
            probs[:, :, 1:] += p_replacement
            assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)))

            return messages, probs

    def reinforce(self, messages, entropies, apply_noise, lengths=None):
        raise NotImplementedError
        if not apply_noise:
            return messages, entropies

        size = messages.size()
        messages = messages.clone().view(size[0] * size[1])
        non_eos_ids = torch.arange(len(messages))[messages != 0]
        target_mask = torch.rand(
            non_eos_ids.numel(),
            generator=self.generator,
            device=self.device,
        ) < self.p
        n_targets = target_mask.sum()

        if n_targets == 0:
            return messages.view(size), entropies

        # target positions & symbols
        target_rows = non_eos_ids[target_mask]
        target_symbols = messages[target_rows]

        # find candidate symbols different from target symbols
        candidate_symbols = (
            torch.arange(1, self.vocab_size)
            .unsqueeze(0)
            .expand(n_targets, -1))
        candidate_mask = candidate_symbols != target_symbols.unsqueeze(-1)
        candidate_symbols = candidate_symbols[candidate_mask].view(-1, self.vocab_size - 2)

        # sample replacement symbols
        replacement_ids = torch.randint(
            size=(n_targets,),
            high=self.vocab_size - 2,
            generator=self.generator,
            device=self.device)
        replacement_symbols = candidate_symbols[torch.arange(n_targets), replacement_ids]

        # replace
        messages[target_rows] = replacement_symbols

        return messages.view(size), entropies
