import torch
import numpy as np
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Channel(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        error_prob: float,
        max_len: int,
        vocab_size: int,
        device: torch.device = torch.device("cpu"),
        seed: int = 42
    ):
        super().__init__()
        self.p = torch.tensor(error_prob, device=device, requires_grad=False)
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
    def process(
        messages: torch.Tensor,
        probs: torch.Tensor,
        apply_noise: bool
    ):
        return

    # @staticmethod
    # def tensor_binary_entropy(p: torch.Tensor):
    #     q = 1 - p
    #     min_real = torch.finfo(p.dtype).min
    #     log2_p = torch.clamp(torch.log2(p), min=min_real)
    #     log2_q = torch.clamp(torch.log2(q), min=min_real)
    #     return -p * log2_p - q * log2_q

    def sample_targets(self, *size):
        return torch.rand(
            *size,
            generator=self.generator,
            device=self.device
        ) < self.p

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

    def forward(self, messages: torch.Tensor, probs: torch.Tensor, *args):
        return (
            self.process(messages, probs, apply_noise=True),
            self.process(messages, probs, apply_noise=False),
        )


class NoChannel(Channel):
    def process(self, messages, probs, apply_noise):
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

    def process(self, messages, probs, apply_noise):
        if not apply_noise:
            placeholder_probs = torch.zeros_like(messages[:, :, :1])
            messages = torch.cat([messages, placeholder_probs], dim=-1)
            probs = torch.cat([probs, placeholder_probs], dim=-1)
            return messages, probs

        elif self.training:
            target_mask = self.sample_targets(messages.size()[:-1])

            # append a column for erased symbols (creates new tensors)
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
            target_mask = self.sample_targets(non_eos_mask.sum())
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


class DeletionChannel(Channel):

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

    def gs(self, messages, probs, apply_noise):
        if messages.is_cuda:
            self.shift = torch.compile(self.shift)

        if not apply_noise:
            return messages, probs

        elif self.training:
            # sample target symbols
            size = messages.size()
            messages = messages.clone()

            target_mask = self.sample_targets(size[:-1])
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages, probs

            # shift target symbols to the end of the message
            self.shift(messages, target_mask)

            # get new target positions
            n_deleted = torch.sum(target_mask.int().clamp(0, 1), dim=1).unsqueeze(-1)
            n_deleted_2 = target_mask.int().clamp(0, 1).sum(1, keepdim=True)
            assert torch.all(n_deleted_2 == n_deleted)
            target_mask = (
                torch.arange(size[1] - 1, -1, step=-1)
                .unsqueeze(0)
                .expand(size[:-1])
            ) < n_deleted

            # delete
            replacement_probs = torch.zeros_like(messages[target_mask])
            replacement_probs[..., 0] = 1
            messages[target_mask] = replacement_probs

            # adjust probabilities
            probs = probs.clone()
            probs[..., 1:] *= 1 - self.p
            probs[..., 0] = 1 - probs[..., 1:].sum(-1)

            return messages, probs

        else:
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[2])
            discrete_symbols = messages.argmax(-1)
            non_eos_ids = torch.arange(len(messages))[discrete_symbols != 0]
            target_mask = self.sample_targets(non_eos_ids.numel())
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages.view(size), probs

            # get target positions & symbols
            target_rows = non_eos_ids[target_mask]
            target_mask = torch.zeros_like(messages[:, 0]).bool()
            target_mask[target_rows] = True
            messages = messages.view(size)
            target_mask = target_mask.view(size[:-1])

            # shift target symbols to the end of the message
            self.shift(messages, target_mask)

            # get new target positions
            n_deleted = target_mask.int().clamp(0, 1).sum(1, keepdim=True)
            target_mask = (
                torch.arange(size[1] - 1, -1, step=-1)
                .unsqueeze(0)
                .expand(size[:-1])
            ) < n_deleted

            # delete
            replacement_probs = torch.zeros_like(messages[target_mask])
            replacement_probs[..., 0] = 1
            messages[target_mask] = replacement_probs

            # adjust probabilities
            probs = probs.clone().view(size[0] * size[1], size[2])
            probs[non_eos_ids, 1:] *= 1 - self.p
            probs[non_eos_ids, 0] = 1 - probs[non_eos_ids, 1:].sum(-1)
            probs = probs.view(size)

            return messages, probs


class SymmetricChannel(Channel):
    """
    Replaces each symbol with a different symbol with a given probability.
    The replacement symbol is randomly sampled from a uniform distribution.
    """

    def process(self, messages, probs, apply_noise):
        if not apply_noise:
            return messages, probs

        elif self.training:
            # reshape & sample targets
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[-1])
            target_mask = self.sample_targets(messages[:, 1:].size())
            # target_mask = self.sample_targets(size[0] * size[1])
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages.view(size), probs

            # get target positions & highest symbol probs
            rows = torch.arange(len(target_mask))
            cols = torch.arange(size[-1] - 1)
            target_rows = rows.expand(size[-1] - 1, -1).t()[target_mask]
            target_cols = cols.expand(target_mask.size())[target_mask]

            # compute probability adjustment for each symbol with a target
            targets = messages.clone()[target_rows, target_cols]
            adjustment = targets.expand(size[-1] - 1, -1).t() / (size[-1] - 2)
            adjustment[torch.arange(len(adjustment)), target_cols] = -targets

            # adjust relaxed symbols
            messages[target_rows, 1:] += adjustment
            messages = messages.view(size)

            # assert torch.allclose(messages.sum(-1), torch.ones_like(messages.sum(-1)))

            # adjust symbol probabilities
            probs = probs.clone()
            p_replacement = (1 - probs[:, :, 1:] - probs[:, :, :1])
            p_replacement *= self.p / (size[-1] - 2)
            probs[:, :, 1:] *= 1 - self.p
            probs[:, :, 1:] += p_replacement
            # assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]))

            return messages, probs

        else:
            # reshape, apply argmax, exclude EOS, sample symbols to be replaced
            size = messages.size()
            messages = messages.clone().view(size[0] * size[1], size[-1])
            discrete_symbols = messages.argmax(-1)
            non_eos_ids = torch.arange(len(messages))[discrete_symbols != 0]
            target_mask = self.sample_targets(non_eos_ids.numel())
            n_targets = target_mask.sum()

            if n_targets == 0:
                return messages.view(size), probs

            # get target positions & symbols
            target_rows = non_eos_ids[target_mask]
            target_symbols = discrete_symbols[target_rows]

            # find candidate symbols different from target symbols
            n_targets = target_symbols.numel()
            symbols = torch.arange(1, messages.size(-1)).expand(n_targets, -1)
            candidate_mask = symbols != target_symbols.unsqueeze(-1)
            candidate_symbols = symbols[candidate_mask].view(-1, size[-1] - 2)

            # sample replacement symbols
            replacement_ids = torch.randint(
                size=(n_targets,),
                high=messages.size(-1) - 2,
                generator=self.generator,
                device=self.device,
            )
            idx = (torch.arange(n_targets), replacement_ids)
            replacement_symbols = candidate_symbols[idx]

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

            return messages, probs
