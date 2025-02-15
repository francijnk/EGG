import torch
import numpy as np
import torch.nn as nn
from egg.core.util import find_lengths
from scipy.optimize import minimize_scalar
from abc import ABCMeta, abstractmethod
from typing import Optional

from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs


# TODO instead of entropies, adjust probs


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

        # compute maximum achievable entropies
        self.max_symbol_entropy = torch.tensor(self._max_symbol_entropy())
        # self.max_message_entropy = torch.tensor(self._max_message_entropy())
        self.max_message_entropy = {
            i: self._max_message_entropy(i)
            for i in range(self.max_len + 1)
        }  # for every message length

    @abstractmethod
    def gs(
            probs: torch.Tensor,
            entropies: torch.Tensor,
            apply_noise: bool):
        return

    @abstractmethod
    def reinforce(
            messages: torch.Tensor,
            entropies: torch.Tensor,
            apply_noise: bool, **kwargs):
        return

    @staticmethod
    def tensor_binary_entropy(p: torch.Tensor):
        q = 1 - p
        min_real = torch.finfo(p.dtype).min
        log2_p = torch.clamp(torch.log2(p), min=min_real)
        log2_q = torch.clamp(torch.log2(q), min=min_real)
        return -p * log2_p - q * log2_q

    @staticmethod
    def binary_entropy(p: float):
        if p == 0. or p == 1.:
            return 0.
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    @abstractmethod
    def _max_symbol_entropy(self, vocab_size: Optional[int]):
        return

    def _max_message_entropy(self, length: int):
        return self._max_symbol_entropy(self.vocab_size - 1) * length

    # @abstractmethod
    # def _message_entropy(self, eos_prob, max_suffix_entropy):
    #     return

    # def _max_message_entropy(self, max_len=None, max_iter=5000):
    #     if max_len is None:
    #         max_len = self.max_len

    #    if max_len == 1:
    #        entropy = lambda p: -self._message_entropy(p, 0)
    #        optimal_eos_prob = minimize_scalar(
    #            entropy,
    #            method='bounded', bounds=(0., 1.),
    #            options={'maxiter': max_iter})
    #        return entropy(optimal_eos_prob.x)  # , [optimal_eos_prob.x]

    #    max_suffix_entropy = self._max_message_entropy(max_len - 1)
    #    entropy = lambda p: -self._message_entropy(p, max_suffix_entropy)
    #    optimal_eos_prob = minimize_scalar(
    #        entropy,
    #        method='bounded', bounds=(0., 1.),
    #        options={'maxiter': max_iter})
    #    # eos_probs = [optimal_eos_prob.x] + eos_probs

    #    return -entropy(optimal_eos_prob.x)  # , eos_probs

    def compute_max_entropy(self, length_probs):
        min_real = torch.finfo(length_probs.dtype).min
        length_log2_prob = torch.clamp(torch.log2(length_probs), min=min_real)
        entropy_length = (-length_probs * length_log2_prob).sum(-1)

        # print(length_probs)
        max_entropy = entropy_length.clone()
        for i in range(len(length_probs)):
            max_entropy_i = self.max_message_entropy[i]
            max_entropy += length_probs[i] * max_entropy_i
        # print(max_entropy)

        return max_entropy

    def update_values(self, output_dict):
        # compute entropy of message length
        length_probs = output_dict['length_probs']
        # min_positive = torch.finfo(length_probs.dtype).tiny
        min_real = torch.finfo(length_probs.dtype).min
        length_log2_prob = torch.clamp(torch.log2(length_probs), min=min_real)
        entropy_length = (-length_probs * length_log2_prob).sum(-1)

        # adjust entropy values to cover message length variability
        # TODO if we bring deletion ch. back, entopy length w/o needs to be adjusted
        output_dict['entropy_msg'] += entropy_length
        output_dict['entropy_msg_nn'] += entropy_length

        # exclude appended EOS
        output_dict['entropy_smb'] = output_dict['entropy_smb'][:, :-1]
        output_dict['entropy_smb_nn'] = output_dict['entropy_smb_nn'][:, :-1]

        # compute_redundancy
        entropy_msg = output_dict['entropy_msg']
        entropy_smb = output_dict['entropy_smb']
        entropy_msg_nn = output_dict['entropy_msg_nn']
        entropy_smb_nn = output_dict['entropy_smb_nn']
        length_probs = output_dict['length_probs']

        # output_dict['redund_msg'] = 1 - entropy_msg / self.max_message_entropy
        output_dict['redundancy_smb'] = \
            (1 - entropy_smb / self.max_symbol_entropy).mean(-1)
        # output_dict['redund_msg_nn'] = 1 - entropy_msg_nn / self.max_message_entropy
        output_dict['redundancy_smb_nn'] = \
            (1 - entropy_smb_nn / self.max_symbol_entropy).mean(-1)
        # max_symbol_entropy_nn should be adjusted TODO

        # max_entropy_adj = torch.zeros_like(entropy_msg)
        max_entropy_adj = entropy_length.clone()
        for i in range(length_probs.size(1)):
            length_prob_i = length_probs[:, i]
            max_entropy_i = self.max_message_entropy[i]
            max_entropy_adj += length_prob_i * max_entropy_i

        # assume empty messages have redundancy 1
        output_dict['max_entropy'] = max_entropy_adj
        output_dict['redundancy_msg'] = torch.where(
            max_entropy_adj > 0,
            1 - entropy_msg / max_entropy_adj,
            1)

        # TODO redundancy w/o noise

        # TODO remove this check later
        # mask = output_dict['redundancy_msg'] < 0
        # if torch.any(mask):
        #    # print("msg", output_dict['message'][mask])
        #    print("message entropy", entropy_msg[mask])
        #    print("max entropy adj", max_entropy_adj[mask])
        #    print("redund", output_dict['redundancy_msg'][mask])
        #    print("length probs", length_probs[mask])
        #    print("length probs", length_probs[mask].sum(-1))
        #    max_entropies = torch.tensor(list(self.max_message_entropy.values()))
        #    max_entropies = max_entropies.unsqueeze(0)
        #    print(length_probs[mask] * max_entropies)
        #    print((length_probs[mask] * max_entropies).sum(-1) + entropy_length[mask])
        #    print("")

        return output_dict

    def forward(self, messages, **kwargs):
        # GS
        if messages.dim() == 3:
            symbol_entropies = kwargs['entropy']

            msg, entropies = self.gs(messages, symbol_entropies, True)
            msg_nn, entropies_nn = self.gs(messages, symbol_entropies, False)

            output_dict = {
                # 'message': _messages,
                # 'message_nn': messages,
                'entropy_msg': torch.zeros_like(msg[:, 0, 0]),
                'entropy_msg_nn': torch.zeros_like(msg_nn[:, 0, 0]),
                'entropy_smb': entropies.detach(),
                'entropy_smb_nn': entropies_nn.detach(),
                'length_probs': torch.zeros(
                    messages.size(0),
                    messages.size(1) + 1,
                    requires_grad=False).to(messages),
                'length_probs_nn': torch.zeros(  # TODO if deletion comes back
                    messages.size(0),
                    messages.size(1) + 1,
                    requires_grad=False).to(messages),
            }

            return msg, msg_nn, output_dict

        # Reinforce
        else:
            raise NotImplementedError


class NoChannel(Channel):
    def _max_symbol_entropy(self, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size - 1
        return np.log2(vocab_size)

    def gs(self, probs, entropies, apply_noise):
        return probs, entropies

    def reinforce(self, probs, entropies, apply_noise, **kwargs):
        return probs, entropies

    # def _entropy(self, eos_prob):
    #    # uniform symbol distribution maximizes entropy of 1 symbol messages
    #    return (
    #        self.binary_entropy(eos_prob)
    #        # + eos_prob * 0
    #        + (1 - eos_prob) * (
    #            np.log2(se
    #        )
    #    )
    # np.log2(self.vocab_size)

    # def _message_entropy(self, eos_prob, max_suffix_entropy):
    #     return (
    #         self.binary_entropy(eos_prob)
    #         # + eos_prob * 0
    #         + (1 - torch.tensor(eos_prob)) * (
    #             torch.log2(torch.tensor(self.vocab_size) - 1)
    #             + max_suffix_entropy
    #         )
    #     )


class ErasureChannel(Channel):
    """
    Erases a symbol from a message with a given probability
    """

    def _max_symbol_entropy(self, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size - 1
        return (
            self.binary_entropy(self.p.item())
            + (1 - self.p.item()) * np.log2(vocab_size)
            # + self.p * 0
        )

    # def _max_msg_entropy(self, length):
    #     max_entropy = 0
    #     for i in range(length - 1):
    #         max_entropy +=  (
    #             self.binary_entropy(self.p.item())
    #             + (1 - self.p.item()) * np.log2(self.vocab_size - 1)
    #             # + self.p.item() * 0
    #         )
    #     # only the last symbol might be EOS
    #     max_entropy += (
    #         self.binary_entropy(self.p.item())
    #         + (1 - self.p.item()) * np.log2(self.vocab_size)
    #         # + self.p.item() * 0
    #    )
    #     return max_entropy
    # def _entropy(self, eos_prob):
    #     error_prob = self.p.item()
    #     erased_prob = (1 - eos_prob) * error_prob
    #     return (
    #         - eos_prob * np.log2(eos_prob)
    #         - erased_prob * np.log2(erased_prob)
    #         - (1 - eos_prob) * (1 - erased_prob) * (
    #             np.log2(1 - eos_prob)
    #             + np.log2(1 - erased_prob)
    #             - np.log2(self.vocab_size - 1)
    #        )
    #     )

    # def _message_entropy(self, eos_prob, max_suffix_entropy):
    #     error_prob = self.p.item()
    #     return (
    #         self.binary_entropy(eos_prob)
    #         # + eos_prob * 0
    #         + (1 - eos_prob) * (
    #             self.binary_entropy(error_prob)
    #             + error_prob * max_suffix_entropy
    #             + (1 - error_prob) * (
    #                 np.log2(self.vocab_size - 1)
    #                 + max_suffix_entropy
    #             )
    #         )
    #     )

    def gs(self, probs, entropies, apply_noise):
        if not apply_noise:
            placeholder_probs = torch.zeros_like(probs[:, :, :1])
            probs = torch.cat([probs, placeholder_probs], dim=-1)
            return probs, entropies

        elif self.training:
            target_mask = torch.rand(
                probs.size()[:-1],
                generator=self.generator,
                device=self.device,
            ) < self.p

            # append a column for erased symbols
            placeholder_probs = torch.zeros_like(probs[:, :, :1])
            probs = torch.cat([probs, placeholder_probs], dim=-1)

            if target_mask.sum() == 0:
                return probs, entropies

            # create a replacement probability array and replace
            erased_probs = torch.zeros_like(probs)
            erased_probs[:, :, 0] = probs[:, :, 0]
            erased_probs[:, :, -1] = 1 - probs[:, :, 0]

            target_probs = torch.zeros_like(probs).to(torch.bool)
            target_probs[target_mask] = 1
            probs = torch.where(target_probs, erased_probs, probs)

            # adjust symbol entropies
            p_eos = probs[:, :, 0].detach()
            h_eos = self.tensor_binary_entropy(p_eos)
            H_non_eos = (entropies - h_eos) / (1 - p_eos)
            entropies = (
                h_eos  # + p_eos * 0
                + (1 - p_eos) * (
                    self.tensor_binary_entropy(self.p)
                    + (1 - self.p) * H_non_eos
                )
            )

            return probs, entropies

        else:
            # append a column for erased symbols
            placeholder_probs = torch.zeros_like(probs[:, :, :1])
            probs = torch.cat([probs, placeholder_probs], dim=-1)

            # apply argmax, exclude EOS, sample batch rows to be replaced
            discrete_symbols = probs.argmax(-1)
            non_eos_mask = discrete_symbols != 0
            non_eos_symbols = discrete_symbols[non_eos_mask]
            target_mask = torch.rand(
                non_eos_mask.sum(),
                generator=self.generator,
                device=self.device
            ) < self.p
            n_targets = target_mask.sum()

            if n_targets == 0:
                return probs, entropies

            # prepare the index and source of replacement
            target_probs = torch.zeros_like(probs).bool()
            target_probs[non_eos_mask] = torch.where(
                target_mask.unsqueeze(-1),
                torch.ones(target_mask.size(0), probs.size(-1)).bool(),
                False)
            erased_probs = torch.zeros_like(probs)
            erased_probs[:, :, -1] = 1

            # replace
            probs = torch.where(target_probs, erased_probs, probs)

            # adjust entropy
            entropies = entropies.clone()
            entropies[non_eos_mask] = (
                self.tensor_binary_entropy(self.p)
                + (1 - self.p) * entropies[non_eos_mask]
            )

            return probs, entropies

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

        non_eos_positions = torch.arange(1, size[-1]).expand(size[0], -1)
        target_symbols = non_eos_positions[target_mask]
        target_rows = torch.arange(size[0]).unsqueeze(1)
        target_rows = target_rows.expand(-1, size[-1])[target_mask]

        messages[target_rows, target_symbols] = self.vocab_size
        return messages  # TODO check


class DeletionChannel(Channel):
    pass


class SymmetricChannel(Channel):
    """
    Replaces each symbol with a different symbol with a given probability.
    The replacement symbol is randomly sampled from a uniform distribution.
    """

    def _max_symbol_entropy(self, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size - 1
        return np.log2(vocab_size)

    def gs(self, probs, entropies, apply_noise):
        if not apply_noise:
            return probs, entropies

        elif self.training:
            # reshape & sample targets
            size = probs.size()
            probs = probs.clone().view(size[0] * size[1], size[-1])
            target_mask = torch.rand(
                probs[:, 1:].size(),
                generator=self.generator,
                device=self.device,
            ) < self.p
            n_targets = target_mask.sum()

            if n_targets == 0:
                # TODO entropy should still be adjusted (also for other channels)
                return probs.view(size), entropies

            # get target positions & probs
            non_eos_positions = torch.arange(1, size[-1]).expand(len(probs), -1)
            target_symbols = non_eos_positions[target_mask]
            target_rows = torch.arange(len(probs)).unsqueeze(1)
            target_rows = target_rows.expand(-1, size[-1] - 1)[target_mask]
            target_probs = probs[target_rows, target_symbols]

            # find candidate symbols different from target symbols
            non_eos_positions = non_eos_positions[0].expand(n_targets, -1)
            candidate_mask = non_eos_positions != target_symbols.unsqueeze(-1)
            candidate_symbols = non_eos_positions[candidate_mask]
            candidate_symbols = candidate_symbols.view(-1, size[-1] - 2)

            # sample replacement symbols
            replacement_ids = torch.randint(
                size=(n_targets,),
                high=probs.size(-1) - 2,
                generator=self.generator,
                device=self.device)
            replacement_symbols = \
                candidate_symbols[torch.arange(n_targets), replacement_ids]

            # adjust symbol probabilities
            adjustment = torch.zeros_like(probs)
            adjustment[target_rows, target_symbols] -= target_probs
            adjustment[target_rows, replacement_symbols] += target_probs

            probs = (probs + adjustment).view(size)

            # adjust entropy
            p_eos = probs[:, :, 0].detach()
            h_eos = self.tensor_binary_entropy(p_eos)
            H_non_eos = (entropies - h_eos) / (1 - p_eos)
            entropies = (
                h_eos  # + p_eos * 0
                + (1 - p_eos) * (
                    self.tensor_binary_entropy(self.p)
                    + self.p * torch.log2(torch.tensor(probs.size(-1) - 2))
                    + (1 - self.p) * H_non_eos
                )
            )

            return probs, entropies

        else:
            # reshape, apply argmax, exclude EOS, sample symbols to be replaced
            size = probs.size()
            probs = probs.clone().view(size[0] * size[1], size[-1])
            discrete_symbols = probs.argmax(-1)
            non_eos_ids = torch.arange(len(probs))[discrete_symbols != 0]
            target_mask = torch.rand(
                non_eos_ids.numel(),
                generator=self.generator,
                device=self.device,
            ) < self.p
            n_targets = target_mask.sum()

            if n_targets == 0:
                return probs.view(size), entropy

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
                high=probs.size(-1) - 2,
                generator=self.generator,
                device=self.device)
            replacement_symbols = candidate_symbols[torch.arange(n_targets), replacement_ids]

            # replace probabilities
            probs[target_rows] = 0
            probs[target_rows, replacement_symbols] = 1
            probs = probs.view(size)

            # adjust entropy for all non EOS symbols
            non_eos_mask = probs.argmax(-1) != 0
            entropies = entropies.clone()
            entropies[non_eos_mask] = (
                self.tensor_binary_entropy(self.p)
                + self.p * torch.log2(torch.tensor(probs.size(-1) - 2))
                + (1 - self.p) * entropies[non_eos_mask]
            )

            return probs, entropies

    def reinforce(self, messages, entropies, apply_noise, lengths=None):
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

        # TODO entropy
        return messages.view(size), entropies
