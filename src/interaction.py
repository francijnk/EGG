# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as distrib

from egg.core.batch import Batch


@dataclass(repr=True, eq=True)
class LoggingStrategy:
    store_sender_input: bool = True
    store_receiver_input: bool = True
    store_labels: bool = True
    store_aux_input: bool = True
    store_message: bool = True
    store_symbol_logits: bool = True
    store_receiver_output: bool = True
    store_message_length: bool = True
    store_message_nn: bool = True
    store_symbol_logits_nn: bool = True
    store_receiver_output_nn: bool = True
    store_message_length_nn: bool = True

    def filtered_interaction(
        self,
        sender_input: Optional[torch.Tensor],
        receiver_input: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        aux_input: Optional[Dict[str, torch.Tensor]],
        message: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
        receiver_output: Optional[torch.Tensor],
        message_length: Optional[torch.Tensor],
        message_nn: Optional[torch.Tensor],
        logits_nn: Optional[torch.Tensor],
        receiver_output_nn: Optional[torch.Tensor],
        message_length_nn: Optional[torch.Tensor],
        aux: Dict[str, torch.Tensor],
    ):

        return Interaction(
            sender_input=sender_input if self.store_sender_input else None,
            receiver_input=receiver_input if self.store_receiver_input else None,
            labels=labels if self.store_labels else None,
            aux_input=aux_input if self.store_aux_input else None,
            message=message if self.store_message else None,
            logits=logits if self.store_symbol_logits else None,
            receiver_output=receiver_output if self.store_receiver_output else None,
            message_length=message_length if self.store_message_length else None,
            message_nn=message_nn if self.store_message_nn else None,
            logits_nn=logits_nn if self.store_symbol_logits_nn else None,
            receiver_output_nn=receiver_output_nn if self.store_receiver_output_nn else None,
            message_length_nn=message_length_nn if self.store_message_length_nn else None,
            aux=aux,
        )

    @classmethod
    def minimal(cls):
        args = [False] * 7
        return cls(*args)

    @classmethod
    def maximal(cls):
        return cls()


@dataclass(repr=True, unsafe_hash=True)
class Interaction:
    # incoming data
    sender_input: Optional[torch.Tensor]
    receiver_input: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    aux_input: Optional[Dict[str, torch.Tensor]]

    # what agents produce
    message: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    receiver_output: Optional[torch.Tensor]
    message_length: Optional[torch.Tensor]

    # what would happen if no noise was added
    message_nn: Optional[torch.Tensor]
    logits_nn: Optional[torch.Tensor]
    receiver_output_nn: Optional[torch.Tensor]
    message_length_nn: Optional[torch.Tensor]

    # auxilary info
    aux: Dict[str, torch.Tensor]

    def __add__(self, other) -> "Interaction":
        """
        Defines the behaviour of the + operator between two Interaction objects
        """

        def _check_cat(lst):
            if all(x is None for x in lst):
                return None
            # if some but not all are None: not good
            if any(x is None for x in lst):
                raise RuntimeError(
                    "Appending empty and non-empty interactions logs. "
                    "Normally this shouldn't happen!"
                )
            return torch.cat(lst, dim=0)

        def _combine_aux_dicts(dict1, dict2):
            new_dict = {}
            keys = set(list(dict1.keys()) + list(dict2.keys()))
            for k in keys:
                if k in dict1 and k in dict2:
                    new_dict[k] = torch.cat((dict1[k], dict2[k]))
                else:
                    new_dict[k] = dict1[k] if k in dict1 else dict2[k]
            return new_dict

        interactions = [self, other]
        return Interaction(
            sender_input=_check_cat([x.sender_input for x in interactions]),
            receiver_input=_check_cat([x.receiver_input for x in interactions]),
            labels=_check_cat([x.labels for x in interactions]),
            aux_input=_combine_aux_dicts(self.aux_input, other.aux_input),
            message=_check_cat([x.message for x in interactions]),
            logits=_check_cat([x.logits for x in interactions]),
            receiver_output=_check_cat([x.receiver_output for x in interactions]),
            message_length=_check_cat([x.message_length for x in interactions]),
            message_nn=_check_cat([x.message_nn for x in interactions]),
            logits_nn=_check_cat([x.logits_nn for x in interactions]),
            receiver_output_nn=_check_cat([x.receiver_output_nn for x in interactions]),
            message_length_nn=_check_cat([x.message_length_nn for x in interactions]),
            aux=_combine_aux_dicts(self.aux, other.aux),
        )

    @property
    def size(self):
        interaction_fields = [
            self.sender_input,
            self.receiver_input,
            self.labels,
            self.message,
            self.logits,
            self.receiver_output,
            self.message_length,
            self.message_nn,
            self.logits_nn,
            self.receiver_output_nn,
            self.message_length_nn,
        ]
        for t in interaction_fields:
            if t is not None:
                return t.size(0)
        raise RuntimeError("Cannot determine interaction log size; it is empty.")

    def to(self, *args, **kwargs) -> "Interaction":
        """Moves all stored tensor to a device. For instance, it might be not
        useful to store the interaction logs in CUDA memory."""

        def _to(x):
            if x is None or not torch.is_tensor(x):
                return x
            return x.to(*args, **kwargs)

        self.sender_input = _to(self.sender_input)
        self.receiver_input = _to(self.receiver_input)
        self.labels = _to(self.labels)
        self.message = _to(self.message)
        self.logits = _to(self.logits)
        self.receiver_output = _to(self.receiver_output)
        self.message_length = _to(self.message_length)
        self.message_nn = _to(self.message_nn)
        self.logits_nn = _to(self.logits_nn)
        self.receiver_output_nn = _to(self.receiver_output_nn)
        self.message_length_nn = _to(self.message_length_nn)

        if self.aux_input:
            self.aux_input = dict((k, _to(v)) for k, v in self.aux_input.items())
        if self.aux:
            self.aux = dict((k, _to(v)) for k, v in self.aux.items())

        return self

    @staticmethod
    def from_iterable(interactions: Iterable["Interaction"]) -> "Interaction":
        def _check_cat(lst):
            if all(x is None for x in lst):
                return None
            # if some but not all are None: not good
            if any(x is None for x in lst):
                raise RuntimeError(
                    "Appending empty and non-empty interactions logs. "
                    "Normally this shouldn't happen!"
                )
            return torch.cat(lst, dim=0)

        assert interactions, "interaction list must not be empty"
        has_aux_input = interactions[0].aux_input is not None
        for x in interactions:
            assert len(x.aux) == len(interactions[0].aux)
            if has_aux_input:
                assert len(x.aux_input) == len(
                    interactions[0].aux_input
                ), "found two interactions of different aux_info size"
            else:
                assert (
                    not x.aux_input
                ), "some aux_info are defined some are not, this should not happen"

        aux_input = None
        if has_aux_input:
            aux_input = {}
            for k in interactions[0].aux_input:
                aux_input[k] = _check_cat([x.aux_input[k] for x in interactions])
        aux = {}
        for k in interactions[0].aux:
            aux[k] = _check_cat([x.aux[k] for x in interactions])

        return Interaction(
            sender_input=_check_cat(
                [x.sender_input for x in interactions]),
            receiver_input=_check_cat(
                [x.receiver_input for x in interactions]),
            labels=_check_cat(
                [x.labels for x in interactions]),
            aux_input=aux_input,
            message=_check_cat(
                [x.message for x in interactions]),
            logits=_check_cat(
                [x.logits for x in interactions]),
            message_length=_check_cat(
                [x.message_length for x in interactions]),
            receiver_output=_check_cat(
                [x.receiver_output for x in interactions]),
            message_nn=_check_cat(
                [x.message_nn for x in interactions]),
            logits_nn=_check_cat(
                [x.logits_nn for x in interactions]),
            message_length_nn=_check_cat(
                [x.message_length_nn for x in interactions]),
            receiver_output_nn=_check_cat(
                [x.receiver_output_nn for x in interactions]),
            aux=aux,
        )

    @staticmethod
    def empty() -> "Interaction":
        return Interaction(
            None, None, None, {}, None, None, None,
            None, None, None, None, None, {})

    @staticmethod
    def gather_distributed_interactions(log: "Interaction") -> Optional["Interaction"]:
        assert (
            distrib.is_initialized()
        ), "torch.distributed must be initialized beforehand"
        world_size = distrib.get_world_size()

        def send_collect_tensor(tnsr):
            assert tnsr is not None

            tnsr = tnsr.contiguous().cuda()
            lst = [torch.zeros_like(tnsr) for _ in range(world_size)]
            distrib.all_gather(lst, tnsr)
            return torch.cat(lst, dim=0).to("cpu")

        def send_collect_dict(d):
            if not d or d is None:
                return {}

            new_d = {}
            for k, v in d.items():
                if v is not None:
                    v = send_collect_tensor(v)
                new_d[k] = v

            return new_d

        interaction_as_dict = dict(
            (name, getattr(log, name))
            for name in [
                "sender_input",
                "receiver_input",
                "labels",
                "message",
                "logits",
                "message_length",
                "receiver_output",
                "message_nn",
                "logits_nn",
                "message_length_nn",
                "receiver_output_nn",
            ]
        )

        interaction_as_dict = send_collect_dict(interaction_as_dict)

        synced_aux_input = send_collect_dict(log.aux_input)
        interaction_as_dict["aux_input"] = synced_aux_input
        synced_aux = send_collect_dict(log.aux)
        interaction_as_dict["aux"] = synced_aux

        synced_interacton = Interaction(**interaction_as_dict)

        assert log.size * world_size == synced_interacton.size
        return synced_interacton
