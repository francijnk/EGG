# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import json
import time
import pathlib
import argparse
from collections import defaultdict
from datetime import timedelta

import torch.utils.data

import egg.core as core

from ancm.trainers import Trainer
from ancm.util import (
    DataHandler,
    Dump,
    build_optimizer,
    print_training_results,
    is_jsonable,
)
from ancm.archs import (
    Sender, Receiver,
    RnnSenderGS,
    loss_rf, loss_gs,
    SenderReceiverRnnReinforce,
    SenderReceiverRnnGS,
)
from ancm.callbacks import (
    CustomProgressBarLogger,
    TrainingEvaluationCallback,
)
from ancm.eval import relative_message_entropy


def get_params(params):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=None, help="Path to .npz inpur data file")
    parser.add_argument("--results_folder", type=str, default='runs', help="Output folder")
    parser.add_argument("--filename", type=str, default=None, help="Output file name (no extension)")

    parser.add_argument(
        "--channel", type=str, default=None,
        help="Communication channel type {erasure, symmetric, deletion} (default: None)")
    parser.add_argument(
        "--error_prob", type=float, default=None, help="Probability of error per symbol (default: 0.0)")
    parser.add_argument(
        "--sender_hidden", type=int, default=64, help="Size of the hidden layer of Sender (default: 64)")
    parser.add_argument(
        "--receiver_hidden", type=int, default=64, help="Size of the hidden layer of Receiver (default: 64)")
    parser.add_argument(
        "--embedding", type=int, default=12,
        help="Dimensionality of the embedding hidden layer for the agents (default: 12)")
    parser.add_argument(
        "--sender_cell", type=str, default="lstm",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: lstm)")
    parser.add_argument(
        "--receiver_cell", type=str, default="lstm",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: lstm)")
    parser.add_argument(
        "--sender_lr", type=float, default=1e-1, help="Learning rate for Sender's parameters (default: 1e-1)")
    parser.add_argument(
        "--receiver_lr", type=float, default=1e-1, help="Learning rate for Receiver's parameters (default: 1e-1)")
    parser.add_argument(
        "--length_cost", type=float, default=1e-2, help="Message length cost")
    parser.add_argument(
        "--mode", type=str, default="gs",
        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training (default: gs)")
    parser.add_argument(
        "--n_permutations_train", type=int, default=None,
        help="Number of order permutations of the objects in the train set")
    parser.add_argument(
        "--image_input", action="store_true", default=False,
        help="Run the image data variant of the game")
    parser.add_argument(
        "--optim", type=str, default="rmsprop", help="Optimizer to use [adam, rmsprop] (default: rmsprop)")
    parser.add_argument(
        "--no_shuffle", action="store_false", default=True,
        help="Do not shuffle train data before every epoch (default: False)")

    # RF-specific
    parser.add_argument(
        "--sender_entropy_coeff", type=float, default=0.01,
        help="RF sender entropy coefficient (default: 0.01)")
    parser.add_argument(
        "--receiver_entropy_coeff", type=float, default=0.001,
        help="RF receiver entropy coefficient (default: 0.001)")

    # GS-specific
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="GS temperature for the sender (default: 1.0)")
    parser.add_argument(
        "--trainable_temperature", action="store_true", default=False, help="Enable trainable temperature")
    parser.add_argument(
        "--temperature_decay", default=None, type=float,
        help="Factor, by which the temperature is decreased every epoch.")
    parser.add_argument(
        "--temperature_minimum", default=None, type=float, help="Minimum temperature value.")

    # W&B
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="WandB entity name")
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument(
        "--wandb_run_id", type=str, default=None, help="WandB run id")
    parser.add_argument(
        "--wandb_group", type=str, default=None, help="WandB group name")

    args = core.init(parser, params)
    check_args(args)
    print(args)
    return args


def check_args(args):

    args.channel = args.channel.lower() if args.channel else args.channel
    assert (
        args.channel is None
        or args.channel in ("erasure", "symmetric", "deletion")
    ), 'The only channels implemented are "erasure", "symmetric", "deletion"'

    args.mode = args.mode.lower()
    assert args.mode in ("rf", "gs")

    if args.channel is None or args.error_prob == 0:
        args.error_prob = 0.0
        args.channel = 'none'

    decay = args.temperature_decay
    minimum = args.temperature_minimum
    assert (
        (decay is None and minimum is None)
        or (decay is not None and minimum is not None)
    )

    if args.results_folder is not None:
        os.makedirs(os.path.dirname(args.results_folder), exist_ok=True)

    args.results_folder = pathlib.Path(args.results_folder) \
        if args.results_folder is not None else None


def main(params):
    opts = get_params(params)

    device = torch.device("cuda" if opts.cuda else "cpu")

    data_handler = DataHandler(opts)
    train_data, validation_data, test_data, train_eval_data = \
        data_handler.load_data(opts)

    if opts.channel == 'erasure' and opts.error_prob != 0:
        vocab_size = opts.vocab_size + 1
    else:
        vocab_size = opts.vocab_size

    _sender = Sender(
        n_features=data_handler.n_features,
        n_hidden=opts.sender_hidden,
        image_input=opts.image_input)
    _receiver = Receiver(
        n_features=data_handler.n_features,
        linear_units=opts.receiver_hidden,
        image_input=opts.image_input)
    if opts.mode == 'rf':
        sender = core.RnnSenderReinforce(
            _sender,
            opts.vocab_size,
            opts.embedding,
            opts.sender_hidden,
            opts.max_len,
            cell=opts.sender_cell)
        receiver = core.RnnReceiverReinforce(
            agent=core.ReinforceWrapper(_receiver),
            vocab_size=vocab_size,
            embed_dim=opts.embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell)
        game = SenderReceiverRnnReinforce(
            sender, receiver,
            loss=loss_rf,
            vocab_size=opts.vocab_size,
            channel_type=opts.channel,
            error_prob=opts.error_prob,
            length_cost=opts.length_cost,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=opts.receiver_entropy_coeff,
            device=device,
            seed=opts.random_seed)
    elif opts.mode == 'gs':
        sender = RnnSenderGS(
            _sender,
            opts.vocab_size,
            opts.embedding,
            opts.sender_hidden,
            opts.max_len,
            opts.temperature,
            opts.sender_cell,
            opts.trainable_temperature)
        receiver = core.RnnReceiverGS(
            _receiver,
            vocab_size,
            opts.embedding,
            opts.receiver_hidden,
            opts.receiver_cell)
        game = SenderReceiverRnnGS(
            sender, receiver,
            loss=loss_gs,
            vocab_size=opts.vocab_size,
            channel_type=opts.channel,
            error_prob=opts.error_prob,
            length_cost=opts.length_cost,
            device=device,
            seed=opts.random_seed)

    optimizer = build_optimizer(game, opts)

    callbacks = [
        TrainingEvaluationCallback(opts, game.channel),
        CustomProgressBarLogger(
            opts,
            train_data_len=len(train_data),
            test_data_len=len(validation_data)),
    ]

    if opts.mode == "gs" and not opts.trainable_temperature \
            and opts.temperature_decay is not None:
        callbacks.append(core.TemperatureUpdater(
            agent=sender,
            decay=opts.temperature_decay,
            minimum=opts.temperature_minimum))

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=None,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks)

    t_start = time.monotonic()
    trainer.train(n_epochs=opts.n_epochs)
    t_end = time.monotonic()
    dump_dict = {}

    # results on the eval subset of the training set (VISA only)
    if train_eval_data is not None:
        train_dump = Dump(game, train_eval_data, opts, device)
        dump_dict['train'] = {
            'evaluation': train_dump.get_eval_dict(),
            'messages': train_dump.get_message_logs(),
        }

    test_dump = Dump(game, test_data, opts, device)
    dump_dict['test'] = {
        'evaluation': test_dump.get_eval_dict(),
        'messages': test_dump.get_message_logs(),
    }

    # if we evaluated on the train set, compute KLD between the protocols
    if train_eval_data is not None:
        for key in dump_dict['train']['evaluation']:
            if key != 'no noise':
                probs_p, probs_q = train_dump.probs, test_dump.probs
            else:
                print('im alive!')
                probs_p, probs_q = train_dump.probs_nn, test_dump.probs_nn

            kld = relative_message_entropy(probs_p, probs_q).item()
            dump_dict['train']['evaluation'][key]['KLD_train_test'] = kld
            dump_dict['test']['evaluation'][key]['KLD_train_test'] = kld

    # save training time
    training_time = timedelta(seconds=t_end - t_start)
    evaluation_time = timedelta(seconds=time.monotonic() - t_start)

    sec_per_epoch = training_time.seconds / opts.n_epochs
    minutes, seconds = divmod(sec_per_epoch, 60)

    training_time = str(training_time).split('.', maxsplit=1)[0]
    evaluation_time = str(evaluation_time).split('.', maxsplit=1)[0]
    training_time_per_epoch = f'{int(minutes):02}:{int(seconds):02}'

    dump_dict['opts'] = {k: v for k, v in vars(opts).items() if is_jsonable(v)}
    dump_dict['training_time'] = {
        'training': training_time,
        'evaluation': evaluation_time,
        'training_per_epoch': training_time_per_epoch,
    }

    if opts.results_folder:
        opts.results_folder.mkdir(exist_ok=True)
        with open(opts.results_folder / f'{opts.filename}-results.json', 'w') as f:
            json.dump(dump_dict, f, indent=4)

        print(f"Results saved to {opts.results_folder / opts.filename}-results.json")

    print('Training time:', training_time)
    print('Training time per epoch:', training_time_per_epoch)
    print('Evaluation time:', evaluation_time)

    print_training_results(dump_dict)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
