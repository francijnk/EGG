# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import json
import time
import pathlib
import argparse
from datetime import timedelta

import torch.utils.data

import egg.core as core

from src.trainers import Trainer
from src.util import (
    DataHandler,
    Dump,
    build_optimizer,
    print_training_results,
    is_jsonable,
)
from src.archs import SenderReceiverRnnGS
from src.callbacks import (
    CustomProgressBarLogger,
    TrainingEvaluationCallback,
    ReceiverResetCallback,
    TemperatureAnnealer,
)
from src.eval import relative_message_entropy


def get_params(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to .npz input file"
    )
    parser.add_argument(
        "--results_folder", type=str, default='runs', help="Output folder"
    )
    parser.add_argument(
        "--filename", type=str, default=None, help="Output file name (no extension)"
    )

    parser.add_argument(
        "--channel", type=str, default=None,
        help="Communication channel type {erasure, deletion} "
        "(default: None)"
    )
    parser.add_argument(
        "--error_prob", type=float, default=None, help="Probability of error "
        "per symbol (default: 0.0)"
    )
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument(
        "--sender_hidden", type=int, default=64, help="Size of the hidden "
        "layer of Sender (default: 64)"
    )
    parser.add_argument(
        "--receiver_hidden", type=int, default=64,
        help="Size of the hidden layer of Receiver (default: 64)"
    )
    parser.add_argument(
        "--embedding", type=int, default=12,
        help="Dimensionality of the embedding hidden layer for the agents "
        "(default: 12)",
    )
    parser.add_argument(
        "--sender_cell", type=str, default="lstm",
        help="Type of the cell used for Sender {rnn, gru, lstm} "
        "(default: lstm)"
    )
    parser.add_argument(
        "--receiver_cell", type=str, default="lstm",
        help="Type of the cell used for Receiver {rnn, gru, lstm} "
        "(default: lstm)"
    )
    parser.add_argument(
        "--sender_lr", type=float, default=1e-1,
        help="Learning rate for Sender's parameters (default: 1e-1)"
    )
    parser.add_argument(
        "--receiver_lr", type=float, default=1e-1,
        help="Learning rate for Receiver's parameters (default: 1e-1)"
    )
    parser.add_argument(
        "--length_cost", type=float, default=1e-2,
        help="Message length cost (default: 1e-2)",
    )
    parser.add_argument(
        "--kld_coeff", type=float, default=0.0,
        help="KLD loss coefficient (default: 0.0)"
    )
    parser.add_argument(
        "--image_input", action="store_true", default=False,
        help="Run the image data variant of the game (default: False)"
    )
    parser.add_argument(
        "--optim", type=str, default="adamw",
        help="Optimizer to use {adam, adamw, rmsprop} (default: adamw)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0,
        help="Weight decay coefficient (default: 0)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
        help="Number of initial training steps, during which length cost is "
        "not applied (default: 0)",
    )
    parser.add_argument("--loss", type=str, default='weighted_attributes')
    parser.add_argument(
        "--receiver_reset_freq", default=None,
        type=lambda x: None if x == 'None' else int(x),
        help="Number of epochs between receiver parameter resets "
        "(default: None)"
    )
    parser.add_argument(
        "--no_shuffle", action="store_true", default=False,
        help="Do not shuffle train data before every epoch (default: False)",
    )
    parser.add_argument(
        "--double_precision", action="store_true", default=False,
        help="Use double predicion floating point numbers (default: False)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.,
        help="GS temperature (default: 1.0)",
    )
    parser.add_argument(
        "--temperature_start", type=float, default=None,
        help="Initial GS temperature (default: None)"
    )
    parser.add_argument(
        "--temperature_end", type=float, default=None,
        help="Final GS temperature (default: None)"
    )

    # W&B
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="WandB entity name (default: None)",
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="WandB project name (default: None)",
    )
    parser.add_argument(
        "--wandb_run_id", type=str, default=None,
        help="WandB run id (default: None)",
    )
    parser.add_argument(
        "--wandb_group", type=str, default=None,
        help="WandB group name (default: None)",
    )

    args = core.init(parser, params)
    check_args(args)
    print(args)
    return args


def check_args(args):
    args.channel = args.channel.lower() if args.channel else args.channel

    if args.channel is not None \
            and args.channel.lower() not in ('erasure', 'deletion'):
        raise ValueError(f"Unknown channel type: {args.channel}")

    if args.channel is None or args.error_prob == 0:
        args.error_prob = 0.0
        args.channel = None

    if args.results_folder is not None:
        os.makedirs(os.path.dirname(args.results_folder), exist_ok=True)

    args.results_folder = pathlib.Path(args.results_folder) \
        if args.results_folder is not None else None


def main(params):
    opts = get_params(params)

    if opts.double_precision:
        torch.set_default_dtype(torch.float64)

    data_handler = DataHandler(opts)
    train_data, eval_train_data, eval_test_data = data_handler.load_data(opts)

    game = SenderReceiverRnnGS(opts)

    optimizer = build_optimizer(game, opts)
    callbacks = [
        TrainingEvaluationCallback(opts, game.channel),
        CustomProgressBarLogger(
            opts,
            train_data_len=len(train_data),
            test_data_len=len(eval_test_data),
        ),
        ReceiverResetCallback(game, opts),
    ]

    if not (opts.temperature_start is None and opts.temperature_end is None):
        callbacks.append(TemperatureAnnealer(game, opts))

    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=None,
        train_data=train_data,
        validation_data=eval_test_data,
        callbacks=callbacks,
    )

    t_start = time.monotonic()
    trainer.train(n_epochs=opts.n_epochs)
    t_end = time.monotonic()
    dump_dict = {}

    # results on the eval subset of the training set
    train_dump = Dump(
        game, eval_train_data, data_handler.eval_train_sample_types, opts
    )
    train_mapping = data_handler.eval_train_mapping
    dump_dict['train'] = {
        'evaluation': train_dump.get_eval_dict(),
        'messages': train_dump.get_message_logs(train_mapping),
    }

    test_dump = Dump(
        game, eval_test_data, data_handler.eval_test_sample_types, opts
    )
    test_mapping = data_handler.eval_train_mapping
    dump_dict['test'] = {
        'evaluation': test_dump.get_eval_dict(),
        'messages': test_dump.get_message_logs(test_mapping),
    }

    for key in dump_dict['train']['evaluation']:
        logits_p, logits_q = (train_dump.logits_nn, test_dump.logits_nn) \
            if key == 'sent' else (train_dump.logits, test_dump.logits)

        kld_train = relative_message_entropy(logits_p, logits_q).item()
        kld_test = relative_message_entropy(logits_q, logits_p).item()
        dump_dict['train']['evaluation'][key]['kld_train_test'] = kld_train
        dump_dict['test']['evaluation'][key]['kld_train_test'] = kld_train
        dump_dict['train']['evaluation'][key]['kld_test_train'] = kld_test
        dump_dict['test']['evaluation'][key]['kld_test_train'] = kld_test

    # save training time
    training_time = timedelta(seconds=t_end - t_start)
    evaluation_time = timedelta(seconds=time.monotonic() - t_end)

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
        filepath = opts.results_folder / f'{opts.filename}-results.json'
        with open(filepath, 'w') as f:
            json.dump(dump_dict, f, indent=4)
        print(f"Results saved to {filepath}")

    print('Training time:', training_time)
    print('Training time per epoch:', training_time_per_epoch)
    print('Evaluation time:', evaluation_time)

    print_training_results(dump_dict)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
