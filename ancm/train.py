# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import json
import time
import pathlib
import argparse
import operator
from collections import defaultdict
from datetime import timedelta

import torch.utils.data

import egg.core as core
# from egg.core.util import move_to

from ancm.trainers import Trainer
from ancm.util import (
    DataHandler,
    build_optimizer,
    dump_sender_receiver,
    get_results_dict,
    print_training_results,
    is_jsonable,
)
from ancm.archs import (
    Sender, Receiver,
    loss_rf, loss_gs,
    SenderReceiverRnnReinforce,
    SenderReceiverRnnGS,
)
from ancm.callbacks import (
    CustomProgressBarLogger,
    TrainingMetricsCallback,
)


def get_params(params):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=None, help="Path to .npz data file to load")
    parser.add_argument("--channel", type=str, default=None, help="Communication channel type {erasure, symmetric, deletion} (default: None)")
    parser.add_argument("--error_prob", type=float, default=0., help="Probability of error per symbol (default: 0.0)")
    parser.add_argument("--no_shuffle", action="store_false", default=True, help="Do not shuffle train data before every epoch (default: False)")
    parser.add_argument("--sender_hidden", type=int, default=50, help="Size of the hidden layer of Sender (default: 50)")
    parser.add_argument("--receiver_hidden", type=int, default=50, help="Size of the hidden layer of Receiver (default: 50)")
    parser.add_argument("--embedding", type=int, default=10, help="Dimensionality of the embedding hidden layer for the agents (default: 10)")
    parser.add_argument("--sender_cell", type=str, default="rnn", help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)")
    parser.add_argument("--receiver_cell", type=str, default="rnn", help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)")
    parser.add_argument("--sender_lr", type=float, default=1e-1, help="Learning rate for Sender's parameters (default: 1e-1)")
    parser.add_argument("--receiver_lr", type=float, default=1e-1, help="Learning rate for Receiver's parameters (default: 1e-1)")
    parser.add_argument("--sender_entropy_coeff", type=float, default=0.01)
    parser.add_argument("--receiver_entropy_coeff", type=float, default=0.001)
    parser.add_argument("--length_cost", type=float, default=1e-2, help="Message length cost")
    parser.add_argument("--mode", type=str, default="gs", help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {gs only at the moment} (default: rf)")
    parser.add_argument("--optim", type=str, default="rmsprop", help="Optimizer to use [adam, rmsprop] (default: rmsprop)")
    parser.add_argument("--temperature", type=float, default=1.0, help="GS temperature for the sender (default: 1.0)")
    parser.add_argument("--trainable_temperature", action="store_true", default=False, help="Enable trainable temperature")
    parser.add_argument("--temperature_decay", default=0.9, type=float)
    parser.add_argument("--temperature_minimum", default=0.5, type=float)
    parser.add_argument("--results_folder", type=str, default='runs', help="Folder where file with dumped messages will be created")
    parser.add_argument("--filename", type=str, default=None, help="Filename (no extension)")
    parser.add_argument("--debug", action="store_true", default=False, help="Run egg/objects_game with pdb enabled")
    parser.add_argument("--image_input", action="store_true", default=False, help="Run image data variant of the game")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run id")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB project name")

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

    if args.results_folder is not None:
        os.makedirs(os.path.dirname(args.results_folder), exist_ok=True)

    args.results_folder = pathlib.Path(args.results_folder) \
        if args.results_folder is not None else None

    if args.debug:
        import pdb

        pdb.set_trace()


def main(params):
    opts = get_params(params)

    device = torch.device("cuda" if opts.cuda else "cpu")

    data_handler = DataHandler(opts)
    train_data, validation_data, test_data, aux_train_data = \
        data_handler.load_data(opts)

    if opts.channel == 'erasure' and opts.error_prob != 0:
        receiver_vocab_size = opts.vocab_size + 1
    else:
        receiver_vocab_size = opts.vocab_size

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
            vocab_size=receiver_vocab_size,
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
        sender = core.RnnSenderGS(
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
            receiver_vocab_size,
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
        TrainingMetricsCallback(
            vocab_size=opts.vocab_size,
            max_len=opts.max_len,
            channel_type=opts.channel,
            error_prob=opts.error_prob,
            sender=_sender,
            receiver=_receiver,
            dataloader=validation_data,
            device=device,
            image_input=opts.image_input,
            bs=opts.batch_size),
        CustomProgressBarLogger(
            opts,
            train_data_len=len(train_data),
            test_data_len=len(validation_data)),
    ]

    if opts.mode == "gs" and not opts.trainable_temperature:
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
    second_val = (opts.channel is not None and opts.error_prob > 0.)
    trainer.train(n_epochs=opts.n_epochs, second_val=second_val)
    t_end = time.monotonic()

    def evaluate(dataloader):
        results, messages = defaultdict(dict), []
        message_counts = defaultdict(lambda: defaultdict(int))

        apply_noise = opts.error_prob > 0. and opts.channel is not None
        receiver = game.receiver

        if opts.channel == 'erasure' and opts.error_prob != 0:
            receiver_vocab_size = opts.vocab_size + 1
        else:
            receiver_vocab_size = opts.vocab_size

        dump = dump_sender_receiver(
            game, dataloader, apply_noise=apply_noise, max_len=opts.max_len,
            vocab_size=receiver_vocab_size, device=device)

        # Unique targets
        unique_dict = defaultdict(int)
        if opts.image_input:
            for i in range(len(dump)):
                target_attrs = [
                    str(int(attr_values[i]))
                    for attr_values in dump.target_attributes.values()]
                target_repr = ','.join(target_attrs)
                unique_dict[target_repr] += 1
        else:
            for s_inp in dump.sender_inputs:
                target = ','.join([
                    str(int(x)) for x in s_inp.nonzero().squeeze().tolist()])
                unique_dict[target] += 1

        # Evaluation in the same setting as during training
        output_key = 'noise' if apply_noise else 'no noise'
        results[output_key] = get_results_dict(
            dump, receiver, opts, unique_dict,
            noise=apply_noise)

        for s_inp, msg, r_inp, r_out, label, t_attr, d_attr in dump:
            if opts.image_input:
                # For the Obverter dataset, we save object features rather than
                # images (color, shape, position, rotation)
                target_vec = ','.join([str(attr) for attr in t_attr.values()])
                candidate_vex = [
                    ','.join([str(attr) for attr in attr_dict.values()])
                    for attr_dict in d_attr]
                message = ','.join([str(x) for x in msg.tolist()])
                message_log = {
                    'target_obj': target_vec,
                    'candidate_objs': candidate_vex,
                    'message': message,
                    'message-no-noise': None,
                    'label': label}

            else:
                # VISA concepts are sparse binary tensors, hence we represent each
                # object as a set of features that it does have
                target_vec = ','.join([
                    str(x) for x in s_inp.nonzero().squeeze().tolist()])
                candidate_vex = [
                    ','.join([
                        str(x) for x in candidate.nonzero().squeeze().tolist()])
                    for candidate in r_inp]
                message = ','.join([str(x) for x in msg.tolist()])
                message_log = {
                    'target_obj': target_vec,
                    'candidate_objs': candidate_vex,
                    'message': message,
                    'message-no-noise': None,
                    'label': label,
                    'target_attributes': t_attr,
                    'distractor_attributes': d_attr}

            messages.append(message_log)
            message_counts[output_key][message] += 1

        # If we applied noise during training, disable it and evaluate again
        if apply_noise:
            dump = dump_sender_receiver(
                game, dataloader, apply_noise=False,
                max_len=opts.max_len,
                vocab_size=opts.vocab_size,
                device=device)

            results['no_noise'] = get_results_dict(dump, receiver, opts, unique_dict, False)

            # Iterating through Dump without noise
            for i, (s_inp, msg, r_inp, _, _, _, _) in enumerate(dump):
                if opts.image_input:  # Obverter
                    target_vec = ','.join([attr for attr in t_attr.values()])
                    candidate_vex = [
                        ','.join([attr for attr in attr_dict.values()])
                        for attr_dict in d_attr]
                    message = ','.join([str(x) for x in msg.tolist()])

                else:  # VISA
                    target_vec = ','.join([
                        str(x) for x in s_inp.nonzero().squeeze().tolist()])
                    candidate_vex = [
                        ','.join([
                            str(x) for x in candidate.nonzero().squeeze().tolist()])
                        for candidate in r_inp]
                    message = ','.join([str(x) for x in msg.tolist()])

                message_log = messages[i]
                assert message_log['target_obj'] == target_vec
                assert message_log['candidate_objs'] == candidate_vex

                message_log['message-no-noise'] = message
                message_counts['no_noise'][message] += 1

        for key in message_counts:
            message_counts[key] = sorted(
                message_counts[key].items(),
                key=operator.itemgetter(1),
                reverse=True)

        return results, messages, message_counts

    output_dict = {}

    # get results on the train and test test
    if aux_train_data is not None:
        results, messages, message_counts = evaluate(aux_train_data)
        output_dict['train'] = {
            'results': results,
            'messages': messages,
            'message_counts': message_counts}
    results, messages, message_counts = evaluate(test_data)
    output_dict['test'] = {
        'results': results,
        'messages': messages,
        'message_counts': message_counts}

    training_time = timedelta(seconds=t_end - t_start)
    evaluation_time = timedelta(seconds=time.monotonic() - t_start)

    sec_per_epoch = training_time.seconds / opts.n_epochs
    minutes, seconds = divmod(sec_per_epoch, 60)

    training_time = str(training_time).split('.', maxsplit=1)[0]
    evaluation_time = str(evaluation_time).split('.', maxsplit=1)[0]
    training_time_per_epoch = f'{int(minutes):02}:{int(seconds):02}'

    print_training_results(output_dict)

    opts_dict = {k: v for k, v in vars(opts).items() if is_jsonable(v) and k != 'optimizer'}
    output_dict['opts'] = opts_dict
    output_dict['training_time'] = {
        'training': training_time,
        'evaluation': evaluation_time,
        'training_per_epoch': training_time_per_epoch}

    if opts.results_folder:
        opts.results_folder.mkdir(exist_ok=True)
        with open(opts.results_folder / f'{opts.filename}-results.json', 'w') as f:
            json.dump(output_dict, f, indent=4)

        print(f"Results saved to {opts.results_folder / opts.filename}-results.json")

    print('Training time:', training_time)
    print('Training time per epoch:', training_time_per_epoch)
    print('Evaluation time:', evaluation_time)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
