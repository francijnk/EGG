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
    RnnSenderGS,
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
    train_data, validation_data, test_data, aux_train_data = \
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
        TrainingMetricsCallback(
            vocab_size=opts.vocab_size,
            max_len=opts.max_len,
            channel=game.channel,
            channel_type=opts.channel,  # TODO
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

    if opts.mode == "gs" and not opts.trainable_temperature \
            and opts.temperature_decay is not None:
        callbacks.append(core.TemperatureUpdater(
            agent=sender,
            decay=opts.temperature_decay,
            minimum=opts.temperature_minimum))

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=None,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks)

    t_start = time.monotonic()
    trainer.train(n_epochs=opts.n_epochs)
    t_end = time.monotonic()

    def evaluate(dataloader):
        results, messages = {}, []
        message_counts = defaultdict(int)

        receiver = game.receiver

        dump = dump_sender_receiver(
            game, dataloader, max_len=opts.max_len,
            vocab_size=vocab_size, mode=opts.mode, device=device)

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

        results = get_results_dict(dump, receiver, opts, unique_dict)

        for s_inp, msg, msg_nn, r_inp, r_out, r_out_nn, \
                label, t_attr, d_attr, ch_out in dump:

            if opts.image_input:
                # For the Obverter dataset, we save object features rather than
                # images (color, shape, position, rotation)
                target_vec = ','.join([str(int(attr)) for attr in t_attr.values()])
                candidate_vex = [
                    ','.join([str(int(attr)) for attr in attr_dict.values()])
                    for attr_dict in d_attr]
                message = ','.join([str(int(x)) for x in msg.tolist()])
                message_nn = ','.join([str(int(x)) for x in msg_nn.tolist()])
                #print(r_out, "receiver_pred")
                message_log = {
                    'target_obj': target_vec,
                    'candidate_objs': candidate_vex,
                    'label': label,
                    'message': message,
                    'prediction': r_out,
                    'message_no_noise': message_nn,
                    'prediction_no_noise': r_out_nn,
                }

            else:
                # VISA concepts are sparse binary tensors, hence we represent each
                # object as a set of features that it does have
                target_vec = ','.join([
                    str(x) for x in s_inp.nonzero().squeeze().tolist()])
                candidate_vex = [
                    ','.join([
                        str(x) for x in candidate.nonzero().squeeze().tolist()])
                    for candidate in r_inp]
                message = ','.join([str(int(x)) for x in msg.tolist()])
                message_nn = ','.join([str(int(x)) for x in msg_nn.tolist()])
                message_log = {
                    'target_obj': target_vec,
                    'candidate_objs': candidate_vex,
                    'label': label,
                    'message': message,
                    'prediction': r_out,
                    'message_no_noise': message_nn,
                    'prediction_no_noise': r_out_nn,
                    'target_attributes': t_attr,
                    'distractor_attributes': d_attr,
                }

            message_log['entropy'] = ch_out['entropy_msg']
            message_log['entropy_no_noise'] = ch_out['entropy_msg_nn']
            message_log['redundancy'] = ch_out['redundancy_msg']

            messages.append(message_log)
            message_counts[message] += 1

        message_counts = sorted(
            message_counts.items(),
            key=operator.itemgetter(1),
            reverse=True)

        return {
            'results': results,
            'messages': messages,
            'message_counts': message_counts,
        }

    output_dict = {}

    # results on the eval subset of the training set (VISA)
    if aux_train_data is not None:
        game.train()
        output_dict['train'] = evaluate(aux_train_data)

    # results on the test set
    game.eval()
    output_dict['test'] = evaluate(test_data)

    # save training time
    training_time = timedelta(seconds=t_end - t_start)
    evaluation_time = timedelta(seconds=time.monotonic() - t_start)

    sec_per_epoch = training_time.seconds / opts.n_epochs
    minutes, seconds = divmod(sec_per_epoch, 60)

    training_time = str(training_time).split('.', maxsplit=1)[0]
    evaluation_time = str(evaluation_time).split('.', maxsplit=1)[0]
    training_time_per_epoch = f'{int(minutes):02}:{int(seconds):02}'

    def make_jsonable(x, key=None):
        if isinstance(x, torch.Tensor):
            print(key, 'processing tensor')
            try:
                return x.item()
            except:
                if x.numel() < 100:
                    return x.tolist()
                else:
                    return 'none'
        if isinstance(x, dict):
            return {make_jsonable(k): make_jsonable(v, k) for k, v in x.items()}
        if isinstance(x, list) or isinstance(x, tuple):
            return [make_jsonable(item) for item in x]

        return x

    opts_dict = {k: make_jsonable(v.item()) if isinstance(v, torch.Tensor)
                 else v for k, v in vars(opts).items() if is_jsonable(v) and k != 'optimizer'}
    # opts_dict = {k: v for k, v in vars(opts) if is_jsonable(v) and k != 'optimizer'}
    output_dict['opts'] = opts_dict
    output_dict['training_time'] = {
        'training': training_time,
        'evaluation': evaluation_time,
        'training_per_epoch': training_time_per_epoch,
    }

    if opts.results_folder:
        opts.results_folder.mkdir(exist_ok=True)
        with open(opts.results_folder / f'{opts.filename}-results.json', 'w') as f:
            json.dump(make_jsonable(output_dict), f, indent=4)

        print(f"Results saved to {opts.results_folder / opts.filename}-results.json")

    print('Training time:', training_time)
    print('Training time per epoch:', training_time_per_epoch)
    print('Evaluation time:', evaluation_time)

    print_training_results(output_dict)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
