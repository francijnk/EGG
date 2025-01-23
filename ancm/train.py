# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import argparse
import operator
import pathlib
import json
import time
from datetime import timedelta
from collections import defaultdict

import torch.utils.data

import egg.core as core
from egg.core.util import move_to

from ancm.trainers import Trainer
from ancm.util import (
    dump_sender_receiver,
    truncate_messages,
    is_jsonable,
    CustomDataset,
    DataHandler,
    build_optimizer,
)
from ancm.metrics import (
    compute_mi_input_msgs,
    # compute_conceptual_alignment,
    compute_max_rep,
    compute_redundancy_msg,
    compute_redundancy_smb,
    compute_redundancy_smb_adjusted,
    compute_top_sim,
    # compute_posdis,
    # compute_bosdis,
)
from ancm.archs import (
    SenderReinforce, ReceiverReinforce,
    loss,
    SenderReceiverRnnReinforce,
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
    parser.add_argument("--sender_embedding", type=int, default=10, help="Dimensionality of the embedding hidden layer for Sender (default: 10)")
    parser.add_argument("--receiver_embedding", type=int, default=10, help="Dimensionality of the embedding hidden layer for Receiver (default: 10)")
    parser.add_argument("--sender_cell", type=str, default="rnn", help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)")
    parser.add_argument("--receiver_cell", type=str, default="rnn", help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)")
    parser.add_argument("--sender_lr", type=float, default=1e-1, help="Learning rate for Sender's parameters (default: 1e-1)")
    parser.add_argument("--receiver_lr", type=float, default=1e-1, help="Learning rate for Receiver's parameters (default: 1e-1)")
    parser.add_argument("--sender_entropy_coeff", type=float, default=0.01)
    parser.add_argument("--receiver_entropy_coeff", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=None, help="LR decay, 1.0 for no decay (default: no decay)")
    parser.add_argument("--length_cost", type=float, default=1e-2, help="Message length cost")
    parser.add_argument("--evaluate", action="store_true", default=False, help="Evaluate trained model on test data")
    parser.add_argument("--results_folder", type=str, default='runs', help="Folder where file with dumped messages will be created")
    parser.add_argument("--filename", type=str, default=None, help="Filename (no extension)")
    parser.add_argument("--debug", action="store_true", default=False, help="Run egg/objects_game with pdb enabled")
    parser.add_argument("--images", action="store_true", default=False, help="Run image data variant of the game")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run id")

    args = core.init(parser, params)
    check_args(args)
    print(args)
    return args


def check_args(args):
    if args.results_folder is not None:
        os.makedirs(os.path.dirname(args.results_folder), exist_ok=True)

    if args.debug:
        import pdb

        pdb.set_trace()

    args.channel = args.channel.lower() if args.channel else args.channel
    assert (
        args.channel is None
        or args.channel in ("erasure", "symmetric", "deletion", "truncation")
    ), 'The only channels implemented are "erasure", "symmetric", "deletion" and "truncation"'

    args.results_folder = (
        pathlib.Path(args.results_folder) if args.results_folder is not None else None
    )

    if (not args.evaluate) and args.results_folder:
        print(
            "| WARNING --results_folder was set without --evaluate. Evaluation will not be performed nor any results will be dumped. Please set --evaluate"
        )


def main(params):
    opts = get_params(params)

    # device = torch.device("cuda" if opts.cuda else "cpu")
    #device = torch.device("cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed manually to make runs reproducible
    # You need to set this again if you do multiple runs of the same model
    torch.manual_seed(42)

    # When running on the CuDNN backend two further options must be set for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_handler = DataHandler(opts)
    train_data, validation_data, test_data = data_handler.load_data(opts)

    if opts.channel == 'erasure' and opts.error_prob != 0:
        receiver_vocab_size = opts.vocab_size + 1
    else:
        receiver_vocab_size = opts.vocab_size

    _sender = SenderReinforce(
        n_features=data_handler.n_features,
        n_hidden=opts.sender_hidden,
        image = opts.images)
    _sender = _sender.to(device)
    _receiver = ReceiverReinforce(
        n_features=data_handler.n_features,
        linear_units=opts.receiver_hidden,
        image=opts.images)
    _receiver = _receiver.to(device)
    sender = core.RnnSenderReinforce(
        _sender,
        opts.vocab_size,
        opts.sender_embedding,
        opts.sender_hidden,
        opts.max_len,
        cell=opts.sender_cell)
    sender = sender.to(device)
    receiver = core.RnnReceiverReinforce(
        agent=core.ReinforceWrapper(_receiver),
        vocab_size=receiver_vocab_size,
        embed_dim=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        cell=opts.receiver_cell)
    receiver = receiver.to(device)
    game = SenderReceiverRnnReinforce(
        sender, receiver,
        loss=loss,
        vocab_size=opts.vocab_size,
        channel_type=opts.channel,
        error_prob=opts.error_prob,
        length_cost=opts.length_cost,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=opts.receiver_entropy_coeff,
        device=device,
        seed=opts.random_seed)
    game = game.to(device)
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
            bs=opts.batch_size),
    ]

    callbacks.append(CustomProgressBarLogger(
        opts,
        train_data_len=len(train_data),
        test_data_len=len(validation_data),
    ))
    trainer = Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=None,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks)

    t_start = time.monotonic()
    if opts.error_prob == 0. or not opts.channel:
        trainer.train(n_epochs=opts.n_epochs, second_val=False)
    else:
        trainer.train(n_epochs=opts.n_epochs, second_val=True)
    training_time = timedelta(seconds=time.monotonic()-t_start)
    sec_per_epoch = training_time.seconds / opts.n_epochs
    minutes, seconds = divmod(sec_per_epoch, 60)

    time_total = str(training_time).split('.', maxsplit=1)[0]
    time_per_epoch = f'{int(minutes):02}:{int(seconds):02}'

    if opts.evaluate:
        output_dict = defaultdict(dict)

        # Standard evaluation – same setting as during training
        apply_noise = opts.error_prob != 0. and opts.channel is not None
        sender_inputs, messages, receiver_inputs, receiver_outputs, labels = \
            dump_sender_receiver(
                game, test_data, apply_noise=apply_noise,
                variable_length=True, max_len=opts.max_len, vocab_size=receiver_vocab_size,
                device=device)

        padded_messages = torch.nn.utils.rnn.pad_sequence(messages, batch_first=True)

        # to get new additional accuracy for truncated messages (where one symbol is removed)
        new_messages, new_receiver_inputs, new_labels = truncate_messages(
            padded_messages, receiver_inputs, labels)

        dataset = CustomDataset(new_messages, new_receiver_inputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)

        predictions = []
        for b_messages, b_inputs in dataloader:
            b_messages = b_messages.to(device)
            b_inputs = b_inputs.to(device)
            outputs = receiver(b_messages, b_inputs)
            predictions.extend(outputs[0])

        predictions = torch.Tensor(predictions)
        predictions = predictions.to(device)
        new_labels = torch.Tensor((new_labels))
        new_labels = new_labels.to(device)
        compared = torch.eq(predictions, new_labels)
        accuracy2 = compared.float().mean().item()

        actual_vocab = set(int(s) for m in messages for s in m.tolist())
        actual_vocab_size = len(actual_vocab)

        receiver_outputs = move_to(receiver_outputs, device)
        receiver_outputs = torch.stack(receiver_outputs)
        labels = move_to(labels, device)
        labels = torch.stack(labels)

        accuracy = torch.mean((receiver_outputs == labels).float()).item()
        # alignment = compute_conceptual_alignment(
        #     test_data, _receiver, _sender, device, opts.batch_size)
        redund_msg = compute_redundancy_msg(messages)
        redund_smb = compute_redundancy_smb(
            messages, opts.max_len, opts.vocab_size, opts.channel, opts.error_prob)
        redund_smb_adj = compute_redundancy_smb_adjusted(
            messages, opts.channel, opts.error_prob, alphabet=actual_vocab, erased_symbol=opts.vocab_size)
        topographic_rho = compute_top_sim(sender_inputs, messages)
        # posdis = compute_posdis(sender_inputs, messages)
        # bosdis = compute_bosdis(sender_inputs, messages, opts.vocab_size)
        maxrep = compute_max_rep(messages).mean().item()

        output_dict['results']['accuracy'] = accuracy
        output_dict['results']['accuracy2'] = accuracy2
        # output_dict['results']['embedding_alignment'] = alignment
        output_dict['results']['redundancy_msg'] = redund_msg
        output_dict['results']['redundancy_smb'] = redund_smb
        output_dict['results']['redundancy_smb_adj'] = redund_smb_adj
        output_dict['results']['topographic_rho'] = topographic_rho
        # output_dict['results']['pos_dis'] = posdis
        # output_dict['results']['bos_dis'] = bosdis
        output_dict['results']['max_rep'] = maxrep
        output_dict['results']['actual_vocab_size'] = actual_vocab_size

        unique_dict = {}
        for elem in sender_inputs:
            target = ""
            if elem.dim() == 2:
                elem = elem[0]
            for dim in elem:
                target += f"{str(int(dim.item()))}-"
            target = target[:-1]
            if target not in unique_dict:
                unique_dict[target] = True

        mi_result = compute_mi_input_msgs(sender_inputs, messages)
        output_dict['results'].update(mi_result)
        entropy_msg = f"{mi_result['entropy_msg']:.3f}"
        entropy_inp = f"{mi_result['entropy_inp']:.3f}"
        mi = f"{mi_result['mi_msg_inp']:.3f}"
        entropy_inp_dim = f"{[round(x, 3) for x in mi_result['entropy_inp_dim']]}"
        mi_dim = f'{[round(x, 3) for x in mi_result["mi_msg_inp_dim"]]}'
        t_rho = f'{topographic_rho:.3f}'
        # p_dis = f'{posdis:.3f}'
        # b_dis = f'{bosdis:.3f}'
        redund_msg = f'{redund_msg:.3f}'
        redund_smb = f'{redund_smb:.3f}'
        redund_smb_adj = f'{redund_smb_adj:.3f}'
        max_repetitions = f'{maxrep:.2f}'

        # If we applied noise during training,
        # compute results after disabling noise in the test phase as well
        if opts.error_prob != 0:
            sender_inputs_nn, messages_nn, receiver_inputs_nn, \
                receiver_outputs_nn, labels_nn = dump_sender_receiver(
                    game, test_data,
                    apply_noise=False,
                    variable_length=True, max_len=opts.max_len,
                    vocab_size=opts.vocab_size, device=device)

            padded_messages_nn = torch.nn.utils.rnn.pad_sequence(messages_nn, batch_first=True)

            # to get new additional accuracy for truncated messages (where one symbol is removed)
            new_messages_nn, new_receiver_inputs_nn, new_labels_nn = truncate_messages(
                padded_messages_nn, receiver_inputs_nn, labels_nn)

            predictions_nn = []
            for b_messages, b_inputs in dataloader:
                b_messages = b_messages.to(device)
                b_inputs = b_inputs.to(device)
                outputs = receiver(b_messages, b_inputs)
                predictions_nn.extend(outputs[0])

            predictions_nn = torch.Tensor(predictions_nn)
            new_labels_nn = torch.Tensor((new_labels_nn))
            compared_nn = torch.eq(predictions_nn, new_labels_nn)
            accuracy2_nn = compared_nn.float().mean().item()

            receiver_outputs_nn = move_to(receiver_outputs_nn, device)
            receiver_outputs_nn = torch.stack(receiver_outputs_nn)
            labels_nn = move_to(labels_nn, device)
            labels_nn = torch.stack(labels_nn)

            actual_vocab_nn = set(int(s) for m in messages_nn for s in m.tolist())
            actual_vocab_size_nn = len(actual_vocab_nn)

            accuracy_nn = torch.mean((receiver_outputs_nn == labels_nn).float()).item()
            redund_msg_nn = compute_redundancy_msg(messages_nn)
            redund_smb_nn = compute_redundancy_smb(
                messages_nn, opts.max_len, opts.vocab_size, None, 0.0)
            redund_smb_adj_nn = compute_redundancy_smb_adjusted(
                messages_nn, None, 0.0, actual_vocab_nn)
            top_sim_nn = compute_top_sim(sender_inputs_nn, messages_nn)
            # posdis_nn = compute_posdis(sender_inputs_nn, messages_nn)
            # bosdis_nn = compute_bosdis(sender_inputs_nn, messages_nn, opts.vocab_size)
            max_rep_nn = compute_max_rep(messages_nn).mean().item()

            output_dict['results-no-noise']['accuracy'] = accuracy_nn
            output_dict['results-no-noise']['accuracy2'] = accuracy2_nn
            # output_dict['results-no-noise']['embedding_alignment'] = alignment
            output_dict['results-no-noise']['redundancy_msg'] = redund_msg_nn
            output_dict['results-no-noise']['redundancy_smb'] = redund_smb_nn
            output_dict['results-no-noise']['redundancy_smb_adj'] = redund_smb_adj_nn
            output_dict['results-no-noise']['topographic_rho'] = top_sim_nn
            # output_dict['results-no-noise']['pos_dis'] = posdis_nn
            # output_dict['results-no-noise']['bos_dis'] = bosdis_nn
            output_dict['results-no-noise']['max_rep'] = max_rep_nn
            output_dict['results-no-noise']['actual_vocab_size'] = actual_vocab_size_nn

            acc_str = f'{accuracy:.2f} / {accuracy_nn:.2f}'
            acc2_str = f'{accuracy2:.2f} / {accuracy2_nn:.2f}'
            mi_result_nn = compute_mi_input_msgs(sender_inputs_nn, messages_nn)
            output_dict['results-no-noise'].update(mi_result_nn)
            entropy_msg += f" / {mi_result_nn['entropy_msg']:.3f}"
            entropy_inp += f" / {mi_result_nn['entropy_inp']:.3f}"
            mi += f" / {mi_result_nn['mi_msg_inp']:.3f}"
            mi_dim_nn = f"{[round(x, 3) for x in mi_result_nn['mi_msg_inp_dim']]}"
            t_rho += f" / {top_sim_nn:.3f}"
            # p_dis += f'/ {posdis_nn:.3f}'
            # b_dis += f'/ {bosdis_nn:.3f}'
            redund_msg += f' / {redund_msg_nn:.3f}'
            redund_smb += f' / {redund_smb_nn:.3f}'
            redund_smb_adj += f' / {redund_smb_adj_nn:.3f}'
            max_repetitions += f' / {max_rep_nn:.2f}'

            print("|")
            print("|\033[1m Results (with noise / without noise)\033[0m\n|")
        else:
            acc_str = f'{accuracy:.2f}'
            acc2_str = f'{accuracy2:.2f}'
            print("|\n|\033[1m Results\033[0m\n|")

        align = 40
        print("|" + "H(msg) =".rjust(align), entropy_msg)
        print("|" + "H(target objs) =".rjust(align), entropy_inp)
        print("|" + "I(target objs; msg) =".rjust(align), mi)
        print("|\n| Separately for each object vector dimension")
        if opts.error_prob != 0:
            print("|" + "H(target objs) =".rjust(align), entropy_inp_dim)
            print("|" + "I(target objs; msg) =".rjust(align), mi_dim, "(with noise)")
            print("|" + "I(target objs; msg) =".rjust(align), mi_dim_nn, "(no noise)")
        else:
            print("|" + "H(target objs) =".rjust(align), entropy_inp_dim)
            print("|" + "I(target objs; msg) =".rjust(align), mi_dim)
        print('|')
        print("|" + "Accuracy:".rjust(align), acc_str)
        print("|" + "Accuracy2:".rjust(align), acc2_str)
        print("|")
        # print("|" + "Embedding alignment:".rjust(align) + f" {alignment:.2f}")
        print("|" + "Redundancy (message level):".rjust(align), redund_msg)
        print("|" + "Redundancy (symbol level):".rjust(align), redund_smb)
        print("|" + "Redundancy (symbol level, adjusted):".rjust(align), redund_smb_adj)
        print("|" + "Max num. of symbol reps.:".rjust(align) + f" {max_repetitions}")
        print("|" + "Topographic rho:".rjust(align) + f" {t_rho}")
        # print("|" + "PosDis:".rjust(align) + f" {p_dis}")
        # print("|" + "BosDis:".rjust(align) + f" {b_dis}")

        if opts.results_folder:
            opts.results_folder.mkdir(exist_ok=True)

            messages_dict = {}

            msg_dict = defaultdict(int)
            for sender_input, message, receiver_input, receiver_output, label \
                    in zip(
                        sender_inputs, messages, receiver_inputs,
                        receiver_outputs, labels):
                target_vec = ','.join([str(int(x)) for x in sender_input.tolist()])
                message = ','.join([str(int(x)) for x in message.tolist()])
                candidate_vex = [','.join([str(int(x)) for x in candidate])
                                 for candidate in receiver_input.tolist()]
                message_log = {
                    'target_vec': target_vec,
                    'candidate_vex': candidate_vex,
                    'message': message}
                if opts.error_prob != 0:
                    message_log['message_no_noise'] = None
                message_log['label'] = label.item()

                m_key = f'{target_vec}#' + ';'.join(candidate_vex)
                messages_dict[m_key] = message_log
                msg_dict[message] += 1

            sorted_msgs = sorted(msg_dict.items(), key=operator.itemgetter(1), reverse=True)

            if opts.error_prob != 0.:
                msg_dict_nn = defaultdict(int)
                for sender_input, message, receiver_input, receiver_output, label \
                        in zip(
                            sender_inputs_nn, messages_nn, receiver_inputs_nn,
                            receiver_outputs_nn, labels_nn):
                    target_vec = ','.join([str(int(x)) for x in sender_input.tolist()])
                    candidate_vex = [','.join([str(int(c)) for c in candidate])
                                     for candidate in receiver_input.tolist()]
                    message = ','.join([str(int(x)) for x in message.tolist()])

                    m_key = f'{target_vec}#' + ';'.join(candidate_vex)
                    messages_dict[m_key]['message_no_noise'] = message
                    msg_dict_nn[message] += 1

                sorted_msgs_nn = sorted(msg_dict_nn.items(), key=operator.itemgetter(1), reverse=True)

            lexicon_size = str(len(msg_dict.keys())) if opts.error_prob == 0 \
                else f'{len(msg_dict.keys())} / {len(msg_dict_nn.keys())}'

            if opts.error_prob == 0 or opts.channel is None:
                print("|")
                print("|" + "Unique target objects:".rjust(align), len(unique_dict.keys()))
                print("|" + "Lexicon size:".rjust(align), lexicon_size)
                print("|" + "Vocab size:".rjust(align), f"{actual_vocab_size}/{opts.vocab_size}")
            else:
                print("|")
                print("|" + "Unique target objects:".rjust(align), len(unique_dict.keys()))
                print("|" + "Lexicon size:".rjust(align), lexicon_size)

                if receiver_vocab_size != opts.vocab_size:
                    print("|" + "Vocab size:".rjust(align), f"{actual_vocab_size}/{actual_vocab_size_nn} out of {receiver_vocab_size}/{opts.vocab_size}")
                else:
                    print("|" + "Vocab size:".rjust(align), f"{actual_vocab_size}/{opts.vocab_size}")

            output_dict['results']['unique_targets'] = len(unique_dict.keys())
            output_dict['results']['unique_msg'] = len(msg_dict.keys())
            if opts.error_prob != 0:
                output_dict['results']['unique_msg_no_noise'] = len(msg_dict_nn.keys())
            # output_dict['results']['embedding_alignment'] = alignment
            output_dict['messages'] = [v for v in messages_dict.values()]
            output_dict['message_counts'] = sorted_msgs
            if opts.error_prob != 0:
                output_dict['message_counts_no_noise'] = sorted_msgs_nn
                if opts.channel == 'erasure':
                    output_dict['erased_symbol'] = opts.vocab_size
            opts_dict = {k: v for k, v in vars(opts).items() if is_jsonable(v)}
            output_dict['opts'] = opts_dict
            output_dict['training_time'] = {
                'total': time_total,
                'per_epoch': time_per_epoch}

            with open(opts.results_folder / f'{opts.filename}-results.json', 'w') as f:
                json.dump(output_dict, f, indent=4)

            print(f"| Results saved to {opts.results_folder / opts.filename}-results.json")

    print('| Total training time:', time_total)
    print('| Training time per epoch:', time_per_epoch)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
