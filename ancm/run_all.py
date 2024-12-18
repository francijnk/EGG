import os
import math
import random
import subprocess
import shutil
import argparse
from tqdm import tqdm
from collections import defaultdict
from uuid import uuid4
from distutils.dir_util import copy_tree
from multiprocessing import Process, get_context

import time
from datetime import timedelta

from ancm.train import main
from egg.core import init


random_seeds = [i+1 for i in range(5)]
data_seeds = [42]
error_probs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
channels = 'erasure deletion symmetric'.split()
max_lengths = [2,3]

slr = 5e-3
rlr = 1e-3
length_cost = 1e-3
vocab_size = 10
hidden_units = 50
n_epochs = 10

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--max_len', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=4)
args = parser.parse_args()

max_lengths = [args.max_len] if args.max_len else max_lengths


def get_opts(error_prob, channel, max_len, random_seed, data_seed, filename, results_dir):
    opts = [
        f'--error_prob {error_prob}',
        f'--max_len {max_len}',
        f'--vocab_size {vocab_size}',
        f'--sender_lr {slr}',
        f'--receiver_lr {rlr}',
        f'--length_cost {length_cost}',
        f'--sender_hidden {hidden_units}',
        f'--receiver_hidden {hidden_units}',
        f'--random_seed {random_seed}',
        f'--data_seed {data_seed}',
        f'--filename {filename}',
        f'--dump_results_folder {results_dir}',
        f'--n_epochs {n_epochs}',
        '--perceptual_dimensions [2]*8',
        #'--n_distractors 4',
        '--sender_embedding 10',
        '--receiver_embedding 10',
        '--sender_entropy_coeff 0.01',
        '--receiver_entropy_coeff 0.001',
        '--sender_cell lstm',
        '--receiver_cell lstm',
        '--mode rf',
        '--evaluate',
        '--validation_freq 1'
    ]
    if channel is not None:
        opts.append(f'--channel {channel}')
    return opts

def task(channel, error_prob, max_len, rs, ds):
    if channel:
        results_dir = f'channel_{channel}/error_prob_{error_prob:.2f}'
    else:
        results_dir = 'baseline'
    output_dir = os.path.join(args.output_dir, results_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{max_len}_{rs}'
    opts = get_opts(0.0, channel, max_len, rs, ds, filename, output_dir)
    process = subprocess.Popen(
        ['python3', '-m' 'ancm.train']
        + [o for opt in opts for o in opt.split()])
    exitcode = process.wait()

if __name__ == '__main__':

    num_runs = len(max_lengths) * (
        len(random_seeds) * len(data_seeds)
        + (len(channels) * len(error_probs[1:])
           * len(random_seeds) * len(data_seeds)))
    t_start = time.monotonic()

    processes = defaultdict(list)
    count = 1
    for max_len in max_lengths:
        for rs in random_seeds:
            for ds in data_seeds:
                print('---', f'{count}/{num_runs}', '---')
                count += 1
                task(None, 0.0, max_len, rs, ds)
                # ctx = get_context('spawn')
                # processes[max_len].append(
                #     ctx.Process(
                #         target=task,
                #         args=(None, 0.0, max_len, rs, ds)))

    for max_len in max_lengths:
        for channel in channels:
            for pr in error_probs[1:]:
                for rs in random_seeds:
                    for ds in data_seeds:
                        task(channel, pr, max_len, rs, ds)
                        # ctx = get_context('spawn')
                        # processes[max_len].append(
                        #    ctx.Process(
                        #        target=task,
                        #        args=(channel, pr, max_len, rs, ds)))

    #for max_len in processes:
    #    random.shuffle(processes[max_len])

    all_processes = [p for max_len in processes for p in processes[max_len]]
    num_processes = len(all_processes)
    print('Running', num_processes, 'jobs')

    #num_batches = 0
    #for max_len in processes:
    #    n_batches = math.ceil(len(processes[max_len]) / args.batch_size)
    #    num_batches += n_batches
    #batch_count = 1

    #for max_len in processes:
    #    print('=== max len:', max_len, '===')
    #    t_start_max_len = time.monotonic()
    #    for j, batch_start in enumerate(range(0, len(processes[max_len]), args.batch_size)):
    #        batch = processes[max_len][batch_start:batch_start+args.batch_size]
    #        for process in batch:
    #            process.start()
    #        for process in batch:
    #            process.join()
    #        elapsed = timedelta(seconds=time.monotonic()-t_start)
    #        elapsed_max_len = timedelta(seconds=time.monotonic()-t_start_max_len)
    #        elapsed_per_batch = elapsed_max_len.seconds / (j+1)
    #        minutes, seconds = divmod(elapsed_per_batch, 60)
    #        elapsed = str(elapsed).split('.', maxsplit=1)[0]
    #        elapsed_per_batch = f'{int(minutes):02}:{int(seconds):02}'
    #        print(f'batch {batch_count}/{num_batches} completed! Elapsed time: {elapsed} ({elapsed_per_batch} per batch)')
    #        batch_count += 1

    training_time = timedelta(seconds=time.monotonic()-t_start)
    sec_per_run = training_time.seconds / num_runs
    minutes, seconds = divmod(sec_per_run, 60)

    time_total = str(training_time).split('.', maxsplit=1)[0]
    time_per_run = f'{int(minutes):02}:{int(seconds):02}'

    with open(os.path.join(args.output_dir, f'training_time_{uuid4()}.txt'), 'w') as fp:
        fp.write('max lengths: ' + ', '.join([str(x) for x in max_lengths]) + '\n')
        fp.write(time_total + '\n')
        fp.write(time_per_run)

    baseline_dir = os.path.join(args.output_dir, 'baseline')
    for channel in channels:
        channel_baseline_dir = os.path.join(args.output_dir, f'channel_{channel}', 'error_prob_0.0')
        copy_tree(baseline_dir, channel_baseline_dir)
    shutil.rmtree(baseline_dir)
