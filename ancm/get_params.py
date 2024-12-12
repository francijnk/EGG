import os
import math
import random
import subprocess
import shutil
import argparse
from tqdm import tqdm
from collections import defaultdict
from uuid import uuid4

import time
from datetime import timedelta

from ancm.train import main
from egg.core import init


random_seeds = [i+1 for i in range(10)]
data_seeds = [42]
error_probs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
channels = 'erasure deletion symmetric'.split()
max_lengths = [2, 3, 5, 10]

slr = 5e-3
rlr = 1e-3
length_cost = 1e-3
vocab_size = 10
hidden_units = 50
n_epochs = 10


def get_opts(error_prob, channel, max_len, random_seed, data_seed):
    if channel:
        _channel = f'{channel}_{error_prob:.2f}'
    else:
        _channel = f'{baseline}'
    filename = f'{_channel}_{max_len}_{random_seed}'
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
        # f'--dump_results_folder {results_dir}',
        f'--n_epochs {n_epochs}',
        '--perceptual_dimensions [2]*8',
        '--n_distractors 4',
        '--sender_embedding 10',
        '--receiver_embedding 10',
        '--sender_entropy_coeff 0.01',
        '--receiver_entropy_coeff 0.001',
        '--sender_cell lstm',
        '--receiver_cell lstm',
        '--mode rf',
        '--evaluate',
        '--validation_freq 1',
    ]
    if channel is not None:
        opts.append(f'--channel {channel}')
    return opts


if __name__ == '__main__':

    all_params = []
    for max_len in max_lengths:
        for rs in random_seeds:
            for ds in data_seeds:
                params = get_opts(0., None, max_len, rs, ds)
                all_params.append(' '.join(params))

    for max_len in max_lengths:
        for channel in channels:
            for pr in error_probs[1:]:
                for rs in random_seeds:
                    for ds in data_seeds:
                        params = get_opts(pr, channel, max_len, rs, ds)
                        all_params.append(' '.join(params))


    params_file = 'ancm/jobs/params.txt'
    with open(params_file, 'w') as fp:
        fp.write('\n'.join(all_params))

    print(len(all_params), 'lines saved to', params_file)
