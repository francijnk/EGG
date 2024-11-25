from train import main
from egg.core import init
import time
from datetime import timedelta
import subprocess
import shutil
from distutils.dir_util import copy_tree

random_seeds = [i+1 for i in range(5)]
data_seeds = [42]
error_probs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
max_lengths = [2, 3, 5, 10]
channels = 'erasure deletion symmetric'.split()

slr = 5e-3
rlr = 1e-3
length_cost = 1e-3
vocab_size = 10
hidden_units = 50

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
        '--perceptual_dimensions [2]*8',
        '--sender_embedding 10',
        '--receiver_embedding 10',
        '--n_epochs 3',
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

num_runs = (len(max_lengths) * len(random_seeds) * len(data_seeds)
    + (len(channels) * len(max_lengths) * len(error_probs[1:])
        * len(random_seeds) * len(data_seeds)))
run_count = 1
t_start = time.monotonic()

print(">>>", "baseline".center(10), "<<<")
for max_len in max_lengths:
    for rs in random_seeds:
        for ds in data_seeds:
            print("="*3, f"{run_count} / {num_runs}".center(10), "="*3)
            results_dir = f'runs/baseline/'
            filename = f'{max_len}_{ds}_{rs}'
            opts = get_opts(0.0, None, max_len, rs, ds, filename, results_dir)
            process = subprocess.Popen(
                ['python3', 'ancm/train.py']
                + [o for opt in opts for o in opt.split()])
            exitcode = process.wait()

            elapsed = timedelta(seconds=time.monotonic()-t_start)
            elapsed_per_run = elapsed.seconds / run_count
            minutes, seconds = divmod(elapsed_per_run, 60)
            elapsed = str(elapsed).split('.', maxsplit=1)[0]
            elapsed_per_run = f'{int(minutes):02}:{int(seconds):02}'
            print(f"elapsed time: {elapsed} ({elapsed_per_run} per run)")
            print('')
            run_count += 1
 
for channel in channels:
    print(">>>", f"{channel}".center(10), "<<<")
    for max_len in max_lengths:
        for pr in error_probs[1:]:
            for rs in random_seeds:
                for ds in data_seeds:
                    print("="*3, f"{run_count} / {num_runs}".center(10), "="*3)
                    print(f"error prob: {pr}")
                    print(f"seed: {rs} / {ds}")
                    print(f"max_len: {max_len}")

                    results_dir = f'runs/channel_{channel}/error_prob_{pr}/'
                    filename = f'{max_len}_{ds}_{rs}'
                    opts = get_opts(pr, channel, max_len, rs, ds, filename, results_dir)

                    process = subprocess.Popen(
                        ['python3', 'ancm/train.py']
                        + [o for opt in opts for o in opt.split()])
                    exitcode = process.wait()

                    elapsed = timedelta(seconds=time.monotonic()-t_start)
                    elapsed_per_run = elapsed.seconds / run_count
                    minutes, seconds = divmod(elapsed_per_run, 60)
                    elapsed = str(elapsed).split('.', maxsplit=1)[0]
                    elapsed_per_run = f'{int(minutes):02}:{int(seconds):02}'
                    print(f"elapsed time: {elapsed} ({elapsed_per_run} per run)")
                    print('')
                    run_count += 1
 
training_time = timedelta(seconds=time.monotonic()-t_start)
sec_per_run = training_time.seconds / run_count 
minutes, seconds = divmod(sec_per_run, 60)

time_total = str(training_time).split('.', maxsplit=1)[0]
time_per_run = f'{int(minutes):02}:{int(seconds):02}'

print("Total training time:", time_total)
print("       Time per run:", time_per_run)

for channel in channels:
    copy_tree("runs/baseline", f"runs/channel_{channel}/error_pr_0.0")
shutil.rmtree("runs/baseline")
