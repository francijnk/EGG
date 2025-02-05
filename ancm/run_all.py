from ancm.train import main
from egg.core import init
import time
from datetime import timedelta
import subprocess
from uuid import uuid4

random_seeds = [i + 1 for i in range(5)]
error_probs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
channels = [None, 'erasure', 'deletion', 'symmetric']
max_lengths = [2,3,4,5,8]

slr = 5e-3
rlr = 1e-3
length_cost = 0.01
vocab_size = 10
hidden_units = 128
n_epochs = 20
emb = 16
tau = 2.


run_count = 1
t_start = time.monotonic()
for channel in channels:
    for max_len in max_lengths:
        for pr in error_probs:
            for seed in random_seeds:
                print("="*3, f"{run_count} / {len(max_lengths) * len(error_probs) * len(random_seeds)}".center(10), "="*3)
                print(f"erasure_pr: {pr}")
                print(f"seed: {seed}")
                print(f"max_len: {max_len}")

                filename = f'{channel}_{max_len}_{seed}'
            
                opts = [
                    f'--error_prob {pr:.2f}',
                    f'--max_len {max_len}',
                    f'--vocab_size {vocab_size}',
                    f'--sender_lr {slr}',
                    f'--receiver_lr {rlr}',
                    f'--length_cost {length_cost}',
                    f'--sender_hidden {hidden_units}',
                    f'--receiver_hidden {hidden_units}',
                    f'--random_seed {seed}',
                    f'--filename {filename}',
                    f'--n_epochs {n_epochs}',
                    f'--temperature {tau}',
                    f'--embedding {emb}',
                    '--image_input',
                    '--data_path ancm/data/input_data/obverter-5-100-64.npz',
                    #'--data_path ancm/data/input_data/visa-4-250.npz',
                    '--optim adam',
                    '--n_permutations_train 5',
                    '--sender_entropy_coeff 0.01',
                    '--receiver_entropy_coeff 0.001',
                    '--sender_cell lstm',
                    '--receiver_cell lstm',
                    '--validation_freq 1',
                    '--wandb_project cezary_snellius ',
                    '--wandb_entity koala-lab',
                    f'--wandb_run_id {max_len}_{channel}_{seed}_{uuid4()}',
                    '--results_folder C:/Users/franc/Documents/GitHub/ANCM_EGG/ancm/runs/results_02_05_img',
                ]

                if channel is not None:
                    opts.append(f'--channel {channel}')

                process = subprocess.Popen(
                    ['python3', '-m', 'ancm.train']
                    + [o for opt in opts[:-1] for o in opt.split()]
                    )
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
sec_per_run = training_time.seconds / (len(random_seeds) * len(max_lengths) * len(error_probs))
minutes, seconds = divmod(sec_per_run, 60)

time_total = str(training_time).split('.', maxsplit=1)[0]
time_per_run = f'{int(minutes):02}:{int(seconds):02}'

print("Total training time:", time_total)
print("       Time per run:", time_per_run)

