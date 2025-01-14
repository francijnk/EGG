import sys
import time
from datetime import timedelta
import subprocess

# load parameters from a file
with open(sys.argv[2]) as fp:
    param_lines = fp.read().split('\n')

print(param_lines[0], param_lines[-1])
param_sets = [[p for p in line.split()] for line in param_lines]


def task(*param_set):
    output_dir = sys.argv[1]
    command = f'python3 -m ancm.train --dump_results_folder {output_dir}'.split() \
        + list(*param_set)
    process = subprocess.Popen(command)
    _ = process.wait()


# training loop
t_start = time.monotonic()
run_count = 1
for param_set in param_sets:
    print(run_count, '/', len(param_sets))

    task(param_set)

    elapsed = timedelta(seconds=time.monotonic() - t_start)
    elapsed_per_run = elapsed.seconds / run_count
    minutes, seconds = divmod(elapsed_per_run, 60)
    elapsed = str(elapsed).split('.', maxsplit=1)[0]
    elapsed_per_run = f'{int(minutes):02}:{int(seconds):02}'
    print(f"elapsed time: {elapsed} ({elapsed_per_run} per run)")
    print('')
    run_count += 1


training_time = timedelta(seconds=time.monotonic() - t_start)
sec_per_run = training_time.seconds / len(param_sets)
minutes, seconds = divmod(sec_per_run, 60)

time_total = str(training_time).split('.', maxsplit=1)[0]
time_per_run = f'{int(minutes):02}:{int(seconds):02}'

print("Total training time:", time_total)
print("       Time per run:", time_per_run)
