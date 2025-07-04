import sys
from uuid import uuid4

random_seeds = [i + 1 for i in range(20)]
error_probs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
channels = 'erasure deletion'.split()

slr = 5e-4
rlr = 1e-4
length_cost = 0.01
vocab_size = 10
hidden_units = 128
n_epochs = 60
emb = 16
weight_decay = 0.01
temperature_start = 1.5
optimizer = 'adamw'
warmup = 250
label_coeff = 0.5
features_coeff = 1


def get_opts(error_prob, channel, max_len, random_seed):
    if channel:
        _channel = f'{channel}_{error_prob:.2f}'
    else:
        _channel = 'baseline'
    filename = f'{_channel}_{max_len}_{random_seed}'
    opts = [
        f'--error_prob {error_prob:.2f}',
        f'--max_len {max_len}',
        f'--vocab_size {vocab_size}',
        f'--sender_lr {slr}',
        f'--receiver_lr {rlr}',
        f'--label_coeff {label_coeff}',
        f'--features_coeff {features_coeff}',
        f'--length_cost {length_cost}',
        f'--sender_hidden {hidden_units}',
        f'--temperature_start {temperature_start}',
        '--temperature_end 1',
        f'--receiver_hidden {hidden_units}',
        f'--random_seed {random_seed}',
        f'--filename {filename}',
        f'--n_epochs {n_epochs}',
        f'--embedding {emb}',
        f'--optim {optimizer}',
        f'--warmup_steps {warmup}',
        f'--weight_decay {weight_decay}',
        '--sender_cell lstm',
        '--receiver_cell lstm',
        '--validation_freq 1',
        # '--wandb_project cezary_snellius ',
        # '--wandb_entity koala-lab',
        # f'--wandb_run_id {max_len}_{_channel}_{random_seed}_{uuid4()}'
        # '--results_folder runs_01_23/'
    ]
    if channel is not None:
        opts.append(f'--channel {channel}')
    return opts


if __name__ == '__main__':
    all_params = []
    max_len = sys.argv[1]

    # baseline (0 error prob)
    if 0. in error_probs:
        for rs in random_seeds:
            params = get_opts(0., None, max_len, rs)
            all_params.append(' '.join(params))

    # nonzero error probs for each channel
    for channel in channels:
        for pr in [p for p in error_probs if p > 0]:
            for rs in random_seeds:
                params = get_opts(pr, channel, max_len, rs)
                all_params.append(' '.join(params))

    params_fpath = sys.argv[2]
    with open(params_fpath, 'w') as fp:
        fp.write('\n'.join(all_params))

    print(len(all_params), 'lines saved to', params_fpath)
