import sys

random_seeds = [i + 1 for i in range(5)]
error_probs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
channels = 'erasure deletion symmetric'.split()
max_lengths = [2, 3, 4, 5, 8]

slr = 5e-3
rlr = 1e-3
length_cost = 0.01
vocab_size = 10
hidden_units = 50
n_epochs = 40


def get_opts(error_prob, channel, max_len, random_seed):
    if channel:
        _channel = f'{channel}_{error_prob:.2f}'
        group = channel
    else:
        _channel = 'baseline'
        group = 'baseline'
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
        f'--filename {filename}',
        f'--n_epochs {n_epochs}',
        # '--data_path ancm/data/input_data/visa-4-500.npz',
        '--sender_embedding 10',
        '--receiver_embedding 10',
        '--sender_entropy_coeff 0.01',
        '--receiver_entropy_coeff 0.001',
        '--sender_cell lstm',
        '--receiver_cell lstm',
        '--evaluate',
        '--validation_freq 1',
        '--wandb_project cezary_snellius ',
        '--wandb_entity koala-lab',
        f'--wandb_group {group}',
        f'--wandb_run_id {max_len}_{_channel}_{random_seed}'
        # '--results_folder runs_01_23/'
    ]
    if channel is not None:
        opts.append(f'--channel {channel}')
    return opts


if __name__ == '__main__':
    all_params = []
    for max_len in max_lengths:
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

    params_fpath = sys.argv[1]
    with open(params_fpath, 'w') as fp:
        fp.write('\n'.join(all_params))

    print(len(all_params), 'lines saved to', params_fpath)
