import os
import gc
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


def parse_fname(fname):
    fname = fname[:fname.index('-')]
    fname_split = fname.split('_')

    channel = fname_split[0]
    error_prob = float(fname_split[1]) if not len(fname_split) == 3 else 0.
    max_len = int(fname_split[-2])
    seed = int(fname_split[-1])

    return channel, error_prob, max_len, seed


def load_data(input_dir):

    history_train = defaultdict(list)
    history_val = defaultdict(list)
    results = defaultdict(lambda: defaultdict(list))

    for fname in os.listdir(path=input_dir):
        fpath = os.path.join(input_dir, fname)
        if fname.endswith('csv'):
            channel, error_prob, max_len, seed = parse_fname(fname)
            df = pd.read_csv(fpath)

            df_train = df[df.phase == 'train']
            if error_prob != 0.:
                df_noise = df[df.phase == 'val']
                df_no_noise = df[df.phase == 'val (nn)']
            else:
                df_noise = df[df.phase == 'val']
                df_no_noise = df[df.phase == 'val']

            df_noise = df_noise.assign(noise=['noise' for _ in range(len(df_noise))])
            df_noise = df_noise.reset_index(drop=True)
            df_noise['accuracy'] = df_noise['accuracy'] / 100

            df_no_noise = df_no_noise.assign(noise=['no noise' for _ in range(len(df_noise))])
            df_no_noise = df_no_noise.reset_index(drop=True)
            df_no_noise['accuracy'] = df_no_noise['accuracy'] / 100

            history_val[(max_len, channel, error_prob)].append(df_noise)
            history_val[(max_len, channel, error_prob)].append(df_no_noise)
            history_train[(max_len, channel, error_prob)].append(df_train)

        elif fname.endswith('json'):
            channel, error_prob, max_len, seed = parse_fname(fname)
            with open(fpath) as file:
                fdata = json.load(file)

            for dataset_key in fdata:  # train or test
                for condition_key in fdata[dataset_key]:  # noise / no noise
                    results['max_len'].append(max_len)
                    results['channel'].append(channel)
                    results['error_prob'].append(error_prob)
                    results['noise'].append(condition_key)
                    results['unique_targets'].append(fdata['results']['unique_targets'])
                    results['unique_messages'].append(fdata['results']['unique_msg'])
                    avg_len = np.mean([m['message'].count(',') for m in fdata['messages']]).item()
                    results['avg_length'].append(avg_len)
                    for metric_key, metric_val in fdata[dataset_key][condition_key].items():
                        # _get = lambda metric: fdata['results'][dataset_key][condition_key][metric]
                        results[metric_key].append(_get(metric_val))

    channels = set(results['channel']) - {'baseline'}

    # history val: export to DataFrame and handle baseline results
    for max_len, channel, error_prob in list(history_val.keys()):
        if channel != 'baseline':
            continue
        key = (max_len, channel, error_prob)
        for c in channels:
            new_key = (max_len, c, error_prob)
            history_val[new_key] = history_val[key]
            history_train[new_key] = history_train[key]
        del history_val[key]
        del history_train[key]

    for max_len, channel, error_prob in history_val:
        for df in history_val[(max_len, channel, error_prob)]:
            df['max_len'] = max_len
            df['channel'] = channel
            df['error_prob'] = error_prob
 
    for max_len, channel, error_prob in history_train:
        for df in history_train[(max_len, channel, error_prob)]:
            df['max_len'] = max_len
            df['channel'] = channel
            df['error_prob'] = error_prob

    history_val = pd.concat([df for key in history_val for df in history_val[key]], ignore_index=True)
    history_train = pd.concat([df for key in history_train for df in history_train[key]], ignore_index=True)

    # results: export to DataFrame and handle baseline results
    results = pd.DataFrame(results)

    baseline_df_list = []
    for channel in channels:
        df = results[results['channel'] == 'baseline'].copy()
        df['channel'] = channel
        baseline_df_list.append(df)
    baseline_df = pd.concat(baseline_df_list, ignore_index=True)

    results = results.drop(
        results[results['channel'] == 'baseline'].index)
    results = pd.concat([baseline_df, results], ignore_index=True)

    return history_train, history_val, results


def get_long_data(history_val, metrics):
    data_long = pd.melt(
        history_val,
        id_vars='epoch max_len channel error_prob noise'.split(),
        value_vars=metrics, var_name='metric', value_name='value', ignore_index=True)
    # data_long.dropna(inplace=True)
    return data_long


def close_plot(plot):
    plt.close()
    gc.collect()
