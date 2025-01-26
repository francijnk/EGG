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

    data_train = defaultdict(list)
    data_val = defaultdict(list)
    data_test = defaultdict(list)

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

            data_val[(max_len, channel, error_prob)].append(df_noise)
            data_val[(max_len, channel, error_prob)].append(df_no_noise)
            data_train[(max_len, channel, error_prob)].append(df_train)

        elif fname.endswith('json'):
            channel, error_prob, max_len, seed = parse_fname(fname)
            with open(fpath) as file:
                fdata = json.load(file)

            r_nn = 'results-no-noise' if 'results-no-noise' in fdata else 'results'

            data_test['max_len'].append(max_len)
            data_test['channel'].append(channel)
            data_test['error_prob'].append(error_prob)
            data_test['noise'].append('noise')
            data_test['accuracy'].append(fdata['results']['accuracy'])
            data_test['accuracy2'].append(fdata['results']['accuracy2'])
            data_test['redundancy_message'].append(fdata['results']['redundancy_msg'])
            data_test['redundancy_symbol'].append(fdata['results']['redundancy_smb'])
            data_test['redundancy_symbol_adj'].append(fdata['results']['redundancy_smb_adj'])
            # data_test['redundancy_symbol_adj2'].append(fdata['results']['redundancy_smb_adj2'])
            data_test['max_rep'].append(fdata['results']['max_rep'])
            # data_test['embedding_alignment'].append(fdata['results']['embedding_alignment'])
            data_test['topographic_rho'].append(fdata['results']['topographic_rho'])
            # data_test['pos_dis'].append(fdata['results']['pos_dis'])
            # data_test['bos_dis'].append(fdata['results']['bos_dis'])
            data_test['unique_targets'].append(fdata['results']['unique_targets'])
            data_test['unique_messages'].append(fdata['results']['unique_msg'])
            avg_len = np.mean([m['message'].count(',')  for m in fdata['messages']]).item()
            data_test['avg_length'].append(avg_len)
            data_test['actual_vocab_size'].append(fdata['results']['actual_vocab_size'])

            data_test['max_len'].append(max_len)
            data_test['channel'].append(channel)
            data_test['error_prob'].append(error_prob)
            data_test['noise'].append('no noise')
            data_test['accuracy'].append(fdata[r_nn]['accuracy'])
            data_test['accuracy2'].append(fdata[r_nn]['accuracy2'])
            data_test['redundancy_message'].append(fdata[r_nn]['redundancy_msg'])
            data_test['redundancy_symbol'].append(fdata[r_nn]['redundancy_smb'])
            data_test['redundancy_symbol_adj'].append(fdata[r_nn]['redundancy_smb_adj'])
            # data_test['redundancy_symbol_adj2'].append(fdata[r_nn]['redundancy_smb_adj2'])
            data_test['max_rep'].append(fdata[r_nn]['max_rep'])
            # data_test['embedding_alignment'].append(fdata[r_nn]['embedding_alignment'])
            data_test['topographic_rho'].append(fdata[r_nn]['topographic_rho'])
            # data_test['pos_dis'].append(fdata[r_nn]['pos_dis'])
            # data_test['bos_dis'].append(fdata[r_nn]['bos_dis'])
            data_test['unique_targets'].append(fdata['results']['unique_targets'])
            if 'unique_msg_no_noise' in fdata['results']:
                data_test['unique_messages'].append(fdata['results']['unique_msg_no_noise'])
            else:
                data_test['unique_messages'].append(fdata['results']['unique_msg'])
            if 'message_no_noise' in fdata['messages'][0]:
                avg_len = np.mean([m['message_no_noise'].count(',')  for m in fdata['messages']]).item()
            else:
                avg_len = np.mean([m['message'].count(',')  for m in fdata['messages']]).item()
            data_test['avg_length'].append(avg_len)
            data_test['actual_vocab_size'].append(fdata[r_nn]['actual_vocab_size'])

    channels = set(data_test['channel']) - {'baseline'}

    # data val: export to DataFrame and handle baseline results
    for max_len, channel, error_prob in list(data_val.keys()):
        if channel != 'baseline':
            continue
        key = (max_len, channel, error_prob)
        for c in channels:
            new_key = (max_len, c, error_prob)
            data_val[new_key] = data_val[key]
            data_train[new_key] = data_train[key]
        del data_val[key]
        del data_train[key]

    for max_len, channel, error_prob in data_val:
        for df in data_val[(max_len, channel, error_prob)]:
            df['max_len'] = max_len
            df['channel'] = channel
            df['error_prob'] = error_prob
        for df in data_train[(max_len, channel, error_prob)]:
            df['max_len'] = max_len
            df['channel'] = channel
            df['error_prob'] = error_prob
    data_val = pd.concat([df for key in data_val for df in data_val[key]], ignore_index=True)
    data_train = pd.concat([df for key in data_val for df in data_val[key]], ignore_index=True)

    # data_test: export to DataFrame and handle baseline results
    data_test = pd.DataFrame(data_test)

    baseline_df_list = []
    for channel in channels:
        df = data_test[data_test['channel'] == 'baseline'].copy()
        df['channel'] = channel
        baseline_df_list.append(df)
    baseline_df = pd.concat(baseline_df_list, ignore_index=True)

    data_test = data_test.drop(
        data_test[data_test['channel'] == 'baseline'].index)
    data_test = pd.concat([baseline_df, data_test], ignore_index=True)

    return data_train, data_val, data_test


def get_long_train_data(data_train, metrics):
    data_long = pd.melt(
        data_val,
        id_vars='epoch max_len channel error_prob'.split(),
        value_vars=metrics, var_name='metric', value_name='value', ignore_index=True)
    # data_long.dropna(inplace=True)
    return data_long


def get_long_val_data(data_val, metrics):
    data_long = pd.melt(
        data_val,
        id_vars='epoch max_len channel error_prob noise'.split(),
        value_vars=metrics, var_name='metric', value_name='value', ignore_index=True)
    # data_long.dropna(inplace=True)
    return data_long


def get_long_data(data, metrics):
    if 'noise' in data.columns:
        return get_long_val_data(data, metrics)
    else:
        return get_long_train_data(data, metrics)


def close_plot(plot):
    plt.close()
    gc.collect()
