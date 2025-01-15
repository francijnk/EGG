import os
import argparse
import pandas as pd
import seaborn as sns

from collections import namedtuple, defaultdict

import ancm.analyse.style
from ancm.analyse.util import load_data, get_long_val_data, close_plot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--recompute', '-r', action='store_true')
    return parser.parse_args()


def plot_test(data_test, output_dir):
    test_long = pd.melt(
        pd.DataFrame(data_test),
        id_vars='max_len channel error_prob noise'.split(),
        value_vars=None, var_name='metric', value_name='value', ignore_index=True)
    test_long = test_long.sort_values('max_len')
    test_long.max_len = test_long.max_len.astype(str)
    test_long.value = test_long.value.astype(float)
    test_long.error_prob = test_long.error_prob.astype(float)

    value_x_tick = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # (1) Accuracy
    df = test_long[test_long.metric.isin(['accuracy', 'accuracy2', 'embedding_alignment'])]
    plot = sns.relplot(
        df,
        x='error_prob', y='value',
        col='max_len', row='channel', style='noise',
        kind='line', 
        errorbar=None,  # ('se', 2), 
        marker='o',
        markersize=8, hue='metric',
        facet_kws=dict(margin_titles=True), legend=True)
    # sns.move_legend(
    #    plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #    title="max message length", frameon=False,)

    plot.set(xticks=value_x_tick)
    plot.savefig(os.path.join(output_dir, "test_accuracy.png"), dpi=400)
    close_plot(plot)

    # (2) Redundancy
    df = test_long[test_long.metric.isin([
        'redundancy_message',
        'redundancy_symbol',
        'redundancy_symbol_adj1',
        'redundancy_symbol_adj2'])]
    plot = sns.relplot(
        df,
        x='error_prob', y='value', hue='metric',
        col='max_len', row='channel', style='noise',
        errorbar=None, # errorbar=('se',2),
        marker='o', kind='line', markersize=8,
        facet_kws=dict(margin_titles=True), legend=True)
    # sns.move_legend(
    #    plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #     title="redundancy", frameon=False,)
    plot.set(xticks=value_x_tick)
    plot.savefig(os.path.join(output_dir, "test_redundancy.png"), dpi=400)
    close_plot(plot)

    # (3) Max. rep.
    df = test_long[test_long.metric == 'max_rep']
    plot = sns.relplot(
        df, col='max_len', x='error_prob', y='value', kind='line',
        errorbar=('se',2), style='noise', row='channel', marker='o',
        markersize=8, # hue='max_len',
        facet_kws=dict(margin_titles=True), legend=True)
    # sns.move_legend(
    #     plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #     title="avg len of max rep subsequence", frameon=False,)
    plot.set(xticks=value_x_tick)
    plot.savefig(os.path.join(output_dir, "test_max_rep.png"), dpi=400)
    close_plot(plot)

    # (4) Compositionality
    df = test_long[(test_long.metric.isin('topographic_rho pos_dis bos_dis'.split()))]
    # sns.set_palette(sns.color_palette("Set2", 3))
    plot = sns.relplot(
        df,
        x='error_prob', y='value', style='noise',
        col='max_len', row='channel',
        errorbar=None, # errorbar=('se',2),
        marker='o', markersize=8, kind='line',
        hue='metric', facet_kws=dict(margin_titles=True), legend=True)
    # sns.move_legend(
    #    plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #    title="max message length", frameon=False,)
    plot.set(xticks=value_x_tick)
    plot.savefig(os.path.join(output_dir, "test_compositionality.png"), dpi=400)
    close_plot(plot)


def plot_training(data_val, output_dir):
    metrics = ['accuracy']
    df = get_long_val_data(data_val, metrics)

    sns.set_palette(sns.husl_palette(4, 0.76))
    plot = sns.relplot(
        legend=True, data=df,
        x='epoch', y='value', hue='channel',
        row='max_len', col='error_prob', style='noise',
        errorbar=None,  # errorbar=('se', 2),
        kind='line', facet_kws=dict(margin_titles=True))
    plot.set_titles("{col_name}")
    plot.savefig(os.path.join(output_dir, "training_accuracy.png"))
    close_plot(plot)

    metrics = ['alignment']
    df = get_long_val_data(data_val, metrics)

    sns.set_palette(sns.husl_palette(4, 0.76))
    plot = sns.relplot(
        legend=True, data=df,
        x='epoch', y='value', hue='channel',
        row='max_len', col='error_prob', style='noise',
        errorbar=None,  # errorbar=('se', 2),
        kind='line', facet_kws=dict(margin_titles=True))
    plot.set_titles("{col_name}")
    plot.savefig(os.path.join(output_dir, "training_alignment.png"))
    close_plot(plot)

    # redundancy
    metrics = ['redund_msg', 'redund_smb', 'redund_smb_adj', 'redund_smb_adj2']
    df = get_long_val_data(data_val, metrics)

    for channel in pd.unique(df.channel):
        df_channel = df[df.channel == channel]
        sns.set_palette(sns.husl_palette(4))
        plot = sns.relplot(
            legend=True, data=df,
            row='max_len', col='error_prob',
            x='epoch', y='value', kind='line',
            hue='metric', style='noise',
            errorbar=None,
            # errorbar=('se', 2),
            facet_kws=dict(margin_titles=True))
        plot.set_titles("{col_name}")
        plot.savefig(os.path.join(output_dir, f"training_redundancy_{channel}.png"))
        close_plot(plot)

    # lexicon size
    metrics = ['lexicon_size']
    df = get_long_val_data(data_val, metrics)

    sns.set_palette(sns.husl_palette(4, 0.76))
    plot = sns.relplot(
        legend=True, data=df,
        row='max_len', col='error_prob',
        x='epoch', y='value', kind='line',
        hue='channel', style='noise',
        errorbar=None,
        # errorbar=('se', 2),
        facet_kws=dict(margin_titles=True))
    plot.set_titles("{col_name}")
    plot.savefig(os.path.join(output_dir, "training_lexicon.png"))
    close_plot(plot)

    metrics = ['actual_vocab_size']
    df = get_long_val_data(data_val, metrics)

    sns.set_palette(sns.husl_palette(4, 0.76))
    plot = sns.relplot(
        legend=True, data=df,
        x='epoch', y='value', hue='channel',
        row='max_len', col='error_prob', style='noise',
        errorbar=None,  # errorbar=('se', 2),
        kind='line', facet_kws=dict(margin_titles=True))
    plot.set_titles("{col_name}")
    plot.savefig(os.path.join(output_dir, "training_actual_vs.png"))
    close_plot(plot)

    # length, max rep, actual_vs
    metrics = ['length', 'max_rep']
    df = get_long_val_data(data_val, metrics)

    for channel in pd.unique(df.channel):
        df_channel = df[df.channel == channel]
        sns.set_palette(sns.husl_palette(3))
        plot = sns.relplot(
            legend=True, data=df,
            row='max_len', col='error_prob',
            x='epoch', y='value', kind='line',
            hue='metric', style='noise',
            errorbar=None,
            # errorbar=('se', 2),
            facet_kws=dict(margin_titles=True))
        plot.set_titles("{col_name}")
        plot.savefig(os.path.join(output_dir, f"training_msg_len_{channel}.png"))
        close_plot(plot)

    # compositionality
    metrics = 'top_sim pos_dis bos_dis'.split()
    df = get_long_val_data(data_val, metrics)
    for channel in pd.unique(df.channel):
        df_channel = df[df.channel == channel]
        sns.set_palette(sns.color_palette("husl", 3))
        plot = sns.relplot(
            data=df_channel,
            x='epoch', y='value',
            row='max_len', col='error_prob', style='noise',
            hue='metric', kind='line',
            errorbar=None,
            facet_kws=dict(margin_titles=True, legend_out=True))
        plot.set_titles("{col_name}")
        plot.savefig(os.path.join(output_dir, f"training_compositionality_{channel}.png"))
        close_plot(plot)


def analyse(data_test, output_dir):
    channels = pd.unique(data_test['channel'])
    max_lengths = pd.unique(data_test['max_len'])
    error_probs = pd.unique(data_test['error_prob'])

    lines = []
    for channel in channels:
        lines.append("\n" + "#"*20)
        lines.append("# " + channel.center(16) + " #")
        lines.append("#"*20)
        for max_len in sorted(max_lengths):
            data = data_test.loc[
                (data_test.channel == channel) & (data_test.max_len == max_len)]
            header = (
                "error_prob ".ljust(12) +
                "acc ".ljust(6) +
                "acc2 ".ljust(6) +
                "acc2_nn ".ljust(10) +
                "unique_msg ".ljust(12) +
                "unique_msg_nn ".ljust(16) +
                "avg_len ".ljust(8) +
                "avg_len_nn ".ljust(12) +
                "max_rep ".ljust(8) +
                "max_rep_nn ".ljust(12) +
                "act_vs ".ljust(8))
            lines.append("\n" + f"max_len = {max_len}")
            lines.append("=" * len(header))
            lines.append(header)
            for error_prob in error_probs:
                rows_noise = data.loc[
                    (data.error_prob == error_prob) & (data.noise == 'noise')]
                rows_no_noise = data.loc[
                    (data.error_prob == error_prob) & (data.noise == 'no noise')]

                unique_msg = rows_noise['unique_messages'].mean()
                acc = rows_no_noise['accuracy'].mean()
                acc2 = rows_noise['accuracy2'].mean()
                acc2_nn = rows_no_noise['accuracy2'].mean()
                unique_msg_nn = rows_no_noise['unique_messages'].mean()
                avg_len = rows_noise['avg_length'].mean()
                avg_len_nn = rows_no_noise['avg_length'].mean()
                max_rep = rows_noise['max_rep'].mean()
                max_rep_nn = rows_noise['max_rep'].mean()
                actual_vs = rows_noise['actual_vocab_size'].mean()
                # actual_vs_nn = rows_no_noise['actual_vocab_size'].mean()

                # TODO standard errors

                lines.append(
                    f"{error_prob:.2f}".ljust(12) +
                    f"{acc:.2f}".ljust(6) +
                    f"{acc2:.2f}".ljust(6) +
                    f"{acc2_nn:.2f}".ljust(10) +
                    f"{unique_msg:.2f}".ljust(12) +
                    f"{unique_msg_nn:.2f}".ljust(16) +
                    f"{avg_len:.2f}".ljust(8) +
                    f"{avg_len_nn:.2f}".ljust(12) +
                    f"{max_rep:.2f}".ljust(8) +
                    f"{max_rep_nn:.2f}".ljust(12) +
                    f"{actual_vs:.2f}".ljust(8))
            lines.append("=" * len(header))

    with open(os.path.join(output_dir, "stats.txt"), "w") as file:
        file.write("\n".join(lines))


def main():
    args = parse_args()

    processed_data_path = os.path.join(args.i, 'processed')
    if args.recompute or not os.path.isdir(processed_data_path):
        # export data and save it
        os.makedirs(processed_data_path, exist_ok=True)
        data_test, data_val = load_data(args.i)

        data_test.to_csv(os.path.join(args.i, 'processed', 'data_test.csv'))
        data_val.to_csv(os.path.join(args.i, 'processed', 'data_val.csv'))
    else:
        # load already exported data
        data_test = pd.read_csv(os.path.join(args.i, 'processed', 'data_test.csv'))
        data_val = pd.read_csv(os.path.join(args.i, 'processed', 'data_val.csv'))

    os.makedirs(args.o, exist_ok=True)

    plot_test(data_test, args.o)
    plot_training(data_val, args.o)
    analyse(data_test, args.o)


if __name__ == '__main__':
    main()
