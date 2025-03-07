import os
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tabulate

import ancm.analyse.style
from ancm.analyse.util import load_data, get_long_data, close_plot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    return parser.parse_args()


def plot_final(df, output_dir, dataset):
    df_long = pd.melt(
        df,
        id_vars='max_len channel error_prob condition'.split(),
        value_vars=None, var_name='measure',
        value_name='value', ignore_index=True,
    )
    df_long = df_long.sort_values('max_len')
    df_long.max_len = df_long.max_len.astype(str)
    df_long.value = df_long.value.astype(float)
    df_long.error_prob = df_long.error_prob.astype(float)

    value_x_tick = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # (1) Accuracy
    metrics = ['accuracy', 'accuracy_symbol_removal', 'topsim', 'redundancy']
    df_acc = df_long[df_long.measure.isin(metrics)]
    plot = sns.relplot(
        df_acc,
        x='error_prob', y='value',
        col='max_len', row='channel',
        style='condition', hue='measure',
        kind='line',
        errorbar=None,  # ('se', 2),
        marker='o',
        markersize=8,
        facet_kws=dict(margin_titles=True), legend=True)
    # sns.move_legend(
    #    plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #    title="max message length", frameon=False,)

    plot.set(xticks=value_x_tick)
    fname = f'final_{dataset}_accuracy.pdf'
    plot.savefig(
        os.path.join(output_dir, fname),
        format='pdf',
        dpi=None,
        pad_inches=0.01,
        bbox_inches='tight',
    )
    close_plot(plot)

    # (2) Redundancy
    # df_r = df_long[df_long.measure == 'redundancy']
    # plot = sns.relplot(
    #     df_r,
    #     x='error_prob', y='value', hue='measure',
    #     col='max_len', row='channel', style='condition',
    #     errorbar=None,  # errorbar=('se',2),
    #     marker='o', kind='line', markersize=8,
    #     facet_kws=dict(margin_titles=True), legend=True,
    # )
    # sns.move_legend(
    #    plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #     title="redundancy", frameon=False,)
    # plot.set(xticks=value_x_tick)
    # fname = f'final_{dataset}_redundancy.pdf'
    # plot.savefig(
    #     os.path.join(output_dir, fname),
    #     format='pdf',
    #     dpi=None,
    #     pad_inches=0.01,
    #     bbox_inches='tight',
    # )
    # close_plot(plot)

    # (3) KLD
    df_kld = df_long[df_long.measure.apply(lambda x: x.startswith('KLD'))]
    plot = sns.relplot(
        df_kld,
        x='error_prob', y='value', hue='measure',
        col='max_len', row='channel', style='condition',
        errorbar=None,  # errorbar=('se',2),
        marker='o', kind='line', markersize=8,
        facet_kws=dict(margin_titles=True), legend=True,
    )
    # sns.move_legend(
    #    plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #     title="redundancy", frameon=False,)
    plot.set(xticks=value_x_tick)
    fname = f'final_{dataset}_kld.pdf'
    plot.savefig(
        os.path.join(output_dir, fname),
        format='pdf',
        dpi=None,
        pad_inches=0.01,
        bbox_inches='tight',
    )
    close_plot(plot)

    # (3) Max. rep.
    # df = test_long[test_long.metric == 'max_rep']
    # plot = sns.relplot(
    #    df, col='max_len', x='error_prob', y='value', kind='line',
    #    errorbar=('se',2), style='noise', row='channel', marker='o',
    #    markersize=8, # hue='max_len',
    #    facet_kws=dict(margin_titles=True), legend=True)
    # sns.move_legend(
    #     plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #     title="avg len of max rep subsequence", frameon=False,)
    # lot.set(xticks=value_x_tick)
    # plot.savefig(os.path.join(output_dir, "test_max_rep.png"), dpi=400)
    # close_plot(plot)

    # (4) Compositionality
    # df = test_long[(test_long.metric.isin('topographic_rho pos_dis bos_dis'.split()))]
    # sns.set_palette(sns.color_palette("Set2", 3))
    # plot = sns.relplot(
    #     df,
    #     x='error_prob', y='value', style='noise',
    #     col='max_len', row='channel',
    #     errorbar=None, # errorbar=('se',2),
    #     marker='o', markersize=8, kind='line',
    #     hue='metric', facet_kws=dict(margin_titles=True), legend=True)
    # sns.move_legend(
    #    plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3,
    #    title="max message length", frameon=False,)
    # plot.set(xticks=value_x_tick)
    # plot.savefig(os.path.join(output_dir, "test_compositionality.png"), dpi=400)
    # close_plot(plot)


def plot_history(history_df, output_dir, dataset):
    metrics = ['accuracy', 'redundancy', 'topsim']  # , 'temperature']
    df = get_long_data(history_df, metrics, dataset)
    sns.set_palette(sns.husl_palette(4, 0.76))
    plot = sns.relplot(
        legend=True, data=df,
        x='epoch', y='value', hue='channel',
        row='max_len', col='error_prob', style='condition',
        errorbar=None,  # errorbar=('se', 2),
        kind='line', facet_kws=dict(margin_titles=True))
    plot.set_titles("{col_name}")
    fname = f'history_{dataset}_redundancy.pdf'
    plot.savefig(
        os.path.join(output_dir, fname),
        format='pdf',
        dpi=None,
        pad_inches=0.01,
        bbox_inches='tight',
    )
    close_plot(plot)

    # metrics = ['alignment']
    # df = get_long_val_data(data_val, metrics)

    # sns.set_palette(sns.husl_palette(4, 0.76))
    # plot = sns.relplot(
    #     legend=True, data=df,
    #     x='epoch', y='value', hue='channel',
    #     row='max_len', col='error_prob', style='noise',
    #     errorbar=None,  # errorbar=('se', 2),
    #     kind='line', facet_kws=dict(margin_titles=True))
    # plot.set_titles("{col_name}")
    # plot.savefig(os.path.join(output_dir, "training_alignment.png"))
    # close_plot(plot)

    # redundancy
    for channel in pd.unique(df.channel):
        df_channel = history_df[history_df.channel == channel]
        sns.set_palette(sns.husl_palette(4))
        plot = sns.relplot(
            legend=True, data=df_channel,
            row='max_len', col='error_prob',
            x='epoch', y='redundancy', kind='line',
            hue='channel', style='condition',
            errorbar=None,  # errorbar=('se', 2),
            facet_kws=dict(margin_titles=True))
        plot.set_titles("{col_name}")
        fname = f'history_{dataset}_{channel}_redundancy.pdf'
        plot.savefig(
            os.path.join(output_dir, fname),
            format='pdf',
            dpi=None,
            pad_inches=0.01,
            bbox_inches='tight',
        )
        close_plot(plot)

    # temperature, entropy
    for channel in pd.unique(df.channel):
        print('entropy channel')
        # print(history_df[history_df.channel == channel]['entropy_msg'])
        df_channel = get_long_data(
            history_df[history_df.channel == channel],
            ['temperature', 'entropy_msg', 'entropy_msg_as_a_whole'],
            dataset,
        )
        print(df_channel)
        sns.set_palette(sns.husl_palette(4))
        plot = sns.relplot(
            legend=True, data=df_channel,
            row='max_len', col='error_prob',
            x='epoch', y='value', kind='line',
            hue='measure', style='condition',
            errorbar=None,
            # errorbar=('se', 2),
            facet_kws=dict(margin_titles=True))
        plot.set_titles("{col_name}")
        fname = f'history_{dataset}_{channel}_entropy.pdf'
        plot.savefig(
            os.path.join(output_dir, fname),
            format='pdf',
            dpi=None,
            pad_inches=0.01,
            bbox_inches='tight',
        )
        close_plot(plot)

    # lexicon size
    sns.set_palette(sns.husl_palette(4, 0.76))
    plot = sns.relplot(
        legend=True, data=history_df,
        row='max_len', col='error_prob',
        x='epoch', y='lexicon_size', kind='line',
        hue='channel', style='condition',
        errorbar=None,
        # errorbar=('se', 2),
        facet_kws=dict(margin_titles=True, sharey=False))
    plot.set_titles("{col_name}")
    fname = f'history_{dataset}_lexicon.pdf'
    plot.savefig(
        os.path.join(output_dir, fname),
        format='pdf',
        dpi=None,
        pad_inches=0.01,
        bbox_inches='tight',
    )
    close_plot(plot)

    # actual vocab size
    sns.set_palette(sns.husl_palette(4, 0.76))
    plot = sns.relplot(
        legend=True, data=history_df,
        x='epoch', y='actual_vocab_size', hue='channel',
        kind='line', facet_kws=dict(margin_titles=True)
    )
    plot.set_titles("{col_name}")
    fname = f'history_{dataset}_actual_vs.pdf'
    plot.savefig(
        os.path.join(output_dir, fname),
        format='pdf',
        dpi=None,
        pad_inches=0.01,
        bbox_inches='tight',
    )
    close_plot(plot)

    # length, max rep, actual_vs
    metrics = ['length', 'max_rep']
    for channel in pd.unique(df.channel):
        df_channel = get_long_data(
            history_df[history_df.channel == channel],
            metrics,
            dataset,
        )
        sns.set_palette(sns.husl_palette(3))
        plot = sns.relplot(
            legend=True, data=df,
            row='max_len', col='error_prob',
            x='epoch', y='value', kind='line',
            hue='measure', style='condition',
            errorbar=None,  # errorbar=('se', 2),
            facet_kws=dict(margin_titles=True),
        )
        plot.set_titles("{col_name}")
        fname = f'history_{dataset}_{channel}_msg_len.pdf'
        plot.savefig(
            os.path.join(output_dir, fname),
            format='pdf',
            dpi=None,
            pad_inches=0.01,
            bbox_inches='tight',
        )
        close_plot(plot)


def analyse(df, output_dir, dataset):
    print(type(df), df.columns)
    channels = pd.unique(df['channel'])
    max_lengths = pd.unique(df['max_len'])
    error_probs = pd.unique(df['error_prob'])

    def get_stats(values):
        try:
            mean = values.mean()
            # std_er = np.std(values, ddof=1) / np.sqrt(np.size(values))
            return f"{mean:.2f}"  # , f"{std_er:.2f}"
        except:
            return "--"  # , "--"

    lines = []
    header = [
        "error prob",  #.ljust(12) +
        "accuracy",  #.ljust(6) +
        "unique msg",  #.ljust(12) +
        "unique msg (nn)",  #.ljust(16) +
        "avg_len",  #.ljust(8) +
        "max_rep",  # .ljust(8) +
    ]
    header = header + header[1:]

    for i, channel in enumerate(channels):
        for j, max_len in enumerate(sorted(max_lengths)):
            data = df.loc[
                (data_test.channel == channel) & (data_test.max_len == max_len)]
            # lines.append(header)
            for l, error_prob in enumerate(error_probs):
                rows_noise = data.loc[
                    (data.error_prob == error_prob) & (data.condition == 'noise')]
                rows_no_noise = data.loc[
                    (data.error_prob == error_prob) & (data.condition == 'no noise')]

                unique_msg = rows_noise['unique_messages'].mean()
                acc = rows_no_noise['accuracy']
                acc2 = rows_noise['symbol_removal_accuracy']
                acc2_nn = rows_no_noise['symbol_removal_accuracy']
                unique_msg_nn = rows_no_noise['unique_msg']
                avg_len = rows_noise['average_length']
                avg_len_nn = rows_no_noise['average_length']
                max_rep = rows_noise['max_rep']#
                max_rep_nn = rows_noise['max_rep']
                actual_vs = rows_noise['actual_vocab_size']
                redundancy = rows_noise['redundancy']
                entropy = rows_noise['entropy_msg']
                # actual_vs_nn = rows_no_noise['actual_vocab_size'].mean()
                if i == 0:
                    lines.append([
                        error_prob,
                        get_stats(acc),
                        get_stats(unique_msg),
                        get_stats(unique_msg_nn),
                        get_stats(avg_len),
                        get_stats(max_rep),
                    ])
                else:
                    idx = l + j * (len(error_probs))
                    print('appending to', idx)
                    lines[idx].extend([
                        get_stats(acc),
                        get_stats(unique_msg),
                        get_stats(unique_msg_nn),
                        get_stats(avg_len),
                        get_stats(max_rep),
                    ])

                    #f"{error_prob:.2f}"#.ljust(12) +
                    #f"{acc:.2f}"#.ljust(6) +
                    #f"{acc2:.2f}"#.ljust(6) +
                    #f"{acc2_nn:.2f}"#.ljust(10) +
                    #f"{unique_msg:.2f}"#.ljust(12) +
                    #f"{unique_msg_nn:.2f}"#.ljust(16) +
                    #f"{avg_len:.2f}"#.ljust(8) +
                    #f"{avg_len_nn:.2f}"#.ljust(12) +
                    #f"{max_rep:.2f}"#.ljust(8) +
                    #f"{max_rep_nn:.2f}"#.ljust(12) +
                    #f"{actual_vs:.2f}"#.ljust(8) +
                    #f"{redundancy:.2f}"#.ljust(8) +
                    #f"{redundancy_smb:.2f}")#.ljust(8))
            #lines.append("=" * len(header))
    table = tabulate.tabulate(lines, header, tablefmt='latex_booktabs')
    print(table)
    #with open(os.path.join(output_dir, f"stats-{dataset}.txt"), "w") as file:
    #    file.write("\n".join(lines))


def plot_test_perchannel(data_test, out_dir, key):
    test_long = pd.melt(pd.DataFrame(data_test),
        id_vars='max_len channel error_prob condition'.split(),
        value_vars=None, var_name='measure', value_name='value', ignore_index=True)
    test_long = test_long.sort_values('max_len')
    test_long.max_len = test_long.max_len.astype(str)
    test_long.value = test_long.value.astype(float)
    test_long.error_prob = test_long.error_prob.astype(float)

    channels = pd.unique(test_long['channel'])

    col_names = ['accuracy', 'accuracy_symbol_removal', 'redundancy', 'topsim']
    df_metrics = test_long[test_long.measure.isin(col_names)]

    for channel in channels:
        df = df_metrics.loc[
                (df_metrics.channel == channel)]
        value_x_tick = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        plot = sns.relplot(df, x = "error_prob", y= 'value', row="max_len",
                           col="measure", hue="max_len", style="condition", kind='line',
                           marker ='o', markersize=8, facet_kws={"margin_titles": True})

        for ax in plot.axes.flatten():
            ax.tick_params(labelbottom=True)

        plot.set(xticks=value_x_tick)
        (plot
        .set_axis_labels("Error Probability", "Value")
        .set_titles(col_template="{col_name}", row_template="max len {row_name}")
        .set_xticklabels(value_x_tick)
        .tight_layout())
        plot.fig.subplots_adjust(top=0.95)
        plot.fig.suptitle(f'{channel}', fontsize = '30' )
        plot.savefig(
            os.path.join(out_dir, f"{key}_{channel}.pdf"),
            format='pdf',
            dpi=None,
            pad_inches=0.01,
            bbox_inches='tight',
        )
        close_plot(plot)


def main():
    args = parse_args()

    processed_data_path = os.path.join(args.i, 'processed')

    # export data and save it
    os.makedirs(processed_data_path, exist_ok=True)
    history_train, history_test, results_train, results_test = load_data(args.i)

    print('\ncolumns (training history):', history_train.columns)
    print('\ncolumns (final results):', results_train.columns, '\n')
    # data_train.to_csv(os.path.join(args.i, 'processed', 'data_train.csv'))
    # data_test.to_csv(os.path.join(args.i, 'processed', 'data_test.csv'))
    # data_val.to_csv(os.path.join(args.i, 'processed', 'data_val.csv'))

    os.makedirs(args.o, exist_ok=True)

    plot_final(results_train, args.o, 'train')
    plot_final(results_test, args.o, 'test')
    plot_history(history_train, args.o, 'train')
    plot_history(history_test, args.o, 'test')
    # analyse(results_train, args.o, 'train')
    # analyse(results_test, args.o, 'test')
    plot_test_perchannel(results_test, args.o, 'test')
    plot_test_perchannel(results_train, args.o, 'train')


if __name__ == '__main__':
    main()
