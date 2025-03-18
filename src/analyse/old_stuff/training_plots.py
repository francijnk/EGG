import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import seaborn as sns
import style


confidence = 0.95
RESULTS_DIR = 'runs/'



all_dfs = defaultdict(list)
for d in os.listdir('runs/'):
    if not d.startswith('channel') or not os.path.isdir(os.path.join('runs', d)):
        continue
    channel = d[d.index('_')+1:]
    for dd in os.listdir(os.path.join(RESULTS_DIR, d)):
        if not dd.startswith('error_prob_') or not os.path.isdir(os.path.join(RESULTS_DIR, d)):
            continue
        error_prob = dd.strip('error_prob_')
        directory = os.path.join(RESULTS_DIR, d, dd)

        for file in os.listdir(directory):
            if file.endswith('csv'):
                max_len, _, seed = (int(item) for item in file[:file.index('-')].split('_'))
            else:
                continue

            fpath = os.path.join(directory, file)
            df = pd.read_csv(fpath)
            if error_prob != 0.:
                df_noise = df[df.phase == 'val']
                df_no_noise = df[df.phase == 'val (no noise)']
            else:
                df_noise = df[df.phase == 'val']
                df_no_noise = df[df.phase == 'val']
            df_noise = df_noise.assign(noise=['noise' for _ in range(len(df_noise))])
            df_no_noise = df_no_noise.assign(noise=['no noise' for _ in range(len(df_noise))])
            df_noise = df_noise.reset_index(drop=True)
            df_no_noise = df_no_noise.reset_index(drop=True)
            df_no_noise['accuracy'] = df_no_noise['accuracy'] / 100
            df_noise['accuracy'] = df_noise['accuracy'] / 100
            all_dfs[(max_len, channel, error_prob)].append(df_noise)
            all_dfs[(max_len, channel, error_prob)].append(df_no_noise)

metrics = 'accuracy length alignment top_sim pos_dis bos_dis'.split()

comp_long = defaultdict(list)
acc_long = defaultdict(list)

for max_len, channel, error_prob in all_dfs:
    for i in range(len(all_dfs[(max_len, channel, error_prob)][0])):
        for metric in metrics:
            vals = [df[metric][i] for df in all_dfs[(max_len, channel, error_prob)]]
            vals = [v for v in vals if v == v]
            noise = [df['noise'][i] for df in all_dfs[(max_len, channel, error_prob)]]
            for v, n in zip(vals, noise):
                if metric in 'redundancy top_sim pos_dis bos_dis'.split():
                    comp_long['epoch'].append(i+1)
                    comp_long['max_len'].append(max_len)
                    comp_long['channel'].append(channel)
                    comp_long['error_prob'].append(error_prob)
                    comp_long['metric'].append(metric)
                    comp_long['value'].append(v)
                    comp_long['noise'].append(n)
                if metric in 'accuracy redundancy'.split():
                    acc_long['epoch'].append(i+1)
                    acc_long['error_prob'].append(error_prob)
                    acc_long['max_len'].append(max_len)
                    acc_long['channel'].append(channel)
                    acc_long['metric'].append('accuracy')
                    acc_long['value'].append(v)
                    acc_long['noise'].append(n)


os.makedirs('figures/img', exist_ok=True)
hue_order = ['top_sim', 'pos_dis', 'bos_dis', 'accuracy']
sns.set_palette(sns.color_palette("husl", 4))
plot = sns.relplot(
    data=comp_long, row='max_len', col='error_prob', style='noise',
    x='epoch', y='value', hue='metric', kind='line', errorbar=None, facet_kws=dict(margin_titles=True, legend_out=True))
plot.set_titles("{col_name}")
plot.savefig("figures/img/training_compositionality.png") 

sns.set_palette(sns.husl_palette(4, 0.76))
acc_plot = sns.relplot(legend=True,
    data=acc_long, row='max_len', col='error_prob', x='epoch', y='value', kind='line', errorbar=('se',2),facet_kws=dict(margin_titles=True))
acc_plot.set_titles("{col_name}")
acc_plot.savefig("figures/img/training_accuracy.png") 

# sns.set_palette(sns.hls_palette(3, h=0.2))

