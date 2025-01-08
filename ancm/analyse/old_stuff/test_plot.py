import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import seaborn as sns
import style


df = pd.read_csv('figures/data/test_long.csv')
df = df.sort_values('max_len')
df.max_len = df.max_len.apply(str)
comp_df = df[(df.metric.isin('accuracy redundancy'.split()))]
value_x_tick = [0, 0.05, 0.10, 0.15, 0.20, 0.25]


#sns.set_palette(sns.color_palette("Set2", 3))
os.makedirs('figures/img', exist_ok=True)
sns.set(font_scale=1.9)
plot = sns.relplot(comp_df, col='metric', x='error_prob', y='value', kind='line', errorbar=('se',2), style='noise', row='channel', marker='o', markersize=8, hue='max_len', facet_kws=dict(margin_titles=True), legend=True)
sns.move_legend(plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3, title="max message length", frameon=False,)
plot.set(xticks=value_x_tick)
plot.savefig("figures/img/test_plot.png", dpi=400) 


comp_df = df[(df.metric.isin('topographic_rho pos_dis bos_dis'.split()))]
value_x_tick = [0, 0.05, 0.10, 0.15, 0.20, 0.25]


#sns.set_palette(sns.color_palette("Set2", 3))
os.makedirs('figures/img', exist_ok=True)
sns.set(font_scale=1.9)
plot = sns.relplot(comp_df, col='metric', x='error_prob', y='value', kind='line', errorbar=('se',2), row='channel', marker='o', markersize=8, hue='max_len', facet_kws=dict(margin_titles=True), legend=True)
sns.move_legend(plot, "upper center", bbox_to_anchor=(.5, 1.04), ncol=3, title="max message length", frameon=False,)
plot.set(xticks=value_x_tick)
plot.savefig("figures/img/test_plot_2.png", dpi=400) 
