import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import statistics
from collections import defaultdict


confidence = 0.95


print('')
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
results_nn = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

data_long = defaultdict(list)
for d in os.listdir('runs/'):
    directory = os.path.join('runs', d)
    if not d.startswith('erasure') or not os.path.isdir(directory):
        continue
    erasure_pr = float(d.strip('erasure_pr_'))


    for file in os.listdir(directory):
        if file.endswith('json'):
            max_len, seed = (int(item.strip('.json')) for item in file[:file.index('-')].split('_'))

            with open(os.path.join(directory, file)) as fp:
                data = json.load(fp)

            r_nn = 'results-no-noise' if 'results-no-noise' in data else 'results'
            results[max_len][erasure_pr]['accuracy'].append(data['results']['accuracy'])
            results[max_len][erasure_pr]['embedding_alignment'].append(data['results']['embedding_alignment'])
            results[max_len][erasure_pr]['topographic_rho'].append(data['results']['topographic_rho']*100)
            results[max_len][erasure_pr]['pos_dis'].append(data['results']['pos_dis']*100)
            results[max_len][erasure_pr]['bos_dis'].append(data['results']['bos_dis']*100)
            results[max_len][erasure_pr]['unique_targets'].append(data['results']['unique_targets'])
            results[max_len][erasure_pr]['unique_msg'].append(data['results']['unique_msg'])
            if 'unique_msg_no_noise' in data['results']:
                results[max_len][erasure_pr]['unique_msg_no_noise'].append(data['results']['unique_msg_no_noise'])
            else:
                results[max_len][erasure_pr]['unique_msg_no_noise'] = results[max_len][erasure_pr]['unique_msg']


            results_nn[max_len][erasure_pr]['accuracy'].append(data[r_nn]['accuracy']/100)
            results_nn[max_len][erasure_pr]['topographic_rho'].append(data[r_nn]['topographic_rho'])
            results_nn[max_len][erasure_pr]['pos_dis'].append(data[r_nn]['pos_dis'])
            results_nn[max_len][erasure_pr]['bos_dis'].append(data[r_nn]['bos_dis'])
            results_nn[max_len][erasure_pr]['unique_targets'].append(data['results']['unique_targets'])
            results_nn[max_len][erasure_pr]['unique_msg'].append(data['results']['unique_msg'])
            if 'unique_msg_no_noise' in data['results']:
                results_nn[max_len][erasure_pr]['unique_msg_no_noise'].append(data['results']['unique_msg_no_noise'])
            else:
                results_nn[max_len][erasure_pr]['unique_msg_no_noise'] = results[max_len][erasure_pr]['unique_msg']

            data_long['max_len'].append(max_len)
            data_long['erasure_pr'].append(erasure_pr)
            data_long['noise'].append('no noise')
            data_long['accuracy'].append(data['results']['accuracy']/100)
            data_long['embedding_alignment'].append(data['results']['embedding_alignment']/100)
            data_long['topographic_rho'].append(data['results']['topographic_rho'])
            data_long['pos_dis'].append(data['results']['pos_dis'])
            data_long['bos_dis'].append(data['results']['bos_dis'])
            #data_long['unique_msg'].append(data['results']['unique_msg'])
            #data_long['unique_targets'].append(data['results']['unique_targets'])
            #if 'unique_msg_no_noise' in data['results']:
            #    data_long['unique_msg_no_noise'].append(data['results']['unique_msg_no_noise'])
            #else:
            #    data_long['unique_msg_no_noise'].append(data['results']['unique_msg'])

            data_long['max_len'].append(max_len)
            data_long['erasure_pr'].append(erasure_pr)
            data_long['noise'].append('noise')
            data_long['accuracy'].append(data[r_nn]['accuracy']/100)
            data_long['embedding_alignment'].append(data[r_nn]['embedding_alignment']/100)
            data_long['topographic_rho'].append(data[r_nn]['topographic_rho'])
            data_long['pos_dis'].append(data[r_nn]['pos_dis'])
            data_long['bos_dis'].append(data[r_nn]['bos_dis'])
#print(data_long)
#data_long = sorted(data_long, key=lambda x: int(x['max_len']))
data_long = pd.melt(
    pd.DataFrame(data_long),
    id_vars='max_len erasure_pr noise'.split(),
    value_vars=None, var_name='metric', value_name='value', ignore_index=True)
data_long.to_csv('figures/data/test_long.csv', index=False)

aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
aggregated_nn = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for max_len in results:
    for erasure_pr in results[max_len]:
        for metric in results[max_len][erasure_pr]:
            output = {}
            for k, rd in zip(('noise', 'no_noise'), (results, results_nn)):
                vals = results[max_len][erasure_pr][metric]
                if len(vals) == 1:
                    output[k] = (vals[0], None)
                    continue
                mean = np.mean(vals)
                if metric.startswith('unique'):
                    sd = statistics.stdev(vals)
                    output[k] = (mean, sd)
                else:
                    stde = stats.sem(vals)
                    h = stde * stats.t.ppf((1+confidence) / 2., len(vals)-1)
                    output[k] = (mean.item(), h.item())
            aggregated[metric][max_len][erasure_pr] = output['noise']
            aggregated_nn[metric][max_len][erasure_pr] = output['no_noise']


os.makedirs('figures/data/', exist_ok=True)
with open('figures/data/means-noise.json', 'w') as fp:
    json.dump(aggregated, fp, indent=4)

with open('figures/data/means-no-noise.json', 'w') as fp:
    json.dump(aggregated_nn, fp, indent=4)
