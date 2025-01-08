import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import statistics
from collections import defaultdict


CONFIDENCE = 0.95
RESULTS_DIR = 'runs/'

print('')
results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
results_nn = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

data_long = defaultdict(list)
for d in os.listdir(RESULTS_DIR):
    if not d.startswith('channel') or not os.path.isdir(os.path.join('runs', d)):
        continue

    channel = d[d.index('_')+1:]
    for dd in os.listdir(os.path.join(RESULTS_DIR, d)):
        if not dd.startswith('error_prob_') or not os.path.isdir(os.path.join(RESULTS_DIR, d)):
            continue
        error_prob = float(dd.strip('error_prob_'))
        directory = os.path.join(RESULTS_DIR, d, dd)

        print(directory)
        for file in os.listdir(directory):
            if file.endswith('json'):
                max_len, _, seed = (int(item.strip('.json')) for item in file[:file.index('-')].split('_'))
                print(file, max_len, seed, error_prob)
                with open(os.path.join(directory, file)) as fp:
                    data = json.load(fp)

                r_nn = 'results-no-noise' if 'results-no-noise' in data else 'results'
                results[channel][max_len][error_prob]['accuracy'].append(data['results']['accuracy'])
                results[channel][max_len][error_prob]['embedding_alignment'].append(data['results']['embedding_alignment'])
                results[channel][max_len][error_prob]['topographic_rho'].append(data['results']['topographic_rho']*100)
                results[channel][max_len][error_prob]['pos_dis'].append(data['results']['pos_dis']*100)
                results[channel][max_len][error_prob]['bos_dis'].append(data['results']['bos_dis']*100)
                results[channel][max_len][error_prob]['unique_targets'].append(data['results']['unique_targets'])
                if 'unique_msg_no_noise' in data['results']:
                    results[channel][max_len][error_prob]['unique_msg_no_noise'].append(data['results']['unique_msg_no_noise'])
                else:
                    results[channel][max_len][error_prob]['unique_msg_no_noise'].append(data['results']['unique_msg'])
                results[channel][max_len][error_prob]['unique_msg'].append(data['results']['unique_msg'])

                avg_len = np.mean([m['message'].count(',') + 1 for m in data['messages']]).item()
                results[channel][max_len][error_prob]['avg_len'].append(avg_len)


                results_nn[channel][max_len][error_prob]['accuracy'].append(data[r_nn]['accuracy']/100)
                results_nn[channel][max_len][error_prob]['topographic_rho'].append(data[r_nn]['topographic_rho'])
                results_nn[channel][max_len][error_prob]['pos_dis'].append(data[r_nn]['pos_dis'])
                results_nn[channel][max_len][error_prob]['bos_dis'].append(data[r_nn]['bos_dis'])
                results_nn[channel][max_len][error_prob]['unique_targets'].append(data['results']['unique_targets'])
                if 'unique_msg_no_noise' in data['results']:
                    results_nn[channel][max_len][error_prob]['unique_msg_no_noise'].append(data['results']['unique_msg_no_noise'])
                else:
                    results_nn[channel][max_len][error_prob]['unique_msg_no_noise'].append(data['results']['unique_msg'])
                results_nn[channel][max_len][error_prob]['unique_msg'].append(data['results']['unique_msg'])
                results_nn[channel][max_len][error_prob]['avg_len'] = avg_len

                data_long['max_len'].append(max_len)
                data_long['channel'].append(channel)
                data_long['error_prob'].append(error_prob)
                data_long['noise'].append('no noise')
                data_long['accuracy'].append(data['results']['accuracy']/100)
                data_long['redundancy'].append(data['results']['redundancy_msg_lvl'])
                data_long['embedding_alignment'].append(data['results']['embedding_alignment']/100)
                data_long['topographic_rho'].append(data['results']['topographic_rho'])
                data_long['pos_dis'].append(data['results']['pos_dis'])
                data_long['bos_dis'].append(data['results']['bos_dis'])

                data_long['max_len'].append(max_len)
                data_long['channel'].append(channel)
                data_long['error_prob'].append(error_prob)
                data_long['noise'].append('noise')
                data_long['accuracy'].append(data[r_nn]['accuracy']/100)
                data_long['redundancy'].append(data[r_nn]['redundancy_msg_lvl'])
                data_long['embedding_alignment'].append(data[r_nn]['embedding_alignment']/100)
                data_long['topographic_rho'].append(data[r_nn]['topographic_rho'])
                data_long['pos_dis'].append(data[r_nn]['pos_dis'])
                data_long['bos_dis'].append(data[r_nn]['bos_dis'])

print(data_long)
print(pd.DataFrame(data_long))
os.makedirs('figures/data/', exist_ok=True)
data_long = pd.melt(
    pd.DataFrame(data_long),
    id_vars='max_len channel error_prob noise'.split(),
    value_vars=None, var_name='metric', value_name='value', ignore_index=True)
data_long.to_csv('figures/data/test_long.csv', index=False)

aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
aggregated_nn = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
for channel in results:
    for max_len in results[channel]:
        for error_prob in results[channel][max_len]:
            for metric in results[channel][max_len][error_prob]:
                output = {}
                for k, rd in zip(('noise', 'no_noise'), (results, results_nn)):
                    vals = results[channel][max_len][error_prob][metric]
                    if len(vals) == 1:
                        output[k] = (vals[0], None)
                        continue
                    mean = np.mean(vals)
                    if metric.startswith('unique') or metric == 'avg_len':
                        sd = statistics.stdev(vals)
                        output[k] = (mean, sd)
                    else:
                        stde = stats.sem(vals)
                        h = stde * stats.t.ppf((1+CONFIDENCE) / 2., len(vals)-1)
                        output[k] = (mean.item(), h.item())
                aggregated[channel][metric][max_len][error_prob] = output['noise']
                aggregated_nn[channel][metric][max_len][error_prob] = output['no_noise']


with open('figures/data/means-noise.json', 'w') as fp:
    json.dump(aggregated, fp, indent=4)

with open('figures/data/means-no-noise.json', 'w') as fp:
    json.dump(aggregated_nn, fp, indent=4)
