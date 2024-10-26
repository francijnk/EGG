import json
import numpy as np
import statistics
from collections import defaultdict
import os

with open('figures/data/means-noise.json') as fp:
    data = json.load(fp)


results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
data_long = defaultdict(list)
for d in os.listdir('runs/'):
    directory = os.path.join('runs', d)
    if not d.startswith('erasure') or not os.path.isdir(directory):
        continue
    erasure_pr = float(d.strip('erasure_pr_'))
    for file in os.listdir(directory):
        if file.endswith('json'):
            max_len, seed = (int(item.strip('.json')) for item in file[:file.index('-')].split('_'))
            if max_len==10:
                continue
            with open(os.path.join(directory, file)) as fp:
                data = json.load(fp)

            results[erasure_pr][max_len]['unique_targets'].append(data['results']['unique_targets'])
            results[erasure_pr][max_len]['unique_msg'].append(data['results']['unique_msg'])
            if 'unique_msg_no_noise' in data['results']:
                results[erasure_pr][max_len]['unique_msg_no_noise'].append(data['results']['unique_msg_no_noise'])
            else:
                results[erasure_pr][max_len]['unique_msg_no_noise'].append(data['results']['unique_msg'])
            avg_len = np.mean([m['message'].count(',')  for m in data['messages']]).item()
            results[erasure_pr][max_len]['avg_len'].append(avg_len)

cols = []
for p in results:
    for ml in sorted(results[p], key=int):
        for v in  results[p][ml]:
            if v != 'unique_targets':
                cols.append((ml, v))
    break
print(cols)

for p in results:
    for ml in sorted(results[p], key=int):
        print(results[p][ml]['unique_targets'])


for p in results:
    fields = [
        f'{p:.2f}',
    ]

    for ml in sorted(results[p], key=int):
        res = results[p][ml]
        for key in ('unique_msg_no_noise', 'unique_msg', 'avg_len'):
            vals = res[key]
            if len(vals) == 0:
                mean, sd = '--', '--'
            elif len(vals) == 1:
                mean = vals[0]
                sd = '--'
            else:
                mean = np.mean(vals)
                sd = statistics.stdev(vals)
            # print(p, ml, mean, sd)
            fields.extend([mean, sd])

    print(' & '.join([f'{x:.1f}' if isinstance(x, float) else str(x) for x in fields]), '\\\\')


