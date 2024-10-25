import json


with open('figures/data/means-noise.json') as fp:
    data = json.load(fp)


key2desc = {
    'accuracy': 'Acc.',
    'embedding_alignment': 'Emb. al.',
    'topographic_rho': 'Top. $\\rho$',
    'pos_dis': 'Pos. dis',
    'bos_dis': 'Bos. dis',
    #'unique_msg_no_noise': 'Unique messages sent',
    #'unique_msg': 'Unique messages received',
}


for metric in data:
    if metric not in key2desc:
        continue
    line = [key2desc[metric]]
    for max_len in data[metric]:
        if max_len != '2':
            continue
        if metric == 'f1':
            continue
        for noise_level in data[metric][max_len]:
            mean, ci = data[metric][max_len][noise_level]
            mean, ci = mean/100, ci/100
            if metric == 'accuracy':
                pass
                # mean, ci = 100 * mean, 100 * ci
            line.extend((f"{round(mean, 4):.3f}", f"{round(ci,4):.3f}"))
    print(' & '.join(line) + ' \\\\')


print('n cells:', len(line))
