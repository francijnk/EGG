import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))
# TODO check cosine one last time and probably remove


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_distractors', '-d', type=int, required=True)
    parser.add_argument('--n_samples_train', type=int, default=128)
    parser.add_argument('--n_samples_eval_train', type=int, default=4)
    parser.add_argument('--n_samples_eval_test', type=int, default=16)
    parser.add_argument('--no_homonyms', action='store_true')
    parser.add_argument('--uk', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def path(*paths):
    return os.path.normpath(os.path.join(current_dir, *paths))


def extract_visa(args):
    directory = path(f'visa_dataset/{"UK" if args.uk else "US"}/')

    concepts = defaultdict(dict)
    homonyms = []
    for file in directory:
        if not os.path.isfile(path(file)):
            continue
        tree = ET.parse(path(directory, file))

        concept_category = tree.getroot().attrib['category']
        for concept in tree.getroot().iter('concept'):
            attribute_dict = defaultdict(int)
            for category in concept:
                attributes = [a.strip() for a in category.text.split()
                              if a.strip()]
                for a in attributes:
                    attribute_dict[a] = 1
            if concept.attrib['name'].find('_(') == -1:
                concept_name = concept.attrib['name']
            else:
                concept_name = concept.attrib['name'][:concept.attrib['name'].find('_(')]

            concepts['homonyms'][concept_name] = {
                'category': concept_category,
                'attributes': attribute_dict}

            # removing homonyms
            if concept_name not in concepts['no-homonyms'] \
                    and concept_name not in homonyms:
                concepts['no-homonyms'][concept_name] = {
                    'category': concept_category,
                    'attributes': attribute_dict}
            elif concept_name in homonyms:
                print('skipping', concept_name)
            else:
                del concepts['no-homonyms'][concept_name]
                homonyms.append(concept_name)
                print('deleting', concept_name)

    for ds, concept_dicts in concepts.items():

        all_attributes = {
            k: None for cd in concept_dicts.values() for k in cd['attributes']
        }

        output_dict = {
            'concept': list(concept_dicts.keys()),
            'category': [cd['category'] for cd in concept_dicts.values()]
        }

        # remove attributes, which do not differ among the concepts
        for a in all_attributes:
            a_values = [cd['attributes'][a] for cd in concept_dicts.values()]
            if len(set(a_values)) > 1:
                output_dict[a] = a_values

        output_path = path(f'visa-{ds}.csv')
        df = pd.DataFrame(output_dict)
        df.to_csv(output_path, index=False)

        print(f'visa-{ds}')
        print('number of concepts:', len(df))
        print('number of attributes:', len(df.columns) - 2)
        print('')


def sample(data, n_distractors, n_samples):
    n_features = data.shape[1] - 1

    concepts = np.array(data.iloc[:, 1:].values, dtype=np.int8)
    mapping, category_ids = np.unique(data.iloc[:, 0], return_inverse=True)
    positions = np.arange(n_distractors + 1)

    input_features = np.zeros(
        (len(data) * n_samples, n_distractors + 1, n_features),
        dtype=np.int8)
    categories = np.empty_like(input_features[..., 0])
    labels = np.empty_like(input_features[:, 0, 0])

    # cos_sims = cosine_similarity(concepts)

    for concept_i in range(len(data)):
        candidate_ids = np.delete(np.arange(len(data)), concept_i, axis=0)
        # candidate_probs = None
        # candidate_probs = np.delete(cos_sims[concept_i], concept_i, axis=0)
        # candidate_probs += np.clip(
        #     np.quantile(candidate_probs, q=0.10),
        #     a_min=1e-6, a_max=None,
        # )
        # candidate_pr^obs[candidate_probs > 0],
        # q=0.,
        # )  # to ensure all probabilities are positive
        # candidate_probs **= 0.5
        # candidate_probs /= candidate_probs.sum()
        candidate_probs = None
        # print(candidate_probs.max(), candidate_probs.min(), candidate_probs.mean())
        # print(candidate_probs[:20])
        # print(candidate_probs.shape, candidate_probs.sum())
        assert concept_i not in candidate_ids

        for sample_j in range(n_samples):
            idx = concept_i * n_samples + sample_j
            sample = np.empty_like(input_features[0])

            sampled_ids = np.random.choice(
                candidate_ids,
                size=n_distractors,
                p=candidate_probs,
                replace=False,
            )
            concept_pos = np.random.randint(0, n_distractors + 1)
            sampled_pos = positions[positions != concept_pos]
            sample[concept_pos] = concepts[concept_i]
            sample[sampled_pos] = concepts[sampled_ids]

            input_features[idx] = sample
            labels[idx] = concept_pos
            categories[idx, concept_pos] = category_ids[concept_i]
            categories[idx, sampled_pos] = category_ids[sampled_ids]

    size = input_features.shape
    reshaped = input_features.reshape(size[0] * size[1], -1, order='C')
    _, unique = np.unique(reshaped, axis=0, return_inverse=True)
    _sorted = np.sort(unique.reshape(size[:2]))
    print(
        'Unique concept sets:',
        np.unique(_sorted, axis=0).shape[0],
        '/', input_features.shape[0])
    mapping = mapping.astype('U')
    mapping = np.array(
        (mapping,),
        dtype=[('category', mapping.dtype, mapping.shape)],
    )

    categories = np.stack([
        np.array((sample,), dtype=[('category', np.int8, n_distractors + 1,)])
        for sample in categories
    ])
    return input_features, labels, categories, mapping


def export_visa(args, visa):
    np.random.seed(args.seed)

    features = visa.iloc[:, 1:]
    labels = visa.iloc[:, 0]

    print('Total concepts:', features.shape[0])
    print('Total features:', features.shape[1] - 1)

    # 80% train, 20% test
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.2)

    print('Train concepts:', len(train_features))
    print('Test concepts:', len(test_features))

    train, train_targets, train_cat, train_mapping = sample(
        train_features, args.n_distractors, args.n_samples_train,
    )
    eval_train, eval_train_targets, eval_train_cat, eval_train_mapping = sample(
        train_features, args.n_distractors, args.n_samples_eval_train,
    )
    eval_test, eval_test_targets, eval_test_cat, eval_test_mapping = sample(
        test_features, args.n_distractors, args.n_samples_eval_test,
    )

    print('Training samples:', len(train))
    print('Evaluation samples (train):', len(eval_train))
    print('Evaluation samples (test):', len(eval_test))

    fname = f'visa-{args.n_distractors + 1}-{args.n_samples_train}.npz'
    np.savez_compressed(
        path('..', 'input_data', fname),
        train=train,
        train_targets=train_targets,
        train_attributes=train_cat,
        train_attribute_mapping=train_mapping,
        eval_train=eval_train,
        eval_train_targets=eval_train_targets,
        eval_train_attributes=eval_train_cat,
        eval_train_attribute_mapping=eval_train_mapping,
        eval_test=eval_test,
        eval_test_targets=eval_test_targets,
        eval_test_attributes=eval_test_cat,
        eval_test_attribute_mapping=eval_test_mapping,
        allow_pickle=False,
    )

    print(f'data saved to .../data/input_data/{fname}')


def plot_visa(visa, fpath):
    visa = visa.copy().sort_values(by=['category'])
    features = visa.values[:, 2:]
    categories = visa.values[:, 1]
    cat_labels, cat_ids = np.unique(categories, return_index=True)

    fig, ax = plt.subplots(sharey=True, figsize=(5 / 2.54, 5 / 2.54), dpi=3200)
    [ax.spines[s].set(linewidth=0.4) for s in ax.spines]
    ax.spy(features, aspect='equal', alpha=(features).astype(float))
    ax.set_facecolor('white')
    ax.tick_params(
        axis='both',
        left=False, right=False,
        bottom=False, top=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
        labeltop=False,
    )

    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.hlines(cat_ids[1:] - 0.5, xmin=xmin, xmax=xmax, colors='grey', alpha=0.25, zorder=-1, linewidths=0.4)
    plt.tight_layout()
    plt.savefig(fpath, format='pdf', dpi=None, backend='pgf', pad_inches=0.2 / 72, bbox_inches='tight')
    plt.close()


def plot_pair_counts(pair_counts, fpath):
    fig, ax = plt.subplots()
    data = pair_counts.copy()
    data[data == -1] = 0
    ax.imshow(data)
    plt.tight_layout()
    plt.savefig(fpath, format='pdf', dpi=72, pad_inches=0)
    plt.close()


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(args.seed)
    os.makedirs(path('../input_data/'), exist_ok=True)

    extract_visa(args)
    visa_fname = 'visa-no-homonyms.csv' if args.no_homonyms else 'visa-homonyms.csv'
    visa = pd.read_csv(path(visa_fname))

    export_visa(args, visa)
    plot_visa(visa, path('visa-distribution.pdf'))
