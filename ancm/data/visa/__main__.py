import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_distractors', '-d', type=int, required=True)
    parser.add_argument('--n_samples_train', type=int, default=128)
    parser.add_argument('--n_samples_train_eval', type=int, default=4)
    parser.add_argument('--n_samples_test', type=int, default=16)
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
                              if a.strip()]  # and not a.startswith('beh')]
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

    concepts = np.array(data.iloc[:, 1:].values, dtype=np.int64)
    categories, category_ids = np.unique(data.iloc[:, 0], return_inverse=True)
    positions = np.arange(n_distractors + 1)

    sample_features = np.zeros(
        (len(data) * n_samples, n_distractors + 1, n_features),
        dtype=np.int64)
    sample_categories = np.empty_like(sample_features[..., 0])
    target_positions = np.empty_like(sample_features[:, 0, 0])

    for concept_i in range(len(data)):
        candidate_ids = np.delete(np.arange(len(data)), concept_i, axis=0)
        assert concept_i not in candidate_ids

        for sample_j in range(n_samples):
            idx = concept_i * n_samples + sample_j
            sample = np.empty_like(sample_features[0])

            sampled_ids = np.random.choice(
                candidate_ids,
                size=n_distractors,
                replace=False)
            concept_pos = np.random.randint(0, n_distractors + 1)
            sampled_pos = positions[positions != concept_pos]
            sample[concept_pos] = concepts[concept_i]
            sample[sampled_pos] = concepts[sampled_ids]

            sample_features[idx] = sample
            target_positions[idx] = concept_pos
            sample_categories[idx, concept_pos] = category_ids[concept_i]
            sample_categories[idx, sampled_pos] = category_ids[sampled_ids]

    sample_categories = np.array(
        (sample_categories,),
        dtype=[('category', np.int64, sample_categories.shape)])
    categories = np.char.array(categories, unicode=False)
    mapping = np.array(
        (categories,),
        dtype=[('category', categories.dtype, categories.shape)])

    return sample_features, sample_categories, target_positions, mapping


def _sample(data_concepts, n_distractors, n_samples):
    n_features = data_concepts.shape[1] - 1  # exclude the category column

    data_categories = data_concepts.iloc[:, 0]  # .astype('category')
    # data_categories = data_categories.cat.codes
    data_concepts = np.array(data_concepts.iloc[:, 1:].values, dtype='int')

    sample_sets, labels, categories = [], [], defaultdict(list)
    for concept_i in range(len(data_concepts)):
        target_category = data_categories.iloc[concept_i]  # data_concepts.iloc[concept_i,0]

        distractor_ids = np.delete(np.arange(len(data_concepts)), concept_i, axis=0)
        assert concept_i not in distractor_ids

        for sample_j in range(n_samples):
            target_pos = np.random.randint(0, n_distractors + 1)
            distractor_ind = np.random.choice(distractor_ids, size=n_distractors, replace=False)
            distractor_pos = [i for i in range(n_distractors + 1) if i != target_pos]

            sample_set = np.zeros((1, n_distractors + 1, n_features), dtype=np.int64)
            sample_set[0, target_pos] = data_concepts[concept_i]
            for distr_pos, distr_ind in zip(distractor_pos, distractor_ind):
                sample_set[0, distr_pos] = data_concepts[distr_ind]

            sample_sets.append(sample_set)
            labels.append(target_pos)
            # categories.append((category, *distractor_categories))

            categories['target_category'].append(target_category)
            for k, distr_ind in enumerate(distractor_ind):
                distr_cat = data_categories.iloc[distr_ind]
                categories[f'distr{k}_category'].append(distr_cat)

    input_data = np.vstack(sample_sets)

    labels = np.array(labels, dtype=np.int64)
    # categories = np.array(
    #    categories, dtype=np.dtype([('category', np.int64)]))

    # create a DataFrame & code categories as integers
    category_df = pd.DataFrame(categories, dtype='category')
    new_categories = pd.unique(category_df.values.ravel('K'))
    for col in category_df.columns:
        category_df[col] = category_df[col].cat.set_categories(
            new_categories=new_categories,
            ordered=True)
        category_df[col] = category_df[col].cat.codes

    # export to Numpy
    categories = np.array(category_df, dtype=np.int64)
    categories = np.array(
        list(map(tuple, categories)),
        dtype=np.dtype([
            (col, np.int64) for col in category_df.columns
        ])
    )

    return input_data, labels, categories


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

    train, train_cat, train_targets, train_mapping = sample(
        train_features, args.n_distractors, args.n_samples_train,
    )
    eval_train, eval_train_cat, eval_train_targets, eval_train_mapping = sample(
        train_features, args.n_distractors, args.n_samples_train_eval,
    )
    eval_test, eval_test_cat, eval_test_targets, eval_test_mapping = sample(
        test_features, args.n_distractors, args.n_samples_test,
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
    # fig = plt.figure(figsize=(7, 4))
    visa = visa.copy().sort_values(by=['category'])
    features = visa.values[:, 2:]
    categories = visa.values[:, 1]
    cat_labels, cat_ids = np.unique(categories, return_index=True)
    # cat_ticks = cat_ids.tolist() + [len(visa)]
    # cat_ticks = [
    #     (cat_ticks[i] + cat_ticks[i + 1]) / 2
    #     for i in range(len(cat_ids))
    # ]
    # print(features.shape, features[0])
    # print(categories.shape, categories[:10])
    # print(cat_ids.shape, cat_ids[:10])
    # print(features, type(features), features.dtype)

    fig, ax = plt.subplots(sharey=True)
    ax.spy(features, aspect='equal', alpha=(features).astype(float))
    ax.set_facecolor('white')
    ax.tick_params(
        axis='both',
        # left=True, right=True,
        left=False, right=False,
        bottom=False, top=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
        labeltop=False,
    )
    # ax.set_ylim(bottom=(len(visa) - 1), top=0)
    # cat_sep = (cat_ids - 0.5).tolist() + [len(visa) - 0.5]
    # ax.set_yticks(cat_sep)
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ax.hlines(cat_ids - 0.5, xmin=xmin, xmax=xmax, colors='grey', alpha=0.25, zorder=-1, linewidths=0.75)
    plt.tight_layout()
    plt.savefig(fpath, format='pdf', dpi=None, pad_inches=0.01, bbox_inches='tight')
    plt.close()


def plot_pair_counts(pair_counts, fpath):
    fig, ax = plt.subplots()
    data = pair_counts.copy()
    data[data == -1] = 0
    ax.imshow(data)
    plt.tight_layout()
    plt.savefig(fpath, format='pdf', dpi=None, pad_inches=0)
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
