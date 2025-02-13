import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import xml.etree.ElementTree as ET


current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_distractors', '-d', type=int, required=True)
    parser.add_argument('--n_samples_train', type=int, default=128)
    parser.add_argument('--n_samples_train_eval', type=int, default=8)
    parser.add_argument('--n_samples_test', type=int, default=16)
    parser.add_argument('--us', action='store_true')
    args = parser.parse_args()
    return args


def extract_visa(args):
    if args.us:
        directory = 'visa_dataset/UK'
    else:
        directory = 'visa_dataset/US'

    concepts_no_homo = {}
    concepts_w_homo = {}
    homonyms = []
    for file in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, file)):
            continue
        tree = ET.parse(os.path.join(directory, file))

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

            concepts_w_homo[concept_name] = {
                'category': concept_category,
                'attributes': attribute_dict}

            # removing homonyms
            if concept_name not in concepts_no_homo and concept_name not in homonyms:
                concepts_no_homo[concept_name] = {
                    'category': concept_category,
                    'attributes': attribute_dict}
            elif concept_name in homonyms:
                print('skipping', concept_name)
            else:
                del concepts_no_homo[concept_name]
                homonyms.append(concept_name)
                print('deleting', concept_name)

    for ds, concepts in zip(('homonyms', 'no-homonyms'), (concepts_w_homo, concepts_no_homo)):
        all_attributes = set([k for cd in concepts.values() for k in cd['attributes'].keys()])
        output_dict = {
            'concept': list(concepts.keys()),
            'category': [cd['category'] for cd in concepts.values()]}

        # remove attributes, which do not differ among the concepts
        for attribute in all_attributes:
            attribute_values = [cd['attributes'][attribute] for cd in concepts.values()]
            if len(set(attribute_values)) > 1:
                output_dict[attribute] = attribute_values
        output = pd.DataFrame(output_dict)

        output_path = os.path.join(current_dir, 'visa', f'visa-{ds}.csv')
        output.to_csv(output_path, index=False)

        print(f'visa-{ds}')
        print('number of concepts:', len(output))
        print('number of attributes:', len(output.columns) - 2)
        print('')


def sample(data_concepts, n_distractors, n_samples):
    n_features = data_concepts.shape[1] - 1  # exclude the category column

    data_categories = data_concepts.iloc[:, 0]  # .astype('category')
    # data_categories = data_categories.cat.set_categories(
    #     new_categories=data_categories.unique(),
    #     ordered=True)
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
                categories[f'distr_{k}_category'].append(distr_cat)

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
            (col, np.int64)
            for col in category_df.columns
        ])
    )

    return input_data, labels, categories


def export_visa(args):
    np.random.seed(42)
    visa_fpath = os.path.join(current_dir, 'visa-homonyms.csv')
    visa = pd.read_csv(visa_fpath)

    features = visa.iloc[:, 1:]
    textlabels = visa.iloc[:, :1]

    print('Total concepts:', features.shape[0])
    print('Total features:', features.shape[1] - 1)

    # 80% train, 20% test
    train_features, test_features, train_textlabels, test_labels = train_test_split(
        features, textlabels, test_size=0.2)

    print('Train concepts:', len(train_features))
    print('Test concepts:', len(test_features))

    train, train_labels, train_categories = sample(
        train_features, args.n_distractors, args.n_samples_train)
    train_eval, train_eval_labels, train_eval_categories = sample(
        train_features, args.n_distractors, args.n_samples_train_eval)
    test, test_labels, test_categories = sample(
        test_features, args.n_distractors, args.n_samples_test)

    print('Training samples:', len(train_labels))
    print('Evaluation samples (train):', len(train_eval_labels))
    print('Evaluation samples (test):', len(test_labels))

    npz_fname = f"visa-{args.n_distractors}-{args.n_samples_train}.npz"
    npz_fpath = os.path.join(current_dir, '..', 'input_data', npz_fname)
    np.savez_compressed(
        npz_fpath,
        train=train, train_labels=train_labels, train_attributes=train_categories,
        valid=test, valid_labels=test_labels, valid_attributes=test_categories,
        test=test, test_labels=test_labels, test_attributes=test_categories,
        train_eval=train_eval, train_eval_labels=train_eval_labels,
        train_eval_attributes=train_eval_categories,
        n_distractors=args.n_distractors)


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(42)

    os.makedirs(os.path.join(current_dir, 'visa'), exist_ok=True)
    visa_csv = os.path.join(current_dir, 'visa-homonyms.csv')
    if not os.path.isfile(visa_csv):
        extract_visa(args)

    export_visa(args)
