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
    parser.add_argument('--n_samples_train', type=int, default=100)
    parser.add_argument('--n_samples_val', type=int, default=20)
    parser.add_argument('--n_samples_test', type=int, default=20)
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
            check = False
            for val in attribute_values:
                if val != attribute_values[0]:
                    check = True
            if check:
                output_dict[attribute] = attribute_values
        output = pd.DataFrame(output_dict)

        output_path = os.path.join(current_dir, 'visa', f'visa-{ds}.csv')
        output.to_csv(output_path, index=False)

        print(f'visa-{ds}')
        print('number of concepts:', len(output))
        print('number of attributes:', len(output.columns) - 2)
        print('')


def reshape(data_concepts, n_distractors, n_features, n_samples, data_distractors=None):
    if data_distractors is None:  # if no df for distractors is provided, use df for concepts
        data_distractors = data_concepts

    labels = []
    # categories = data_concepts.iloc[:,0]

    data_concepts = np.array(data_concepts.iloc[:, 1:].values, dtype='int')
    sample_sets = []
    for concept_i in range(len(data_concepts)):
        # category = categories.iloc[concept_i]  # data_concepts.iloc[concept_i,0]
        distractors_i = np.delete(data_concepts, concept_i, axis=0)
        for sample_j in range(n_samples):

            idx = np.random.randint(distractors_i.shape[0], size=n_distractors)
            distractors_ij = distractors_i[idx]

            target_pos = np.random.randint(0, n_distractors + 1)
            distractor_pos = [i for i in range(n_distractors + 1) if i != target_pos]

            sample_set = np.zeros((1, n_distractors + 1, n_features), dtype=np.int64)
            sample_set[0, target_pos] = data_concepts[concept_i]
            for distr_k, distr_pos in enumerate(distractor_pos):
                sample_set[0, distr_pos] = distractors_ij[distr_k]

            labels.append(target_pos)
            sample_sets.append(sample_set)

    data_reshaped = np.vstack(sample_sets)

    return data_reshaped, labels


def export_visa(args):
    np.random.seed(42)
    visa_fpath = os.path.join(current_dir, 'visa-homonyms.csv')
    visa = pd.read_csv(visa_fpath)

    features = visa.iloc[:, 1:]
    textlabels = visa.iloc[:, :1]
    n_features = features.shape[1] - 1  # exclude the category column

    # divide 70% for train, 15% test and val
    train_features, temp_features, train_textlabels, temp_labels = train_test_split(
        features, textlabels, test_size=0.3)
    val_features, test_features, val_textlabels, test_textlabels = train_test_split(
        temp_features, temp_labels, test_size=0.5)

    train, train_labels = reshape(
        train_features, args.n_distractors, n_features, args.n_samples_train, features)
    val, val_labels = reshape(
        val_features, args.n_distractors, n_features, args.n_samples_val, features)
    test, test_labels = reshape(
        test_features, args.n_distractors, n_features, args.n_samples_test, features)

    print('train:', len(train_labels))
    print('val:', len(val_labels))
    print('test:', len(test_labels))

    npz_fname = f"visa-{args.n_distractors}-{args.n_samples_train}.npz"
    npz_fpath = os.path.join(current_dir, '..', 'input_data', npz_fname)
    np.savez_compressed(
        npz_fpath,
        train=train, train_labels=train_labels,
        valid=val, valid_labels=val_labels,
        test=test, test_labels=test_labels,
        n_distractors=args.n_distractors)


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(42)

    os.makedirs(os.path.join(current_dir, 'visa'), exist_ok=True)
    visa_csv = os.path.join(current_dir, 'visa-homonyms.csv')
    if not os.path.isfile(visa_csv):
        extract_visa(args)

    export_visa(args)
