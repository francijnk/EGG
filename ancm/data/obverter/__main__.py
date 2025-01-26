import os
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from ancm.data.obverter.render import (
    colors,
    object_types,
    render_scenes,
    get_object_fname,
)


current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_distractors', '-d', type=int, required=True)
    parser.add_argument('--n_samples_train', type=int, default=300)
    parser.add_argument('--n_samples_test', type=int, default=20)
    parser.add_argument('--n_img', type=int, default=150)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def load_image(color, obj_type, resolution, idx=None, n_img=None):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    assets_dir = os.path.join(current_dir, 'assets')
    fname = get_object_fname(color, obj_type, idx, n_img)
    color, obj_type, _, xpos, ypos, rotation = \
        fname.split('.')[0].split('_')

    image = Image.open(os.path.join(assets_dir, fname)).convert("RGB")
    image = transform(image)
    image = image.numpy()

    return image, (color, obj_type, int(xpos), int(ypos), int(rotation))


def pick_random_color(exclude=None):
    if exclude is None:
        exclude = []
    elif not isinstance(exclude, list):
        exclude = [exclude]

    available_colors = [
        item for item in colors.keys()
        if item not in exclude]

    return random.choice(available_colors)


def pick_random_shape(exclude=None):
    if exclude is None:
        exclude = []
    elif not isinstance(exclude, list):
        exclude = [exclude]

    available_object_types = [
        item for item in object_types
        if item not in exclude]

    return random.choice(available_object_types)


def export_input_data(n_distractors, n_samples, n_img, resolution):
    sample_sets, labels, attributes = [], [], defaultdict(list)

    n_same_shape = int(0.15 * n_samples)
    n_same_color = int(0.1 * n_samples)

    for shape in tqdm(object_types):
        for color in colors:
            for i in range(n_samples):
                image_idx = i % n_img
                target_image, (color, shape, xpos, ypos, rotation) = \
                    load_image(color, shape, resolution, image_idx)

                target_pos = np.random.randint(0, n_distractors + 1)
                distractor_pos = [i for i in range(n_distractors + 1) if i != target_pos]

                sample_set = np.zeros(
                    (1, n_distractors + 1, *target_image.shape), dtype=np.int64)
                sample_set[0, target_pos] = target_image

                used_combinations = [(shape, color)]
                for j in range(n_distractors):
                    if i < n_same_shape:
                        # same shape
                        exclude = [item[1] for item in used_combinations]
                        distr_shape = shape
                        distr_color = pick_random_color(exclude)
                    elif i - n_same_shape < n_same_color:
                        # same color
                        exclude = [item[0] for item in used_combinations]
                        distr_color = color
                        distr_shape = pick_random_shape(exclude)
                    else:
                        # random choice
                        distr_shape = pick_random_shape()
                        exclude = [
                            item[1] for item in used_combinations
                            if item[0] == distr_shape]
                        distr_color = pick_random_color(exclude=color)

                    used_combinations.append((distr_shape, distr_color))

                    distr_image, (_, _, distr_xpos, distr_ypos, distr_rot) = \
                        load_image(distr_color, distr_shape, resolution, image_idx)
                    sample_set[0, distractor_pos[j]] = distr_image

                    attributes[f'distr_{j}_color'].append(distr_color)
                    attributes[f'distr_{j}_shape'].append(distr_shape)
                    attributes[f'distr_{j}_xpos'].append(distr_xpos)
                    attributes[f'distr_{j}_ypos'].append(distr_ypos)
                    attributes[f'distr_{j}_rotation'].append(distr_rot)

                sample_sets.append(sample_set)
                labels.append(target_pos)
                attributes['target_color'].append(color)
                attributes['target_shape'].append(shape)
                attributes['target_xpos'].append(xpos)
                attributes['target_ypos'].append(ypos)
                attributes['target_rotation'].append(rotation)

    input_data = np.vstack(sample_sets)
    labels = np.array(labels, dtype=np.int64)

    # create a DataFrame & code each category as integers
    attribute_df = pd.DataFrame(attributes)
    for col in attribute_df.columns:
        attribute_df[col] = attribute_df[col].astype('category')
        attribute_df[col] = attribute_df[col].cat.set_categories(
            new_categories=attribute_df[col].unique(),
            ordered=True)
        attribute_df[col] = attribute_df[col].cat.codes

    attributes = np.array(attribute_df, dtype=np.int64)
    attributes = np.array(
        list(map(tuple, attributes)),
        dtype=np.dtype([
            (attribute, np.int64)
            for attribute in attribute_df.columns
        ])
    )

    return input_data, labels, attributes


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    render_scenes(args)

    train, train_labels, train_attributes = export_input_data(
        args.n_distractors, args.n_samples_train, args.n_img, args.resolution)
    test, test_labels, test_attributes = export_input_data(
        args.n_distractors, args.n_samples_test, args.n_img, args.resolution)

    print(train.shape)
    print('Number of shapes:', len(object_types))
    print('Number of colors:', len(colors.keys()))
    print('Train samples:', len(train_labels))
    print('Test/Eval samples:', len(test_labels))

    npz_fpath = os.path.join(
        current_dir, '..', 'input_data',
        f'obverter-{args.n_distractors + 1}-{args.n_samples_train}-{args.resolution}.npz')

    np.savez_compressed(
        npz_fpath,
        train=train, train_labels=train_labels, train_attributes=train_attributes,
        valid=test, valid_labels=test_labels, valid_attributes=test_attributes,
        test=test, test_labels=test_labels, test_attributes=test_attributes,
        n_distractors=args.n_distractors)
