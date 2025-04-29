import os
import gc
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from itertools import product, cycle
from collections import defaultdict
from torchvision import transforms

from .render import (
    colors,
    shapes,
    render_scenes,
    get_object_fname,
)

attribute_keys = ('shape', 'color', 'x', 'y')


def path(*paths):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, *paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--n_distractors', '-d', type=int, required=True)
    parser.add_argument('--n_samples_train', type=int, default=256)
    parser.add_argument('--n_samples_eval_train', type=int, default=12)
    parser.add_argument('--n_samples_eval_test', type=int, default=36)
    parser.add_argument('--n_images', type=int, default=20)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_image(shape, color, x=None, y=None, resolution=128):
    fname = get_object_fname(shape, color, x, y)
    if fname is None:
        print(shape, color, x, y, 'not found!')
    assert fname is not None

    shape, color, x, y, _, _ = fname.split('.')[0].split('_')

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    image = Image.open(path('assets_128', fname)).convert("RGB")
    image = transform(image)
    image = image.numpy()

    return image, (shape, color, int(x), int(y))


def get_select_random(args):
    tf = cycle([True, False])

    def select_random(used_combos, object_tuples, mode=None):
        # mode = mode if mode is not None else sample_mode()
        if mode is None:
            mode = 'unique_shape_color' if next(tf) else 'random'

        if mode == 'unique_shape_color':
            # combinations of shape and color are unique in each sample
            exclude = [tup[:2] for tup in used_combos]
            available = [
                (*tup, x, y) for tup, x, y
                in product(object_tuples, [0, 1], [0, 1])
                if tup[:2] not in exclude
            ]
        else:  # no constraints: shape, color and shade not necessarily unique
            available = [
                (*tup, x, y) for tup, x, y
                in product(object_tuples, [0, 1], [0, 1])
                if (*tup, x, y) not in used_combos
            ]

        return random.choice(available), mode

    return select_random


def export_input_data(object_tuples, args, select, n_samples):
    n_distractors = args.n_distractors
    positions = np.arange(n_distractors + 1)

    resolution = args.resolution
    size = (len(object_tuples) * n_samples, n_distractors + 1)
    target_positions = np.empty(size[0], dtype=np.int8)

    attributes = np.empty((*size, len(attribute_keys)), dtype=np.int8)
    shared_attributes = {k: [] for k in attribute_keys}
    sampling_mode = []
    input_features = np.empty(
        (*size, 3, resolution, resolution),
        dtype=np.float16,
    )

    mapping = defaultdict(lambda: (lambda x: x))
    mapping.update({
        'shape': lambda x: shapes.index(x),
        'color': lambda x: colors.index(x),
    })

    pbar = tqdm(total=size[0], desc='Sampling distractors')
    for t, t_tuple in enumerate(object_tuples):
        for s in range(n_samples):
            idx = t * n_samples + s
            t_image, t_attributes = load_image(
                *t_tuple,
                random.choice([0, 1]),
                random.choice([0, 1]),
                resolution=resolution,
            )

            t_pos = np.random.randint(0, n_distractors + 1)
            d_pos = positions[positions != t_pos]

            target_positions[idx] = t_pos
            input_features[idx, t_pos] = t_image
            attributes[idx, t_pos] = [
                mapping[k](v) for k, v in zip(attribute_keys, t_attributes)
            ]

            used_combos, mode = [t_attributes], None
            for d in range(n_distractors):
                d_attributes, mode = select(used_combos, object_tuples, mode)
                d_image, d_attributes = load_image(
                    *d_attributes,
                    resolution=resolution)
                used_combos.append(d_attributes)

                input_features[idx, d_pos[d]] = d_image
                attributes[idx, d_pos[d]] = [mapping[k](v) for k, v in zip(attribute_keys, d_attributes)]

            for k, key in enumerate(attribute_keys):
                shared_attributes[key].append(
                    sum(1 for c in used_combos[1:] if c[k] == t_attributes[k])
                )

            if input_features.shape[0] < 2000:
                sampling_mode.append(mode)

            pbar.update(1)
            gc.collect()

    pbar.close()

    for attribute, counts in shared_attributes.items():
        print(
            f'Avg number of distractors sharing {attribute} with '
            'the target:', np.round(np.mean(counts), 2)
        )

    attributes = np.stack([
        np.array(
            tuple(np.moveaxis(sample, 1, 0)),
            dtype=[(k, np.int8, (n_distractors + 1,)) for k in attribute_keys],
        ) for sample in attributes
    ])

    # save attribute mapping
    maps = (
        np.array(shapes),
        np.array(colors),
        np.array([0, 1], dtype=np.int8),
        np.array([0, 1], dtype=np.int8),
    )
    mapping = np.array(
        maps,
        dtype=[(k, m.dtype, m.shape) for k, m in zip(attribute_keys, maps)],
    )
    sample_modes = np.array(sampling_mode) if sampling_mode else None

    return input_features, target_positions, attributes, mapping, sample_modes


if __name__ == '__main__':
    args = parse_args()

    if args.render:
        render_scenes(args)

    np.random.seed(args.seed)
    random.seed(args.seed)

    all_objects = list(product(shapes, colors))
    test_objects = [(s, c) for s, c in zip(shapes * 2, cycle(colors))]
    train_objects = [t for t in all_objects if t not in test_objects]

    print('\ntest objects:')
    pprint(test_objects)

    select_fn = get_select_random(args)

    train, train_targets, train_attr, train_mapping, _ = \
        export_input_data(train_objects, args, select_fn, args.n_samples_train)
    eval_train, eval_train_targets, eval_train_attr, \
        eval_train_mapping, eval_train_sample_modes = export_input_data(
            train_objects, args, select_fn, args.n_samples_eval_train)
    eval_test, eval_test_targets, eval_test_attr, eval_test_mapping, \
        eval_test_sample_modes = export_input_data(
            test_objects, args, select_fn, args.n_samples_eval_test)

    train_shapes = len({t[0] for t in train_objects})
    train_colors = len({t[1] for t in train_objects})

    print('\nNumber of shapes:', len(shapes))
    print('Number of colors:', len(colors), '\n')
    print('Number of shapes in the train set:', train_shapes)
    print('Number of colors in the train set:', train_colors, '\n')
    print('Unique tuples in the train set:', len(train_objects))
    print('Unique tuples in the test set:', len(test_objects), '\n')
    print('Train samples:', len(train))
    print('Eval samples (train):', len(eval_train))
    print('Eval samples (test):', len(eval_test), '\n')

    os.makedirs(path('../input_data/'), exist_ok=True)
    fname = f'obverter-{args.n_distractors + 1}-' \
        f'{args.n_samples_train}-{args.resolution}.npz'

    np.savez_compressed(
        path('..', 'input_data', fname),
        train=train,
        train_targets=train_targets,
        train_attributes=train_attr,
        train_attribute_mapping=train_mapping,
        eval_train=eval_train,
        eval_train_targets=eval_train_targets,
        eval_train_attributes=eval_train_attr,
        eval_train_attribute_mapping=eval_train_mapping,
        eval_train_sample_modes=eval_train_sample_modes,
        eval_test=eval_test,
        eval_test_targets=eval_test_targets,
        eval_test_attributes=eval_test_attr,
        eval_test_attribute_mapping=eval_test_mapping,
        eval_test_sample_modes=eval_test_sample_modes,
        train_objects=np.array(train_objects),
        test_objects=np.array(test_objects),
    )

    print(f'data saved to .../data/input_data/{fname}')
