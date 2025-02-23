import os
import random
import argparse
import itertools
import numpy as np
from collections import defaultdict
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ancm.data.obverter.render import (
    colors,
    object_types,
    render_scene,
    render_scenes,
    get_object_fname,
)


current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_distractors', '-d', type=int, required=True)
    parser.add_argument('--n_samples_train', type=int, default=100)
    parser.add_argument('--n_samples_train_eval', type=int, default=10)
    parser.add_argument('--n_samples_test', type=int, default=30)
    parser.add_argument('--n_img', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--mode', type=str, default='random')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def load_image(shape, color, x=None, y=None, rotation=None, idx=None, resolution=128):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    assets_dir = os.path.join(current_dir, 'assets')

    fname = get_object_fname(color, shape, x, y, rotation, idx)
    if fname is not None:
        color, obj_type, _, xpos, ypos, rotation = \
            fname.split('.')[0].split('_')
    else:
        idx = 0
        while get_object_fname(color, shape, idx=idx) is not None:
            idx += 1

        if rotation is None:
            rotation = random.uniform(0, 360)
        else:
            if shape in 'sphere cylinder ellipsoid cone'.split() and rotation != 0:
                raise ValueError
            if shape == 'box' and rotation not in (0, 1):
                raise ValueError
            _rotation = None
            while _rotation is None:
                r = random.uniform(0, 360)
                r = render_scene.rotation_to_categorical(r, shape)
                if r == rotation:
                    _rotation = r

        bounds = {
            -2: (-3., -1.8),
            -1: (-1.8, -0.6),
            0: (-0.6, 0.6),
            1: (0.6, 1.8),
            2: (1.8, 3.),
        }
        if x is None:
            _xcat = random.randint(-2, 2)
            x = random.uniform(*bounds[_xcat])
        if y is None:
            _ycat = random.randint(-2, 2)
            y = random.uniform(*bounds[_ycat])
        location = [x, y]
        print(
            'Rendering a missing scene...'
            f'({shape}, {color}, {location}, {rotation})'
        )
        render_scene(idx, shape, color, location, rotation, resolution)

        fname = get_object_fname(color, shape, location, None)
        color, obj_type, _, xpos, ypos, rotation = \
            fname.split('.')[0].split('_')

    image = Image.open(os.path.join(assets_dir, fname)).convert("RGB")
    image = transform(image)
    image = image.numpy()

    return image, (obj_type, color, int(xpos), int(ypos), int(rotation))


def pick_random_color(tuples, exclude=None):
    if exclude is None:
        exclude = []
    elif not isinstance(exclude, list):
        exclude = [exclude]

    available_colors = [item[1] for item in tuples if item not in exclude]
    return random.choice(available_colors)


def pick_random_shape(tuples, exclude=None):
    if exclude is None:
        exclude = []
    elif not isinstance(exclude, list):
        exclude = [exclude]

    available_shapes = [item[0] for item in tuples if item not in exclude]
    return random.choice(available_shapes)


def select_random_distractor(attributes, used_combos, object_tuples):
    # assert len(attributes) == 5
    # shape, color, xpos, ypos, rotation = attributes

    exclude = [item[0] for item in used_combos]
    d_shape = pick_random_shape(object_tuples, exclude=exclude)
    exclude = [item[1] for item in used_combos if item[0] == d_shape]
    d_color = pick_random_color(object_tuples, exclude=exclude)

    d_xpos, d_ypos, d_rotation = None, None, None

    combination = (d_shape, d_color, d_xpos, d_ypos, d_rotation)
    used_combos.append(combination)
    return combination, used_combos


def export_input_data(object_tuples, args, select_fn, n_samples):
    n_distractors = args.n_distractors
    size = (len(object_tuples) * n_samples, n_distractors + 1)
    positions = np.arange(n_distractors + 1)
    _colors = list(colors)
    keys = ('shape', 'color', 'xpos', 'ypos', 'rotation')

    target_positions = np.empty(size[0], dtype=np.int64)
    target_positions[:] = -1
    attributes = defaultdict(lambda: np.empty(size, dtype=np.int64))
    sample_features = []

    def map_shape_color(attr, key):
        if key == 'shape':
            return object_types.index(attr)
        elif key == 'color':
            return _colors.index(attr)
        else:
            return attr

    # n_same_shape = int(0.1 * n_samples)
    # n_same_color = int(0.1 * n_samples)
    # n_same_location = int(0.2 * n_samples)

    for tuple_i, (shape, color) in tqdm(enumerate(object_tuples), total=len(object_tuples)):
        for sample_j in range(n_samples):
            idx = tuple_i * n_samples + sample_j
            image_idx = sample_j % args.n_img

            target_image, target_attributes = load_image(
                shape, color, idx=image_idx, resolution=args.resolution)

            target_pos = np.random.randint(0, n_distractors + 1)
            distractor_pos = positions[positions != target_pos]

            sample = np.empty((size[1], *target_image.shape), dtype=np.int64)
            sample[target_pos] = target_image

            target_positions[idx] = target_pos
            for k, key in enumerate(keys):
                attributes[key][idx, target_pos] = \
                    map_shape_color(target_attributes[k], key)

            used_combos = [target_attributes]
            for distr_k in range(n_distractors):
                d_attributes, used_combos = \
                    select_fn(target_attributes, used_combos, object_tuples)
                d_image, d_attributes = \
                    load_image(*d_attributes, resolution=args.resolution)
                d_pos = distractor_pos[distr_k]

                sample[d_pos] = d_image
                for key_k, key in enumerate(keys):
                    attributes[key][idx, d_pos] = \
                        map_shape_color(d_attributes[key_k], key)

            sample_features.append(sample.reshape(1, *sample.shape))

    input_data = np.vstack(sample_features)

    attributes = np.array(
        tuple(array for array in attributes.values()),
        dtype=[(key, np.int64, size) for key in keys],
    )

    # map remaining attributes and save mappings
    mappings = [
        np.char.array(object_types, unicode=False),
        np.char.array(_colors, unicode=False),
    ]
    for key in keys[2:]:
        mapping, mapped = np.unique(attributes[key], return_inverse=True)
        mappings.append(mapping)
        attributes[key] = mapped.reshape(attributes[key].shape)
    mapping = np.array(
        tuple(mappings),
        dtype=[
            (key, mapping.dtype, mapping.shape)
            for key, mapping in zip(keys, mappings)
        ],
    )

    return input_data, attributes, target_positions, mapping


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    render_scenes(args)

    if args.mode == 'random':
        select_fn = select_random_distractor
        fname = f'obverter-{args.n_distractors + 1}-{args.n_samples_train}-{args.resolution}.npz'
    else:
        raise NotImplementedError

    all_objects = list(itertools.product(object_types, colors))
    train_objects, test_objects = train_test_split(all_objects, test_size=0.25)

    train, train_attr, train_targets, train_mapping = \
        export_input_data(train_objects, args, select_fn, args.n_samples_train)
    eval_train, eval_train_attr, eval_train_targets, eval_train_mapping = \
        export_input_data(train_objects, args, select_fn, args.n_samples_train_eval)
    eval_test, eval_test_attr, eval_test_targets, eval_test_mapping = \
        export_input_data(test_objects, args, select_fn, args.n_samples_test)

    print('Number of shapes:', len(object_types))
    print('Number of colors:', len(colors.keys()))
    print('Unique tuples in the train set:', len(train_objects))
    print('Unique tuples in the test set:', len(test_objects))
    print('Train samples:', len(train))
    print('Eval samples (train):', len(eval_train))
    print('Eval samples (test):', len(eval_test))

    npz_fpath = os.path.join(
        current_dir, '..', 'input_data', fname)

    np.savez_compressed(
        npz_fpath,
        train=train,
        train_targets=train_targets,
        train_attributes=train_attr,
        train_attribute_mapping=train_mapping,
        eval_train=eval_train,
        eval_train_targets=eval_train_targets,
        eval_train_attributes=eval_train_attr,
        eval_train_attribute_mapping=eval_train_mapping,
        eval_test=eval_test,
        eval_test_targets=eval_test_targets,
        eval_test_attributes=eval_test_attr,
        eval_test_attribute_mapping=eval_test_mapping,
        allow_pickle=False,
    )
