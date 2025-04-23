# Based on https://github.com/benbogin/obverter/

import os
import random
import numpy as np
from vapory import (
    Scene, Camera, LightSource, Plane,
    Texture, Pigment, Finish,
    Box, Sphere, Cylinder, Cone,
    Torus, Superellipsoid,
)
from tqdm import tqdm
from itertools import product


resolution = 128
current_dir = os.path.dirname(os.path.abspath(__file__))


shapes = ['cube', 'sphere', 'cylinder', 'torus', 'ellipsoid', 'octahedron']
palette = {
    'white': [1, 1, 1], 'gray': [0.5, 0.5, 0.5], 'red': [1, 0, 0],
    'yellow': [1, 1, 0.1], 'green': [0, 1, 0],
    'cyan': [0, 1, 1], 'blue': [0, 0, 1], 'magenta': [1, 0, 1]
}
colors = list(palette.keys())


def render_scene(idx, shape, color, location, rotation):
    assert shape in shapes and color in colors

    def rotation_to_categorical(rotation, shape):
        # if rotational symmetry is present, return 0
        if shape in ('sphere', 'torus', 'cone', 'cylinder'):
            return 0

        rotation = rotation % 180  # ellipsoid and cube have reflection symmetry
        if 0 <= rotation < 22.5 or 157.5 <= rotation <= 180:
            rotation = 0  # front view
        elif 22.5 <= rotation < 67.5:
            rotation = 1  # angle view (L)
        elif 67.5 <= rotation < 112.5:
            rotation = 2  # side view
        elif 112.5 <= rotation < 157.5:
            rotation = 3  # angle view (R)
        else:
            raise ValueError

        # cube has 2 symmetry axes
        if shape in ('cube', 'octahedron'):
            rotation = rotation % 2
        return rotation

    location_cat = [int(location[0] + 3) // 6, int(location[1] + 1.5) // 6]
    # location_cat = [int(location[0] // 2.5), int(location[1] // 2.5)]
    # location_cat = [(location[0] + 3) // 6, int(location[1] + 1.5) // 3]
    rotation_cat = rotation_to_categorical(rotation, shape)

    filename = \
        f'{shape}_{color}_{location_cat[0]}_{location_cat[1]}_' \
        f'{idx}_{rotation_cat}'
    filepath = os.path.join(current_dir, f'assets_{resolution}', filename)

    location = [
        location[0] + np.random.normal(0, 0.15),
        location[1] + np.random.normal(0, 0.15),
    ]
    size = 2
    radius = size / 2
    finish = ('ambient', 0.3, 'diffuse', 0.7)
    attributes = [
        Texture(Pigment('color', palette[color])),
        Finish(*finish),
        'rotate', (0, rotation, 0),
        'translate', (location[0], 0.0, location[1]),
    ]
    if shape == 'cube':
        location.insert(1, size / 2)
        obj = Box(
            [-size / 2, 0, -size / 2],
            [size / 2, size, size / 2],
            *attributes,
        )
    if shape == 'sphere':
        attributes.extend(['translate', (0, radius * 1.25, 0)])
        obj = Sphere([0, 0, 0], radius * 1.25, *attributes)
    if shape == 'torus':
        attributes.extend(['translate', (0, radius / 2, 0)])
        obj = Torus(radius, radius / 2, *attributes)
    if shape == 'ellipsoid':
        attributes.extend(['translate', (0, radius / 1.8, 0)])
        obj = Sphere([0, 0, 0], radius, 'scale', (1, 1 / 1.8, 1 * 1.8), *attributes)
    if shape == 'cylinder':
        obj = Cylinder([0, 0, 0], [0, size * 1.1, 0], radius * 0.8, *attributes)
    if shape == 'octahedron':
        obj = Superellipsoid(
            [2, 2],
            'scale', 1.8,
            'translate', [0, 1.8, 0],
            *attributes
        )
    if shape == 'cone':
        obj = Cone(
            [0, 0, 0], radius * 1.2,
            [0, size * 1.4, 0], 0.,
            'rotate', [180, 0, 0],
            'translate', [0, size * 1.4, 0],
            *attributes
        )

    # camera = Camera('location', [0, 8, -7], 'look_at', [0, 0, 0])
    camera = Camera('location', [0, 8.5, -7], 'look_at', [0, -1, 2])
    light = LightSource([-12, 10, -3], 'color', [1, 1, 1], 'adaptive', 2)
    # light = LightSource([-12, 10, -8], 'color', [1, 1, 1], 'adaptive', 2)
    # light = LightSource([0, 10, 1.5], 'color', [1, 1, 1], 'adaptive', 2)
    checker = Texture(
        Pigment('checker', 'color', [.47, .6, .74], 'color', [.34, .48, .6]),
        'scale', 3,
        'translate', (1.5, 3.5 - 2.5),
    )
    chessboard = Plane([0, 1, 0], 0.0, 'hollow', checker, Finish(*finish))

    scene = Scene(camera, objects=[light, obj, chessboard])
    scene.render(
        filepath, width=resolution, height=resolution,
        antialiasing=0.0001, quality=11,
    )


def get_object_fname(shape, color, x=None, y=None, idx=None):
    if idx is not None:
        assert shape is not None and color is not None
        # assert x in (-2.5, 0, 2.5) and y in (-2.5, 0, 2.5)
        # assert x in (-3, 3) and y in (-1.5, 1.5)
        assert x in (-3, 3)
        assert y in (-1.5, 4.5)

        # x, y = int(x // 2.5), int(y // 2.5)
        # x, y = (x + 3) // 3, int((y + 1.5) // 3)
        x, y = (x + 3) // 6, int((y + 1.5) // 6)
        prefix = f'{shape}_{color}_{x}_{y}_{idx}_'

        def f(fname):
            return fname.startswith(prefix) and 'pov-state' not in fname

    else:
        # assert x in (-1, 0, 1, None) and y in (-1, 0, 1, None)
        assert x in (0, 1, None) and y in (0, 1, None)

        def f(fname):
            _s, _c, _x, _y = fname.split('.')[0].split('_')[:4]
            if fname.endswith('pov-state'):
                return False
            for a, _a in [(shape, _s), (color, _c), (x, int(_x)), (y, int(_y))]:
                if a is not None and a != _a:
                    return False
            return True

    assets_dir = os.path.join(current_dir, f'assets_{resolution}')
    all_images = os.listdir(assets_dir)
    matches = list(filter(f, all_images))

    return random.choice(matches) if matches else None


def render_scenes(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    assets_dir = os.path.join(current_dir, f'assets_{resolution}')
    os.makedirs(assets_dir, exist_ok=True)

    total_steps = len(colors) * (len(shapes)) * 2 * 2 * args.n_images
    pbar = tqdm(total=total_steps)
    # for shape, color, x, y in product(shapes, colors, [-2.5, 0, 2.5], [-2.5, 0, 2.5]):
    for shape, color, x, y in product(shapes, colors, [-3, 3], [-1.5, 4.5]):
        for idx in range(args.n_images):
            filename = get_object_fname(shape, color, x, y, idx=idx)
            if filename is not None:
                pbar.update(1)
                pbar.set_description("Checking for missing scenes")
                continue
            else:
                location = [x, y]
                rotation = random.uniform(0, 360)
                pbar.set_description("Rendering scenes")
                render_scene(idx, shape, color, location, rotation)
                pbar.update(1)
                pbar.refresh()
    pbar.close()
