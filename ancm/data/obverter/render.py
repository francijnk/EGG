# Sources:
# (1) https://github.com/benbogin/obverter/blob/master/data.py
# (2) https://github.com/benbogin/obverter/blob/master/create_ds.py
import shutil
import os
import pickle
import random
from PIL import Image
import numpy as np
from vapory import *
from tqdm import tqdm


colors = {
    'red': [1, 0, 0],
    'blue': [0, 0, 1],
    'green': [0, 1, 0],
    'white': [1] * 3,
    'gray': [0.5] * 3,
    'yellow': [1, 1, 0.1],
    'cyan': [0, 1, 1],
    'magenta': [1, 0, 1]}
object_types = ['box', 'sphere', 'cylinder', 'torus', 'ellipsoid', 'cone']

current_dir = os.path.dirname(os.path.abspath(__file__))


def render_scene(idx, object_type, color, location, rotation, resolution):
    assert (object_type in object_types)
    assert (color in colors)

    def location_to_categorical(location):
        # split [-3, 3] into [-3, -1) [-1, 1) [1, 3]
        # if location < -1:
        #     return -1
        # elif location < 1:
        #     return 0
        # else:
        #     return 1

        # split [-3, 3] into [-3, -1.8), [-1.8, -0.6) [-0.6, 0.6)
        # [0.6, 1.8), [1.8, 3]
        if location < -1.8:
            return -2
        elif location < -0.6:
            return -1
        elif location < 0.6:
            return 0
        elif location < 1.8:
            return 1
        else:
            return 2

    def rotation_to_categorical(rotation, object_type):
        # if rotational symmetry is present, return 0
        if object_type in 'sphere torus cone cylinder'.split():
            return 0

        # ellipsoid and box exhibit reflection symmetry
        if object_type == 'box':  # 2 symmetry axes
            rotation = rotation % 90
        elif object_type == 'ellipsoid':
            rotation = rotation % 180

        if 0 <= rotation < 22.5 or 157.5 <= rotation <= 180:
            rotation = 0  # front view
        elif 22.5 <= rotation < 67.5:
            rotation = 1  # angle view (L)
        elif 67.5 <= rotation < 112.5:
            rotation = 2  # side view
        elif 112.5 <= rotation < 157.5:
            rotation = 3  # angle view (R)
        else:
            rotation = None
        assert rotation is not None

        if object_type == 'box':
            rotation = rotation % 2
        return rotation

    location_cat = [location_to_categorical(loc) for loc in location]
    rotation_cat = rotation_to_categorical(rotation, object_type)

    filename = \
        f'{color}_{object_type}_{idx}_{location_cat[0]}_' \
        f'{location_cat[1]}_{rotation_cat}'
    filepath = os.path.join(current_dir, 'assets', filename)

    color = colors[color]
    size = 2
    radius = size / 2
    attributes = [
        Texture(Pigment('color', color)),
        Finish('ambient', 0.7),
        'rotate', (0, rotation, 0),
        'translate', (location[0], 0, location[1]),
    ]
    if object_type == 'box':
        location.insert(1, size / 2)
        obj = Box ([-size / 2, 0, -size /2], [size/2, size, size/2], *attributes) 
    if object_type == 'sphere':
        attributes.extend(['translate', (0, radius, 0)])
        obj = Sphere([0, 0, 0], radius, *attributes)
    if object_type == 'torus':
        attributes.extend(['translate', (0, radius / 2, 0)])
        obj = Torus(radius, radius / 2, *attributes)
    if object_type == 'ellipsoid':
        attributes.extend(['translate', (0, radius * 0.45, 0)])
        obj = Sphere([0, 0, 0], radius, 'scale', (0.75, 0.45, 1.5), *attributes)
    if object_type == 'cylinder':
        obj = Cylinder([0, 0, 0], [0, size * 1.5, 0], radius, *attributes)
    if object_type == 'cone':
        obj = Cone([0, 0, 0], radius, [0, 1.75 * size, 0], 0., *attributes)

    camera = Camera('location', [0, 7, -6], 'look_at', [0, 0, 0])
    light = LightSource([-8, 12, -5], 'color', [1, 1, 1])

    chessboard = Plane(
        [0, 1, 0], 0.30, 'hollow',
        Texture(
            Pigment(
                'checker',
                'color', [.47, .6, .74],
                'color', [.34, 0.48, 0.6]),
            'scale', 2.4, 'translate', (1.2, 0, 1.2)),
         Finish('ambient', 0.5))
    scene = Scene(camera, objects=[light, obj, chessboard])
    scene.render(filepath, width=resolution, height=resolution, antialiasing=0.001)


def get_object_fname(color, shape, x=None, y=None, rotation=None, idx=None):
    assert all([var is None for var in [x, y, rotation]]) or idx is None

    assets_dir = os.path.join(current_dir, 'assets')
    all_images = os.listdir(assets_dir)

    if x is None and y is None and idx is None:
        prefix = f'{color}_{shape}_'
        _filter = lambda x: x.startswith(prefix)
    elif x is None and y is None and rotation is None:
        prefix = f'{color}_{shape}_{idx}_'
        _filter = lambda x: x.startswith(prefix)
    else:
        def _filter(fname):
            _color, _shape, _, _x, _y, _rotation = fname.split('.')[0].split('_')
            if color != _color or shape != shape:
                return False
            if x is not None and int(x) != _x:
                return False
            if y is not None and int(y) != _y:
                return False
            if rotation is not None and int(rotation) !=  _rotation:
                return False
            return True

    matches = [fname for fname in all_images if _filter(fname)]
    if not matches:
        return None
    else:
        return random.choice(matches)


def render_scenes(args):
    random.seed(args.seed)
    assets_dir = os.path.join(current_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    print("Rendering scenes...")
    n_samples_img = args.n_img  
    total_steps = len(colors) * len(object_types) * n_samples_img
    with tqdm(total=total_steps) as pbar:
        for color in colors:
            for object_type in object_types:
                for idx in range(n_samples_img):
                    location = [random.uniform(-3, 3), random.uniform(-3, 3)]
                    rotation = random.uniform(0, 360)

                    filename = get_object_fname(color, object_type, idx=idx) 
                    if filename is not None:
                        tqdm.write("%s.png exists, skipping" % filename)
                        pbar.update(1)
                        continue
                    else:
                        resolution = 128 if args.resolution == 64 else args.resolution
                        render_scene(idx, object_type, color, location, rotation, resolution)
                        pbar.update(1)
    print("Finished")
