import mitsuba as mi
import os
from argparse import ArgumentParser
mi.set_variant('scalar_rgb')

from mitsuba import ScalarTransform4f as T
sensor_width = 512
sensor_height = 512
sensor_sep = 25            
phi = 140  
radius = 4          # view distance form center
theta = 50         # view angle from upright axis
spp = 512


def load_sensor(r, phi, theta):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': sensor_width,
            'height': sensor_height,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })


parser = ArgumentParser()
parser.add_argument("--scene_file", type=str)
parser.add_argument("--shape_id", type=str)
parser.add_argument("--cat", type=str)
conf = parser.parse_args()

scene = mi.load_file(conf.scene_file)
sensor = load_sensor(radius, phi, theta)
image = mi.render(scene, spp=spp, sensor=sensor)
mi.util.write_bitmap(f'test_images_vat/images/{conf.cat}/result_image_phi-{phi}-theta-{theta}-radius-{radius}_{conf.shape_id}.png', image)