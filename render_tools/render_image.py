# import mitsuba as mi

# mi.set_variant('cuda_ad_rgb')

# image = mi.render(scene)


import mitsuba as mi
import os
mi.set_variant('scalar_rgb')
import datetime
import multiprocessing
import os
import torch
import numpy as np


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

def run_script(script):
    os.system(script)

# phis = [sensor_sep * i for i in range(1,8)]
# print(phis)

# for i, phi in enumerate(phis):

root_path = '/root/autodl-tmp/skj/PointFlowRenderer/test_images_vat'
xml_path = os.path.join(root_path, 'xml_files')
scripts = []
cuda_sel = 0

dirList = os.listdir(xml_path)
for cat in dirList[2:]:
    cat_folder = os.path.join(xml_path, cat)
    xml_file_list = os.listdir(cat_folder)
    image_save_file = os.path.join('test_images_vat/images', cat)
    if not os.path.exists(image_save_file):
        os.makedirs(image_save_file) 
    for file in xml_file_list:
        scene_file = os.path.join(cat_folder, file)
        shape_id = file.split('_')[0]
        scripts.append(["python render_image_call.py --scene_file {} --shape_id {} --cat {}".format(scene_file, shape_id, cat)])

while len(scripts) != 0:
    processes = []
    for script in scripts[:12]:
        p = multiprocessing.Process(target=run_script, args=(script))
        # total_jobs += 1
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    scripts = scripts[12:]

# print(total_jobs)
print("Complete")
        
        
        # try:
        #     scene = mi.load_file(scene_file)
        #     sensor = load_sensor(radius, phi, theta)
        #     image = mi.render(scene, spp=spp, sensor=sensor)
        #     mi.util.write_bitmap(f'test_images/images/{cat}/result_image_phi-{phi}-theta-{theta}-radius-{radius}_{shape_id}.png', image)
        # except:
        #     pass

        
