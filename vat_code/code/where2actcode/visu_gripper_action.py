"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
from utils_opengl import get_global_position_from_camera
import cv2
import json
from argparse import ArgumentParser

from sapien.core import Pose
from env_opengl import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot

in_fn = sys.argv[1]
out_dir = '/'.join(in_fn.split('/')[:-1])
with open(in_fn, 'r') as fin:
    primact_type = fin.readline().rstrip()
    dir1 = np.array([float(x) for x in fin.readline().rstrip().split()], dtype=np.float32)
    dir2 = np.array([float(x) for x in fin.readline().rstrip().split()], dtype=np.float32)

dir1 /= np.linalg.norm(dir1)
dir3 = np.cross(dir1, dir2)
dir2 = np.cross(dir3, dir1)
dir2 /= np.linalg.norm(dir2)

# setup env
env = Env()

# setup camera
cam = Camera(env, dist=1)

# load the red dot
object_urdf_fn = 'sphere.urdf'
object_material = env.get_material(4, 4, 0.01)
env.load_object(object_urdf_fn, object_material)
rd_ids = [l.get_id() for l in env.object.get_links()] 

# set dir1 dir2
dir1 = cam.get_metadata()['mat44'][:3, :3] @ dir1
dir2 = cam.get_metadata()['mat44'][:3, :3] @ dir2
dir3 = np.cross(dir1, dir2)

# get pixel 3D position (cam/world)
position_world = np.array([0, 0, 0], dtype=np.float32)

# compute final pose
rotmat = np.eye(4).astype(np.float32)
rotmat[:3, 0] = dir2
rotmat[:3, 1] = dir3
rotmat[:3, 2] = dir1

final_rotmat = np.array(rotmat, dtype=np.float32)
final_rotmat[:3, 3] = position_world - dir1 * 0.11
final_pose = Pose().from_transformation_matrix(final_rotmat)
position2 = final_rotmat[:, 3]
depth2 = np.linalg.inv(cam.get_metadata()['mat44']) @ position2
depth2 = depth2[0]

start_rotmat = np.array(rotmat, dtype=np.float32)
start_rotmat[:3, 3] = position_world - dir1 * 0.11 - dir1 * 0.05 * 3
start_pose = Pose().from_transformation_matrix(start_rotmat)
position1 = start_rotmat[:, 3]
depth1 = np.linalg.inv(cam.get_metadata()['mat44']) @ position1
depth1 = depth1[0]

action_direction = None
if 'left' in primact_type:
    action_direction = dir2
elif 'up' in primact_type:
    action_direction = dir3

### viz the EE gripper position
loader = env.scene.create_urdf_loader()
loader.fix_root_link = True
robot_material = env.get_material(4, 4, 0.01)
robot1 = loader.load('./robots/panda_gripper_white.urdf', {"material": robot_material})
link_ids1 = [l.get_id() for l in robot1.get_links()] + rd_ids
robot2 = loader.load('./robots/panda_gripper_white.urdf', {"material": robot_material})
link_ids2 = [l.get_id() for l in robot2.get_links()] + rd_ids
robot3 = loader.load('./robots/panda_gripper_white.urdf', {"material": robot_material})
link_ids3 = [l.get_id() for l in robot3.get_links()] + rd_ids
robot1.set_root_pose(start_pose)
open_joint_angles = [0, 0, 0, 0, 0, 0, 0.04, 0.04]
half_joint_angles = [0, 0, 0, 0, 0, 0, 0.02, 0.02]
closed_joint_angles = [0, 0, 0, 0, 0, 0, 0, 0]
if 'pulling' in primact_type:
    robot1.set_qpos(open_joint_angles)
else:
    robot1.set_qpos(closed_joint_angles)
if 'pushing' in primact_type:
    robot2.set_qpos(closed_joint_angles)
    robot3.set_qpos(closed_joint_angles)
else:
    robot2.set_qpos(half_joint_angles)
    robot3.set_qpos(closed_joint_angles)
robot2.set_root_pose(final_pose)

if action_direction is not None:
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - dir1 * 0.11 + action_direction * 0.05 * 3
    end_pose = Pose().from_transformation_matrix(end_rotmat)
    position3 = end_rotmat[:, 3]
    depth3 = np.linalg.inv(cam.get_metadata()['mat44']) @ position3
    depth3 = depth3[0]
    robot3.set_root_pose(end_pose)

if primact_type == 'pulling':
    robot3.set_root_pose(start_pose)
    position3 = start_rotmat[:, 3]
    depth3 = np.linalg.inv(cam.get_metadata()['mat44']) @ position3
    depth3 = depth3[0]

if primact_type == 'pushing':
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - dir1 * 0.11 + dir1 * 0.01 * 3
    end_pose = Pose().from_transformation_matrix(end_rotmat)
    position3 = end_rotmat[:, 3]
    depth3 = np.linalg.inv(cam.get_metadata()['mat44']) @ position3
    depth3 = depth3[0]
    robot3.set_root_pose(end_pose)

env.render()

rgb, _ = cam.get_observation()
object_mask = cam.get_object_mask()
#seg_mask = cam.camera.get_segmentation()
#link_mask1 = np.zeros((rgb.shape[0], rgb.shape[1])).astype(np.float32)
#link_mask2 = np.zeros((rgb.shape[0], rgb.shape[1])).astype(np.float32)
#link_mask3 = np.zeros((rgb.shape[0], rgb.shape[1])).astype(np.float32)
#for lid in np.unique(seg_mask):
#    if lid in link_ids1:
#        link_mask1[seg_mask == lid] = 1
#    if lid in link_ids2:
#        link_mask2[seg_mask == lid] = 1
#    if lid in link_ids3:
#        link_mask3[seg_mask == lid] = 1

def hide(actor):
    for link in actor.get_links():
        link.hide_visual()

def unhide(actor):
    for link in actor.get_links():
        link.unhide_visual()

frgb = np.ones((rgb.shape[0], rgb.shape[1], 3)).astype(np.float32)
fmask = np.zeros((rgb.shape[0], rgb.shape[1])).astype(np.float32)

def compose_img(front_rgb, front_mask, light):
    global frgb, fmask
    frgb[front_mask>0.5] = frgb[front_mask>0.5] * (1-light) + front_rgb[front_mask>0.5] * light
    fmask[front_mask>0.5] = fmask[front_mask>0.5] * (1-light) + front_mask[front_mask>0.5] * light

unhide(robot1); hide(robot2); hide(robot3);
env.render()
rgb1, _ = cam.get_observation()
link_mask1 = cam.get_object_mask()

hide(robot1); unhide(robot2); hide(robot3);
env.render()
rgb2, _ = cam.get_observation()
link_mask2 = cam.get_object_mask()

hide(robot1); hide(robot2); unhide(robot3);
env.render()
rgb3, _ = cam.get_observation()
link_mask3 = cam.get_object_mask()


depths = np.array([-depth1, -depth2, -depth3], dtype=np.float32)
depth_order = np.argsort(depths)

rgbs = [rgb1, rgb2, rgb3]
link_masks = [link_mask1, link_mask2, link_mask3]
lights = [0.6, 0.8, 1]
for di in depth_order:
    compose_img(rgbs[di], link_masks[di], lights[di])

#frgb[:, :, 0] = (rgb1[:, :, 0] * link_mask1 + rgb2[:, :, 0] * link_mask2 + rgb3[:, :, 0] * link_mask3) / (link_mask1 + link_mask2 + link_mask3 + 1e-12)
#frgb[:, :, 1] = (rgb1[:, :, 1] * link_mask1 + rgb2[:, :, 1] * link_mask2 + rgb3[:, :, 1] * link_mask3) / (link_mask1 + link_mask2 + link_mask3 + 1e-12)
#frgb[:, :, 2] = (rgb1[:, :, 2] * link_mask1 + rgb2[:, :, 2] * link_mask2 + rgb3[:, :, 2] * link_mask3) / (link_mask1 + link_mask2 + link_mask3 + 1e-12)
final_rgb = np.zeros((rgb.shape[0], rgb.shape[1], 4)).astype(np.float32)
final_rgb[:, :, :3] = frgb * np.expand_dims(fmask, axis=-1) + 1.0 * (1-np.expand_dims(fmask, axis=-1))
final_rgb[:, :, 3] = object_mask

Image.fromarray((final_rgb*255).astype(np.uint8)).save(os.path.join(out_dir, in_fn.split('/')[-1].replace('.txt', '.png')))

# close env
env.close()

