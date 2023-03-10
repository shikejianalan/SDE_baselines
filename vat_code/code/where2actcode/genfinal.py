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
from utils import get_global_position_from_camera, save_h5, radian2degree, degree2radian
# import cv2
import json
from argparse import ArgumentParser
import torch
import copy
import time
import imageio

from sapien.core import Pose, ArticulationJointType
#from pointnet2_ops.pointnet2_utils import furthest_point_sample
# from ppo_actor_critic2 import PPO
# from td3 import ReplayBuffer
# from td3 import TD3
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
import random
from tensorboardX import SummaryWriter
from scipy.spatial.transform import Rotation as R
from models.model_3d_task_actor_dir_RL_raw import ActorNetwork
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from models.model_pn2_ae import Network
import ipdb
# import traceback
# import faulthandler
# faulthandler.enable()

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--shape_id', type=str)
parser.add_argument('--category', type=str)
parser.add_argument('--cnt_id', type=int)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_degree', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--every_bonus', action='store_true', default=False, help='no_gui [default: False]')
# parser.add_argument('--zz_penalty', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_initial_position', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_initial_dir', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_initial_up_dir', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_joint_origins', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_ctpt_dis_to_joint', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--pn_feat', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--pred_world_xyz', type=int, default=0)
parser.add_argument('--pred_residual_world_xyz', type=int, default=0)
parser.add_argument('--pred_residual_cambase_xyz', type=int, default=0)
parser.add_argument('--pred_residual_root_qpos', type=int, default=1)
parser.add_argument('--up_norm_dir', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--final_dist', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--replay_buffer_size', type=int, default=5e5)
parser.add_argument('--pos_range', type=float, default=0.5)
parser.add_argument('--explore_noise_scale', type=float, default=0.01)
parser.add_argument('--eval_noise_scale', type=float, default=0.01)
parser.add_argument('--noise_decay', type=float, default=0.8)
# parser.add_argument('--sparse_reward', type=float, default=0.1)
parser.add_argument('--guidance_reward', type=float, default=0.0)
parser.add_argument('--decay_interval', type=int, default=100)
parser.add_argument('--q_lr', type=float, default=3e-4)
parser.add_argument('--policy_lr', type=float, default=3e-4)
parser.add_argument('--threshold', type=float, default=0.15)
parser.add_argument('--task_upper', type=float, default=30)
parser.add_argument('--task_lower', type=float, default=30)
parser.add_argument('--success_reward', type=int, default=10)
# parser.add_argument('--target_margin', type=float, default=2)
parser.add_argument('--HER_move_margin', type=float, default=2)
parser.add_argument('--target_part_state', type=str, default='random-middle')
parser.add_argument('--num_steps', type=int, default=4)
parser.add_argument('--with_step', type=int, default=1)
parser.add_argument('--update_itr2', type=int, default=2)
parser.add_argument('--early_stop', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_HER', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--HER_only_success', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--HER_only_attach', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--sample_num', type=int, default=2)
parser.add_argument('--eval_epoch', type=int, default=100)
parser.add_argument('--out_gif', type=int, default=1)
parser.add_argument('--out_png', type=int, default=0)
parser.add_argument('--wp_rot', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--use_direction_world', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--up_norm_thresh', type=float, default=0)
parser.add_argument('--use_random_up', action='store_true', default=False, help='no_gui [default: False]')
# parser.add_argument('--eval_train_set', type=int, default=0)
# parser.add_argument('--eval_val_set', type=int, default=1)
parser.add_argument('--state_axes', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--state_axes_all', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--sample_sameConf_diffZ', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--sample_sameTask_diffProposal', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--wp_xyz', type=int, default=1)
parser.add_argument('--coordinate_system', type=str, default='cambase')
parser.add_argument('--sample_type', type=str, default='random')
parser.add_argument('--num_offset', type=int, default=0)



args = parser.parse_args()

# set random seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

train_shape_list, val_shape_list = [], []
# train_file_dir = "../stats/train_10cats_train_data_list.txt"
# val_file_dir = "../stats/train_10cats_test_data_list.txt"
train_file_dir = "../stats/train_where2actPP_train_data_list.txt"
val_file_dir = "../stats/train_where2actPP_test_data_list.txt"
all_shape_list = []
if args.primact_type == 'pushing' or args.primact_type == 'pulling':
    all_cat_list = ['StorageFurniture', 'Microwave', 'Refrigerator', 'Door']
    eval_cat_list = [' Safe', 'WashingMachine', 'Table']

len_shape = {}
len_train_shape = {}
len_val_shape = {}
shape_cat_dict = {}
cat_shape_id_list = {}
val_cat_shape_id_list = {}
train_cat_shape_id_list = {}
for cat in all_cat_list:
    len_shape[cat] = 0
    len_train_shape[cat] = 0
    len_val_shape[cat] = 0
    cat_shape_id_list[cat] = []
    train_cat_shape_id_list[cat] = []
    val_cat_shape_id_list[cat] = []

with open(train_file_dir, 'r') as fin:
    for l in fin.readlines():
        shape_id, cat = l.rstrip().split()
        if cat not in all_cat_list:
            continue
        train_shape_list.append(shape_id)
        all_shape_list.append(shape_id)
        shape_cat_dict[shape_id] = cat
        len_shape[cat] += 1
        len_train_shape[cat] += 1
        cat_shape_id_list[cat].append(shape_id)
        train_cat_shape_id_list[cat].append(shape_id)

with open(val_file_dir, 'r') as fin:
    for l in fin.readlines():
        shape_id, cat = l.rstrip().split()
        if cat not in all_cat_list:
            continue
        val_shape_list.append(shape_id)
        all_shape_list.append(shape_id)
        shape_cat_dict[shape_id] = cat
        len_shape[cat] += 1
        len_val_shape[cat] += 1
        cat_shape_id_list[cat].append(shape_id)
        val_cat_shape_id_list[cat].append(shape_id)

len_train_shape_list = len(train_shape_list)
len_val_shape_list = len(val_shape_list)
len_all_shape_list = len(all_shape_list)

EP_MAX = 500
batch_size = args.batch_size
hidden_dim = 512
policy_target_update_interval = 3 # delayed update for the policy network and target networks
DETERMINISTIC = True  # DDPG: deterministic policy gradient
# explore_noise_scale = args.explore_noise_scale
# eval_noise_scale = args.eval_noise_scale
reward_scale = 1.
noise_decay = args.noise_decay
decay_interval = args.decay_interval
threshold_gripper_distance = args.threshold

# state
# 0     : door's current qpos
# 1     : distance to task
# 2     : task
# 3-5   : start_gripper_root_position
# 6-8   : contact_point_position_world
# 9-11  : gripper_finger_position
# 12-19 : gripper_qpos
# 20-28 : up, forward, left
#       : up
# 29-31 : joint_origins
# 32    : state_ctpt_dis_to_joint
# 33-37 : step_idx

record = [0 for i in range(100)]

shape_id = args.shape_id
trial_id = args.trial_id
primact_type = args.primact_type

# out_dir = os.path.join(args.out_dir,
#                        '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))
out_dir = args.out_dir % trial_id
print('out_dir: ', out_dir)

save_dir = os.path.join(out_dir, '45783_task40_diffProposal_1')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

result_succ_dir = os.path.join(save_dir, 'result_succ_imgs')
if not os.path.exists(result_succ_dir):
    os.mkdir(result_succ_dir)
result_fail_dir = os.path.join(save_dir, 'result_fail_imgs')
if not os.path.exists(result_fail_dir):
    os.mkdir(result_fail_dir)
# result_tmp_dir = os.path.join(out_dir, 'result_tmp_imgs')
# if not os.path.exists(result_tmp_dir):
#     os.mkdir(result_tmp_dir)


# ckpt_path = "/home/wuruihai/where2actCode/logs/exp-model_3d_task_actor_dir_RL-pushing-StorageFurniture-2021032962/ckpts/%d-network.pth" % args.eval_epoch
ckpt_path = os.path.join(out_dir, 'ckpts', '%d-network.pth' % args.eval_epoch)

actor = ActorNetwork(feat_dim=128, num_steps=5).to(device)
actor.load_state_dict(torch.load(ckpt_path))
actor.eval()
# load


# setup env
print("creating env")
env = Env(show_gui=(not args.no_gui))
print("env creared")
cam_theta = 3.7343466485504053
cam_phi = 0.3176779188918737
cam = Camera(env, theta=cam_theta, phi=cam_phi)
print("camera created")
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)


# load shape
# state = 'random-closed-middle'
state = 'random-middle'

### viz the EE gripper position
# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)

state_RL = []
# all_reward = []
target_part_state = args.target_part_state

if args.pred_world_xyz + args.pred_residual_world_xyz + args.pred_residual_root_qpos + args.pred_residual_cambase_xyz != 1:
    raise ValueError

tot_succ_epoch = -1
tot_done_epoch = 0
tot_fail_epoch = 0
mat44 = cam.get_metadata()['mat44']


saved_task_degree = 40
saved_shape_id = 45783
saved_joint_angles = [1.1453292608261108]
saved_x, saved_y = 232, 242     # 267, 249 ?


saved_num = 0

robot_loaded = 0
object_material = env.get_material(4, 4, 0.01)

valid_shape = [0 for idx in range(len_train_shape_list + len_val_shape_list)]
num_same_ctpt = 1000

# for epoch in range(EP_MAX):
for epoch in range(1):
    print('epoch: ', epoch)
    torch.cuda.empty_cache()

    if not args.no_gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

    torch.cuda.empty_cache()

    record[(tot_succ_epoch + 1) % 100] = 0

    # if args.eval_train_set == 1:
    #     idx_shape = random.randint(0, len_train_shape_list - 1)
    #     while valid_shape[idx_shape] == -1:
    #         idx_shape = random.randint(0, len_train_shape_list - 1)
    #     shape_id = train_shape_list[idx_shape]
    # elif args.eval_val_set == 1:
    #     idx_shape = random.randint(0, len_val_shape_list - 1)
    #     while valid_shape[idx_shape] == -1:
    #         idx_shape = random.randint(0, len_val_shape_list - 1)
    #     shape_id = val_shape_list[idx_shape]

    selected_cat = all_cat_list[random.randint(0, len(all_cat_list) - 1)]
    # print('category: ', selected_cat)
    shape_id = val_cat_shape_id_list[selected_cat][random.randint(0, len_val_shape[selected_cat] - 1)]

    flog = open(os.path.join(out_dir, 'log.txt'), 'a')
    env.flog = flog

    # idx_shape = random.randint(0, len_train_shape_list - 1)
    # while valid_shape[idx_shape] == -1:
    #     idx_shape = random.randint(0, len_train_shape_list - 1)
    # shape_id = train_shape_list[idx_shape]


    # if args.sample_sameConf_diffZ:
    #     shape_id = saved_shape_id
    # if args.sample_sameTask_diffProposal and epoch % num_same_ctpt != 0:
    #     shape_id = saved_shape_id
    # else:
    #     saved_shape_id = shape_id
    shape_id = saved_shape_id
    print('shape_id: ', shape_id)

    object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
    # state = 'random-closed-middle'
    target_part_state = args.target_part_state
    joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, target_part_id=-1)
    # if args.sample_sameConf_diffZ:
    #     joint_angles = saved_joint_angles
    # if args.sample_sameTask_diffProposal and epoch % num_same_ctpt != 0:
    #     joint_angles = saved_joint_angles
    # else:
    #     saved_joint_angles = joint_angles
    saved_joint_angles[0] -= degree2radian(saved_task_degree-9)
    joint_angles = saved_joint_angles
    env.render()
    # target_part_joint_idx = env.find_target_part_joint_idx(target_part_id=target_part_id)

    ### use the GT vision
    rgb, depth = cam.get_observation()

    # get movable link mask
    object_link_ids = env.movable_link_ids
    gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

    # sample a pixel to interact
    xs, ys = np.where(gt_movable_link_mask > 0)
    if len(xs) == 0:
        env.scene.remove_articulation(env.object)
        flog.close()
        continue

    idx = np.random.randint(len(xs))
    x, y = xs[idx], ys[idx]
    target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    env.set_target_object_part_actor_id(target_part_id)
    tot_trial = 0
    while tot_trial < 100 and ((env.target_object_part_joint_type != ArticulationJointType.REVOLUTE) or (
            env.get_target_part_axes_dir(target_part_id) != 1)):
        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
        tot_trial += 1
    old_target_part_id = target_part_id
    if args.sample_sameConf_diffZ:
        x, y = saved_x, saved_y
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
    if args.sample_sameTask_diffProposal and epoch % num_same_ctpt != 0:
        x, y = saved_x, saved_y
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
    else:
        saved_x, saved_y = x, y
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
    if (env.target_object_part_joint_type != ArticulationJointType.REVOLUTE) or (
            env.get_target_part_axes_dir(target_part_id) != 1):
        env.scene.remove_articulation(env.object)
        flog.close()
        # valid_shape[idx_shape] = -1
        continue
    target_part_joint_idx = env.target_object_part_joint_id

    joint_angle_lower = env.joint_angles_lower[target_part_joint_idx]
    joint_angle_upper = env.joint_angles_upper[target_part_joint_idx]
    joint_angle_lower_degree = radian2degree(joint_angle_lower)
    joint_angle_upper_degree = radian2degree(joint_angle_upper)
    task_upper = min(joint_angle_upper_degree, args.task_upper)

    task_lower = args.task_lower
    # task_lower = max(args.task_lower, radian2degree(target_part_qpos))    # task = np.pi * 30 / 180
    task_degree = random.random() * (task_upper - task_lower) + task_lower
    task_degree = saved_task_degree
    # if args.sample_sameConf_diffZ:
    #     task_degree = saved_task_degree
    if args.primact_type == 'pulling':
        task_degree = -task_degree
    task = degree2radian(task_degree)
    print(task_lower, task_upper)
    print("task:", task_degree)

    # joint_angles = env.update_joint_angle(joint_angles, target_part_joint_idx, target_part_state, task_degree)
    env.set_object_joint_angles(joint_angles)
    env.render()
    # print("angles:", joint_angles)

    ### use the GT vision
    rgb, depth = cam.get_observation()
    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
    gt_nor = cam.get_normal_map()

    # get movable link mask
    object_link_ids = env.movable_link_ids
    gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

    # sample a pixel to interact
    xs, ys = np.where(gt_movable_link_mask > 0)
    if len(xs) == 0:
        env.scene.remove_articulation(env.object)
        flog.close()
        continue

    idx = np.random.randint(len(xs))
    x, y = xs[idx], ys[idx]
    target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    env.set_target_object_part_actor_id(target_part_id)
    tot_trial = 0
    while tot_trial < 100 and target_part_id != old_target_part_id:
        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
        tot_trial += 1
    if target_part_id != old_target_part_id:
        env.scene.remove_articulation(env.object)
        flog.close()
        continue
    if args.sample_sameConf_diffZ:
        x, y = saved_x, saved_y
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
    if args.sample_sameTask_diffProposal and epoch % num_same_ctpt != 0:
        x, y = saved_x, saved_y
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
    else:
        saved_x, saved_y = x, y
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
    print("x, y:", x, y)

    joint_origins = env.get_target_part_origins_new(target_part_id=target_part_id)
    joint_axes = env.get_target_part_axes(target_part_id=target_part_id)
    print("joint_origins:", joint_origins)
    print("joint_axes", joint_axes)
    axes_dir = env.get_target_part_axes_dir(target_part_id)
    if axes_dir != 1:
        flog.close()
        # print(joint_axes)
        # print(axes_dir)
        print("old:", old_target_part_id)
        print("new:", target_part_id)
        print("not fitable axis!!!!!!")
        # for i in range(10000):
        #     env.step()
        #     env.render()
        # robot.wait_n_steps(10000)
        env.scene.remove_articulation(env.object)
        continue

    # get pixel 3D pulling direction (cam/world)
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam

    # sample a random direction in the hemisphere (cam/world)
    action_direction_cam = np.random.randn(3).astype(np.float32)
    action_direction_cam /= np.linalg.norm(action_direction_cam)
    if action_direction_cam @ direction_cam > 0:
        action_direction_cam = -action_direction_cam
    action_direction_cam = -direction_cam
    action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam

    # get pixel 3D position (cam/world)
    position_cam = cam_XYZA[x, y, :3]
    position_cam_xyz1 = np.ones((4), dtype=np.float32)
    position_cam_xyz1[:3] = position_cam
    position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
    position_world = position_world_xyz1[:3]

    state_joint_origins = joint_origins
    # !!!!!!!!!!
    # state_joint_origins[axes_dir] = position_world[axes_dir]
    state_joint_origins[-1] = position_world[-1]
    state_ctpt_dis_to_joint = np.linalg.norm(state_joint_origins - position_world)
    state_door_dir = position_world - state_joint_origins

    # compute camera-base frame (camera-center, world-up-z, camera-front-x)
    dist = 5.0
    pos = np.array([dist * np.cos(cam_phi) * np.cos(cam_theta), dist * np.cos(cam_phi) * np.sin(cam_theta), dist * np.sin(cam_phi)])


    out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
    # with Image.open(os.path.join(cur_dir, 'interaction_mask_%d.png' % result_idx)) as fimg:
    #     out3 = (np.array(fimg, dtype=np.float32) > 127)
    pt = out[x, y, :3]
    ptid = np.array([x, y], dtype=np.int32)
    mask = (out[:, :, 3] > 0.5)
    mask[x, y] = False
    pc = out[mask, :3]
    # pcids = grid_xy[:, mask].T
    # out3 = out3[mask]
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000 - 1]
    pc = pc[idx, :]
    pc = np.vstack([pt, pc])
    pc[:, 0] -= 5   # cam

    input_pcs = torch.tensor(pc, dtype=torch.float32).reshape(1, 30000, 3).to(device)
    # input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)  # B x 3N x 3   # point cloud
    batch_size = 1

    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, args.num_point_per_shape).long().reshape(-1)  # BN
    if args.sample_type == 'fps':
        input_pcid2 = furthest_point_sample(input_pcs, args.num_point_per_shape).long().reshape(-1)  # BN
    elif args.sample_type == 'random':
        pcs_id = ()
        for batch_idx in range(input_pcs.shape[0]):
            idx = np.arange(input_pcs[batch_idx].shape[0])
            np.random.shuffle(idx)
            while len(idx) < args.num_point_per_shape:
                idx = np.concatenate([idx, idx])
            idx = idx[:args.num_point_per_shape]
            pcs_id = pcs_id + (torch.tensor(np.array(idx)), )
        input_pcid2 = torch.stack(pcs_id, dim=0).long().reshape(-1)
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
    pc = input_pcs[0].detach().cpu().numpy()

    if epoch % num_same_ctpt == 0:
        save_h5(os.path.join(save_dir, 'cam_XYZA_final%d_%d.h5' % (epoch // num_same_ctpt + args.num_offset,task_degree)),
                [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                 (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                 (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])
        Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
            os.path.join(save_dir, 'interaction_mask_final%d_%d.png' % (epoch // num_same_ctpt + args.num_offset,task_degree)))
env.close()