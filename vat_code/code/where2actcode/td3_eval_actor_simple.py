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
from models.model_3d_task_score_topk import ActionScore
from models.model_3d_task_critic_updir_RL import Network as Critic
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from models.model_pn2_ae import Network
import ipdb
# import traceback
# import faulthandler
# faulthandler.enable()
import torch.nn.functional as F
from data_task_traj_RL import SAPIENVisionDataset
import utils

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
parser.add_argument('--sameTask_diffProposal', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--wp_xyz', type=int, default=1)
parser.add_argument('--coordinate_system', type=str, default='cambase')
parser.add_argument('--sample_type', type=str, default='random')
parser.add_argument('--num_offset', type=int, default=0)
parser.add_argument('--affordance_dir', type=str, default='xxx')
parser.add_argument('--affordance_epoch', type=int, default=0)
parser.add_argument('--critic_dir', type=str, default='xxx')
parser.add_argument('--critic_epoch', type=str, default='0')
parser.add_argument('--val_data_dir', type=str, help='data directory')
parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir7', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir8', type=str, default='xxx', help='data directory')
parser.add_argument('--val_data_dir9', type=str, default='xxx', help='data directory')
parser.add_argument('--val_num_data_uplimit', type=int, default=100000)
parser.add_argument('--angle_system', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='save_dir', help='data directory')
parser.add_argument('--saved_json_dir', type=str, default=None)

parser.add_argument('--drawer', type=int, default=0)
parser.add_argument('--slider', type=int, default=0)


def bgs(d6s):
    # print(d6s[0, 0, 0] *d6s[0, 0, 0] + d6s[0, 1, 0] * d6s[0, 1, 0] + d6s[0, 2, 0] *d6s[0, 2, 0])
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    # print(torch.stack([b1, b2, b3], dim=1).shape)
    # print(torch.stack([b1, b2, b3], dim=1)[0])
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)



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
if args.drawer + args.slider != 1:
    raise ValueError
if args.primact_type == 'pushing' or args.primact_type == 'pulling':
    if args.slider:
        all_cat_list = ['Window']
        eval_cat_list = ['StorageFurniture']
    if args.drawer:
        all_cat_list = ['StorageFurniture']
        eval_cat_list = ['Safe', 'WashingMachine']

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

EP_MAX = 5000
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

# task = np.pi * args.task / 180
pos_range = args.pos_range
rot_range = np.pi * 45 / 180
action_range = torch.tensor([pos_range, pos_range, pos_range, rot_range, rot_range, rot_range]).to(device)
action_dim = 6
state_dim = 1 + 1 + 1 + 3 + 3 + 8  # cur_obj_qpos, dis_to_target, cur_gripper_info, cur_step_idx, final_task(degree), contact_point_xyz, gripper_xyz
if args.with_step:
    state_dim += args.num_steps + 1
if args.state_initial_position:
    state_dim += 3
if args.state_initial_dir:
    state_dim += 9
if args.state_initial_up_dir:
    state_dim += 3
if args.state_joint_origins:
    state_dim += 3
if args.state_ctpt_dis_to_joint:
    state_dim += 1
if args.pn_feat:
    state_dim += args.pn_feat_dim
if args.state_axes:
    state_dim += 1
if args.state_axes_all:
    state_dim += 3
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


# replay_buffer_size = args.replay_buffer_size
# replay_buffer = ReplayBuffer(replay_buffer_size)
# td3 = TD3(replay_buffer, state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, action_range=action_range, q_lr=args.q_lr, policy_lr=args.policy_lr, device=device, pred_world_xyz=args.pred_world_xyz).to(device)

if args.pn_feat:
    PNPP = Network(feat_dim=args.pn_feat_dim).to(device)
    pnpp_ckpt_path = args.pnpp_ckpt_path
    saved_pnpp = torch.load(pnpp_ckpt_path)
    PNPP.load_state_dict(saved_pnpp)
    PNPP.train()

shape_id = args.shape_id
trial_id = args.trial_id
primact_type = args.primact_type

# out_dir = os.path.join(args.out_dir,
#                        '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))
out_dir = args.out_dir % trial_id
print('out_dir: ', out_dir)

save_dir = os.path.join(out_dir, args.save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

result_succ_dir = os.path.join(save_dir, 'result_succ_imgs')
if not os.path.exists(result_succ_dir):
    os.mkdir(result_succ_dir)
result_fail_dir = os.path.join(save_dir, 'result_fail_imgs')
if not os.path.exists(result_fail_dir):
    os.mkdir(result_fail_dir)


# load actor
actor = ActorNetwork(feat_dim=128, num_steps=5).to(device)
actor.load_state_dict(torch.load(os.path.join(out_dir, 'ckpts', '%d-network.pth' % args.eval_epoch)))
actor.eval()

affordance = ActionScore(feat_dim=128).to(device)
affordance.load_state_dict(torch.load(args.affordance_dir % args.affordance_epoch))
affordance.eval()
# torch.save(affordance.state_dict(), args.affordance_dir % (args.affordance_epoch * -1), _use_new_zipfile_serialization=False)
# exit(0)

critic = Critic(feat_dim=128, num_steps=5).to(device)
critic.load_state_dict(torch.load(os.path.join(args.critic_dir, 'ckpts', '%s-network.pth' % args.critic_epoch)))
critic.eval()


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

# all_reward = []
target_part_state = args.target_part_state

if args.pred_world_xyz + args.pred_residual_world_xyz + args.pred_residual_root_qpos + args.pred_residual_cambase_xyz != 1:
    raise ValueError

tot_succ_epoch = -1
tot_done_epoch = 0
tot_fail_epoch = 0
tot_contact_error, tot_grasp_error, tot_not_fitable_axis = 0, 0, 0


saved_num = 0

robot_loaded = 0
object_material = env.get_material(4, 4, 0.01)

valid_shape = [0 for idx in range(len_train_shape_list + len_val_shape_list)]
num_same_ctpt = 500


# saved_shape_id = 45297
# saved_cam_theta, saved_cam_phi = 2.0139454731613187, 0.636293728762917
# saved_task_degree = 30
# saved_joint_angles = [1.510947823524475 - degree2radian(saved_task_degree)]
# saved_x, saved_y = 383, 362


# saved_shape_id = 45850
# saved_cam_theta, saved_cam_phi = 2.704867919059725, 0.5380882738937182
# saved_task_degree = 32.01195599825837
# saved_joint_angles = [1.0382184982299805 - degree2radian(saved_task_degree)]
# saved_x, saved_y = 302, 258
#
# found_contact_point = False
# saved_aff_pos = None

if args.sameTask_diffProposal:
    with open(args.saved_json_dir, 'r') as fin:
        result_data = json.load(fin)
    saved_shape_id = result_data['shape_id']
    saved_camera_metadata = result_data['camera_metadata']
    saved_cam_theta, saved_cam_phi = saved_camera_metadata['theta'], saved_camera_metadata['phi']
    saved_task_degree = result_data['actual_task']
    saved_joint_angles = result_data['joint_angles']
    saved_pixel_idx = result_data['pixel_locs']
    saved_x, saved_y = saved_pixel_idx[0], saved_pixel_idx[1]
    found_contact_point = False
    saved_aff_pos = None


# load data
val_data_list = []
for root, dirs, files in os.walk(args.val_data_dir):
    for dir in dirs:
        val_data_list.append(os.path.join(args.val_data_dir, dir))
    break
if args.val_data_dir2 != 'xxx':
    for root, dirs, files in os.walk(args.val_data_dir2):
        for dir in dirs:
            val_data_list.append(os.path.join(args.val_data_dir2, dir))
        break
if args.val_data_dir3 != 'xxx':
    for root, dirs, files in os.walk(args.val_data_dir3):
        for dir in dirs:
            val_data_list.append(os.path.join(args.val_data_dir3, dir))
        break
if args.val_data_dir4 != 'xxx':
    for root, dirs, files in os.walk(args.val_data_dir4):
        for dir in dirs:
            val_data_list.append(os.path.join(args.val_data_dir4, dir))
        break
if args.val_data_dir5 != 'xxx':
    for root, dirs, files in os.walk(args.val_data_dir5):
        for dir in dirs:
            val_data_list.append(os.path.join(args.val_data_dir5, dir))
        break
if args.val_data_dir6 != 'xxx':
    for root, dirs, files in os.walk(args.val_data_dir6):
        for dir in dirs:
            val_data_list.append(os.path.join(args.val_data_dir6, dir))
        break
if args.val_data_dir7 != 'xxx':
    for root, dirs, files in os.walk(args.val_data_dir7):
        for dir in dirs:
            val_data_list.append(os.path.join(args.val_data_dir7, dir))
        break


data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction', 'gripper_forward_direction', \
        'result', 'task_motion', 'gt_motion', 'task_waypoints', 'cur_dir', 'shape_id', 'trial_id', 'is_original', 'position', 'camera_metadata', 'joint_angles', 'ori_pixel_ids']
val_dataset = SAPIENVisionDataset([args.primact_type], [], data_features, buffer_max_num=512,
                                  img_size=224, only_true_data=True,
                                  no_true_false_equal=False, angle_system=0,
                                  EP_MAX=30000, degree_lower=10,
                                  cur_primact_type=args.primact_type, critic_mode=False, train_mode=False)
val_dataset.load_data(val_data_list, wp_xyz=args.wp_xyz, coordinate_system=args.coordinate_system,
                      num_data_uplimit=args.val_num_data_uplimit)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                             num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                             worker_init_fn=utils.worker_init_fn)
val_batches = enumerate(val_dataloader, 0)

succ_cnt = 0

for epoch in range(EP_MAX):
    print('epoch: ', epoch)
    torch.cuda.empty_cache()
    if epoch >= 20 and succ_cnt==0:
        break
    if epoch >= 50 and saved_num==0:
        break
    batch_ind, batch = next(val_batches)

    shape_id = batch[data_features.index('shape_id')][0]
    if args.sameTask_diffProposal:
        shape_id = saved_shape_id
    print('shape_id: ', shape_id, type(shape_id))

    # if shape_id != '103452':
    #     continue

    camera_metadata = batch[data_features.index('camera_metadata')][0]
    cam_theta, cam_phi = camera_metadata['theta'], camera_metadata['phi']
    if args.sameTask_diffProposal:
        cam_theta, cam_phi = saved_cam_theta, saved_cam_phi
    cam.change_pose(phi=cam_phi, theta=cam_theta, random_position=False, restrict_dir=True)
    if not args.no_gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

    torch.cuda.empty_cache()
    record[(tot_succ_epoch + 1) % 100] = 0

    object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
    # state = 'random-closed-middle'
    target_part_state = args.target_part_state
    joint_angles = env.load_object(object_urdf_fn, object_material, state=target_part_state, target_part_id=-1)
    joint_angles = batch[data_features.index('joint_angles')][0].tolist()
    if args.sameTask_diffProposal:
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
        continue

    idx = np.random.randint(len(xs))
    # x, y = xs[idx], ys[idx]
    # target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    # env.set_target_object_part_actor_id(target_part_id)
    # tot_trial = 0
    # while tot_trial < 100 and ((env.target_object_part_joint_type != ArticulationJointType.REVOLUTE) or (
    #         env.get_target_part_axes_dir(target_part_id) != 1)):
    #     idx = np.random.randint(len(xs))
    #     x, y = xs[idx], ys[idx]
    #     target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    #     env.set_target_object_part_actor_id(target_part_id)
    #     tot_trial += 1
    # old_target_part_id = target_part_id
    pixel_locs = batch[data_features.index('ori_pixel_ids')][0]
    x, y = pixel_locs[0], pixel_locs[1]
    if args.sameTask_diffProposal:
        x, y = saved_x, saved_y
    target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    old_target_part_id = target_part_id
    env.set_target_object_part_actor_id(target_part_id)

    if (env.target_object_part_joint_type != ArticulationJointType.PRISMATIC):
        env.scene.remove_articulation(env.object)
        # valid_shape[idx_shape] = -1
        continue
    print("old_axes:", env.get_target_part_axes(target_part_id))
    print("old_origins:", env.get_target_part_origins_new(target_part_id))
    target_part_joint_idx = env.target_object_part_joint_id

    joint_angle_lower = env.joint_angles_lower[target_part_joint_idx]
    joint_angle_upper = env.joint_angles_upper[target_part_joint_idx]
    # joint_angle_lower_degree = radian2degree(joint_angle_lower)
    # joint_angle_upper_degree = radian2degree(joint_angle_upper)
    task_upper = min(joint_angle_upper - joint_angle_lower, args.task_upper)

    task_lower = args.task_lower
    # task_lower = max(args.task_lower, radian2degree(target_part_qpos))    # task = np.pi * 30 / 180
    task_degree = random.random() * (task_upper - task_lower) + task_lower
    task_degree = batch[data_features.index('gt_motion')][0]
    if args.sameTask_diffProposal:
        task_degree = saved_task_degree
    # if args.primact_type == 'pulling':
    #     task_degree = -task_degree
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
    # gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
    gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

    # sample a pixel to interact
    xs, ys = np.where(gt_movable_link_mask > 0)
    if len(xs) == 0:
        env.scene.remove_articulation(env.object)
        continue

    idx = np.random.randint(len(xs))
    # x, y = xs[idx], ys[idx]
    # target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    # env.set_target_object_part_actor_id(target_part_id)
    # tot_trial = 0
    # while tot_trial < 100 and target_part_id != old_target_part_id:
    #     idx = np.random.randint(len(xs))
    #     x, y = xs[idx], ys[idx]
    #     target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    #     env.set_target_object_part_actor_id(target_part_id)
    #     tot_trial += 1
    # if target_part_id != old_target_part_id:
    #     env.scene.remove_articulation(env.object)
    #     flog.close()
    #     continue
    pixel_locs = batch[data_features.index('ori_pixel_ids')][0]
    x, y = pixel_locs[0], pixel_locs[1]
    if args.sameTask_diffProposal:
        x, y = saved_x, saved_y
    target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
    env.set_target_object_part_actor_id(target_part_id)
    gt_movable_link_mask = cam.get_movable_link_mask([target_part_id])
    print("x, y:", x, y)

    joint_origins = env.get_target_part_origins_new(target_part_id=target_part_id)
    joint_axes = env.get_target_part_axes(target_part_id=target_part_id)
    # print("joint_origins:", joint_origins)
    # print("joint_axes", joint_axes)
    axes_dir = env.get_target_part_axes_dir_new(target_part_id)
    if args.drawer:
        if axes_dir != 0:
            tot_not_fitable_axis += 1
            # for i in range(10000):
            #     env.step()
            #     env.render()
            # robot.wait_n_steps(10000)
            env.scene.remove_articulation(env.object)
            continue
    if args.slider:
        if axes_dir != 1:
            print("axes_dir != 1")
            tot_not_fitable_axis += 1
            # for i in range(10000):
            #     env.step()
            #     env.render()
            # robot.wait_n_steps(10000)
            env.scene.remove_articulation(env.object)
            continue

    # get pixel 3D pulling direction (cam/world)
    mat44 = cam.get_metadata()['mat44']
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
    print('position_world: ', position_world, type(position_world))

    # state_joint_origins = joint_origins
    # # !!!!!!!!!!
    # # state_joint_origins[axes_dir] = position_world[axes_dir]
    # state_joint_origins[-1] = position_world[-1]
    # state_ctpt_dis_to_joint = np.linalg.norm(state_joint_origins - position_world)
    # state_door_dir = position_world - state_joint_origins

    # compute camera-base frame (camera-center, world-up-z, camera-front-x)
    dist = 5.0
    pos = np.array([dist * np.cos(cam_phi) * np.cos(cam_theta), dist * np.cos(cam_phi) * np.sin(cam_theta), dist * np.sin(cam_phi)])
    cb_up = np.array([0, 0, 1], dtype=np.float32)
    cb_left = np.cross(cb_up, action_direction_cam)
    cb_left /= np.linalg.norm(cb_left)
    cb_forward = np.cross(cb_left, cb_up)
    cb_forward /= np.linalg.norm(cb_forward)
    base_mat44 = np.eye(4)
    base_mat44[:3, :3] = np.vstack([cb_forward, cb_left, cb_up]).T
    base_mat44[:3, 3] = pos  # cambase2world
    cam2cambase = np.linalg.inv(base_mat44) @ cam.get_metadata()['mat44']  # cam2cambase
    cam2cambase = cam2cambase[:3, :3]
    position_cam -= np.array([5, 0, 0])
    cb_position = (cam2cambase @ position_cam.T).T


    out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
    out3 = (gt_movable_link_mask > 0).astype(np.uint8) * 255
    out3 = np.array(out3, dtype=np.float32) > 127
    pt = out[x, y, :3]
    ptid = np.array([x, y], dtype=np.int32)
    mask = (out[:, :, 3] > 0.5)
    mask[x, y] = False
    pc = out[mask, :3]
    # pcids = grid_xy[:, mask].T
    out3 = out3[mask]
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000 - 1]
    out3 = out3[idx]
    pc = pc[idx, :]
    pc = np.vstack([pt, pc])
    out3 = np.append(True, out3)
    input_movables = out3
    pc[:, 0] -= 5   # cam, norm

    input_pcs = torch.tensor(pc, dtype=torch.float32).reshape(1, 30000, 3).to(device)
    input_movables = torch.tensor(input_movables, dtype=torch.float32).reshape(1, 30000, 1).to(device)
    # input_pcs = batch[data_features.index('pcs')][0].to(device)  # B x 3N x 3   # point cloud
    # input_movables = batch[data_features.index('pc_movables')][0].reshape(1, 30000, 1).to(device)  # B x 3N  # movable part
    # print('input_pcs', input_pcs.shape)
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
    input_movables = input_movables[input_pcid1, input_pcid2, :].reshape(batch_size, args.num_point_per_shape, -1)
    pc = input_pcs[0].detach().cpu().numpy()


    if args.coordinate_system == 'world':
        world_pcs = (cam.get_metadata()['mat44'][:3, :3] @ pc.T).T
        world_pcs = torch.from_numpy(np.array(world_pcs, dtype=np.float32)).unsqueeze(0).float().to(device)
    elif args.coordinate_system == 'cam':
        cam_pcs = torch.from_numpy(pc).unsqueeze(0)
    elif args.coordinate_system == 'cambase':
        cb_pc = (cam2cambase @ pc.T).T    # cambase
        cb_pcs = torch.from_numpy(np.array(cb_pc, dtype=np.float32)).unsqueeze(0)


    ################### 出需要的图, 比如input的一帧，结束的pc
    if epoch == 0:
        rgb_pose, _ = cam.get_observation()
        fimg = (rgb_pose * 255).astype(np.uint8)
        fimg = Image.fromarray(fimg)
        fimg.save(os.path.join(save_dir, 'BEGIN_%d.png' % args.num_offset))  # first frame

        save_h5(os.path.join(save_dir, 'BEGIN_%d.h5' % args.num_offset),
                [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                 (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                 (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])
        Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
            os.path.join(save_dir, 'BEGIN_%d.png' % args.num_offset))

    # exit(0)
    ######################


    if args.coordinate_system == 'world':
        task_degree_tensor = torch.from_numpy(np.array(task_degree)).float().view(1, 1).to(device)

        with torch.no_grad():
            pred_action_score_map = affordance.inference_action_score(world_pcs, task_degree_tensor)
            pred_action_score_map = pred_action_score_map.cpu().numpy()
        pred_action_score_map = pred_action_score_map * input_movables.cpu().numpy()
        aff_max_idx = np.argmax(pred_action_score_map)
        aff_pos = world_pcs.view(-1, 3)[aff_max_idx]

        if args.sameTask_diffProposal:
            if not found_contact_point:
                print('here')
                pred_action_score_map = pred_action_score_map.reshape(-1)
                aff_score_sorted_index = np.argsort(-pred_action_score_map)
                print(pred_action_score_map)
                print(aff_score_sorted_index)
                num_aff_pts = torch.sum(input_movables).item()
                selected_idx = np.random.randint(0, int(num_aff_pts * 0.002))
                aff_pos = world_pcs.view(-1, 3)[aff_score_sorted_index[selected_idx]]
                found_contact_point = True
                saved_aff_pos = aff_pos
            else:
                aff_pos = saved_aff_pos

        aff_pos_tensor = aff_pos.view(1, 3)
        position_world = aff_pos.reshape(3).detach().cpu().numpy()
        # position_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ position_world
        # position_cam += [5, 0, 0]   # un-norm
        # position_world = cam.get_metadata()['mat44'][:3, :3] @ position_cam
        # aff_pos_tensor = torch.from_numpy(aff_pos).view(1, -1).to(device)
        with torch.no_grad():
            if not args.sameTask_diffProposal:
                traj = actor.sample_n(world_pcs, task_degree_tensor, aff_pos_tensor, rvs=100)
                gt_score = torch.sigmoid(critic.forward_n(world_pcs, task_degree_tensor, traj, aff_pos_tensor, rvs=100)[0])
                gt_score = gt_score.view(1, 100, 1).max(dim=1)
            else:
                traj = actor.sample_n(world_pcs, task_degree_tensor, aff_pos_tensor, rvs=10)
                gt_score = torch.sigmoid(critic.forward_n(world_pcs, task_degree_tensor, traj, aff_pos_tensor, rvs=10)[0])
                gt_score = gt_score.view(1, 10, 1).max(dim=1)
            # print(gt_score[1])
        recon_traj = traj[gt_score[1]][0]
        recon_dir = recon_traj[:, 0, :]
        recon_dir = recon_dir.reshape(-1, 3, 2)
        recon_dir = bgs(recon_dir)
        recon_wps = recon_traj[:, 1:, :]

    recon_wps = recon_wps.detach().cpu().numpy()
    recon_dir = recon_dir.detach().cpu().numpy()

    up = recon_dir[0, :, 0]
    forward = recon_dir[0, :, 1]
    left = recon_dir[0, :, 2]

    if args.coordinate_system == 'cambase':
        up = (np.linalg.inv(cam2cambase) @ up.T).T
        forward = (np.linalg.inv(cam2cambase) @ forward.T).T
        left = (np.linalg.inv(cam2cambase) @ left.T).T
    if args.coordinate_system == 'cambase' or args.coordinate_system == 'cam':
        up = (cam.get_metadata()['mat44'][:3, :3] @ up.T).T
        forward = (cam.get_metadata()['mat44'][:3, :3] @ forward.T).T
        left = (cam.get_metadata()['mat44'][:3, :3] @ left.T).T

    # if args.pred_residual_cambase_xyz:
    #     # cam
    #     up = (np.linalg.inv(cam2cambase) @ up.T).T
    #     forward = (np.linalg.inv(cam2cambase) @ forward.T).T
    #     left = (np.linalg.inv(cam2cambase) @ left.T).T
    #     # world
    #     up = (cam.get_metadata()['mat44'][:3, :3] @ up.T).T
    #     forward = (cam.get_metadata()['mat44'][:3, :3] @ forward.T).T
    #     left = (cam.get_metadata()['mat44'][:3, :3] @ left.T).T

    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    action_direction_world = up

    start_rotmat = np.array(rotmat, dtype=np.float32)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if args.use_direction_world:
        start_rotmat[:3, 3] = position_world - (-direction_world) * 0.15  # add displacement(lase column)
        start_pose = Pose().from_transformation_matrix(start_rotmat)
        start_gripper_root_position = position_world - (-direction_world) * 0.15
    else:
        start_rotmat[:3, 3] = position_world - action_direction_world * 0.15  # add displacement(lase column)
        start_pose = Pose().from_transformation_matrix(start_rotmat)
        start_gripper_root_position = position_world - action_direction_world * 0.15

    end_rotmat = start_rotmat.copy()
    end_rotmat[:3, 3] = position_world - action_direction_world * 0.08

    if robot_loaded == 0:
        robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
        robot_loaded = 1
    else:
        robot.load_gripper(robot_urdf_fn, robot_material)
    env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)

    state_joint_origins = joint_origins
    # !!!!!!!!!!
    # state_joint_origins[axes_dir] = position_world[axes_dir]
    state_joint_origins[-1] = position_world[-1]
    state_ctpt_dis_to_joint = np.linalg.norm(state_joint_origins - position_world)
    state_door_dir = position_world - state_joint_origins

    final_distance = 100
    t0 = time.time()

    out_info = dict()
    out_info['out_dir'] = out_dir
    out_info['shape_id'] = shape_id
    out_info['category'] = args.category
    out_info['cnt_id'] = args.cnt_id
    out_info['primact_type'] = args.primact_type
    out_info['trial_id'] = args.trial_id

    out_info['random_seed'] = args.random_seed
    out_info['pixel_locs'] = [int(x), int(y)]
    out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
    out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
    if env.target_object_part_joint_type == ArticulationJointType.REVOLUTE:
        out_info['target_object_part_joint_type'] = "REVOLUTE"
    elif env.target_object_part_joint_type == ArticulationJointType.PRISMATIC:
        out_info['target_object_part_joint_type'] = 'PRISMATIC'
    else:
        out_info['target_object_part_joint_type'] = str(env.target_object_part_joint_type)
    out_info['direction_camera'] = direction_cam.tolist()
    out_info['direction_world'] = direction_world.tolist()
    out_info['mat44'] = str(cam.get_metadata()['mat44'])
    out_info['gripper_direction_camera'] = action_direction_cam.tolist()
    out_info['gripper_direction_world'] = action_direction_world.tolist()
    out_info['position_cam'] = position_cam.tolist()
    out_info['position_world'] = position_world.tolist()
    out_info['gripper_forward_direction_world'] = forward.tolist()
    # out_info['gripper_forward_direction_camera'] = forward_cam.tolist()

    # out_info['pred_world_xyz'] = args.pred_world_xyz
    # out_info['pred_residual_world_xyz'] = args.pred_residual_world_xyz

    # setup camera
    # cam = Camera(env, random_position=True)

    out_info['camera_metadata'] = cam.get_metadata_json()

    out_info['object_state'] = target_part_state
    # if joint_angles is None:
    #    joint_angles = env.load_object(object_urdf_fn, object_material, state=state)
    # else:
    #    print(joint_angles)
    # print("LOADED shape")
    out_info['joint_angles'] = joint_angles
    out_info['joint_angles_lower'] = env.joint_angles_lower
    out_info['joint_angles_upper'] = env.joint_angles_upper
    out_info['start_rotmat_world'] = start_rotmat.tolist()

    # out_info['start_rotmat_world'] = start_rotmat.tolist()
    # print("position_world", position_world)
    # print("action_direction_world", action_direction_world)
    # print("start:", start_rotmat)

    # move back
    # env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)
    robot.robot.set_root_pose(start_pose)
    env.render()
    # robot.wait_n_steps(1000000000000)

    # activate contact checking
    env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)



    if args.primact_type == 'pulling':
        init_success = True
        success_grasp = False
        try:
            robot.open_gripper()
            robot.move_to_target_pose(end_rotmat, 3000)
            robot.wait_n_steps(2000)
            robot.close_gripper()
            robot.wait_n_steps(600)
            now_qpos = robot.robot.get_qpos().tolist()
            finger1_qpos = now_qpos[-1]
            finger2_qpos = now_qpos[-2]
            # print(finger1_qpos, finger2_qpos)
            if finger1_qpos + finger2_qpos > 0.01:
                success_grasp = True
        except Exception:
            init_success = False
        if not (success_grasp and init_success):
            tot_grasp_error += 1
            print('grasp error!')
            env.scene.remove_articulation(env.object)
            env.scene.remove_articulation(robot.robot)
            continue


    if not args.no_gui:
        ### wait to start
        env.wait_to_start()
        pass

    ### main steps
    out_info['start_target_part_qpos'] = env.get_target_part_qpos()
    init_target_part_qpos = env.get_target_part_qpos()

    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

    num_steps = args.num_steps  # waypoints = num_step + 1
    out_info["num_steps"] = num_steps + 1
    step_one_hot = np.eye(num_steps + 1)
    waypoints = []
    dense_waypoints = []

    success = True
    robot.close_gripper()

    gripper_finger_position = position_world - 0.02 * action_direction_world

    try:
        robot.wait_n_steps(400)    # 尝试一下看看
        succ_images = []
        rgb_pose, _ = cam.get_observation()
        fimg = (rgb_pose*255).astype(np.uint8)
        fimg = Image.fromarray(fimg)
        succ_images.append(fimg)
        begin_img = fimg
        # fimg.save(os.path.join(save_dir, '%d_%d.png' % (args.num_offset, epoch)))  # first frame

        for step_idx in range(num_steps):
            waypoint = recon_wps[0, step_idx]
            print(waypoint)

            # get rotmat and move
            final_rotmat = start_rotmat.copy()
            # if args.pred_residual_root_qpos:
            #     final_rotmat[:3, 3] += waypoint[0] * forward + waypoint[1] * left + waypoint[2] * up
            # if args.pred_residual_world_xyz:
            #     final_rotmat[0, 3] += waypoint[0]
            #     final_rotmat[1, 3] += waypoint[1]
            #     final_rotmat[2, 3] += waypoint[2]
            # if args.pred_residual_cambase_xyz:
            #     waypoint[0:3] = (np.linalg.inv(cam2cambase) @ waypoint[0:3].T).T  # cambase2cam
            #     waypoint[0:3] = (cam.get_metadata()['mat44'][:3, :3] @ waypoint[0:3].T).T     # cam2world
            #     final_rotmat[0, 3] += waypoint[0]
            #     final_rotmat[1, 3] += waypoint[1]
            #     final_rotmat[2, 3] += waypoint[2]
            # if args.pred_world_xyz:
            #     final_rotmat[0, 3] = waypoint[0]
            #     final_rotmat[1, 3] = waypoint[1]
            #     final_rotmat[2, 3] = waypoint[2]

            if args.wp_xyz == 0:
                final_rotmat[:3, 3] += waypoint[0] * forward + waypoint[1] * left + waypoint[2] * up
            elif args.coordinate_system == 'world':
                final_rotmat[0, 3] += waypoint[0]
                final_rotmat[1, 3] += waypoint[1]
                final_rotmat[2, 3] += waypoint[2]
            # elif args.coordinate_system == 'cam':
            #     waypoint[0:3] = (cam.get_metadata()['mat44'][:3, :3] @ waypoint[0:3].T).T  # cam2world
            #     final_rotmat[0, 3] += waypoint[0]
            #     final_rotmat[1, 3] += waypoint[1]
            #     final_rotmat[2, 3] += waypoint[2]
            # elif args.coordinate_system == 'cambase':
            #     waypoint[0:3] = (np.linalg.inv(cam2cambase) @ waypoint[0:3].T).T  # cambase2cam
            #     waypoint[0:3] = (cam.get_metadata()['mat44'][:3, :3] @ waypoint[0:3].T).T  # cam2world
            #     final_rotmat[0, 3] += waypoint[0]
            #     final_rotmat[1, 3] += waypoint[1]
            #     final_rotmat[2, 3] += waypoint[2]

            if args.wp_rot:
                try:
                    # print("wps:", waypoint[3], waypoint[4], waypoint[5])
                    r = R.from_euler('XYZ', [waypoint[3], waypoint[4], waypoint[5]], degrees=False)
                    final_rotmat[:3, :3] = final_rotmat[:3, :3] @ r.as_matrix()
                except Exception:
                    success = False
                    print("?!!!!!!!!!!!!!!!!!!")
                    # print("wps!:", waypoint[3], waypoint[4], waypoint[5])
                    # ipdb.set_trace()
                    break

            try:
                imgs, cur_waypoints = robot.move_to_target_pose(final_rotmat, 2500, cam=cam, vis_gif=True, vis_gif_interval=500, visu=True)
                cur_waypoints2 = robot.wait_n_steps(600, visu=True)
                dense_waypoints.extend(cur_waypoints)
                dense_waypoints.extend(cur_waypoints2)
                if args.primact_type == 'pulling':
                    robot.close_gripper()
                    cur_waypoints3 = robot.wait_n_steps(400, visu=True)
                    dense_waypoints.extend(cur_waypoints3)
            except Exception:
                success = False
                break
            cur_waypoint = robot.robot.get_qpos().tolist()

            if args.out_gif:
                succ_images.extend(imgs)

            ''' calculate reward  (use radian) '''
            stop = False
            distance = np.abs((init_target_part_qpos - task) - env.get_target_part_qpos())
            final_distance = radian2degree(distance)
            print("dis:", radian2degree((init_target_part_qpos - task) - env.get_target_part_qpos()))

            waypoints.append(cur_waypoint)
            dense_waypoints.append(cur_waypoint)

            if distance < degree2radian(abs(task_degree) * 0.22):  # bonus
                print("done!")
                record[(tot_succ_epoch + 1) % 100] = 1
                tot_done_epoch += 1
                stop = True
                end_step = step_idx + 1
                # export SUCC GIF Image
                if args.out_gif:
                    try:
                        imageio.mimsave(os.path.join(result_succ_dir, '%d_%d_%.3f_%.3f_%d.gif' % (args.num_offset, epoch, task_degree, radian2degree(init_target_part_qpos), step_idx+1)), succ_images)
                    except:
                        pass

            if step_idx == num_steps - 1 and distance >= degree2radian(abs(task_degree) * 0.22):
                stop = True
                end_step = step_idx + 1
                if tot_fail_epoch < tot_done_epoch:
                    tot_fail_epoch += 1
                    if args.out_gif:
                        try:
                            imageio.mimsave(os.path.join(result_fail_dir, '%d_%d_%.3f_%.3f_%.3f_%d.gif' % (args.num_offset, epoch, task_degree, radian2degree(init_target_part_qpos), radian2degree(distance), step_idx+1)), succ_images)
                        except:
                            pass

            # try:
            #     imageio.mimsave(os.path.join(result_tmp_dir, '%d_%.3f_%.3f_%d.gif' % (
            #     tot_done_epoch, task_degree, radian2degree(init_target_part_qpos), step_idx + 1)), succ_images)
            # except:
            #     pass
            # break

            if args.early_stop and stop:
                break

        rgb_pose, _ = cam.get_observation()
        fimg = (rgb_pose*255).astype(np.uint8)
        fimg = Image.fromarray(fimg)
        end_img = fimg

    except ContactError:
        success = False

    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
    out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
    out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()
    out_info['task'] = task_degree
    actual_task = radian2degree(init_target_part_qpos - env.get_target_part_qpos())
    out_info['actual_task'] = actual_task

    # close the file
    env.scene.remove_articulation(robot.robot)

    # if epoch % 100 == 0:
    #     save_h5(os.path.join(save_dir, 'cam_XYZA_%d.h5' % (epoch // num_same_ctpt + args.num_offset)),
    #             [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
    #              (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
    #              (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])
    #     Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
    #         os.path.join(save_dir, 'interaction_mask_%d.png' % (epoch // num_same_ctpt + args.num_offset)))
    succ_cnt = succ_cnt+1
    if success:
        out_info['result'] = 'VALID'
        out_info['final_target_part_qpos'] = env.get_target_part_qpos()
        out_info['part_motion'] = out_info['final_target_part_qpos'] - out_info['start_target_part_qpos']
        out_info['part_motion_degree'] = out_info['part_motion'] * 180.0 / 3.1415926535
    else:
        out_info['result'] = 'CONTACT_ERROR'
        out_info['part_motion'] = 0.0
        out_info['part_motion_degree'] = 0.0
        print('contact_error')
        tot_contact_error += 1
        env.scene.remove_articulation(env.object)
        continue
    env.scene.remove_articulation(env.object)

    tot_succ_epoch += 1
    out_info['waypoints'] = waypoints
    out_info['dense_waypoints'] = dense_waypoints
    # for idx in range(1000000000000000):
    #     env.render()

    # save results

    if abs(task_degree - actual_task) <= (abs(task_degree) * 0.22):
    # if True:
        with open(os.path.join(save_dir, 'result_%d_%d.json' % (args.num_offset, epoch)), 'w') as fout:
            json.dump(out_info, fout)
        save_h5(os.path.join(save_dir, 'cam_XYZA_%d_%d.h5' % (args.num_offset, epoch)),
                [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'),
                 (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'),
                 (cam_XYZA_pts.astype(np.float32), 'pc', 'float32')])
        Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
            os.path.join(save_dir, 'interaction_mask_%d_%d.png' % (args.num_offset, epoch)))
        begin_img.save(os.path.join(save_dir, 'begin_%d_%d.png' % (args.num_offset, epoch)))  # first frame
        end_img.save(os.path.join(save_dir, 'end_%d_%d.png' % (args.num_offset, epoch)))  # first frame
        saved_num += 1
        print('success_trajectory: ', saved_num)


    if args.no_gui:
        # close env
        # env.close()
        pass
    else:
        if success:
            print('[Successful Interaction] Done. Ctrl-C to quit.')
            ### wait forever
            # robot.wait_n_steps(100000000000)
            # env.close()
        else:
            print('[Unsuccessful Interaction] invalid gripper-object contact.')
            # close env
            # env.close()

    # loop 结束后还需要再更新一次state
    accuracy = sum(record) / len(record)
    print(
        'Episode: {}/{}  | Final Distance: {:.4f}  | Accuracy: {:.4f}  | Running Time: {:.4f}'.format(
            epoch + 1, EP_MAX, final_distance, accuracy, time.time() - t0
        )
    )
    print('accu: ', tot_done_epoch / (batch_ind + 1))
    print('num_contact_error: ', tot_contact_error)
    print('num_grasp_error: ', tot_grasp_error)
    # print('num_not_fitable_axis: ', tot_not_fitable_axis)
    print('\n')
env.close()
