import os
import sys
import shutil
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils
from utils import get_global_position_from_camera
from sapien.core import Pose
from eval_env import Env
from camera import Camera
from robots.panda_robot import Robot
import imageio
import cv2
import datetime

from pointnet2_ops.pointnet2_utils import furthest_point_sample

# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--model_version', type=str)
parser.add_argument('--model_epoch', type=int, help='epoch')
parser.add_argument('--shape_id', type=str, help='shape id')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--algorithm', type=str, default='w2a')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
eval_conf = parser.parse_args()


# load train config
train_conf = torch.load(os.path.join('logs', eval_conf.exp_name, 'conf.pth'))

# load model
model_def = utils.get_model_module(eval_conf.model_version)

# set up device
device = torch.device(eval_conf.device)
print(f'Using device: {device}')
start = datetime.datetime.now()
start = start.strftime("%Y-%m-%d-%H:%M:%S")
print(start)
# check if eval results already exist. If so, delete it.
result_dir = os.path.join('logs', eval_conf.exp_name, f'visu_action_heatmap_proposals-{eval_conf.shape_id}-model_epoch_{eval_conf.model_epoch}-{eval_conf.result_suffix}-{start}')
if os.path.exists(result_dir):
    if not eval_conf.overwrite:
        response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
        if response != 'y':
            sys.exit()
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
print(f'\nTesting under directory: {result_dir}\n')

# create models
network = model_def.Network(train_conf.feat_dim, train_conf.rv_dim, train_conf.rv_cnt)

# load pretrained model
print('Loading ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), eval_conf.model_epoch)
data_to_restore = torch.load(os.path.join('logs', eval_conf.exp_name, 'ckpts', '%d-network.pth' % eval_conf.model_epoch))
network.load_state_dict(data_to_restore, strict=False)
print('DONE\n')

# send to device
network.to(device)

# set models to evaluation mode
network.eval()

# setup env
env = Env()

# setup camera
cam = Camera(env, theta=np.pi/3 *2.3, phi=np.pi/5)
#cam = Camera(env, fixed_position=True)
#cam = Camera(env)
env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)
mat33 = cam.mat44[:3, :3]

# load shape
object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % eval_conf.shape_id
object_material = env.get_material(4, 4, 0.01)
state = 'open'
if np.random.random() < 0.5:
    state = 'closed'
print('Object State: %s' % state)
state = "indicated"
articu_angle = np.pi/2 * (1/2)
flag = env.load_object(object_urdf_fn, object_material, articu_angle, state=state)
cur_qpos = env.get_object_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    cur_new_qpos = env.get_object_qpos()
    invalid_contact = False
    for c in env.scene.get_contacts():
        for p in c.points:
            if abs(p.impulse @ p.impulse) > 1e-4:
                invalid_contact = True
                break
        if invalid_contact:
            break
    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Object Not Still!')
    flog.close()
    env.close()
    exit(1)


### use the GT vision
rgb, depth = cam.get_observation()
object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
gt_handle_mask = cam.get_handle_mask()

# sample a pixel to interact
xs, ys = np.where(gt_movable_link_mask>0)
#xs, ys = np.where(gt_handle_mask>0)
if len(xs) == 0:
    print('No Movable Pixel! Quit!')
    env.close()
    exit(1)
idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(result_dir, 'rgb.png'))
marked_rgb = (rgb*255).astype(np.uint8)
marked_rgb = cv2.circle(marked_rgb, (y, x), radius=3, color=(0, 0, 255), thickness=5)
Image.fromarray(marked_rgb).save(os.path.join(result_dir, 'point_to_interact.png'))

# get pixel 3D position (world)
position_world = get_global_position_from_camera(cam, depth, y, x)[:3]
print('Position World: %f %f %f' % (position_world[0], position_world[1], position_world[2]))

# prepare input pc
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
pt = cam_XYZA[x, y, :3]
ptid = np.array([x, y], dtype=np.int32)
mask = (cam_XYZA[:, :, 3] > 0.5)
mask[x, y] = False
pc = cam_XYZA[mask, :3]
grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448
pcids = grid_xy[:, mask].T
pc_movable = (gt_movable_link_mask > 0)[mask]
idx = np.arange(pc.shape[0])
np.random.shuffle(idx)
while len(idx) < 30000:
    idx = np.concatenate([idx, idx])
idx = idx[:30000-1]
pc = pc[idx, :]
pc_movable = pc_movable[idx]
pcids = pcids[idx, :]
pc = np.vstack([pt, pc])
pc_movable = np.append(True, pc_movable)
pcids = np.vstack([ptid, pcids])
pc[:, 0] -= 5
pc = torch.from_numpy(pc).unsqueeze(0).to(device)

input_pcid = furthest_point_sample(pc, 2048).long().reshape(-1)
pc = pc[:, input_pcid, :3]  # 1 x N x 3
pc_movable = pc_movable[input_pcid.cpu().numpy()]     # N
pcids = pcids[input_pcid.cpu().numpy()]
pccolors = rgb[pcids[:, 0], pcids[:, 1]]

# push through unet
feats = network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F

# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot = Robot(env, robot_urdf_fn, robot_material)

with torch.no_grad():
    # push through the network
    if eval_conf.algorithm == 'w2a':
        pred_6d = network.inference_actor(pc)[0]  # RV_CNT x 6
    elif eval_conf.algorithm == 'vat':
        pred_6d, pred_dist = network.inference_actor(pc)  # RV_CNT x 6
        pred_6d = pred_6d[0]
        

    pred_Rs = network.actor.bgs(pred_6d.reshape(-1, 3, 2))    # RV_CNT x 3 x 3
    pred_action_score_map = network.inference_action_score(pc)[0] # N
    pred_action_score_map = pred_action_score_map.cpu().numpy()

    # save action_score_map
    # pc = mat33 @ pc
    world_pc = (mat33 @ pc[0].cpu().numpy().T).T
    fn = os.path.join(result_dir, 'action_score_map_full_world')
    utils.render_pts_label_png(fn,  world_pc, pred_action_score_map)
    # print(pc.shape, pred_action_score_map.shape)

    init_pc = pc[0].cpu().numpy()
    fn = os.path.join(result_dir, 'action_score_map_full_init')
    utils.render_pts_label_png(fn,  init_pc, pred_action_score_map)
    # print(pc.shape, pred_action_score_map.shape)

    pred_action_score_map *= pc_movable
    fn = os.path.join(result_dir, 'action_score_map')
    utils.render_pts_label_png(fn,  world_pc, pred_action_score_map)

    fn = os.path.join(result_dir, 'input')
    utils.export_pts_color_pts(fn,  world_pc, pccolors)
    utils.export_pts_color_obj(fn,  world_pc, pccolors)

    # show actor-results with critic-preds
    succ_images = []
    def plot_figure(idx, up, forward, result, final_dist = 0.05):
        # cam to world
        up = mat33 @ up
        forward = mat33 @ forward
        # print(final_dist)
        up = np.array(up, dtype=np.float32)
        forward = np.array(forward, dtype=np.float32)
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)
        rotmat = np.eye(4).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up
        rotmat[:3, 3] = position_world + up * final_dist - up * 0.15
        pose = Pose().from_transformation_matrix(rotmat)
        robot.robot.set_root_pose(pose)
        env.render()
        rgb_final_pose, _ = cam.get_observation()
        fimg = (rgb_final_pose*255).astype(np.uint8)
        fimg = Image.fromarray(fimg)
        
        if result > 0.5:
            succ_images.append(fimg)
            fimg.save(os.path.join(result_dir, 'SUCC-%d.png' % idx))
        else:
            fimg.save(os.path.join(result_dir, 'FAIL-%d.png' % idx))

    for i in range(train_conf.rv_cnt):
        gripper_direction_camera = pred_Rs[i:i+1, :, 0]
        gripper_forward_direction_camera = pred_Rs[i:i+1, :, 1]

        if eval_conf.algorithm == 'w2a':
            result_score = network.inference_critic(pc,gripper_direction_camera,gripper_forward_direction_camera,abs_val=True).item()
        elif eval_conf.algorithm == 'vat':
            pred_dist_j = pred_dist[:, i, :]
            result_score = network.inference_critic(pc,gripper_direction_camera,gripper_forward_direction_camera,pred_dist_j,abs_val=True).item()

        # result_score = network.inference_critic(pc, gripper_direction_camera, gripper_forward_direction_camera, abs_val=True).item()
        result = (result_score > 0.5)
        if eval_conf.algorithm == 'w2a':
            plot_figure(i, gripper_direction_camera[0].cpu().numpy(), gripper_forward_direction_camera[0].cpu().numpy(), result)
        else:
            plot_figure(i, gripper_direction_camera[0].cpu().numpy(), gripper_forward_direction_camera[0].cpu().numpy(), result, pred_dist_j.cpu().numpy())
    # export SUCC GIF Image
    try:
        imageio.mimsave(os.path.join(result_dir, 'all_succ.gif'), succ_images)
    except:
        pass

# close env
env.close()

