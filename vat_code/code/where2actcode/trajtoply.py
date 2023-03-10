import os
import matplotlib
# from data import PartNetDataset
import numpy as np
import h5py
import json
import random
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append('../')
import utils
import matplotlib
import random

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import Camera
from PIL import Image
import argparse

Colors = ['blue', 'red', 'darkviolet', 'gold', 'lightpink', 'green', 'yellow']


# from rand_cmap import rand_cmap
# cmap = rand_cmap(300, type='bright', first_color_black=True, last_color_black=False, verbose=False)
def dist(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2])


def callen(p):
    L = len(p)
    ret = 0
    for i in range(L - 1):
        ret = ret + dist(p[i], p[i + 1])
        if dist(p[i], p[i + 1]) > 0.1:
            return -10
    ret = ret + dist(p[0], p[L - 1]) * 0.3
    return ret


def getdist(p1, p2):
    ret = 0
    L = 5
    for i in range(min(L, len(p1), len(p2))):
        ret = ret + dist(p1[i], p2[i])
    return ret


def load_semantic_colors(filename):
    semantic_colors = {}
    id2name = {}
    tot = -1
    with open(filename, 'r') as fin:
        for l in fin.readlines():
            tot += 1
            semantic, r, g, b = l.rstrip().split()
            semantic_colors[semantic] = (int(r), int(g), int(b))
            id2name[tot] = semantic
    id2name[-1] = '-1'
    semantic_colors['-1'] = (0, 0, 0)
    return semantic_colors, id2name


def draw_geo(ax, p, color, maker='.', s=20, alpha=0.5, rot=None):
    # print(p.shape)
    if rot is not None:
        p = (rot * p.transpose()).transpose()
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=[color], alpha=alpha, marker=maker, s=s)


def draw_direction(ax, cp, directions, rot=None):
    # print(cp)
    len_line = 0.2
    if rot is not None:
        cp = (rot * cp.reshape(-1, 3).T).T.A.reshape(3)
        directions = (rot * directions.T).T.A
    up_line = np.array([cp, cp + directions[0] * len_line])
    forward_line = np.array([cp, cp + directions[1] * len_line])
    left_line = np.array([cp, cp + directions[2] * len_line])

    # print(up_line)
    ax.plot(up_line[:, 0], up_line[:, 1], up_line[:, 2], c='blue', linewidth=3)
    ax.plot(forward_line[:, 0], forward_line[:, 1], forward_line[:, 2], c='yellow', linewidth=3)
    ax.plot(left_line[:, 0], left_line[:, 1], left_line[:, 2], c='green', linewidth=3)


def get_waypoints_cam(json_content):
    mat44 = np.array(json_content['camera_metadata']['mat44'])
    position_world = np.array(json_content['position_world'])  # contact_point
    # position_cam = (np.linalg.inv(mat44[:3, :3]) @ position_world.T).T
    # position_cam -= np.array([5, 0, 0])   # normalize to (0,0,0)
    # position_world = (mat44[:3, :3] @ position_world.T).T
    # waypoints = json_content['waypoints']
    waypoints = []
    up = np.array(json_content['gripper_direction_world'])
    forward = np.array(json_content['gripper_forward_direction_world'])
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    start_rotmat_world = rotmat

    # up_cam = mat44[:3, :3].T @ up
    # forward_cam = mat44[:3, :3].T @ forward
    # left_cam = mat44[:3, :3].T @ left

    gripper_finger_position = position_world
    gripper_position = []
    gripper_direction = []

    # the first waypoint
    gripper_position.append(gripper_finger_position)
    gripper_direction.append([up, forward, left])

    # waypoints_position
    for idx in range(len(waypoints)):
        gripper_position.append(
            gripper_finger_position + [waypoints[idx][0], waypoints[idx][1], waypoints[idx][2]])
        # gripper_direction.append(waypoints[idx][3:6])

    # waypoints_direction
    for idx in range(10, len(waypoints), 10):
        final_rotmat = start_rotmat_world.copy()
        final_rotmat[0, 3] += waypoints[idx][0]
        final_rotmat[1, 3] += waypoints[idx][1]
        final_rotmat[2, 3] += waypoints[idx][2]
        try:
            r = R.from_euler('XYZ', [waypoints[idx][3], waypoints[idx][4], waypoints[idx][5]], degrees=False)
            final_rotmat[:3, :3] = final_rotmat[:3, :3] @ r.as_matrix()
            forward = final_rotmat[:3, 0]
            left = final_rotmat[:3, 1]
            up = final_rotmat[:3, 2]
        except Exception:
            forward = np.zeros(3)
            left = np.zeros(3)
            up = np.zeros(3)
        gripper_direction.append([up, forward, left])

    gripper_position = np.array(gripper_position)
    gripper_position = (np.linalg.inv(mat44[:3, :3]) @ gripper_position.T).T  # world2cam
    gripper_direction = np.array(gripper_direction)
    gripper_direction = (np.linalg.inv(mat44[:3, :3]) @ gripper_direction.T).T

    return gripper_position, gripper_direction


def get_waypoints_cam_dense(json_content):
    mat44 = np.array(json_content['camera_metadata']['mat44'])
    position_world = np.array(json_content['position_world'])  # contact_point
    # position_cam = (np.linalg.inv(mat44[:3, :3]) @ position_world.T).T
    # position_cam -= np.array([5, 0, 0])   # normalize to (0,0,0)
    # position_world = (mat44[:3, :3] @ position_world.T).T
    # waypoints = json_content['dense_waypoints']
    waypoints = []
    up = np.array(json_content['gripper_direction_world'])
    forward = np.array(json_content['gripper_forward_direction_world'])
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    start_rotmat_world = rotmat

    # up_cam = mat44[:3, :3].T @ up
    # forward_cam = mat44[:3, :3].T @ forward
    # left_cam = mat44[:3, :3].T @ left

    gripper_finger_position = position_world
    gripper_position = []
    gripper_direction = []

    # the first waypoint
    gripper_position.append(gripper_finger_position)
    gripper_direction.append([up, forward, left])

    # waypoints_position
    for idx in range(len(waypoints)):
        gripper_position.append(
            gripper_finger_position + waypoints[idx][0] * forward + waypoints[idx][1] * left + waypoints[idx][2] * up)
        # gripper_direction.append(waypoints[idx][3:6])

    # waypoints_direction
    for idx in range(10, len(waypoints), 10):
        final_rotmat = start_rotmat_world.copy()
        final_rotmat[0, 3] += waypoints[idx][0]
        final_rotmat[1, 3] += waypoints[idx][1]
        final_rotmat[2, 3] += waypoints[idx][2]
        try:
            r = R.from_euler('XYZ', [waypoints[idx][3], waypoints[idx][4], waypoints[idx][5]], degrees=False)
            final_rotmat[:3, :3] = final_rotmat[:3, :3] @ r.as_matrix()
            forward = final_rotmat[:3, 0]
            left = final_rotmat[:3, 1]
            up = final_rotmat[:3, 2]
        except Exception:
            forward = np.zeros(3)
            left = np.zeros(3)
            up = np.zeros(3)
        gripper_direction.append([up, forward, left])

    gripper_position = np.array(gripper_position)
    gripper_position = (np.linalg.inv(mat44[:3, :3]) @ gripper_position.T).T  # world2cam
    gripper_direction = np.array(gripper_direction)
    gripper_direction = (np.linalg.inv(mat44[:3, :3]) @ gripper_direction.T).T

    return gripper_position, gripper_direction


def get_mask(h5_file, mask_file):
    cam_XYZA_id1 = h5_file['id1'][:].astype(np.int64)
    cam_XYZA_id2 = h5_file['id2'][:].astype(np.int64)
    cam_XYZA_pts = h5_file['pc'][:].astype(np.float32)
    out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
    out3 = np.array(mask_file, dtype=np.float32) > 127
    mask = (out[:, :, 3] > 0.5)
    # print('mask: ', mask.shape)
    pc = out[mask, :3]
    out3 = out3[mask]
    # print('pc', pc.shape)
    # print('out3', out3.shape)
    return pc, out3


''' main '''


def t2pc(opt_dir, input_dir, afford=0, way_id1=0, way_id2=0, way_id3=0, p_id=0, diff=0, p_id2=0, p_id3=0):
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)

    output_dir = opt_dir

    if afford == 1:
        json_file = open(os.path.join(input_dir, 'result_%d_%d.json' % (p_id, way_id1)), 'r')
        json_content = json.load(json_file)
        gripper_position, a = get_waypoints_cam_dense(json_content)

        tmp = [(gripper_position[0][0], gripper_position[0][1], gripper_position[0][2])]
        vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        if diff == 1:
            PlyData([el], text=False).write(output_dir + f'/traj{way_id1}_start.ply')
        else:
            PlyData([el], text=False).write(output_dir + '/traj_start.ply')
        gripper_position = gripper_position[1:]
        tmp = [(gripper_position[i][0], gripper_position[i][1], gripper_position[i][2]) for i in
               range(gripper_position.shape[0])]
        vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=False).write(output_dir + f'/traj{p_id}_{way_id1}_den_big.ply')

        if diff == 1:
            json_file = open(os.path.join(input_dir, 'result_%d_%d.json' % (p_id2, way_id2)), 'r')
        else:
            json_file = open(os.path.join(input_dir, 'result_%d_%d.json' % (p_id, way_id2)), 'r')
        json_content = json.load(json_file)
        gripper_position, a = get_waypoints_cam_dense(json_content)
        if diff == 1:
            tmp = [(gripper_position[0][0], gripper_position[0][1], gripper_position[0][2])]
            vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
            PlyData([el], text=False).write(output_dir + f'/traj{way_id2}_start.ply')
        gripper_position = gripper_position[1:]
        tmp = [(gripper_position[i][0], gripper_position[i][1], gripper_position[i][2]) for i in
               range(gripper_position.shape[0])]
        vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=False).write(output_dir + f'/traj{p_id2}_{way_id2}_den_big.ply')

        if diff == 1:
            json_file = open(os.path.join(input_dir, 'result_%d_%d.json' % (p_id3, way_id3)), 'r')
        else:
            json_file = open(os.path.join(input_dir, 'result_%d_%d.json' % (p_id, way_id3)), 'r')
        json_content = json.load(json_file)
        gripper_position, a = get_waypoints_cam_dense(json_content)
        if diff == 1:
            tmp = [(gripper_position[0][0], gripper_position[0][1], gripper_position[0][2])]
            vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
            PlyData([el], text=False).write(output_dir + f'/traj{way_id3}_start.ply')
        gripper_position = gripper_position[1:]
        tmp = [(gripper_position[i][0], gripper_position[i][1], gripper_position[i][2]) for i in
               range(gripper_position.shape[0])]
        vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=False).write(output_dir + f'/traj{p_id3}_{way_id3}_den_big.ply')

    else:
        json_file = open(os.path.join(input_dir, 'result.json'), 'r')
        json_content = json.load(json_file)
        gripper_position, a = get_waypoints_cam_dense(json_content)
        tmp = [(gripper_position[0][0], gripper_position[0][1], gripper_position[0][2])]
        vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=False).write(output_dir + '/traj_start.ply')

        gripper_position, a = get_waypoints_cam(json_content)
        tmp = [(gripper_position[i][0], gripper_position[i][1], gripper_position[i][2]) for i in
               range(len(gripper_position))]
        vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=False).write(output_dir + f'/traj.ply')

        tmp = []

        gripper_position, a = get_waypoints_cam_dense(json_content)
        tmp = [(gripper_position[i][0], gripper_position[i][1], gripper_position[i][2]) for i in
               range(len(gripper_position))]
        vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=False).write(output_dir + f'/traj_dense.ply')


print('done')