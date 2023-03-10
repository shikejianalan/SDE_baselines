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
# import utils
import matplotlib
import random
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import Camera
from PIL import Image

Colors = ['blue','red','darkviolet','gold','lightpink','green','yellow']
# from rand_cmap import rand_cmap
# cmap = rand_cmap(300, type='bright', first_color_black=True, last_color_black=False, verbose=False)
def dist(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2])
def callen(p):
    L = len(p)
    ret = 0
    for i in range(L-1):
        ret = ret + dist(p[i],p[i+1])
        if dist(p[i],p[i+1])>0.1:
            return -10
    ret = ret + dist(p[0],p[L-1])*0.3
    return ret
def getdist(p1,p2):
    ret = 0
    L = 5
    for i in range(min(L,len(p1),len(p2))):
        ret = ret + dist(p1[i],p2[i])
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
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=[color], alpha = alpha,marker=maker, s=s)

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



def create_color_list(num):
    colors = np.ndarray(shape=(num, 3))
    random.seed(30)
    for i in range(0, num):
        colors[i, 0] = random.randint(0, 255)
        colors[i, 1] = random.randint(0, 255)
        colors[i, 2] = random.randint(0, 255)
    return colors
COLOR_LIST = create_color_list(100)
for i in range(100):
    COLOR_LIST[i] = (float(COLOR_LIST[i][0]) / 255.0, float(COLOR_LIST[i][1]) / 255.0, float(COLOR_LIST[i][2]) / 255.0)
COLOR_LIST = tuple(COLOR_LIST)
# print(type(COLOR_LIST))
def draw_geo1(ax, p, color, maker='.', s=20, alpha=0.5, rot=None):
    # print(p.shape)
    if rot is not None:
        p = (rot * p.transpose()).transpose()
    ax.quiver(p[:-1,0], p[:-1,1],p[:-1,2], p[1:,0] - p[:-1,0], p[1:,1] - p[:-1,1],p[1:,2]-p[:-1,2], color=[color],alpha=alpha,length=1.1,arrow_length_ratio=0.5)
def draw_waypoint(ax,obj,coord_rot,i):
#    L = len(obj)
#    L = L // 3
    draw_geo1(ax=ax, p=obj, color=Colors[i], rot=coord_rot, s=50, alpha=0.7)
#    draw_geo1(ax=ax, p=obj[L:L * 2], color=Colors[i], rot=coord_rot, s=50, alpha=0.5)
#    draw_geo1(ax=ax, p=obj[L * 2:], color=Colors[i], rot=coord_rot, s=50, alpha=0.3)

def draw_partnet_objects(num, objects, ax, fig, object_names=None, type_list=None, print_name_list=None,
                         edge_list=None,
                         figsize=None, rep='boxes',
                         leafs_only=False, use_id_as_color=False, visu_edges=False, vis_only_pc=False, sem_colors_filename=None, len_x=None, len_y=None, save_fig_name='xxx', part_list=[],
                         print_label=False, print_obj_name=False,
                         ):

    # if sem_colors_filename is not None:
    #     sem_colors, id2name = load_semantic_colors(filename=sem_colors_filename)
    #     for sem in sem_colors:
    #         sem_colors[sem] = (float(sem_colors[sem][0]) / 255.0, float(sem_colors[sem][1]) / 255.0, float(sem_colors[sem][2]) / 255.0)
    #         # if vis_only_pc:
    #         #     sem_colors[sem] = (float(sem_colors['chair'][0]) / 255.0, float(sem_colors['chair'][1]) / 255.0, float(sem_colors['chair'][2]) / 255.0)

    # else:
    #     sem_colors = None

    t = (num-1)//3 + 1
    coord_rot = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    choose_list = []
    choose = 0
    maxn = 0.0
    for i in range(t):
        if callen(objects[i]) > maxn:
            maxn = callen(objects[i])
            choose = i
    draw_waypoint(ax,objects[choose],coord_rot,0)
    choose_list.append(objects[choose])
    maxn = 0.0
    choose = t
    for i in range(t,2*t):
        tmp = 100.0
        for k in range(len(choose_list)):
            tmp = min(tmp, getdist(objects[i], choose_list[k]))
        tmp = tmp + callen(objects[i])
        if tmp > maxn:
            maxn = tmp
            choose = i
    draw_waypoint(ax,objects[choose], coord_rot,1)
    choose_list.append(objects[choose])
    maxn = 0.0
    choose = t
    for i in range(2*t, num):
        tmp = 100.0
        for k in range(len(choose_list)):
            tmp = min(tmp, getdist(objects[i], choose_list[k]))
        tmp = tmp + callen(objects[i])
        if tmp > maxn:
            maxn = tmp
            choose = i
    draw_waypoint(ax,objects[choose], coord_rot,2)
    choose_list.append(objects[choose])
    """
    for i, obj in enumerate(objects):
        rep = type_list[i]

        # ax.set_aspect('equal')

        # transform coordinates so z is up (from y up)
        # coord_rot = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coord_rot = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # coord_rot = None


        if rep == 'geos_wp':      
            # tuple[0]: pc

            if len(obj[1]) != 1:    # None
                # tuple[1]: waypoints
                # draw_geo(ax=ax, p=obj[1], color='red', maker='^', rot=coord_rot, s=50)
                if i==0:
                    L = len(obj)
                    L = L//3
                    draw_geo(ax=ax, p=obj[:L], color=Colors[0], rot=coord_rot, s=50,alpha=0.7)
                    draw_geo(ax=ax, p=obj[L:L*2], color=Colors[0], rot=coord_rot, s=50,alpha=0.5)
                    draw_geo(ax=ax, p=obj[L*2:], color=Colors[0], rot=coord_rot, s=50,alpha=0.3)
                    choose_list.append(obj)
                elif i%t == 0:
                    maxn = 0.0
                    choose = i-t+1
                    for j in range(i-t+1,i+1):
                        tmp = 100.0
                        for k in range(len(choose_list)):
                            tmp = min(tmp,getdist(objects[j],choose_list[k])) 
                        if tmp > maxn:
                            maxn = tmp
                            choose = j
                    L = len(objects[choose])
                    L = L//3
                    draw_geo(ax=ax, p=objects[choose][:L], color=Colors[i//t], rot=coord_rot, s=50,alpha=0.7)
                    draw_geo(ax=ax, p=objects[choose][L:L*2], color=Colors[i//t], rot=coord_rot, s=50,alpha=0.5)
                    draw_geo(ax=ax, p=objects[choose][L*2:], color=Colors[i//t], rot=coord_rot, s=55,alpha=0.3)
                    choose_list.append(objects[choose])

            continue
    """

def get_waypoints_cam(json_content):
    mat44 = np.array(json_content['camera_metadata']['mat44'])
    position_world = np.array(json_content['position_world'])  # contact_point
    # position_cam = (np.linalg.inv(mat44[:3, :3]) @ position_world.T).T
    # position_cam -= np.array([5, 0, 0])   # normalize to (0,0,0)
    # position_world = (mat44[:3, :3] @ position_world.T).T
    waypoints = json_content['waypoints']
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

    gripper_finger_position = position_world + 0.01*up
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
    gripper_position = (np.linalg.inv(mat44[:3, :3]) @ gripper_position.T).T         # world2cam
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
def h52pc(opt_dir,input_dir,afford=0,way_id1=0,p_id=0):
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    if afford==1:
        h5_file = h5py.File(os.path.join(input_dir, f'cam_XYZA_{p_id}_{way_id1}.h5'), 'r')
        mask_file = Image.open(os.path.join(input_dir, f'interaction_mask_{p_id}_{way_id1}.png'))
    else :
        h5_file = h5py.File(os.path.join(input_dir, f'cam_XYZA.h5'), 'r')
        mask_file = Image.open(os.path.join(input_dir, f'interaction_mask.png'))
    output_dir=opt_dir
    pc, mask = get_mask(h5_file, mask_file)
    pc -= np.array([[5, 0, 0] for i in range(pc.shape[0])])  # normalize to(0,0,0)
    idx_list = np.where(mask<0.5)
    np.random.shuffle(idx_list)
    idx_list = idx_list[:10000]
    pc1 = pc[idx_list]
    idx_list = np.where(mask>0.5)
    np.random.shuffle(idx_list)
    idx_list = idx_list[:10000]
    pc2 = pc[idx_list]
    tmp = [(pc1[i][0], pc1[i][1], pc1[i][2]) for i in range(pc1.shape[0])]
    vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(output_dir+"/body1.ply")
    tmp = [(pc2[i][0], pc2[i][1], pc2[i][2]) for i in range(pc2.shape[0])]
    vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(output_dir+"/door1.ply")

print('done')