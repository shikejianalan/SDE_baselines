from plyfile import PlyData, PlyElement
import ipdb
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
obj_dir = os.path.join("0625pcd_final.ply")
ply_data = PlyData.read(obj_dir)
data = ply_data.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
data_np = np.zeros((data_pd.shape[0], 4), dtype=np.float)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    if i > 2:
        break
    data_np[:, i] = data_pd[name]

pc = data_np[:, :3]
x_norm=-0.06817176192998886
y_norm=-0.0668630450963974
z_norm=0.08109600841999054
scale=0.57854706
pc[:, 0] -= x_norm
pc[:, 1] -= y_norm
pc[:, 2] -= z_norm

pc /= scale

input_dir = '.'
output_dir= './real_v'
json_file = open(os.path.join(input_dir, 'sup_mask_0625.json'), 'r')
json_content = json.load(json_file)
mask = json_content['mask']
tmp = []

for i in range(pc.shape[0]):
    if mask[i] == 1:
        tmp.append((pc[i][0],pc[i][1],pc[i][2]))

vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
PlyData([el], text=False).write(output_dir+'/out_door.ply')

tmp = []

for i in range(pc.shape[0]):
    if mask[i] == 0:
        tmp.append((pc[i][0],pc[i][1],pc[i][2]))

vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
PlyData([el], text=False).write(output_dir+'/door_body.ply')

json_file = open(os.path.join(input_dir, 'sup_mask_0625_drawer.json'), 'r')
json_content = json.load(json_file)
mask = json_content['mask']
tmp = []

for i in range(pc.shape[0]):
    if mask[i] == 1:
        tmp.append((pc[i][0],pc[i][1],pc[i][2]))

vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
PlyData([el], text=False).write(output_dir+'/out_drawer.ply')

tmp = []

for i in range(pc.shape[0]):
    if mask[i] == 0:
        tmp.append((pc[i][0],pc[i][1],pc[i][2]))

vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
PlyData([el], text=False).write(output_dir+'/drawer_body.ply')