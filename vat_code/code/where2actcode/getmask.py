from plyfile import PlyData, PlyElement
import ipdb
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
obj_dir = os.path.join("pcd_final.ply")
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

obj_dir = os.path.join("real_.ply")
ply_data = PlyData.read(obj_dir)
data = ply_data.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
data_np = np.zeros((data_pd.shape[0], 4), dtype=np.float)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    if i > 2:
        break
    data_np[:, i] = data_pd[name]

pc1 = data_np[:, :3]

mat44 = np.array([[ 0.61131721, -0.78792599,  0.07391821,  0.00685607],
                 [-0.40453437, -0.39140059, -0.82653344, -0.36229953],
                 [ 0.68017881,  0.47537166, -0.55801306,  1.37500313],
                 [ 0.,          0.,          0.,          1.        ]])

pc1 = (mat44[:3, :3] @ pc1.T).T

print(pc.shape)
print(pc1.shape)

mask = []

for i in range(pc.shape[0]):
    flag = 0
    if i % 1000 == 0:
        print(i)
    for j in range(pc1.shape[0]):
        if abs(pc[i][0]-pc1[j][0])<=0.01 and abs(pc[i][1]-pc1[j][1])<=0.01 and abs(pc[i][2]-pc1[j][2])<=0.01:
            flag = 1
            break
    mask.append(int(flag))

out_info = {'mask':mask}
save_dir='.'
with open(os.path.join(save_dir, 'mask_0619.json'), 'w') as fout:
    json.dump(out_info, fout)
    