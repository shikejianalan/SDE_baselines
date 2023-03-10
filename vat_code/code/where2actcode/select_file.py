from plyfile import PlyData, PlyElement
import ipdb
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
obj_dir = os.path.join("0618pcd2_mask.ply")
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
idx_list = np.array(range(pc.shape[0]))
np.random.shuffle(idx_list)
idx_list = idx_list[:30000]
pc = pc[idx_list]
pc1 = pc[:10000]

print(pc.shape)
print(pc1.shape)

tmp = []

for i in range(pc.shape[0]):
    ret = 0
    if i % 1000 == 0:
        print(i)
    for j in range(pc1.shape[0]):
        if abs(pc[i][0]-pc1[j][0])<=0.005 and abs(pc[i][1]-pc1[j][1])<=0.005 and abs(pc[i][2]-pc1[j][2])<=0.005:
            ret = ret + 1
        if ret > 10:
            break
    if ret>10:
        tmp.append((pc[i][0],pc[i][1],pc[i][2]))

vertex = np.array(tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
PlyData([el], text=False).write("./real_.ply")
    