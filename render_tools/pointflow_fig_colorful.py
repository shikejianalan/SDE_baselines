# import numpy as np
# # import torch
# def standardize_bbox(pcl, points_per_object):
#     # pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
#     # np.random.shuffle(pt_indices)
#     # pcl = pcl[pt_indices] # n by 3
#     mins = np.amin(pcl, axis=0)
#     maxs = np.amax(pcl, axis=0)
#     center = ( mins + maxs ) / 2.
#     scale = np.amax(maxs-mins)
#     print("Center: {}, Scale: {}".format(center, scale))
#     result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
#     return result

# xml_head = \
# """
# <scene version="0.6.0">
#     <integrator type="path">
#         <integer name="maxDepth" value="-1"/>
#     </integrator>
#     <sensor type="perspective">
#         <float name="farClip" value="100"/>
#         <float name="nearClip" value="0.1"/>
#         <transform name="toWorld">
#             <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
#         </transform>
#         <float name="fov" value="25"/>
        
#         <sampler type="ldsampler">
#             <integer name="sampleCount" value="256"/>
#         </sampler>
#         <film type="hdrfilm">
#             <integer name="width" value="1600"/>
#             <integer name="height" value="1200"/>
#             <rfilter type="gaussian"/>
#             <boolean name="banner" value="false"/>
#         </film>
#     </sensor>
    
#     <bsdf type="roughplastic" id="surfaceMaterial">
#         <string name="distribution" value="ggx"/>
#         <float name="alpha" value="0.05"/>
#         <float name="intIOR" value="1.46"/>
#         <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
#     </bsdf>
    
# """

# xml_ball_segment = \
# """
#     <shape type="sphere">
#         <float name="radius" value="0.025"/>
#         <transform name="toWorld">
#             <translate x="{}" y="{}" z="{}"/>
#         </transform>
#         <bsdf type="diffuse">
#             <rgb name="reflectance" value="{},{},{}"/>
#         </bsdf>
#     </shape>
# """

# xml_tail = \
# """
#     <shape type="rectangle">
#         <ref name="bsdf" id="surfaceMaterial"/>
#         <transform name="toWorld">
#             <scale x="10" y="10" z="1"/>
#             <translate x="0" y="0" z="-0.5"/>
#         </transform>
#     </shape>
    
#     <shape type="rectangle">
#         <transform name="toWorld">
#             <scale x="10" y="10" z="1"/>
#             <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
#         </transform>
#         <emitter type="area">
#             <rgb name="radiance" value="6,6,6"/>
#         </emitter>
#     </shape>
# </scene>
# """

# def colormap(x,y,z):
#     vec = np.array([x,y,z])
#     vec = np.clip(vec, 0.001,1.0)
#     norm = np.sqrt(np.sum(vec**2))
#     vec /= norm
#     return [vec[0], vec[1], vec[2]]
# xml_segments = [xml_head]

# path = '/root/autodl-tmp/skj/where2act_vat/code/logs/final_exp-model_3d-pushing-vat-None-train_3d-2023-02-23-20:50:08/visu_action_heatmap_proposals-7128-model_epoch_299-nothing/action_score_map_full_world.pts'
# loaded_pts_list = np.loadtxt(path).astype(np.float32)
# pcl = loaded_pts_list
# # pcl = np.load('chair_pcl.npy')
# print(pcl)
# print(pcl.shape)
# pcl = standardize_bbox(pcl, 2048)
# pcl = pcl[:,[2,0,1]]
# pcl[:,0] *= -1
# pcl[:,2] += 0.0125

# for i in range(pcl.shape[0]):
#     color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
#     xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
# xml_segments.append(xml_tail)

# xml_content = str.join('', xml_segments)

# with open('init.xml', 'w') as f:
#     f.write(xml_content)


import numpy as np
import torch
from scipy.spatial import KDTree
import os

def color_points(cloud, c_list, N):
    # 创建一个KDTree
    tree = KDTree(cloud)
    
    # 初始化一个颜色数组
    colors = np.ones((len(cloud), 3))
    colors[:, 0:3] = np.array([135/255,206/255,235/255])
    
    # 针对每个c点找到最近的N个点并染色
    for c in c_list:
        print(c)
        distances, indices = tree.query(c, k=N)
        red = np.zeros((N, 3))
        for i in range(N):
            red[i][0] = 1 - (distances[i]/np.max(distances))**3 + 0.01
            print(red[i][0])
        for index in indices:
            colors[index][0:3] = red[0]
            
    # 将未染色的点变为白色
    # white = np.ones((len(cloud), 3)) - colors
    # colors += white
    # colors = colors/np.max(colors)
    return colors


def standardize_bbox(pcl, points_per_object):
    # pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    # np.random.shuffle(pt_indices)
    # pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>

        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_marked_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.01"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""


xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]



# load pc
# path = '/root/autodl-tmp/skj/Manipulation_SDE/code/logs/multi_debug/7128/action_score_map_full_world.pt'
# loaded_pts_list = np.loadtxt(path).astype(np.float32)
# pcl = loaded_pts_list
# print(pcl.shape)
root_path = '/root/autodl-tmp/skj/Manipulation_SDE/code/logs/2023-03-02-06:28:17-pushing-vat-highest'
save_path = 'xml_files'

dirList = os.listdir(root_path)
for cat in dirList:
    print(cat)
    cat_dir = os.path.join(root_path, cat)
    for root, dirts, files in os.walk(cat_dir): # root, dirs, files
            for file in files:
                if file.endswith('.shapes'):
                    file_prefix = file[:-12]
                    shape_file = os.path.join(root, file)
                    result_file = os.path.join(root, file_prefix + '.results')

                    shape_result = torch.load(shape_file)
                    pcl = shape_result['world_pc'][0].numpy()
                    # print(pcl.shape)

                    # load positions
                    # point_path = '/root/autodl-tmp/skj/Manipulation_SDE/code/logs/2023-03-01-07:46:49-pushing-w2a-highest/Kettle/102730_0.63_2.41_0.80.results'
                    results = torch.load(result_file)
                    false_waypoints=[]
                    true_waypoints=[]
                    false_vectors = results['false_vectors']
                    true_vectors = results['true_vectors']
                    for false_vector in false_vectors:
                        distance = false_vector[1]-false_vector[0]
                        for i in range(70):
                            false_waypoints.append(false_vector[0] + distance/20 * i)
                    false_points_length = len(false_waypoints)
                    if len(true_vectors) != 0:
                        for true_vector in true_vectors:
                            distance = true_vector[1]-true_vector[0]
                            for i in range(70):
                                true_waypoints.append(true_vector[0] + distance/20 * i)

                    all_waypoints = np.vstack((np.array(false_waypoints), np.array(true_waypoints)))
                    # print(all_waypoints.shape)
                    # print(contact_points)
                    # print(contact_points.shape)

                    pcl = np.vstack((pcl, all_waypoints))

                    # print(pcl.shape)

                    pcl = standardize_bbox(pcl, 2048)
                    pcl, waypoints = pcl[:10000], pcl[10000:]
                    
                    # color = color_points(pcl, np.array(contact_points), 5)
                    # print(pcl.shape, waypoints.shape)
                    # color_ = pcl[:, 4]
                    xml_segments = [xml_head]
                    for i in range(pcl.shape[0]):
                        color = colormap(135/255,206/255,235/255)
                        xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
                        # xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color[i].tolist()))
                    for point in waypoints[:false_points_length]:
                        color = colormap(255, 0, 235/255)
                        xml_segments.append(xml_marked_ball_segment.format(point[0],point[1],point[2], *color))
                    for point in waypoints[false_points_length:]:
                        color = colormap(0, 102/255, 51/255)
                        xml_segments.append(xml_marked_ball_segment.format(point[0],point[1],point[2], *color))


                    xml_segments.append(xml_tail)

                    xml_content = str.join('', xml_segments)
                    save_cat_root = os.path.join('test_images_vat', 'xml_files', cat)
                    if not os.path.exists(save_cat_root):
                        os.makedirs(save_cat_root)
                    with open(os.path.join(save_cat_root, f'{file_prefix}.xml'), 'w') as f:
                        f.write(xml_content)