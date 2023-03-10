import os,sys
import trimesh
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
import argparse


def read_ply_xyz(filename):
    """ read XYZ point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def write_obj(points, faces, filename):
    with open(filename, 'w') as F:
        for p in points:
            F.write('v %f %f %f\n'%(p[0], p[1], p[2]))
        
        for f in faces:
            F.write('f %d %d %d\n'%(f[0], f[1], f[2]))


def convert_point_cloud_to_balls(pc_ply_filename, output_filename=None, sphere_r=0.008):
    if output_filename is None:
        pc_dir = os.path.dirname(pc_ply_filename)
        pc_name = pc_ply_filename.split('\\')[-1].split('.')[0]
        output_filename = os.path.join(pc_dir, pc_name+'_{:.4f}.obj'.format(sphere_r))

    pc = read_ply_xyz(pc_ply_filename)

    points = []
    faces = []

    for pts in (pc):
        sphere_m = trimesh.creation.uv_sphere(radius=sphere_r, count=[3,3])
        sphere_m.apply_translation(pts)

        faces_offset = np.array(sphere_m.faces) + len(points)
        faces.extend(faces_offset)
        points.extend(np.array(sphere_m.vertices))
    
    points = np.array(points)
    faces = np.array(faces)
    #print(points.shape, faces.shape)
    finale_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    if False: # only true for EPN results
        finale_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2., [0,0,1]))
    finale_mesh.export(output_filename)
    

def pc2b(pc_path):
    pc_ply_files = [f for f in os.listdir(pc_path) if f.endswith('.ply')]
    output_path = pc_path+'_spheres'
    if not os.path.exists(output_path): os.makedirs(output_path)
    for pc_name in tqdm(pc_ply_files):
        pc_filename = os.path.join(pc_path, pc_name)
        if '.ply' in pc_filename:
            if 'input' in pc_filename:
                sphere_r = 0.004
            else:
                if 'traj' in pc_filename:
                    sphere_r = 0.033
                    if 'start' in pc_filename:
                        sphere_r = 0.042
                    if 'dense' in pc_filename:
                        sphere_r = 0.025    
                else:
                    sphere_r = 0.008
        #sphere_r = 0.006
            out_pc_filename = os.path.join(output_path, pc_name[:-4]+'.obj')
            convert_point_cloud_to_balls(pc_filename, out_pc_filename, sphere_r=sphere_r)