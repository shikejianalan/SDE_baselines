import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np
import imageio
from argparse import ArgumentParser

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import look_at_view_transform,FoVOrthographicCameras,PointsRasterizationSettings,PointsRenderer,PulsarPointsRenderer,PointsRasterizer,AlphaCompositor,NormWeightedCompositor

from torch import Tensor
from colorsys import hls_to_rgb

def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

def get_affordance_map_color(conf, affordance: Tensor):
    """
    Input: affordance N (value from 0 to 1)
    Output: the color for the affordance_map N x 3
    """
    b = 7. / 12.
    k = 1. / 18. - b
    affordance_color = torch.zeros(affordance.size(0), 3, device=conf.device)
    affordance_color[:, 0] = k * affordance + b # H value
    affordance_color[:, 1] = 1. # S value
    affordance_color[:, 2] = .5 # L value
    
    return hsl2rgb_torch(affordance_color)


def affordance_render(conf, pre_pts, affordance, view=[6, 0, 270], radius=0.0075, zoom=0.6, out_fn=None):
    pts_colors = get_affordance_map_color(conf, affordance)
    # print("pre_pts: ", pre_pts.size()) 
    # print("pts_colors: ", pts_colors.size())
    point_cloud = Pointclouds(points=[pre_pts], features=[pts_colors])
    # Initialize a camera.
    R, T = look_at_view_transform(view[0], view[1], view[2])
    cameras = FoVOrthographicCameras(device=conf.device, R=R*zoom, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=conf.render_img_size,
        radius=radius,
        points_per_pixel=10
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )
    images = renderer(point_cloud)
    
    output_img = images[0, ..., :3].cpu().numpy()
    np.rot90(output_img)
    output_img = output_img * 255.0
    output_img = output_img.astype(np.uint8)
    """
    print("output max: ", np.max(output_img))
    print("dtype: ", output_img.dtype)
    """
    if out_fn != None: 
        imageio.imwrite(out_fn, output_img)
    return {"torch_image": images[0, ..., :3], "np_image": output_img}



parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--render_img_size', default=(512, 512), help='the output image size')
parser.add_argument('--mode', default='test', help='which data to visualize')
conf = parser.parse_args()

if conf.mode == 'test':
    root_file = '/root/autodl-tmp/skj/where2act_vat/code/logs/final_exp-model_3d-pushing-vat-None-train_3d-2023-02-23-20:50:08/visu_action_heatmap_proposals-7128-model_epoch_299-nothing'
    # root_file = '/root/autodl-tmp/skj/where2act/code/logs/exp-model_3d_critic-pushing-None-train_3d_critic/visu_critic_heatmap-7296-model_epoch_299-nothing'
    pts_file = os.path.join(root_file, 'action_score_map.pts')
    affordance_file = os.path.join(root_file, 'action_score_map.label')
    loaded_pts_list = np.loadtxt(pts_file).astype(np.float32)
    loaded_pts = torch.from_numpy(loaded_pts_list).to(conf.device)

    loaded_afford_list = np.loadtxt(affordance_file).astype(np.float32)
    loaded_afford = torch.from_numpy(loaded_afford_list).to(conf.device)

    affordance_render(conf, loaded_pts, loaded_afford, out_fn='/root/autodl-tmp/skj/where2act/code/results/result6.png')
    print('done')

else:
    with open('readme.txt', 'r') as f:
        pass