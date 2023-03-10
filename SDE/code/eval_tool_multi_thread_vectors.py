import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--shape_id", type=int)
parser.add_argument("--camera_angle_phi", type=float, default='0.6283185')
parser.add_argument("--camera_angle_theta", type=float,default='2.40855436775217')
parser.add_argument("--articu_angle", type=float)
parser.add_argument("--save_dir", type=str)
parser.add_argument('--cuda_sel', type=str,default='2')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--primact_type', type=str, help='the primact type')
parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

parser.add_argument('--model_path', type=str)
# network and sampler setting
parser.add_argument('--sampler', type=str, default='EM', help='Sampler options: EM, PC and ODE')
parser.add_argument('--cond_len', type=int, default=256, help='The dimension of the condition')
## SDE network options
parser.add_argument('--manipSDE_sigma', type=float, default=25.0)
parser.add_argument('--manipSDE_snr', type=float, default=0.2)
parser.add_argument('--manipSDE_num_steps', type=int, default=500)
parser.add_argument('--manipSDE_input_dim', type=int, default=9)
parser.add_argument('--manipSDE_cond_res_num', type=int, default=5)
parser.add_argument('--manipSDE_feat_len', type=int, default=128)
parser.add_argument('--manipSDE_time_embed_len', type=int, default=128)

# test options:
parser.add_argument('--how_many_times', type=int, default=20)

parser.add_argument('--vis_gif', type=bool, default=True)
parser.add_argument('--visu', type=bool, default=True)

"""render options"""
parser.add_argument('--render_img_size', type=int, default=512)

conf = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(conf.cuda_sel)

import numpy as np
import random
import cv2

import torch
import imageio
import model_SDE
import model_SDE_union
#import subprocess
#import concurrent.futures
from PIL import Image
from env import Env
from camera import Camera
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from robots.panda_robot import Robot
from sapien.core import Pose

from tqdm import tqdm

if conf.seed >= 0:
    #conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

class Evaluator:
    def __init__(self, conf) -> None:
        self.conf = conf
        self.env = Env()
        self.robot_urdf_fn = './robots/panda_gripper.urdf'
        self.robot_material = self.env.get_material(4, 4, 0.01)

    def get_test_env(self, phi, theta, shape_id, articu_angle):
        self.cam = Camera(self.env, phi=phi, theta=theta)
        #cam = Camera(env)
        self.env.set_controller_camera_pose(self.cam.pos[0], self.cam.pos[1], self.cam.pos[2], np.pi+self.cam.theta, -self.cam.phi)

        # load shape
        object_urdf_fn = '../data/where2act_original_sapien_dataset/{}/mobility_vhacd.urdf'.format(shape_id)
        object_material = self.env.get_material(4, 4, 0.01)
        state = "indicated"
        flag = self.env.load_object(object_urdf_fn, object_material, articu_angle, state=state)
        # if flag == False:
        #     return False
        self.env.step()
        self.env.render()

        mat44 = self.cam.get_metadata()['mat44']
        ### use the GT vision
        rgb, depth = self.cam.get_observation()
        object_link_ids = self.env.movable_link_ids
        # print("object_link_ids", object_link_ids)
        gt_movable_link_mask = self.cam.get_movable_link_mask(object_link_ids)
        gt_handle_mask = self.cam.get_handle_mask()

        # prepare input pc
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = self.cam.compute_camera_XYZA(depth)
        self.generate_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts,
                        gt_movable_link_mask, mat44,
                        device=self.conf.device)

    def generate_pc(self, cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, movable_link_mask, mat44, device=None):
        num_point_per_shape = 10000
        out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
        mask = (out[:, :, 3] > 0.5)
        pc = out[mask, :3]
        movable_link_mask = movable_link_mask[mask]
        idx = np.arange(pc.shape[0])
        np.random.shuffle(idx)
        while len(idx) < 30000:
            idx = np.concatenate([idx, idx])
        idx = idx[:30000]
        pc = pc[idx, :]
        movable_link_mask = movable_link_mask[idx]
        self.movable_link_mask = movable_link_mask.reshape(1, 30000, 1)
        pc[:, 0] -= 5

        world_pc = (mat44[:3, :3] @ pc.T).T
        pc = np.array(pc, dtype=np.float32)
        world_pc = np.array(world_pc, dtype=np.float32)
        pc = torch.from_numpy(pc).float().unsqueeze(0).to(device)
        world_pc = torch.from_numpy(world_pc).float().unsqueeze(0).to(device).contiguous()
        input_pcid1 = torch.arange(1).unsqueeze(1).repeat(1, num_point_per_shape).long().reshape(-1)  # BN
        #print(world_pc.device)
        input_pcid2 = furthest_point_sample(world_pc, num_point_per_shape).long().reshape(-1)  # BN

        self.pc = pc[input_pcid1, input_pcid2, :].reshape(1, num_point_per_shape, -1)  # 1 * N * 3
        self.world_pc = world_pc[input_pcid1, input_pcid2, :].reshape(1, num_point_per_shape, -1)
        movables = self.movable_link_mask[input_pcid1, input_pcid2.cpu().detach()]
        self.movables = movables.reshape(1, num_point_per_shape, 1)

    def network_infer(self, network: model_SDE_union.AssembleModel):
        init_pc_world = self.world_pc
        init_pc_cam = self.pc
        init_mask = self.movables
        # print("init_mask: ", init_mask.shape)
        # print("self.movables: ", self.movables.shape)
        # print("init_mask id: ", id(init_mask))
        # print("self.movables id: ", id(self.movables))
        init_mask = init_mask.reshape(10000)
        # print("init_mask: ", init_mask.shape)
        # print("self.movables: ", self.movables.shape)
        # print("init_mask id: ", id(init_mask))
        # print("self.movables id: ", id(self.movables))
        # exit()
        pc_cam = init_pc_cam[0].detach().cpu().numpy()
        with torch.no_grad():
            input_pcs = init_pc_cam.to(self.conf.device)
            whole_feats = network.get_whole_feats(input_pcs)
            result = network.get_pose(whole_feats)["x_states"][-1]
            #result = result.detach().cpu().numpy()
        
        posi = result[0, :3].detach().cpu().numpy()
        self.pose = result[:, 3:]
        l2_norm = np.sqrt(np.sum(pc_cam - posi, axis=-1) ** 2)
        print("l2_norm: ", l2_norm)
        print("min l2_norm: ", np.min(l2_norm))

        self.p_id = np.argmin(l2_norm)

        position_cam = pc_cam[self.p_id]
        # print(position_cam)
        position_cam[0] += 5
        position_cam_xyz1 = np.ones((4), dtype=np.float32)
        position_cam_xyz1[:3] = position_cam
        self.position_world_xyz1 = self.cam.get_metadata()['mat44'] @ position_cam_xyz1
        self.position_world = self.position_world_xyz1[:3]
        position_cam[0] -= 5
        init_pc_world[0][0] = init_pc_world[0][self.p_id]
        init_pc_cam[0][0] = init_pc_cam[0][self.p_id]
        init_mask = init_mask.reshape(1, 10000, 1)
        init_mask[0][0] = 1
        return result.detach().clone()

    def test_contact_once(self, object_link_id, robot_first_time):
        # conf = self.conf
        # cam = self.cam
        # env = self.env
        # robot = self.robot
        self.env.set_target_object_part_actor_id(object_link_id)
        position_world = self.position_world
        position_world_xyz1 = self.position_world_xyz1
        up = self.pose[0, :3].detach().cpu().numpy()
        forward = self.pose[0, 3:].detach().cpu().numpy()
        up = self.cam.get_metadata()['mat44'][:3, :3] @ up
        forward = self.cam.get_metadata()['mat44'][:3, :3] @ forward

        up = np.array(up, dtype=np.float32)
        forward = np.array(forward, dtype=np.float32)

        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        forward = np.cross(left, up)
        forward /= np.linalg.norm(forward)

        action_direction_world = up

        rotmat = np.eye(4).astype(np.float32)
        rotmat[:3, 0] = forward
        rotmat[:3, 1] = left
        rotmat[:3, 2] = up

        # final_dist = 0.3 + np.random.rand() * 0.25 + trial_id * 0.05
        final_dist = 0.05
        print(final_dist)

        final_rotmat = np.array(rotmat, dtype=np.float32)
        final_rotmat[:3, 3] = position_world - action_direction_world * final_dist - action_direction_world * 0.1
        if self.conf.primact_type == 'pushing':
            final_rotmat[:3, 3] = position_world + action_direction_world * final_dist - action_direction_world * 0.15
        final_pose = Pose().from_transformation_matrix(final_rotmat)

        start_rotmat = np.array(rotmat, dtype=np.float32)
        start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
        start_pose = Pose().from_transformation_matrix(start_rotmat)

        end_rotmat = np.array(rotmat, dtype=np.float32)
        end_rotmat[:3, 3] = position_world - action_direction_world * 0.1
        if robot_first_time:
            self.robot = Robot(self.env, self.robot_urdf_fn, self.robot_material, open_gripper=('pulling' in self.conf.primact_type))
        else:
            self.robot.load_gripper(self.robot_urdf_fn, self.robot_material, open_gripper=('pulling' in self.conf.primact_type))
        self.robot.robot.set_root_pose(start_pose)
        self.env.render()

        # activate contact checking
        self.env.start_checking_contact(self.robot.hand_actor_id, self.robot.gripper_actor_ids, 'pushing' in self.conf.primact_type)
        target_link_mat44 = self.env.get_target_part_pose().to_transformation_matrix()

        success = True
        target_link_mat44 = self.env.get_target_part_pose().to_transformation_matrix()
        position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
        #succ_images = []
        succ_imgs = None
        succ_points = None
        if self.conf.primact_type == 'pulling':
            try:
                init_success = True
                success_grasp = False
                print("try to grasp")
                # imgs = robot.wait_n_steps(1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                # succ_images.extend(imgs)
                try:
                    self.robot.open_gripper()
                    #robot.move_to_target_pose(end_rotmat, 2000)
                    succ_imgs, succ_points = self.robot.move_to_target_pose_visu(end_rotmat, 2000, visu=self.conf.visu, vis_gif=self.conf.vis_gif, cam=self.cam)
                    self.robot.wait_n_steps(2000)
                    self.robot.close_gripper()
                    self.robot.wait_n_steps(2000)
                    now_qpos = self.robot.robot.get_qpos().tolist()
                    finger1_qpos = now_qpos[-1]
                    finger2_qpos = now_qpos[-2]
                    # print(finger1_qpos, finger2_qpos)
                    if finger1_qpos + finger2_qpos > 0.01:
                        success_grasp = True
                except Exception:
                    init_success = False
                if not (success_grasp and init_success):
                    print('grasp_fail')
                    success = False
                else:
                    try:
                        succ_imgs, succ_points = self.robot.move_to_target_pose_visu(final_rotmat, 2000, visu=self.conf.visu, vis_gif=self.conf.vis_gif, cam=self.cam)
                        self.robot.wait_n_steps(2000)
                    except Exception:
                        print("fail")
                        success = False
            except Exception:
                success = False
        else :
            try:
                self.robot.close_gripper()
                #succ_images = []
                try:
                    succ_imgs, succ_points = self.robot.move_to_target_pose_visu(final_rotmat, 2000, visu=self.conf.visu, vis_gif=self.conf.vis_gif, cam=self.cam)
                    self.robot.wait_n_steps(2000)
                except Exception:
                    print("fail")
                    success = False
            except Exception:
                success = False
        self.env.scene.remove_articulation(self.robot.robot)
        return {"success": success, "succ_imgs": succ_imgs, "succ_points": succ_points, "contact_points": self.position_world}
    
    def concat_images(self, images, row_n, filename):
        image_count = len(images)
        img = images[0]
        width, height = img.size
        result_width = width * row_n
        result_height = height * ((image_count + row_n - 1) // row_n)
        result = Image.new(img.mode, (result_width, result_height))
        for idx, image in enumerate(images):
            result.paste(image, (idx % row_n * width, idx // row_n * height))
        result.save(filename)
    
    def diversity_score_l2(self, all_results):
        all_dist = 0
        for cur in all_results:
            for other in all_results:
                cur_dist = torch.sqrt(torch.sum((other - cur) ** 2))
                all_dist += cur_dist
        return all_dist

    def test_contact(self, network, shape_id, camera_angle, articu_angle, save_dir):
        succ_time = 0
        total_time = 0
        ds_value = 0.
        robot_first_time = True
        print("current shape_id: ", shape_id)
        ds_results = []
        all_ds_results = []
        # all_motions = []
        all_pid = []
        afford_list = []
        all_position = []
        for test_idx in range(self.conf.how_many_times):
            self.get_test_env(camera_angle[0], camera_angle[1], shape_id, articu_angle)
            # if succ == False:
            #     print("loading not success")
            #     self.env.scene.remove_articulation(self.env.object)
            #     continue
            total_time += 1
            result = self.network_infer(network)
            all_ds_results.append(result)
            #movable_test_flag = False
            #for object_link_id in self.env.movable_link_ids: # test all movable link ids
            """
            if movable_test_flag:
                print("reload")
                self.env.scene.remove_articulation(self.env.object)
                print("reach here 1")
                self.get_test_env(camera_angle[0], camera_angle[1], shape_id, articu_angle)
                print("reach here 2")
            movable_test_flag = True
            """
            if_succ_result = self.test_contact_once(self.env.movable_link_ids[self.movables[0, self.p_id, 0] - 1], robot_first_time)
            robot_first_time = False
            #print("reach here 3")
            if if_succ_result["success"]:
                succ_time += 1
                ds_results.append(result)
                succ_gif = if_succ_result["succ_imgs"]
                afford_list.append(succ_gif[0])
                all_pid.append(self.p_id)
                all_position.append(if_succ_result['contact_points'])
                # print("succ_points: ", if_succ_result["succ_points"][0])
                # succ_points = torch.tensor(if_succ_result["succ_points"])
                # all_motions.append(succ_points)
                # print("succ_points: ", succ_points.size())
                # exit()
                succ_gif[0].save(os.path.join(save_dir, "{}_{:.2f}_{:.2f}_{:.2f}_{}_succ.gif".format(shape_id, camera_angle[0], camera_angle[1], articu_angle, test_idx)),
                    save_all=True, append_images=succ_gif[1:], optimize=False, duration=200, loop=0)
                #break
            self.env.scene.remove_articulation(self.env.object)
            #self.env.scene.remove_articulation(self.robot.robot)
        if len(afford_list) > 0:
            ds_results = torch.cat(ds_results, dim=0)
            ds_value = self.diversity_score_l2(ds_results)
            self.concat_images(afford_list, 5, os.path.join(save_dir, "{}_{:.2f}_{:.2f}_{:.2f}_possible_ways.png".format(shape_id, camera_angle[0], camera_angle[1], articu_angle)))
            save_obj_motion = {"world_pc": self.world_pc.cpu(), "all_pos": torch.tensor(np.array(all_position)).cpu(), "pids": torch.tensor(all_pid).cpu(), "all_results": ds_results.cpu()}
            torch.save(save_obj_motion, os.path.join(save_dir, "{}_{:.2f}_{:.2f}_{:.2f}_succ.shapes".format(shape_id, camera_angle[0], camera_angle[1], articu_angle)))
        return_dict = {"shape_id": shape_id, "succ_time": succ_time, "total_time": total_time, "ds_value": float(ds_value) / (total_time ** 2), "all_vectors": torch.cat(all_ds_results, dim=0).cpu()}
        print(return_dict)
        torch.save(return_dict, os.path.join(save_dir, "{}_{:.2f}_{:.2f}_{:.2f}.results".format(shape_id, camera_angle[0], camera_angle[1], articu_angle)))
        self.env.close()
        # return return_dict
    
    """
    def test_contact_para():
        

        scripts = ['script_{}.py'.format(i) for i in range(200)]

        while scripts:
            subprocess_args = [["python", script] for script in scripts[:16]]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(subprocess.call, args) for args in subprocess_args]
                concurrent.futures.wait(futures)
            scripts = scripts[16:]
    """

    
    def get_camera_angle_list(self, base_angle=[np.pi/5, np.pi/3 *2.2], time=8):
        camera_angle_list = [base_angle,]
        for _ in range(time):
            new_angle = camera_angle_list[-1]
            new_angle[-1] += np.pi/3 * 0.2
            camera_angle_list.append(new_angle)
        return camera_angle_list
    
    def get_articu_angle_list(self, base_angle=np.pi/2*0.1, time=5):
        articu_angle_list = [base_angle,]
        for _ in range(time):
            new_angle = articu_angle_list[-1]
            new_angle += np.pi/2 * 0.1
            articu_angle_list.append(new_angle)
        return articu_angle_list


network = model_SDE_union.AssembleModel(conf)
network.load_state_dict(torch.load(conf.model_path))
network = network.to(conf.device)
network.eval()
evaluator = Evaluator(conf)
camera_angle = [conf.camera_angle_phi, conf.camera_angle_theta]
evaluator.test_contact(network, conf.shape_id, camera_angle, conf.articu_angle, conf.save_dir)



