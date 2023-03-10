import numpy as np
import random
import cv2
import os
import torch
import imageio
# import model_SDE
# import model_SDE_union
#import subprocess
#import concurrent.futures
from PIL import Image
from env import Env
from camera import Camera
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from robots.panda_robot import Robot
from sapien.core import Pose
from argparse import ArgumentParser
from tqdm import tqdm
import utils

class Evaluator:
    def __init__(self, conf) -> None:
        self.conf = conf
        self.env = Env()
        self.robot_urdf_fn = './robots/panda_gripper.urdf'
        self.robot_material = self.env.get_material(4, 4, 0.01)
        self.algorithm = conf.algorithm
        self.num_point_per_shape = 10000

    def get_test_env(self, phi, theta, shape_id, articu_angle):
        self.cam = Camera(self.env, phi=phi, theta=theta)
        #cam = Camera(env)
        self.env.set_controller_camera_pose(self.cam.pos[0], self.cam.pos[1], self.cam.pos[2], np.pi+self.cam.theta, -self.cam.phi)

        # load shape
        object_urdf_fn = '/root/autodl-tmp/skj/where2act/data/where2act_original_sapien_dataset/{}/mobility_vhacd.urdf'.format(shape_id)
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
        world_pc, pc = self.generate_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts,
                        gt_movable_link_mask, mat44,
                        device=self.conf.device)
        return world_pc, pc

    def generate_pc(self, cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, movable_link_mask, mat44, device=None):
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
        input_pcid1 = torch.arange(1).unsqueeze(1).repeat(1, self.num_point_per_shape).long().reshape(-1)  # BN
        #print(world_pc.device)
        input_pcid2 = furthest_point_sample(world_pc, self.num_point_per_shape).long().reshape(-1)  # BN

        self.pc = pc[input_pcid1, input_pcid2, :].reshape(1, self.num_point_per_shape, -1)  # 1 * N * 3
        self.world_pc = world_pc[input_pcid1, input_pcid2, :].reshape(1, self.num_point_per_shape, -1)
        movables = self.movable_link_mask[input_pcid1, input_pcid2.cpu().detach()]
        self.movables = movables.reshape(1, self.num_point_per_shape, 1)

        return self.world_pc, self.pc

    def network_infer(self, network):
        init_pc_world = self.world_pc
        init_pc_cam = self.pc
        init_mask = self.movables
        init_mask = init_mask.reshape(self.num_point_per_shape)
        pc_cam = init_pc_cam[0].detach().cpu().numpy()
        with torch.no_grad():
            ###################
            # prepare input pcs
            ###################
            input_pcs = init_pc_cam.to(self.conf.device)
            ##########################
            # sample pixel to interact
            ##########################
            pc_score = network.inference_action_score(input_pcs)
            pc_score = pc_score.detach().cpu().numpy()

        if self.conf.policy == 'prob':
            pc_score[pc_score<0.5] = 0
            pp = pc_score[0]+1e-12
            
            self.p_id = np.random.choice(len(pc_score[0]), 1, p=pp/pp.sum())[0]

        elif self.conf.policy == 'highest':
            accu = 0.95
            # print(result)
            result = pc_score.reshape(self.num_point_per_shape)
            xs = np.where(result > accu)[0]
            while len(xs) < 100 and accu >= 0.1:
                accu = accu - 0.05
                xs = np.where(result > accu)[0]
                # print("len:", len(xs))
            # print("length:", len(xs))
            # print("epi_score: ", accu)

            self.p_id = xs[random.randint(0, len(xs) - 1)]
        
        position_cam = pc_cam[self.p_id]
        # print(position_cam)
        position_cam[0] += 5
        position_cam_xyz1 = np.ones((4), dtype=np.float32)
        position_cam_xyz1[:3] = position_cam
        self.position_world_xyz1 = self.cam.get_metadata()['mat44'] @ position_cam_xyz1
        self.position_world = self.position_world_xyz1[:3]
        position_cam[0] -= 5
        ####################################
        # move the posi to front of input_pc
        ####################################
        init_pc_world[0][0] = init_pc_world[0][self.p_id]
        init_pc_cam[0][0] = init_pc_cam[0][self.p_id]
        init_mask = init_mask.reshape(1, self.num_point_per_shape, 1)
        init_mask[0][0] = 1
        #################
        # generate action
        #################
        with torch.no_grad():
            if self.algorithm == 'w2a':
                pred_6d = network.inference_actor(init_pc_cam)[0]  # RV_CNT x 6
                pred_Rs = network.actor.bgs(pred_6d.reshape(-1, 3, 2)).detach().cpu().numpy()
            elif self.algorithm == 'vat':
                pred_6d, pred_dist = network.inference_actor(init_pc_cam)  # RV_CNT x 6
                pred_6d = pred_6d[0]
                pred_Rs = network.actor.bgs(pred_6d.reshape(-1, 3, 2)).detach().cpu().numpy()

        gripper_direction_camera = pred_Rs[:, :, 0]
        gripper_forward_direction_camera = pred_Rs[:, :, 1]
        rvs = gripper_direction_camera.shape[0]
        action_scores = []
        ########################
        # score proposed actions
        ########################
        for j in range(rvs):
            up = gripper_direction_camera[j]
            forward = gripper_forward_direction_camera[j]

            # up = cam.get_metadata()['mat44'][:3, :3] @ up
            # forward = cam.get_metadata()['mat44'][:3, :3] @ forward
            up = torch.FloatTensor(up).view(1, -1).to(self.conf.device)
            forward = torch.FloatTensor(forward).view(1, -1).to(self.conf.device)

            with torch.no_grad():
                # ipdb.set_trace()
                # print("pc_shape:", init_pc_world.shape)
                # print("up_shape:", up.shape)
                up = up.view(1, -1)
                if self.algorithm == 'w2a':
                    critic_score = network.inference_critic(init_pc_cam,up,forward,abs_val=True)
                elif self.algorithm == 'vat':
                    pred_dist_j = pred_dist[:, j, :]
                    # print(pred_dist_j.shape, pred_dist.shape)
                    critic_score = network.inference_critic(init_pc_cam,up,forward,pred_dist_j,abs_val=True)
            action_scores.append(critic_score.item())

        action_scores = np.array(action_scores)
        ########################################
        # sample from proposals with score > 0.5
        ########################################
        action_scores[action_scores<0.5] = 0
        action_score_sum = np.sum(action_scores) + 1e-12
        pp =  action_scores + 1e-12
        try:
            proposal_id = np.random.choice(len(action_scores), 1, p=action_scores/action_scores.sum())
            # proposal_id = [np.argmax(action_scores)]
            print('proposal id', proposal_id)
        except:
            proposal_id = [np.argmax(action_scores)]
            print('proposal id', proposal_id)
        print('proposal_score:', action_scores[proposal_id])
        selected_up = gripper_direction_camera[proposal_id]
        # print(selected_up.shape)
        selected_forward = gripper_forward_direction_camera[proposal_id]
        if self.algorithm == 'vat':
            self.pred_dist = pred_dist[:, proposal_id, :].detach().cpu().numpy()
        # print(selected_forward.shape)
        self.pose = torch.from_numpy(np.hstack([selected_up, selected_forward]))
        # print(self.pose.shape)
        # print(position_cam.shape)
        position_cam_tensor = torch.from_numpy(position_cam.reshape(1, -1))
        # print(position_cam_tensor.shape)
        result = torch.hstack((position_cam_tensor, self.pose))
        self.pose = self.pose.to(self.conf.device)
        result = result.to(self.conf.device)
        # print(result.shape)
        return result

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
        if self.algorithm == 'vat':
            final_dist = self.pred_dist[0,0]
        else:
            final_dist = 0.05
        if self.conf.primact_type == 'pulling':
            final_dist = -final_dist
        print(final_dist)

        final_rotmat = np.array(rotmat, dtype=np.float32)
        # final_rotmat[:3, 3] = position_world - action_direction_world * final_dist - action_direction_world * 0.1
        # if self.conf.primact_type == 'pushing':
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
                except Exception as e:
                    print('Errror :' + str(e))
                    print("fail")
                    success = False
            except Exception as e:
                print('Errror :' + str(e))
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
    
    def test_contact(self, network, shape_id_list, camera_angle_list, articu_angle_list, save_dir, aff_dir):
        succ_time = 0
        total_time = 0
        robot_first_time = True
        for shape_id in shape_id_list:
            print("current shape_id: ", shape_id)
            for camera_angle in camera_angle_list:
                for articu_angle in articu_angle_list:
                    ds_results = []
                    # all_motions = []
                    all_pid = []
                    afford_list = []
                    all_position = []

                    for test_idx in range(self.conf.how_many_times):
                        world_pc, pc = self.get_test_env(camera_angle[0], camera_angle[1], shape_id, articu_angle)
                        # if succ == False:
                        #     print("loading not success")
                        #     self.env.scene.remove_articulation(self.env.object)
                        #     continue
                        total_time += 1
                        result = self.network_infer(network)
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

                    # save pt file

                    angle_dir = os.path.join(aff_dir, f"{shape_id}", f"{articu_angle}")
                    if not os.path.exists(angle_dir):
                        os.makedirs(angle_dir)
                    fn = os.path.join(angle_dir, f'action_score_map_full_world.pt')
                    save_world_pc = world_pc[0].cpu().numpy()
                    utils.export_pts(fn, save_world_pc)
                    fn1 = os.path.join(angle_dir, f'action_score_map_full_init.pt')
                    save_pc = pc[0].cpu().numpy()
                    utils.export_pts(fn1, save_pc)


                    if len(afford_list) > 0:

                        self.concat_images(afford_list, 5, os.path.join(save_dir, "{}_{:.2f}_{:.2f}_{:.2f}_possible_ways.png".format(shape_id, camera_angle[0], camera_angle[1], articu_angle)))
                        save_obj_motion = {"obj": self.pc.cpu(), "pids": torch.tensor(all_pid).cpu()}
                        torch.save(save_obj_motion, os.path.join(save_dir, "{}_{:.2f}_{:.2f}_{:.2f}_succ.pt".format(shape_id, camera_angle[0], camera_angle[1], articu_angle)))
                        np.save(os.path.join(angle_dir,'all_world_positions.npy'), np.array(all_position))
        return {"succ_time": succ_time, "total_time": total_time, "ds_list": []}
    
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



