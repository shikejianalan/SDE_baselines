import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from subprocess import call
# from datagen import DataGen
from data_task_traj_RL import SAPIENVisionDataset
import utils
from utils import calc_part_motion_degree
from pointnet2_ops.pointnet2_utils import furthest_point_sample
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))
import render_using_blender as render_utils
from tensorboardX import SummaryWriter
import ipdb


def train(conf, train_shape_list, train_data_list, val_data_list, all_train_data_list=None):
    # create training and validation datasets and data loaders
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction_world', 'gripper_forward_direction_world', \
            'result', 'task_motion', 'gt_motion', 'task_waypoints', 'cur_dir', 'shape_id', 'trial_id', 'is_original', 'position_world']

    ''' input:  task， init position， contact point, waypoint '''
     
    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.ActionScore(conf.feat_dim, topk=conf.topk)

    
    actor = None
    critic = None


    # send parameters to device
    network.to(conf.device)
    network.load_state_dict(torch.load(conf.affordance_dir % conf.eval_epoch))

    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, angle_system=conf.angle_system, EP_MAX=conf.num_train, degree_lower=conf.degree_lower, cur_primact_type=conf.primact_type, critic_mode=True, only_true_data=False, affordance_mode=True)

    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal, angle_system=conf.angle_system, EP_MAX=conf.num_eval, degree_lower=conf.degree_lower, cur_primact_type=conf.primact_type, critic_mode=True, only_true_data=False, affordance_mode=True)

    ### load data for the current epoch
    val_dataset.load_data(val_data_list, wp_xyz=conf.wp_xyz, num_data_uplimit=conf.val_num_data_uplimit)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    val_num_batch = len(val_dataloader)


    # start training
    start_time = time.time()

    last_train_console_log_step, last_val_console_log_step = None, None



    val_batches = enumerate(val_dataloader, 0)
    network.eval()
    idx_step = 0
    ### train for every batch
    for val_batch_ind, val_batch in val_batches:
        idx_step += 1
        # val_step = (epoch + val_fraction_done) * train_num_batch - 1

        # set models to evaluation mode

        with torch.no_grad():
            # forward pass (including logging)
            score_forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                    step=idx_step, epoch=-1, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                    log_console=None, log_tb=not conf.no_tb_log, tb_writer=None, lr=None, actor=actor, critic=critic)


def score_forward(batch, data_features, network, conf, \
            is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
            log_console=False, log_tb=False, tb_writer=None, lr=None, actor=None, critic=None):
    # prepare input
    input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)  # B x 3N x 3   # point cloud
    # print("shape0:", input_pcs.shape)
    # input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)  # B x 3N x 2
    input_movables = torch.cat(batch[data_features.index('pc_movables')], dim=0).to(conf.device)  # B x 3N  # movable part
    mat44 = np.array(batch[data_features.index('mat44')])[0]
    batch_size = input_pcs.shape[0]

    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    # print('pcid2', input_pcid2.shape)
    # # random sample pts
    # pcs_id = ()
    # for batch_idx in range(input_pcs.shape[0]):
    #     idx = np.arange(input_pcs[batch_idx].shape[0])
    #     np.random.shuffle(idx)
    #     while len(idx) < conf.num_point_per_shape:
    #         idx = np.concatenate([idx, idx])
    #     idx = idx[:conf.num_point_per_shape]
    #     pcs_id = pcs_id + (torch.tensor(np.array(idx)), )
    # input_pcid2 = torch.stack(pcs_id, dim=0).long().reshape(-1)
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
    # print("shape1:", input_pcs.shape)
    # input_pxids = input_pxids[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
    input_movables = input_movables[input_pcid1, input_pcid2].reshape(batch_size, conf.num_point_per_shape)

    # input_dirs1 = torch.cat(batch[data_features.index('gripper_direction_camera')], dim=0).to(conf.device)  # B x 3 # up作为feature
    # input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction_camera')], dim=0).to(conf.device)  # B x 3   # forward
    # input_dirs1 = torch.cat(batch[data_features.index('gripper_direction_world')], dim=0).to(conf.device)  # B x 3 # up作为feature
    # input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction_world')], dim=0).to(conf.device)  # B x 3   # forward
    # print('output: ', batch[data_features.index('task_motion')])
    actual_motion = torch.Tensor(batch[data_features.index('gt_motion')]).to(conf.device)  # B  # 度数
    # gt_motion = torch.Tensor(batch[data_features.index('gt_motion')])  # B  # 真实角度, 在visuliaze的时候有用
    # task_waypoints = torch.Tensor(batch[data_features.index('task_waypoints')]).to(conf.device)     # 取 waypoint, 4*3 (初始一定是(0,0,0), gripper坐标系)
    # print(task_waypoints.shape)
    # task_traj = torch.cat([torch.cat([input_dirs1, input_dirs2], dim=1).view(conf.batch_size, 1, 6), task_waypoints], dim=1).view(conf.batch_size, conf.num_steps, 6)  # up和forward两个方向拼起来 + waypoints
    # contact_point = torch.Tensor(batch[data_features.index('position_world')]).to(conf.device)

    # forward through the network
    # print('input_pcs: ', input_pcs.shape)
    # print('task_traj: ', task_traj)
    # print('cp: ', contact_point)
    # print(task_traj.shape)
    # print(contact_point.shape)
    # print(input_pcs.shape)
    # loss = network.get_loss(input_pcs, actual_motion, contact_point, critic=critic, actor=actor)  # B x 2, B x F x N
    # pred_result_logits = network(task_motion, task_traj, contact_point)  # B x 2, B x F x N
    with torch.no_grad():
        pred_action_score_map = network.inference_action_score(input_pcs, actual_motion)
        pred_action_score_map = pred_action_score_map.cpu().numpy()
    # ipdb.set_trace()
    pred_action_score_map = pred_action_score_map * input_movables[0].view(-1, 1).cpu().numpy()

    # save action_score_map
    base_dir = conf.base_dir

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    fn = os.path.join(base_dir, 'action_score_map_full_' + str(step) + "_" + str(actual_motion[0].cpu().detach().numpy()))
    pc_world = input_pcs[0].cpu().numpy()
    pc_cam = (np.linalg.inv(mat44[:3, :3]) @ pc_world.T).T
    # ctpts = (np.linalg.inv(mat44[:3, :3]) @ ctpts.T).T
    utils.render_pts_label_png(fn,  pc_cam, pred_action_score_map)




if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--data_dir_prefix', type=str, help='data directory')
    # parser.add_argument('--val_data_fn', type=str, help='data directory', default='data_tuple_list_val_subset.txt')
    parser.add_argument('--train_shape_fn', type=str, help='training shape file that indexs all shape-ids')
    parser.add_argument('--ins_cnt_fn', type=str, help='a file listing all category instance count')
    parser.add_argument('--actor_path', type=str, help='a file listing all category instance count')
    parser.add_argument('--critic_path', type=str, help='a file listing all category instance count')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    #parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    # parser.add_argument('--abs_thres', type=float, default=0.01, help='abs thres')
    # parser.add_argument('--rel_thres', type=float, default=0.5, help='rel thres')
    # parser.add_argument('--dp_thres', type=float, default=0.5, help='dp thres')
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--wp_xyz', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--rotation_loss', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--buffer_max_num', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=500)
    parser.add_argument('--sample_succ', action='store_true', default=False)
    parser.add_argument('--angle_system', type=int, default=0)
    parser.add_argument('--num_train', type=int, default=2000)
    parser.add_argument('--num_eval', type=int, default=200)
    parser.add_argument('--degree_lower', type=int, default=15)

    # loss weights
    parser.add_argument('--lbd_kl', type=float, default=1.0)
    parser.add_argument('--lbd_dir', type=float, default=1.0)
    parser.add_argument('--lbd_recon', type=float, default=1.0)

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    parser.add_argument('--actor_eval_epoch', type=str, default='-1', help='num batch every visu')
    parser.add_argument('--critic_eval_epoch', type=str, default='-1', help='num batch every visu')
    parser.add_argument('--topk', type=int, default=10, help='num batch every visu')

    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--offline_data_dir2', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir3', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir4', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir5', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir6', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir7', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir8', type=str, default='xxx', help='data directory')
    parser.add_argument('--offline_data_dir9', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_dir2', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir3', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir4', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir5', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir6', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir7', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir8', type=str, default='xxx', help='data directory')
    parser.add_argument('--val_data_dir9', type=str, default='xxx', help='data directory')
    parser.add_argument('--train_num_data_uplimit', type=int, default=100000)
    parser.add_argument('--val_num_data_uplimit', type=int, default=100000)

    # parse args
    conf = parser.parse_args()


    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.primact_type}-{conf.category_types}-{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)
    conf.tb_dir = os.path.join(conf.exp_dir, 'tb')
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.mkdir(conf.exp_dir)
        os.mkdir(conf.tb_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
        if not conf.no_visu:
            os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # prepare data_dir
    # conf.data_dir = conf.data_dir_prefix + '-' + conf.exp_name
    # if os.path.exists(conf.data_dir):
    #     if not conf.resume:
    #         if not conf.overwrite:
    #             response = input('A data_dir named "%s" already exists, overwrite? (y/n) ' % conf.data_dir)
    #             if response != 'y':
    #                 exit(1)
    #         shutil.rmtree(conf.data_dir)
    # else:
    #     if conf.resume:
    #         raise ValueError('ERROR: no data_dir named %s to resume!' % conf.data_dir)
    # if not conf.resume:
    #     os.mkdir(conf.data_dir)

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    if not conf.resume:
        torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog


    # backup python files used for this training
    if not conf.resume:
        os.system('cp datagen.py data.py models/%s.py %s %s' % (conf.model_version, __file__, conf.exp_dir))
     
    # set training device
    device = torch.device(conf.device)
    conf.device = device

    if conf.category_types is None:
        conf.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture', 'Switch', 'TrashCan', 'Window']
    else:
        conf.category_types = conf.category_types.split(',')

    # read train_shape_fn
    train_shape_list = []
    # with open(conf.train_shape_fn, 'r') as fin:
    #     for l in fin.readlines():
    #         shape_id, category = l.rstrip().split()
    #         if category in conf.category_types:
    #             train_shape_list.append((shape_id, category))
    # utils.printout(flog, 'len(train_shape_list): %d' % len(train_shape_list))


    train_data_list = []
    # all_train_data_list = []
    for root, dirs, files in os.walk(conf.offline_data_dir):
        for dir in dirs:
            train_data_list.append(os.path.join(conf.offline_data_dir, dir))
            # all_train_data_list.append(os.path.join(conf.offline_data_dir, dir))
        break       # 只找一级目录
    if conf.offline_data_dir2 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir2):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir2, dir))
            break       # 只找一级目录
    if conf.offline_data_dir3 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir3):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir3, dir))
            break       # 只找一级目录
    if conf.offline_data_dir4 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir4):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir4, dir))
            break       # 只找一级目录
    if conf.offline_data_dir5 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir5):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir5, dir))
            break       # 只找一级目录
    if conf.offline_data_dir6 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir6):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir6, dir))
            break       # 只找一级目录
    if conf.offline_data_dir7 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir7):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir7, dir))
            break       # 只找一级目录
    if conf.offline_data_dir8 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir8):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir8, dir))
            break       # 只找一级目录
    if conf.offline_data_dir9 != 'xxx':
        for root, dirs, files in os.walk(conf.offline_data_dir9):
            for dir in dirs:
                train_data_list.append(os.path.join(conf.offline_data_dir9, dir))
            break       # 只找一级目录
    utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))
    print('train_data_list: ', train_data_list)

    val_data_list = []
    for root, dirs, files in os.walk(conf.val_data_dir):
        for dir in dirs:
            val_data_list.append(os.path.join(conf.val_data_dir, dir))
        break
    if conf.val_data_dir2 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir2):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir2, dir))
            break
    if conf.val_data_dir3 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir3):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir3, dir))
            break
    if conf.val_data_dir4 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir4):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir4, dir))
            break
    if conf.val_data_dir5 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir5):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir5, dir))
            break
    if conf.val_data_dir6 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir6):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir6, dir))
            break
    if conf.val_data_dir7 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir7):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir7, dir))
            break
    if conf.val_data_dir8 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir8):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir8, dir))
            break
    if conf.val_data_dir9 != 'xxx':
        for root, dirs, files in os.walk(conf.val_data_dir9):
            for dir in dirs:
                val_data_list.append(os.path.join(conf.val_data_dir9, dir))
            break

     
    ### start training
    print('train_data_list: ', train_data_list[0])
    train(conf, train_shape_list, train_data_list, val_data_list)


    ### before quit
    # close file log
    flog.close()