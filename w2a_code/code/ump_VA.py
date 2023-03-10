"""
    Train the Action Scoring Module only
"""

import os
import time
import datetime
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
from datagen import DataGen
from data import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from umpnet_baseline.model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))
import render_using_blender as render_utils


def train(conf, train_shape_list=None, train_data_list=None, val_data_list=None, all_train_data_list=None):
    print('start')
    # create training and validation datasets and data loaders
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction_camera', 'gripper_forward_direction_camera', \
            'result', 'cur_dir', 'shape_id', 'trial_id', 'is_original', 'rgb-d']
     
    # Initialization of model, optimizer, replay buffer
    network = Model(num_directions=conf.num_direction, model_type=conf.model_type)
    print(network)

    pos_optimizer = torch.optim.Adam(network.pos_model.parameters(), lr=conf.learning_rate, betas=(0.9, 0.95))
    dir_optimizer = torch.optim.Adam(network.dir_model.parameters(), lr=conf.learning_rate, betas=(0.9, 0.95))
    pos_scheduler = torch.optim.lr_scheduler.StepLR(pos_optimizer, step_size=conf.learning_rate_decay, gamma=0.5)
    dir_scheduler = torch.optim.lr_scheduler.StepLR(dir_optimizer, step_size=conf.learning_rate_decay, gamma=0.5)
    # replay_buffer = ReplayBuffer(conf.replay_buffer_dir, conf.replay_buffer_size)

    # Set device
    device_pos = torch.device(f'cuda:0')
    device_dir = torch.device(f'cuda:0')
    network = network.to(device_pos, device_dir)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TotalLoss'
    # if not conf.no_tb_log:
    #     # https://github.com/lanpa/tensorboard-pytorch
    #     from tensorboardX import SummaryWriter
    #     train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'train'))
    #     val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'val'))

    # # send parameters to device
    # device = torch.cuda.set_device(conf.device)
    # network.to(conf.device)
    # utils.optimizer_to_device(network_opt, conf.device)

    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres, img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal)
    
    train_dataset.load_data(train_data_list)

    val_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres, img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal)
    
    val_dataset.load_data(val_data_list)
    utils.printout(conf.flog, str(val_dataset))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    val_num_batch = len(val_dataloader)

    # create a data generator
    datagen = DataGen(conf.num_processes_for_datagen, conf.flog)

        # sample succ
    if conf.sample_succ:
        sample_succ_list = []
        sample_succ_dirs = []

    # start training
    start_time = time.time()

    last_train_console_log_step, last_val_console_log_step = None, None

    # if resume
    start_epoch = 0
    # if conf.resume:
    #     # figure out the latest epoch to resume
    #     for item in os.listdir(os.path.join(conf.exp_dir, 'ckpts')):
    #         if item.endswith('-train_dataset.pth'):
    #             start_epoch = max(start_epoch, int(item.split('-')[0]))

    #     # load states for network, optimizer, lr_scheduler, sample_succ_list
    #     data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % start_epoch))
    #     network.load_state_dict(data_to_restore)
    #     data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % start_epoch))
    #     network_opt.load_state_dict(data_to_restore)
    #     data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % start_epoch))
    #     network_lr_scheduler.load_state_dict(data_to_restore)

    #     # rmdir and make a new dir for the current sample-succ directory
    #     old_sample_succ_dir = os.path.join(conf.data_dir, 'epoch-%04d_sample-succ' % (start_epoch - 1))
    #     utils.force_mkdir(old_sample_succ_dir)

    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):
        ### collect data for the current epoch
        # if epoch > start_epoch:
        #     utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Waiting epoch-{epoch} data ]')
        #     train_data_list = datagen.join_all()
        #     utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Gathered epoch-{epoch} data ]')
        #     cur_data_folders = []
        #     for item in train_data_list:
        #         item = '/'.join(item.split('/')[:-1])
        #         if item not in cur_data_folders:
        #             cur_data_folders.append(item)
        #     for cur_data_folder in cur_data_folders:
        #         with open(os.path.join(cur_data_folder, 'data_tuple_list.txt'), 'w') as fout:
        #             for item in train_data_list:
        #                 if cur_data_folder == '/'.join(item.split('/')[:-1]):
        #                     fout.write(item.split('/')[-1]+'\n')

            # load offline-generated sample-random data
            # for item in all_train_data_list:
            #     # valid_id_l = conf.num_interaction_data_offline + conf.num_interaction_data * (epoch-1)
            #     # # print(valid_id_l)
            #     # # print(valid_id_r)
            #     # # print(int(item.split('_')[-1]))
            #     # # print(item)
            #     # valid_id_r = conf.num_interaction_data_offline + conf.num_interaction_data * epoch
            #     # if valid_id_l <= int(item.split('_')[-1]) < valid_id_r:
            #         # print('add new data')
            #     train_data_list.append(item)

        ### start generating data for the next epoch
        # sample succ
        # if conf.sample_succ:
        #     if conf.resume and epoch == start_epoch:
        #         sample_succ_list = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-sample_succ_list.pth' % start_epoch))
        #     else:
        #         torch.save(sample_succ_list, os.path.join(conf.exp_dir, 'ckpts', '%d-sample_succ_list.pth' % epoch))
        #     for item in sample_succ_list:
        #         datagen.add_one_recollect_job(item[0], item[1], item[2], item[3], item[4], item[5], item[6])
        #     sample_succ_list = []
        #     sample_succ_dirs = []
        #     cur_sample_succ_dir = os.path.join(conf.data_dir, 'epoch-%04d_sample-succ' % epoch)
        #     utils.force_mkdir(cur_sample_succ_dir)

        # start all jobs
        # datagen.start_all()
        # utils.printout(conf.flog, f'  [ {strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} Started generating epoch-{epoch+1} data ]')

        # ### load data for the current epoch
        # if conf.resume and epoch == start_epoch:
        #     train_dataset = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-train_dataset.pth' % start_epoch))
        # else:
        #     train_dataset.load_data(all_train_data_list)
        # utils.printout(conf.flog, str(train_dataset))
        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
        #         num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
        # train_num_batch = len(train_dataloader)

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1

        ### train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step
            
            # save checkpoint
            if train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    # Save model and optimizer
                    save_state = {
                        'pos_state_dict': network.pos_model.state_dict(),
                        'dir_state_dict': network.dir_model.state_dict(),
                        'pos_optimizer': pos_optimizer.state_dict(),
                        'dir_optimizer': dir_optimizer.state_dict(),
                        'epoch': epoch + 1
                    }
                    torch.save(save_state, os.path.join(conf.exp_dir, 'ckpts', 'latest.pth'))
                    shutil.copyfile(
                        os.path.join(conf.exp_dir, 'ckpts','latest.pth'),
                        os.path.join(conf.exp_dir, 'ckpts','epoch_%06d.pth' % (epoch + 1))
                    )

                    # torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    # torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    # torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    # torch.save(train_dataset, os.path.join(conf.exp_dir, 'ckpts', '%d-train_dataset.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()

            # forward pass
            # print(len(batch[0]))


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--data_dir_prefix', type=str, help='data directory')
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_fn', type=str, help='data directory', default='data_tuple_list_val_subset.txt')
    parser.add_argument('--train_shape_fn', type=str, help='training shape file that indexs all shape-ids')
    parser.add_argument('--ins_cnt_fn', type=str, help='a file listing all category instance count')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:2', help='cpu or cuda:x for using cuda on GPU number x')
    #parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--abs_thres', type=float, default=0.01, help='abs thres')
    parser.add_argument('--rel_thres', type=float, default=0.5, help='rel thres')
    parser.add_argument('--dp_thres', type=float, default=0.5, help='dp thres')
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')

    # training parameters
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--buffer_max_num', type=int, default=20000)
    parser.add_argument('--num_processes_for_datagen', type=int, default=20)
    parser.add_argument('--num_interaction_data_offline', type=int, default=5)
    parser.add_argument('--num_interaction_data', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--sample_succ', action='store_true', default=False)

    # loss weights

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # umpnet conf
    parser.add_argument('--learning_rate', default=8e-3, type=float, help='learning rate of the optimizer')
    parser.add_argument('--learning_rate_decay', default=500, type=int, help='learning rate decay')
    parser.add_argument('--num_direction', default=64, type=int, help='number of directions')
    parser.add_argument('--model_type', default='sgn_mag', type=str, choices=['sgn', 'mag', 'sgn_mag'], help='model_type')


    # parse args
    conf = parser.parse_args()
    start = datetime.datetime.now()
    start = start.strftime("%Y-%m-%d-%H:%M:%S")
    print(start)
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.primact_type}-{conf.category_types}-{conf.exp_suffix}-{start}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
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
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
        if not conf.no_visu:
            os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # prepare data_dir
    conf.data_dir = conf.data_dir_prefix + '-' + conf.exp_name
    if os.path.exists(conf.data_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A data_dir named "%s" already exists, overwrite? (y/n) ' % conf.data_dir)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.data_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no data_dir named %s to resume!' % conf.data_dir)
    if not conf.resume:
        os.mkdir(conf.data_dir)

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

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # backup python files used for this training
    if not conf.resume:
        os.system('cp datagen.py data.py models/%s.py %s %s' % (conf.model_version, __file__, conf.exp_dir))
     
    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device
    
    # parse params
    utils.printout(flog, 'primact_type: %s' % str(conf.primact_type))

    if conf.category_types is None:
        conf.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture', 'Switch', 'TrashCan', 'Window']
    else:
        conf.category_types = conf.category_types.split(',')
    utils.printout(flog, 'category_types: %s' % str(conf.category_types))
    
    # read cat2freq
    conf.cat2freq = dict()
    with open(conf.ins_cnt_fn, 'r') as fin:
        for l in fin.readlines():
            category, _, freq = l.rstrip().split()
            conf.cat2freq[category] = int(freq)
    utils.printout(flog, str(conf.cat2freq))

    # read train_shape_fn
    train_shape_list = []
    with open(conf.train_shape_fn, 'r') as fin:
        for l in fin.readlines():
            shape_id, category = l.rstrip().split()
            if category in conf.category_types:
                train_shape_list.append((shape_id, category))
    utils.printout(flog, 'len(train_shape_list): %d' % len(train_shape_list))
    
    with open(os.path.join(conf.offline_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_train_data_list = [os.path.join(conf.offline_data_dir, l.rstrip()) for l in fin.readlines()]
    utils.printout(flog, 'len(all_train_data_list): %d' % len(all_train_data_list))
    if conf.resume:
        train_data_list = None
    else:
        train_data_list = []
        for item in all_train_data_list:
            if int(item.split('_')[-1]) < conf.num_interaction_data_offline + 100:
                train_data_list.append(item)
        utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))
    
    with open(os.path.join(conf.val_data_dir, conf.val_data_fn), 'r') as fin:
        val_data_list = [os.path.join(conf.val_data_dir, l.rstrip()) for l in fin.readlines()]
    utils.printout(flog, 'len(val_data_list): %d' % len(val_data_list))
     
    ### start training
    train(conf, train_shape_list, train_data_list, val_data_list, all_train_data_list)