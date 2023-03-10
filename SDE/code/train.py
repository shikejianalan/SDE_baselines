import os
import random
import numpy as np
import torch
import utils
from torch import nn
from data import SAPIENVisionDataset
from argparse import ArgumentParser
from affor_tools.affordance_render import affordance_render
from model_SDE import AssembleModel
from torch.utils.tensorboard import SummaryWriter
from pointnet2_ops.pointnet2_utils import furthest_point_sample


def main(conf, train_data_list):
    log_main_folder = os.path.join('logs', conf.exp_name)
    writer_log_path = os.path.join(log_main_folder, 'log')
    if not os.path.exists(writer_log_path):
        os.makedirs(writer_log_path)
    # Use tensorboard
    writer = SummaryWriter(writer_log_path)
    print('experiment start.')
    print('-'*70)
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction_camera', 'gripper_forward_direction_camera', \
            'result', 'cur_dir', 'shape_id', 'trial_id', 'is_original']

    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres, img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal)
    train_dataset.load_data(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
                num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)
    print("train_num_batch: ", train_num_batch)
    print('-'*70)

    """Create new model"""
    network = AssembleModel(conf)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network = nn.DataParallel(network)
    else:
        print("Single GPU Training!")
    network = network.to(conf.device)
    if conf.cont_train_start > 0:
        print('Continue training, load model from path: ', conf.cont_path)
        network.load_state_dict(torch.load(conf.cont_path, map_location=conf.device))
    elif conf.cont_train_start < 0:
        print('cont_train_start cannot smaller than 0!')
        raise ValueError

    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr)

    # model save path
    model_save_path = os.path.join(log_main_folder, 'model_save')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    iter_num = 0
    for epoch in range(conf.cont_train_start, conf.epochs):
        network.train()
        for train_batch_ind, batch in enumerate(train_dataloader, 0):
            # prepare input
            input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)     # B x 3N x 3
            input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)     # B x 3N x 2
            input_movables = torch.cat(batch[data_features.index('pc_movables')], dim=0).to(conf.device)     # B x 3N
            batch_size = input_pcs.shape[0]

            input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
            input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)                 # BN
            input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
            input_pxids = input_pxids[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
            input_movables = input_movables[input_pcid1, input_pcid2].reshape(batch_size, conf.num_point_per_shape)

            input_dirs1 = torch.cat(batch[data_features.index('gripper_direction_camera')], dim=0).to(conf.device)     # B x 3
            input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction_camera')], dim=0).to(conf.device)     # B x 3
            
            # prepare gt
            gt_result = torch.Tensor(batch[data_features.index('result')]).long().to(conf.device)     # B
            # gripper_img_target = torch.cat(batch[data_features.index('gripper_img_target')], dim=0).to(conf.device)     # B x 3 x H x W

            # print(gt_result.bool())

            network_opt.zero_grad()
            losses = network(input_pcs, input_dirs1, input_dirs2, gt_result.float()[:, None])
            afford_loss = torch.mean(losses["afford_loss"])
            pose_loss = losses["pose_loss"][gt_result.bool()]
            if pose_loss.size(0) == 0: # If no positive gt_result
                pose_loss = 0.
            else:
                pose_loss = torch.mean(pose_loss)
            total_loss = afford_loss + pose_loss
            total_loss.backward()
            network_opt.step()
            if (iter_num + 1) % conf.print_loss == 0:
                print("epoch {}, afford loss at ".format(epoch), iter_num, ": ", afford_loss.item())
                print("epoch {}, pose loss at ".format(epoch), iter_num, ": ", pose_loss.item())
                writer.add_scalars('Training loss:', {"afford loss": afford_loss.item(), "pose loss": pose_loss.item()}, iter_num)
            iter_num += 1

            """
            print("input_pcs.size(): ", input_pcs.size())
            # For test only
            print("gt_result: ", gt_result)
            exit()
            """
        if (epoch + 1) % conf.epoch_save == 0:
            utils.save_network(network, os.path.join(model_save_path, 'model_epoch_{}.pth'.format(epoch)))
        if (epoch + 1) % conf.val_every_epochs == 0:
            network.eval()
            print('validation at epoch {}.'.format(epoch))
            # The following content should be changed
            val_batch_size = input_pcs.size(0)
            val_num_point = input_pcs.size(1)
            gen_results = network.get_affordance(network.get_whole_feats(input_pcs))["x_states"][-1].reshape(val_batch_size, val_num_point)
            
            render_imgs = []
            for b_num in range(val_batch_size):
                render_imgs.append(affordance_render(conf, input_pcs[b_num], torch.clamp(gen_results[b_num], 0., 1.))["torch_image"])
            render_imgs = torch.stack(render_imgs, dim=0)
            writer.add_images('epoch_{}_gen'.format(epoch), render_imgs, 0, dataformats='NHWC')
    utils.save_network(network, os.path.join(model_save_path, 'model_final.pth'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    # experimental setting
    parser.add_argument('--exp_name', type=str, default='my_exp_1', help='Please set your exp name, all the output will be saved in the folder with this exp_name.')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')

    # dataset setting
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--buffer_max_num', type=int, default=20000)
    parser.add_argument('--abs_thres', type=float, default=0.01, help='abs thres')
    parser.add_argument('--rel_thres', type=float, default=0.5, help='rel thres')
    parser.add_argument('--dp_thres', type=float, default=0.5, help='dp thres')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--num_interaction_data_offline', type=int, default=5)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)

    # training setting
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--epoch_save', type=int, default=100)
    parser.add_argument('--val_every_epochs', type=int, default=100)
    parser.add_argument('--cont_train_start', type=int, default=0, help='If you want to continue training, please set this option greater than 0.')
    parser.add_argument('--cont_path', type=str, default=None, help='The model path for continue training.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--print_loss', type=int, default=100)

    # network and sampler setting
    parser.add_argument('--sampler', type=str, default='PC_origin', help='Sampler options: EM, PC and ODE')
    parser.add_argument('--cond_len', type=int, default=128, help='The dimension of the condition')
    ## Affordance network options
    parser.add_argument('--affordSDE_sigma', type=float, default=25.0)
    parser.add_argument('--affordSDE_snr', type=float, default=0.2)
    parser.add_argument('--affordSDE_num_steps', type=int, default=500)
    parser.add_argument('--affordSDE_input_dim', type=int, default=1)
    parser.add_argument('--affordSDE_cond_res_num', type=int, default=5)
    parser.add_argument('--affordSDE_feat_len', type=int, default=128)
    parser.add_argument('--affordSDE_time_embed_len', type=int, default=128)
    ## Pose network options
    parser.add_argument('--poseSDE_sigma', type=float, default=25.0)
    parser.add_argument('--poseSDE_snr', type=float, default=0.2)
    parser.add_argument('--poseSDE_num_steps', type=int, default=500)
    parser.add_argument('--poseSDE_input_dim', type=int, default=6)
    parser.add_argument('--poseSDE_cond_res_num', type=int, default=5)
    parser.add_argument('--poseSDE_feat_len', type=int, default=128)
    parser.add_argument('--poseSDE_time_embed_len', type=int, default=128)

    """render options"""
    parser.add_argument('--render_img_size', type=int, default=512)

    conf = parser.parse_args()

    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    if conf.category_types is None:
        conf.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture', 'Switch', 'TrashCan', 'Window']
    else:
        conf.category_types = conf.category_types.split(',')

    with open(os.path.join(conf.offline_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_train_data_list = [os.path.join(conf.offline_data_dir, l.rstrip()) for l in fin.readlines()]

    train_data_list = []
    for item in all_train_data_list:
        if int(item.split('_')[-1]) < conf.num_interaction_data_offline:
            train_data_list.append(item)

    main(conf, train_data_list)
