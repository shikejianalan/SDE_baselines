import random
import os
import numpy as np

# import model_SDE_union
from argparse import ArgumentParser
from where2act_eval_tool_v1 import Evaluator
import utils
import torch
print(torch.cuda.device_count())


shape_id_list_all = [7119, 7263, 7296, 46906, 46981, 47088, 10620, 10685, 10905, 103242, 103253, 103255, 9032, 9035, 9041, 102154, 102165, 102177]
shape_id_list_micro =  [7119]#, 7263, 7296]
def main(conf):
    log_main_folder = os.path.join(conf.base_dir, conf.exp_name, conf.primact_type)
    gif_dir = os.path.join(log_main_folder, "gif")
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    aff_dir = os.path.join(log_main_folder, 'aff_files')
    if not os.path.exists(aff_dir):
        os.makedirs(aff_dir)
    #####################
    # diffusion model
    #####################
    # network = model_SDE_union.AssembleModel(conf)
    # network.load_state_dict(torch.load(conf.model_path))
    # network = network.to(conf.device)
    # network.eval()
    #####################
    # where2act model
    #####################
    # load train config
    train_conf = torch.load(os.path.join(conf.model_path, 'conf.pth'))

    # load model
    model_def = utils.get_model_module(conf.model_version, conf.algorithm)

    # set up device
    device = torch.device(conf.device)
    print(f'Using device: {device}')

    # create models
    network = model_def.Network(train_conf.feat_dim, train_conf.rv_dim, train_conf.rv_cnt)

    # load pretrained model
    print('Loading ckpt from ', os.path.join(conf.model_path, 'ckpts'), conf.model_epoch)

    data_to_restore = torch.load(os.path.join(conf.model_path, 'ckpts', '%d-network.pth' % conf.model_epoch), map_location=conf.device)
    network.load_state_dict(data_to_restore, strict=False)
    print('DONE\n')

    # send to device
    network.to(device)
    evaluator = Evaluator(conf)
    #camera_angle_list = evaluator.get_camera_angle_list()
    #articu_angle_list = evaluator.get_articu_angle_list()
    camera_angle_list = [[np.pi/5, np.pi/3 *2.3],]
    articu_angle_list = [np.pi/2 * (1/2), np.pi/2 * (2/3)]
    if conf.shape_type == "all":
        shape_id_list = shape_id_list_all
    elif conf.shape_type == "microwave":
        shape_id_list = shape_id_list_micro
    results = evaluator.test_contact(network, shape_id_list, camera_angle_list, articu_angle_list, gif_dir, aff_dir)
    succ_rate = results["succ_time"] / results["total_time"]
    print("succ_rate is: ", succ_rate)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--base_dir', type=str, default='logs', help='the base dir of the experiment')
    parser.add_argument('--exp_name', type=str, default='exp-model_3d-pushing-None-train_3d_1', help='Please set your exp name, all the output will be saved in the folder with this exp_name.')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_epoch', type=int, default=299)
    parser.add_argument('--model_version', type=str)
    # network and sampler setting
    parser.add_argument('--sampler', type=str, default='EM', help='Sampler options: EM, PC and ODE')
    parser.add_argument('--cond_len', type=int, default=128, help='The dimension of the condition')
    ## SDE network options
    parser.add_argument('--manipSDE_sigma', type=float, default=25.0)
    parser.add_argument('--manipSDE_snr', type=float, default=0.2)
    parser.add_argument('--manipSDE_num_steps', type=int, default=500)
    parser.add_argument('--manipSDE_input_dim', type=int, default=9)
    parser.add_argument('--manipSDE_cond_res_num', type=int, default=5)
    parser.add_argument('--manipSDE_feat_len', type=int, default=128)
    parser.add_argument('--manipSDE_time_embed_len', type=int, default=128)

    # test options:
    parser.add_argument('--how_many_times', type=int, default=1)
    parser.add_argument('--shape_type', type=str, default="microwave")
    parser.add_argument('--visu', default = True)
    parser.add_argument('--vis_gif', type=bool, default=True)
    parser.add_argument('--algorithm', type=str, default = 'w2a')
    parser.add_argument('--policy', type=str, default = 'prob')

    """render options"""
    parser.add_argument('--render_img_size', type=int, default=512)

    conf = parser.parse_args()

    if conf.seed >= 0:
        #conf.seed = random.randint(1, 10000)
        random.seed(conf.seed)
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)
    main(conf)