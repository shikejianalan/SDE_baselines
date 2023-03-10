import os
import random
import numpy as np
import torch
import utils
from data import SAPIENVisionDataset
from argparse import ArgumentParser
from affor_tools.affordance_render import affordance_render


def main(conf, train_data_list):
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction_camera', 'gripper_forward_direction_camera', \
            'result', 'cur_dir', 'shape_id', 'trial_id', 'is_original']

    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num, \
            abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres, img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal)
    train_dataset.load_data(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
                num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)
    print("train_num_batch: ", train_num_batch)

    train_batches = enumerate(train_dataloader, 0)

    for train_batch_ind, batch in train_batches:
        input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)     # B x 3N x 3
        print("input_pcs.size(): ", input_pcs.size())
        """For test only"""
        affordance_pcs = input_pcs[8]
        afford_map = torch.zeros(affordance_pcs.size(0), device=conf.device)
        affordance_render(conf, affordance_pcs, afford_map, out_fn="./output/afford_test_zero_2.png")
        afford_map = torch.ones(affordance_pcs.size(0), device=conf.device)
        affordance_render(conf, affordance_pcs, afford_map, out_fn="./output/afford_test_one_2.png")
        afford_map_yellow = afford_map * .75
        affordance_render(conf, affordance_pcs, afford_map_yellow, out_fn="./output/afford_test_yellow_2.png")
        afford_map_green = afford_map * .25
        affordance_render(conf, affordance_pcs, afford_map_green, out_fn="./output/afford_test_green_2.png")
        exit()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='random seed (for reproducibility) [specify -1 means to generate a random one]')

    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')

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
    parser.add_argument('--batch_size', type=int, default=32)

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
