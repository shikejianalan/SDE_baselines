import pc2b
import h5toply
import trajtoply
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', default='./ks-work', help='category name for training')
parser.add_argument('--input_path', default='/root/autodl-tmp/skj/where2act/data/gt_data-exp-model_3d_critic-pushing-None-train_3d_critic/epoch-0000_sample-succ/7310_Microwave_3_pushing_0', help='category name for training')
parser.add_argument('--way_id1', default=0,type=int, help='category name for training')
parser.add_argument('--way_id2', default=0,type=int, help='category name for training')
parser.add_argument('--way_id3', default=0,type=int, help='category name for training')
parser.add_argument('--p_id', default=0,type=int, help='category name for training')
parser.add_argument('--p_id2', default=0,type=int, help='category name for training')
parser.add_argument('--p_id3', default=0,type=int, help='category name for training')
parser.add_argument('--afford', default=0,type=int, help='category name for training')
parser.add_argument('--diff', default=0,type=int, help='category name for training')
FLAG = parser.parse_args()

trajtoply.t2pc(opt_dir=FLAG.opt_path,input_dir=FLAG.input_path,afford=FLAG.afford,way_id1=FLAG.way_id1,way_id2=FLAG.way_id2,way_id3=FLAG.way_id3,p_id=FLAG.p_id,diff=FLAG.diff,p_id2=FLAG.p_id2,p_id3=FLAG.p_id3)

h5toply.h52pc(opt_dir=FLAG.opt_path,input_dir=FLAG.input_path,afford=FLAG.afford,way_id1=FLAG.way_id1,p_id=FLAG.p_id)

pc2b.pc2b(pc_path=FLAG.opt_path)