import datetime
import multiprocessing
import os
import torch
import numpy as np

def run_script(script):
    os.system(script)


device_count = torch.cuda.device_count()
print(device_count)

all_test_lists_dir1 = '/root/autodl-tmp/skj/where2act/stats/train_10cats_test_data_list.txt'
all_test_lists_dir2 = '/root/autodl-tmp/skj/where2act/stats/test_5cats_data_list.txt'
cats_dict = {}
to_do_processes = []
device = 0
for file_name in [all_test_lists_dir1, all_test_lists_dir2]:
    with open(file_name, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in cats_dict.keys():
                cats_dict[cat] = []
            cats_dict[cat].append((shape_id))
            # to_do_processes.append()

# print(cats_dict.keys())
# for key in cats_dict.keys():
#     print(len(cats_dict[key]))

scripts = []
cuda_sel = 0
# total_jobs = 0
angle_count = 0
angle_list = np.load('angles_list.npy')
print(angle_list.shape)
start = datetime.datetime.now()
start = start.strftime("%Y-%m-%d-%H:%M:%S")
for cat_key in cats_dict.keys():
    save_dir = os.path.join('logs', f'{start}-pushing-vat-highest', cat_key)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for shape_id in cats_dict[cat_key]:
        # print(shape_id, cuda_sel)
        scripts.append(["xvfb-run -a python eval_tool_multi_thread_baseline.py --shape_id {} --articu_angle {} --save_dir {} --cuda_sel {}".format(shape_id, angle_list[angle_count], save_dir, cuda_sel)])
        angle_count += 1
        cuda_sel += 1
        if cuda_sel >= 4:
            cuda_sel = 0

while len(scripts) != 0:
    processes = []
    for script in scripts[:12]:
        p = multiprocessing.Process(target=run_script, args=(script))
        # total_jobs += 1
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    scripts = scripts[12:]

# print(total_jobs)
print("Complete")
# result_dir = ''
# for cat_key in cats_dict.keys():
#     shape_list = cats_dict[cat_key]
#     for shape in shape_list:
#         shape_file = os.path.join(result_dir, shape, 'result.dict')