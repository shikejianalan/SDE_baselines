import numpy as np
import os
import torch
result_main_path = '/root/autodl-tmp/skj/Manipulation_SDE/code/logs/2023-03-01-07:46:49-pushing-w2a-highest'

CatList = os.listdir(result_main_path) # current directory
total_result = {}
avg_result = {}
for cat in CatList:
    print(cat)
    total_result[cat] = []
    cat_dir = os.path.join(result_main_path, cat)
    if os.path.isdir(cat_dir) == True:
        # print(cat_dir)

        for i, j, k in os.walk(cat_dir): # root, dirs, files
            for file in k:
                if file.endswith('.results'):
                    result_file = os.path.join(i, file)
                    result = torch.load(result_file)
                    single_srr = result['succ_time']/result['total_time']
                    total_result[cat].append(single_srr)

print(total_result)
all_train_avg = []
all_test_avg = []
for cat in ['Box', 'Switch', 'TrashCan', 'Refrigerator', 'Kettle', 'Window', 'Faucet', 'StorageFurniture', 'Microwave', 'Door']:
    avg_cat = round(np.array(total_result[cat]).sum()/len(total_result[cat])*100, 1)
    avg_result[cat] = avg_cat
    all_train_avg.append(avg_cat)

avg_result['all_train_avg'] = round(np.array(all_train_avg).sum()/len(all_train_avg), 1)

for cat in ['KitchenPot', 'WashingMachine', 'Bucket', 'Safe', 'Table']:
    avg_cat = round(np.array(total_result[cat]).sum()/len(total_result[cat])*100, 1)
    avg_result[cat] = avg_cat
    all_test_avg.append(avg_cat)

avg_result['all_test_avg'] = round(np.array(all_test_avg).sum()/len(all_test_avg), 1)

print(avg_result)

# write result to latex.txt
with open('latex_result.txt', 'a') as f:
    f.write('Where2Act &')
    for cat in ['all_train_avg', 'Box', 'Switch', 'TrashCan', 'Refrigerator', 'Kettle', 'Window', 'Faucet', 'StorageFurniture', 'Microwave', 'Door','all_test_avg' , 'KitchenPot', 'WashingMachine', 'Bucket', 'Safe', 'Table']:
        if cat == 'Table':
            f.write(f' {avg_result[cat]} \\\ \n')
        else:
            f.write(f' {avg_result[cat]} &')

    f.close()


