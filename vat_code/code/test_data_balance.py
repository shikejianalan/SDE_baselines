import numpy as np
import os
import json
from utils import printout

data_dir = '/root/autodl-tmp/skj/where2act/data/gt_data-train_10cats_train_data-pushing_5epochs'

data_list = os.listdir(data_dir)

print(len(data_list))
positive = 0
negative = 0
no_result = 0
for f in data_list:
    result_dir = os.path.join(data_dir, f, 'result.json')
    try:
        with open(result_dir, 'r') as f:
            result = json.load(f)
            success = result['result']
            if success == 'VALID':
                positive += 1
            else:
                negative += 1
    except:
        no_result += 1
    
print('positive', positive)
print('negative', negative)
print('no result', no_result)
print(positive + negative + no_result)