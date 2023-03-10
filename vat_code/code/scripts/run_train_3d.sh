xvfb-run -a python train_3d.py \
    --exp_suffix train_3d \
    --model_version model_3d \
    --primact_type pushing \
    --data_dir_prefix ../data/gt_data \
    --offline_data_dir ../data/gt_data-train_10cats_pushing_vat_5epochs \
    --val_data_dir ../data/gt_data-test_10cats_pushing_vat_5epochs \
    --val_data_fn data_tuple_list.txt \
    --train_shape_fn ../stats/train_10cats_train_data_list.txt \
    --ins_cnt_fn ../stats/ins_cnt_15cats.txt \
    --buffer_max_num 2000 \
    --num_processes_for_datagen 50 \
    --num_interaction_data_offline 1 \
    --num_interaction_data 1 \
    --sample_succ \
    --pretrained_critic_ckpt /root/autodl-tmp/skj/where2act_vat/code/logs/exp-model_3d_critic-pushing-None-train_3d_critic-2023-02-23-02:28:44/ckpts/270-network.pth \
    --device cuda:1 \
    --overwrite

# xvfb-run -a python train_3d.py \
#     --exp_suffix train_3d \
#     --model_version model_3d \
#     --primact_type pulling \
#     --data_dir_prefix ../data/gt_data \
#     --offline_data_dir ../data/gt_data-train_10cats_train_data-pulling_5epochs \
#     --val_data_dir ../data/gt_data-train_10cats_test_data-pulling_5epochs \
#     --val_data_fn data_tuple_list.txt \
#     --train_shape_fn ../stats/train_10cats_train_data_list.txt \
#     --ins_cnt_fn ../stats/ins_cnt_15cats.txt \
#     --buffer_max_num None \
#     --num_processes_for_datagen 50 \
#     --num_interaction_data_offline 1 \
#     --num_interaction_data 1 \
#     --sample_succ \
#     --pretrained_critic_ckpt /root/autodl-tmp/skj/where2act/code/logs/exp-model_3d_critic-pulling-None-train_3d_critic/ckpts/199-network.pth \
#     --device cuda:1 \
#     --overwrite
