CUDA_VISIBLE_DEVICES=3 xvfb-run -a python eval_3d_task_traj_score.py \
    --exp_suffix 2021051338 \
    --model_version model_3d_task_score_topk \
    --primact_type pushing \
    --category_types StorageFurniture \
    --data_dir_prefix ../data/gt_data \
    --offline_data_dir /home/wuruihai/where2actCode/data/gt_data-fixcam-Type-single_cabinet_train_data-pushing0p05_TD3_sample_concat/allShape_StorageFurniture_2021051338/NEW_TRAIN1 \
    --val_data_dir /media/wuruihai/sixt/where2act/logs/exp-model_3d_task_actor_dir_RL_raw-pushing-None-52601/45783_task40_diffProposal_1/EVAL \
    --affordance_dir /home/wuruihai/where2actCode/logs/exp-model_3d_task_score_topk-pushing-None-2021051338/ckpts/%d-network.pth \
    --train_shape_fn ../stats/train_StorageFurniture_train_data_list.txt \
    --ins_cnt_fn ../stats/ins_cnt_single_cabinet.txt \
    --buffer_max_num 512 \
    --feat_dim 128   \
    --num_steps 5    \
    --batch_size 1  \
    --angle_system 1 \
    --num_train 30000 \
    --num_eval  30000  \
    --eval_epoch 30 \
    --degree_lower 10 \
    --base_dir /media/wuruihai/sixt/where2act/logs/exp-model_3d_task_actor_dir_RL_raw-pushing-None-52502/45783_task40_diffProposal_1/eval_heatmap_png \
    --train_num_data_uplimit 720  \
    --val_num_data_uplimit 50  \




