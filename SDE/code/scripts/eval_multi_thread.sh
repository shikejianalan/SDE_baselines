export CUDA_VISIBLE_DEVICES=0,1,2,3
TIME=$(date "+%Y%m%d%H%M%S")
PRIMACT_TYPE=pushing
POLICY=highest
ALGORITHM=w2a
SAVE_DIR=${TIME}_${PRIMACT_TYPE}_${ALGORITHM}_${POLICY}

xvfb-run -a python eval_tool_multi_thread_baseline.py \
    --shape_id 7128 \
    --camera_angle_phi 0.6283185 \
    --camera_angle_theta 2.40855436775217 \
    --articu_angle 0.7853981633 \
    --save_dir logs/multi_debug \
    --primact_type $PRIMACT_TYPE \
    --model_path /root/autodl-tmp/skj/where2act_vat/code/logs/exp-model_3d-pushing-w2a-None-train_3d_5epochs \
    --model_epoch 299 \
    --model_version model_3d \
    --cuda_sel 0 \
    --algorithm $ALGORITHM \
    --policy $POLICY 
