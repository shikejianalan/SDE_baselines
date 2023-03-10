export CUDA_VISIBLE_DEVICES=0,1,2,3
TIME=$(date "+%Y%m%d%H%M%S")
PRIMACT_TYPE=pushing
POLICY=prob
ALGORITHM=vat
EXP_NAME=eval_${TIME}_${PRIMACT_TYPE}_${ALGORITHM}_${POLICY}

xvfb-run -a python eval.py --exp_name $EXP_NAME \
    --seed 100 \
    --primact_type $PRIMACT_TYPE \
    --cond_len 256 \
    --model_path /root/autodl-tmp/skj/where2act_vat/code/logs/final_exp-model_3d-pushing-vat-None-train_3d-2023-02-23-20:50:08 \
    --model_epoch 299 \
    --model_version model_3d \
    --sampler EM \
    --algorithm $ALGORITHM \
    --policy $POLICY 

# export CUDA_VISIBLE_DEVICES=0
# TIME=$(date "+%Y%m%d%H%M%S")
# PRIMACT_TYPE=pulling
# EXP_NAME=eval_${TIME}_${PRIMACT_TYPE}_SDE_union_1999_after_debug_robot_fix_1

# xvfb-run -a python eval.py --exp_name $EXP_NAME \
#     --seed 100 \
#     --primact_type $PRIMACT_TYPE \
#     --cond_len 256 \
#     --model_path /home/jcheng/research_files/r10_AdaAfford/codes/Manip_Diff/code/logs/models/SDE_union_pulling_model_epoch_1999.pth \
#     --sampler EM
