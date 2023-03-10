xvfb-run -a python visu_action_heatmap_proposals.py \
  --exp_name final_exp-model_3d-pushing-vat-None-train_3d-2023-02-23-20:50:08 \
  --model_epoch 299 \
  --model_version model_3d \
  --shape_id $1 \
  --algorithm vat \
  --overwrite

