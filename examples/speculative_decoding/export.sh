export HF_HUB_CACHE=/mnt/md0/hf_cache/
export HF_TOKEN=$HF_TOKEN

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_yolo_mar22_r4/train_logs/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_yolo_mar22_r4/hf_export-final-1-epoch/

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_pretrain_yolo_mar23_r4/train_logs/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_pretrain_yolo_mar23_r4/hf_export-final-2-epoch/

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_r4_continue_debuglongctx/train_logs/checkpoint-1365/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_pretrain_yolo_mar23_r4/hf_export-longctx-step1365/

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_r4_continue_longctx_bs96/train_logs/checkpoint-1365/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_pretrain_yolo_mar23_r4/hf_export-longctx-bs96-step1365/

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_r4_continue_longctx_bs24_lr1e-4/train_logs/checkpoint-1364/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_pretrain_yolo_mar23_r4/hf_export-longctx-bs24_128k-step1365/

MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_r4_continue_longctx_bs96_lr2e-4-128k-swa128/train_logs/
EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_posttrain_128k_mar27_r4/hf_export-longctx-swa128/

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_r4_continue_longctx_bs96_lr2e-4-128k-fix/train_logs/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_posttrain_128k_mar27_r4/hf_export-longctx/

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_r4_continue_swa128/train_logs/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_r4_continue_swa128/hf_export-ft-experiment-swa/

# MODEL_PATH=/mnt/md0/eagle-training-metadata/train_logs/hpo/eagle3_gptoss_hpo_20260322_223101/run-4/checkpoint-2375/
# EXPORT_PATH=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_hpo_mar22_r3/hf_export-20260322_223101-run-4-checkpoint-2375/

mkdir -p $EXPORT_PATH

python scripts/export_hf_checkpoint.py --model_path $MODEL_PATH --export_path $EXPORT_PATH

# STEPS:
# Edit config.json:
# - change max_position_embeddings to 131072
# - add yarn configuration matching GPT-OSS 120B
# - set this on the top-level and eagle-level config:
# - -     "eagle_aux_hidden_state_layer_ids": [24,30,36],
# - set this on the top-level config:
#     "norm_before_fc": true,

# "target_layer_count": 36,
# "layer_types": ["sliding_attention"],
# "sliding_window": 128,
