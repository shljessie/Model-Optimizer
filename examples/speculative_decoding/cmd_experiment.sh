export HF_HUB_CACHE=/mnt/md0/hf_cache/
export HF_TOKEN=$HF_TOKEN

BASE_MODEL=openai/gpt-oss-120b
DATA=/root/eagle/ModelOptNew/eagle_training_data/all_longcontext_samples_with_templates.jsonl
INIT_CKPT=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_pretrain_yolo_mar23_r4/train_logs/
LR=2e-4
TRAIN_BS=1
GRAD_ACCUM_STEPS=16
LOGDIR=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_r4_continue_longctx_bs96_lr2e-4-128k-swa128/train_logs/
export GPU_PER_NODE=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

echo "Using $GPU_PER_NODE GPU(s) per node: $CUDA_VISIBLE_DEVICES"
echo "Logging to $LOGDIR"

mkdir -p $LOGDIR

./launch_train.sh --model $BASE_MODEL \
            --output_dir $LOGDIR \
            --data $DATA  \
            --num_epochs 1 \
            --training_seq_len 131072 \
            --lr $LR \
            --warmup_steps 500 --decay_steps 500 \
            --eagle_config eagle_config.json --train_bs $TRAIN_BS --mix_hidden_states True --gradient_accumulation_steps $GRAD_ACCUM_STEPS --bucket_granularity 8192 --ar_validate_steps -1 --log_steps 8 --num_ttt_steps 3 --init_from_checkpoint $INIT_CKPT

echo "Done! Logs and checkpoints are saved in $LOGDIR"
