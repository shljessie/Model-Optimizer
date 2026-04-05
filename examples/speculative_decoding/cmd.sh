export HF_HUB_CACHE=/mnt/md0/hf_cache/
export HF_TOKEN=$HF_TOKEN

# BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
BASE_MODEL=openai/gpt-oss-120b
DATA=/mnt/md0/eagle-training-metadata/training_data/gptoss_eagle3_round5/all_pretrain_samples_with_templates.jsonl
# DATA=/root/eagle/ModelOptNew/eagle_training_data/all_pretrain_samples_with_templates.jsonl
# DATA=/root/eagle/ModelOptNew/eagle_training_data/all_pretrain_samples_downsampled_30k.jsonl
LR=4e-4
TRAIN_BS=1
GRAD_ACCUM_STEPS=256
export GPU_PER_NODE=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
MAX_GRAD_NORM=5
EFFECTIVE_BS=$((TRAIN_BS * GRAD_ACCUM_STEPS * GPU_PER_NODE))
echo "Effective batch size: $EFFECTIVE_BS (TRAIN_BS: $TRAIN_BS, GRAD_ACCUM_STEPS: $GRAD_ACCUM_STEPS, GPU_PER_NODE: $GPU_PER_NODE)"
# LOGDIR=/mnt/md0/eagle-training-metadata/train_logs/eagle_experiments/temp_logs_gptoss120b_remotehidden_anteater1epoch_mixhiddens_lr$LR-bs$TRAIN_BS-1gpu_r3_$VLLM_PORT-$(date +%Y%m%d_%H%M%S)
LOGDIR=/mnt/md0/eagle-training-metadata/train_logs/eagle3_gptoss_pretrain_4layers_fullattn_bs$EFFECTIVE_BS-lr$LR-maxgradnorm$MAX_GRAD_NORM/train_logs/

echo "Using $GPU_PER_NODE GPU(s) per node: $CUDA_VISIBLE_DEVICES"
echo "Logging to $LOGDIR"

mkdir -p $LOGDIR

./launch_train.sh --model $BASE_MODEL \
            --output_dir $LOGDIR \
            --data $DATA  \
            --num_epochs 1 \
            --training_seq_len 4096 \
            --lr $LR \
            --warmup_steps 500 --decay_steps 500 \
            --eagle_config eagle_config.json --train_bs $TRAIN_BS --max_grad_norm $MAX_GRAD_NORM --gradient_accumulation_steps $GRAD_ACCUM_STEPS --mix_hidden_states True --bucket_granularity 0 --ar_validate_steps -1 --log_steps 1 --num_ttt_steps 3 # --draft_vocab_cache /root/eagle/ModelOptNew/examples/speculative_decoding/draft_vocab_cache/Llama-3.2-1B-Instruct/d2t.pt

echo "Done! Logs and checkpoints are saved in $LOGDIR"
