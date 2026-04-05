source /root/eagle/vllm-dev-agentic/tokens.local.sh
# source /root/eagle/vllm-dev-agentic/.venv/bin/activate
source /root/eagle/vllm_centml_fork/.venv_py312/bin/activate

# MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
# DRAFTER=amazon/Qwen3-Coder-30B-A3B-Instruct-P-EAGLE
# DRAFTER=z-lab/Qwen3-Coder-30B-A3B-DFlash
# DRAFTER=lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge
# MODEL=nvidia/nemotron-super-rl-021826
# python3 prepare_data.py --dataset speed --config all

# MODEL=Qwen/Qwen3.5-122B-A10B
# PROJECT_DIR=qwen35_122b
# MODEL=Qwen/Qwen3.5-35B-A3B
# PROJECT_DIR=qwen35_35b
# MODEL=Qwen/Qwen3.5-27B
# PROJECT_DIR=qwen35_27b
# MODEL=openai/gpt-oss-20b
# DRAFTER=amazon/GPT-OSS-20B-P-EAGLE
# MODEL=Qwen/Qwen3-8B

# MODEL=moonshotai/Kimi-K2.5
# DRAFTER=lightseekorg/kimi-k2.5-eagle3

export CUDA_VISIBLE_DEVICES=3

export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

MODEL=openai/gpt-oss-120b
# DRAFTER=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_yolo_mar22_r4/hf_export-final-1-epoch/
DRAFTER=/mnt/md0/eagle-training-checkpoints/eagle3_gptoss_r4_continue_swa128/hf_export-ft-experiment-swa/

rm -f acceptance_rate.json responses.jsonl

DATASET=qualitative
DRAFT_LENGTH=3
BS=128
# PROJECT_DIR=gptoss_eagle_develop_yolopretrain_final_1epoch-mar23-k$DRAFT_LENGTH-bs$BS-speed$DATASET
PROJECT_DIR=gptoss_eagle_develop_pretrain-finetune-swa-mar23-k$DRAFT_LENGTH-bs$BS-speed$DATASET
mkdir -p $PROJECT_DIR
NUM_REQUESTS=880
echo "$PROJECT_DIR/log.txt"
python3 run.py --model_dir $MODEL --tokenizer $MODEL --draft_model_dir $DRAFTER --trust_remote_code --dataset speed --dataset_path data/speed/$DATASET --tp_size 1 --ep_size 1 --draft_length $DRAFT_LENGTH --output_length 4096 --engine VLLM --concurrency $BS --num_requests $NUM_REQUESTS --show_progress --speculative_algorithm EAGLE3 --postprocess gptoss --save_dir $PROJECT_DIR > $PROJECT_DIR/log.txt 2>&1 &

# DRAFTER=RedHatAI/Qwen3-8B-speculator.eagle3
# DRAFTER=/mnt/md0/speculators-training/qwen3_8b_magpie/checkpoints/9
# for DATASET in qualitative throughput_1k throughput_8k throughput_16k; do
#     for DRAFT_LENGTH in 3 7; do
#         # PROJECT_DIR=gpt_oss_20b_p_eagle_k$DRAFT_LENGTH-bs1
#         BS=128
#         if [ "$DATASET" = "qualitative" ]; then
#             BS=32
#         fi
#         PROJECT_DIR=qwen3_8b_speculators_local-normfix--withyarn-10epochs-magpie100k-mar1_k$DRAFT_LENGTH-bs$BS-speed$DATASET
#         mkdir -p $PROJECT_DIR
#         NUM_REQUESTS=1536
#         # 1536 for throughput_xx, 880 for qualitative
#         if [ "$DATASET" = "qualitative" ]; then
#             NUM_REQUESTS=880
#         fi

#         CUDA_VISIBLE_DEVICES=$GPU python3 run.py --model_dir $MODEL --tokenizer $MODEL --draft_model_dir $DRAFTER --dataset speed --dataset_path data/speed/$DATASET --tp_size 1 --ep_size 1 --draft_length $DRAFT_LENGTH --output_length 4096 --engine VLLM --concurrency $BS --num_requests $NUM_REQUESTS --show_progress --speculative_algorithm EAGLE3 --save_dir $PROJECT_DIR > $PROJECT_DIR/log.txt 2>&1 &
#         echo "Launched with PID $! on GPU $GPU with DRAFT_LENGTH $DRAFT_LENGTH"
#         GPU=$((GPU+1))
#     done
# done

# for DRAFT_LENGTH in 3 7; do
#     # PROJECT_DIR=gpt_oss_20b_p_eagle_k$DRAFT_LENGTH-bs1
#     PROJECT_DIR=qwen3_8b_speculators_public_k$DRAFT_LENGTH-bs32_speed1k
#     mkdir -p $PROJECT_DIR
#     CUDA_VISIBLE_DEVICES=$GPU python3 run.py --model_dir $MODEL --tokenizer $MODEL --draft_model_dir $DRAFTER --dataset speed --dataset_path data/speed/qualitative --tp_size 1 --ep_size 1 --draft_length $DRAFT_LENGTH --output_length 4096 --engine VLLM --concurrency 32 --num_requests 880 --show_progress --speculative_algorithm EAGLE3 --save_dir $PROJECT_DIR > $PROJECT_DIR/log.txt 2>&1 &
#     echo "Launched with PID $! on GPU $GPU with DRAFT_LENGTH $DRAFT_LENGTH"
#     GPU=$((GPU+1))
# done

# DRAFTER=RedHatAI/gpt-oss-20b-speculator.eagle3

# for DRAFT_LENGTH in 3; do
#     PROJECT_DIR=gpt_oss_20b_eagle3_k$DRAFT_LENGTH-bs1
#     mkdir -p $PROJECT_DIR
#     CUDA_VISIBLE_DEVICES=$GPU python3 run.py --model_dir $MODEL --tokenizer $MODEL --draft_model_dir $DRAFTER --dataset speed --dataset_path data/speed/qualitative --tp_size 1 --ep_size 1 --draft_length $DRAFT_LENGTH --output_length 512 --engine VLLM --concurrency 1 --num_requests 880 --show_progress --speculative_algorithm EAGLE3 --postprocess gptoss --save_dir $PROJECT_DIR > $PROJECT_DIR/log.txt 2>&1 &
#     echo "Launched with PID $! on GPU $GPU with DRAFT_LENGTH $DRAFT_LENGTH"
#     GPU=$((GPU+1))
# done

# wait for all procsesses to finish
wait

# for DRAFT_LENGTH in 4 10 18; do
#     python3 run.py --model_dir $MODEL --tokenizer $MODEL --draft_model_dir $DRAFTER --dataset humaneval --dataset_path openai/openai_humaneval --tp_size 1 --ep_size 1 --draft_length $DRAFT_LENGTH --output_length 256 --engine VLLM --concurrency 1 --show_progress --parallel_drafting
# done

# for DRAFT_LENGTH in 3 7 11 15; do
#     python3 run.py --model_dir $MODEL --tokenizer $MODEL --draft_model_dir $DRAFTER --dataset humaneval --dataset_path openai/openai_humaneval --tp_size 1 --ep_size 1 --draft_length $DRAFT_LENGTH --output_length 256 --engine VLLM --concurrency 1 --show_progress --speculative_algorithm EAGLE3
# done

# for DRAFT_LENGTH in 3 7 11 15; do
# done
