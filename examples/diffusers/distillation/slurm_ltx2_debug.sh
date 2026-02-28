#!/bin/bash
# LTX-2 Distillation Debug Script (General Trainer)
#
# Short debug run using mock data -- no real data or text encoder needed.
# Tests the full training pipeline: model loading, strategy, training loop,
# checkpointing, and validation.
#
# Usage:
#   ./slurm_ltx2_debug.sh [options] [config_overrides...]
#
# Examples:
#   # Launch with defaults (1 node, 8 GPUs, 30 min, mock data)
#   ./slurm_ltx2_debug.sh
#
#   # Dry run
#   ./slurm_ltx2_debug.sh --dry-run
#
#   # Override config values
#   ./slurm_ltx2_debug.sh optimization.steps=50 optimization.batch_size=2

set -eo pipefail

######################
### Default Config ###
######################

JOB_NAME="ltx2-general-debug"
PARTITION="batch_block1,interactive,batch_short,batch_singlenode"
ACCOUNT="adlr_psx_numerics"
TIME_LIMIT="00:30:00"

NUM_NODES=1
GPUS_PER_NODE=8

ROOT_PATH="/lustre/fs1/portfolios/adlr/projects/adlr_psx_numerics/users/mxin/ltx"
MODELOPT_PATH="$ROOT_PATH/sources/Model-Optimizer"
DISTILL_PATH="$MODELOPT_PATH/examples/diffusers/distillation"
CONTAINER="$ROOT_PATH/containers/ltx-distillation2.sqsh"

MODEL_PATH="$ROOT_PATH/models/LTX-2"
TRANSFORMER_PATH="$MODEL_PATH/ltx-2-19b-dev.safetensors"
GEMMA_PATH="$ROOT_PATH/models/gemma-3-12b-it-qat-q4_0-unquantized"

# Output
OUTPUT_DIR="$ROOT_PATH/results/$JOB_NAME"

# Training config
CONFIG="$DISTILL_PATH/configs/ltx2_distillation.yaml"
OVERRIDES=""

DRY_RUN=false

######################
### Parse Args     ###
######################

while [[ $# -gt 0 ]]; do
    case $1 in
        --job-name)     JOB_NAME="$2";       shift 2 ;;
        --partition)    PARTITION="$2";       shift 2 ;;
        --account)      ACCOUNT="$2";        shift 2 ;;
        --time)         TIME_LIMIT="$2";     shift 2 ;;
        --nodes)        NUM_NODES="$2";      shift 2 ;;
        --gpus)         GPUS_PER_NODE="$2";  shift 2 ;;
        --config)       CONFIG="$2";         shift 2 ;;
        --container)    CONTAINER="$2";      shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";     shift 2 ;;
        --model-path)   TRANSFORMER_PATH="$2"; shift 2 ;;
        --gemma-path)   GEMMA_PATH="$2";     shift 2 ;;
        --dry-run)      DRY_RUN=true;        shift ;;
        --help|-h)
            echo "Usage: $0 [options] [config_overrides...]"
            echo ""
            echo "Options:"
            echo "  --job-name NAME       Job name (default: $JOB_NAME)"
            echo "  --partition PART      SLURM partition"
            echo "  --account ACCT        SLURM account"
            echo "  --time TIME           Time limit HH:MM:SS (default: $TIME_LIMIT)"
            echo "  --nodes N             Number of nodes (default: $NUM_NODES)"
            echo "  --gpus N              GPUs per node (default: $GPUS_PER_NODE)"
            echo "  --config FILE         Training config YAML"
            echo "  --container FILE      Container image"
            echo "  --output-dir DIR      Output directory"
            echo "  --model-path PATH     LTX-2 checkpoint path"
            echo "  --gemma-path PATH     Gemma model directory"
            echo "  --dry-run             Print command without submitting"
            echo ""
            echo "Config overrides (passed to trainer as OmegaConf dotlist):"
            echo "  optimization.steps=100"
            echo "  distillation.distillation_alpha=0.5"
            exit 0
            ;;
        *)
            OVERRIDES="$OVERRIDES $1"
            shift
            ;;
    esac
done

######################
### Validate       ###
######################

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

if [[ ! -f "$CONTAINER" ]] && [[ "$DRY_RUN" == "false" ]]; then
    echo "Warning: Container not found: $CONTAINER"
fi

######################
### Setup          ###
######################

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

MOUNTS="/lustre:/lustre,$MODELOPT_PATH:/opt/modelopt"

ACCELERATE_CONFIG="$DISTILL_PATH/configs/accelerate/fsdp_ltx2.yaml"

# Thin wrapper that delegates to src.run (avoids relative-import issues)
TRAINER_SCRIPT="$DISTILL_PATH/train_general.py"
TRAINER_CWD="$DISTILL_PATH"

######################
### Build Command  ###
######################

read -r -d '' CONTAINER_CMD <<'CMDEOF' || true
set -eo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=
export WANDB_API_KEY=

echo "=== Node $(hostname) | SLURM_NODEID=$SLURM_NODEID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

MACHINE_RANK=${SLURM_NODEID:-0}
CMDEOF

# Append accelerate launch command.
# Debug overrides: mock data, short run, frequent checkpoints, skip validation,
# time-limit save at 25 min (gives 5 min buffer in a 30-min job).
CONTAINER_CMD="$CONTAINER_CMD
cd $TRAINER_CWD

accelerate launch \\
    --config_file $ACCELERATE_CONFIG \\
    --num_processes $TOTAL_GPUS \\
    --num_machines $NUM_NODES \\
    --machine_rank \$MACHINE_RANK \\
    --rdzv_backend c10d \\
    --main_process_ip MASTER_IP_PLACEHOLDER \\
    --main_process_port 29500 \\
    $TRAINER_SCRIPT \\
    --config $CONFIG \\
    model.model_path=$TRANSFORMER_PATH \\
    model.text_encoder_path=$GEMMA_PATH \\
    output_dir=$OUTPUT_DIR \\
    distillation.quant_cfg=NVFP4_DEFAULT_CFG \\
    distillation.calibration_size=4 \\
    distillation.calibration_n_steps=4 \\
    distillation.use_mock_data=true \\
    distillation.mock_data_samples=200 \\
    optimization.steps=100 \\
    optimization.batch_size=1 \\
    optimization.gradient_accumulation_steps=1 \\
    optimization.warmup_steps=5 \\
    checkpoints.interval=50 \\
    checkpoints.must_save_by=25 \\
    validation.interval=50 \\
    validation.video_dims='[256,256,9]' \\
    validation.skip_initial_validation=true \\
    wandb.enabled=false \\
    $OVERRIDES
"

######################
### Print Info     ###
######################

echo "========================================"
echo "LTX-2 Distillation Debug (General Trainer)"
echo "========================================"
echo "Job Name:        $JOB_NAME"
echo "Partition:       $PARTITION"
echo "Account:         $ACCOUNT"
echo "Time Limit:      $TIME_LIMIT"
echo "Nodes:           $NUM_NODES"
echo "GPUs/Node:       $GPUS_PER_NODE"
echo "Total GPUs:      $TOTAL_GPUS"
echo "Config:          $CONFIG"
echo "Model:           $TRANSFORMER_PATH"
echo "Gemma:           $GEMMA_PATH"
echo "Output Dir:      $OUTPUT_DIR"
echo "Container:       $CONTAINER"
echo "Overrides:       $OVERRIDES"
echo "========================================"
echo ""

######################
### Build Script   ###
######################

read -r -d '' SBATCH_SCRIPT <<EOF || true
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --time=$TIME_LIMIT
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:$GPUS_PER_NODE
#SBATCH --exclusive
#SBATCH --dependency=singleton

set -eo pipefail

head_node=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address | head -n 1)
echo "Master node: \$head_node (\$head_node_ip)"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"

read -r -d '' CONTAINER_CMD <<'INNEREOF' || true
$CONTAINER_CMD
INNEREOF
CONTAINER_CMD=\${CONTAINER_CMD//MASTER_IP_PLACEHOLDER/\$head_node_ip}

srun --nodes=\$SLURM_NNODES --ntasks-per-node=1 \\
    --output="$LOG_DIR/slurm-%j-%t.out" \\
    --error="$LOG_DIR/slurm-%j-%t.err" \\
    --no-container-remap-root --no-container-mount-home \\
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \\
    bash -c "\$CONTAINER_CMD"
EOF

######################
### Submit Job     ###
######################

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN ==="
    echo ""
    echo "$SBATCH_SCRIPT"
    echo ""
else
    echo "Submitting job..."
    echo "$SBATCH_SCRIPT" | sbatch
    echo ""
    echo "Job submitted! Logs: $LOG_DIR/"
fi
