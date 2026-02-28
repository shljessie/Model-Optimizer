#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Easy-to-use SLURM Launch Script for LTX-2 Distillation Training
#
# This script provides a simple interface to launch multi-node training jobs.
# It handles container setup, environment variables, and SLURM configuration.
#
# Usage:
#   ./slurm/launch.sh [options]
#
# Examples:
#   # Launch with defaults (2 nodes, 8 GPUs each)
#   ./slurm/launch.sh
#
#   # Launch with 4 nodes
#   ./slurm/launch.sh --nodes 4
#
#   # Launch with custom config
#   ./slurm/launch.sh --config configs/my_experiment.yaml --nodes 4
#
#   # Dry run (print command without submitting)
#   ./slurm/launch.sh --dry-run

set -eo pipefail

######################
### Default Config ###
######################

# Job settings
JOB_NAME="ltx2-distillation"
PARTITION="batch_block1,interactive,batch_short"
ACCOUNT="adlr_psx_numerics"
TIME_LIMIT="00:30:00"

# Cluster resources
NUM_NODES=1
GPUS_PER_NODE=8
# CPUS_PER_TASK=64

# Paths (modify these for your environment)
ROOT_PATH="/path/to/workdir"
MODELOPT_PATH="$ROOT_PATH/sources/Model-Optimizer"
DISTILL_PATH="$MODELOPT_PATH/examples/diffusers/distillation"
CONTAINER="$ROOT_PATH/containers/ltx-distillation2.sqsh"

# Model paths
MODEL_PATH="$ROOT_PATH/models/LTX-2"
TRANSFORMER_PATH="$MODEL_PATH/ltx-2-19b-dev.safetensors"
GEMMA_PATH="$ROOT_PATH/models/gemma-3-12b-it-qat-q4_0-unquantized"

# Output paths
OUTPUT_DIR="$ROOT_PATH/results/$JOB_NAME"

# Training config (absolute paths)
CONFIG="$DISTILL_PATH/configs/ltx2_distillation.yaml"
OVERRIDES=""

# Flags
DRY_RUN=false

######################
### Parse Args     ###
######################

while [[ $# -gt 0 ]]; do
    case $1 in
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --container)
            CONTAINER="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options] [config_overrides...]"
            echo ""
            echo "Options:"
            echo "  --job-name NAME     Job name (default: $JOB_NAME)"
            echo "  --partition PART    SLURM partition (default: $PARTITION)"
            echo "  --account ACCT      SLURM account (default: $ACCOUNT)"
            echo "  --time TIME         Time limit HH:MM:SS (default: $TIME_LIMIT)"
            echo "  --nodes N           Number of nodes (default: $NUM_NODES)"
            echo "  --gpus N            GPUs per node (default: $GPUS_PER_NODE)"
            echo "  --config FILE       Training config YAML (default: $CONFIG)"
            echo "  --container FILE    Container image (default: $CONTAINER)"
            echo "  --output-dir DIR    Output directory (default: $OUTPUT_DIR)"
            echo "  --model-path PATH   Model checkpoint path"
            echo "  --dry-run           Print command without submitting"
            echo "  --help              Show this help message"
            echo ""
            echo "Config overrides (passed to trainer):"
            echo "  distillation.distillation_alpha=0.5"
            echo "  optimization.learning_rate=1e-5"
            exit 0
            ;;
        *)
            # Collect as config overrides
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
    echo "  Set CONTAINER env var or use --container option"
fi

######################
### Setup          ###
######################

# Create output and log directories
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Total GPUs
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Container mounts
MOUNTS="/lustre:/lustre,$MODELOPT_PATH:/opt/modelopt"

# Accelerate config (absolute paths)
ACCELERATE_CONFIG="$DISTILL_PATH/configs/accelerate/fsdp_ltx2.yaml"

TRAINER_SCRIPT="$DISTILL_PATH/train_general.py"
TRAINER_CWD="$DISTILL_PATH"

######################
### Build Command  ###
######################

# Full command to run inside the container
# - Use MASTER_IP_PLACEHOLDER for the master node IP (replaced at runtime)
# - Add any setup, exports, or preprocessing here
read -r -d '' CONTAINER_CMD <<'CMDEOF' || true
set -eo pipefail

# ============ Environment Setup ============
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=
export WANDB_API_KEY=

# ============ Debug Info ============
echo "Container started on $(hostname)"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Machine rank for this node (SLURM_NODEID is 0, 1, 2, ... for each node)
MACHINE_RANK=${SLURM_NODEID:-0}
echo "MACHINE_RANK: $MACHINE_RANK"

# ============ Training ============
CMDEOF

# Append the accelerate launch command (these variables expand now)
# Note: MACHINE_RANK is evaluated at runtime inside the container
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
    $OVERRIDES
"

######################
### Print Info     ###
######################

echo "========================================"
echo "LTX-2 Distillation - SLURM Launch"
echo "========================================"
echo "Job Name:        $JOB_NAME"
echo "Partition:       $PARTITION"
echo "Account:         $ACCOUNT"
echo "Time Limit:      $TIME_LIMIT"
echo "Nodes:           $NUM_NODES"
echo "GPUs/Node:       $GPUS_PER_NODE"
echo "Total GPUs:      $TOTAL_GPUS"
echo "Config:          $CONFIG"
echo "Output Dir:      $OUTPUT_DIR"
echo "Container:       $CONTAINER"
echo "Overrides:       $OVERRIDES"
echo "========================================"
echo ""

######################
### Build Script   ###
######################

# Build the final sbatch script (single source of truth)
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
#SBATCH --comment={"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading"}}

set -eo pipefail

# Get master node info
head_node=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address | head -n 1)
echo "Master node: \$head_node (\$head_node_ip)"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"
echo "SLURM_NNODES: \$SLURM_NNODES"

# Container command (with master IP placeholder replaced)
read -r -d '' CONTAINER_CMD <<'INNEREOF' || true
$CONTAINER_CMD
INNEREOF
CONTAINER_CMD=\${CONTAINER_CMD//MASTER_IP_PLACEHOLDER/\$head_node_ip}

# Run on all nodes (--nodes and --ntasks match SLURM allocation)
# Use srun --output to get separate log files per node (%t = task/node id, %j = job id)
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
    echo "=== DRY RUN - Would submit: ==="
    echo ""
    echo "$SBATCH_SCRIPT"
    echo ""
else
    echo "Submitting job..."
    echo "$SBATCH_SCRIPT" | sbatch
    echo ""
    echo "Job submitted! Check logs/ for output."
fi