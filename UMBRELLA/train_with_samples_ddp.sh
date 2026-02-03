#!/bin/bash

################################################################################
# UMBRELLA Multi-GPU DDP Training Script
#
# Distributed Data Parallel (DDP) training across multiple GPUs.
#
# Usage:
#   ./train_with_samples_ddp.sh [--gpus 0,1,2,3] [--task mixed] [--epochs 3]
#
# Examples:
#   ./train_with_samples_ddp.sh                      # Use all available GPUs
#   ./train_with_samples_ddp.sh --gpus 0,1           # Use GPUs 0 and 1
#   ./train_with_samples_ddp.sh --gpus 0,1,2,3 --epochs 10
#   ./train_with_samples_ddp.sh --gpus 0,1 --task T1 --batch-size 16
#
# Features:
#   - Multi-GPU training with DDP
#   - Uses umbrella_llava_train.yaml configuration
#   - Supports 3D (sMRI/dMRI) and 4D (fMRI) images
#   - LLaVA-Next format tokenization
#   - Automatic GPU detection and assignment
#   - Synchronized batch normalization across GPUs
#
# Requirements:
#   - PyTorch with CUDA support (distributed package)
#   - transformers, monai, pyyaml
#   - Multiple GPUs available
#   - Sample data in project/dataloaders/sample_data/sex_comparison_conversations/
#
# Performance Tips:
#   - Increase batch size for multi-GPU training (divided by number of GPUs)
#   - Use gradient accumulation for larger effective batch sizes
#   - Enable mixed precision (fp16/bf16) for faster training and less memory
#
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

# Configuration
CONFIG_FILE="$PROJECT_DIR/project/config/umbrella_llava_train.yaml"
TRAINING_SCRIPT="$PROJECT_DIR/project/training/main_umbrella_training_fixed.py"
SAMPLE_DATA_DIR="$PROJECT_DIR/project/dataloaders/sample_data/sex_comparison_conversations"
OUTPUT_DIR="$PROJECT_DIR/outputs/umbrella_training_ddp"

# Default parameters
TASK="mixed"
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=0.00005
GPUS=""
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            grep '^#' "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Auto-detect GPUs if not specified
if [ -z "$GPUS" ]; then
    NUM_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo -e "${RED}ERROR: No GPUs detected. Cannot run DDP training on CPU.${NC}" >&2
        exit 1
    fi
    GPUS=$(seq 0 $((NUM_GPUS - 1)) | paste -sd ',' -)
    echo -e "${GREEN}Auto-detected $NUM_GPUS GPU(s): $GPUS${NC}"
fi

# Convert GPUS string to array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC_PER_NODE=${#GPU_ARRAY[@]}

# Print configuration
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}UMBRELLA DDP TRAINING CONFIGURATION${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "Script Directory: ${GREEN}$SCRIPT_DIR${NC}"
echo -e "Config File: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Training Script: ${GREEN}$TRAINING_SCRIPT${NC}"
echo -e "Sample Data: ${GREEN}$SAMPLE_DATA_DIR${NC}"
echo -e "Output Directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "${BLUE}---------------${NC}"
echo -e "GPUs: ${CYAN}$GPUS${NC} (${NPROC_PER_NODE} GPU(s))"
echo -e "Number of Nodes: ${GREEN}$NUM_NODES${NC}"
echo -e "Master Address: ${GREEN}$MASTER_ADDR${NC}"
echo -e "Master Port: ${GREEN}$MASTER_PORT${NC}"
echo -e "${BLUE}---------------${NC}"
echo -e "Task Type: ${GREEN}$TASK${NC}"
echo -e "Epochs: ${GREEN}$EPOCHS${NC}"
echo -e "Batch Size (per GPU): ${GREEN}$BATCH_SIZE${NC}"
echo -e "Effective Batch Size: ${GREEN}$((BATCH_SIZE * NPROC_PER_NODE))${NC}"
echo -e "Learning Rate: ${GREEN}$LEARNING_RATE${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}" >&2
    exit 1
fi

# Check if training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo -e "${RED}ERROR: Training script not found: $TRAINING_SCRIPT${NC}" >&2
    exit 1
fi

# Check if sample data exists
if [ ! -d "$SAMPLE_DATA_DIR" ]; then
    echo -e "${YELLOW}WARNING: Sample data directory not found: $SAMPLE_DATA_DIR${NC}"
    echo -e "${YELLOW}Please ensure sample JSON files are in the correct location${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"
echo ""

# GPU check
echo -e "${BLUE}GPU Availability Check:${NC}"
python3 << 'PYEOF'
import torch
print(f"Total GPU(s) available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
PYEOF
echo ""

# Set environment variables for DDP
export CUDA_VISIBLE_DEVICES="$GPUS"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export NPROC_PER_NODE="$NPROC_PER_NODE"
export NUM_NODES="$NUM_NODES"
export NODE_RANK="$NODE_RANK"

echo -e "${BLUE}Starting UMBRELLA DDP Training...${NC}"
echo -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# Run DDP training using torchrun
# torchrun handles process launching and environment setup
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$TRAINING_SCRIPT" \
    --task "$TASK" \
    --config "$CONFIG_FILE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$SAMPLE_DATA_DIR"

TRAINING_EXIT_CODE=$?

# Print summary
echo ""
echo -e "${BLUE}================================================================================${NC}"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ DDP TRAINING COMPLETED SUCCESSFULLY${NC}"
    echo -e "Results saved to: ${GREEN}$OUTPUT_DIR${NC}"
    echo -e "Total processes: ${CYAN}$(($NPROC_PER_NODE * $NUM_NODES))${NC}"
else
    echo -e "${RED}✗ DDP TRAINING FAILED (exit code: $TRAINING_EXIT_CODE)${NC}"
    exit $TRAINING_EXIT_CODE
fi
echo -e "${BLUE}================================================================================${NC}"
