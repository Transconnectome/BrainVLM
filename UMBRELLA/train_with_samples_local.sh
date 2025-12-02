#!/bin/bash

################################################################################
# UMBRELLA Local Training Script (Single GPU)
#
# Usage:
#   ./train_with_samples_local.sh [--task T1|T2|T3|mixed] [--epochs 3] [--batch-size 8]
#
# Examples:
#   ./train_with_samples_local.sh                    # Default: mixed T1-T3
#   ./train_with_samples_local.sh --task T1           # Q&A only
#   ./train_with_samples_local.sh --task T2 --epochs 10
#   ./train_with_samples_local.sh --batch-size 16
#
# Features:
#   - Single GPU training (no DDP)
#   - Uses umbrella_llava_train.yaml configuration
#   - Supports 3D (sMRI/dMRI) and 4D (fMRI) images
#   - LLaVA-Next format tokenization
#   - Automatic sample data detection
#
# Requirements:
#   - PyTorch with CUDA support
#   - transformers, monai, pyyaml
#   - Sample data in project/dataloaders/sample_data/sex_comparison_conversations/
#
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

# Configuration
CONFIG_FILE="$PROJECT_DIR/project/config/umbrella_llava_train.yaml"
TRAINING_SCRIPT="$PROJECT_DIR/project/training/main_umbrella_training_fixed.py"
SAMPLE_DATA_DIR="$PROJECT_DIR/project/dataloaders/sample_data/sex_comparison_conversations"
OUTPUT_DIR="$PROJECT_DIR/outputs/umbrella_training_local"

# Default parameters
TASK="mixed"
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=0.00005

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

# Print configuration
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}UMBRELLA LOCAL TRAINING CONFIGURATION${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "Script Directory: ${GREEN}$SCRIPT_DIR${NC}"
echo -e "Config File: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Training Script: ${GREEN}$TRAINING_SCRIPT${NC}"
echo -e "Sample Data: ${GREEN}$SAMPLE_DATA_DIR${NC}"
echo -e "Output Directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "${BLUE}---------------${NC}"
echo -e "Task Type: ${GREEN}$TASK${NC}"
echo -e "Epochs: ${GREEN}$EPOCHS${NC}"
echo -e "Batch Size: ${GREEN}$BATCH_SIZE${NC}"
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
if torch.cuda.is_available():
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected. Training will be slow on CPU.")
PYEOF
echo ""

# Run training
echo -e "${BLUE}Starting UMBRELLA Training...${NC}"
echo ""

python3 "$TRAINING_SCRIPT" \
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
    echo -e "${GREEN}✓ TRAINING COMPLETED SUCCESSFULLY${NC}"
    echo -e "Results saved to: ${GREEN}$OUTPUT_DIR${NC}"
else
    echo -e "${RED}✗ TRAINING FAILED (exit code: $TRAINING_EXIT_CODE)${NC}"
    exit $TRAINING_EXIT_CODE
fi
echo -e "${BLUE}================================================================================${NC}"
