#!/bin/bash
# UMBRELLA: LLaVA-based Brain MRI Training Script
# DeepSpeed Stage 2 + CPU Offload Mode
# Optimized for 40GB GPUs

set +x

cd /pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini

module load python
module load cpe/23.03
module load cudatoolkit/12.9

conda activate /pscratch/sd/h/heehaw/anaconda/BrainVLM_llava

export LIBRARY_PATH=$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/pscratch/sd/h/heehaw
export HF_HOME=/pscratch/sd/h/heehaw/huggingface
export TORCH_HOME=/pscratch/sd/h/heehaw/
export DS_SKIP_CUDA_CHECK=1

# DeepSpeed Launcher
# --num_gpus: Number of GPUs to use
deepspeed --num_gpus 1 project/training/main_umbrella_training.py \
      --deepspeed project/config/deepspeed_config.json \
      --config /pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini/project/config/umbrella_llava_train_BMIRegression_7B.yaml \
      --train-data ./sample_data/BMI_regression_conversations_10000subjects/train_conversations.jsonl \
      --eval-data ./sample_data/BMI_regression_conversations_10000subjects/validation_conversations.jsonl \
      --modality sMRI \
      --output-dir ./hf_results/umbrella_BMIRegression_DeepSpeed \
      --eval-output-dir /pscratch/sd/h/heehaw/BrainVLM/UMBRELLA_gemini/eval_predictions_BMIRegression_DeepSpeed
