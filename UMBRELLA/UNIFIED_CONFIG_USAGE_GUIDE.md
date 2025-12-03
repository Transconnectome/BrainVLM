# UMBRELLA Unified Configuration - Usage Guide

Quick reference for using the consolidated configuration system.

---

## Overview

**Single Source of Truth**: `UMBRELLATrainingConfig`
**HuggingFace Compatible**: Converts to `UMBRELLATrainingArgs` via factory method
**Zero AttributeErrors**: All required attributes properly propagated

---

## Basic Usage

### 1. Load Configuration from YAML

```python
from training.main_umbrella_training_fixed import UMBRELLATrainingConfig

# Load from YAML
config = UMBRELLATrainingConfig.from_yaml("umbrella_llava_train.yaml")

# Override specific settings
config.batch_size = 4
config.learning_rate = 1e-4
config.task_filter = "same_sex_comparison"
```

### 2. Create Programmatically

```python
config = UMBRELLATrainingConfig(
    model_name="llava-hf/llava-interleave-qwen-0.5b-hf",
    train_json_path="data/train.json",
    eval_json_path="data/eval.json",
    modality="T1",
    batch_size=2,
    learning_rate=5e-5,
    num_epochs=50,
    enable_task_aware_loss=True,
    mask_human_turns=True,
    use_wandb=True
)
```

### 3. Convert to Training Arguments

```python
# Create HuggingFace-compatible training arguments
training_args = config.to_training_args(
    eval_dataset_available=True  # Set based on whether you have eval data
)

# Now use with UMBRELLATrainer
trainer = UMBRELLATrainer(
    model=model,
    args=training_args,  # Correct type, all attributes present
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    tokenizer=tokenizer
)
```

---

## Configuration Attributes

### Model Settings
```python
config.model_name = "llava-hf/llava-interleave-qwen-0.5b-hf"
config.tokenizer_name = None  # Uses model_name if None
```

### Data Settings
```python
config.train_json_path = "data/train.json"  # File or directory
config.eval_json_path = "data/eval.json"    # Optional
config.task_filter = "same_sex_comparison"  # Optional task filtering
```

### Modality Settings
```python
config.modality = "T1"  # or "rsfMRI"
config.img_size = [96, 96, 96]  # 3D volume
config.patch_size = [10, 10, 10]

# Multi-modality support
config.T1_img_size = [96, 96, 96]
config.T1_patch_size = [10, 10, 10]
config.rsfMRI_img_size = [96, 96, 96, 24]  # 4D volume
config.rsfMRI_patch_size = [16, 16, 16, 3]
```

### Training Hyperparameters
```python
config.batch_size = 2
config.gradient_accumulation_steps = 1
config.learning_rate = 5e-5
config.num_epochs = 50
config.max_seq_length = 2048
config.max_images_per_sample = 10
config.warmup_steps = 500
```

### Memory and Performance
```python
config.enable_memory_aware_batching = True
config.memory_budget_gb = 30.0
config.gradient_checkpointing = True
config.mixed_precision = "bf16"  # or "fp16"
```

### Multi-turn Masking
```python
config.mask_human_turns = True      # Mask user turns in loss
config.mask_padding_tokens = True   # Mask padding tokens
```

### Task-Aware Loss
```python
config.enable_task_aware_loss = True
config.task_type_weights = {
    "same_sex_comparison": 1.0,
    "different_sex_comparison": 1.2
}
```

### Dummy Loss Support
```python
config.enable_dummy_loss = True
config.dummy_loss_weight = 0.1
```

### Logging Settings
```python
config.logging_steps = 1
config.save_steps = 500
config.eval_steps = 500
config.save_total_limit = 3

# Advanced logging
config.log_turn_distribution = True
config.log_image_statistics = True
config.log_memory_usage = False
```

### Gradient Normalization
```python
config.normalize_gradients_by_batch_size = True
config.base_batch_size = 32
```

### Weights & Biases
```python
config.use_wandb = True
config.wandb_project = "umbrella-training"
config.wandb_api_key = "your-api-key"  # Or set in YAML
```

---

## Complete Example

### Python Script

```python
#!/usr/bin/env python3
"""UMBRELLA training with unified config."""

import logging
from pathlib import Path
from training.main_umbrella_training_fixed import (
    UMBRELLATrainingConfig,
    UMBRELLATrainingPipeline
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create configuration
config = UMBRELLATrainingConfig.from_yaml("config/umbrella_llava_train.yaml")

# Override settings
config.train_json_path = "data/sex_comparison/train"
config.eval_json_path = "data/sex_comparison/eval"
config.task_filter = "same_sex_comparison"
config.batch_size = 4
config.num_epochs = 30
config.use_wandb = True

# Verify paths
assert Path(config.train_json_path).exists()
assert Path(config.eval_json_path).exists()

# Create and run pipeline
pipeline = UMBRELLATrainingPipeline(config)
pipeline.train()
```

### Command Line

```bash
python main_umbrella_training_fixed.py \
    --config config/umbrella_llava_train.yaml \
    --train-data data/sex_comparison/train \
    --eval-data data/sex_comparison/eval \
    --modality T1 \
    --task-filter same_sex_comparison \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --output-dir results/sex_comparison
```

---

## Factory Method Details

### to_training_args() Signature

```python
def to_training_args(self, eval_dataset_available: bool = False) -> UMBRELLATrainingArgs:
    """
    Convert config to UMBRELLATrainingArgs for HuggingFace Trainer.

    Args:
        eval_dataset_available: Whether evaluation dataset is available.
                               Affects evaluation_strategy and related settings.

    Returns:
        UMBRELLATrainingArgs instance with all attributes properly set.
    """
```

### Attribute Mapping

**Standard HuggingFace TrainingArguments**:
- output_dir, num_train_epochs, per_device_train_batch_size
- gradient_accumulation_steps, learning_rate, warmup_steps
- logging_steps, eval_steps, save_steps, save_total_limit
- fp16, bf16, save_strategy, evaluation_strategy
- report_to, load_best_model_at_end

**UMBRELLA-Specific Extensions**:
- mask_human_turns, mask_padding_tokens
- enable_task_aware_loss, task_type_weights
- enable_memory_aware_batching, memory_budget_gb
- enable_dummy_loss, dummy_loss_weight
- log_turn_distribution, log_image_statistics, log_memory_usage
- normalize_gradients_by_batch_size, base_batch_size

---

## Verification

### Check Attribute Presence

```python
config = UMBRELLATrainingConfig.from_yaml("config.yaml")
training_args = config.to_training_args()

# Verify type
assert isinstance(training_args, UMBRELLATrainingArgs)
print(f"Type: {type(training_args).__name__}")

# Verify attributes
assert hasattr(training_args, 'task_type_weights')
assert hasattr(training_args, 'enable_task_aware_loss')
assert hasattr(training_args, 'mask_human_turns')
print("All attributes present ✓")

# Check values
print(f"Task-aware loss: {training_args.enable_task_aware_loss}")
print(f"Mask human turns: {training_args.mask_human_turns}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
```

### Debug Logging

The training pipeline automatically logs attribute verification:

```
==================================================================================
CREATING TRAINING ARGUMENTS (UNIFIED CONFIG)
==================================================================================
  Training arguments created successfully
  Type: UMBRELLATrainingArgs
  Has task_type_weights: True
  Has enable_task_aware_loss: True
  Has mask_human_turns: True
==================================================================================
```

---

## Common Patterns

### Pattern 1: Task-Specific Training

```python
# Sex comparison training
config = UMBRELLATrainingConfig.from_yaml("config.yaml")
config.task_filter = "same_sex_comparison"
config.task_type_weights = {"same_sex_comparison": 1.0}
config.output_dir = "results/same_sex"

pipeline = UMBRELLATrainingPipeline(config)
pipeline.train()
```

### Pattern 2: Memory-Constrained Training

```python
config = UMBRELLATrainingConfig.from_yaml("config.yaml")
config.batch_size = 1
config.gradient_accumulation_steps = 8
config.enable_memory_aware_batching = True
config.memory_budget_gb = 15.0
config.gradient_checkpointing = True
config.mixed_precision = "bf16"

pipeline = UMBRELLATrainingPipeline(config)
pipeline.train()
```

### Pattern 3: Quick Experimentation

```python
config = UMBRELLATrainingConfig(
    model_name="llava-hf/llava-interleave-qwen-0.5b-hf",
    train_json_path="data/small_train.json",
    batch_size=4,
    num_epochs=5,
    use_wandb=False,
    save_steps=100,
    logging_steps=10
)

pipeline = UMBRELLATrainingPipeline(config)
pipeline.train()
```

### Pattern 4: Production Training

```python
config = UMBRELLATrainingConfig.from_yaml("config/production.yaml")
config.use_wandb = True
config.wandb_project = "umbrella-production"
config.save_total_limit = 5
config.eval_steps = 100
config.save_steps = 500
config.enable_task_aware_loss = True
config.log_turn_distribution = True
config.log_image_statistics = True

pipeline = UMBRELLATrainingPipeline(config)
pipeline.train()
```

---

## Troubleshooting

### AttributeError: 'TrainingArguments' object has no attribute 'X'

**Cause**: Using vanilla TrainingArguments instead of UMBRELLATrainingArgs

**Solution**: Use factory method
```python
# Wrong
training_args = TrainingArguments(...)

# Correct
training_args = config.to_training_args()
```

### Missing Configuration Attributes

**Cause**: YAML file missing new attributes

**Solution**: Attributes have defaults, no action needed
```python
# All new attributes have defaults in dataclass
config = UMBRELLATrainingConfig.from_yaml("old_config.yaml")
# Works fine - uses defaults for missing attributes
```

### Type Mismatch Errors

**Cause**: Passing config directly to trainer

**Solution**: Convert to training args
```python
# Wrong
trainer = UMBRELLATrainer(args=config)

# Correct
training_args = config.to_training_args()
trainer = UMBRELLATrainer(args=training_args)
```

---

## Migration from Old Code

### Before (Broken)

```python
# Created vanilla TrainingArguments
training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    # ... 20 more lines of manual mapping
)

# Missing UMBRELLA-specific attributes
trainer = UMBRELLATrainer(args=training_args)  # AttributeError!
```

### After (Fixed)

```python
# One line conversion
training_args = config.to_training_args(
    eval_dataset_available=(eval_dataset is not None)
)

# All attributes present
trainer = UMBRELLATrainer(args=training_args)  # Works!
```

---

## Best Practices

### 1. Load from YAML for Production
```python
# Recommended
config = UMBRELLATrainingConfig.from_yaml("config.yaml")
```

### 2. Override Only What You Need
```python
config = UMBRELLATrainingConfig.from_yaml("config.yaml")
config.batch_size = 4  # Override specific setting
```

### 3. Use Factory Method
```python
# Always convert via factory method
training_args = config.to_training_args()
# Never create UMBRELLATrainingArgs manually
```

### 4. Verify Configuration
```python
# Log configuration before training
logger.info(f"Batch size: {config.batch_size}")
logger.info(f"Task-aware loss: {config.enable_task_aware_loss}")
logger.info(f"Mask human turns: {config.mask_human_turns}")
```

### 5. Keep YAML Organized
```yaml
# umbrella_llava_train.yaml
dataset:
  T1:
    img_size: [96, 96, 96]
  rsfMRI:
    img_size: [96, 96, 96, 24]

model:
  hf_name: "llava-hf/llava-interleave-qwen-0.5b-hf"
  T1:
    patch_size: [10, 10, 10]
  rsfMRI:
    patch_size: [16, 16, 16, 3]

trainer:
  per_device_batch_size: 2
  learning_rate: 5e-5
  max_epochs: 50
```

---

## Summary

**Key Points**:
1. Use `UMBRELLATrainingConfig` as single source of truth
2. Convert to `UMBRELLATrainingArgs` via `to_training_args()`
3. All UMBRELLA-specific attributes now accessible
4. No more AttributeError on task_type_weights
5. Backward compatible with existing code

**Quick Start**:
```python
config = UMBRELLATrainingConfig.from_yaml("config.yaml")
training_args = config.to_training_args()
trainer = UMBRELLATrainer(args=training_args, ...)
trainer.train()
```

**References**:
- Implementation: `main_umbrella_training_fixed.py`
- Analysis: `CONFIG_CONSOLIDATION_ANALYSIS.md`
- Completion Report: `CONFIG_CONSOLIDATION_COMPLETION_REPORT.md`
