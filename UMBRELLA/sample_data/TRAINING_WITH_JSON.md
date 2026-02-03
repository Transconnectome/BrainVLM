# Training with JSON  Format

## Overview

This guide explains how to use the  JSON format and dataloaders for training LLaVA-style vision-language models on brain imaging data.

---

## Quick Start

### 1. Generate Training Data

```bash
# Generate  format JSON files
cd sample_data
python3 generate_sex_comparison_conversations_.py
```

This creates:
```
sex_comparison_conversations_/
├── train/           # 200 training examples
├── validation/      # 200 validation examples
├── test/            # 200 test examples
└── samples/         # 10 sample examples
```

### 2. Validate Format

```bash
# Validate all JSON files
python3 validate_json_format_.py
```

Expected output:
```
✓ VALIDATION PASSED
All files conform to  format specification
```

### 3. Load Data

```python
from project.dataloaders_ import UMBRELLADataLoader

# Create dataset
dataset = UMBRELLADataLoader(
    json_dir="sample_data/sex_comparison_conversations_",
    split="train",
    tokenizer=tokenizer,
    processor=processor,
    image_size=224,
    max_length=2048
)

# Create dataloader
dataloader = dataset.create_dataloader(
    batch_size=4,
    shuffle=True,
    num_workers=2
)
```

### 4. Train Model

```python
# Training loop
for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

---

## Detailed Usage

### Dataset Initialization

```python
from project.dataloaders_ import UMBRELLADataLoader

dataset = UMBRELLADataLoader(
    json_dir="path/to/conversations_",      # Root directory with train/val/test
    image_root="/path/to/images",              # Optional: root for relative paths
    split="train",                             # "train", "validation", or "test"
    tokenizer=tokenizer,                       # LLaVA tokenizer
    processor=processor,                       # LLaVA image processor
    image_size=224,                            # Target image size
    normalize=True,                            # Min-max normalization [0,1]
    standardize=True,                          # Z-score standardization
    max_length=2048,                           # Max sequence length
    add_generation_prompt=False                # Add empty assistant prompt
)
```

### Dataloader Creation

```python
# Single dataloader
train_loader = dataset.create_dataloader(
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

# Or create all splits at once
from project.dataloaders_ import create_umbrella_dataloaders

dataloaders = create_umbrella_dataloaders(
    json_dir="path/to/conversations_",
    tokenizer=tokenizer,
    processor=processor,
    batch_size=4,
    image_size=224,
    max_length=2048,
    num_workers=2
)

train_loader = dataloaders["train"]
val_loader = dataloaders["validation"]
test_loader = dataloaders["test"]
```

### Batch Structure

Each batch contains:

```python
{
    "input_ids": torch.Tensor,        # [batch_size, max_length]
    "attention_mask": torch.Tensor,   # [batch_size, max_length]
    "pixel_values": torch.Tensor,     # [batch_size, num_images, 3, H, W]
    "labels": torch.Tensor,           # [batch_size, max_length]
    "metadata": List[Dict],           # Original metadata for each example
    "task_ids": List[str]             # Task IDs for each example
}
```

---

## Training Examples

### Example 1: Basic Training Loop

```python
import torch
from torch.optim import AdamW
from project.dataloaders_ import create_umbrella_dataloaders

# Initialize model (example)
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Create dataloaders
dataloaders = create_umbrella_dataloaders(
    json_dir="sample_data/sex_comparison_conversations_",
    tokenizer=processor.tokenizer,
    processor=processor,
    batch_size=2,
    max_length=2048
)

# Setup training
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
model.train()
for epoch in range(3):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloaders["train"]):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device) if batch["pixel_values"] is not None else None
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloaders["train"])
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
```

### Example 2: HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments
from project.dataloaders_ import UMBRELLADataLoader

# Initialize datasets
train_dataset = UMBRELLADataLoader(
    json_dir="sample_data/sex_comparison_conversations_",
    split="train",
    tokenizer=processor.tokenizer,
    processor=processor
)

val_dataset = UMBRELLADataLoader(
    json_dir="sample_data/sex_comparison_conversations_",
    split="validation",
    tokenizer=processor.tokenizer,
    processor=processor
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./brainvlm_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="tensorboard"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=UMBRELLADataLoader.collate_fn
)

# Train
trainer.train()
```

### Example 3: Multi-GPU Training with DeepSpeed

```python
from transformers import Trainer, TrainingArguments
import deepspeed

# DeepSpeed config
ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2
    }
}

# Training arguments with DeepSpeed
training_args = TrainingArguments(
    output_dir="./brainvlm_deepspeed",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    deepspeed=ds_config,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=UMBRELLADataLoader.collate_fn
)

# Train with DeepSpeed
trainer.train()
```

---

## Configuration Parameters

### Image Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 224 | Target size for vision encoder |
| `normalize` | True | Min-max normalization to [0,1] |
| `standardize` | True | Z-score standardization |

### Tokenization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_length` | 2048 | Maximum sequence length |
| `add_generation_prompt` | False | Add empty assistant prompt for generation |

### DataLoader

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Number of examples per batch |
| `shuffle` | True (train) | Shuffle training data |
| `num_workers` | 0 | Number of data loading workers |
| `pin_memory` | True | Pin memory for faster GPU transfer |
| `drop_last` | True (train) | Drop incomplete last batch |

---

## Memory Optimization

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Batch Size Reduction

```python
# Effective batch size = per_device_batch_size * gradient_accumulation_steps
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    ...
)
```

---

## Monitoring and Logging

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/brainvlm")

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Training step
        loss = train_step(batch)

        # Log to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Loss/train", loss, global_step)

    # Log validation
    val_loss = validate(val_loader)
    writer.add_scalar("Loss/validation", val_loss, epoch)

writer.close()
```

### Weights & Biases

```python
import wandb

wandb.init(project="brainvlm", config={
    "learning_rate": 2e-5,
    "batch_size": 4,
    "epochs": 3
})

# Log during training
wandb.log({"loss": loss, "epoch": epoch})
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Enable mixed precision training
5. Reduce max_length

```python
# Example: Memory-efficient config
dataset = UMBRELLADataLoader(
    ...,
    max_length=1024,      # Reduced from 2048
)

dataloader = dataset.create_dataloader(
    batch_size=1,         # Reduced batch size
    ...
)

model.gradient_checkpointing_enable()
```

### Slow Data Loading

**Solutions:**
1. Increase num_workers
2. Use pin_memory=True
3. Preprocess images offline
4. Use faster storage (SSD)

```python
dataloader = dataset.create_dataloader(
    batch_size=4,
    num_workers=4,        # Parallel loading
    pin_memory=True,      # Faster GPU transfer
    ...
)
```

### Image Loading Errors

**Check:**
1. Image paths are correct
2. .nii.gz files exist and are valid
3. nibabel is installed: `pip install nibabel`
4. Sufficient disk space

```python
# Verify image loading
from project.dataloaders_ import load_image

try:
    image = load_image("path/to/image.nii.gz", modality="sMRI")
    print(f"Image loaded: shape={image.shape}")
except Exception as e:
    print(f"Error loading image: {e}")
```

### Tokenization Issues

**Check:**
1. Tokenizer is compatible with model
2. Processor is initialized correctly
3. max_length is appropriate

```python
# Test tokenization
from project.dataloaders_ import format_conversation

conversations = [...]  # Your conversations
prompt = format_conversation(conversations)
print(f"Prompt: {prompt}")
print(f"Length: {len(prompt)} characters")

# Tokenize
tokens = processor.tokenizer(prompt, return_tensors="pt")
print(f"Token IDs shape: {tokens['input_ids'].shape}")
```

---

## Best Practices

### 1. Data Validation

Always validate your JSON files before training:
```bash
python3 validate_json_format_.py
```

### 2. Start with Small Batch

Begin with a small batch to ensure everything works:
```python
dataset = UMBRELLADataLoader(...)
small_loader = dataset.create_dataloader(batch_size=1, shuffle=False)

# Test one batch
batch = next(iter(small_loader))
outputs = model(**batch)
print(f"Loss: {outputs.loss}")
```

### 3. Monitor GPU Usage

```python
import torch

print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 4. Save Checkpoints

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f"checkpoint_epoch_{epoch}.pt")

# Load checkpoint
checkpoint = torch.load("checkpoint_epoch_2.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 5. Use Validation Set

Always evaluate on validation set:
```python
model.eval()
with torch.no_grad():
    val_loss = 0
    for batch in val_loader:
        outputs = model(**batch)
        val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
```

---

## Example Training Commands

### Local Training (1 GPU)

```bash
python train.py \
    --json_dir sample_data/sex_comparison_conversations_ \
    --output_dir ./output \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --max_length 2048
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 train.py \
    --json_dir sample_data/sex_comparison_conversations_ \
    --output_dir ./output \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 2e-5
```

### DeepSpeed Training

```bash
deepspeed --num_gpus=4 train.py \
    --deepspeed ds_config.json \
    --json_dir sample_data/sex_comparison_conversations_ \
    --output_dir ./output \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --batch_size 2 \
    --epochs 5
```

---

## Next Steps

1. **Data Generation:** Generate your JSON conversations
2. **Validation:** Validate format compliance
3. **Testing:** Test dataloader with small batch
4. **Training:** Run full training with monitoring
5. **Evaluation:** Evaluate on test set
6. **Deployment:** Save and deploy trained model

---

## Resources

- **JSON Format:** `JSON_FORMAT__SPECIFICATION.md`
- **Validation Script:** `validate_json_format_.py`
- **Example Generator:** `generate_sex_comparison_conversations_.py`
- **Sample Data:** `sex_comparison_conversations_/samples/`
- **Dataloaders:** `project/dataloaders_/`

---

**Version:** 2.0
**Date:** 2025-11-25
**Author:** BrainVLM Team
