# UMBRELLA Training - Quick Start Guide

## Basic Setup

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install omegaconf wandb

# Medical imaging
pip install nibabel monai

# Data handling
pip install pandas numpy scikit-learn
```

### 2. Prepare Configuration

Edit `project/config/umbrella_llava_train.yaml`:

```yaml
dataset:
  T1:
    study_sample: ["ABCD"]
    json_file: ["/path/to/abcd_t1_train.json"]
    data_root: ["/path/to/ABCD/T1/"]
    img_size: 128

  rsfMRI:
    study_sample: ["ABCD"]
    json_file: ["/path/to/abcd_fmri_train.json"]
    data_root: ["/path/to/ABCD/fMRI/"]
    img_size: [96, 96, 96]
    sequence_length: 20

  dMRI:
    study_sample: []  # Optional: set to [] to skip dMRI
    json_file: []
    data_root: []

trainer:
  per_device_batch_size: 2
  num_train_epochs: 50
  learning_rate: 0.00005
```

### 3. Prepare JSON Files

Each dataset requires a JSON file with sample definitions:

**Format**:
```json
[
  {
    "task_id": "age_estimation",
    "subject_id": "sub-0001",
    "modality_paths": {
      "sMRI": "sub-0001/anat/T1w.nii.gz",
      "fMRI": "sub-0001/func/",
      "dMRI": "sub-0001/dwi/dwi.nii.gz"
    },
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nEstimate the age of this brain"
      },
      {
        "from": "gpt",
        "value": "Based on morphometric analysis, this brain appears to be approximately 45 years old"
      }
    ],
    "metadata": {
      "age": 45,
      "sex": 1
    }
  }
]
```

### 4. Run Training

```bash
cd /path/to/UMBRELLA

# Single GPU training
python project/main_umbrella_training.py --config project/config/umbrella_llava_train.yaml

# Multi-GPU training with DDP
torchrun --nproc_per_node=4 project/main_umbrella_training.py --config project/config/umbrella_llava_train.yaml

# With DeepSpeed
deepspeed --num_gpus=4 project/main_umbrella_training.py --config project/config/umbrella_llava_train.yaml
```

---

## Understanding the Modalities

### fMRI (Resting-State Functional MRI)

**Data Format**:
```
subject_dir/
├── frame_0.pt    # Temporal frame 0
├── frame_1.pt    # Temporal frame 1
├── ...
├── frame_N.pt    # Temporal frame N
└── global_stats.pt  # Normalization statistics
```

**Processing**:
- Loads N temporal frames as a 4D sequence
- Applies dataset-specific padding to 96×96×96
- Applies z-score normalization with global statistics
- Temporal downsampling based on TR
- Output shape: (1, 96, 96, 96, 20) for 20-frame sequences

**Supported Datasets**:
- ABCD: Padding (0,1,0,0,0,0), downsample 2.5×, TR=0.8s
- UKB: Padding (3,9,0,0,10,8), downsample 1.0×, TR=0.735s
- HCP: Padding (3,9,0,0,10,8), downsample 1.0×, TR=0.72s
- HBN: Special rest/task handling
- ABIDE: Negative padding for cropping

### T1 (T1-Weighted sMRI)

**Data Format**:
```
T1w.nii.gz    # NIfTI format 3D volume
```

**Processing**:
- Loads NIfTI file (any size)
- Resizes to 128×128×128
- Applies MONAI intensity normalization (z-score)
- Training: Random axis flips for augmentation
- Eval: No augmentation
- Output shape: (1, 128, 128, 128)

### dMRI (Diffusion MRI)

**Data Format**:
```
dwi.nii.gz    # NIfTI format 3D volume
```

**Processing**:
- Same as T1
- Resizes to 128×128×128
- Applies MONAI intensity normalization
- Output shape: (1, 128, 128, 128)

---

## Training Modes

### Single Modality Training

To train on only one modality, set others to empty:

```yaml
dataset:
  T1:
    study_sample: ["ABCD"]
    json_file: ["/path/to/t1.json"]
    data_root: ["/path/to/T1/"]

  rsfMRI:
    study_sample: []  # Skip fMRI
    json_file: []
    data_root: []

  dMRI:
    study_sample: []  # Skip dMRI
    json_file: []
    data_root: []
```

### Multi-Modality Training

Include all modalities:

```yaml
dataset:
  T1:
    study_sample: ["ABCD"]
    json_file: ["/path/to/t1.json"]
    data_root: ["/path/to/T1/"]

  rsfMRI:
    study_sample: ["ABCD"]
    json_file: ["/path/to/fmri.json"]
    data_root: ["/path/to/fMRI/"]

  dMRI:
    study_sample: ["ABCD"]
    json_file: ["/path/to/dmri.json"]
    data_root: ["/path/to/dMRI/"]
```

### Multiple Dataset Training

Combine datasets from different studies:

```yaml
dataset:
  T1:
    study_sample: ["ABCD", "UKB", "HCP"]
    json_file: [
      "/path/to/abcd_t1.json",
      "/path/to/ukb_t1.json",
      "/path/to/hcp_t1.json"
    ]
    data_root: [
      "/data/ABCD/T1/",
      "/data/UKB/T1/",
      "/data/HCP/T1/"
    ]
```

---

## Monitoring Training

### WandB Dashboard

1. Set API key in config:
```yaml
wandb:
  API_KEY: "YOUR_WANDB_API_KEY"
```

2. Training metrics logged:
   - Training loss per epoch
   - Validation loss and metrics
   - Gradient norms
   - Generation samples (every 50 steps)
   - Learning rate schedule

3. Access at: https://wandb.ai/your-username/UMBRELLA

### Local Logs

```bash
# View training logs
tail -f hf_logs/*/training.log

# View generation samples
cat generation_logs.json | jq .
```

---

## Model Architecture Details

### Patch Embedding Module

The `PatchEmbed` class converts images to patch tokens:

**T1 (128×128×128 with 10×10×10 patches)**:
- Number of patches: (128/10)³ = 12³ = 1,728 patches
- Patch dimension: 10×10×10×1 = 1,000 pixels
- Projection: 1,000 → 768 (embedding dimension)

**fMRI (96×96×96×20 with 16×16×16×3 patches)**:
- Number of patches: (96/16)³ × (20/3) = 6³ × 6.67 ≈ 1,440 patches
- Patch dimension: 16×16×16×3 = 12,288 voxels
- Projection: 12,288 → 768

**dMRI (128×128×128 with 10×10×10 patches)**:
- Same as T1: 1,728 patches

### Vision Tower Freezing

Only the custom patch embeddings are trainable:
- Vision encoder (CLIP-ViT layers): **FROZEN**
- Pre/post layer norms: **FROZEN**
- Patch embeddings: **TRAINABLE** (modality-specific)
- Multi-modal projector: **FROZEN**
- Language model: **FROZEN**

### Loss Computation

```python
# For single modality batch
if len(modalities) == 1:
    dummy_loss = 0 * sum(param)  # for unused modalities
    actual_loss = cross_entropy(logits, labels)
    total_loss = dummy_loss + actual_loss

# For multi-modality batch
else:
    total_loss = cross_entropy(concat_logits, concat_labels)
```

---

## Common Configurations

### Lightweight (GPU with 8GB VRAM)

```yaml
trainer:
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
  max_epochs: 20

dataset:
  rsfMRI:
    sequence_length: 10  # Reduced from 20
```

### Standard (GPU with 24GB VRAM)

```yaml
trainer:
  per_device_batch_size: 2
  gradient_accumulation_steps: 2
  max_epochs: 50

dataset:
  rsfMRI:
    sequence_length: 20
```

### High Performance (A100 with 40GB VRAM)

```yaml
trainer:
  per_device_batch_size: 8
  gradient_accumulation_steps: 1
  max_epochs: 50

dataset:
  rsfMRI:
    sequence_length: 30
```

---

## Troubleshooting

### Error: "No datasets specified"

**Cause**: All modalities set to empty

**Solution**: Configure at least one modality in the config file

### Error: "Frame not found"

**Cause**: fMRI frames missing or misnamed

**Solution**:
- Check frames are named `frame_0.pt`, `frame_1.pt`, etc.
- Verify frame directory path matches `modality_paths.fMRI` in JSON

### Error: "Stats file not found"

**Cause**: Missing `global_stats.pt` for fMRI normalization

**Solution**:
- Generate stats during preprocessing
- Or set `input_scaling_method: "none"` (not recommended)

### CUDA Out of Memory

**Solution**:
1. Reduce `per_device_batch_size`
2. Reduce `sequence_length` for fMRI
3. Reduce `img_size` for T1/dMRI
4. Enable gradient checkpointing

### Training is slow

**Checklist**:
- GPU utilization > 50%? Check with `nvidia-smi`
- `num_workers` > 0 in DataLoader?
- Sufficient I/O bandwidth (SSD preferred)
- Consider using mixed precision training

---

## Next Steps

1. **Prepare Data**: Create JSON files and organize data
2. **Configure**: Edit `umbrella_llava_train.yaml`
3. **Test**: Run on small subset first
4. **Scale**: Increase dataset size and adjust hyperparameters
5. **Evaluate**: Monitor WandB dashboard and evaluate on test set

---

**Version**: 1.0
**Last Updated**: November 20, 2025
**Maintainer**: BrainVLM Team
