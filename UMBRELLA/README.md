# UMBRELLA: Production-Ready Training System for Brain MRI Vision-Language Models

## ✅ Status: Production Ready

**Latest Update**: December 1, 2025
**Code Status**: Clean (experimental code removed)
**Tokenization**: Validated (4/4 tests passing)
**Training Script**: `project/training/main_umbrella_training_fixed.py`

---

## Quick Start

### 1. Run Tokenization Validation
```bash
python project/tests/validate_tokenization.py --verbose
```
**Expected**: 4/4 tests passing ✅

### 2. Train with Production Script
```bash
python project/training/main_umbrella_training_fixed.py \
    --config project/config/umbrella_llava_train.yaml \
    --train-json ./data/train.json \
    --modality T1
```

---

## What's New in This Cleanup

### ✅ Code Removed
- `project/main_umbrella_training.py` - Outdated experimental script
- `project/utils/umbrella_trainer.py` - Unused experimental trainer
- `project/utils/dynamic_trainer.py` - Broken dynamic trainer
- `project/utils/training_example.py` - Sample code

**Why**: These were experimental implementations with broken imports. The production training is now handled entirely by `project/training/main_umbrella_training_fixed.py`

### ✅ Documentation Consolidated
- Moved 28 archived/outdated docs to `.archive/`
- Kept only essential production documentation
- Root directory now has 9 core docs (down from 47)

### ✅ What Remains
- `project/training/main_umbrella_training_fixed.py` - **Single production training script**
- `project/dataset/umbrella_dataset_fixed.py` - **Clean dataset implementation**
- `project/dataset/umbrella_collator.py` - **LLaVA-compatible data collator**
- `project/tests/validate_tokenization.py` - **Tokenization validation (4/4 passing)**

---

## Project Structure (Cleaned)

```
UMBRELLA/
├── README.md                                    # This file
├── TOKENIZATION_GUIDE.md                        # Tokenization documentation
├── TRAINING_QUICKSTART.md                       # Training guide
├── CURRENT_DATASET_STRUCTURE.md                 # Data format reference
├── DATASET_QUICK_REFERENCE.md                   # Quick lookup
├── LLAVA_JSON_QUICK_REFERENCE.md               # JSON format reference
├── project/
│   ├── training/
│   │   ├── main_umbrella_training_fixed.py     # ✅ PRODUCTION TRAINING SCRIPT
│   │   ├── umbrella_utils.py                   # Training utilities
│   │   └── llava_conversation_handler.py       # Conversation formatting
│   ├── dataset/
│   │   ├── umbrella_dataset_fixed.py           # ✅ PRODUCTION DATASET
│   │   └── umbrella_collator.py                # ✅ DATA COLLATION
│   ├── tests/
│   │   └── validate_tokenization.py            # ✅ VALIDATION (4/4 PASSING)
│   ├── config/
│   │   └── umbrella_llava_train.yaml           # Training configuration
│   └── model/
│       └── patch_embed.py                      # Vision encoder patches
├── sample_data/                                # Sample data and formats
├── sample_scripts/                             # Training scripts
└── .archive/                                   # Old documentation and code
    ├── experimental_code/                      # Removed experimental files
    ├── phase_reports/                          # Phase completion reports
    ├── session_history/                        # Session documentation
    └── technical_notes/                        # Archived technical analysis
```

---

## Production Training Pipeline

### Main Training Script
**File**: `project/training/main_umbrella_training_fixed.py`

**Features**:
- ✅ LLaVA-Next tokenization format (`<|im_start|>user ... <|im_end|>`)
- ✅ Proper user turn masking (label = -100, excluded from loss)
- ✅ Generic image tokens for all modalities
- ✅ Standard HuggingFace Trainer (production-proven)
- ✅ W&B integration for experiment tracking
- ✅ Comprehensive error handling and validation

**Model**: `llava-hf/llava-interleave-qwen-0.5b-hf` (optimized for multi-modal)

**Data Format**: JSON v2 (role/content structure)
```json
{
  "conversations": [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": [...]}
  ],
  "images": [...]
}
```

---

## Tokenization System (Validated)

### Test Results
All 4 tests passing ✅:
1. ✅ **LLaVA-Next Format Generation** - Verifies proper format and markers
2. ✅ **Image Token Uniformity** - All modalities use generic `<image>` token
3. ✅ **User Turn Masking** - Loss exclusion working correctly
4. ✅ **JSON v2 Format Parsing** - Array content structure handled

### Run Validation
```bash
python project/tests/validate_tokenization.py --verbose
```

---

## Dataset Overview

### Supported Modalities
- **T1 (sMRI)**: Structural MRI (3D volumes)
- **fMRI (rsfMRI)**: Resting-state functional MRI (4D time series)
- **dMRI**: Diffusion MRI (3D volumes)

### Data Format
Organize your data as JSON with role/content structure:
```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this brain scan"},
        {"type": "image", "modality": "T1", "image_path": "/path/to/image.nii.gz"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "This is a T1-weighted MRI..."}]
    }
  ]
}
```

See `CURRENT_DATASET_STRUCTURE.md` for detailed schema.

---

## Configuration

Edit `project/config/umbrella_llava_train.yaml`:

```yaml
model_name: "llava-hf/llava-interleave-qwen-0.5b-hf"
img_size: [120, 120, 120]  # 3D image dimensions
modality: "T1"  # Options: T1, fMRI, dMRI

training:
  epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4

wandb:
  use_wandb: true
  wandb_api_key: "your_key_here"
```

---

## Key Changes & Migrations

### Breaking Changes
**None** - This cleanup only removes unused experimental code

### Recommended Usage
- Use `project/training/main_umbrella_training_fixed.py` for ALL training
- Use `project/tests/validate_tokenization.py` before each training run
- Use W&B for experiment tracking

### What Was Removed
- Experimental multi-task training (use single runs per modality)
- Broken import paths (fixed in production script)
- Unused trainer utilities (HF Trainer is standard)
- Experimental dynamic batching (use HF Trainer configs)

---

## Documentation Reference

### Core Documentation (Root)
- `TOKENIZATION_GUIDE.md` - Tokenization implementation details
- `TRAINING_QUICKSTART.md` - How to run training
- `CURRENT_DATASET_STRUCTURE.md` - Data organization
- `LLAVA_JSON_QUICK_REFERENCE.md` - JSON format reference

### Archived Documentation (`.archive/`)
- `technical_notes/` - Implementation analysis and design docs
- `phase_reports/` - Phase completion reports
- `experimental_code/` - Removed code (for reference)

---

## Next Steps

1. **Validate Tokenization**: `python project/tests/validate_tokenization.py`
2. **Configure Training**: Edit `project/config/umbrella_llava_train.yaml`
3. **Prepare Data**: Format data as JSON v2 (see `CURRENT_DATASET_STRUCTURE.md`)
4. **Run Training**: Use `project/training/main_umbrella_training_fixed.py`
5. **Monitor**: Track experiments with W&B integration

---

## Support Documentation

| Topic | File |
|-------|------|
| Tokenization Implementation | `TOKENIZATION_GUIDE.md` |
| Training Steps | `TRAINING_QUICKSTART.md` |
| Data Format | `CURRENT_DATASET_STRUCTURE.md` |
| JSON Schema | `LLAVA_JSON_QUICK_REFERENCE.md` |
| Dataset Details | `DATASET_QUICK_REFERENCE.md` |

---

## Data Structure and Organization

This section documents the organization of neuroimaging data across three modalities: structural MRI (sMRI), functional MRI (fMRI), and diffusion MRI (dMRI). Understanding this structure is essential for implementing dataloaders.

### Subject ID Format

All subject identifiers follow the NDAR (National Database for Autism Research) format:
- **Pattern**: `NDARINV` + 10 alphanumeric characters
- **Example**: `NDARINVXXXXXXXXXX`
- **Character Set**: Uppercase letters (A-Z) and digits (0-9)

---

### sMRI Data Structure

**Location**: `{DATA_ROOT}/sMRI/`

**Total Subjects**: ~11,370

**Structure**: Flat directory containing preprocessed T1-weighted NIfTI files

```
sMRI/
├── NDARINVXXXXXXXXXX.nii.gz
├── NDARINVYYYYYYYYYY.nii.gz
├── NDARINVZZZZZZZZZZ.nii.gz
└── ... (~11,370 files)
```

**File Details**:
- **Format**: NIfTI compressed (`.nii.gz`)
- **Content**: Preprocessed T1-weighted structural MRI
- **Naming**: `{SUBJECT_ID}.nii.gz`
- **Expected Shape**: `[D, H, W]` (typically 120x120x120 after preprocessing)

**Example Paths**:
```python
# Single subject T1 image
smri_path = f"{DATA_ROOT}/sMRI/NDARINVXXXXXXXXXX.nii.gz"

# Glob pattern for all subjects
all_smri = glob.glob(f"{DATA_ROOT}/sMRI/NDARINV*.nii.gz")
```

**Loading Example**:
```python
import nibabel as nib
import numpy as np

def load_smri(subject_id, data_root):
    """Load T1-weighted sMRI for a subject."""
    path = f"{data_root}/sMRI/{subject_id}.nii.gz"
    img = nib.load(path)
    data = img.get_fdata()  # Shape: [D, H, W]
    return data.astype(np.float32)
```

---

### fMRI Data Structure

**Location**: `{DATA_ROOT}/fMRI/`

**Total Subjects**: ~9,180

**Structure**: Contains imaging data in multiple formats

```
fMRI/
├── hdf5/                    # (Not used in UMBRELLA)
│   ├── NDARINVXXXXXXXXXX.h5
│   └── ... (~9,180 files)
│
└── img/                     # Frame-wise PyTorch tensors
    ├── NDARINVXXXXXXXXXX/
    │   ├── global_stats.pt
    │   ├── frame_0.pt
    │   ├── frame_1.pt
    │   ├── frame_2.pt
    │   └── ... (up to frame_361.pt)
    ├── NDARINVYYYYYYYYYY/
    │   ├── global_stats.pt
    │   └── frame_*.pt
    └── ... (~9,180 subject directories)
```

#### fMRI img/ Directory Details

Each subject directory contains:
- **`global_stats.pt`**: Global statistics for normalization
- **`frame_N.pt`**: Individual fMRI frames (N = 0 to ~361)
  - Format: PyTorch tensor (`.pt`)
  - Content: Single timepoint of 4D fMRI data
  - Frame count varies by subject (typically 362 frames)

**Example Paths**:
```python
# Subject fMRI directory
subj_dir = f"{DATA_ROOT}/fMRI/img/NDARINVXXXXXXXXXX/"

# Global stats file
stats_path = f"{subj_dir}/global_stats.pt"

# Individual frame
frame_path = f"{subj_dir}/frame_100.pt"

# All frames for a subject
frame_files = sorted(glob.glob(f"{subj_dir}/frame_*.pt"))
```

**Loading Example**:
```python
import torch
import os
from pathlib import Path

def load_fmri_frames(subject_id, data_root, frame_indices=None):
    """Load fMRI frames for a subject.

    Args:
        subject_id: Subject identifier (e.g., 'NDARINVXXXXXXXXXX')
        data_root: Root directory of data
        frame_indices: List of frame indices to load, or None for all

    Returns:
        dict with 'frames' (list of tensors) and 'global_stats' (tensor)
    """
    subj_dir = Path(data_root) / "fMRI" / "img" / subject_id

    # Load global stats
    global_stats = torch.load(subj_dir / "global_stats.pt")

    # Get all frame files
    frame_files = sorted(subj_dir.glob("frame_*.pt"))

    if frame_indices is not None:
        frame_files = [subj_dir / f"frame_{i}.pt" for i in frame_indices]

    frames = [torch.load(f) for f in frame_files]

    return {
        'frames': frames,
        'global_stats': global_stats,
        'num_frames': len(frames)
    }

def get_fmri_subject_list(data_root):
    """Get list of all subjects with fMRI data."""
    img_dir = Path(data_root) / "fMRI" / "img"
    subjects = [d.name for d in img_dir.iterdir() if d.is_dir()]
    return sorted(subjects)
```

---

### dMRI Data Structure

**Location**: `{DATA_ROOT}/dMRI/`

**Total Subjects**: ~8,340

**Structure**: Flat directory containing preprocessed diffusion MRI NIfTI files

```
dMRI/
├── NDARINVXXXXXXXXXX.nii.gz
├── NDARINVYYYYYYYYYY.nii.gz
├── NDARINVZZZZZZZZZZ.nii.gz
└── ... (~8,340 files)
```

**File Details**:
- **Format**: NIfTI compressed (`.nii.gz`)
- **Content**: Preprocessed diffusion MRI (likely FA maps or other derived metrics)
- **Naming**: `{SUBJECT_ID}.nii.gz`

**Example Paths**:
```python
# Single subject dMRI
dmri_path = f"{DATA_ROOT}/dMRI/NDARINVXXXXXXXXXX.nii.gz"

# Glob pattern for all subjects
all_dmri = glob.glob(f"{DATA_ROOT}/dMRI/NDARINV*.nii.gz")
```

**Loading Example**:
```python
import nibabel as nib
import numpy as np

def load_dmri(subject_id, data_root):
    """Load diffusion MRI for a subject."""
    path = f"{data_root}/dMRI/{subject_id}.nii.gz"
    img = nib.load(path)
    data = img.get_fdata()
    return data.astype(np.float32)
```

---

### Cross-Modality Subject Matching

Not all subjects have data for all three modalities. Here's how to find subjects with specific modality combinations:

```python
import os
from pathlib import Path

def get_subjects_by_modality(data_root):
    """Get subject lists for each modality."""
    data_root = Path(data_root)

    # sMRI subjects
    smri_subjects = {
        f.stem for f in (data_root / "sMRI").glob("NDARINV*.nii.gz")
    }

    # dMRI subjects
    dmri_subjects = {
        f.stem for f in (data_root / "dMRI").glob("NDARINV*.nii.gz")
    }

    # fMRI subjects (from img directory)
    fmri_subjects = {
        d.name for d in (data_root / "fMRI" / "img").iterdir()
        if d.is_dir() and d.name.startswith("NDARINV")
    }

    return {
        'smri': smri_subjects,
        'dmri': dmri_subjects,
        'fmri': fmri_subjects
    }

def get_multimodal_subjects(data_root, modalities=['smri', 'fmri']):
    """Get subjects with data in all specified modalities."""
    all_subjects = get_subjects_by_modality(data_root)

    # Find intersection
    common = None
    for mod in modalities:
        if common is None:
            common = all_subjects[mod]
        else:
            common = common & all_subjects[mod]

    return sorted(common)

# Example usage
# subjects_with_both = get_multimodal_subjects(DATA_ROOT, ['smri', 'fmri'])
# subjects_all_three = get_multimodal_subjects(DATA_ROOT, ['smri', 'fmri', 'dmri'])
```

---

### Complete Dataloader Implementation Pattern

Here's a comprehensive pattern for building a multi-modal dataloader:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path

class UMBRELLADataset(Dataset):
    """Multi-modal brain MRI dataset for UMBRELLA."""

    def __init__(
        self,
        data_root,
        modalities=['smri'],  # Options: 'smri', 'fmri', 'dmri'
        subject_list=None,    # Optional: list of subject IDs to use
        transform=None,
        fmri_frames=None      # None for all, or list of indices
    ):
        self.data_root = Path(data_root)
        self.modalities = modalities
        self.transform = transform
        self.fmri_frames = fmri_frames

        # Get subjects with all requested modalities
        if subject_list is not None:
            self.subjects = sorted(subject_list)
        else:
            self.subjects = get_multimodal_subjects(
                self.data_root,
                self.modalities
            )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        sample = {'subject_id': subject_id}

        # Load requested modalities
        if 'smri' in self.modalities:
            smri_path = self.data_root / 'sMRI' / f'{subject_id}.nii.gz'
            smri_data = nib.load(smri_path).get_fdata()
            sample['smri'] = torch.tensor(smri_data, dtype=torch.float32)

        if 'dmri' in self.modalities:
            dmri_path = self.data_root / 'dMRI' / f'{subject_id}.nii.gz'
            dmri_data = nib.load(dmri_path).get_fdata()
            sample['dmri'] = torch.tensor(dmri_data, dtype=torch.float32)

        if 'fmri' in self.modalities:
            fmri_dir = self.data_root / 'fMRI' / 'img' / subject_id

            # Load global stats
            global_stats = torch.load(fmri_dir / 'global_stats.pt')

            # Load frames
            if self.fmri_frames is None:
                frame_files = sorted(fmri_dir.glob('frame_*.pt'))
            else:
                frame_files = [fmri_dir / f'frame_{i}.pt' for i in self.fmri_frames]

            frames = torch.stack([torch.load(f) for f in frame_files])

            sample['fmri'] = frames
            sample['fmri_stats'] = global_stats

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage
"""
dataset = UMBRELLADataset(
    data_root='/path/to/data',
    modalities=['smri', 'fmri'],
    fmri_frames=list(range(0, 362, 10))  # Every 10th frame
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

for batch in dataloader:
    smri = batch['smri']      # [B, D, H, W]
    fmri = batch['fmri']      # [B, T, D, H, W]
    subject_ids = batch['subject_id']
    # ... training loop
"""
```

---

### Data Summary

| Modality | Subjects | File Format | Structure | Typical Shape |
|----------|----------|-------------|-----------|---------------|
| sMRI | ~11,370 | `.nii.gz` | Flat directory | [120, 120, 120] |
| fMRI | ~9,180 | `.pt` frames | Subject directories | [362, D, H, W] |
| dMRI | ~8,340 | `.nii.gz` | Flat directory | [D, H, W] |

### Implementation Notes

1. **File Existence Checking**: Always verify file existence before loading, as some subjects may have missing data
2. **Memory Management**: For fMRI, load frames on-demand rather than all at once
3. **Normalization**: Use `global_stats.pt` for proper fMRI normalization
4. **Subject Matching**: Cross-modality matching is essential; use set operations

---

## LLaVA Architecture vs BLIP

This implementation uses **LLaVA architecture**, which differs from BLIP in several key ways:

| Feature | LLaVA | BLIP-2 |
|---------|-------|--------|
| Projector | Linear MLP | Q-Former |
| Vision-Language Interface | Direct concatenation | Cross-attention |
| Training | End-to-end instruction tuning | Stage-wise training |
| Loss | Unified NLL loss | Multiple losses (ITC, ITM, LM) |

UMBRELLA uses the simpler LLaVA approach with linear projection, making it more efficient and easier to train while maintaining competitive performance.

## Prompt Format

The model uses the LLaVA chat format:

```
USER: <image>
You are a neurologist and now you are analyzing T1-weighted MRI images.
Estimate sex of subject from this image.
ASSISTANT: male
```

For JSON-format prompts (recommended for UMBRELLA):
```json
{
  "image": "<image_path>",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are a neurologist analyzing T1-weighted MRI images. Estimate the subject's sex."
    },
    {
      "from": "gpt",
      "value": "male"
    }
  ]
}
```

## Dependencies

- PyTorch >= 2.0
- Transformers >= 4.31
- MONAI (for medical image loading)
- timm (for vision models)
- wandb (for logging)
- nibabel (for NIfTI loading)

## References

- [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485)
- [LLaVA-1.5: Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)
- [UMBRELLA Vision Documents](../projects/BrainVLM/docs/VISION_AND_STRATEGY/)

## License

See the main BrainVLM repository for license information.
