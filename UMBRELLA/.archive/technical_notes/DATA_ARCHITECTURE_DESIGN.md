# UMBRELLA Data Architecture Design

## Executive Summary

This document presents a hybrid data loading architecture for UMBRELLA:

1. **JSON-Based Loading (sMRI, dMRI)**: Single generic dataset class that reads from JSON files
2. **Inheritance-Based Loading (fMRI)**: BaseDataset pattern with dataset-specific subclasses

This design maximizes simplicity for static modalities while preserving flexibility for temporal fMRI data.

---

## 1. Architecture Overview

### 1.1 Design Philosophy

| Modality | Loading Pattern | Rationale |
|----------|----------------|-----------|
| **sMRI** | JSON-based | Static 3D volumes, no temporal complexity, text prompts pre-formatted |
| **dMRI** | JSON-based | Static 3D volumes, same simplicity as sMRI |
| **fMRI** | Inheritance-based | Temporal sequences require dataset-specific padding, TR, downsampling |

### 1.2 Key Principles

1. **Single Responsibility**: Each component handles one concern
2. **Open/Closed**: Easy to add new datasets without modifying existing code
3. **Configuration over Code**: Dataset variations expressed in JSON, not hard-coded
4. **Unified Output**: All datasets produce consistent output format

---

## 2. JSON-Based Loading (sMRI/dMRI)

### 2.1 Design Overview

A single `StaticBrainDataset` class that:
- Takes a JSON file path as input
- Loads image paths and text prompts from JSON
- Returns formatted data ready for training

**No per-dataset subclassing required**. To add a new sMRI dataset, just create a new JSON file.

### 2.2 JSON File Structure

```json
[
  {
    "task_id": "A",
    "subject_id": "sub-0001",
    "modality_paths": {
      "sMRI": "/path/to/T1w.nii.gz",
      "dMRI": "/path/to/dMRI.nii.gz"
    },
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nYou are a neurologist analyzing T1-weighted MRI. What is the estimated age of this subject?"
      },
      {
        "from": "gpt",
        "value": "Based on the brain morphometry, the estimated age is 45 years."
      }
    ]
  },
  {
    "task_id": "B",
    "subject_id": "sub-0002",
    "modality_paths": {
      "sMRI": "/path/to/T1w.nii.gz"
    },
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nClassify the biological sex of this subject."
      },
      {
        "from": "gpt",
        "value": "male"
      }
    ]
  }
]
```

### 2.3 Implementation Design

```python
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (
    LoadImage, Compose, AddChannel, Resize,
    NormalizeIntensity, RandAxisFlip
)

class StaticBrainDataset(Dataset):
    """
    Generic dataset for static brain images (sMRI, dMRI).
    Loads data from JSON file - no subclassing needed.
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        img_size: int = 128,
        mode: str = 'train',
        modality: str = 'sMRI',  # or 'dMRI'
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.mode = mode
        self.modality = modality
        self.max_length = max_length

        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # Filter for entries that have requested modality
        self.data = [
            entry for entry in self.data
            if modality in entry.get('modality_paths', {})
        ]

        # Setup image transforms
        self.image_transform = self._build_transforms()
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)

    def _build_transforms(self):
        """Build augmentation pipeline"""
        img_size = (self.img_size, self.img_size, self.img_size)

        if self.mode == 'train':
            return Compose([
                AddChannel(),
                Resize(img_size),
                RandAxisFlip(prob=0.5),
                NormalizeIntensity()
            ])
        else:
            return Compose([
                AddChannel(),
                Resize(img_size),
                NormalizeIntensity()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        # Load image
        image_path = entry['modality_paths'][self.modality]
        image = self.image_loader(image_path)
        image = self.image_transform(image)
        image = torch.tensor(image)

        # Extract text from conversations
        conversations = entry['conversations']
        instruction = conversations[0]['value']  # Human turn
        answer = conversations[1]['value']       # GPT turn

        # Tokenize
        full_text = instruction + answer
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Create labels (mask instruction tokens)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Find answer start position for masking
        # (Simplified - production code should use proper tokenization)

        return {
            'pixel_values': {self.modality: image},
            'input_ids': {self.modality: input_ids},
            'attention_mask': {self.modality: attention_mask},
            'labels': {self.modality: labels},
            'subject_id': entry['subject_id'],
            'task_id': entry['task_id']
        }
```

### 2.4 JSON Generation Script Pattern

```python
def generate_umbrella_json(
    metadata_csv: str,
    image_dir: str,
    output_json: str,
    task_config: dict
):
    """
    Generate JSON file for UMBRELLA from metadata and images.

    Args:
        metadata_csv: Path to CSV with subject metadata
        image_dir: Directory containing images
        output_json: Output JSON path
        task_config: Task-specific configuration
    """
    import pandas as pd
    import os

    df = pd.read_csv(metadata_csv)
    entries = []

    for _, row in df.iterrows():
        subject_id = row['subject_id']

        # Find image
        image_path = os.path.join(image_dir, f"{subject_id}.nii.gz")
        if not os.path.exists(image_path):
            continue

        # Generate conversation based on task
        if task_config['task'] == 'age':
            age = row['age']
            instruction = f"<image>\nYou are a neurologist. Estimate the age of this subject."
            answer = f"{int(age)}"
        elif task_config['task'] == 'sex':
            sex = 'male' if row['sex'] == 1 else 'female'
            instruction = f"<image>\nClassify the biological sex."
            answer = sex

        entry = {
            'task_id': task_config['task_id'],
            'subject_id': subject_id,
            'modality_paths': {
                'sMRI': image_path
            },
            'conversations': [
                {'from': 'human', 'value': instruction},
                {'from': 'gpt', 'value': answer}
            ]
        }
        entries.append(entry)

    with open(output_json, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"Generated {len(entries)} entries to {output_json}")
```

---

## 3. Inheritance-Based Loading (fMRI)

### 3.1 Design Overview

fMRI requires dataset-specific handling because:
1. Different datasets have different **padding requirements** (image dimensions vary)
2. Different **temporal resolutions (TR)** require different downsampling
3. Different **background/padding values** for normalization
4. Future: Random frame selection strategies may differ

### 3.2 Abstract Base Class

```python
from abc import ABC, abstractmethod
import os
import json
import torch
from torch.utils.data import Dataset

class BasefMRIDataset(Dataset, ABC):
    """
    Abstract base class for fMRI datasets.

    Subclasses must implement:
    - get_padding_config(): Dataset-specific padding
    - get_temporal_config(): TR and downsampling
    - get_background_value(): For padding
    """

    def __init__(
        self,
        json_path: str,
        data_root: str,
        tokenizer,
        sequence_length: int = 20,
        stride_within_seq: int = 1,
        input_scaling_method: str = 'minmax',
        mode: str = 'train',
        max_length: int = 128,
    ):
        self.json_path = json_path
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride_within_seq = stride_within_seq
        self.input_scaling_method = input_scaling_method
        self.mode = mode
        self.max_length = max_length

        # Calculate derived values
        self.sample_duration = sequence_length * stride_within_seq

        # Load JSON entries
        self._load_json()

        # Build data list
        self.data = self._build_data_list()

    def _load_json(self):
        """Load and parse JSON file"""
        with open(self.json_path, 'r') as f:
            self.json_entries = json.load(f)

    def _build_data_list(self):
        """Build list of (subject_id, subject_path, start_frame, ...) tuples"""
        data = []

        for entry in self.json_entries:
            subject_id = entry['subject_id']
            subject_path = os.path.join(
                self.data_root, 'img', subject_id
            )

            if not os.path.exists(subject_path):
                continue

            # Count available frames
            num_frames = len([
                f for f in os.listdir(subject_path)
                if f.startswith('frame_')
            ])

            # Generate start positions
            session_duration = num_frames - self.sample_duration + 1
            stride = max(1, self.sample_duration // 2)  # 50% overlap

            for start_frame in range(0, session_duration, stride):
                data.append({
                    'subject_id': subject_id,
                    'subject_path': subject_path,
                    'start_frame': start_frame,
                    'num_frames': num_frames,
                    'entry': entry
                })

        return data

    # ==================== ABSTRACT METHODS ====================

    @abstractmethod
    def get_padding_config(self) -> tuple:
        """
        Return padding configuration for torch.nn.functional.pad.

        Returns:
            tuple: (left, right, top, bottom, front, back) padding
        """
        pass

    @abstractmethod
    def get_temporal_config(self) -> dict:
        """
        Return temporal resolution configuration.

        Returns:
            dict: {
                'tr': float,  # Repetition time in seconds
                'target_tr': float,  # Target TR after downsampling
                'downsample_ratio': float  # Computed from TR ratio
            }
        """
        pass

    @abstractmethod
    def get_background_value(self, image: torch.Tensor) -> float:
        """
        Return the background/padding value for this dataset.

        Args:
            image: Loaded fMRI tensor

        Returns:
            float: Value to use for padding
        """
        pass

    # ==================== COMMON METHODS ====================

    def load_sequence(self, subject_path, start_frame):
        """Load fMRI frame sequence"""
        frames = []

        for i in range(start_frame, start_frame + self.sample_duration, self.stride_within_seq):
            frame_path = os.path.join(subject_path, f'frame_{i}.pt')
            frame = torch.load(frame_path, weights_only=False).unsqueeze(0)
            frames.append(frame)

        sequence = torch.cat(frames, dim=4)

        # Apply normalization
        sequence = self._apply_normalization(subject_path, sequence)

        return sequence

    def _apply_normalization(self, subject_path, sequence):
        """Apply input scaling based on global statistics"""
        if self.input_scaling_method == 'none':
            return sequence

        stats_path = os.path.join(subject_path, 'global_stats.pt')
        stats = torch.load(stats_path, weights_only=False)

        if self.input_scaling_method == 'minmax':
            sequence = sequence / stats['global_max']

        elif self.input_scaling_method == 'znorm_zeroback':
            background = sequence == 0
            sequence = (sequence - stats['global_mean']) / stats['global_std']
            sequence[background] = 0

        return sequence

    def _apply_padding(self, sequence):
        """Apply dataset-specific padding"""
        background_value = self.get_background_value(sequence)
        padding = self.get_padding_config()

        # Permute for padding: (1, C, H, W, T) -> (1, T, C, H, W)
        sequence = sequence.permute(0, 4, 1, 2, 3)
        sequence = torch.nn.functional.pad(sequence, padding, value=background_value)
        sequence = sequence.permute(0, 2, 3, 4, 1)

        return sequence

    def _apply_downsampling(self, sequence):
        """Apply temporal downsampling based on TR difference"""
        config = self.get_temporal_config()
        ratio = config.get('downsample_ratio', 1.0)

        if ratio == 1.0:
            return sequence

        # Random start for training, fixed for eval
        T = sequence.shape[-1]
        target_T = int(T / ratio)

        if self.mode == 'train':
            start = torch.randint(0, int(ratio), (1,)).item()
        else:
            start = 0

        indices = torch.arange(start, T, int(ratio))[:target_T]
        sequence = sequence[..., indices]

        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # Load sequence
        sequence = self.load_sequence(
            sample['subject_path'],
            sample['start_frame']
        )

        # Apply dataset-specific transformations
        sequence = self._apply_padding(sequence)
        sequence = self._apply_downsampling(sequence)

        # Get text from JSON entry
        entry = sample['entry']
        conversations = entry['conversations']
        instruction = conversations[0]['value']
        answer = conversations[1]['value']

        # Tokenize
        full_text = instruction + answer
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'pixel_values': {'fMRI': sequence},
            'input_ids': {'fMRI': encoding['input_ids'].squeeze(0)},
            'attention_mask': {'fMRI': encoding['attention_mask'].squeeze(0)},
            'labels': {'fMRI': encoding['input_ids'].squeeze(0)},
            'subject_id': sample['subject_id']
        }
```

### 3.3 Dataset-Specific Implementations

#### ABCD fMRI Dataset

```python
class ABCDfMRIDataset(BasefMRIDataset):
    """ABCD Study fMRI dataset"""

    def get_padding_config(self) -> tuple:
        """ABCD images are 96x96x95, pad to 96x96x96"""
        return (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)

    def get_temporal_config(self) -> dict:
        """ABCD TR = 0.8s"""
        return {
            'tr': 0.8,
            'target_tr': 2.0,  # Common target
            'downsample_ratio': 2.5  # 2.0 / 0.8
        }

    def get_background_value(self, image: torch.Tensor) -> float:
        """ABCD background is 0"""
        return 0.0


class UKBfMRIDataset(BasefMRIDataset):
    """UK Biobank fMRI dataset"""

    def get_padding_config(self) -> tuple:
        """UKB images need significant padding"""
        return (3, 9, 0, 0, 10, 8)

    def get_temporal_config(self) -> dict:
        """UKB TR = 0.735s"""
        return {
            'tr': 0.735,
            'target_tr': 2.0,
            'downsample_ratio': 2.72
        }

    def get_background_value(self, image: torch.Tensor) -> float:
        """UKB uses first voxel as background"""
        return image.flatten()[0].item()


class HCPfMRIDataset(BasefMRIDataset):
    """Human Connectome Project (S1200) fMRI dataset"""

    def get_padding_config(self) -> tuple:
        """HCP images need similar padding to UKB"""
        return (3, 9, 0, 0, 10, 8)

    def get_temporal_config(self) -> dict:
        """HCP TR = 0.72s"""
        return {
            'tr': 0.72,
            'target_tr': 2.0,
            'downsample_ratio': 2.78
        }

    def get_background_value(self, image: torch.Tensor) -> float:
        """HCP uses first voxel as background"""
        return image.flatten()[0].item()


class HBNfMRIDataset(BasefMRIDataset):
    """Healthy Brain Network fMRI dataset"""

    def __init__(self, *args, input_type: str = 'rest', **kwargs):
        self.input_type = input_type
        super().__init__(*args, **kwargs)

    def get_padding_config(self) -> tuple:
        """HBN has different sizes for rest vs task"""
        if self.input_type == 'rest':
            # (81, 95, 81) -> (96, 96, 96)
            return (7, 8, 1, 0, 7, 8)
        else:
            # Task is (96, 96, 95) -> (96, 96, 96)
            return (0, 1, 0, 0, 0, 0)

    def get_temporal_config(self) -> dict:
        """HBN TR = 0.8s"""
        return {
            'tr': 0.8,
            'target_tr': 2.0,
            'downsample_ratio': 2.5
        }

    def get_background_value(self, image: torch.Tensor) -> float:
        return image.flatten()[0].item()


class ABIDEfMRIDataset(BasefMRIDataset):
    """ABIDE I/II fMRI dataset"""

    def get_padding_config(self) -> tuple:
        """ABIDE (97, 115, 97) -> (96, 96, 96) using negative padding"""
        return (0, -1, -10, -9, -1, 0)  # Crop instead of pad

    def get_temporal_config(self) -> dict:
        """ABIDE varies by site, use average"""
        return {
            'tr': 2.0,  # Varies, using common value
            'target_tr': 2.0,
            'downsample_ratio': 1.0
        }

    def get_background_value(self, image: torch.Tensor) -> float:
        return image.flatten()[0].item()
```

---

## 4. Integration Strategy

### 4.1 Unified Training Usage

```python
from dataset.static_brain_dataset import StaticBrainDataset
from dataset.fmri_datasets import ABCDfMRIDataset, UKBfMRIDataset
from dataset.data_utils import InterleaveDataset, CustomDataCollatorWithPadding
from torch.utils.data import DataLoader

def create_training_datasets(config, tokenizer):
    """Create all datasets for UMBRELLA training"""

    datasets = []

    # sMRI datasets (JSON-based)
    if config.use_smri:
        smri_dataset = StaticBrainDataset(
            json_path=config.smri_json_path,
            tokenizer=tokenizer,
            img_size=config.img_size,
            mode='train',
            modality='sMRI'
        )
        datasets.append(smri_dataset)

    # dMRI datasets (JSON-based)
    if config.use_dmri:
        dmri_dataset = StaticBrainDataset(
            json_path=config.dmri_json_path,
            tokenizer=tokenizer,
            img_size=config.img_size,
            mode='train',
            modality='dMRI'
        )
        datasets.append(dmri_dataset)

    # fMRI datasets (Inheritance-based)
    if config.use_fmri:
        # ABCD
        if 'ABCD' in config.fmri_datasets:
            abcd_fmri = ABCDfMRIDataset(
                json_path=config.abcd_fmri_json,
                data_root=config.abcd_data_root,
                tokenizer=tokenizer,
                sequence_length=config.sequence_length,
                mode='train'
            )
            datasets.append(abcd_fmri)

        # UKB
        if 'UKB' in config.fmri_datasets:
            ukb_fmri = UKBfMRIDataset(
                json_path=config.ukb_fmri_json,
                data_root=config.ukb_data_root,
                tokenizer=tokenizer,
                sequence_length=config.sequence_length,
                mode='train'
            )
            datasets.append(ukb_fmri)

    # Interleave all datasets
    combined_dataset = InterleaveDataset(datasets, shuffle=True)

    return combined_dataset


def create_dataloader(dataset, tokenizer, config):
    """Create DataLoader with custom collator"""

    collator = CustomDataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=config.max_length
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
```

### 4.2 Configuration Example

```yaml
# config/umbrella_training.yaml

# Common settings
img_size: 128
max_length: 128
batch_size: 8
num_workers: 4

# Modality flags
use_smri: true
use_dmri: true
use_fmri: true

# sMRI/dMRI (JSON-based)
smri_json_path: ./data/json/abcd_ukb_smri_train.json
dmri_json_path: ./data/json/abcd_dmri_train.json

# fMRI (Inheritance-based)
fmri_datasets: ['ABCD', 'UKB']
sequence_length: 20

# Dataset-specific paths
abcd_fmri_json: ./data/json/abcd_fmri_train.json
abcd_data_root: /data/ABCD/fMRI/
ukb_fmri_json: ./data/json/ukb_fmri_train.json
ukb_data_root: /data/UKB/fMRI/
```

---

## 5. Design Rationale

### 5.1 Why JSON for sMRI/dMRI?

| Reason | Explanation |
|--------|-------------|
| **Simplicity** | No temporal dimension, just single 3D volumes |
| **Pre-formatted text** | Prompts can be carefully crafted offline |
| **Zero metadata processing** | All computation done at JSON generation time |
| **Easy scaling** | Add new datasets by creating new JSON files |
| **Version control** | JSON files can be tracked and versioned |
| **Reproducibility** | Exact prompts are recorded, not generated |

### 5.2 Why Inheritance for fMRI?

| Reason | Explanation |
|--------|-------------|
| **Dataset-specific parameters** | Padding, TR, background values vary significantly |
| **Temporal complexity** | Sequence loading, downsampling need flexibility |
| **Future extensibility** | Easy to add site-specific preprocessing |
| **Performance** | Subclass can optimize for specific data layout |
| **Type safety** | Abstract methods ensure required implementations |

### 5.3 Why Not Pure JSON for fMRI?

1. **Padding varies per image**: Would need padding in JSON per entry
2. **TR differences**: Downsampling ratios need to be computed
3. **Frame-level loading**: Need to handle variable frame counts
4. **Memory efficiency**: Can't pre-compute everything like with sMRI

---

## 6. Future Extensibility

### 6.1 Adding a New sMRI Dataset

1. Create metadata CSV
2. Run JSON generation script
3. Add path to config
4. Done!

```bash
python scripts/generate_json.py \
    --metadata /data/NewDataset/metadata.csv \
    --image_dir /data/NewDataset/images/ \
    --output ./data/json/new_dataset_smri.json \
    --task age
```

### 6.2 Adding a New fMRI Dataset

1. Create new subclass:

```python
class NewDatasetfMRI(BasefMRIDataset):
    """New dataset fMRI"""

    def get_padding_config(self) -> tuple:
        # Measure your images and set padding
        return (0, 0, 0, 0, 0, 0)

    def get_temporal_config(self) -> dict:
        return {
            'tr': 1.5,  # Your TR
            'target_tr': 2.0,
            'downsample_ratio': 1.33
        }

    def get_background_value(self, image: torch.Tensor) -> float:
        return 0.0
```

2. Generate JSON for the dataset
3. Register in factory or config
4. Done!

### 6.3 Adding New Tasks

Just create new JSON files with different conversations:

```json
{
    "task_id": "brain_region_qa",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nIdentify the brain region with the highest activation."
        },
        {
            "from": "gpt",
            "value": "The highest activation is in the dorsolateral prefrontal cortex."
        }
    ]
}
```

---

## 7. Implementation Plan

### 7.1 File Organization

```
UMBRELLA/
├── project/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── static_brain_dataset.py    # JSON-based sMRI/dMRI
│   │   ├── base_fmri_dataset.py       # Abstract BaseClass for fMRI
│   │   ├── fmri_datasets.py           # ABCD, UKB, HCP, HBN, ABIDE
│   │   ├── data_utils.py              # Collator, InterleaveDataset
│   │   └── json_generators/           # Scripts to generate JSON
│   │       ├── generate_smri_json.py
│   │       ├── generate_dmri_json.py
│   │       └── generate_fmri_json.py
│   ├── config/
│   │   └── umbrella_training.yaml
│   └── ...
├── data/
│   └── json/                          # Generated JSON files
│       ├── abcd_smri_train.json
│       ├── abcd_fmri_train.json
│       ├── ukb_smri_train.json
│       └── ...
```

### 7.2 Implementation Checklist

#### Phase 1: Core Infrastructure
- [ ] Create `static_brain_dataset.py` with StaticBrainDataset
- [ ] Create `base_fmri_dataset.py` with BasefMRIDataset
- [ ] Migrate data_utils.py (CustomDataCollatorWithPadding, InterleaveDataset)

#### Phase 2: fMRI Implementations
- [ ] Implement ABCDfMRIDataset
- [ ] Implement UKBfMRIDataset
- [ ] Implement HCPfMRIDataset (S1200)
- [ ] Implement HBNfMRIDataset
- [ ] Implement ABIDEfMRIDataset

#### Phase 3: JSON Generation
- [ ] Create generate_smri_json.py script
- [ ] Create generate_fmri_json.py script
- [ ] Generate ABCD JSON files (sMRI, fMRI)
- [ ] Generate UKB JSON files (sMRI, fMRI)

#### Phase 4: Integration
- [ ] Create unified training script
- [ ] Update config system
- [ ] Test with single modality
- [ ] Test with multi-modality
- [ ] Performance benchmarking

### 7.3 Dependencies

```yaml
# Required packages
torch >= 2.0
monai >= 1.0
nibabel >= 4.0
transformers >= 4.30
pandas >= 2.0
numpy >= 1.24
```

---

## 8. Summary

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| sMRI/dMRI loading | JSON-based | Simplicity, pre-formatted prompts |
| fMRI loading | Inheritance | Dataset-specific parameters |
| Output format | Unified dict | Consistent collation |
| Text source | JSON conversations | Reproducibility |
| Padding abstraction | Abstract method | Type safety |
| TR handling | Abstract method | Dataset flexibility |

### Benefits

1. **Separation of concerns**: Static vs temporal modalities handled appropriately
2. **Scalability**: Easy to add datasets without code changes (for sMRI)
3. **Type safety**: Abstract methods enforce implementation
4. **Reproducibility**: JSON records exact prompts used
5. **Maintainability**: Clear inheritance hierarchy
6. **Performance**: Dataset-specific optimizations possible

### Success Criteria

- [ ] All existing datasets can be loaded with new architecture
- [ ] Adding new sMRI dataset requires only JSON file
- [ ] Adding new fMRI dataset requires only new subclass
- [ ] Unified training works with all modalities
- [ ] Performance comparable to current implementation
- [ ] Code is well-documented and testable

---

## Appendix: Quick Reference

### JSON Entry Schema

```json
{
    "task_id": "string",
    "subject_id": "string",
    "modality_paths": {
        "sMRI": "path/to/file.nii.gz",
        "dMRI": "path/to/file.nii.gz"
    },
    "conversations": [
        {"from": "human", "value": "instruction"},
        {"from": "gpt", "value": "answer"}
    ]
}
```

### Dataset Abstract Methods

```python
class BasefMRIDataset:
    @abstractmethod
    def get_padding_config(self) -> tuple:
        """Return (left, right, top, bottom, front, back) padding"""

    @abstractmethod
    def get_temporal_config(self) -> dict:
        """Return {'tr': float, 'target_tr': float, 'downsample_ratio': float}"""

    @abstractmethod
    def get_background_value(self, image: torch.Tensor) -> float:
        """Return padding value for this dataset"""
```

### Output Format (All Datasets)

```python
{
    'pixel_values': {modality: tensor},
    'input_ids': {modality: tensor},
    'attention_mask': {modality: tensor},
    'labels': {modality: tensor},
    'subject_id': str
}
```
