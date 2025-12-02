# UMBRELLA Dataset Quick Reference

Quick lookup guide for using the UMBRELLA dataset classes.

---

## TL;DR - Common Tasks

### Load ABCD fMRI Data
```python
from project.dataset import create_fmri_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')

dataset = create_fmri_dataset(
    'ABCD',
    json_file='abcd_fmri_train.json',
    data_root='/data/ABCD/fMRI/',
    tokenizer=tokenizer,
    sequence_length=20,
    train=True
)
```

### Load T1 (sMRI) Data
```python
from project.dataset import T1JSONDataset

dataset = T1JSONDataset(
    json_file='abcd_t1_train.json',
    data_root='/data/ABCD/T1/',
    tokenizer=tokenizer,
    img_size=128,
    mode='train'
)
```

### Load dMRI Data
```python
from project.dataset import dMRIJSONDataset

dataset = dMRIJSONDataset(
    json_file='abcd_dmri_train.json',
    data_root='/data/ABCD/dMRI/',
    tokenizer=tokenizer,
    img_size=128,
    mode='train'
)
```

### Combine All Modalities
```python
from torch.utils.data import ConcatDataset

combined = ConcatDataset([abcd_fmri, t1_dataset, dmri_dataset])
loader = DataLoader(combined, batch_size=8, num_workers=4)

for batch in loader:
    fmri_img = batch['pixel_values'].get('rsfMRI')   # (B, 1, 96, 96, 96, 20)
    t1_img = batch['pixel_values'].get('T1')        # (B, 1, 128, 128, 128)
    dmri_img = batch['pixel_values'].get('dMRI')    # (B, 1, 128, 128, 128)
    tokens = batch['input_ids']                      # {modality: tensor}
```

---

## Class Selection Guide

### When to Use Which Dataset?

| Need | Use | Reason |
|------|-----|--------|
| ABCD fMRI | `ABCDfMRIDataset` or `create_fmri_dataset('ABCD', ...)` | Inherited, fast |
| UKB fMRI | `UKBfMRIDataset` or `create_fmri_dataset('UKB', ...)` | Inherited, fast |
| HCP fMRI | `HCPfMRIDataset` or `create_fmri_dataset('HCP', ...)` | Inherited, fast |
| HBN fMRI | `HBNfMRIDataset(input_type='rest')` | Special rest/task handling |
| ABIDE fMRI | `ABIDEfMRIDataset` | Handles cropping via negative padding |
| Any sMRI/T1 | `T1JSONDataset` | Single class handles all |
| Any dMRI | `dMRIJSONDataset` | Single class handles all |
| Raw output | Use `*Raw` variant (e.g., `T1JSONDatasetRaw`) | Different return format |

---

## API Reference

### fMRI Datasets

#### Direct Instantiation
```python
from project.dataset import ABCDfMRIDataset

dataset = ABCDfMRIDataset(
    json_file: str,              # Path to JSON with samples
    data_root: str,              # Root directory for fMRI frames
    modality: str = 'fMRI',      # Modality type
    tokenizer = None,            # HF tokenizer
    image_processor = None,      # Custom processor (optional)
    max_seq_length: int = 128,   # Max token length
    sequence_length: int = 20,   # Number of fMRI frames
    stride_within_seq: int = 1,  # Frame skip factor
    stride_between_seq: int = 1, # Sequence skip factor
    input_scaling_method: str = 'znorm_zeroback',  # Normalization
    shuffle_time_sequence: bool = False,
    train: bool = True,          # Training mode
    add_context: bool = False    # Add demographic context
)
```

#### Factory Function
```python
from project.dataset import create_fmri_dataset

dataset = create_fmri_dataset(
    dataset_name: str,           # 'ABCD', 'UKB', 'HCP', 'HBN', 'ABIDE'
    json_file: str,
    data_root: str,
    output_format: str = 'hf',   # 'hf' or 'raw'
    **kwargs                      # Passed to dataset class
)
```

#### Available Datasets
- `ABCDfMRIDataset`
- `UKBfMRIDataset`
- `HCPfMRIDataset`
- `HBNfMRIDataset` (supports `input_type='rest'|'task'`)
- `ABIDEfMRIDataset`

#### Raw Variants
- `UKBfMRIDatasetRaw`
- `HBNfMRIDatasetRaw`
- `ABIDEfMRIDatasetRaw`

### T1/sMRI Dataset

```python
from project.dataset import T1JSONDataset, create_t1_dataset

# Direct instantiation
dataset = T1JSONDataset(
    json_file: str,
    data_root: str,
    modality: str = 'sMRI',      # or 'T1'
    tokenizer = None,
    image_processor = None,
    max_seq_length: int = 128,
    img_size: int = 128,         # Resize target
    mode: str = 'train',         # or 'eval'
    add_context: bool = False
)

# Factory function
dataset = create_t1_dataset(
    json_file: str,
    data_root: str,
    output_format: str = 'hf',   # 'hf' or 'raw'
    **kwargs
)

# Raw variant
raw_dataset = T1JSONDatasetRaw(...)
```

### dMRI Dataset

```python
from project.dataset import dMRIJSONDataset, create_dmri_dataset

# Direct instantiation
dataset = dMRIJSONDataset(
    json_file: str,
    data_root: str,
    modality: str = 'dMRI',
    tokenizer = None,
    image_processor = None,
    max_seq_length: int = 128,
    img_size: int = 128,         # Resize target
    mode: str = 'train',         # or 'eval'
    add_context: bool = False
)

# Factory function
dataset = create_dmri_dataset(
    json_file: str,
    data_root: str,
    output_format: str = 'hf',
    **kwargs
)

# Raw variant
raw_dataset = dMRIJSONDatasetRaw(...)
```

---

## JSON Input Format

### Required JSON Schema

```json
[
  {
    "task_id": "age_estimation",
    "subject_id": "sub-0001",
    "modality_paths": {
      "sMRI": "relative/path/to/T1w.nii.gz",
      "dMRI": "relative/path/to/dMRI.nii.gz",
      "fMRI": "sub-0001/img/"  // For fMRI: dir with frame_*.pt files
    },
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is the estimated age?"
      },
      {
        "from": "gpt",
        "value": "Based on brain morphometry, approximately 45 years."
      }
    ],
    "metadata": {
      "sex": 1,              // 1=male, 0=female
      "age": 45,
      "scanner": "Siemens"
    }
  }
]
```

### Key Points
- Paths in `modality_paths` are relative to `data_root`
- For fMRI: must point to directory with `frame_*.pt` files
- For T1/dMRI: must point to `.nii` or `.nii.gz` file
- `conversations` are pre-formatted (no metadata extraction)
- `metadata` is optional but recommended

---

## Output Format

All datasets return:
```python
{
    'pixel_values': {
        'rsfMRI': torch.Tensor,  # fMRI: (1, H, W, D, T)
        'T1': torch.Tensor,      # sMRI: (1, H, W, D)
        'dMRI': torch.Tensor     # dMRI: (1, H, W, D)
    },
    'input_ids': {
        modality: torch.Tensor   # (seq_length,)
    },
    'attention_mask': {
        modality: torch.Tensor   # (seq_length,)
    },
    'labels': {
        modality: torch.Tensor   # (seq_length,), with instruction masked
    },
    'subject_id': 'sub-0001',
    'task_id': 'age_estimation',
    'metadata': {
        'sex': 1,
        'age': 45,
        ...
    }
}
```

### Raw Format Output (for `*Raw` variants)
```python
{
    'image': torch.Tensor,           # (1, H, W, D) or (1, H, W, D, T)
    'subject_id': 'sub-0001',
    'task_id': 'age_estimation',
    'instruction': 'USER: <image>\n... ASSISTANT: ',
    'answer': 'Based on brain morphometry...',
    'metadata': {...}
}
```

---

## Normalization Methods

### For fMRI
```python
input_scaling_method: str = 'znorm_zeroback'
```
Options:
- `'none'` - No normalization
- `'minmax'` - Normalize to [0, 1] using global_max
- `'znorm_zeroback'` - Z-norm with zero background preserved
- `'znorm_minback'` - Z-norm with all voxels normalized

### For T1/dMRI
Uses MONAI's `NormalizeIntensity()` - z-score normalization

---

## Dataset-Specific Parameters

### ABCD fMRI
- Padding: `(0, 1, 0, 0, 0, 0)`
- Downsample: `2.5×` (TR 0.8s)
- Background: First element

### UKB fMRI
- Padding: `(3, 9, 0, 0, 10, 8)`
- Downsample: `1.0×` (TR 0.735s)
- Background: First element

### HCP (S1200) fMRI
- Padding: `(3, 9, 0, 0, 10, 8)`
- Downsample: `1.0×` (TR 0.72s)
- Background: First element

### HBN fMRI
- Rest padding: `(7, 8, 1, 0, 7, 8)`
- Task padding: `(0, 1, 0, 0, 0, 0)`
- Downsample: `1.0×` (TR 0.8s)
- Special: Rest data cropped to 96 on 4th dim

### ABIDE fMRI
- Padding: `(0, -1, -10, -9, -1, 0)` (negative = crop)
- Downsample: `1.0×` (TR 2.0s)
- Background: First element

---

## Utility Functions

### load_json(json_file: str)
Load JSON sample definitions.

### load_nifti(file_path: str) -> np.ndarray
Load NIfTI files with nibabel.

### load_pt_frames(subject_path: str, start_frame: int, num_frames: int, ...) -> torch.Tensor
Load fMRI frames from .pt files.

### normalize_fmri(data: torch.Tensor, stats_path: str, method: str)
Apply normalization using global statistics.

### pad_tensor(tensor: torch.Tensor, padding: tuple, value: float)
Apply dataset-specific padding.

### format_conversation(conversations: list, modality: str)
Convert conversation list to (instruction, answer) tuple.

### tokenize_conversation(instruction: str, answer: str, tokenizer, ...)
Tokenize text with optional instruction masking for training.

### get_num_frames(subject_path: str) -> int
Count available frames in directory.

### resolve_path(relative_path: str, data_root: str) -> str
Resolve path against data_root.

---

## Common Patterns

### Create Dataset and DataLoader
```python
from torch.utils.data import DataLoader
from project.dataset import create_fmri_dataset

dataset = create_fmri_dataset(
    'ABCD',
    json_file='train.json',
    data_root='/data/ABCD/fMRI/',
    tokenizer=tokenizer,
    sequence_length=20,
    train=True
)

loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=True
)

for batch in loader:
    pixel_values = batch['pixel_values']  # {modality: tensor}
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
```

### Combine Multiple Datasets
```python
from torch.utils.data import ConcatDataset, DataLoader

fmri = create_fmri_dataset('ABCD', ...)
t1 = T1JSONDataset(...)
dmri = dMRIJSONDataset(...)

combined = ConcatDataset([fmri, t1, dmri])
loader = DataLoader(combined, batch_size=8, num_workers=4)
```

### Different Modes
```python
# Training
train_dataset = ABCDfMRIDataset(..., train=True, input_scaling_method='znorm_zeroback')

# Evaluation
eval_dataset = ABCDfMRIDataset(..., train=False, input_scaling_method='znorm_zeroback')
```

### Raw Format for Compatibility
```python
# If you need raw output instead of HF format
raw_dataset = create_fmri_dataset('UKB', ..., output_format='raw')

sample = raw_dataset[0]
# sample['fmri_sequence'], sample['subject_name'], sample['target'], ...
```

---

## Troubleshooting

### Common Issues

**"No sMRI path found"**
- Check JSON has correct modality_paths key
- Key should contain 'smri' or 't1' (case-insensitive)

**"Frame not found: frame_X.pt"**
- Verify subject directory exists at data_root
- Check frames are numbered sequentially from 0

**"Stats file not found"**
- For fMRI, ensure global_stats.pt exists in subject directory
- Generated during preprocessing

**Path resolution fails**
- Verify data_root ends without trailing slash
- Check paths in JSON are relative, not absolute

---

## Performance Tips

1. **Use num_workers >= 4** for efficient data loading
2. **Set pin_memory=True** for GPU training
3. **Adjust sequence_length** to balance GPU memory vs temporal coverage
4. **Use stride_within_seq > 1** to reduce computational cost
5. **Normalize ahead of time** to avoid redundant computation

---

## Dataset Statistics

| Dataset | N Subjects | N Tasks | Total Samples | Est. Size |
|---------|-----------|--------|--------------|-----------|
| ABCD | 11,370 | Multiple | ~50K | ~5 TB |
| UKB | 9,180 | Multiple | ~40K | ~4 TB |
| HCP | ~1,200 | 4 | ~4.8K | ~500 GB |
| HBN | ~2,570 | 2 | ~10K | ~1 TB |
| ABIDE | ~1,113 | 1 | ~1K | ~500 GB |

---

**Version**: 1.0
**Last Updated**: November 20, 2025
**Status**: Ready for Production
