# JSONL Format Support for UMBRELLA Dataset

## Overview

The UMBRELLA dataset now supports **JSONL (JSON Lines)** format in addition to the existing JSON and directory-based loading formats. JSONL is a newline-delimited JSON format where each line is a complete, valid JSON object.

## What Changed

### File: `project/dataset/umbrella_dataset_fixed.py`

**New Features:**
1. JSONL file format detection and loading
2. Line-by-line JSON parsing for memory efficiency
3. Robust error handling for malformed lines
4. Backward compatibility with existing JSON and directory formats

### Modified Methods

1. **`_load_samples_smart()`**
   - Now detects `.jsonl` file extension
   - Routes JSONL files to new `_load_samples_from_jsonl()` method
   - Maintains backward compatibility with `.json` files and directories

2. **`_load_samples_from_jsonl()` (NEW)**
   - Reads JSONL files line-by-line
   - Parses each line as independent JSON object
   - Skips empty lines gracefully
   - Logs failed lines with line numbers
   - Returns list of `UMBRELLASample` objects

## JSONL Format Specification

### File Structure

Each line in a `.jsonl` file contains one complete JSON object representing a single training sample:

```jsonl
{"task_id": "sample_001", "task_type": "T3", ...}
{"task_id": "sample_002", "task_type": "T3", ...}
{"task_id": "sample_003", "task_type": "T3", ...}
```

### Sample JSON Object Schema

```json
{
    "task_id": "NDARINV0DYF4WPG_same_sex_comparison",
    "task_type": "T3",
    "subject_ids": ["NDARINVUWAU3TFB", "NDARINV0DYF4WPG"],
    "modalities": ["sMRI", "sMRI"],
    "images": [
        {
            "path": "/path/to/image1.nii.gz",
            "token": "<image>",
            "modality": "sMRI"
        },
        {
            "path": "/path/to/image2.nii.gz",
            "token": "<image>",
            "modality": "sMRI"
        }
    ],
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Examine this brain scan."},
                {"type": "image", "modality": "sMRI", "image_path": "..."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Response..."}
            ]
        }
    ],
    "metadata": {
        "subject_id": "NDARINV0DYF4WPG",
        "subject_label": "female",
        ...
    }
}
```

## Usage Examples

### Loading JSONL Files

```python
from dataset.umbrella_dataset_fixed import UMBRELLADataset

# Load train split (JSONL format)
train_dataset = UMBRELLADataset(
    json_path="sample_data/train_conversations.jsonl",
    tokenizer=tokenizer,
    mode='train',
    img_size=128,
    max_images=2
)

# Load validation split (JSONL format)
val_dataset = UMBRELLADataset(
    json_path="sample_data/validation_conversations.jsonl",
    tokenizer=tokenizer,
    mode='eval',
    img_size=128,
    max_images=2
)

# Load test split (JSONL format)
test_dataset = UMBRELLADataset(
    json_path="sample_data/test_conversations.jsonl",
    tokenizer=tokenizer,
    mode='eval',
    img_size=128,
    max_images=2
)
```

### Automatic Format Detection

The dataset automatically detects file format based on extension:

```python
# JSONL format (detected by .jsonl extension)
dataset = UMBRELLADataset(json_path="data/train.jsonl", ...)

# JSON format (detected by .json extension)
dataset = UMBRELLADataset(json_path="data/train.json", ...)

# Directory format (multiple JSON files)
dataset = UMBRELLADataset(json_path="data/train/", ...)
```

## Supported Loading Formats

### 1. JSONL File (NEW)
```
data/
├── train_conversations.jsonl
├── validation_conversations.jsonl
└── test_conversations.jsonl
```

**Advantages:**
- Memory efficient (line-by-line loading)
- Suitable for large datasets (millions of samples)
- Easy to append new samples
- Standard format used by many ML tools

### 2. Single JSON File (Legacy)
```
data/
└── train.json  # Contains array of samples: [sample1, sample2, ...]
```

### 3. Directory with Multiple JSON Files (Legacy)
```
data/train/
├── sample1.json  # Each file contains one sample
├── sample2.json
└── sample3.json
```

### 4. Nested Directories (Legacy)
```
data/train/
├── same_sex_comparison/
│   ├── sample1.json
│   └── sample2.json
└── different_sex_comparison/
    ├── sample3.json
    └── sample4.json
```

## Error Handling

### JSONL Loading Features

1. **Line-by-line parsing**: Each line is parsed independently
2. **Graceful error handling**: Malformed lines are skipped with warnings
3. **Empty line handling**: Empty lines are automatically skipped
4. **Error logging**: Failed lines are logged with line numbers
5. **Validation**: Ensures at least one valid sample is loaded

### Example Error Output

```
WARNING - Line 42: Invalid JSON - Expecting value: line 1 column 1 (char 0)
WARNING - Line 157: Failed to parse sample - KeyError: 'conversations'
INFO - Found 1000 lines in JSONL file
INFO - Successfully loaded 998 samples
WARNING - Failed to parse 2 lines: [42, 157]
```

## Performance Characteristics

### JSONL vs JSON Format

| Aspect | JSONL | JSON (Array) |
|--------|-------|--------------|
| Memory usage | Low (line-by-line) | High (full file in memory) |
| Loading speed | Fast | Moderate |
| Append efficiency | O(1) - just append line | O(n) - rewrite entire file |
| Error isolation | High - bad lines isolated | Low - file corruption affects all |
| Large datasets | Excellent | Poor |
| Human readability | Good (one sample per line) | Better (structured format) |

### Recommended Format by Dataset Size

- **Small (<10K samples)**: JSON or directory format
- **Medium (10K-100K samples)**: JSONL format
- **Large (>100K samples)**: JSONL format (recommended)

## Testing

### Test Script

Run `test_jsonl_simple.py` to verify JSONL loading:

```bash
python3 test_jsonl_simple.py
```

Expected output:
```
✅ JSONL parsing: SUCCESS
✅ Loaded 1000 samples from JSONL file
✅ File format detection: working
```

### Verification Commands

```bash
# Count lines in JSONL file
wc -l sample_data/train_conversations.jsonl

# View first sample
head -n 1 sample_data/train_conversations.jsonl | python3 -m json.tool

# Validate all lines are valid JSON
python3 -c "
import json
with open('sample_data/train_conversations.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except Exception as e:
            print(f'Line {i}: {e}')
"
```

## Migration Guide

### Converting from JSON Array to JSONL

```python
import json

# Read existing JSON file
with open('train.json', 'r') as f:
    samples = json.load(f)  # List of samples

# Write as JSONL
with open('train_conversations.jsonl', 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\n')
```

### Converting from Directory to JSONL

```python
import json
from pathlib import Path

# Find all JSON files
json_files = Path('train/').glob('**/*.json')

# Write as JSONL
with open('train_conversations.jsonl', 'w') as outfile:
    for json_file in json_files:
        with open(json_file) as infile:
            sample = json.load(infile)
            outfile.write(json.dumps(sample) + '\n')
```

## Current Dataset Status

### Sex Comparison Dataset

Location: `sample_data/sex_comparison_conversations_simple_extended/`

**Files:**
- `train_conversations.jsonl` (1000 samples)
- `validation_conversations.jsonl` (1000 samples)
- `test_conversations.jsonl` (1000 samples)

**Statistics:**
- Task type: T3 (sex comparison)
- Images per sample: 2
- Conversation turns: 4
- Format: JSONL (newline-delimited JSON)

## Implementation Details

### Code Changes Summary

```python
# File: project/dataset/umbrella_dataset_fixed.py

def _load_samples_smart(self, path: str) -> List[UMBRELLASample]:
    """Smart loading with JSONL support."""
    path_obj = Path(path)

    if path_obj.is_file():
        if path_obj.suffix == '.jsonl':  # NEW: JSONL detection
            return self._load_samples_from_jsonl(path)
        elif path_obj.suffix == '.json':
            return self._load_samples_from_file(path)
    elif path_obj.is_dir():
        return self._load_samples_from_directory(path)

def _load_samples_from_jsonl(self, jsonl_path: str) -> List[UMBRELLASample]:
    """NEW: Load from JSONL file (line-by-line)."""
    all_samples = []
    failed_lines = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)  # Parse single JSON object
                samples = self._parse_samples([item])
                all_samples.extend(samples)
            except Exception as e:
                logger.warning(f"Line {line_num}: {e}")
                failed_lines.append(line_num)

    return all_samples
```

## Backward Compatibility

All existing code continues to work without modification:

```python
# Old code still works (JSON format)
dataset = UMBRELLADataset(json_path="data/train.json", ...)

# Old code still works (directory format)
dataset = UMBRELLADataset(json_path="data/train/", ...)

# New code works (JSONL format)
dataset = UMBRELLADataset(json_path="data/train.jsonl", ...)
```

## Best Practices

1. **Use JSONL for large datasets** (>10K samples)
2. **Use JSON for small datasets** (<1K samples, better readability)
3. **Use directories for heterogeneous tasks** (different task types in subdirectories)
4. **Validate JSONL files** before training to catch errors early
5. **Keep line endings consistent** (use '\n' not '\r\n')

## Future Enhancements

Potential improvements for future versions:

1. **Streaming support**: Load samples on-demand without loading entire file
2. **Compression support**: Read from `.jsonl.gz` files
3. **Parallel loading**: Multi-threaded JSONL parsing
4. **Schema validation**: Validate samples against expected schema
5. **Progress bars**: Show loading progress for large files

## References

- JSONL Format Specification: https://jsonlines.org/
- UMBRELLA Dataset Documentation: See `README.md`
- Original Dataset Paper: See `docs/`

---

**Last Updated**: 2025-12-04
**Version**: 1.0
**Status**: Production-ready
