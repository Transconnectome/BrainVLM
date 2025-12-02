# Quick Start: Directory-Based Data Loading

## TL;DR - What Changed

Your training data can now be **individual JSON files in a directory** instead of one monolithic JSON file.

```bash
# OLD WAY (still works):
--train-json ./data/train.json  # Single file with all samples

# NEW WAY (now supported):
--train-data ./sample_data/train/  # Directory with individual files
```

---

## Basic Usage

### Train on All Samples in Directory
```bash
python project/training/main_umbrella_training_fixed.py \
    --config config/umbrella_llava_train.yaml \
    --train-data ./sample_data/sex_comparison_conversations/train/ \
    --eval-data ./sample_data/sex_comparison_conversations/validation/ \
    --modality T1
```

**What it does**: Loads all 200 JSON files from train directory

---

### Train on Specific Task Type Only
```bash
python project/training/main_umbrella_training_fixed.py \
    --config config/umbrella_llava_train.yaml \
    --train-data ./sample_data/sex_comparison_conversations/train/ \
    --task-filter same_sex_comparison \
    --modality T1
```

**What it does**: Only loads samples with "same_sex_comparison" in their task_id (~100 files)

---

## Task Filtering Examples

### Same-Sex Comparisons Only
```bash
--task-filter same_sex_comparison
```
Loads: `NDARINV*_same_sex_comparison.json` files only

### Different-Sex Comparisons Only
```bash
--task-filter different_sex_comparison
```
Loads: `NDARINV*_different_sex_comparison.json` files only

### All Tasks (No Filter)
```bash
# Simply omit the --task-filter argument
```
Loads: All JSON files in directory

---

## Data Structure Requirements

Each JSON file should contain ONE sample in this format:

```json
{
  "task_id": "NDARINV00CY2LNV_same_sex_comparison",
  "task_type": "T3",
  "subject_ids": ["NDARINV1", "NDARINV2"],
  "modalities": ["sMRI", "sMRI"],
  "images": [
    {"path": "/path/to/image1.nii.gz", "token": "<image>", "modality": "sMRI"},
    {"path": "/path/to/image2.nii.gz", "token": "<image>", "modality": "sMRI"}
  ],
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this scan."},
        {"type": "image", "modality": "sMRI", "image_path": "/path/to/image1.nii.gz"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "This is a T1-weighted MRI showing..."}
      ]
    }
  ],
  "metadata": {"source": "ABCD"}
}
```

---

## Complete Training Command

```bash
python project/training/main_umbrella_training_fixed.py \
    --config config/umbrella_llava_train.yaml \
    --train-data ./sample_data/sex_comparison_conversations/train/ \
    --eval-data ./sample_data/sex_comparison_conversations/validation/ \
    --modality T1 \
    --task-filter same_sex_comparison \
    --batch-size 2 \
    --learning-rate 5e-5 \
    --output-dir ./results/same_sex_only \
    --no-wandb
```

**Explanation**:
- `--train-data`: Directory with training JSON files
- `--eval-data`: Directory with validation JSON files
- `--modality`: T1 (sMRI) or rsfMRI
- `--task-filter`: Only load specific task type (optional)
- `--batch-size`: Override config batch size
- `--learning-rate`: Override config learning rate
- `--output-dir`: Where to save checkpoints
- `--no-wandb`: Disable Weights & Biases logging

---

## Troubleshooting

### Error: "No JSON files found in directory"
**Solution**: Check that your directory contains .json files
```bash
ls ./sample_data/sex_comparison_conversations/train/*.json
```

### Error: "Training data not found"
**Solution**: Verify the path is correct
```bash
# Absolute path is safer:
--train-data /full/path/to/sample_data/sex_comparison_conversations/train/

# Or relative path from project root:
--train-data ./sample_data/sex_comparison_conversations/train/
```

### Warning: "Failed to load X files"
**Solution**: Check those specific JSON files for format errors
```bash
python -m json.tool failing_file.json
```

### No samples loaded with task filter
**Solution**: Verify your filter string matches task_id patterns
```bash
# Check actual task_ids in your files:
grep "task_id" ./sample_data/sex_comparison_conversations/train/*.json | head -5
```

---

## Performance Tips

1. **Use absolute paths** for image files in JSON (faster loading)
2. **Filter early** using `--task-filter` to load only needed samples
3. **Monitor logs** to see how many samples were loaded
4. **Test with small directory first** before full training

---

## Backward Compatibility

If you have existing single JSON files, they still work:

```bash
# This still works (old format):
--train-data ./data/train.json
```

The system automatically detects whether you're pointing to a file or directory.

---

## Quick Validation

Test that your data loads correctly without training:

```bash
python test_directory_loading.py
```

Expected output:
```
================================================================================
TEST SUMMARY
================================================================================
  Directory Structure: PASS
  JSON Format: PASS
  Mock Dataset Loading: PASS
  Task Filtering: PASS
  Collator Compatibility: PASS

Total: 5/5 tests passed
================================================================================

ALL TESTS PASSED - Refactoring successful!
```

---

## Next Steps

1. ✅ Verify your data structure matches requirements
2. ✅ Run validation test (`python test_directory_loading.py`)
3. ✅ Test training with small directory first
4. ✅ Scale up to full dataset

---

## Additional Help

- Full documentation: `REFACTORING_COMPLETE_REPORT.md`
- Training guide: `TRAINING_WITH_JSON.md`
- Format spec: `JSON_FORMAT_SPECIFICATION.md`

**Status**: Ready to use! 🚀
