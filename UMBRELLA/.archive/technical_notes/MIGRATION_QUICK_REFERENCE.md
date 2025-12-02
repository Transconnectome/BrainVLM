# Migration Quick Reference - V2 to Primary

**Date**: 2025-11-27
**Status**: Complete ✅

---

## TL;DR

All "V2" and "_v2" suffixes have been removed. Use the standard names now.

---

## Import Changes

### OLD (V2) ❌
```python
from project.dataloaders_v2 import (
    ImageLoaderV2,
    ConversationProcessorV2,
    T1JSONDatasetV2,
    UMBRELLADataLoaderV2
)
```

### NEW (Primary) ✅
```python
from project.dataloaders import (
    ImageLoader,
    ConversationProcessor,
    T1JSONDataset,
    UMBRELLADataLoader
)
```

---

## Class Name Changes

| Old Name (V2) | New Name (Primary) |
|---------------|-------------------|
| `ImageLoaderV2` | `ImageLoader` |
| `ConversationProcessorV2` | `ConversationProcessor` |
| `T1JSONDatasetV2` | `T1JSONDataset` |
| `UMBRELLADataLoaderV2` | `UMBRELLADataLoader` |

---

## File Path Changes

| Old Path (V2) | New Path (Primary) |
|--------------|-------------------|
| `project/dataloaders_v2/` | `project/dataloaders/` |
| `sample_data/sex_comparison_conversations_v2/` | `sample_data/sex_comparison_conversations/` |
| `generate_sex_comparison_conversations_v2.py` | `generate_sex_comparison_conversations.py` |
| `validate_json_format_v2.py` | `validate_json_format.py` |

---

## Documentation Changes

| Old Name (V2) | New Name (Primary) |
|--------------|-------------------|
| `LLAVA_JSON_V2_INDEX.md` | `LLAVA_JSON_INDEX.md` |
| `LLAVA_JSON_V2_IMPLEMENTATION_REPORT.md` | `LLAVA_JSON_IMPLEMENTATION_REPORT.md` |
| `LLAVA_JSON_V2_QUICK_REFERENCE.md` | `LLAVA_JSON_QUICK_REFERENCE.md` |
| `JSON_FORMAT_V2_SPECIFICATION.md` | `JSON_FORMAT_SPECIFICATION.md` |
| `TRAINING_WITH_JSON_V2.md` | `TRAINING_WITH_JSON.md` |

---

## Usage Examples

### Data Generation
```bash
# Generate sex comparison conversations
cd sample_data
python generate_sex_comparison_conversations.py

# Validate generated data
python validate_json_format.py
```

### Training Code
```python
from project.dataloaders import create_umbrella_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_umbrella_dataloaders(
    json_dir="sample_data/sex_comparison_conversations",
    batch_size=32,
    num_workers=4
)

# Use in training loop
for batch in train_loader:
    images = batch['images']
    conversations = batch['conversations']
    # ... train model
```

### Direct Dataset Usage
```python
from project.dataloaders import T1JSONDataset

# Create dataset
dataset = T1JSONDataset(
    json_dir="sample_data/sex_comparison_conversations/train",
    image_dir="path/to/images"
)

# Get a sample
sample = dataset[0]
```

---

## Migration Checklist for Your Code

- [ ] Replace `dataloaders_v2` with `dataloaders` in imports
- [ ] Remove `V2` suffix from all class names
- [ ] Update data directory paths (remove `_v2`)
- [ ] Update script references (remove `_v2.py`)
- [ ] Update documentation links
- [ ] Test imports work correctly
- [ ] Regenerate data if needed
- [ ] Run validation on generated data

---

## Quick Find/Replace Commands

For sed (Unix/Mac):
```bash
# Update Python imports
sed -i '' 's/dataloaders_v2/dataloaders/g' your_file.py
sed -i '' 's/ImageLoaderV2/ImageLoader/g' your_file.py
sed -i '' 's/ConversationProcessorV2/ConversationProcessor/g' your_file.py
sed -i '' 's/T1JSONDatasetV2/T1JSONDataset/g' your_file.py
sed -i '' 's/UMBRELLADataLoaderV2/UMBRELLADataLoader/g' your_file.py

# Update file paths
sed -i '' 's/sex_comparison_conversations_v2/sex_comparison_conversations/g' your_file.py
```

For VS Code (Find and Replace):
```
Find:    dataloaders_v2
Replace: dataloaders

Find:    V2
Replace: (empty - just remove)
```

---

## Troubleshooting

### Import Error: "No module named 'dataloaders_v2'"
**Solution**: Update import to use `dataloaders` instead of `dataloaders_v2`

### Import Error: "cannot import name 'ImageLoaderV2'"
**Solution**: Use `ImageLoader` instead of `ImageLoaderV2`

### FileNotFoundError: "sex_comparison_conversations_v2"
**Solution**: Update path to `sex_comparison_conversations` or regenerate data

### Module has no attribute 'V2'
**Solution**: Remove "V2" suffix from class names in your code

---

## Documentation Links

- Full Migration Report: `V1_TO_PRIMARY_MIGRATION_COMPLETE.md`
- LLaVA JSON Index: `LLAVA_JSON_INDEX.md`
- Training Guide: `sample_data/TRAINING_WITH_JSON.md`
- Format Specification: `sample_data/JSON_FORMAT_SPECIFICATION.md`

---

## Support

If you encounter issues after migration:
1. Check this quick reference
2. Review the full migration report
3. Ensure all imports use new class names
4. Regenerate data if directory structure changed

---

**Last Updated**: 2025-11-27
**Migration Status**: Complete ✅
