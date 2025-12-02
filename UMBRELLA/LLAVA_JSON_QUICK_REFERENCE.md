# LLaVA JSON  - Quick Reference Guide

## 🚀 Quick Start (3 Commands)

```bash
# 1. Generate data
cd sample_data
python3 generate_sex_comparison_conversations_.py

# 2. Validate format
python3 validate_json_format_.py

# 3. Use in training
python3 your_training_script.py
```

---

## 📋 JSON Format (At a Glance)

```json
{
    "task_id": "unique_identifier",
    "task_type": "T3",
    "subject_ids": ["subject1", "subject2"],
    "modalities": ["sMRI", "sMRI"],
    "images": [
        {"path": "...", "token": "<image>", "modality": "sMRI"}
    ],
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {"type": "image", "modality": "sMRI", "image_path": "..."}
            ]
        }
    ],
    "metadata": {...}
}
```

**Critical:** Use `<image>` (NOT `<image_sMRI>`)

---

## 🔧 Basic Usage

### Load Data

```python
from project.dataloaders_ import UMBRELLADataLoader

dataset = UMBRELLADataLoader(
    json_dir="sex_comparison_conversations_",
    split="train",
    tokenizer=tokenizer,
    processor=processor
)

dataloader = dataset.create_dataloader(batch_size=4)
```

### Training Loop

```python
for batch in dataloader:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        labels=batch["labels"]
    )
    loss = outputs.loss
    loss.backward()
```

---

## 📁 File Locations

```
UMBRELLA/
├── sample_data/
│   ├── generate_sex_comparison_conversations_.py  # Data generator
│   ├── validate_json_format_.py                   # Validator
│   ├── JSON_FORMAT__SPECIFICATION.md              # Full spec
│   ├── TRAINING_WITH_JSON_.md                     # Training guide
│   └── sex_comparison_conversations_/             # Generated data
│
└── project/dataloaders_/
    ├── image_loader_.py              # Image loading
    ├── conversation_processor_.py    # Format conversion
    ├── t1_json_dataset_.py           # Dataset class
    └── umbrella_dataloader_.py       # Main dataloader
```

---

## ✅ Validation Checklist

```bash
# Run validation
python3 validate_json_format_.py

# Expected: ✓ VALIDATION PASSED
```

**Check:**
- [ ] Generic `<image>` tokens
- [ ] Lowercase roles
- [ ] Required fields present
- [ ] Image count matches
- [ ] Modality values valid

---

## 🎯 Key Differences from V1

| Feature | V1 |  |
|---------|----|----|
| Token | `<image_sMRI>` | `<image>` |
| Type | `"image_sMRI"` | `"image"` |
| Images array | Missing | Required |
| Metadata | Basic | Comprehensive |

---

## 🔥 Common Issues & Fixes

### Out of Memory
```python
# Reduce batch size
dataloader = dataset.create_dataloader(batch_size=1)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Image Loading Error
```python
# Check path
from project.dataloaders_ import load_image
image = load_image("path.nii.gz", modality="sMRI")
```

### Slow Loading
```python
# Increase workers
dataloader = dataset.create_dataloader(
    batch_size=4,
    num_workers=4
)
```

---

## 📊 Batch Structure

```python
batch = {
    "input_ids": Tensor[batch, seq_len],
    "attention_mask": Tensor[batch, seq_len],
    "pixel_values": Tensor[batch, num_images, 3, H, W],
    "labels": Tensor[batch, seq_len],
    "metadata": List[Dict],
    "task_ids": List[str]
}
```

---

## 💡 Quick Tips

1. **Always validate before training**
   ```bash
   python3 validate_json_format_.py
   ```

2. **Start with small batch**
   ```python
   batch_size=1  # Test first
   ```

3. **Monitor GPU usage**
   ```python
   print(torch.cuda.memory_allocated() / 1e9)
   ```

4. **Save checkpoints**
   ```python
   torch.save(model.state_dict(), "checkpoint.pt")
   ```

---

## 🔗 Documentation Links

- **Full Spec:** `JSON_FORMAT__SPECIFICATION.md`
- **Training:** `TRAINING_WITH_JSON_.md`
- **Report:** `LLAVA_JSON__IMPLEMENTATION_REPORT.md`

---

## 📞 Quick Commands

```bash
# Generate data
python3 generate_sex_comparison_conversations_.py

# Validate
python3 validate_json_format_.py

# Test processor
cd project/dataloaders_
python3 conversation_processor_.py

# Test dataloader
python3 t1_json_dataset_.py
```

---

## 🎓 Example Configurations

### Memory-Efficient
```python
dataset = UMBRELLADataLoader(..., max_length=1024)
dataloader = dataset.create_dataloader(batch_size=1)
model.gradient_checkpointing_enable()
```

### High-Performance
```python
dataset = UMBRELLADataLoader(..., max_length=2048)
dataloader = dataset.create_dataloader(
    batch_size=8,
    num_workers=4,
    pin_memory=True
)
```

### Multi-GPU
```python
from torch.nn.parallel import DistributedDataParallel
model = DistributedDataParallel(model)
```

---

**Version:** 2.0 | **Status:** ✅ Ready for Production
