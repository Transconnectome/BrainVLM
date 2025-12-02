# Modality Handling - Quick Visual Guide

**Quick Answer**: ✅ **YES - The dataloader fully supports incomplete modalities**

---

## Visual Flow: How Incomplete Modalities Work

### Your JSON File (Mixed Modalities)
```
JSON:
├─ sub-0001: {T1: ✅, fMRI: ✅, dMRI: ✅}  (all modalities)
├─ sub-0002: {T1: ✅, fMRI: ❌, dMRI: ❌}  (T1 only)
├─ sub-0003: {T1: ✅, fMRI: ✅, dMRI: ❌}  (T1 + fMRI only)
└─ sub-0004: {T1: ✅, fMRI: ❌, dMRI: ✅}  (T1 + dMRI only)
```

### Each Dataset Processes This

```
T1JSONDataset:
  └─ Looks for: 'T1', 'image_sMRI', 'sMRI'
     ├─ sub-0001: ✅ Found
     ├─ sub-0002: ✅ Found
     ├─ sub-0003: ✅ Found
     └─ sub-0004: ✅ Found
     Result: 4 samples

fMRIDataset:
  └─ Looks for: 'fMRI', 'rsfMRI', 'rsfmri'
     ├─ sub-0001: ✅ Found
     ├─ sub-0002: ❌ NOT found → SKIP
     ├─ sub-0003: ✅ Found
     └─ sub-0004: ❌ NOT found → SKIP
     Result: 2 samples

dMRIDataset:
  └─ Looks for: 'dMRI', 'dwi'
     ├─ sub-0001: ✅ Found
     ├─ sub-0002: ❌ NOT found → SKIP
     ├─ sub-0003: ❌ NOT found → SKIP
     └─ sub-0004: ✅ Found
     Result: 2 samples
```

### InterleaveDataset Combines Them
```
Dataset pool: 4 T1 + 2 fMRI + 2 dMRI = 8 total
Sampling probabilities: 50% T1, 25% fMRI, 25% dMRI

Random batch sequence:
├─ Batch 1: [T1, T1, fMRI]         (from random selection)
├─ Batch 2: [T1, dMRI, T1]         (from random selection)
├─ Batch 3: [T1, fMRI, fMRI]       (from random selection)
└─ Batch 4: [T1, T1, dMRI]         (from random selection)
```

### Data Collator Adapts to Each Batch
```
Batch 1: [T1, T1, fMRI]
         ↓
Collator detects: T1 + rsfMRI
         ↓
Output:
{
  'T1': {pixel_values, input_ids, attention_mask, labels},
  'rsfMRI': {pixel_values, input_ids, attention_mask, labels}
}

Batch 2: [T1, dMRI, T1]
         ↓
Collator detects: T1 + dMRI
         ↓
Output:
{
  'T1': {pixel_values, input_ids, attention_mask, labels},
  'dMRI': {pixel_values, input_ids, attention_mask, labels}
}

Batch 3: [T1, T1]  (only T1 samples)
         ↓
Collator detects: T1 only
         ↓
Output:
{
  'T1': {pixel_values, input_ids, attention_mask, labels}
}
```

### Trainer Handles Each Batch Type
```
Single-Modality Batch (e.g., [T1, T1, T1]):
  compute_loss():
    ├─ dummy_loss = 1e-7 × (rsfMRI_params + dMRI_params)
    ├─ actual_loss = real_loss(T1)
    └─ total_loss = dummy_loss + actual_loss ✅

  Gradients:
    ├─ ∂loss/∂T1_params = large (real signal)
    ├─ ∂loss/∂rsfMRI_params = small (dummy signal)
    └─ ∂loss/∂dMRI_params = small (dummy signal)

Multi-Modality Batch (e.g., [T1, fMRI]):
  compute_loss():
    ├─ No dummy loss needed
    ├─ actual_loss = unified_loss(T1_samples + fMRI_samples)
    └─ total_loss = actual_loss ✅

  Gradients:
    ├─ ∂loss/∂T1_params = from T1 samples
    ├─ ∂loss/∂rsfMRI_params = from fMRI samples
    └─ ∂loss/∂dMRI_params = 0 (not in batch)
```

---

## Concrete Example: JSON File

### What You Can Write
```json
[
  {
    "task_id": "age_prediction",
    "subject_id": "sub-ABCD-0001",
    "modality_paths": {
      "image_sMRI": "/data/ABCD/T1/sub-0001/T1w.nii.gz",
      "rsfMRI": "/data/ABCD/fMRI/sub-0001/",
      "dMRI": "/data/ABCD/dMRI/sub-0001/dwi.nii.gz"
    },
    "conversations": [
      {"from": "human", "value": "<image>\nHow old is this subject?"},
      {"from": "gpt", "value": "Based on the brain MRI..."}
    ]
  },
  {
    "task_id": "age_prediction",
    "subject_id": "sub-ABCD-0002",
    "modality_paths": {
      "image_sMRI": "/data/ABCD/T1/sub-0002/T1w.nii.gz"
    },
    "conversations": [...]
  },
  {
    "task_id": "age_prediction",
    "subject_id": "sub-UKB-0001",
    "modality_paths": {
      "image_sMRI": "/data/UKB/T1/sub-0001/T1w.nii.gz",
      "rsfMRI": "/data/UKB/fMRI/sub-0001/"
    },
    "conversations": [...]
  }
]
```

✅ **This works perfectly!**

- sub-ABCD-0001: Has all 3 modalities
- sub-ABCD-0002: Has only T1 (fMRI/dMRI datasets skip it)
- sub-UKB-0001: Has T1 + fMRI (dMRI dataset skips it)

---

## Key Flexibility Points

### 1. Different Modalities Per Subject
```
Same JSON file can have:
- Some subjects with all 3 modalities
- Some subjects with only T1
- Some subjects with T1 + fMRI
- Some subjects with T1 + dMRI
- etc.

✅ Each dataset picks what it needs
```

### 2. Flexible Key Names
```
These all work (case-insensitive):
{
  "image_sMRI": "..."      ✅
  or
  "T1": "..."              ✅
  or
  "sMRI": "..."            ✅
  or
  "t1": "..."              ✅
}
```

### 3. Optional Modalities
```
Don't need to include all modalities:

Option A - Complete:
{ "modality_paths": {"image_sMRI": "...", "rsfMRI": "...", "dMRI": "..."} }

Option B - Partial (also works):
{ "modality_paths": {"image_sMRI": "..."} }

✅ Both valid!
```

### 4. Batch Composition
```
Training batches automatically adjust:

Single modality:  [T1, T1, T1]           → Uses dummy loss
Multi-modality:   [T1, fMRI, T1]        → Uses unified loss
Sparse:          [T1, dMRI, fMRI]      → Uses unified loss

✅ All handled automatically
```

---

## What Happens Internally

### Step 1: Dataset Loading
```
Load JSON → Each dataset searches for its modality
├─ T1JSONDataset: "Give me any sample with T1"
├─ fMRIDataset: "Give me samples with rsfMRI (skip others)"
└─ dMRIDataset: "Give me samples with dMRI (skip others)"
```

### Step 2: Interleaving
```
Combine datasets with probability weighting:
T1: 60% (appears most often)
fMRI: 30% (appears less often)
dMRI: 10% (appears least often)
```

### Step 3: Batch Formation
```
Create batches by random sampling:
Batch 1: 3 samples from pool → might be [T1, T1, fMRI]
Batch 2: 3 samples from pool → might be [T1, T1, T1]
Batch 3: 3 samples from pool → might be [T1, dMRI, fMRI]
etc.
```

### Step 4: Collation
```
For each batch, collator:
1. Detects which modalities are present
2. Groups samples by modality
3. Pads each modality independently
4. Returns structure: {modality: {pixel_values, input_ids, ...}}
```

### Step 5: Training
```
For each batch, trainer:
1. Checks if single or multi-modality
2. If single → Apply dummy loss for inactive modalities
3. If multi → Use unified loss for all samples
4. Backprop updates all embeddings
```

---

## FAQ: Will This Work?

### Q: My JSON has only T1 for all subjects?
**A**: ✅ **YES** - T1JSONDataset will load all, fMRI/dMRI will be empty

### Q: Some subjects have T1, some have T1+fMRI?
**A**: ✅ **YES** - fMRI dataset skips those without fMRI

### Q: I want all 3 modalities but some subjects are missing dMRI?
**A**: ✅ **YES** - dMRI dataset skips subjects without dMRI

### Q: Will batches have mixed modalities?
**A**: ✅ **YES** - InterleaveDataset samples from all available

### Q: Will the trainer handle single-modality batches?
**A**: ✅ **YES** - Dummy loss mechanism ensures gradient flow

### Q: Do I need to do anything special in my JSON?
**A**: ❌ **NO** - Just include what you have in `modality_paths`

### Q: What if a key name is different?
**A**: ✅ **Works** - Flexible key matching (case-insensitive)

---

## Real-World Example

**Scenario**: You have ABCD and UKB datasets
- ABCD: All subjects have T1 + fMRI
- UKB: All subjects have T1 only (no fMRI)

**Your JSON**:
```json
[
  {"modality_paths": {"image_sMRI": "abcd/T1/...", "rsfMRI": "abcd/fMRI/"}, ...},  // ABCD
  {"modality_paths": {"image_sMRI": "abcd/T1/...", "rsfMRI": "abcd/fMRI/"}, ...},  // ABCD
  {"modality_paths": {"image_sMRI": "ukb/T1/..."}, ...},  // UKB (no fMRI)
  {"modality_paths": {"image_sMRI": "ukb/T1/..."}, ...},  // UKB (no fMRI)
]
```

**Result**:
- T1JSONDataset: Uses all 4 subjects
- fMRIDataset: Uses only 2 subjects (ABCD)
- Training batches: Mix of T1-only and T1+fMRI
- Dummy loss: Handles T1-only batches automatically

✅ **Works seamlessly!**

---

## Summary

| Aspect | Support | Ease |
|--------|---------|------|
| **Incomplete modalities** | ✅ Yes | Very easy - just omit them |
| **Variable per subject** | ✅ Yes | Datasets handle it automatically |
| **Single-modality batches** | ✅ Yes | Dummy loss does the work |
| **Mixed batches** | ✅ Yes | Collator adapts automatically |
| **Configuration needed** | ❌ No | Works out of the box |
| **Special handling** | ❌ No | Everything is automatic |

---

**Bottom Line**: Your system is **designed for exactly this scenario**. Just include the modalities you have, and everything else happens automatically!
