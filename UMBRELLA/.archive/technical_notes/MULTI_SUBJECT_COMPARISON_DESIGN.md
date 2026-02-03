# Multi-Subject MRI Comparison - Implementation Design

**Date**: November 20, 2025
**Status**: Design Specification Ready for Implementation
**Priority**: Medium (Optional extension)

---

## Executive Summary

**User Question**: Can the dataloader handle multi-subject comparison format?

**Answer**: **NO** - Requires moderate architectural changes

**Recommended Solution**: Option E - Hybrid with `comparison_mode` flag

**Implementation Effort**: 9-14 hours total (can be done incrementally)

---

## Problem Statement

### Current Limitation

Current dataloader assumes:
- One `subject_id` per JSON entry (string)
- One image path per modality (string or single path)
- Single sample output from `__getitem__`

### User's Requirement

Support multi-subject comparison:
```json
{
  "subject_id": ["sub-1", "sub-2"],
  "modality_paths": {
    "image_sMRI": ["/path/to/sub-1", "/path/to/sub-2"],
    "image_fMRI": [[frames_sub1], [frames_sub2]]
  }
}
```

### Use Cases

1. **Contrastive Learning**: Compare two subjects, identify disease/differences
2. **Metric Learning**: Learn similarity between subject pairs
3. **Classification with Reference**: Classify test subject relative to reference
4. **Clinical Comparison**: Report differences in brain patterns

---

## Recommended Solution: Option E (Hybrid with Flag)

### Why This Option?

✅ Achieves user's goal (paired subject training)
✅ Backward compatible (default `comparison_mode=False`)
✅ Minimal disruption to existing code
✅ Clear, explicit API
✅ Extensible to N>2 subjects
✅ Production-ready approach

### Architecture Overview

```
JSON Entry with ["sub-1", "sub-2"]
           ↓
T1JSONDataset(comparison_mode=True)
           ↓
__getitem__() → _get_comparison_item()
           ↓
Output: {'pixel_values': {'T1': (N, 1, H, W, D)}, 'comparison_mode': True}
           ↓
ComparisonAwareCollator._collate_comparison()
           ↓
Batch: {'T1': {'pixel_values': (B, N, 1, H, W, D), 'comparison_mode': True}}
           ↓
CustomTrainer._compute_comparison_loss()
           ↓
Model fusion: flatten (B,N) → (B*N), encode, reshape (B,N,tokens,hidden)
           ↓
LLM generates answer, compute NLL loss
```

---

## Implementation Plan

### Phase 1: Dataset Changes (2-3 hours, Low Risk)

#### File: `project/dataset/t1_json_dataset.py`

**Step 1.1: Add comparison_mode parameter**

Location: Lines 54-80 in `__init__`

```python
def __init__(
    self,
    json_file: str,
    data_root: str,
    modality: str = 'sMRI',
    tokenizer=None,
    image_processor=None,
    max_seq_length: int = 128,
    img_size: int = 128,
    mode: str = 'train',
    add_context: bool = False,
    comparison_mode: bool = False,  # NEW PARAMETER
    **kwargs
):
    super().__init__()
    ...
    self.comparison_mode = comparison_mode
```

**Step 1.2: Add _get_comparison_item method**

Add after `_load_and_process_image()` method (after line 185):

```python
def _get_comparison_item(self, index: int) -> Dict[str, Any]:
    """Get a multi-subject comparison sample.

    Returns stacked images from multiple subjects with comparison metadata.

    Args:
        index: Sample index

    Returns:
        Dict with:
        - pixel_values[modality]: (N, 1, H, W, D) stacked images
        - comparison_mode: True
        - num_subjects: N
        - subject_ids: ['sub-1', 'sub-2', ...]
        - input_ids, attention_mask, labels: (seq_len,)
    """
    self.randomize()
    sample = self.samples[index]

    # Extract subject IDs and verify list format
    subject_ids = sample.get('subject_id', [])
    if not isinstance(subject_ids, list):
        subject_ids = [subject_ids]

    modality_paths = sample.get('modality_paths', {})
    conversations = sample.get('conversations', [])
    metadata = sample.get('metadata', {})

    # Get image paths for all subjects
    smri_paths = None
    for key in modality_paths:
        if 'smri' in key.lower() or 't1' in key.lower():
            smri_paths = modality_paths[key]
            break

    if smri_paths is None:
        raise ValueError(f"No sMRI paths found in sample {index}")

    # Ensure smri_paths is a list
    if not isinstance(smri_paths, list):
        smri_paths = [smri_paths]

    if len(smri_paths) != len(subject_ids):
        raise ValueError(
            f"Number of paths ({len(smri_paths)}) != number of subject_ids ({len(subject_ids)})"
        )

    # Load all subjects' images
    images = []
    for i, path in enumerate(smri_paths):
        try:
            resolved_path = resolve_path(path, self.data_root)
            img = self._load_and_process_image(resolved_path)
            images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image for subject {subject_ids[i]}: {e}")

    # Stack images: (N, 1, H, W, D)
    stacked_images = torch.stack(images, dim=0)

    # Process text
    inst, answer = self._process_text(conversations, metadata)

    # Determine modality key
    modality_key = None
    for key in ['T1', 'image_sMRI', 'sMRI', 't1']:
        if key in self.modality or self.modality.lower() in key.lower():
            modality_key = key
            break
    if modality_key is None:
        modality_key = 'T1'  # Default

    # Format output
    inputs = {
        'pixel_values': {modality_key: stacked_images},
        'input_ids': {},
        'attention_mask': {},
        'labels': {},
        'subject_ids': subject_ids,
        'comparison_mode': True,
        'num_subjects': len(subject_ids)
    }

    # Tokenize if tokenizer available
    if self.tokenizer is not None:
        token_dict = tokenize_conversation(
            inst, answer, self.tokenizer, self.max_seq_length
        )
        inputs['input_ids'][modality_key] = token_dict['input_ids']
        inputs['attention_mask'][modality_key] = token_dict['attention_mask']
        inputs['labels'][modality_key] = token_dict['labels']

    # Add metadata
    inputs['task_id'] = sample.get('task_id', '')
    inputs['metadata'] = metadata

    return inputs
```

**Step 1.3: Modify __getitem__ to route based on comparison_mode**

Replace current `__getitem__` (lines 287-323) with:

```python
def __getitem__(self, index: int) -> Dict[str, Any]:
    """Get a sample, routing to comparison or single-subject logic.

    Args:
        index: Sample index

    Returns:
        Dict with pixel_values, input_ids, attention_mask, labels
    """
    sample = self.samples[index]

    # Check if this is a multi-subject sample
    is_multi_subject = isinstance(sample.get('subject_id'), list)

    if is_multi_subject:
        if not self.comparison_mode:
            raise ValueError(
                f"Sample {index} has multiple subjects (comparison format) "
                f"but comparison_mode=False. "
                f"Initialize dataset with comparison_mode=True to enable multi-subject comparison."
            )
        return self._get_comparison_item(index)
    else:
        # Single subject (existing logic)
        return self._get_single_item(index)

def _get_single_item(self, index: int) -> Dict[str, Any]:
    """Get single-subject sample (original __getitem__ logic)."""
    self.randomize()
    sample = self.samples[index]
    subject_id = sample.get('subject_id', f'subject_{index}')

    modality_paths = sample.get('modality_paths', {})
    image_path = None
    for key in modality_paths:
        if 'smri' in key.lower() or 't1' in key.lower():
            image_path = modality_paths[key]
            break

    if image_path is None:
        raise ValueError(f"No sMRI path found in sample {index}")

    image_path = resolve_path(image_path, self.data_root)
    image = self._load_and_process_image(image_path)

    conversations = sample.get('conversations', [])
    metadata = sample.get('metadata', {})
    inst, answer = self._process_text(conversations, metadata)

    inputs = self.__preprocess_as_hf__(image, inst, answer)

    return inputs
```

**Summary of Phase 1 Changes:**
- Add `comparison_mode: bool = False` parameter to `__init__`
- Store as `self.comparison_mode`
- Add `_get_comparison_item()` method (~80 lines)
- Add `_get_single_item()` method (extract from current `__getitem__`)
- Modify `__getitem__()` to route based on mode (~15 lines)
- **Total**: ~100 lines, backward compatible

---

### Phase 2: Collator Changes (1-2 hours, Medium Risk)

#### File: `project/utils/data.py`

**Step 2.1: Modify `__call__` to detect comparison mode**

Location: Lines 104-185 in `CustomDataCollatorWithPadding`

```python
def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch of features, handling both standard and comparison modes.

    Args:
        features: List of feature dicts from dataset

    Returns:
        Collated batch dict
    """
    # Check if any feature is in comparison mode
    has_comparison = any(f.get('comparison_mode', False) for f in features)

    if has_comparison:
        return self._collate_comparison(features)
    else:
        return self._collate_standard(features)
```

**Step 2.2: Add _collate_standard method**

Rename current collation logic:

```python
def _collate_standard(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate standard single-subject samples (original logic)."""
    # Existing lines 104-185 moved here
    ...
```

**Step 2.3: Add _collate_comparison method**

Add after `_collate_standard`:

```python
def _collate_comparison(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate comparison mode samples with multi-subject grouping.

    Args:
        features: List of comparison samples

    Returns:
        Batch with comparison_mode=True, pixel_values shape (B, N, 1, H, W, D)
    """
    # Extract modalities present
    modalities = list(set(
        modality
        for feature in features
        for modality in feature['pixel_values'].keys()
    ))

    # Initialize batch dict
    batch = {}

    for modality in modalities:
        # Get features with this modality
        modal_features = [f for f in features if modality in f['pixel_values']]

        if not modal_features:
            continue

        # Get num_subjects from first feature (assume consistent within batch)
        num_subjects = modal_features[0].get('num_subjects', 1)

        # Collect pixel values: each is (N, 1, H, W, D)
        pixel_values = [f['pixel_values'][modality] for f in modal_features]

        # Stack to (B, N, 1, H, W, D)
        batch_pixels = torch.stack(pixel_values, dim=0)

        # Collect text features
        input_ids = [f['input_ids'].get(modality) for f in modal_features]
        attention_masks = [f['attention_mask'].get(modality) for f in modal_features]
        labels = [f['labels'].get(modality) for f in modal_features]

        # Pad text sequences
        text_batch = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }
        padded_text = self._process_modality(text_batch)

        # Assemble modality batch
        batch[modality] = {
            'pixel_values': batch_pixels,
            'input_ids': padded_text['input_ids'],
            'attention_mask': padded_text['attention_mask'],
            'labels': padded_text['labels'],
            'comparison_mode': True,
            'num_subjects': num_subjects
        }

    return batch
```

**Summary of Phase 2 Changes:**
- Detect `comparison_mode` in `__call__`
- Split logic into `_collate_standard` and `_collate_comparison`
- Handle stacking with extra (N,) dimension
- Preserve text tokenization
- **Total**: ~50 lines, low disruption

---

### Phase 3: Trainer Changes (2-3 hours, Medium Risk)

#### File: `project/utils/Trainer.py`

**Step 3.1: Detect comparison mode in compute_loss**

Location: Lines 257-324 in `compute_loss`

```python
def compute_loss(self, model, inputs, return_outputs=False):
    """Compute loss, handling both standard and comparison modes."""
    self._ensure_set_static_graph(model)

    modalities = list(inputs.keys())

    # Check for comparison mode
    is_comparison = any(
        isinstance(inputs[m], dict) and inputs[m].get('comparison_mode', False)
        for m in modalities
    )

    if is_comparison:
        return self._compute_comparison_loss(model, inputs, return_outputs)

    # Existing single/multi-modality logic
    # ... rest of compute_loss ...
```

**Step 3.2: Add _compute_comparison_loss method**

Add after `compute_loss`:

```python
def _compute_comparison_loss(self, model, inputs, return_outputs=False):
    """Compute loss for multi-subject comparison mode.

    Flattens (B, N) subjects for encoding, then reshapes embeddings
    to handle paired/grouped subjects.

    Args:
        model: Model to compute loss with
        inputs: Batch dict with comparison_mode=True, pixel_values (B, N, 1, H, W, D)
        return_outputs: Whether to return model outputs

    Returns:
        Loss scalar or (loss, outputs) tuple
    """
    modalities = list(inputs.keys())

    if len(modalities) != 1:
        raise NotImplementedError(
            "Comparison mode currently supports single modality only. "
            f"Got {len(modalities)} modalities: {modalities}"
        )

    modality = modalities[0]
    modal_inputs = inputs[modality]

    # Extract shapes
    pixel_values = modal_inputs['pixel_values']  # (B, N, 1, H, W, D)
    B, N = pixel_values.shape[:2]
    input_ids = modal_inputs['input_ids']  # (B, seq_len)
    attention_mask = modal_inputs['attention_mask']  # (B, seq_len)
    labels = modal_inputs['labels']  # (B, seq_len)

    # Flatten subjects for vision encoding: (B*N, 1, H, W, D)
    flat_pixels = pixel_values.view(B * N, *pixel_values.shape[2:])

    # Prepare model inputs
    # NOTE: Model forward pass must be modified to:
    # 1. Accept comparison_mode=True
    # 2. Encode B*N images
    # 3. Reshape embeddings to (B, N, tokens, hidden)
    # 4. Fuse N subjects' embeddings into single sequence
    # 5. Match with (B, seq_len) text inputs

    model_inputs = {
        'pixel_values': flat_pixels,
        'input_ids': input_ids,  # Will need special handling in model
        'attention_mask': attention_mask,
        'labels': labels,
        'comparison_mode': True,
        'num_subjects': N,
        'batch_size': B
    }

    # Forward pass (requires model modification)
    outputs = model(**model_inputs)
    loss = outputs.loss

    if self.args.gradient_accumulation_steps > 1:
        loss = loss / self.args.gradient_accumulation_steps

    return (loss, outputs) if return_outputs else loss
```

**Summary of Phase 3 Changes:**
- Detect comparison mode in `compute_loss`
- Route to `_compute_comparison_loss`
- Handle (B, N) flattening and metadata passing
- **Total**: ~60 lines, medium disruption

**⚠️ CRITICAL**: The model forward pass MUST be modified to handle the `comparison_mode` flag. This is Phase 4 and requires architectural changes.

---

### Phase 4: Model Changes (4-6 hours, High Risk, Requires Architecture Decision)

#### Decision Point: Fusion Strategy

The model must decide HOW to fuse multiple subjects' embeddings. Choose one:

**Option A: Concatenation (Simplest)**
```python
# In model forward():
# embeddings: (B, N, tokens, hidden)
fused = embeddings.view(B, N * tokens, hidden)  # (B, N*tokens, hidden)
# Pro: Simple, preserves all information
# Con: Variable sequence length based on N
```

**Option B: Mean Pooling (Moderate)**
```python
# embeddings: (B, N, tokens, hidden)
fused = embeddings.mean(dim=1)  # (B, tokens, hidden)
# Pro: Fixed sequence length
# Con: Loses subject-specific information
```

**Option C: Attention Fusion (Complex)**
```python
# embeddings: (B, N, tokens, hidden)
# Learn to weight subjects
weights = self.comparison_fusion_attn(embeddings)  # (B, N, 1, 1)
fused = (embeddings * weights).sum(dim=1)  # (B, tokens, hidden)
# Pro: Learned fusion
# Con: Requires new parameters, more training time
```

#### Required Model Modification

Pseudo-code for model forward:

```python
def forward(self, pixel_values, input_ids, attention_mask, labels=None,
            comparison_mode=False, num_subjects=None, batch_size=None, **kwargs):

    if comparison_mode and num_subjects:
        # pixel_values: (B*N, 1, H, W, D)
        # Encode all subjects
        vision_outputs = self.vision_tower(pixel_values)
        # vision_outputs: (B*N, num_tokens, hidden_dim)

        # Reshape to (B, N, num_tokens, hidden_dim)
        num_tokens = vision_outputs.shape[1]
        hidden_dim = vision_outputs.shape[2]
        embeddings = vision_outputs.view(batch_size, num_subjects, num_tokens, hidden_dim)

        # Fuse subjects (choose strategy above)
        fused_embeddings = self._fuse_comparison_embeddings(embeddings)
        # fused_embeddings: (B, num_tokens, hidden_dim) or (B, N*num_tokens, hidden_dim)

        # Project to LLM input space
        image_features = self.mm_projector(fused_embeddings)

        # Continue with LLM...
    else:
        # Standard single-subject path
        # ... existing code ...
```

---

## Implementation Checklist

### Pre-Implementation
- [ ] Decide on fusion strategy (concatenation recommended for MVP)
- [ ] Confirm use case and training objectives
- [ ] Plan model architecture changes
- [ ] Create test dataset with multi-subject samples

### Phase 1: Dataset (2-3 hours)
- [ ] Add `comparison_mode` parameter to `__init__`
- [ ] Implement `_get_comparison_item()`
- [ ] Implement `_get_single_item()`
- [ ] Modify `__getitem__()` routing logic
- [ ] Test with sample JSON
- [ ] Verify output shapes: (N, 1, H, W, D)
- [ ] Do the same for `fmri_json_dataset.py` and `dmri_json_dataset.py`

### Phase 2: Collator (1-2 hours)
- [ ] Implement `_collate_comparison()`
- [ ] Implement `_collate_standard()`
- [ ] Modify `__call__()` routing
- [ ] Test batch shapes: (B, N, 1, H, W, D)
- [ ] Verify text padding works correctly
- [ ] Test mixed batches (if needed)

### Phase 3: Trainer (2-3 hours)
- [ ] Add comparison mode detection
- [ ] Implement `_compute_comparison_loss()`
- [ ] Test loss computation
- [ ] Verify gradient flow
- [ ] Test with dummy backward pass

### Phase 4: Model (4-6 hours)
- [ ] Choose fusion strategy
- [ ] Implement fusion logic
- [ ] Update model forward pass
- [ ] Test end-to-end training
- [ ] Validate gradient flow through fusion

### Testing & Validation
- [ ] Unit tests for each component
- [ ] Integration test: data → collator → trainer → model
- [ ] Comparison vs single-subject benchmarking
- [ ] Gradient flow verification
- [ ] Memory usage analysis

---

## Testing Strategy

### Unit Test: Dataset

```python
def test_comparison_mode():
    # JSON with multi-subject
    json_data = [{
        "subject_id": ["sub-001", "sub-002"],
        "modality_paths": {"image_sMRI": [path1, path2]},
        "conversations": [...]
    }]

    dataset = T1JSONDataset(
        json_file=...,
        comparison_mode=True
    )

    sample = dataset[0]

    # Verify shapes
    assert sample['pixel_values']['T1'].shape == (2, 1, 128, 128, 128)
    assert sample['comparison_mode'] == True
    assert sample['num_subjects'] == 2
    assert sample['subject_ids'] == ["sub-001", "sub-002"]
    print("✅ Dataset test passed")
```

### Unit Test: Collator

```python
def test_comparison_collator():
    # Create batch of comparison samples
    features = [sample1, sample2, sample3]  # 3 comparison samples

    collator = CustomDataCollatorWithPadding(...)
    batch = collator(features)

    # Verify shapes
    assert batch['T1']['pixel_values'].shape == (3, 2, 1, 128, 128, 128)
    assert batch['T1']['comparison_mode'] == True
    print("✅ Collator test passed")
```

### Integration Test

```python
def test_end_to_end():
    dataset = T1JSONDataset(..., comparison_mode=True)
    dataloader = DataLoader(dataset, collate_fn=collator, batch_size=4)

    batch = next(iter(dataloader))

    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Verify gradients
    assert model.vision_tower.weight.grad is not None
    assert model.llm.weight.grad is not None
    print("✅ End-to-end test passed")
```

---

## JSON Format Examples

### Single Subject (Original Format)
```json
{
    "task_id": "age_prediction",
    "subject_id": "sub-ABCD-0001",
    "modality_paths": {
        "image_sMRI": "/data/sMRI/sub-ABCD-0001/T1w.nii.gz",
        "rsfMRI": "/data/fMRI/sub-ABCD-0001/"
    },
    "conversations": [
        {"from": "human", "value": "<image>\nHow old is this subject?"},
        {"from": "gpt", "value": "Based on the brain MRI..."}
    ]
}
```

### Multi-Subject Comparison (New Format)
```json
{
    "task_id": "comparison_disease",
    "subject_id": ["sub-ABCD-0001", "sub-ABCD-0002"],
    "modality_paths": {
        "image_sMRI": [
            "/data/sMRI/sub-ABCD-0001/T1w.nii.gz",
            "/data/sMRI/sub-ABCD-0002/T1w.nii.gz"
        ],
        "rsfMRI": [
            "/data/fMRI/sub-ABCD-0001/",
            "/data/fMRI/sub-ABCD-0002/"
        ]
    },
    "conversations": [
        {
            "from": "human",
            "value": "Compare these two brain scans. Which subject shows signs of neurodegeneration?"
        },
        {
            "from": "gpt",
            "value": "Subject 1 (sub-ABCD-0001) shows more pronounced ventricular enlargement..."
        }
    ],
    "metadata": {
        "subject_1": {"age": 65, "sex": "M", "diagnosis": "normal"},
        "subject_2": {"age": 68, "sex": "F", "diagnosis": "MCI"}
    }
}
```

---

## Backward Compatibility

### All changes are backward compatible:

1. **Dataset**: Default `comparison_mode=False` preserves existing behavior
2. **Collator**: `_collate_comparison` only called if comparison samples present
3. **Trainer**: `_compute_comparison_loss` only called if comparison batch detected
4. **Model**: New parameters are optional

### Migration path for existing code:

```python
# Old code still works
dataset = T1JSONDataset(json_file=...)  # comparison_mode=False by default
dataloader = DataLoader(dataset, collate_fn=collator)
trainer.train()

# New code uses new feature
dataset = T1JSONDataset(json_file=..., comparison_mode=True)
dataloader = DataLoader(dataset, collate_fn=collator)
trainer.train()  # Automatically handles comparison samples
```

---

## Alternative Quick Solution (Workaround)

If implementation cannot happen immediately, use this workaround:

### Preprocessing Step

Pre-concatenate comparison subjects offline:

```python
import torch
import nibabel as nib

def create_comparison_sample(path1, path2, output_path):
    """Pre-create concatenated comparison sample."""
    img1 = nib.load(path1).get_fdata()  # (H, W, D)
    img2 = nib.load(path2).get_fdata()  # (H, W, D)

    # Stack: (2, H, W, D)
    stacked = np.stack([img1, img2], axis=0)

    # Save
    torch.save(stacked, output_path)  # saves as .pt
```

### Updated JSON

```json
{
    "task_id": "comparison_A",
    "subject_id": "sub-001_vs_sub-002",
    "modality_paths": {
        "image_sMRI": "/data/preprocessed/comparison_001_002.pt"
    }
}
```

### Modified Dataset

```python
def _load_and_process_image(self, image_file: str) -> torch.Tensor:
    if image_file.endswith('.pt'):
        # Already stacked: (2, H, W, D)
        return torch.load(image_file).unsqueeze(1)  # → (2, 1, H, W, D)
    else:
        # Standard loading
        return super()._load_and_process_image(image_file)
```

**Pros**: Minimal code changes, works immediately
**Cons**: Requires preprocessing, no architecture learning opportunity

---

## Recommended Next Steps

1. **Confirm Vision**: Validate use case and training objectives
2. **Phase 1 Implementation**: Dataset changes (lowest risk)
3. **Phase 2 Implementation**: Collator changes (medium risk)
4. **Phase 3 Implementation**: Trainer changes (medium risk)
5. **Phase 4 Design**: Model fusion strategy decision
6. **Phase 4 Implementation**: Model changes (highest risk)
7. **Testing**: Comprehensive unit and integration tests
8. **Validation**: End-to-end training with real multi-subject data

---

## Summary

| Aspect | Status |
|--------|--------|
| **Can current system handle multi-subject?** | ❌ NO |
| **Is it possible to add?** | ✅ YES |
| **Recommended approach** | Option E (Hybrid with flag) |
| **Implementation effort** | 9-14 hours |
| **Backward compatible** | ✅ YES |
| **Production ready** | ✅ Can be (with testing) |
| **Highest risk phase** | Phase 4 (Model changes) |

---

**Status**: Design Specification Complete
**Next Action**: User decision on whether to implement or use workaround
