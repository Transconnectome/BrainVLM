# Multi-Modality Sequential Extension - Roadmap & Design

**Phase**: 2 (Recommended Extension)
**Status**: Design Complete, Ready for Implementation
**Estimated Effort**: 2-3 hours
**Date**: November 20, 2025

---

## Overview

**Current State** (Scenario 1 & 3):
- ✅ Single subject, single image (T1)
- ✅ Multiple subjects, single image per subject (T1)

**Proposed Extension** (Scenario 2):
- Single subject, multiple modalities (T1 + dMRI, or T1 + other)
- Sequential multi-turn conversation
- Cross-modality integration via transformer attention

**Why This Extension**:
- Completes the versatility spectrum
- Enables comprehensive patient characterization
- Natural clinical workflow (analyze structure, then diffusion, then integrate)
- High clinical value (7/10 multi-turn benefit)

---

## Use Case Example

### Input JSON (Scenario 2: Single Subject, Multi-Modality)

```json
{
    "subject_id": "sub-001",
    "task_id": "comprehensive_assessment",
    "modality_paths": {
        "image_sMRI": "/path/to/sub-001_T1w.nii.gz",
        "image_dMRI": "/path/to/sub-001_FA.nii.gz"
    },
    "conversations": [
        {
            "from": "human",
            "value": "Analyze the structural T1 anatomy:\n<image_sMRI>"
        },
        {
            "from": "gpt",
            "value": "Structural analysis: Normal brain with well-preserved cortical thickness..."
        },
        {
            "from": "human",
            "value": "Now examine white matter integrity:\n<image_dMRI>\nIntegrate with the T1 findings."
        },
        {
            "from": "gpt",
            "value": "White matter analysis: FA values are normal in major tracts. Integrated assessment shows normal structural and microstructural integrity..."
        }
    ],
    "metadata": {
        "age": 45,
        "sex": 1,
        "diagnosis": "healthy_control"
    }
}
```

### Clinical Workflow

```
Step 1: Radiologist examines T1
        → Assesses cortical thickness, ventricular size, any atrophy
        → Generates structural description

Step 2: Radiologist examines FA map (dMRI)
        → Assesses white matter integrity
        → Uses T1 findings as reference
        → Generates integrated analysis

Result: Comprehensive structural + microstructural assessment
```

---

## Design Specification

### Data Flow

```
JSON with multi-modality paths
        ↓
isinstance(subject_id, list)?
├─ YES → _get_multi_subject_sequential() [EXISTING]
└─ NO  → len(modality_paths) > 1?
         ├─ YES → _get_multi_modality_sequential() [NEW]
         └─ NO  → _get_single_item() [EXISTING]
```

### New Method: `_get_multi_modality_sequential()`

**Pseudocode**:
```python
def _get_multi_modality_sequential(self, index: int) -> Dict[str, Any]:
    """
    Process single subject with multiple modalities sequentially.

    Each modality appears in separate conversation turn.
    Transformer attention integrates modalities across turns.
    """
    # 1. Extract data
    sample = self.samples[index]
    subject_id = sample.get('subject_id')  # String
    modality_paths = sample.get('modality_paths', {})
    conversations = sample.get('conversations', [])
    metadata = sample.get('metadata', {})

    # 2. Load images (one per modality)
    images = {}  # modality_key → image tensor
    for modality_key in modality_paths:
        if modality_key.startswith('image_'):
            path = modality_paths[modality_key]
            image = self._load_and_process_image(path)
            images[modality_key] = image

    # 3. Format conversation (convert placeholders per modality)
    formatted_inst, formatted_answer = \
        self._format_multi_modality_conversation(conversations, images.keys())

    # 4. Tokenize
    inputs = {
        'pixel_values': images,  # {'T1': img, 'dMRI': img}
        'input_ids': {},
        'attention_mask': {},
        'labels': {},
        'modalities': list(images.keys()),
        'subject_id': subject_id
    }

    if self.tokenizer is not None:
        token_dict = tokenize_conversation(
            formatted_inst, formatted_answer,
            self.tokenizer, self.max_seq_length
        )
        inputs['input_ids'] = token_dict['input_ids']
        inputs['attention_mask'] = token_dict['attention_mask']
        inputs['labels'] = token_dict['labels']

    inputs['task_id'] = sample.get('task_id', '')
    inputs['metadata'] = metadata

    return inputs
```

**Key Differences from Multi-Subject**:
- Images dict keyed by MODALITY, not indexed
- subject_id remains STRING
- Modality names in output for identification
- Placeholder conversion per modality

### Placeholder Conversion Strategy

**Option A: Explicit Modality Placeholders** (Recommended)
```python
# Input conversation:
"<image_T1>\nAnalyze structure."
"<image_dMRI>\nAnalyze diffusion."

# Conversion: Keep as-is (no regex needed)
# Output: Standard image tokens to collator
```

**Option B: Generic Placeholder (Simpler)**
```python
# Input conversation:
"<image_sMRI>\nAnalyze structure."
"<image_dMRI>\nAnalyze diffusion."

# Conversion: None needed
# Output: Standard image tokens to collator
```

**Recommendation**: Option A (explicit placeholders) - clearer intent

### Output Structure

```python
{
    'pixel_values': {
        'T1': torch.Tensor,      # Shape: (1, H, W, D)
        'dMRI': torch.Tensor     # Shape: (1, H, W, D)
    },
    'input_ids': torch.Tensor,
    'attention_mask': torch.Tensor,
    'labels': torch.Tensor,
    'modalities': ['T1', 'dMRI'],
    'subject_id': 'sub-001',
    'task_id': 'comprehensive_assessment',
    'metadata': {...}
}
```

---

## Implementation Steps

### Step 1: Add Scenario Classifier (5 min)

```python
def _classify_scenario(self, sample: Dict) -> str:
    """Classify sample scenario for routing."""
    subject_id = sample.get('subject_id')
    modality_paths = sample.get('modality_paths', {})

    num_subjects = len(subject_id) if isinstance(subject_id, list) else 1
    num_modalities = len([k for k in modality_paths.keys()
                          if k.startswith('image_')])

    if num_subjects > 1:
        return 'multi_subject' if num_modalities == 1 else 'multi_both'
    elif num_modalities > 1:
        return 'multi_modality'
    else:
        return 'single'
```

### Step 2: Update `__getitem__()` Routing (10 min)

```python
def __getitem__(self, index: int) -> Dict[str, Any]:
    """Smart routing based on scenario."""
    sample = self.samples[index]
    scenario = self._classify_scenario(sample)

    if scenario == 'single':
        return self._get_single_item(index)
    elif scenario == 'multi_modality':
        return self._get_multi_modality_sequential(index)  # NEW
    elif scenario == 'multi_subject':
        return self._get_multi_subject_sequential(index)
    elif scenario == 'multi_both':
        return self._get_multi_subject_multi_modality(index)  # Future
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
```

### Step 3: Implement `_get_multi_modality_sequential()` (45 min)

See pseudocode above. Key implementation points:
- Error handling: verify all modalities have images
- Verification: check output shapes
- Tokenization: use existing tokenizer

### Step 4: Implement `_format_multi_modality_conversation()` (15 min)

```python
def _format_multi_modality_conversation(
    self,
    conversations: List[Dict[str, str]],
    modality_keys: List[str]
) -> Tuple[str, str]:
    """Format multi-modality conversation."""
    instruction_parts = []
    answer = ""

    modality_mapping = {f'image_{k}': f'<image_{k}>'
                        for k in modality_keys}

    for turn in conversations:
        role = turn.get('from', '').lower()
        value = turn.get('value', '')

        # Replace image paths with tokens
        for modality_key, token in modality_mapping.items():
            value = value.replace(modality_key, token)

        if role == 'human':
            instruction_parts.append(value)
        elif role in ['gpt', 'assistant']:
            answer = value

    instruction = " ".join(instruction_parts)
    return instruction, answer
```

### Step 5: Add Tests (45 min)

Create `test_multi_modality_dataset.py`:

```python
class TestMultiModalitySequential(unittest.TestCase):

    def test_multi_modality_loading(self):
        """Test loading multiple modalities for single subject."""
        # Load T1 + dMRI
        # Verify both images in output
        # Verify shapes
        pass

    def test_multi_modality_conversation_format(self):
        """Test conversation formatting with multiple modalities."""
        # Input with <image_T1> and <image_dMRI>
        # Verify placeholders preserved/converted correctly
        pass

    def test_multi_modality_end_to_end(self):
        """Test complete multi-modality pipeline."""
        # Full JSON to model input validation
        pass

    def test_modality_count_validation(self):
        """Test error handling for modality mismatches."""
        pass
```

---

## Collator & Trainer Compatibility

### Good News: Minimal Changes Needed

**Collator (`data.py`)**:
- Currently handles dict of images: `pixel_values[modality_key] = image`
- Multi-modality follows same pattern
- No changes needed ✅

**Trainer (`Trainer.py`)**:
- Loss computation works with modality dict
- Dummy loss mechanism handles multiple modalities
- No changes needed ✅

**Model**:
- Multi-modality follows same tokenization path
- Transformer handles multi-turn naturally
- No changes needed ✅

---

## Performance Characteristics

| Aspect | Scenario 1 (Single) | Scenario 2 (Multi-Mod) | Scenario 3 (Multi-Subj) |
|--------|---------------------|------------------------|-------------------------|
| Memory | Minimal | ~2x (two images) | ~2x (two images) |
| Computation | Baseline | Similar | Similar |
| Tokens | Baseline | +50% (two turns) | +50% (two turns) |
| Attention Overhead | Baseline | Moderate | Justified |

---

## Testing Strategy

### Unit Tests
1. Multi-modality format recognition
2. Image loading for all modalities
3. Conversation formatting
4. Error handling (missing modality, path mismatch)

### Integration Tests
1. End-to-end T1 + dMRI pipeline
2. Tokenization with multi-modality
3. Collator batching
4. Trainer forward pass

### Clinical Validation
1. Verify cross-modality attention
2. Generate diagnostic reports
3. Compare single-modality vs. multi-modality performance

---

## Backward Compatibility

✅ **Zero Breaking Changes**

- Existing single-subject code unchanged
- Existing multi-subject code unchanged
- Scenario classification only affects new cases
- All existing tests still pass

---

## Extension to Scenario 4 (Multi-Subject + Multi-Modality)

Example use case: Compare two patients, each with T1 + dMRI

```json
{
    "subject_id": ["sub-patient", "sub-control"],
    "modality_paths": {
        "image_sMRI": ["/patient_T1.nii.gz", "/control_T1.nii.gz"],
        "image_dMRI": ["/patient_FA.nii.gz", "/control_FA.nii.gz"]
    },
    "conversations": [
        {"from": "human", "value": "Control T1:\n<sub1-image_sMRI>"},
        {"from": "gpt", "value": "Normal structure."},
        {"from": "human", "value": "Control dMRI:\n<sub1-image_dMRI>"},
        {"from": "gpt", "value": "Normal diffusion."},
        {"from": "human", "value": "Patient T1:\n<sub2-image_sMRI>"},
        {"from": "gpt", "value": "Abnormal structure."},
        {"from": "human", "value": "Patient dMRI:\n<sub2-image_dMRI>\nCompare everything."},
        {"from": "gpt", "value": "Integrated comparison: Patient shows..."}
    ]
}
```

**Implementation**:
- Add `_get_multi_subject_multi_modality()` (~120 lines)
- Combines logic from both handlers
- Effort: ~3-4 hours (after Scenario 2)

---

## Recommended Timeline

### Week 1
- Implement Scenario 2 (multi-modality) - 2-3 hours
- Add comprehensive tests - 1 hour
- Total: 3-4 hours work

### Week 2-3
- Validate with multi-modality test cases
- Generate diagnostic reports
- Compare performance metrics

### Week 4+
- Plan Scenario 4 implementation (if needed)
- Optimize attention mechanisms
- Publish results

---

## Success Criteria

✅ **Phase 2 Complete When**:
1. `_get_multi_modality_sequential()` implemented
2. `_classify_scenario()` working correctly
3. All 5+ unit tests passing
4. Backward compatibility verified
5. Integration tests passing
6. Documentation complete

---

## Summary

**Current State**: Scenario 3 (multi-subject) ✅ COMPLETE

**Next Step**: Scenario 2 (multi-modality) - READY FOR IMPLEMENTATION

**Benefits When Complete**:
- Comprehensive patient assessment (structure + diffusion)
- Cross-modality integration via attention
- Natural clinical workflow support
- High medical value

**Effort Required**: 3-4 hours total

**Breaking Changes**: Zero

**Ready to Proceed?** YES - Design complete, ready for coding.

---

**Status**: Design Complete ✅
**Next Action**: Implement Phase 2
**Estimated Completion**: 1-2 weeks with testing
**Priority**: Medium (high clinical value, moderate complexity)
