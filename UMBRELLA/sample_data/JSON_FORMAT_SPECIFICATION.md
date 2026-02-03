# JSON Format  Specification

## Overview

Primary Version.0 of the UMBRELLA JSON conversation format is designed for compatibility with LLaVA-style vision-language models that use generic `<image>` tokens for image inputs.

**Key Design Principles:**
1. Use generic `<image>` tokens (NOT modality-specific tokens)
2. Preserve modality metadata for data loading
3. Support multi-turn conversations with interleaved images and text
4. Include comprehensive metadata for task tracking
5. Maintain compatibility with LLaVA tokenization

---

## Complete Format Specification

```json
{
    "task_id": "string - unique identifier for this task",
    "task_type": "string - task category (e.g., T3 for multi-turn comparison)",
    "subject_ids": ["array", "of", "subject", "IDs"],
    "modalities": ["array", "of", "modalities", "for", "each", "image"],
    "images": [
        {
            "path": "string - absolute or relative path to image file",
            "token": "<image> - generic image token",
            "modality": "string - sMRI, dMRI, or fMRI"
        }
    ],
    "conversations": [
        {
            "role": "string - user or assistant (lowercase)",
            "content": [
                {
                    "type": "text",
                    "text": "string - text content"
                },
                {
                    "type": "image",
                    "modality": "string - sMRI, dMRI, or fMRI",
                    "image_path": "string - path to image file"
                }
            ]
        }
    ],
    "metadata": {
        "subject_id": "string - primary subject ID",
        "subject_label": "string - classification label",
        "reference_id": "string - reference subject ID (if applicable)",
        "reference_label": "string - reference label (if applicable)",
        "comparison_type": "string - same or different",
        "task": "string - task description"
    }
}
```

---

## Field Descriptions

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | string | ✓ | Unique identifier for the task (e.g., "NDARINV007W6H7B_same_sex_comparison") |
| `task_type` | string | ✓ | Task category identifier (e.g., "T3" for multi-turn comparison) |
| `subject_ids` | array[string] | ✓ | List of all subject IDs involved in the task |
| `modalities` | array[string] | ✓ | List of modalities corresponding to each image |
| `images` | array[object] | ✓ | Array of image metadata objects |
| `conversations` | array[object] | ✓ | Multi-turn conversation array |
| `metadata` | object | ✓ | Task-specific metadata |

### Images Array

Each image object contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | ✓ | Absolute or relative path to .nii.gz file |
| `token` | string | ✓ | Must be exactly `"<image>"` (generic token) |
| `modality` | string | ✓ | Image modality: "sMRI", "dMRI", or "fMRI" |

**Important:** The `token` field MUST be `"<image>"`, NOT modality-specific tokens like `"<image_sMRI>"`.

### Conversations Array

Each conversation turn contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | ✓ | Must be "user" or "assistant" (lowercase) |
| `content` | array[object] | ✓ | Array of content items (text and/or images) |

#### Content Items

**Text content:**
```json
{
    "type": "text",
    "text": "string - the actual text content"
}
```

**Image content:**
```json
{
    "type": "image",
    "modality": "sMRI",
    "image_path": "/path/to/image.nii.gz"
}
```

**Important:** The `type` field MUST be `"image"`, NOT `"image_sMRI"` or other modality-specific types.

### Metadata Object

Recommended fields (task-specific):

| Field | Type | Description |
|-------|------|-------------|
| `subject_id` | string | Primary subject identifier |
| `subject_label` | string | Classification label (e.g., "male", "female") |
| `reference_id` | string | Reference subject ID (for comparison tasks) |
| `reference_label` | string | Reference subject label |
| `comparison_type` | string | "same" or "different" |
| `task` | string | Task description |

---

## Complete Example

```json
{
    "task_id": "NDARINV007W6H7B_same_sex_comparison",
    "task_type": "T3",
    "subject_ids": [
        "NDARINV00BD7VDC",
        "NDARINV007W6H7B"
    ],
    "modalities": [
        "sMRI",
        "sMRI"
    ],
    "images": [
        {
            "path": "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256/NDARINV00BD7VDC.nii.gz",
            "token": "<image>",
            "modality": "sMRI"
        },
        {
            "path": "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256/NDARINV007W6H7B.nii.gz",
            "token": "<image>",
            "modality": "sMRI"
        }
    ],
    "conversations": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is a T1-weighted structural MRI scan from a male subject (reference). Please analyze this brain scan carefully."
                },
                {
                    "type": "image",
                    "modality": "sMRI",
                    "image_path": "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256/NDARINV00BD7VDC.nii.gz"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I've examined the male reference scan. Ready for comparison."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this scan relative to the reference. Classify the biological sex and explain key differences."
                },
                {
                    "type": "image",
                    "modality": "sMRI",
                    "image_path": "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256/NDARINV007W6H7B.nii.gz"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "This scan likely belongs to a male subject. Key features align with the reference male scan."
                }
            ]
        }
    ],
    "metadata": {
        "subject_id": "NDARINV007W6H7B",
        "subject_label": "male",
        "reference_id": "NDARINV00BD7VDC",
        "reference_label": "male",
        "comparison_type": "same",
        "task": "sex_classification_via_comparison"
    }
}
```

---

## Tokenization Process

The JSON conversations are converted to LLaVA format as follows:

### Input (JSON)
```json
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Analyze this scan."},
        {"type": "image", "modality": "sMRI", "image_path": "..."}
    ]
}
```

### Output (LLaVA Format)
```
<|im_start|>user Analyze this scan.
<image><|im_end|>
```

### Complete Multi-Turn Example

**Input (multiple turns):**
```json
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "This is a reference scan."},
            {"type": "image", "modality": "sMRI", "image_path": "..."}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Understood."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare with this scan."},
            {"type": "image", "modality": "sMRI", "image_path": "..."}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "This is similar to the reference."}
        ]
    }
]
```

**Output (tokenized):**
```
<|im_start|>user This is a reference scan.
<image><|im_end|><|im_start|>assistant Understood.<|im_end|><|im_start|>user Compare with this scan.
<image><|im_end|><|im_start|>assistant This is similar to the reference.<|im_end|>
```

---

## Key Differences from V1

| Aspect | V1 Format |  Format |
|--------|-----------|-----------|
| **Image tokens** | `<image_sMRI>` | `<image>` (generic) |
| **Content type** | `"image_sMRI"` | `"image"` |
| **Top-level fields** | `id`, `conversations`, `metadata` | `task_id`, `task_type`, `subject_ids`, `modalities`, `images`, `conversations`, `metadata` |
| **Images array** | Not present | Required with path/token/modality |
| **Metadata labels** | `subject_sex`, `reference_sex` | `subject_label`, `reference_label` |
| **Comparison type** | `same_sex`, `different_sex` | `same`, `different` |

---

## Validation Checklist

Use this checklist to validate your JSON files:

- [ ] All required top-level fields present
- [ ] `task_id` is unique and descriptive
- [ ] `task_type` matches task category
- [ ] `subject_ids` array contains all relevant subjects
- [ ] `modalities` array length matches `images` array length
- [ ] All images have `path`, `token`, and `modality` fields
- [ ] All image tokens are exactly `"<image>"`
- [ ] All image paths end with `.nii.gz`
- [ ] All modalities are valid ("sMRI", "dMRI", or "fMRI")
- [ ] All conversation roles are lowercase
- [ ] All content items have `type` field
- [ ] Image content items use `"type": "image"` (NOT `"image_sMRI"`)
- [ ] Number of image references in conversations matches images array length
- [ ] Metadata contains recommended fields

---

## Usage with Dataloaders

The  format is designed to work seamlessly with the provided dataloaders:

```python
from dataloaders_ import UMBRELLADataLoader

# Create dataset
dataset = UMBRELLADataLoader(
    json_dir="path/to/conversations_",
    image_root="/path/to/images",
    split="train",
    tokenizer=tokenizer,
    processor=processor
)

# Get example
example = dataset[0]
# Returns: {input_ids, attention_mask, pixel_values, labels, metadata}
```

See `TRAINING_WITH_JSON_.md` for complete usage instructions.

---

## Migration from V1 to 

To migrate existing V1 JSON files:

1. Add `task_id`, `task_type`, `subject_ids`, `modalities` fields
2. Create `images` array with metadata for each image
3. Change all `"image_sMRI"` types to `"image"`
4. Update metadata field names (`subject_sex` → `subject_label`, etc.)
5. Update comparison_type values (`same_sex` → `same`, `different_sex` → `different`)
6. Validate with `validate_json_format_.py`

---

## Support

For questions or issues with the  format:
- Validation script: `validate_json_format_.py`
- Example generator: `generate_sex_comparison_conversations_.py`
- Sample files: `sex_comparison_conversations_/samples/`

---

**Version:** 2.0
**Date:** 2025-11-25
**Author:** BrainVLM Team
