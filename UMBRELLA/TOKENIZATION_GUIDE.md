### UMBRELLA Tokenization Guide

**Purpose**: Explain how UMBRELLA converts JSON v2 conversations to LLaVA-Next format

---

## Overview

UMBRELLA uses a two-step conversion process:

1. **JSON v2 Parsing**: Extract conversations from structured JSON
2. **LLaVA-Next Formatting**: Convert to model-compatible prompt format

---

## Step 1: JSON v2 Format (Input)

**Standard Format**:
```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Compare these brain scans."},
        {"type": "image", "modality": "sMRI", "image_path": "/path/to/scan1.nii.gz"},
        {"type": "image", "modality": "sMRI", "image_path": "/path/to/scan2.nii.gz"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Based on comparison..."}
      ]
    }
  ]
}
```

**Key Properties**:
- ✅ Roles: `"user"` and `"assistant"` (lowercase only)
- ✅ Content: Array of type-based items
- ✅ Types: `"text"` and `"image"`
- ✅ Image tokens: Generic `<image>` for all modalities

**Invalid Formats (DO NOT USE)**:
```json
// ❌ WRONG: Old role names
{"role": "human", ...}      // Should be "user"
{"role": "gpt", ...}        // Should be "assistant"

// ❌ WRONG: Content as string
{"role": "user", "content": "Text here"}  // Should be array

// ❌ WRONG: Modality-specific tokens
{"type": "image", "token": "<image_sMRI>"}  // Should use generic <image>
```

---

## Step 2: Intermediate Representation

The dataset parses JSON into `ConversationTurn` objects:

```python
@dataclass
class ConversationTurn:
    role: str  # 'user' or 'assistant'
    content: str  # Combined text with <image> tokens
    image_tokens: List[str]  # List of '<image>' tokens
```

**Example Parsing**:

**Input JSON**:
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Compare these scans."},
    {"type": "image", "modality": "sMRI"},
    {"type": "image", "modality": "sMRI"}
  ]
}
```

**Output ConversationTurn**:
```python
ConversationTurn(
    role='user',
    content='Compare these scans. <image> <image>',
    image_tokens=['<image>', '<image>']
)
```

---

## Step 3: LLaVA-Next Format (Output)

**Target Format**:
```
<|im_start|>user <image><image>
Compare these scans.<|im_end|><|im_start|>assistant
Based on comparison...<|im_end|>
```

**Format Rules**:

1. **Turn Start**: `<|im_start|>{role}`
2. **Content**: Text and `<image>` tokens (no extra spacing)
3. **Turn End**: `<|im_end|>`
4. **Multi-turn**: Chain turns sequentially

**Example Multi-Turn**:
```
<|im_start|>user <image>
Analyze this brain scan.<|im_end|><|im_start|>assistant
This is a T1-weighted MRI showing normal structure.<|im_end|><|im_start|>user
What about age?<|im_end|><|im_start|>assistant
Based on cortical thickness, approximately 35-40 years old.<|im_end|>
```

---

## Conversion Function: `_tokenize_conversation()`

**Location**: `umbrella_dataset_fixed.py`, line ~450

**Implementation**:

```python
def _tokenize_conversation(self, sample: UMBRELLASample) -> Dict[str, torch.Tensor]:
    """
    Convert conversation to LLaVA-Next format and tokenize.

    Steps:
    1. Build conversation string with <|im_start|> and <|im_end|>
    2. Tokenize full conversation
    3. Create labels with proper masking
    """
    conversation_parts = []
    turn_boundaries = []

    for turn in sample.conversation:
        turn_start = len(conversation_parts)

        # Add turn markers and content
        conversation_parts.append(f"<|im_start|>{turn.role}")
        conversation_parts.append(turn.content)
        conversation_parts.append("<|im_end|>")

        turn_end = len(conversation_parts)
        turn_boundaries.append((turn_start, turn_end, turn.role))

    # Join all parts
    full_text = ''.join(conversation_parts)

    # Tokenize
    encoding = self.tokenizer(
        full_text,
        add_special_tokens=True,
        padding='max_length',
        max_length=self.max_seq_length,
        truncation=True,
        return_tensors='pt',
        return_offsets_mapping=True
    )

    input_ids = encoding['input_ids'].squeeze(0)
    attention_mask = encoding['attention_mask'].squeeze(0)
    labels = input_ids.clone()

    # Mask padding
    labels[attention_mask == 0] = -100

    # Mask user turns (only assistant contributes to loss)
    for part_start, part_end, role in turn_boundaries:
        if role == 'user':
            # Calculate character positions and mask corresponding tokens
            # [Implementation details...]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
```

---

## Label Masking Strategy

**Purpose**: Train model to generate assistant responses, not user queries

**Rules**:
1. **User turns**: Masked with `-100` (ignored in loss)
2. **Assistant turns**: Active (contribute to loss)
3. **Padding**: Masked with `-100`

**Example**:

**Input IDs**:
```
[<|im_start|>, user, <image>, Analyze, scan, <|im_end|>, <|im_start|>, assistant, T1, MRI, shows, ..., <|im_end|>, <pad>, <pad>]
```

**Labels** (for loss computation):
```
[-100, -100, -100, -100, -100, -100, -100, assistant, T1, MRI, shows, ..., <|im_end|>, -100, -100]
```

**Explanation**:
- User turn (`<|im_start|>user ... <|im_end|>`): All `-100` (masked)
- Assistant turn (`<|im_start|>assistant ... <|im_end|>`): Active (original tokens)
- Padding: `-100` (masked)

---

## Image Token Handling

**Generic Token**: `<image>` for ALL modalities

**Why Generic?**
- Modality information encoded in metadata
- Model learns modality from image features, not token type
- Simplifies tokenization and vocabulary

**Image Embedding Replacement**:

During forward pass, the model:
1. Identifies `<image>` token positions in `input_ids`
2. Replaces token embeddings with encoded image features
3. Processes combined text+image sequence

**Example**:

**Before Embedding Replacement**:
```
Token IDs: [<|im_start|>, user, <image>, <image>, Compare, scans, <|im_end|>]
```

**After Embedding Replacement**:
```
Embeddings: [embed_start, embed_user, IMAGE_FEATURES_1, IMAGE_FEATURES_2, embed_compare, embed_scans, embed_end]
```

Where `IMAGE_FEATURES_i` are the encoded brain scan features from the vision encoder.

---

## Complete Example: End-to-End

### Input JSON:
```json
{
  "task_id": "sex_comparison_001",
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Reference scan (female):"},
        {"type": "image", "modality": "sMRI"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Reference scan received."}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Compare with this scan:"},
        {"type": "image", "modality": "sMRI"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "This appears to be a male subject."}
      ]
    }
  ]
}
```

### Intermediate Representation:
```python
[
    ConversationTurn(role='user', content='Reference scan (female): <image>', image_tokens=['<image>']),
    ConversationTurn(role='assistant', content='Reference scan received.', image_tokens=[]),
    ConversationTurn(role='user', content='Compare with this scan: <image>', image_tokens=['<image>']),
    ConversationTurn(role='assistant', content='This appears to be a male subject.', image_tokens=[])
]
```

### LLaVA-Next Format:
```
<|im_start|>user <image>
Reference scan (female):<|im_end|><|im_start|>assistant
Reference scan received.<|im_end|><|im_start|>user <image>
Compare with this scan:<|im_end|><|im_start|>assistant
This appears to be a male subject.<|im_end|>
```

### Tokenized (with masking):
```
input_ids: [token_ids for the above text]
labels: [-100 for user turns, token_ids for assistant turns, -100 for padding]
```

---

## Testing Tokenization

### Test 1: Format Validation
```python
from dataset.umbrella_dataset_fixed import UMBRELLADataset

dataset = UMBRELLADataset(
    json_path='sample_data/test.json',
    tokenizer=tokenizer,
    img_size=[120, 120, 120]
)

sample = dataset[0]
decoded = tokenizer.decode(sample['input_ids'])

# Assertions
assert '<|im_start|>user' in decoded
assert '<|im_end|>' in decoded
assert 'USER:' not in decoded  # Old format should NOT appear
assert 'ASSISTANT:' not in decoded
```

### Test 2: Label Masking
```python
sample = dataset[0]
labels = sample['labels']
input_ids = sample['input_ids']

# Find assistant tokens
assistant_token_id = tokenizer.encode('<|im_start|>assistant', add_special_tokens=False)[0]
positions = (input_ids == assistant_token_id).nonzero()

# After each assistant start, labels should NOT be -100 (unmasked)
for pos in positions:
    assert labels[pos + 1] != -100  # First token after <|im_start|>assistant should be active
```

### Test 3: Image Token Placement
```python
sample = dataset[0]
decoded = tokenizer.decode(sample['input_ids'])

# Count image tokens
num_image_tokens = decoded.count('<image>')
expected_images = len(dataset.samples[0].image_paths)

assert num_image_tokens == expected_images
```

---

## Common Issues and Fixes

### Issue 1: Wrong Role Names
**Symptom**: "human" or "gpt" in decoded text
**Fix**: Ensure JSON uses "user" and "assistant"

### Issue 2: Missing Special Tokens
**Symptom**: Text looks normal but no `<|im_start|>` or `<|im_end|>`
**Fix**: Check tokenizer has these tokens in vocabulary

### Issue 3: All Labels are -100
**Symptom**: Training loss is 0 or NaN
**Fix**: Verify masking logic only masks user turns, not assistant

### Issue 4: Image Tokens Not Replaced
**Symptom**: Model sees `<image>` as text token
**Fix**: Ensure model's forward pass replaces image tokens with embeddings

---

## References

- **LLaVA-Next Paper**: [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)
- **LLaVA-Next GitHub**: [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **HuggingFace LLaVA**: [llava-hf models](https://huggingface.co/llava-hf)

---

## Summary

| Step | Input | Output |
|------|-------|--------|
| **1. JSON Parsing** | JSON v2 with role/content | ConversationTurn objects |
| **2. Format Conversion** | ConversationTurn objects | LLaVA-Next prompt string |
| **3. Tokenization** | Prompt string | input_ids, attention_mask, labels |
| **4. Masking** | labels | User turns = -100, Assistant turns = active |

**Key Takeaway**: The conversion ensures compatibility with pre-trained LLaVA-Next models while supporting multi-turn conversations with interleaved brain imaging and text.
