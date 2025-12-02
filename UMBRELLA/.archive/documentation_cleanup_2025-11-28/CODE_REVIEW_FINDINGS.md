# UMBRELLA Code Review and Bug Report

**Date**: 2025-11-28
**Reviewer**: Supervisor Agent
**Scope**: Complete UMBRELLA training system with focus on JSON format handling and tokenization

---

## Executive Summary

**Critical Issues Found**: 5
**Status**: Dataset and tokenization code requires significant updates to match JSON v2 format
**Priority**: HIGH - Current code will fail with production JSON files

### Key Findings

1. **CRITICAL**: `_tokenize_conversation()` generates incorrect LLaVA-Next prompt format
2. **CRITICAL**: Role detection logic incomplete for JSON v2 format
3. **CRITICAL**: Image token handling doesn't match specification
4. **HIGH**: Image size configuration expects lists but code assumes single int
5. **MEDIUM**: Main training script missing (file not found at expected path)

---

## Detailed Analysis

### 1. Configuration File Review

**File**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/config/umbrella_llava_train.yaml`

**Status**: ✅ CORRECT

**Findings**:
```yaml
# Image sizes are correctly specified as LISTS
T1:
  img_size: [120, 120, 120]  # 3D sMRI

rsfMRI:
  img_size: [96, 96, 96, 24]  # 4D fMRI with temporal dimension
  sequence_length: 24
```

**Observations**:
- Correctly acknowledges non-isomorphic MRI images
- T1 (sMRI): 3D spatial dimensions [H, W, D]
- rsfMRI: 4D with temporal dimension [H, W, D, T]
- Configuration is well-structured and follows specification

---

### 2. Sample JSON Format Analysis

**File**: `sample_data/sex_comparison_conversations/test/NDARINVD9ZEJ36L_different_sex_comparison.json`

**Status**: ✅ CORRECT - Matches JSON v2 specification

**Format Validation**:
```json
{
  "conversations": [
    {
      "role": "user",  // ✅ Lowercase "user"
      "content": [     // ✅ Array of content items
        {
          "type": "text",  // ✅ Type-based structure
          "text": "..."
        },
        {
          "type": "image",  // ✅ Image content item
          "modality": "sMRI",
          "image_path": "..."
        }
      ]
    },
    {
      "role": "assistant",  // ✅ Lowercase "assistant"
      "content": [
        {"type": "text", "text": "..."}
      ]
    }
  ]
}
```

**Key Observations**:
- ✅ Uses lowercase "user" and "assistant" roles
- ✅ Content is array of type/text/image dicts
- ✅ Generic `<image>` tokens in images array
- ✅ Proper multi-turn conversation structure

---

### 3. UMBRELLADataset Critical Bugs

**File**: `project/dataset/umbrella_dataset.py`

#### Bug 3.1: Incorrect Role Normalization (CRITICAL)

**Location**: Lines 116-122

**Current Code**:
```python
ROLE_MAPPING = {
    'user': 'human',      # New format -> internal
    'assistant': 'gpt',   # New format -> internal
    'human': 'human',     # Backward compatibility
    'gpt': 'gpt'         # Backward compatibility
}
```

**Issue**: Maps "user" → "human" and "assistant" → "gpt", then tokenization uses "human"/"gpt" for format generation.

**Impact**:
- Creates intermediate format that doesn't match LLaVA-Next expectations
- Tokenization then converts back, adding unnecessary complexity
- Risk of bugs during role detection in `_tokenize_conversation()`

**Recommendation**:
- Keep roles as "user"/"assistant" throughout pipeline
- Only convert to LLaVA-Next format during final tokenization
- Remove intermediate "human"/"gpt" conversion

---

#### Bug 3.2: _tokenize_conversation() Generates Wrong Format (CRITICAL)

**Location**: Lines 356-428

**Current Code**:
```python
def _tokenize_conversation(self, sample: UMBRELLASample) -> Dict[str, torch.Tensor]:
    full_text = ""
    for turn in sample.conversation:
        if turn.role == 'human':
            turn_text = f"USER: {turn.content}\n"  # ❌ WRONG
        else:  # gpt
            turn_text = f"ASSISTANT: {turn.content}\n"  # ❌ WRONG
        full_text += turn_text
```

**Issues**:
1. ❌ Generates "USER: ... ASSISTANT: ..." format (not LLaVA-Next)
2. ❌ Missing `<|im_start|>` and `<|im_end|>` special tokens
3. ❌ Role names in uppercase instead of lowercase
4. ❌ Doesn't handle image token placement properly
5. ❌ Fallback masking uses "ASSISTANT:" pattern detection (incorrect)

**Expected LLaVA-Next Format**:
```
<|im_start|>user <image><image>
Compare these brain scans.<|im_end|><|im_start|>assistant
Based on comparison...<|im_end|><|im_start|>user
What about...?<|im_end|><|im_start|>assistant
```

**Current Output**:
```
USER: <image><image>
Compare these brain scans.
ASSISTANT: Based on comparison...
USER: What about...?
ASSISTANT:
```

**Impact**:
- Model trained with wrong format will not work with LLaVA-Next inference
- Incompatible with pre-trained LLaVA checkpoints
- Loss masking may be incorrect due to wrong pattern detection

---

#### Bug 3.3: _parse_content() Partially Correct (MEDIUM)

**Location**: Lines 217-259

**Current Code**:
```python
def _parse_content(self, content_raw: Union[str, List[Dict]]) -> Tuple[str, List[str]]:
    if isinstance(content_raw, list):
        text_parts = []
        image_tokens = []
        for item in content_raw:
            if item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
            elif item.get('type') == 'image':
                image_token = '<image>'  # ✅ Correct generic token
                image_tokens.append(image_token)
                text_parts.append(image_token)
        text_content = '\n'.join(text_parts)  # ⚠️ Joins with newlines
        return text_content, image_tokens
```

**Issues**:
1. ⚠️ Joins text parts with `\n` - should preserve original spacing
2. ✅ Correctly uses generic `<image>` token
3. ✅ Handles both legacy and new formats

**Impact**: Minor - newline insertion may affect token positions but generally acceptable

---

#### Bug 3.4: Image Size Handling (HIGH)

**Location**: Lines 128, 276-295

**Current Code**:
```python
def __init__(self, ..., img_size: int = 128, ...):  # ❌ Takes single int
    self.img_size = img_size

def _define_image_transforms(self):
    img_size = (self.img_size, self.img_size, self.img_size)  # ❌ Assumes isomorphic
```

**Issue**:
- Config specifies `img_size: [120, 120, 120]` (list)
- Dataset expects single integer
- Doesn't support 4D fMRI with temporal dimension `[96, 96, 96, 24]`

**Impact**:
- Will fail when loading from config YAML
- Cannot handle variable-size images per modality
- fMRI temporal dimension completely ignored

**Required Changes**:
```python
def __init__(self, ..., img_size: Union[int, List[int]] = 128, ...):
    if isinstance(img_size, int):
        self.img_size = [img_size, img_size, img_size]
    else:
        self.img_size = img_size  # Support list from config

def _define_image_transforms(self):
    img_size = tuple(self.img_size)  # Use list directly
```

---

### 4. UMBRELLACollator Review

**File**: `project/dataset/umbrella_collator.py`

**Status**: ✅ MOSTLY CORRECT

**Findings**:
- ✅ Properly handles variable number of images per sample
- ✅ Correct padding logic for images
- ✅ Attention mask and label handling looks correct
- ✅ Memory-aware batching implemented correctly
- ⚠️ Assumes images are already correctly tokenized (depends on dataset fix)

**No Critical Issues Found** - Collator will work correctly once dataset is fixed.

---

### 5. Main Training Script Review

**File**: `project/training/main_umbrella_training_integrated.py`

**Status**: ⚠️ INCORRECT IMPORT PATHS

**Issues**:

**Lines 38-52**:
```python
try:
    from umbrella_dataset import UMBRELLADataset  # ❌ Wrong path
    from umbrella_collator import UMBRELLACollator  # ❌ Wrong path
```

**Correct Import Paths**:
```python
from project.dataset.umbrella_dataset import UMBRELLADataset
from project.dataset.umbrella_collator import UMBRELLACollator
```

**Missing Components**:
- No YAML config loading logic
- Hardcoded `img_size=128` instead of reading from config
- Missing modality-specific image size handling
- No integration with config's list-based image sizes

---

## Summary of Required Changes

### Priority 1: CRITICAL (Must Fix Before Training)

1. **Fix `_tokenize_conversation()` in umbrella_dataset.py**
   - Generate proper LLaVA-Next format with `<|im_start|>` and `<|im_end|>`
   - Use lowercase "user" and "assistant"
   - Proper image token insertion
   - Fix masking pattern detection

2. **Fix Role Handling**
   - Remove intermediate "human"/"gpt" conversion
   - Keep "user"/"assistant" throughout pipeline
   - Update ConversationTurn dataclass

3. **Fix Image Size Handling**
   - Support list-based image sizes from config
   - Handle 3D vs 4D images (sMRI vs fMRI)
   - Per-modality image dimensions

### Priority 2: HIGH (Important for Integration)

4. **Update main_umbrella_training_integrated.py**
   - Fix import paths
   - Load from YAML config
   - Support modality-specific image sizes
   - Initialize dataset with correct parameters

5. **Create Shell Scripts**
   - `train_with_samples_local.sh`
   - `train_with_samples_ddp.sh`
   - Clear parameter documentation

### Priority 3: MEDIUM (Quality Improvements)

6. **Update _parse_content()**
   - Better text part joining (preserve original spacing)
   - Add validation for required fields

7. **Add Validation Tests**
   - Test tokenization output format
   - Verify image loading with config sizes
   - Test batch collation

---

## Test Cases Required

### Test 1: Tokenization Format
```python
sample = load_sample("NDARINVD9ZEJ36L_different_sex_comparison.json")
result = dataset._tokenize_conversation(sample)
decoded = tokenizer.decode(result['input_ids'])

assert "<|im_start|>user" in decoded
assert "<|im_end|>" in decoded
assert "USER:" not in decoded  # Old format should not appear
assert "ASSISTANT:" not in decoded
```

### Test 2: Image Size from Config
```python
config = load_yaml("umbrella_llava_train.yaml")
dataset = UMBRELLADataset(..., img_size=config['dataset']['T1']['img_size'])
assert dataset.img_size == [120, 120, 120]
```

### Test 3: Multi-Turn Conversation
```python
# Load multi-turn sample
sample = load_multiturn_sample()
result = dataset._tokenize_conversation(sample)

# Verify multiple <|im_start|> and <|im_end|> pairs
decoded = tokenizer.decode(result['input_ids'])
assert decoded.count("<|im_start|>") >= 4  # At least 2 turns each direction
```

---

## Recommendations

1. **Immediate Actions**:
   - Fix `_tokenize_conversation()` to generate correct LLaVA-Next format
   - Update image size handling to support config lists
   - Fix import paths in main training script

2. **Testing Strategy**:
   - Create unit tests for tokenization
   - Test with actual sample JSON files
   - Validate output format matches LLaVA-Next expectations

3. **Documentation**:
   - Create TOKENIZATION_GUIDE.md explaining conversion process
   - Document image size configuration requirements
   - Add examples of expected input/output formats

4. **Future Enhancements**:
   - Add format validation at dataset load time
   - Implement automatic format detection and conversion
   - Add detailed logging for debugging tokenization issues

---

## Conclusion

The UMBRELLA codebase has a solid architecture but requires critical updates to:
1. Generate correct LLaVA-Next prompt format
2. Handle JSON v2 format properly
3. Support variable-size MRI images from config

Once these fixes are implemented, the system will be ready for production training with the provided sample data.

**Estimated Fix Time**: 3-4 hours
**Testing Time**: 2 hours
**Total**: ~6 hours for complete update and validation
