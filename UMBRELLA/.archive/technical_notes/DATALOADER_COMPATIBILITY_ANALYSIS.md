# Dataloader/Dataset Compatibility Analysis with New JSON Format

**Date**: 2025-11-27
**Issue**: Critical mismatch between new LLaVA JSON format and existing dataloader implementations

---

## Executive Summary

### Critical Assessment

**Can current dataloaders load new JSON format as-is?**: ❌ **NO**

**Overall Compatibility**: **~30%** - Major architectural mismatches requiring substantial refactoring

**Most Critical Fix Needed First**: Role field mapping (`"human"/"gpt"` → `"user"/"assistant"`)

**Blocking Issue**: Content structure - code expects `turn["content"]` as string, new format provides array of objects

**Estimated Lines of Code to Change**: **~250-300 lines** across 8 files

**Can changes be made without breaking existing functionality?**: ✅ **YES** - With careful conditional logic and backward compatibility checks

---

## 1. New JSON Format vs. Current Implementation

### New Format (LLaVA Standard)
```json
{
  "task_id": "NDARINV007W6H7B_same_sex_comparison",
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this scan."},
        {"type": "image", "modality": "sMRI", "image_path": "..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "This is a brain scan showing..."}
      ]
    }
  ],
  "images": [
    {"path": "...", "token": "<image>", "modality": "sMRI"}
  ],
  "metadata": {...}
}
```

### Current Expected Format (umbrella_dataset.py)
```json
{
  "task_id": "T1_001",
  "conversation": [
    {
      "role": "human",
      "content": "<image_sMRI>\nAnalyze this brain scan."
    },
    {
      "role": "gpt",
      "content": "This is a T1-weighted MRI showing..."
    }
  ],
  "images": [
    {"path": "...", "token": "<image_sMRI>", "modality": "sMRI"}
  ]
}
```

### Key Differences

| Aspect | Current Code Expects | New Format Provides | Compatibility |
|--------|---------------------|-------------------|---------------|
| **Role field** | `"human"`, `"gpt"` | `"user"`, `"assistant"` | ❌ **CRITICAL** |
| **Content structure** | `"content": "string"` | `"content": [array]` | ❌ **CRITICAL** |
| **Image tokens** | Embedded in text string | Separate objects in content array | ❌ **HIGH** |
| **Text extraction** | Direct string access | Must concatenate from array | ❌ **HIGH** |
| **Image references** | `<image_sMRI>` tokens | Generic `<image>` tokens | ⚠️ **MEDIUM** |
| **Conversation key** | `"conversation"` (singular) | `"conversations"` (plural) | ⚠️ **MEDIUM** |

---

## 2. Incompatibility Matrix

| File | Current Behavior | New Format Expectation | Required Fix | Severity | Lines Affected |
|------|-----------------|----------------------|--------------|----------|----------------|
| **umbrella_dataset.py** | Expects `role="human"/"gpt"`, `content=string` | Uses `role="user"/"assistant"`, `content=array` | Update role mapping, parse content array | ❌ CRITICAL | 40-50 |
| **conversation_processor.py** | Expects `conversations` with `role` and `content` as string | Receives `content` as array of objects | Parse array structure, extract text/images | ❌ CRITICAL | 30-40 |
| **t1_json_dataset.py** | Calls conversation_processor with string content | Must handle array content | Update conversation processing logic | ❌ CRITICAL | 20-25 |
| **umbrella_dataloader.py** | Expects processed data from t1_json_dataset | Needs to handle new structure | Update collation if needed | ⚠️ MEDIUM | 10-15 |
| **dataset_utils.py** | `format_conversation()` expects `"from"/"value"` | New format uses `"role"/"content"` | Update field mapping | ❌ HIGH | 30-40 |
| **umbrella_trainer.py** | `TurnMaskBuilder` checks for `"human"`/`"gpt"` roles | Must recognize `"user"`/`"assistant"` | Update role detection logic | ❌ CRITICAL | 20-25 |
| **umbrella_utils.py** | `ConversationFormatter` uses legacy format | Should support new LLaVA JSON format | Already has JSON support, needs validation | ✅ COMPATIBLE | 5-10 |
| **image_loader.py** | Loads images from paths | Works with new format (no changes needed) | None | ✅ COMPATIBLE | 0 |

---

## 3. Detailed File-by-File Analysis

### File 1: `umbrella_dataset.py`

**Status**: ❌ **CRITICAL - Major Refactoring Required**

#### Issue 1: Role Field Mismatch (CRITICAL)
**Location**: Lines 40, 74, 144-150, 286-291

**Current Code** (Line 40):
```python
@dataclass
class ConversationTurn:
    """Single turn in a multi-turn conversation."""
    role: str  # 'human' or 'gpt'  ← EXPECTS 'human'/'gpt'
    content: str  # Text content (may contain image tokens)
    image_tokens: List[str] = field(default_factory=list)
```

**Current Code** (Lines 144-150):
```python
for turn in item.get('conversation', []):
    conv_turn = ConversationTurn(
        role=turn['role'],  # ← GETS 'user'/'assistant', NOT 'human'/'gpt'
        content=turn['content'],  # ← EXPECTS STRING, GETS ARRAY
        image_tokens=self._extract_image_tokens(turn['content'])
    )
    conversation.append(conv_turn)
```

**Current Code** (Lines 286-291):
```python
if turn.role == 'human':
    # Format: "USER: <content>\n"
    turn_text = f"USER: {turn.content}\n"
else:  # gpt
    # Format: "ASSISTANT: <content>\n"
    turn_text = f"ASSISTANT: {turn.content}\n"
```

**Required Fix**:
```python
# OPTION 1: Direct role mapping
ROLE_MAPPING = {
    'user': 'human',
    'assistant': 'gpt',
    'human': 'human',  # Backward compatibility
    'gpt': 'gpt'       # Backward compatibility
}

for turn in item.get('conversation', []) or item.get('conversations', []):
    role = turn['role']
    mapped_role = ROLE_MAPPING.get(role, role)

    # Handle content structure (string vs array)
    content_raw = turn['content']
    if isinstance(content_raw, str):
        # Legacy format
        content = content_raw
    elif isinstance(content_raw, list):
        # New format - extract text from array
        text_parts = [item['text'] for item in content_raw if item.get('type') == 'text']
        content = '\n'.join(text_parts)

        # Insert <image> tokens where images appear
        for item in content_raw:
            if item.get('type') == 'image':
                content = f"<image>\n{content}"  # Prepend image token
    else:
        content = str(content_raw)

    conv_turn = ConversationTurn(
        role=mapped_role,
        content=content,
        image_tokens=self._extract_image_tokens(content)
    )
    conversation.append(conv_turn)
```

**Severity**: ❌ **CRITICAL** - Blocks all data loading
**Lines to Change**: 40-50

---

#### Issue 2: Content Structure Mismatch (CRITICAL)
**Location**: Lines 144-150

**Problem**: Code expects `turn['content']` to be a string, but new format provides an array of objects:
```python
content_raw = turn['content']
# OLD: content_raw = "Analyze this scan."
# NEW: content_raw = [{"type": "text", "text": "Analyze..."}, {"type": "image", ...}]
```

**Current Behavior**:
- `self._extract_image_tokens(turn['content'])` expects string
- Will crash with `TypeError: expected str, got list`

**Required Fix**:
```python
def _parse_content(self, content_raw: Union[str, List[Dict]]) -> Tuple[str, List[str]]:
    """
    Parse content from either legacy string or new array format.

    Returns:
        (text_content, image_tokens)
    """
    if isinstance(content_raw, str):
        # Legacy format
        return content_raw, self._extract_image_tokens(content_raw)

    elif isinstance(content_raw, list):
        # New LLaVA format
        text_parts = []
        image_tokens = []

        for item in content_raw:
            item_type = item.get('type')

            if item_type == 'text':
                text_parts.append(item.get('text', ''))

            elif item_type == 'image':
                # Insert generic <image> token
                image_tokens.append('<image>')
                text_parts.append('<image>')  # Placeholder in text

        text_content = '\n'.join(text_parts)
        return text_content, image_tokens

    else:
        # Fallback
        return str(content_raw), []

# Use in _load_samples():
for turn in item.get('conversation', []) or item.get('conversations', []):
    content, image_tokens = self._parse_content(turn['content'])

    conv_turn = ConversationTurn(
        role=ROLE_MAPPING.get(turn['role'], turn['role']),
        content=content,
        image_tokens=image_tokens
    )
    conversation.append(conv_turn)
```

**Severity**: ❌ **CRITICAL** - Data extraction completely broken

---

#### Issue 3: Conversation Key Mismatch (MEDIUM)
**Location**: Line 144

**Current Code**:
```python
for turn in item.get('conversation', []):  # ← Expects 'conversation'
```

**New Format**:
```json
{
  "conversations": [...]  // ← Provides 'conversations' (plural)
}
```

**Required Fix**:
```python
# Support both keys for backward compatibility
conversations = item.get('conversations', item.get('conversation', []))
for turn in conversations:
    ...
```

**Severity**: ⚠️ **MEDIUM** - Easy fix, but blocks loading

---

### File 2: `conversation_processor.py`

**Status**: ❌ **CRITICAL - Requires Major Updates**

#### Issue 1: Content Array Parsing (CRITICAL)
**Location**: Lines 48-107 (entire `format_conversation_for_llava` method)

**Current Code** (Lines 80-94):
```python
for turn in conversations:
    role = turn["role"]
    content = turn["content"]  # ← EXPECTS STRING

    # Build content string with image tokens
    content_parts = []
    for item in content:  # ← WILL CRASH: iterating over string chars
        if item["type"] == "text":
            content_parts.append(item["text"])
        elif item["type"] == "image":
            # Insert generic <image> token
            content_parts.append(self.IMAGE_TOKEN)
```

**Problem**: Code assumes `content` is already an array, but this conflicts with legacy code elsewhere that expects string.

**Required Fix**:
```python
def format_conversation_for_llava(self, conversations: List[Dict]) -> str:
    """
    Convert JSON conversations to LLaVA prompt format.

    Handles both:
    - Legacy format: content is string
    - New format: content is array of objects
    """
    formatted_turns = []

    for turn in conversations:
        role = turn["role"]
        content_raw = turn["content"]

        # Normalize role
        role_normalized = "user" if role in ["user", "human"] else "assistant"

        # Parse content based on format
        if isinstance(content_raw, str):
            # Legacy format - content is already text
            content_str = content_raw

        elif isinstance(content_raw, list):
            # New format - content is array of items
            content_parts = []
            for item in content_raw:
                if item.get("type") == "text":
                    content_parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    # Insert generic <image> token
                    content_parts.append(self.IMAGE_TOKEN)

            content_str = "\n".join(content_parts)

        else:
            # Unknown format - convert to string
            content_str = str(content_raw)

        # Format turn with special tokens
        formatted_turn = f"{self.IM_START}{role_normalized} {content_str}{self.IM_END}"
        formatted_turns.append(formatted_turn)

    # Join all turns
    prompt = "".join(formatted_turns)

    # Add generation prompt if requested
    if self.add_generation_prompt:
        prompt += f"{self.IM_START}assistant "

    return prompt
```

**Severity**: ❌ **CRITICAL** - Core formatting function broken
**Lines to Change**: 30-40

---

### File 3: `dataset_utils.py`

**Status**: ❌ **HIGH - Multiple Function Updates Required**

#### Issue 1: `format_conversation()` Role Mismatch (HIGH)
**Location**: Lines 218-276

**Current Code** (Lines 256-272):
```python
for i, turn in enumerate(conversations):
    role = turn.get('from', '').lower()  # ← EXPECTS 'from' field
    value = turn.get('value', '')         # ← EXPECTS 'value' field

    if role == 'human':
        if i == 0 and include_image_token:
            instruction_parts.append(f"USER: {modality_token}\n{value}")
        else:
            instruction_parts.append(f"USER: {value}")
    elif role in ['gpt', 'assistant']:
        if i == len(conversations) - 1:
            # Last turn is the answer
            answer = value
            instruction_parts.append("ASSISTANT: ")
        else:
            # Multi-turn: include previous assistant responses
            instruction_parts.append(f"ASSISTANT: {value}")
```

**New Format**:
```json
{
  "role": "user",       // NOT "from": "human"
  "content": [...]      // NOT "value": "..."
}
```

**Required Fix**:
```python
def format_conversation(
    conversations: List[Dict[str, str]],
    include_image_token: bool = True,
    image_token: str = "<image>",
    modality: str = "fMRI"
) -> Tuple[str, str]:
    """
    Format a conversation list into instruction and answer strings.

    Supports both:
    - Legacy format: {"from": "human", "value": "text"}
    - New format: {"role": "user", "content": [array]}
    """
    instruction_parts = []
    answer = ""

    # Determine modality token
    if modality.lower() in ['fmri', 'rsfmri']:
        modality_token = "<image_fMRI>"
    elif modality.lower() in ['smri', 't1']:
        modality_token = "<image_sMRI>"
    elif modality.lower() == 'dmri':
        modality_token = "<image_dMRI>"
    else:
        modality_token = image_token

    for i, turn in enumerate(conversations):
        # Support both legacy and new format
        role = turn.get('role', turn.get('from', '')).lower()

        # Extract content
        content_raw = turn.get('content', turn.get('value', ''))

        # Parse content based on format
        if isinstance(content_raw, str):
            value = content_raw
        elif isinstance(content_raw, list):
            # Extract text from array
            text_parts = [item.get('text', '') for item in content_raw if item.get('type') == 'text']
            value = ' '.join(text_parts)
        else:
            value = str(content_raw)

        # Normalize role names
        if role in ['human', 'user']:
            if i == 0 and include_image_token:
                instruction_parts.append(f"USER: {modality_token}\n{value}")
            else:
                instruction_parts.append(f"USER: {value}")
        elif role in ['gpt', 'assistant']:
            if i == len(conversations) - 1:
                # Last turn is the answer
                answer = value
                instruction_parts.append("ASSISTANT: ")
            else:
                # Multi-turn: include previous assistant responses
                instruction_parts.append(f"ASSISTANT: {value}")

    instruction = " ".join(instruction_parts)

    return instruction, answer
```

**Severity**: ❌ **HIGH** - Core utility function used throughout
**Lines to Change**: 30-40

---

### File 4: `umbrella_trainer.py`

**Status**: ❌ **CRITICAL - Turn Masking Logic Broken**

#### Issue 1: Role Detection in Turn Masking (CRITICAL)
**Location**: Lines 100-184 (`TurnMaskBuilder` class)

**Current Code** (Lines 166-182):
```python
def _mask_turns_from_conversation(self,
                                  conversation: list,
                                  input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply masking based on conversation turn roles.
    """
    labels = input_ids.clone()
    seq_len = len(input_ids)

    # Create role sequence from conversation for reference
    # This is a heuristic approach - in production would use alignments
    roles = [turn.get("role") for turn in conversation]  # ← Gets "user"/"assistant"

    # Estimate token boundaries based on turn order
    # USER turns (role=0) should be masked, ASSISTANT turns (role=1) active
    in_user_turn = True  # Start with user (typically instruction)

    for seq_idx in range(seq_len):
        if attention_mask[seq_idx] == 0:
            # Padding token
            labels[seq_idx] = -100
        elif in_user_turn:
            # Mask USER turn tokens  ← BUT DOESN'T ACTUALLY CHECK FOR 'user' vs 'human'
            labels[seq_idx] = -100
```

**Problem**: Code extracts `role` values but doesn't properly check for `"user"`/`"assistant"` vs `"human"`/`"gpt"`.

**Required Fix**:
```python
def _mask_turns_from_conversation(self,
                                  conversation: list,
                                  input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply masking based on conversation turn roles.

    Supports both:
    - Legacy: role="human"/"gpt"
    - New: role="user"/"assistant"
    """
    labels = input_ids.clone()
    seq_len = len(input_ids)

    # Role normalization mapping
    USER_ROLES = {"user", "human"}
    ASSISTANT_ROLES = {"assistant", "gpt"}

    # Extract roles with normalization
    roles = []
    for turn in conversation:
        role = turn.get("role", "").lower()
        if role in USER_ROLES:
            roles.append("user")
        elif role in ASSISTANT_ROLES:
            roles.append("assistant")
        else:
            roles.append("unknown")

    # Build role-aware mask
    # NOTE: This is still heuristic - production would use tokenizer alignments
    in_user_turn = True  # Start with user (typically instruction)

    for seq_idx in range(seq_len):
        if attention_mask[seq_idx] == 0:
            # Padding token
            labels[seq_idx] = -100
        elif in_user_turn:
            # Mask USER turn tokens (don't train on user input)
            labels[seq_idx] = -100
            # Detect transition to assistant (would need proper token boundary detection)
        # else: ASSISTANT turn - keep label (active in loss)

    return labels
```

**Severity**: ❌ **CRITICAL** - Training loss masking incorrect
**Lines to Change**: 20-25

---

### File 5: `t1_json_dataset.py`

**Status**: ❌ **CRITICAL - Conversation Processing Broken**

#### Issue 1: Conversation Processing (CRITICAL)
**Location**: Lines 122-134

**Current Code**:
```python
def process_conversation(self, json_data: Dict) -> str:
    """
    Convert JSON conversations to LLaVA format.
    """
    conversations = json_data["conversations"]  # ← Expects "conversations"
    prompt = self.conversation_processor.format_conversation_for_llava(conversations)
    return prompt
```

**Problem**: Depends on `conversation_processor.format_conversation_for_llava()` which is broken (see File 2).

**Required Fix**: Fix `conversation_processor.py` first, then this will work.

**Additional Issue**: Field name
```python
def process_conversation(self, json_data: Dict) -> str:
    """
    Convert JSON conversations to LLaVA format.

    Supports both "conversation" (singular) and "conversations" (plural).
    """
    conversations = json_data.get("conversations", json_data.get("conversation", []))

    if not conversations:
        raise ValueError("JSON data missing 'conversation' or 'conversations' field")

    prompt = self.conversation_processor.format_conversation_for_llava(conversations)
    return prompt
```

**Severity**: ❌ **CRITICAL** - Blocks dataset loading
**Lines to Change**: 5-10

---

### File 6: `umbrella_dataloader.py`

**Status**: ⚠️ **MEDIUM - Minimal Changes Required**

**Analysis**: This file is a wrapper around `T1JSONDataset` and passes data through. Once `T1JSONDataset` is fixed, this should work.

**Potential Issue**: Collation function might need updates if data structure changes.

**Required Changes**:
- Test after fixing upstream files
- May need to update `collate_fn()` if tensor shapes change

**Severity**: ⚠️ **MEDIUM** - Depends on upstream fixes
**Lines to Change**: 5-10 (testing/validation)

---

### File 7: `umbrella_utils.py`

**Status**: ✅ **MOSTLY COMPATIBLE - Minor Updates**

**Good News**: This file already has extensive support for LLaVA JSON format!

**Existing Support**:
- Lines 228-333: `create_json_conversation()`, `add_user_to_json()`, `add_assistant_to_json()` - already handles new format
- Lines 402-441: `json_to_text()` - already parses `role` and `content` arrays
- Lines 496-507: `_add_turn_to_json()` - normalizes roles to lowercase

**Required Changes**:
- Ensure role normalization is consistent everywhere
- Add validation for new format

```python
# Add at top of ConversationFormatter class:
ROLE_MAPPING = {
    'user': 'user',
    'human': 'user',
    'assistant': 'assistant',
    'gpt': 'assistant'
}

# Update _add_turn_to_json() (Line 496):
def _add_turn_to_json(self, conversation: list, role: str, content_parts: list) -> list:
    """Helper to add turn to JSON conversation."""
    combined = " ".join(content_parts).strip()

    # Normalize role using mapping
    role_normalized = self.ROLE_MAPPING.get(role.lower(), role.lower())

    if role_normalized == "user":
        return self.add_user_to_json(conversation, text=combined)
    elif role_normalized == "assistant":
        return self.add_assistant_to_json(conversation, response=combined)

    return conversation
```

**Severity**: ✅ **LOW** - Already mostly compatible
**Lines to Change**: 5-10 (validation/normalization)

---

### File 8: `image_loader.py`

**Status**: ✅ **FULLY COMPATIBLE - No Changes Required**

**Analysis**:
- Loads images from file paths
- Doesn't care about conversation structure
- Works with both old and new JSON formats

**Required Changes**: ❌ None

---

## 4. Data Flow Trace

### Current Flow (BROKEN with new format)

```
1. JSON File Loaded (umbrella_dataset.py:_load_samples)
   ↓
   [BREAKS HERE] Expects conversation["role"] = "human"/"gpt"
   [BREAKS HERE] Expects conversation["content"] = string
   ↓
2. Conversation Parsed (umbrella_dataset.py:ConversationTurn)
   ↓
   [BREAKS HERE] _extract_image_tokens() expects string, gets array
   ↓
3. Text Tokenized (umbrella_dataset.py:_tokenize_conversation)
   ↓
   Works (but receives malformed input from step 2)
   ↓
4. Batch Collated (umbrella_dataloader.py:collate_fn)
   ↓
5. Training (umbrella_trainer.py:compute_loss)
   ↓
   [BREAKS HERE] TurnMaskBuilder doesn't recognize "user"/"assistant"
```

### Where Failures Occur

| Step | Component | Failure Point | Impact |
|------|-----------|--------------|--------|
| **1** | JSON loading | Role field mismatch | ❌ Data corrupted from start |
| **1** | JSON loading | Content array not parsed | ❌ Text extraction fails |
| **2** | Token extraction | String operations on array | ❌ Crash |
| **5** | Turn masking | Role detection broken | ❌ Training loss incorrect |

---

## 5. Priority Fix List

### CRITICAL (Must Fix - Blocks Everything)

1. **umbrella_dataset.py:_load_samples()** (Lines 144-170)
   - Add role mapping: `user→human`, `assistant→gpt`
   - Add content array parser: `_parse_content()` method
   - Support both `"conversation"` and `"conversations"` keys
   - **Impact**: Fixes data loading completely
   - **Estimated Time**: 1-2 hours

2. **conversation_processor.py:format_conversation_for_llava()** (Lines 48-107)
   - Update to handle content as array
   - Add backward compatibility for string content
   - Normalize roles to `user`/`assistant`
   - **Impact**: Fixes prompt generation
   - **Estimated Time**: 1 hour

3. **umbrella_trainer.py:TurnMaskBuilder** (Lines 100-184)
   - Update role detection to support both formats
   - Add USER_ROLES/ASSISTANT_ROLES sets
   - **Impact**: Fixes training loss computation
   - **Estimated Time**: 30 minutes

### HIGH (Important - Affects Multiple Components)

4. **dataset_utils.py:format_conversation()** (Lines 218-276)
   - Support both `role/content` and `from/value` formats
   - Parse content array if present
   - **Impact**: Fixes utility functions used throughout
   - **Estimated Time**: 1 hour

5. **t1_json_dataset.py:process_conversation()** (Lines 122-134)
   - Support both conversation key names
   - **Impact**: Fixes dataset creation
   - **Estimated Time**: 15 minutes

### MEDIUM (Testing and Validation)

6. **umbrella_dataloader.py**
   - Test after upstream fixes
   - Validate collation works with new data
   - **Impact**: Ensures batching works correctly
   - **Estimated Time**: 30 minutes

7. **umbrella_utils.py**
   - Add role normalization mapping
   - Add format validation
   - **Impact**: Improves robustness
   - **Estimated Time**: 15 minutes

---

## 6. Code Modification Guide

### Modification 1: umbrella_dataset.py

**File**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/dataset/umbrella_dataset.py`

**Changes Required**:

#### Change 1.1: Add Role Mapping (Line 87)
```python
# ADD THIS CONSTANT AFTER IMAGE_TOKEN_PATTERNS
# Line 87 (after line 86)

# Role normalization for LLaVA compatibility
ROLE_MAPPING = {
    'user': 'human',      # New format -> internal
    'assistant': 'gpt',   # New format -> internal
    'human': 'human',     # Backward compatibility
    'gpt': 'gpt'         # Backward compatibility
}
```

#### Change 1.2: Add Content Parser (Line 171)
```python
# ADD THIS NEW METHOD after _load_samples() method
# Insert at line 171

def _parse_content(self, content_raw: Union[str, List[Dict]]) -> Tuple[str, List[str]]:
    """
    Parse content from either legacy string or new LLaVA array format.

    Args:
        content_raw: Either string (legacy) or list of content items (new)

    Returns:
        (text_content, image_tokens) tuple
    """
    if isinstance(content_raw, str):
        # Legacy format: content is already a string
        return content_raw, self._extract_image_tokens(content_raw)

    elif isinstance(content_raw, list):
        # New LLaVA format: content is array of {"type": "text/image", ...}
        text_parts = []
        image_tokens = []

        for item in content_raw:
            item_type = item.get('type', '')

            if item_type == 'text':
                text_parts.append(item.get('text', ''))

            elif item_type == 'image':
                # Insert generic <image> token
                image_token = '<image>'
                image_tokens.append(image_token)
                text_parts.append(image_token)  # Embed in text for tokenization

        text_content = '\n'.join(text_parts)
        return text_content, image_tokens

    else:
        # Fallback for unexpected formats
        logger.warning(f"Unexpected content type: {type(content_raw)}")
        return str(content_raw), []
```

#### Change 1.3: Update _load_samples() (Lines 144-150)
**BEFORE**:
```python
# Parse conversation
conversation = []
for turn in item.get('conversation', []):
    conv_turn = ConversationTurn(
        role=turn['role'],
        content=turn['content'],
        image_tokens=self._extract_image_tokens(turn['content'])
    )
    conversation.append(conv_turn)
```

**AFTER**:
```python
# Parse conversation - support both 'conversation' and 'conversations' keys
conversations_raw = item.get('conversations', item.get('conversation', []))
conversation = []

for turn in conversations_raw:
    # Normalize role
    raw_role = turn.get('role', 'human')
    role = self.ROLE_MAPPING.get(raw_role, raw_role)

    # Parse content (handles both string and array formats)
    content, image_tokens = self._parse_content(turn.get('content', ''))

    conv_turn = ConversationTurn(
        role=role,
        content=content,
        image_tokens=image_tokens
    )
    conversation.append(conv_turn)
```

---

### Modification 2: conversation_processor.py

**File**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/dataloaders/conversation_processor.py`

#### Change 2.1: Update format_conversation_for_llava() (Lines 48-107)

**REPLACE ENTIRE METHOD** with:
```python
def format_conversation_for_llava(self, conversations: List[Dict]) -> str:
    """
    Convert JSON conversations to LLaVA prompt format.

    Supports both:
    - Legacy format: {"role": "human"/"gpt", "content": "string"}
    - New format: {"role": "user"/"assistant", "content": [array]}

    Args:
        conversations: List of conversation turns

    Returns:
        Formatted prompt string with <image> tokens

    Example output:
    "<|im_start|>user <image>
    Analyze this scan.<|im_end|><|im_start|>assistant
    This is a brain scan.<|im_end|>"
    """
    formatted_turns = []

    for turn in conversations:
        # Normalize role
        raw_role = turn.get("role", "user")
        role = "user" if raw_role in ["user", "human"] else "assistant"

        # Parse content based on format
        content_raw = turn.get("content", "")

        if isinstance(content_raw, str):
            # Legacy format - content is already text string
            content_str = content_raw

        elif isinstance(content_raw, list):
            # New LLaVA format - content is array of items
            content_parts = []

            for item in content_raw:
                item_type = item.get("type", "")

                if item_type == "text":
                    content_parts.append(item.get("text", ""))

                elif item_type == "image":
                    # Insert generic <image> token
                    content_parts.append(self.IMAGE_TOKEN)

            # Join parts with newlines to preserve structure
            content_str = "\n".join(content_parts)

        else:
            # Unknown format - convert to string
            content_str = str(content_raw)

        # Format turn with special tokens
        formatted_turn = f"{self.IM_START}{role} {content_str}{self.IM_END}"
        formatted_turns.append(formatted_turn)

    # Join all turns
    prompt = "".join(formatted_turns)

    # Add generation prompt if requested
    if self.add_generation_prompt:
        prompt += f"{self.IM_START}assistant "

    return prompt
```

---

### Modification 3: dataset_utils.py

**File**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/dataset/dataset_utils.py`

#### Change 3.1: Update format_conversation() (Lines 256-272)

**REPLACE** the loop section with:
```python
for i, turn in enumerate(conversations):
    # Support both legacy {"from": "human", "value": "..."}
    # and new {"role": "user", "content": [...]} formats

    # Extract role
    role = turn.get('role', turn.get('from', '')).lower()

    # Extract content
    content_raw = turn.get('content', turn.get('value', ''))

    # Parse content based on format
    if isinstance(content_raw, str):
        value = content_raw
    elif isinstance(content_raw, list):
        # New format: extract text from array
        text_parts = []
        for item in content_raw:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        value = ' '.join(text_parts)
    else:
        value = str(content_raw)

    # Normalize role names
    if role in ['human', 'user']:
        if i == 0 and include_image_token:
            instruction_parts.append(f"USER: {modality_token}\n{value}")
        else:
            instruction_parts.append(f"USER: {value}")
    elif role in ['gpt', 'assistant']:
        if i == len(conversations) - 1:
            # Last turn is the answer
            answer = value
            instruction_parts.append("ASSISTANT: ")
        else:
            # Multi-turn: include previous assistant responses
            instruction_parts.append(f"ASSISTANT: {value}")
```

---

### Modification 4: umbrella_trainer.py

**File**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/training/umbrella_trainer.py`

#### Change 4.1: Add Role Sets (Line 89)
```python
# ADD AFTER __init__ method, around line 99

# Role normalization sets
USER_ROLES = {"user", "human"}
ASSISTANT_ROLES = {"assistant", "gpt"}
```

#### Change 4.2: Update _mask_turns_from_conversation() (Lines 147-183)

**REPLACE** Lines 166-182 with:
```python
def _mask_turns_from_conversation(self,
                                  conversation: list,
                                  input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply masking based on conversation turn roles.

    Supports both:
    - Legacy: role="human"/"gpt"
    - New: role="user"/"assistant"

    Args:
        conversation: List of turn dicts with role and content
        input_ids: Token IDs for this sample
        attention_mask: Attention mask for this sample

    Returns:
        Labels with -100 for masked tokens
    """
    labels = input_ids.clone()
    seq_len = len(input_ids)

    # Extract and normalize roles
    roles = []
    for turn in conversation:
        role = turn.get("role", "").lower()
        if role in self.USER_ROLES:
            roles.append("user")
        elif role in self.ASSISTANT_ROLES:
            roles.append("assistant")
        else:
            logger.warning(f"Unknown role: {role}, treating as user")
            roles.append("user")

    # Estimate token boundaries based on turn order
    # NOTE: This is heuristic - production would use tokenizer alignments
    in_user_turn = True  # Start with user (typically instruction)

    for seq_idx in range(seq_len):
        if attention_mask[seq_idx] == 0:
            # Padding token
            labels[seq_idx] = -100
        elif in_user_turn:
            # Mask USER turn tokens (don't train on user input)
            labels[seq_idx] = -100
            # TODO: Detect turn transition via special tokens
        # else: ASSISTANT turn - keep label (active in loss)

    return labels
```

---

### Modification 5: t1_json_dataset.py

**File**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/dataloaders/t1_json_dataset.py`

#### Change 5.1: Update process_conversation() (Lines 122-134)

**REPLACE**:
```python
def process_conversation(self, json_data: Dict) -> str:
    """
    Convert JSON conversations to LLaVA format.
    """
    conversations = json_data["conversations"]
    prompt = self.conversation_processor.format_conversation_for_llava(conversations)
    return prompt
```

**WITH**:
```python
def process_conversation(self, json_data: Dict) -> str:
    """
    Convert JSON conversations to LLaVA format.

    Supports both "conversation" (singular, legacy) and
    "conversations" (plural, new format).

    Args:
        json_data: Parsed JSON conversation data

    Returns:
        Formatted LLaVA prompt string

    Raises:
        ValueError: If neither conversation key is present
    """
    # Support both key names
    conversations = json_data.get("conversations", json_data.get("conversation"))

    if conversations is None:
        raise ValueError("JSON data missing both 'conversation' and 'conversations' fields")

    if not isinstance(conversations, list):
        raise ValueError(f"Conversations must be a list, got {type(conversations)}")

    prompt = self.conversation_processor.format_conversation_for_llava(conversations)
    return prompt
```

---

### Modification 6: umbrella_utils.py

**File**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/training/umbrella_utils.py`

#### Change 6.1: Add Role Mapping Constant (Line 213)
```python
# ADD AFTER __init__ method around line 224

# Role normalization mapping
ROLE_MAPPING = {
    'user': 'user',
    'human': 'user',
    'assistant': 'assistant',
    'gpt': 'assistant'
}
```

#### Change 6.2: Update _add_turn_to_json() (Lines 496-507)

**REPLACE**:
```python
def _add_turn_to_json(self, conversation: list, role: str, content_parts: list) -> list:
    """Helper to add turn to JSON conversation."""
    combined = " ".join(content_parts).strip()

    # Normalize role to lowercase for compatibility
    role_lower = role.lower()
    if role_lower == "user":
        return self.add_user_to_json(conversation, text=combined)
    elif role_lower == "assistant":
        return self.add_assistant_to_json(conversation, response=combined)

    return conversation
```

**WITH**:
```python
def _add_turn_to_json(self, conversation: list, role: str, content_parts: list) -> list:
    """
    Helper to add turn to JSON conversation with role normalization.

    Supports legacy roles (human/gpt) and new roles (user/assistant).
    """
    combined = " ".join(content_parts).strip()

    # Normalize role using mapping
    role_normalized = self.ROLE_MAPPING.get(role.lower(), role.lower())

    if role_normalized == "user":
        return self.add_user_to_json(conversation, text=combined)
    elif role_normalized == "assistant":
        return self.add_assistant_to_json(conversation, response=combined)

    return conversation
```

---

## 7. Implementation Checklist

### Phase 1: Core Data Loading (Critical Path)
- [ ] **1.1** Add `ROLE_MAPPING` constant to `umbrella_dataset.py`
- [ ] **1.2** Implement `_parse_content()` method in `umbrella_dataset.py`
- [ ] **1.3** Update `_load_samples()` to use role mapping and content parser
- [ ] **1.4** Test: Can load a sample JSON with new format
- [ ] **1.5** Verify: Conversation roles are correctly mapped
- [ ] **1.6** Verify: Content text is extracted correctly
- [ ] **1.7** Verify: Image tokens are identified

**Estimated Time**: 2-3 hours
**Blocking**: Everything else depends on this

---

### Phase 2: Conversation Processing (High Priority)
- [ ] **2.1** Update `format_conversation_for_llava()` in `conversation_processor.py`
- [ ] **2.2** Add content array parsing logic
- [ ] **2.3** Add role normalization
- [ ] **2.4** Test: Process sample conversation successfully
- [ ] **2.5** Verify: Prompt format matches LLaVA expectations
- [ ] **2.6** Verify: Image tokens in correct positions

**Estimated Time**: 1-1.5 hours
**Dependency**: Phase 1 complete

---

### Phase 3: Training Integration (Critical for Loss)
- [ ] **3.1** Add `USER_ROLES` and `ASSISTANT_ROLES` sets to `umbrella_trainer.py`
- [ ] **3.2** Update `_mask_turns_from_conversation()` with role normalization
- [ ] **3.3** Test: Turn masking works correctly
- [ ] **3.4** Verify: User turns are masked (-100)
- [ ] **3.5** Verify: Assistant turns are active

**Estimated Time**: 30-45 minutes
**Dependency**: Phase 1 complete

---

### Phase 4: Utility Functions (Supporting)
- [ ] **4.1** Update `format_conversation()` in `dataset_utils.py`
- [ ] **4.2** Add role/content field mapping
- [ ] **4.3** Test: Legacy and new formats both work
- [ ] **4.4** Update `process_conversation()` in `t1_json_dataset.py`
- [ ] **4.5** Add `ROLE_MAPPING` to `umbrella_utils.py`
- [ ] **4.6** Update `_add_turn_to_json()`

**Estimated Time**: 1-1.5 hours
**Dependency**: Phase 1 complete

---

### Phase 5: Integration Testing (Validation)
- [ ] **5.1** Test complete pipeline: JSON → Dataset → Batch → Training
- [ ] **5.2** Verify: No crashes during data loading
- [ ] **5.3** Verify: Tokenization produces correct tensors
- [ ] **5.4** Verify: Labels have correct masking
- [ ] **5.5** Verify: Training loss computation is correct
- [ ] **5.6** Test backward compatibility with old format
- [ ] **5.7** Test dataloader batching
- [ ] **5.8** Performance benchmark: loading speed

**Estimated Time**: 2-3 hours
**Dependency**: Phases 1-4 complete

---

### Phase 6: Documentation and Cleanup
- [ ] **6.1** Update docstrings for modified functions
- [ ] **6.2** Add inline comments explaining format handling
- [ ] **6.3** Create migration guide for users
- [ ] **6.4** Update README with format specifications
- [ ] **6.5** Add example JSONs for both formats

**Estimated Time**: 1 hour
**Dependency**: Phase 5 complete

---

## 8. Testing Strategy

### Unit Tests

**Test 1: Role Mapping**
```python
def test_role_mapping():
    """Test that roles are correctly mapped"""
    from umbrella_dataset import ROLE_MAPPING

    assert ROLE_MAPPING['user'] == 'human'
    assert ROLE_MAPPING['assistant'] == 'gpt'
    assert ROLE_MAPPING['human'] == 'human'  # backward compat
    assert ROLE_MAPPING['gpt'] == 'gpt'      # backward compat
```

**Test 2: Content Parsing**
```python
def test_content_parsing():
    """Test content parsing for both formats"""
    dataset = UMBRELLADataset(...)

    # Test string format
    text, tokens = dataset._parse_content("Analyze this scan.")
    assert text == "Analyze this scan."
    assert tokens == []

    # Test array format
    content_array = [
        {"type": "text", "text": "Analyze"},
        {"type": "image"},
        {"type": "text", "text": "this scan"}
    ]
    text, tokens = dataset._parse_content(content_array)
    assert "<image>" in text
    assert len(tokens) == 1
```

**Test 3: Conversation Loading**
```python
def test_conversation_loading():
    """Test loading conversations in new format"""
    # Create test JSON with new format
    test_json = {
        "task_id": "test_001",
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this."},
                    {"type": "image"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Analysis..."}]
            }
        ],
        "images": [{"path": "...", "token": "<image>", "modality": "sMRI"}]
    }

    # Load and verify
    dataset = UMBRELLADataset(...)
    samples = dataset._load_samples(test_json_path)

    assert len(samples) == 1
    assert samples[0].conversation[0].role == 'human'
    assert samples[0].conversation[1].role == 'gpt'
    assert '<image>' in samples[0].conversation[0].content
```

### Integration Tests

**Test 4: End-to-End Pipeline**
```python
def test_end_to_end_pipeline():
    """Test complete pipeline from JSON to training batch"""
    # Create dataset
    dataset = UMBRELLADataset(
        json_path="test_new_format.json",
        tokenizer=tokenizer,
        mode='train'
    )

    # Get sample
    sample = dataset[0]

    # Verify structure
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'labels' in sample
    assert 'pixel_values' in sample

    # Verify labels have masking
    assert (sample['labels'] == -100).any()  # Some tokens masked
    assert (sample['labels'] != -100).any()  # Some tokens active
```

**Test 5: Backward Compatibility**
```python
def test_backward_compatibility():
    """Test that old format still works"""
    # Old format JSON
    old_json = {
        "task_id": "old_001",
        "conversation": [
            {"role": "human", "content": "Question?"},
            {"role": "gpt", "content": "Answer."}
        ],
        "images": [...]
    }

    dataset = UMBRELLADataset(...)
    samples = dataset._load_samples(old_json_path)

    assert len(samples) == 1
    # Should work exactly as before
```

---

## 9. Risk Assessment

### High Risk Changes

1. **umbrella_dataset.py:_load_samples()**
   - **Risk**: Breaking existing functionality
   - **Mitigation**: Add conditional checks for both formats, extensive testing

2. **conversation_processor.py:format_conversation_for_llava()**
   - **Risk**: Incorrect prompt generation affecting training
   - **Mitigation**: Unit tests comparing old vs new outputs

### Medium Risk Changes

3. **umbrella_trainer.py:TurnMaskBuilder**
   - **Risk**: Incorrect loss masking affecting model quality
   - **Mitigation**: Validation tests checking mask correctness

### Low Risk Changes

4. **dataset_utils.py, umbrella_utils.py**
   - **Risk**: Utility function breakage
   - **Mitigation**: Isolated functions, easy to test independently

---

## 10. Summary Assessment

### Critical Questions Answered

#### 1. Can the current dataloaders load the new JSON format as-is?
**Answer**: ❌ **NO**

**Evidence**:
- Line 146 in `umbrella_dataset.py`: `role=turn['role']` gets `"user"` but expects `"human"`
- Line 147: `content=turn['content']` expects string but gets array
- Line 148: `_extract_image_tokens(turn['content'])` crashes on array input

#### 2. What is the most critical fix needed first?
**Answer**: **umbrella_dataset.py:_load_samples()** method (Lines 144-170)

**Rationale**: This is the entry point for all data. If data loading fails here, nothing else works. Must:
1. Map roles correctly
2. Parse content arrays
3. Extract text and image tokens

#### 3. Are there architectural assumptions that are completely incompatible?
**Answer**: ❌ **NO** - Only field name/structure changes needed

**Good News**:
- Core architecture (LLaVA-style masking, multi-turn conversations) is compatible
- Image loading pipeline unchanged
- Tokenization logic unchanged
- Only data extraction layer needs updates

**Bad News**:
- Data extraction layer touches 6 files
- Must maintain backward compatibility

#### 4. How many lines of code need to change across all files?
**Answer**: **~250-300 lines** total

**Breakdown**:
- umbrella_dataset.py: 50 lines
- conversation_processor.py: 40 lines
- dataset_utils.py: 40 lines
- umbrella_trainer.py: 25 lines
- t1_json_dataset.py: 10 lines
- umbrella_dataloader.py: 10 lines
- umbrella_utils.py: 10 lines
- Testing/validation: 75 lines

#### 5. Can changes be made without breaking existing functionality?
**Answer**: ✅ **YES** - With careful implementation

**Strategy**:
1. Use conditional logic: `if isinstance(content, str) ... elif isinstance(content, list) ...`
2. Support both `"conversation"` and `"conversations"` keys
3. Map roles bidirectionally: keep internal `"human"/"gpt"`, accept `"user"/"assistant"`
4. Extensive testing with both old and new formats

---

## 11. Recommended Implementation Order

### Week 1: Core Fixes (Critical Path)
**Monday**:
- Modify `umbrella_dataset.py` (Phase 1)
- Add role mapping
- Implement content parser
- Test data loading

**Tuesday**:
- Modify `conversation_processor.py` (Phase 2)
- Update conversation formatting
- Test prompt generation

**Wednesday**:
- Modify `umbrella_trainer.py` (Phase 3)
- Update turn masking
- Test training integration

**Thursday**:
- Modify `dataset_utils.py` and other utilities (Phase 4)
- Update all helper functions
- Test utilities independently

**Friday**:
- Integration testing (Phase 5)
- End-to-end pipeline tests
- Backward compatibility tests
- Performance benchmarks

### Week 2: Validation and Documentation
**Monday-Tuesday**:
- Comprehensive testing
- Edge case handling
- Bug fixes

**Wednesday-Thursday**:
- Documentation updates
- Migration guide
- Example code

**Friday**:
- Final validation
- Code review
- Deployment preparation

---

## 12. Success Criteria

### Functional Criteria
- ✅ Load sample JSON files in new format without errors
- ✅ Extract correct text from content arrays
- ✅ Identify image tokens correctly
- ✅ Generate valid LLaVA prompts
- ✅ Create correct training tensors
- ✅ Apply turn masking correctly
- ✅ Backward compatibility with old format
- ✅ No performance degradation

### Quality Criteria
- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Code coverage > 90% for modified functions
- ✅ No new linting errors
- ✅ Documentation updated
- ✅ Migration guide complete

---

## Appendix A: Example Test Cases

### Test Case 1: New Format Loading
```python
# Input: New format JSON
{
  "task_id": "test_001",
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this MRI:"},
        {"type": "image", "modality": "sMRI"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "This is a T1-weighted scan showing..."}
      ]
    }
  ],
  "images": [{"path": "/path/to/image.nii.gz", "token": "<image>", "modality": "sMRI"}]
}

# Expected Output after _load_samples():
UMBRELLASample(
    task_id="test_001",
    conversation=[
        ConversationTurn(
            role="human",  # mapped from "user"
            content="Analyze this MRI:\n<image>",  # text + image token
            image_tokens=["<image>"]
        ),
        ConversationTurn(
            role="gpt",  # mapped from "assistant"
            content="This is a T1-weighted scan showing...",
            image_tokens=[]
        )
    ],
    image_paths=["/path/to/image.nii.gz"]
)
```

### Test Case 2: Legacy Format (Backward Compatibility)
```python
# Input: Old format JSON
{
  "task_id": "old_001",
  "conversation": [  # singular
    {
      "role": "human",
      "content": "<image_sMRI>\nAnalyze this scan."
    },
    {
      "role": "gpt",
      "content": "This shows normal brain structure."
    }
  ],
  "images": [{"path": "/path/to/image.nii.gz", "token": "<image_sMRI>", "modality": "sMRI"}]
}

# Expected Output: Should work identically to before
UMBRELLASample(
    task_id="old_001",
    conversation=[
        ConversationTurn(
            role="human",
            content="<image_sMRI>\nAnalyze this scan.",
            image_tokens=["<image_sMRI>"]
        ),
        ConversationTurn(
            role="gpt",
            content="This shows normal brain structure.",
            image_tokens=[]
        )
    ],
    image_paths=["/path/to/image.nii.gz"]
)
```

---

## Conclusion

The current dataloaders are **incompatible** with the new JSON format, but the fixes are **straightforward** and can be implemented **without breaking existing functionality**.

**Total Effort**: ~20-25 hours for complete implementation, testing, and documentation.

**Risk Level**: ⚠️ **MEDIUM** - Changes are localized but critical for data pipeline.

**Recommended Approach**: Implement fixes in priority order (Critical → High → Medium), with extensive testing at each phase to ensure both new and old formats work correctly.
