# Multi-Subject MRI Comparison - Quick Summary

**Status**: ✅ Analysis Complete
**Date**: November 20, 2025

---

## Your Question

> Can the current dataloader handle multi-subject comparison format like:
> ```json
> {
>   "subject_id": ["sub-1", "sub-2"],
>   "modality_paths": {
>     "image_sMRI": ["/path/sub-1", "/path/sub-2"],
>     "image_fMRI": [[frames_sub1], [frames_sub2]]
>   }
> }
> ```

---

## Clear Answer: **NO** ❌

### Current Limitations

| Component | Issue | Impact |
|-----------|-------|--------|
| **Dataset `__getitem__`** | Expects `subject_id` as string, not list | ❌ Fails on multi-subject |
| **Image loading** | Expects single path, not list of paths | ❌ Cannot load N subjects |
| **Output tensor** | Returns `(1, H, W, D)` single subject | ❌ Cannot return N subjects stacked |
| **Collator** | Assumes single image per sample | ❌ Breaks on multi-subject tensors |
| **Trainer loss** | No mechanism for grouped subjects | ❌ Cannot compute comparison loss |

### Where It Breaks

**Line-by-line failures:**

1. **`t1_json_dataset.py:287-290`** - `subject_id` assumed string
2. **`t1_json_dataset.py:296-304`** - Image path assumed single string
3. **`t1_json_dataset.py:307-310`** - Single image loading
4. **`data.py:104-185`** - Collator assumes single sample per feature
5. **`Trainer.py:257-324`** - Loss computation for single batches only

---

## Recommended Solution: Option E ✅

### Hybrid Approach with `comparison_mode` Flag

**Pros:**
- ✅ Achieves your goal (paired subject training)
- ✅ Backward compatible (default `False`)
- ✅ Minimal code disruption
- ✅ Clear, explicit API
- ✅ Production-ready

**Implementation:**
```python
# Original code unchanged
dataset = T1JSONDataset(json_file=...)  # Works as before

# New feature: opt-in
dataset = T1JSONDataset(json_file=..., comparison_mode=True)  # Enables multi-subject
```

---

## Implementation Roadmap

### Phase 1: Dataset (2-3 hours) 🟢 Low Risk
- Add `comparison_mode: bool = False` parameter
- Add `_get_comparison_item()` method (~80 lines)
- Modify `__getitem__()` routing logic
- **Output**: Returns `(N, 1, H, W, D)` stacked tensor

### Phase 2: Collator (1-2 hours) 🟡 Medium Risk
- Add `_collate_comparison()` method (~50 lines)
- Route based on `comparison_mode` flag
- **Output**: Batch shape `(B, N, 1, H, W, D)`

### Phase 3: Trainer (2-3 hours) 🟡 Medium Risk
- Add `_compute_comparison_loss()` method (~60 lines)
- Flatten `(B, N)` for encoding, reshape after
- **Output**: Loss computed across subject pairs

### Phase 4: Model (4-6 hours) 🔴 High Risk
- Choose fusion strategy (concat, mean-pool, attention)
- Modify model forward pass for `comparison_mode`
- Handle embedding reshaping for paired subjects
- **Required**: Architecture decision needed

**Total Effort**: 9-14 hours (can be done incrementally)

---

## Key Technical Details

### Data Flow (Multi-Subject)

```
JSON Entry: ["sub-1", "sub-2"]
    ↓
Dataset: Output (2, 1, 128, 128, 128)  [2 subjects stacked]
    ↓
Collator: Batch → (B, 2, 1, 128, 128, 128)  [B batches, 2 subjects each]
    ↓
Trainer: Flatten → (B*2, 1, 128, 128, 128)  [4 images for B=2]
    ↓
Model Vision Encoder: (4, tokens, hidden)
    ↓
Reshape: (B=2, N=2, tokens, hidden)
    ↓
Fusion: Concatenate/mean/attention
    ↓
LLM: Generate comparison answer
    ↓
Loss: NLL on answer tokens
```

### JSON Format Examples

**Single subject (existing):**
```json
{
  "subject_id": "sub-001",
  "modality_paths": {"image_sMRI": "/path/to/sub-001"}
}
```

**Multi-subject (new):**
```json
{
  "subject_id": ["sub-001", "sub-002"],
  "modality_paths": {
    "image_sMRI": ["/path/to/sub-001", "/path/to/sub-002"]
  }
}
```

### Backward Compatibility ✅

- Default `comparison_mode=False` → existing behavior unchanged
- Single-subject samples work as before
- No breaking changes to API
- Can mix single and multi-subject in same dataset (if careful)

---

## Use Cases This Enables

1. **Contrastive Learning**: "Which subject has disease?"
2. **Metric Learning**: Learn similarity between subjects
3. **Reference Classification**: Classify test vs reference brain
4. **Clinical Comparison**: "Compare patient A vs patient B"

---

## Quick Alternative (Workaround)

If you need this immediately without implementing 4 phases:

**Preprocessing approach:**
```python
# Pre-concatenate offline
stacked = torch.stack([img1, img2])  # (2, H, W, D)
torch.save(stacked, "comparison_001_002.pt")

# JSON references pre-concatenated file
{"subject_id": "sub-001_vs_002", "modality_paths": {"image_sMRI": "comparison_001_002.pt"}}

# Minimal dataset changes
# Just handle .pt files differently
```

**Pros**: Works immediately
**Cons**: Requires preprocessing, no architecture learning

---

## Decision Points for User

### Question 1: Do you need this feature?
- **Yes** → Proceed with Phase 1-4 implementation
- **Maybe** → Start with Phase 1 only, decide later
- **No** → Use single-subject format, close this feature

### Question 2: When do you need it?
- **Immediately** → Use workaround (preprocessing)
- **This week** → Start with Phase 1-2 (dataset + collator)
- **Later** → Plan full implementation with Phase 3-4

### Question 3: How many subjects to compare?
- **Always 2** → Simpler implementation
- **Variable (2, 3, 4...)** → Need flexible handling
- **Many (10+)** → Reconsider memory implications

### Question 4: Which modality?
- **T1 only** → Easier, only modify one dataset class
- **T1 + fMRI** → More complex, need to handle variable-length sequences
- **All three** → Most complex, requires all dataset classes modified

---

## Documentation Created

1. **MULTI_SUBJECT_COMPARISON_DESIGN.md** (Long form)
   - Complete implementation specification
   - Code examples for each phase
   - Testing strategy
   - Integration checklist

2. **MULTI_SUBJECT_ANALYSIS_SUMMARY.md** (This file)
   - Quick reference
   - Decision points
   - Summary of recommendations

---

## Recommendation

**Start with Phase 1 (Dataset)** - 2-3 hours
- Lowest risk
- Validates the approach
- Can decide on Phase 2 based on results
- Can use workaround if Phase 2-4 not needed

Then:
- Phase 2 (Collator) - relatively straightforward
- Phase 3 (Trainer) - moderate complexity
- Phase 4 (Model) - requires architecture decision

---

## Next Steps

1. **Decide**: Do you want to implement this feature?
2. **Clarify**: Answer the 4 decision questions above
3. **Plan**: Which phases will you implement?
4. **Start**: Begin with Phase 1 dataset changes
5. **Test**: Verify output shapes at each phase

---

## Files References

**For implementation details**:
→ See `MULTI_SUBJECT_COMPARISON_DESIGN.md`

**For code locations**:
- Dataset: `project/dataset/t1_json_dataset.py` (lines 54-323)
- Collator: `project/utils/data.py` (lines 104-185)
- Trainer: `project/utils/Trainer.py` (lines 257-324)

**For testing**:
→ See `MULTI_SUBJECT_COMPARISON_DESIGN.md` - Testing Strategy section

---

## Summary Table

| Question | Answer |
|----------|--------|
| Can current system handle it? | ❌ NO |
| Can it be added? | ✅ YES |
| Backward compatible? | ✅ YES |
| Recommended approach? | Option E (Hybrid flag) |
| Total implementation time? | 9-14 hours |
| Highest risk phase? | Phase 4 (Model) |
| Quickest workaround? | Preprocessing .pt files |
| Can be done incrementally? | ✅ YES (Phase 1-4) |

---

**Status**: Analysis Complete and Ready for Implementation
**Date**: November 20, 2025
