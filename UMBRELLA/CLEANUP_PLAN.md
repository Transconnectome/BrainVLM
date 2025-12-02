# UMBRELLA Project Cleanup Plan
**Date**: December 1, 2025
**Status**: Execution Phase

## Overview
Comprehensive cleanup of UMBRELLA codebase and documentation to remove experimental/broken code and consolidate to production-ready state.

---

## Phase 1: Code Cleanup (ACTIVE)

### Experimental/Duplicate Training Scripts to Remove

#### 1. **project/main_umbrella_training.py** ❌
- **Type**: Older experimental script
- **Status**: Broken, references non-existent modules
- **Action**: DELETE
- **Reason**: Superseded by `project/training/main_umbrella_training_fixed.py`
- **Dependencies**: None (not imported anywhere)

#### 2. **project/utils/umbrella_trainer.py** ❌
- **Type**: Extended trainer for experimental multi-task training
- **Status**: Unused - only references itself
- **Action**: DELETE
- **Reason**: Not used by production training script (main_umbrella_training_fixed.py)
- **Dependencies**: Zero references (grep found only self-reference)

#### 3. **project/utils/dynamic_trainer.py** ❌
- **Type**: Dynamic batching trainer (experimental)
- **Status**: Unused - broken imports
- **Imports**: Relative imports that fail (line 16-18)
- **Action**: DELETE
- **Reason**: Not used by any production code
- **Dependencies**: Zero references

#### 4. **project/utils/training_example.py** ❌
- **Type**: Example/sample code
- **Status**: Likely incomplete or experimental
- **Action**: DELETE
- **Reason**: Example code not needed in production

#### 5. **test_imports.py** & **test_tokenization.py** (in root) ⚠️
- **Type**: Debugging/test scripts
- **Action**: REVIEW - Move to appropriate test directory if needed
- **Location**: `/UMBRELLA/test_*.py`

### Utilities to Review

#### Memory/Performance Utilities (in project/utils/)
- memory_utils.py
- memory_safety.py
- dynamic_batching.py

**Status**: Check if used by any current code. If not, move to `.archive/experimental/`

---

## Phase 2: Documentation Cleanup (PENDING)

### Root-Level Documentation Issues
- **Count**: 47 markdown files in UMBRELLA root
- **Status**: EXCESSIVE - needs consolidation

### Categories to Consolidate/Archive

#### Phase Reports (Archive to `.archive/`)
- PHASE_*.md files (4-6 files)
- SESSION_*.md files (3-4 files)
- *_COMPLETION_*.md files (8-10 files)
- *_SUMMARY_*.md files (5-6 files)

#### Technical Notes (Keep or Consolidate)
- DUMMY_LOSS_*.md (5 files) → Consolidate to 1
- MODALITY_HANDLING_*.md (2 files) → Consolidate to 1
- DMRI_T1_*.md (2 files) → Consolidate to 1
- LLAVA_JSON_*.md (3 files) → Keep but organize

#### Outdated Implementation Docs (Archive)
- CLEANUP_*.md files (3 files) → Archive old cleanup docs
- *_COMPATIBILITY_*.md files (3 files) → Review and archive
- MIGRATION_*.md files (2 files) → Archive old migrations

#### Verification/Testing Docs (Archive)
- *_VERIFICATION_*.md files (4 files)
- *_CHECKLIST_*.md files (2 files)

### Documentation Organization Strategy
```
UMBRELLA/
├── README.md (KEEP - main documentation)
├── QUICKSTART.md (CREATE - if missing)
├── TRAINING_GUIDE.md (CONSOLIDATE from training docs)
├── TOKENIZATION_GUIDE.md (KEEP)
├── project/ (code)
├── sample_data/ (samples)
├── sample_scripts/ (examples)
└── .archive/
    ├── experimental_code/ (unused Python files)
    ├── phase_reports/ (phase docs)
    ├── session_history/ (session docs)
    └── technical_notes/ (consolidated docs)
```

---

## Phase 3: Import Validation (PENDING)

### Validation Checks
1. ✅ main_umbrella_training_fixed.py imports are working
2. ⏳ All dataset imports work
3. ⏳ All collator imports work
4. ⏳ No broken references in config files

---

## Phase 4: README Updates (PENDING)

### Update Required Sections
- Remove references to multi-task training
- Update to reflect single production script
- Update directory structure to match current layout
- Update usage examples to point to main_umbrella_training_fixed.py

---

## Execution Summary

### Files to Delete Immediately
```
project/main_umbrella_training.py
project/utils/umbrella_trainer.py
project/utils/dynamic_trainer.py
project/utils/training_example.py
```

### Files to Archive
```
All 47 root-level .md files → .archive/documentation_consolidation/
All experimental utilities → .archive/experimental_code/
```

### Files to Keep/Update
```
README.md (update)
project/training/main_umbrella_training_fixed.py (keep as-is)
project/dataset/umbrella_dataset_fixed.py (keep as-is)
project/dataset/umbrella_collator.py (keep as-is)
project/tests/validate_tokenization.py (keep as-is)
```

---

## Next Steps
1. Execute code deletions
2. Create archive structure
3. Move documentation
4. Update README
5. Validate imports
6. Create cleanup summary report

