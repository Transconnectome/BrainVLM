# UMBRELLA Implementation Setup - Cleanup Report

## Summary

Successfully created the UMBRELLA directory from BLIP_MRI with all BLIP-related code removed, leaving only LLaVA-style implementation.

## Repository Setup

- **Repository**: `https://github.com/Transconnectome/BrainVLM`
- **Clone Location**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella`
- **Branch Created**: `umbrella`
- **Status**: Ready for implementation

## Files Deleted (BLIP-specific)

### Main Training Scripts
- `project/main_Bblip_t5_hf_inference.py`
- `project/main_Bblip_t5_hf_joint_T1.py`
- `project/main_Bblip_t5_hf_joint.py`

### Model Files
- `project/model/Bblip_t5_tmp.py`
- `project/model/modeling_blip_2.py`

### Configuration
- `project/config/Brain_blip_t5_train_DeepSpeed_joint.yaml`

### Sample Scripts
- `sample_scripts/BLIP_MRI_Blip_DDP_interactive.sh`
- `sample_scripts/BLIP_MRI_Blip_T1_DDP_interactive.sh`

## Files Renamed

| Original | New |
|----------|-----|
| `main_BLLaVa_hf_joint_T1.py` | `main_umbrella_llava_T1.py` |
| `model/Bblip_t5.py` | `model/patch_embed.py` |
| `config/Brain_LLaVa_train_DeepSpeed_joint.yaml` | `config/umbrella_llava_train.yaml` |
| `sample_scripts/BLIP_MRI_LLaVa_T1_DDP_interactive.sh` | `sample_scripts/UMBRELLA_LLaVa_T1_DDP_interactive.sh` |

## Files Modified

### 1. `main_umbrella_llava_T1.py`
**Changes**:
- Added module docstring
- Updated config path from `Brain_LLaVa_train_DeepSpeed_joint.yaml` to `umbrella_llava_train.yaml`
- Changed WANDB_PROJECT from "BLIP_sMRI" to "UMBRELLA"
- Updated import from `model.Bblip_t5` to `model.patch_embed`
- Improved comments (removed BLIP references like "freeze Qformer")
- Cleaned up code formatting and removed unnecessary TODOs

### 2. `model/patch_embed.py`
**Changes**:
- Added comprehensive module docstring
- Removed all BLIP-related imports:
  - `from lavis.models import load_model`
  - `from transformers import Blip2ForConditionalGeneration`
  - Various unused imports (loralib, deepspeed, sklearn, etc.)
- Kept only essential imports (torch, nn, trunc_normal_)
- Added detailed class and method docstrings
- Improved code documentation

### 3. `config/umbrella_llava_train.yaml`
**Changes**:
- Added header comments identifying it as UMBRELLA config
- Changed `hf_name` from `Salesforce/blip2-flan-t5-xl` to `llava-hf/llava-1.5-7b-hf`
- Added clarifying comment `# LLaVA model (not BLIP)`

### 4. `sample_scripts/UMBRELLA_LLaVa_T1_DDP_interactive.sh`
**Changes**:
- Updated header comments for UMBRELLA
- Changed working directory from BLIP_MRI to UMBRELLA
- Updated conda environment name
- Updated main script reference

### 5. `environment.yaml` and `environment_llava.yaml`
**Changes**:
- Updated environment names from BLIP_MRI to UMBRELLA

### 6. `utils/Trainer.py`
**Changes**:
- Removed BLIP references in Korean comments

## BLIP References Eliminated

### Code Patterns Removed
```python
# Imports removed:
from transformers import Blip2ForConditionalGeneration
from lavis.models import load_model
from lavis.models.blip2_models.blip2 import Blip2Base

# Model loading removed:
model = Blip2ForConditionalGeneration.from_pretrained(...)

# Architecture components removed:
self.qformer = ...
loss_itc = contrastive_loss(...)
loss_ita = image_text_alignment_loss(...)
```

### Code Patterns Preserved (LLaVA)
```python
# LLaVA imports kept:
from transformers import LlavaForConditionalGeneration

# LLaVA model loading kept:
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

# LLaVA architecture kept:
patch_embed = PatchEmbed(...)  # Custom 3D/4D patch embedding
setattr(model.vision_tower.vision_model, "embeddings", patch_embed)
```

## Final Verification

### BLIP References Check
```bash
grep -r "blip\|BLIP" UMBRELLA/
```
**Result**: Only one reference remains - a clarifying comment in config:
```yaml
# LLaVA model (not BLIP)
```
This is intentional and clarifies the architecture choice.

### Directory Structure
```
UMBRELLA/
├── README.md                              # NEW: Documentation
├── environment.yaml                       # UPDATED
├── environment_llava.yaml                 # UPDATED
├── project/
│   ├── main_umbrella_llava_T1.py         # RENAMED & CLEANED
│   ├── config/
│   │   ├── umbrella_llava_train.yaml     # RENAMED & UPDATED
│   │   └── deepspeed/                     # Unchanged
│   ├── dataset/
│   │   ├── dataset_T1_LLaVa.py           # Unchanged (already LLaVA)
│   │   ├── dataset_T1.py                  # Unchanged
│   │   └── ...
│   ├── model/
│   │   └── patch_embed.py                 # RENAMED & CLEANED
│   └── utils/
│       ├── Trainer.py                     # UPDATED
│       └── ...
└── sample_scripts/
    └── UMBRELLA_LLaVa_T1_DDP_interactive.sh  # RENAMED & UPDATED
```

## LLaVA-style Code Preserved

### Architecture
- **Vision Encoder**: CLIP ViT with custom PatchEmbed
- **Projector**: Linear projection (`multi_modal_projector`)
- **Language Model**: Vicuna-7B

### Training Flow
1. Load LLaVA model from HuggingFace
2. Replace vision embeddings with custom 3D/4D PatchEmbed
3. Freeze all parameters except embeddings
4. Train with unified NLL loss

### Prompt Format
```
USER: <image>
You are a neurologist analyzing T1-weighted MRI images.
Estimate sex of subject from this image.
ASSISTANT: male
```

## Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| BrainVLM repository cloned | PASS |
| "umbrella" branch created | PASS |
| BLIP_MRI copied to UMBRELLA | PASS |
| BLIP-specific scripts removed | PASS |
| BLIP imports eliminated | PASS |
| BLIP model loading removed | PASS |
| BLIP-specific losses removed | PASS |
| LLaVA code preserved | PASS |
| No broken imports | PASS |
| Documentation created | PASS |

## Next Steps

1. **Implement UMBRELLA-specific features**:
   - Add JSON-format prompt loading
   - Implement multi-modal ROI tokens
   - Add EVA-ViT as vision encoder option

2. **Update training configuration**:
   - Adjust paths for your data
   - Configure wandb API key
   - Set appropriate batch sizes

3. **Test training**:
   ```bash
   cd UMBRELLA/project
   torchrun --nnodes 1 --nproc_per_node 1 main_umbrella_llava_T1.py
   ```

4. **Commit changes**:
   ```bash
   git add UMBRELLA
   git commit -m "feat: Add UMBRELLA LLaVA-based implementation (cleaned from BLIP_MRI)"
   ```

## Files Location

- **Repository Root**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella`
- **UMBRELLA Directory**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA`
- **Main Training Script**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/main_umbrella_llava_T1.py`
- **Configuration**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/config/umbrella_llava_train.yaml`
- **Patch Embedding**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/project/model/patch_embed.py`

## Repository Structure Update (November 2024)

The BrainVLM-umbrella repository has been moved to the standardized project structure:

**Old Location**: `/Users/apple/Desktop/neuro-ai-research-system/BrainVLM-umbrella`
**New Location**: `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella`

This allows:
- Separation of docs, code, and data
- Multiple code repositories per project
- Better organization for future implementations

See `/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/PROJECT_STRUCTURE.md` for full project organization.
