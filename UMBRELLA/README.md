# UMBRELLA: LLaVA-based Vision-Language Model for Brain MRI Analysis

## Overview

UMBRELLA is a LLaVA-based multimodal foundation model for brain MRI analysis. This directory contains a cleaned implementation that uses only the LLaVA architecture (not BLIP) for vision-language modeling of neuroimaging data.

## Architecture

**Base Model**: LLaVA 1.5 7B (`llava-hf/llava-1.5-7b-hf`)

**Key Components**:
- **Vision Encoder**: CLIP ViT with custom 3D/4D patch embedding
- **Projector**: Linear projection (LLaVA-style, NOT Q-Former)
- **Language Model**: Vicuna-7B

**Training Strategy**:
- Custom patch embedding for 3D T1 and 4D rsfMRI volumes
- Instruction-tuning with JSON-format prompts
- Only patch embedding layer is trainable (vision encoder, projector, and LLM are frozen)

## Directory Structure

```
UMBRELLA/
├── README.md                    # This file
├── environment.yaml             # Conda environment
├── environment_llava.yaml       # LLaVA-specific environment
├── project/
│   ├── main_umbrella_llava_T1.py    # Main training script
│   ├── config/
│   │   ├── umbrella_llava_train.yaml # Training configuration
│   │   └── deepspeed/                # DeepSpeed configs
│   ├── dataset/
│   │   ├── dataset_T1.py            # T1 dataset (generic)
│   │   └── dataset_T1_LLaVa.py      # T1 dataset (LLaVA format)
│   ├── model/
│   │   └── patch_embed.py           # 3D/4D patch embedding
│   └── utils/
│       ├── data.py                  # Data utilities
│       ├── Trainer.py               # Custom HuggingFace Trainer
│       └── utils.py                 # General utilities
└── sample_scripts/
    └── UMBRELLA_LLaVa_T1_DDP_interactive.sh
```

## Key Files

### Main Training Script
`project/main_umbrella_llava_T1.py`
- Loads LLaVA model from HuggingFace
- Replaces vision encoder embeddings with custom PatchEmbed
- Trains with HuggingFace Trainer

### Patch Embedding
`project/model/patch_embed.py`
- Converts 3D T1 MRI (B, C, D, H, W) to patch embeddings
- Converts 4D rsfMRI (B, C, D, H, W, T) to patch embeddings
- Uses learnable positional embeddings

### Dataset
`project/dataset/dataset_T1_LLaVa.py`
- LLaVA-style instruction format: "USER: <image>\n{instruction}\nASSISTANT: {answer}"
- Supports ABCD and UKB cohorts
- Implements proper label masking for causal LM training

## Configuration

Edit `project/config/umbrella_llava_train.yaml`:

```yaml
model:
    hf_name: "llava-hf/llava-1.5-7b-hf"  # LLaVA model
    T1:
        patch_size: [10, 10, 10]  # 3D patch size for T1

dataset:
    T1:
        img_size: [120, 120, 120]
        target: ["sex"]  # or ["age"]
        study_sample: ["ABCD"]
```

## Usage

### Training
```bash
cd project
torchrun --nnodes 1 --nproc_per_node 1 main_umbrella_llava_T1.py
```

### Multi-GPU Training
```bash
torchrun --nnodes 1 --nproc_per_node 4 main_umbrella_llava_T1.py
```

## LLaVA Architecture vs BLIP

This implementation uses **LLaVA architecture**, which differs from BLIP in several key ways:

| Feature | LLaVA | BLIP-2 |
|---------|-------|--------|
| Projector | Linear MLP | Q-Former |
| Vision-Language Interface | Direct concatenation | Cross-attention |
| Training | End-to-end instruction tuning | Stage-wise training |
| Loss | Unified NLL loss | Multiple losses (ITC, ITM, LM) |

UMBRELLA uses the simpler LLaVA approach with linear projection, making it more efficient and easier to train while maintaining competitive performance.

## Prompt Format

The model uses the LLaVA chat format:

```
USER: <image>
You are a neurologist and now you are analyzing T1-weighted MRI images.
Estimate sex of subject from this image.
ASSISTANT: male
```

For JSON-format prompts (recommended for UMBRELLA):
```json
{
  "image": "<image_path>",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nYou are a neurologist analyzing T1-weighted MRI images. Estimate the subject's sex."
    },
    {
      "from": "gpt",
      "value": "male"
    }
  ]
}
```

## Dependencies

- PyTorch >= 2.0
- Transformers >= 4.31
- MONAI (for medical image loading)
- timm (for vision models)
- wandb (for logging)

## References

- [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485)
- [LLaVA-1.5: Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)
- [UMBRELLA Vision Documents](../projects/BrainVLM/docs/VISION_AND_STRATEGY/)

## License

See the main BrainVLM repository for license information.
