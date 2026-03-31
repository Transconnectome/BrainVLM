import multiprocessing
import os
import sys
import json
import torch
import logging
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    AutoModelForCausalLM
)

# Set project module path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# UMBRELLA project module imports
from dataset.umbrella_dataset import UMBRELLADataset
from dataset.umbrella_collator import UMBRELLACollator
from model.patch_embed import PatchEmbed

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@dataclass
class UMBRELLAInferenceConfig:
    """Unified configuration for UMBRELLA Inference loaded from YAML."""
    
    # Paths
    model_path: str = field(default=None, metadata={"help": "Path to the trained model checkpoint (weights)"})
    data_path: str = field(default=None, metadata={"help": "Path to the test/validation jsonl file"})
    output_file: str = field(default="inference_results.jsonl", metadata={"help": "Output file path"})
    
    # Model Config
    base_model_name: str = "llava-hf/llava-interleave-qwen-0.5b-hf"
    modality: str = "sMRI"
    img_size: List[int] = None
    patch_size: List[int] = None
    
    # Specific Modality Configs
    sMRI_img_size: List[int] = None
    sMRI_patch_size: List[int] = None
    fMRI_img_size: List[int] = None
    fMRI_patch_size: List[int] = None
    T1_img_size: List[int] = None
    T1_patch_size: List[int] = None
    FA_img_size: List[int] = None
    FA_patch_size: List[int] = None

    # Generation Config
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    do_sample: bool = False
    
    # System Config
    batch_size: int = 4
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_images_per_sample: int = 8

    @classmethod
    def from_yaml(cls, config_path: str, modality: str = 'sMRI') -> 'UMBRELLAInferenceConfig':
        """Load configuration from the training YAML file."""
        yaml_config = load_config(config_path)
        
        dataset_config = yaml_config.get('dataset', {})
        model_config = yaml_config.get('model', {})
        trainer_config = yaml_config.get('trainer', {})

        modality_dataset_config = dataset_config.get(modality, {})
        modality_model_config = model_config.get(modality, {})
        
        # Handle T1_FA: use T1 config for img_size
        if modality in ["T1", "FA", "T1_FA"]:
            current_img_size = dataset_config.get("T1", {}).get("img_size", [120, 120, 120])
        else:
            current_img_size = modality_dataset_config.get("img_size", [96, 96, 96])
        current_patch_size = modality_model_config.get('patch_size', [16, 16, 16])
        
        sMRI_config = dataset_config.get('sMRI', {})
        sMRI_model_config = model_config.get('sMRI', {})
        fMRI_config = dataset_config.get('fMRI', {})
        fMRI_model_config = model_config.get('fMRI', {})
        T1_config = dataset_config.get('T1', {})
        T1_model_config = model_config.get('T1', {})
        FA_config = dataset_config.get('FA', {})
        FA_model_config = model_config.get('FA', {})

        return cls(
            base_model_name=model_config.get('hf_name', 'llava-hf/llava-interleave-qwen-0.5b-hf'),
            modality=modality,
            img_size=current_img_size,
            patch_size=current_patch_size,
            
            sMRI_img_size=sMRI_config.get('img_size', [96, 96, 96]),
            sMRI_patch_size=sMRI_model_config.get('patch_size', [16, 16, 16]),
            fMRI_img_size=fMRI_config.get('img_size', [96, 96, 96, 24]),
            fMRI_patch_size=fMRI_model_config.get('patch_size', [16, 16, 16, 3]),
            T1_img_size=T1_config.get('img_size', [120, 120, 120]),
            T1_patch_size=T1_model_config.get('patch_size', [20, 20, 20]),
            FA_img_size=FA_config.get('img_size', [120, 120, 120]),
            FA_patch_size=FA_model_config.get('patch_size', [20, 20, 20]),
            
            batch_size=trainer_config.get('per_device_batch_size', 4)
        )

def load_model_with_custom_architecture(cfg: UMBRELLAInferenceConfig, tokenizer: AutoTokenizer):
    """
    1. Load Base Model
    2. Replace with Custom PatchEmbed
    3. Load Fine-tuned Checkpoint weights
    """
    logger.info(f"1. Loading base model architecture: {cfg.base_model_name}")
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg.base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    except:
        logger.warning("LlavaForConditionalGeneration failed, trying AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            torch_dtype=torch.float16 if "cuda" in cfg.device else torch.float32,
            trust_remote_code=True
        )

    logger.info(f"2. Resizing token embeddings to {len(tokenizer)}...")

    # 3. Replace PatchEmbed
    logger.info("3. Replacing PatchEmbed with custom 3D implementation...")
    original_patch_embedding = model.vision_tower.vision_model.embeddings.patch_embedding
    embed_dim = int(original_patch_embedding.out_channels)
    
    patch_embed = PatchEmbed(
        sMRI_size=cfg.sMRI_img_size,
        sMRI_patch_size=cfg.sMRI_patch_size,
        fMRI_size=cfg.fMRI_img_size,
        fMRI_patch_size=cfg.fMRI_patch_size,
        T1_size=cfg.T1_img_size,
        T1_patch_size=cfg.T1_patch_size,
        FA_size=cfg.FA_img_size,
        FA_patch_size=cfg.FA_patch_size,
        embed_dim=embed_dim,
        modality_type=cfg.modality
    )
    setattr(model.vision_tower.vision_model, "embeddings", patch_embed)

    # 4. Load trained weights
    if cfg.model_path and os.path.exists(cfg.model_path):
        logger.info(f"4. Loading fine-tuned weights from: {cfg.model_path}")
        
        if os.path.isdir(cfg.model_path):
            state_dict_path = os.path.join(cfg.model_path, "model.safetensors")
            if not os.path.exists(state_dict_path):
                state_dict_path = os.path.join(cfg.model_path, "pytorch_model.bin")
            
            if os.path.exists(state_dict_path):
                if state_dict_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(state_dict_path)
                else:
                    state_dict = torch.load(state_dict_path, map_location="cpu")
                
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(missing)
                    logger.warning(f"Missing keys: {len(missing)}")
                if unexpected:
                    logger.info(f"Unexpected keys: {len(unexpected)}")
            else:
                logger.error("No model weights found in checkpoint directory!")
                raise FileNotFoundError("Model weights not found")
        else:
            state_dict = torch.load(cfg.model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    
    model.to(cfg.device)
    model.eval()
    return model

def prepare_inputs_for_generation(model, batch, tokenizer, device):
    """Reimplement UMBRELLATrainer._generate_step logic"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    pixel_values = batch.get('pixel_values')
    
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    # Vision Feature Extraction
    image_features_per_sample = None
    if pixel_values is not None:
        with torch.no_grad():
            image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
            sel_feats = image_outputs.hidden_states[model.config.vision_feature_layer]
            
            if hasattr(model.vision_tower, "select_feature"):
                 sel_feats = sel_feats[:, 1:] 
            
            sel_feats = model.multi_modal_projector(sel_feats)
            
            batch_size = input_ids.shape[0]
            total_images = sel_feats.shape[0]
            if batch_size > 0:
                num_imgs_per_batch = total_images // batch_size
                image_features_per_sample = sel_feats.view(batch_size, num_imgs_per_batch, -1, sel_feats.shape[-1])

    inputs_embeds = model.get_input_embeddings()(input_ids)
    
    # Dynamic Merge
    new_input_embeds = []
    new_attention_masks = []
    
    IMAGE_TOKEN_ID = 151646 
    if hasattr(model.config, "image_token_index"):
        IMAGE_TOKEN_ID = model.config.image_token_index

    batch_size = input_ids.shape[0]
    
    for i in range(batch_size):
        cur_ids = input_ids[i]
        cur_emb = inputs_embeds[i]
        cur_mask = attention_mask[i]
        
        img_indices = (cur_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        final_embeds, final_masks = [], []
        last_pos, img_cnt = 0, 0
        
        for pos in img_indices:
            final_embeds.append(cur_emb[last_pos:pos])
            final_masks.append(cur_mask[last_pos:pos])
            
            if image_features_per_sample is not None:
                if img_cnt < image_features_per_sample.size(1):
                    img_feat = image_features_per_sample[i][img_cnt]
                    final_embeds.append(img_feat)
                    final_masks.append(torch.ones((img_feat.shape[0],), device=device, dtype=cur_mask.dtype))
                    img_cnt += 1
            last_pos = pos + 1
        
        final_embeds.append(cur_emb[last_pos:])
        final_masks.append(cur_mask[last_pos:])
        
        new_input_embeds.append(torch.cat(final_embeds, dim=0))
        new_attention_masks.append(torch.cat(final_masks, dim=0))

    # Trigger & Padding
    trigger_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    trigger_embeds = model.get_input_embeddings()(torch.tensor(trigger_ids, device=device))
    trigger_mask = torch.ones(len(trigger_ids), device=device, dtype=attention_mask.dtype)

    max_len = max([t.shape[0] for t in new_input_embeds]) + len(trigger_ids)
    
    final_inputs_embeds = []
    final_attention_masks = []
    
    for embed, mask in zip(new_input_embeds, new_attention_masks):
        embed_with_trigger = torch.cat([embed, trigger_embeds], dim=0)
        mask_with_trigger = torch.cat([mask, trigger_mask], dim=0)
        
        pad_len = max_len - embed_with_trigger.shape[0]
        if pad_len > 0:
            pad_embed = torch.zeros((pad_len, embed.shape[1]), device=device, dtype=embed.dtype)
            pad_mask = torch.zeros((pad_len,), device=device, dtype=mask.dtype)
            final_inputs_embeds.append(torch.cat([pad_embed, embed_with_trigger], dim=0))
            final_attention_masks.append(torch.cat([pad_mask, mask_with_trigger], dim=0))
        else:
            final_inputs_embeds.append(embed_with_trigger)
            final_attention_masks.append(mask_with_trigger)
            
    return torch.stack(final_inputs_embeds), torch.stack(final_attention_masks)

def main():
    parser = argparse.ArgumentParser(description="Inference script for UMBRELLA model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML training config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained checkpoint directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the validation/test jsonl file")
    parser.add_argument("--output_file", type=str, default="inference_results.jsonl")
    parser.add_argument("--modality", type=str, default="sMRI", help="Modality to use from config (sMRI/fMRI/T1/FA/T1_FA)")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    
    args = parser.parse_args()

    # 1. Load Config
    logger.info(f"Loading config from {args.config_path} with modality={args.modality}")
    cfg = UMBRELLAInferenceConfig.from_yaml(args.config_path, modality=args.modality)
    
    cfg.model_path = args.model_path
    cfg.data_path = args.data_path
    cfg.output_file = args.output_file
    if args.batch_size:
        cfg.batch_size = args.batch_size

    # 2. Tokenizer Setup
    logger.info(f"Loading tokenizer from: {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model Load
    model = load_model_with_custom_architecture(cfg, tokenizer)

    # 4. Dataset & Dataloader
    logger.info(f"Loading dataset from {cfg.data_path}...")
    dataset = UMBRELLADataset(
        data_path=cfg.data_path,
        tokenizer=tokenizer,
        mode='eval',
        max_seq_length=2048,
        img_size=tuple(cfg.img_size),
        max_images_per_sample=cfg.max_images_per_sample,
        modality_type=cfg.modality
    )
    
    collator = UMBRELLACollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        collate_fn=collator
    )

    # 5. Inference Loop
    results = []
    logger.info(f"Starting inference on {cfg.device}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            metadata_list = batch.get('metadata', [])
            task_types = batch.get('task_types', [])
            
            inputs_embeds, attention_mask = prepare_inputs_for_generation(model, batch, tokenizer, cfg.device)
            
            # Manual autoregressive generation
            batch_size = inputs_embeds.shape[0]
            generated_tokens = []
            current_embeds = inputs_embeds
            current_mask = attention_mask
            past_key_values = None
            
            for step in range(cfg.max_new_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = model(
                            inputs_embeds=current_embeds,
                            attention_mask=current_mask,
                            use_cache=True,
                            return_dict=True
                        )
                    else:
                        outputs = model(
                            inputs_embeds=current_embeds,
                            attention_mask=current_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True
                        )
                    
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    generated_tokens.append(next_tokens)
                    
                    if (next_tokens == tokenizer.eos_token_id).all():
                        break
                    
                    next_embeds = model.get_input_embeddings()(next_tokens.unsqueeze(-1))
                    current_embeds = next_embeds
                    current_mask = torch.cat([current_mask, torch.ones((batch_size, 1), device=cfg.device, dtype=current_mask.dtype)], dim=1)
            
            if generated_tokens:
                generated = torch.stack(generated_tokens, dim=1)
            else:
                generated = torch.tensor([[]], device=cfg.device, dtype=torch.long).expand(batch_size, 0)
            
            generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                cleaned_text = text.strip()
                meta = metadata_list[i] if i < len(metadata_list) else {}
                task = task_types[i] if i < len(task_types) else 'unknown'
                
                results.append({
                    "task_type": task,
                    "metadata": meta,
                    "model_prediction": cleaned_text
                })

    # 6. Save
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    logger.info("Inference completed successfully.")

if __name__ == "__main__":
    main()
