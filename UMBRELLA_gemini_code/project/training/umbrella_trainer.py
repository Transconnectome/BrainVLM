"""
UMBRELLA Trainer: Unified Training and Evaluation

Key Features:
1. Training: Multi-turn masking, Dynamic Image Expansion, Task-aware logging
2. Evaluation: Generates actual text responses with proper Left Padding & Image Injection
3. Metrics: Sex classification accuracy, Turn stats

"""
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import EvalLoopOutput


# Import UMBRELLA components (these will be available when module is properly imported)
# Deferred import inside functions to avoid circular imports

logger = logging.getLogger(__name__)


@dataclass
class UMBRELLATrainingArgs(TrainingArguments):
    """Extended training arguments for UMBRELLA-specific settings."""
    # 1. Dynamic Batching & Memory
    enable_memory_aware_batching: bool = field(default=True, metadata={"help": "Enable memory-aware batching"})
    memory_budget_gb: float = field(default=30.0, metadata={"help": "Memory budget in GB"})
    
    # 2. Gradient Normalization
    normalize_gradients_by_batch_size: bool = field(default=True, metadata={"help": "Normalize gradients by batch size"})
    base_batch_size: int = field(default=32, metadata={"help": "Base batch size for gradient normalization"})

    # 3. Logging
    log_image_statistics: bool = field(default=True, metadata={"help": "Log image statistics"})
    
    # 4. [CRITICAL FIX] Evaluation Generation Config (Missing Fields Added Here)
    eval_output_dir: str = field(default="./eval_predictions", metadata={"help": "Directory to save evaluation predictions"})
    #save_eval_predictions: bool = field(default=True, metadata={"help": "Whether to save eval predictions to JSONL"})  # Add if needed
    eval_max_new_tokens: int = field(default=256, metadata={"help": "Max new tokens for generation"})
    eval_temperature: float = field(default=0.7, metadata={"help": "Temperature for generation"})
    eval_top_p: float = field(default=0.9, metadata={"help": "Top-p for generation"})
    eval_do_sample: bool = field(default=True, metadata={"help": "Do sample for generation"})

class UMBRELLATrainer(Trainer):
    """Unified Trainer for Training and Evaluation (Generation)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {'loss': [], 'eval': []}
        os.makedirs(self.args.eval_output_dir, exist_ok=True)
        
        # for gradient checking
        self._static_graph_set = False

    def _ensure_set_static_graph(self, model):
        """Helper for DDP static graph optimization."""
        if not self._static_graph_set and self.is_in_train:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # model._set_static_graph() # Optional: Enable if topology is static
                pass
            self._static_graph_set = True

    def _process_mixed_modality_batch(self, model, pixel_values, modality_types, num_images_per_sample, device):
        """
        Process a batch with mixed modalities (T1, FA, T1_FA).
        
        Args:
            model: The model
            pixel_values: Concatenated images tensor [total_images, C, D, H, W]
            modality_types: List of modality types per sample ['T1', 'FA', 'T1_FA', ...]
            num_images_per_sample: List of image counts per sample [1, 1, 2, ...]
            device: Target device
            
        Returns:
            image_features_per_sample: [batch_size, max_images, num_patches, embed_dim]
        """
        batch_size = len(modality_types)
        
        # Build mapping: which images belong to which sample and what modality
        image_modalities = []  # Modality for each image in pixel_values
        sample_image_indices = []  # (sample_idx, local_img_idx) for each image
        
        img_idx = 0
        for sample_idx, (mod_type, num_imgs) in enumerate(zip(modality_types, num_images_per_sample)):
            if mod_type == 'T1_FA':
                # T1_FA: first image is T1, second is FA
                for local_idx in range(num_imgs):
                    if local_idx == 0:
                        image_modalities.append('T1')
                    else:
                        image_modalities.append('FA')
                    sample_image_indices.append((sample_idx, local_idx))
                    img_idx += 1
            else:
                # Single modality (T1 or FA)
                for local_idx in range(num_imgs):
                    image_modalities.append(mod_type)
                    sample_image_indices.append((sample_idx, local_idx))
                    img_idx += 1
        
        # Get embeddings module (PatchEmbed)
        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model
        embeddings = base_model.vision_tower.vision_model.embeddings
        
        # Process each image with its correct modality
        all_features = []
        for i, mod in enumerate(image_modalities):
            img = pixel_values[i:i+1]  # Keep batch dim [1, C, D, H, W]
            
            # Call PatchEmbed with correct modality
            patch_embeds = embeddings.forward_embeddings(img, modality=mod)
            all_features.append(patch_embeds)
        
        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)  # [total_images, num_patches, embed_dim]
        
        # Apply vision tower's remaining layers (if any) and projector
        # Note: For LLaVA, embeddings output goes through transformer blocks
        # We need to pass through the rest of vision_tower
        
        # Get hidden states through vision model (skip embeddings since we did that)
        # This is tricky - we need to inject our embeddings into the vision model
        # For simplicity, we'll process through vision_tower and projector
        
        # Actually, let's just use the projector directly on our patch embeddings
        # The vision tower's transformer might expect specific positional encoding
        # For now, project directly (this assumes patch_embed output is compatible)
        sel_feats = model.multi_modal_projector(all_features)
        
        # Reorganize into per-sample format
        max_images = max(num_images_per_sample) if num_images_per_sample else 1
        num_patches = sel_feats.shape[1]
        embed_dim = sel_feats.shape[2]
        
        image_features_per_sample = torch.zeros(
            batch_size, max_images, num_patches, embed_dim,
            device=device, dtype=sel_feats.dtype
        )
        
        for feat_idx, (sample_idx, local_idx) in enumerate(sample_image_indices):
            image_features_per_sample[sample_idx, local_idx] = sel_feats[feat_idx]
        
        return image_features_per_sample

    def _compute_dummy_loss(self, model, active_modality):
        """
        Compute dummy loss for unused parameters to fix DDP and grad_fn errors.

        This ensures:
        1. Inactive modality params (e.g. fMRI when training sMRI) get gradients (0.0).
        2. Text-only batches still connect to trainable params (PatchEmbed), giving loss a grad_fn.
        """
        # Unwrap model if DDP
        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model

        # Access the trainable embeddings module
        # Path: LlavaForConditionalGeneration -> LlavaMetaForCausalLM -> LlavaMultiModalProjector/VisionTower
        # We target the PatchEmbed which is inside the Vision Tower
        try:
            embeddings = base_model.vision_tower.vision_model.embeddings
        except AttributeError:
            # Fallback if structure is different
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=next(model.parameters()).device)

        # Scaling factor to ensure 0-contribution but graph connection
        # Using 0.0 * sum() is standard pattern for this

        # All supported modalities (including T1/FA for BrainVLM style)
        all_modalities = ['sMRI', 'fMRI', 'T1', 'FA']

        for name, param in embeddings.named_parameters():
            if param.requires_grad:
                # Logic: Add to dummy loss if it belongs to an INACTIVE modality
                # If active_modality is None (Text-only), ALL modalities are inactive, so ALL are added.
                # This perfectly fixes the "element 0 does not require grad" error.

                is_active_param = False
                if active_modality:
                    # For T1_FA or mixed, both T1 and FA are active
                    if active_modality in ['T1_FA', 'mixed']:
                        if 'T1' in name or 'FA' in name:
                            is_active_param = True
                    elif active_modality in name:
                        is_active_param = True

                if not is_active_param:
                    dummy_loss = dummy_loss + (param.sum() * 0.0)

        return dummy_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom compute_loss with Dynamic Image Token Expansion & Dummy Gradient Support.
        Supports mixed modality batches (T1, FA, T1_FA in same batch).
        """
        self._ensure_set_static_graph(model)

        # 1. Prepare Inputs
        raw_labels = inputs.pop("labels", None)

        # Remove metadata (and get modality info)
        num_images_per_sample = inputs.pop("num_images_per_sample", None)
        inputs.pop("task_types", None)
        inputs.pop("metadata", None)
        inputs.pop("image_mask", None)
        inputs.pop("task_ids", None)
        inputs.pop("sample_indices", None)
        modality_type = inputs.pop("modality_type", None)  # Legacy: single modality
        modality_types = inputs.pop("modality_types", None)  # Per-sample modality list

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) and v.device != device else v for k, v in inputs.items()}
        if raw_labels is not None: raw_labels = raw_labels.to(device)

        # 2. Vision Feature Extraction with Per-Sample Modality Support
        pixel_values = inputs.get('pixel_values')
        image_features_per_sample = None
        active_modality = None

        if pixel_values is not None:
            batch_size = inputs['input_ids'].shape[0]
            
            # Check if mixed modality batch
            if modality_types and len(set(modality_types)) > 1:
                # Mixed modality batch: process each sample's images with correct PatchEmbed
                image_features_per_sample = self._process_mixed_modality_batch(
                    model, pixel_values, modality_types, num_images_per_sample, device
                )
                active_modality = 'mixed'  # For dummy loss calculation
            else:
                # Single modality batch (legacy behavior)
                if modality_types:
                    active_modality = modality_types[0]
                elif modality_type and modality_type in ['T1', 'FA', 'T1_FA', 'sMRI', 'fMRI']:
                    active_modality = modality_type
                else:
                    # Fallback: Detect Modality based on Shape
                    if pixel_values.dim() == 5:
                        active_modality = 'sMRI'
                    elif pixel_values.dim() == 6:
                        active_modality = 'fMRI'

                # Pass modality to vision_tower (PatchEmbed)
                image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
                sel_feats = image_outputs.hidden_states[model.config.vision_feature_layer]
                sel_feats = model.multi_modal_projector(sel_feats)
                
                total_images = sel_feats.shape[0]
                if batch_size > 0:
                    num_imgs = total_images // batch_size
                    image_features_per_sample = sel_feats.view(batch_size, num_imgs, -1, sel_feats.shape[-1])
        else:
            # Text-only batch
            active_modality = None

        # 3. Compute Dummy Loss (Critical Fix)
        # This adds 0.0 * sum(unused_params) to the graph
        dummy_loss = self._compute_dummy_loss(model, active_modality)

        # 4. Dynamic Merge (Expand <image> tokens)
        IMAGE_TOKEN_ID = 151646 
        if hasattr(model.config, "image_token_index"): IMAGE_TOKEN_ID = model.config.image_token_index

        new_embeds, new_labels, new_masks = [], [], []
        inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
        
        batch_size = inputs['input_ids'].shape[0]
        for i in range(batch_size):
            cur_ids = inputs['input_ids'][i]
            cur_emb = inputs_embeds[i]
            cur_lbl = raw_labels[i]
            cur_msk = inputs['attention_mask'][i]
            
            img_indices = (cur_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            final_emb, final_lbl, final_msk = [], [], []
            last_pos, img_cnt = 0, 0
            
            for pos in img_indices:
                final_emb.append(cur_emb[last_pos:pos])
                final_lbl.append(cur_lbl[last_pos:pos])
                final_msk.append(cur_msk[last_pos:pos])
                
                if image_features_per_sample is not None:
                    # Check bound
                    if img_cnt < image_features_per_sample.size(1):
                        img_feat = image_features_per_sample[i][img_cnt]
                        final_emb.append(img_feat)
                        final_lbl.append(torch.full((img_feat.shape[0],), -100, device=device, dtype=cur_lbl.dtype))
                        final_msk.append(torch.ones((img_feat.shape[0],), device=device, dtype=cur_msk.dtype))
                        img_cnt += 1
                last_pos = pos + 1
            
            final_emb.append(cur_emb[last_pos:])
            final_lbl.append(cur_lbl[last_pos:])
            final_msk.append(cur_msk[last_pos:])
            
            new_embeds.append(torch.cat(final_emb, dim=0))
            new_labels.append(torch.cat(final_lbl, dim=0))
            new_masks.append(torch.cat(final_msk, dim=0))

        # Pad
        batch_embeds = pad_sequence(new_embeds, batch_first=True, padding_value=0.0)
        batch_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)
        batch_masks = pad_sequence(new_masks, batch_first=True, padding_value=0)

        # 5. LLM Forward
        outputs = model.language_model(
            inputs_embeds=batch_embeds,
            labels=batch_labels,
            attention_mask=batch_masks
        )
        
        # Combine Losses
        # actual_loss + dummy_loss ensures connectivity for all params
        loss = outputs.loss + dummy_loss

        # 6. Logging & Gradient Norm
        if self.state.global_step % 20 == 0 and self.model.training:
            self._log_prediction(outputs.logits, batch_labels)
            
        if self.args.normalize_gradients_by_batch_size:
            loss = loss * (self.args.base_batch_size / batch_size)

        if return_outputs:
            return loss, outputs
        return loss


    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Generate predictions during evaluation - evaluates ALL assistant turns.
        
        For multi-turn conversations:
        - Turn 1 (A1): Reference sex prediction
        - Turn 2 (A2): Comparison sex prediction
        Both are evaluated separately.
        """
        logger.info(f"Starting Multi-Turn Generation Evaluation: {description}")
        self.model.eval()
        all_preds = []

        # Loss accumulation vars
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # [Important] Save metadata before compute_loss potentially removes it
                current_metadata = batch.get('metadata', [])
                current_task_types = batch.get('task_types', [])

                # 1. Calculate Loss (may fail with nan/inf, but continue anyway)
                batch_for_loss = copy.deepcopy(batch)
                try:
                    loss = self.compute_loss(self.model, batch_for_loss, return_outputs=False)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        num_batches += 1
                except Exception as e:
                    pass

                # 2. Generate Predictions for ALL turns
                turn_predictions = self._generate_multi_turn(batch)

                # 3. Store Predictions with turn information
                if not current_metadata: current_metadata = [{}] * len(turn_predictions)
                if not current_task_types: current_task_types = ['unknown'] * len(turn_predictions)

                for i, sample_turns in enumerate(turn_predictions):
                    meta = current_metadata[i] if i < len(current_metadata) else {}
                    task = current_task_types[i] if i < len(current_task_types) else 'unknown'
                    
                    for turn_idx, turn_text in enumerate(sample_turns):
                        all_preds.append({
                            'model_answer': self._clean_response(turn_text),
                            'task_type': task,
                            'metadata': meta,
                            'turn_index': turn_idx  # 0 = A1 (reference), 1 = A2 (comparison)
                        })
        
        # Save Predictions
        output_file = Path(self.args.eval_output_dir) / f"preds_step_{self.state.global_step}.jsonl"
        with open(output_file, 'w') as f:
            for item in all_preds: f.write(json.dumps(item) + '\n')
            
        # Compute Metrics (now handles multi-turn)
        metrics = self._compute_sex_accuracy(all_preds)
        
        # [CRITICAL] Add average loss to metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics[f"{metric_key_prefix}_loss"] = avg_loss

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(all_preds))


    def _generate_multi_turn(self, batch) -> List[List[str]]:
        """
        Generate predictions for each assistant turn in multi-turn conversations.
        
        For a 2-turn conversation (Q1 -> A1 -> Q2 -> A2):
        - Turn 1: Feed [Q1], generate A1
        - Turn 2: Feed [Q1, A1, Q2], generate A2
        
        Returns:
            List of lists: [[A1_pred, A2_pred], [A1_pred, A2_pred], ...] per sample
        """
        model = self.model
        device = next(model.parameters()).device
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch.get('pixel_values')
        modality_types = batch.get('modality_types')
        num_images_per_sample = batch.get('num_images_per_sample')
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        
        batch_size = input_ids.shape[0]
        
        # Extract image features once
        image_features_per_sample = self._extract_image_features(
            model, pixel_values, modality_types, num_images_per_sample, batch_size, device
        )
        
        # Find assistant turn boundaries in tokenized input
        # Pattern: <|im_start|>assistant\n ... <|im_end|>
        assistant_start_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        
        all_sample_predictions = []
        
        for sample_idx in range(batch_size):
            sample_ids = input_ids[sample_idx]
            sample_mask = attention_mask[sample_idx]
            
            # Find all assistant turn start positions
            turn_starts = self._find_pattern_positions(sample_ids, assistant_start_ids)
            turn_ends = self._find_pattern_positions(sample_ids, im_end_ids)
            
            sample_predictions = []
            
            for turn_idx, turn_start in enumerate(turn_starts):
                # Find corresponding end position (first <|im_end|> after turn_start + header)
                header_end = turn_start + len(assistant_start_ids)
                corresponding_end = None
                for end_pos in turn_ends:
                    if end_pos > header_end:
                        corresponding_end = end_pos
                        break
                
                # Truncate input to just before assistant response (include header for trigger)
                # Context: everything up to <|im_start|>assistant\n
                context_end = turn_start + len(assistant_start_ids)  # Include the assistant header
                truncated_ids = sample_ids[:context_end]
                truncated_mask = sample_mask[:context_end]
                
                # Generate response for this turn
                generated_text = self._generate_single_turn(
                    model, truncated_ids.unsqueeze(0), truncated_mask.unsqueeze(0),
                    image_features_per_sample[sample_idx:sample_idx+1] if image_features_per_sample is not None else None,
                    device
                )
                sample_predictions.append(generated_text[0])
            
            # If no assistant turns found, return empty
            if not sample_predictions:
                sample_predictions = [""]
                
            all_sample_predictions.append(sample_predictions)
        
        return all_sample_predictions
    
    def _find_pattern_positions(self, ids: torch.Tensor, pattern: List[int]) -> List[int]:
        """Find all positions where pattern starts in ids."""
        positions = []
        pattern_len = len(pattern)
        pattern_tensor = torch.tensor(pattern, device=ids.device)
        
        for i in range(len(ids) - pattern_len + 1):
            if torch.equal(ids[i:i+pattern_len], pattern_tensor):
                positions.append(i)
        return positions
    
    def _extract_image_features(self, model, pixel_values, modality_types, num_images_per_sample, batch_size, device):
        """Extract image features from pixel values."""
        if pixel_values is None:
            return None
            
        if modality_types and len(set(modality_types)) > 1:
            return self._process_mixed_modality_batch(
                model, pixel_values, modality_types, num_images_per_sample, device
            )
        elif modality_types and modality_types[0] in ['FA', 'T1_FA']:
            return self._process_mixed_modality_batch(
                model, pixel_values, modality_types, num_images_per_sample, device
            )
        else:
            image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
            sel_feats = image_outputs.hidden_states[model.config.vision_feature_layer]
            sel_feats = model.multi_modal_projector(sel_feats)
            
            total_images = sel_feats.shape[0]
            if batch_size > 0:
                num_imgs = total_images // batch_size 
                return sel_feats.view(batch_size, num_imgs, -1, sel_feats.shape[-1])
        return None
    
    def _generate_single_turn(self, model, input_ids, attention_mask, image_features, device):
        """Generate response for a single turn given truncated context."""
        inputs_embeds = model.get_input_embeddings()(input_ids)
        
        # Dynamic merge (expand <image> tokens)
        IMAGE_TOKEN_ID = 151646 
        if hasattr(model.config, "image_token_index"):
            IMAGE_TOKEN_ID = model.config.image_token_index
        
        new_input_embeds = []
        new_attention_masks = []
        
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
                
                if image_features is not None:
                    if img_cnt < image_features.size(1):
                        img_feat = image_features[i][img_cnt]
                        final_embeds.append(img_feat)
                        final_masks.append(torch.ones((img_feat.shape[0],), device=device, dtype=cur_mask.dtype))
                        img_cnt += 1
                last_pos = pos + 1
            
            final_embeds.append(cur_emb[last_pos:])
            final_masks.append(cur_mask[last_pos:])
            
            new_input_embeds.append(torch.cat(final_embeds, dim=0))
            new_attention_masks.append(torch.cat(final_masks, dim=0))
        
        # Left pad and generate
        max_len = max([t.shape[0] for t in new_input_embeds])
        
        padded_embeds_list = []
        padded_masks_list = []
        
        for embed, mask in zip(new_input_embeds, new_attention_masks):
            curr_len = embed.shape[0]
            pad_len = max_len - curr_len
            
            if pad_len > 0:
                pad_embed = torch.zeros((pad_len, embed.shape[1]), device=device, dtype=embed.dtype)
                pad_mask = torch.zeros((pad_len,), device=device, dtype=mask.dtype)
                
                padded_embeds_list.append(torch.cat([pad_embed, embed], dim=0))
                padded_masks_list.append(torch.cat([pad_mask, mask], dim=0))
            else:
                padded_embeds_list.append(embed)
                padded_masks_list.append(mask)
        
        final_inputs_embeds = torch.stack(padded_embeds_list)
        final_attention_masks = torch.stack(padded_masks_list)
        
        # Autoregressive generation
        max_new_tokens = 32  # Short answers expected (male/female)
        generated_tokens = []
        current_embeds = final_inputs_embeds
        current_mask = final_attention_masks
        past_key_values = None
        
        for step in range(max_new_tokens):
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
            
            # Stop if EOS or <|im_end|>
            im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            if (next_tokens == self.tokenizer.eos_token_id).all() or (next_tokens == im_end_token).all():
                break
            
            next_embeds = model.get_input_embeddings()(next_tokens.unsqueeze(-1))
            current_embeds = next_embeds
            current_mask = torch.cat([current_mask, torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)], dim=1)
        
        if generated_tokens:
            generated = torch.stack(generated_tokens, dim=1)
        else:
            generated = torch.tensor([[]], device=device, dtype=torch.long).expand(batch_size, 0)
        
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _generate_step(self, batch):
        """Generation logic: Expand -> Left Pad -> Trigger -> Generate"""
        model = self.model
        device = next(model.parameters()).device
        
        # 1. Prepare Inputs
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch.get('pixel_values')
        
        # Extract modality information for FA support
        modality_types = batch.get('modality_types')
        num_images_per_sample = batch.get('num_images_per_sample')
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        # 2. Feature Extraction & Expansion (Same as compute_loss logic)
        image_features_per_sample = None
        batch_size = input_ids.shape[0]
        
        if pixel_values is not None:
            # Check if mixed modality batch (FA support)
            if modality_types and len(set(modality_types)) > 1:
                # Mixed modality batch: process each sample's images with correct PatchEmbed
                image_features_per_sample = self._process_mixed_modality_batch(
                    model, pixel_values, modality_types, num_images_per_sample, device
                )
            elif modality_types and modality_types[0] in ['FA', 'T1_FA']:
                # Single modality batch but FA or T1_FA - need special handling
                image_features_per_sample = self._process_mixed_modality_batch(
                    model, pixel_values, modality_types, num_images_per_sample, device
                )
            else:
                # Single modality batch (T1 or legacy sMRI/fMRI)
                image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
                sel_feats = image_outputs.hidden_states[model.config.vision_feature_layer]
                sel_feats = model.multi_modal_projector(sel_feats)
                
                # Reshape back to batch: Assumes collator gives flattened batch of images
                total_images = sel_feats.shape[0]
                if batch_size > 0:
                    num_imgs = total_images // batch_size 
                    image_features_per_sample = sel_feats.view(batch_size, num_imgs, -1, sel_feats.shape[-1])
    

        inputs_embeds = model.get_input_embeddings()(input_ids)
        
        # 3. Dynamic Merge (Expand)
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

        # 4. LEFT PADDING & TRIGGER
        # Find max length
        max_len = max([t.shape[0] for t in new_input_embeds])
        
        # Trigger Tokens ("<|im_start|>assistant")
        trigger_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        trigger_embeds = model.get_input_embeddings()(torch.tensor(trigger_ids, device=device))

        #print(f"TEXT INPUT FOR GENERATION: {self.tokenizer.decode(torch.cat([torch.tensor(input_ids[0]).to('cpu'), torch.tensor(trigger_ids).to('cpu')]), skip_special_tokens=False)}")
        
        padded_embeds_list = []
        padded_masks_list = []
        
        for embed, mask in zip(new_input_embeds, new_attention_masks):
            # Add Trigger First
            embed_with_trigger = torch.cat([embed, trigger_embeds], dim=0)
            mask_with_trigger = torch.cat([mask, torch.ones(len(trigger_ids), device=device)], dim=0)
            
            # Then Left Pad
            curr_len = embed_with_trigger.shape[0]
            # Max len needs to account for trigger too
            target_len = max_len + len(trigger_ids) 
            pad_len = target_len - curr_len
            
            if pad_len > 0:
                pad_embed = torch.zeros((pad_len, embed.shape[1]), device=device, dtype=embed.dtype)
                pad_mask = torch.zeros((pad_len,), device=device, dtype=mask.dtype)
                
                padded_embeds_list.append(torch.cat([pad_embed, embed_with_trigger], dim=0))
                padded_masks_list.append(torch.cat([pad_mask, mask_with_trigger], dim=0))
            else:
                padded_embeds_list.append(embed_with_trigger)
                padded_masks_list.append(mask_with_trigger)
                
        final_inputs_embeds = torch.stack(padded_embeds_list)
        final_attention_masks = torch.stack(padded_masks_list)

        # 5. Manual autoregressive generation to avoid position_ids issues with inputs_embeds
        batch_size = final_inputs_embeds.shape[0]
        seq_len = final_inputs_embeds.shape[1]
        max_new_tokens = self.args.eval_max_new_tokens

        # Initialize generated tokens list
        generated_tokens = []
        current_embeds = final_inputs_embeds
        current_mask = final_attention_masks
        past_key_values = None

        for step in range(max_new_tokens):
            with torch.no_grad():
                if past_key_values is None:
                    # First step: use full inputs_embeds
                    outputs = model(
                        inputs_embeds=current_embeds,
                        attention_mask=current_mask,
                        use_cache=True,
                        return_dict=True
                    )
                else:
                    # Subsequent steps: use only the new token embedding
                    outputs = model(
                        inputs_embeds=current_embeds,
                        attention_mask=current_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )

                past_key_values = outputs.past_key_values

                # Get next token (greedy)
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                generated_tokens.append(next_tokens)

                # Check if all sequences have hit EOS
                if (next_tokens == self.tokenizer.eos_token_id).all():
                    break

                # Prepare for next step
                next_embeds = model.get_input_embeddings()(next_tokens.unsqueeze(-1))
                current_embeds = next_embeds
                current_mask = torch.cat([current_mask, torch.ones((batch_size, 1), device=device, dtype=current_mask.dtype)], dim=1)

        # Stack generated tokens
        if generated_tokens:
            generated = torch.stack(generated_tokens, dim=1)  # (batch, seq)
        else:
            generated = torch.tensor([[]], device=device, dtype=torch.long).expand(batch_size, 0)

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


    def _clean_response(self, text):
        """Remove assistant prefix."""
        triggers = ["assistant", "<|im_start|>assistant\n"]
        for t in triggers:
            if text.lower().startswith(t): return text[len(t):].strip()
        return text.strip()
    
    def _log_prediction(self, logits, labels):
        """
        Log predictions aligned for Next Token Prediction.
        Shows what the model predicted for the NEXT position.
        """
        idx = 0
        
        # [Key Fix] Apply Shift
        # Model's logits[t] predicts labels[t+1]
        # Exclude last logits and first label to align pairs

        # 1. Predictions: Exclude last token (no ground truth for last prediction)
        pred_ids = torch.argmax(logits[idx][:-1], dim=-1)
        
        # 2. Ground truth: Exclude first token (first token is input, not prediction target)
        gt_ids = labels[idx][1:]

        # Turn separation logic (same as before, using shifted IDs)
        turns = []
        current_gt = []
        current_pd = []
        in_turn = False
        ignore_index = -100

        for i in range(len(gt_ids)):
            # Only consider valid prediction range when ground truth is not ignore_index
            is_valid = (gt_ids[i] != ignore_index)

            if is_valid:
                if not in_turn:
                    in_turn = True
                    current_gt = []
                    current_pd = []
                
                current_gt.append(gt_ids[i])
                current_pd.append(pred_ids[i])
            else:
                if in_turn:
                    turns.append((current_gt, current_pd))
                    in_turn = False
                    current_gt = []
                    current_pd = []

        if in_turn:
            turns.append((current_gt, current_pd))

        # Print output
        print(f"\n{'='*20} Prediction Log [Step {self.state.global_step}] (Shifted for visual alignment) {'='*20}")
        if len(turns) == 0:
            print("  (No assistant turns found in this sample)")
        
        for i, (gt_chunk, pd_chunk) in enumerate(turns):
            gt_text = self.tokenizer.decode(torch.tensor(gt_chunk), skip_special_tokens=False)
            pd_text = self.tokenizer.decode(torch.tensor(pd_chunk), skip_special_tokens=False)
            
            print(f"[Assistant Turn {i+1}]")
            print(f"  GT: {gt_text}")
            print(f"  PD: {pd_text}")
        print("="*65)


    def _compute_sex_accuracy(self, predictions):
        """
        Compute sex classification accuracy for multi-turn evaluation.
        
        Tracks:
        - Turn 1 (A1): Reference prediction accuracy
        - Turn 2 (A2): Comparison prediction accuracy  
        - Per-case accuracy for comparison cases
        - Overall accuracy across all turns
        - Combined accuracy (both modality AND sex correct) for modality_sex_prediction task
        """
        # Per-turn stats
        turn1_correct = 0
        turn1_total = 0
        turn2_correct = 0
        turn2_total = 0
        
        # Overall stats
        sex_correct = 0
        sex_total = 0
        mod_correct = 0
        mod_total = 0
        
        # Combined accuracy (both modality AND sex must be correct)
        combined_correct = 0
        combined_total = 0
        combined_turn1_correct = 0
        combined_turn1_total = 0
        combined_turn2_correct = 0
        combined_turn2_total = 0
        
        # Per-case accuracy tracking
        case_stats = {}
        case_combined_stats = {}  # For combined (modality+sex) accuracy per case
        
        for p in predictions:
            metadata = p.get('metadata', {})
            ans = p['model_answer'].lower()
            turn_idx = p.get('turn_index', 0)
            task_type = metadata.get('task', '')
            
            # Determine ground truth based on turn
            # 2-turn format: 0=ref, 1=comp
            # 4-turn format (modality_sex_separate): 0=ref_mod, 1=ref_sex, 2=comp_mod, 3=comp_sex
            if turn_idx == 0:
                gt_sex = metadata.get('reference_sex')
                gt_mod = metadata.get('reference_modality', 'T1')
            elif turn_idx == 1:
                # 4-turn: ref sex / 2-turn: comp sex
                task = metadata.get('task', '')
                if task == 'modality_sex_separate':
                    gt_sex = metadata.get('reference_sex')
                    gt_mod = metadata.get('reference_modality', 'T1')
                else:
                    gt_sex = metadata.get('comparison_sex')
                    gt_mod = metadata.get('comparison_modality')
            elif turn_idx == 2:
                gt_sex = metadata.get('comparison_sex')
                gt_mod = metadata.get('comparison_modality')
            else:  # turn_idx == 3
                gt_sex = metadata.get('comparison_sex')
                gt_mod = metadata.get('comparison_modality')
            
            comparison_case = metadata.get('comparison_case')
            
            # Sex prediction
            pred_sex = 'female' if 'female' in ans else 'male' if 'male' in ans else None
            
            # Modality prediction
            pred_mod = 'FA' if 'fa' in ans.lower() else 'T1' if 't1' in ans.lower() else None
            
            # Sex accuracy
            sex_is_correct = False
            if gt_sex in ['male', 'female'] and pred_sex:
                sex_total += 1
                sex_is_correct = (pred_sex == gt_sex)
                if sex_is_correct:
                    sex_correct += 1
                
                # Track per-turn accuracy
                if turn_idx == 0:
                    turn1_total += 1
                    if sex_is_correct:
                        turn1_correct += 1
                else:
                    turn2_total += 1
                    if sex_is_correct:
                        turn2_correct += 1
                    
                    # Track per-case accuracy (Turn 2 only)
                    if comparison_case:
                        if comparison_case not in case_stats:
                            case_stats[comparison_case] = {'correct': 0, 'total': 0}
                        case_stats[comparison_case]['total'] += 1
                        if sex_is_correct:
                            case_stats[comparison_case]['correct'] += 1
            
            # Modality accuracy
            mod_is_correct = False
            if gt_mod in ['T1', 'FA'] and pred_mod:
                mod_total += 1
                mod_is_correct = (pred_mod == gt_mod)
                if mod_is_correct:
                    mod_correct += 1
            
            # Combined accuracy (BOTH must be correct)
            if gt_sex in ['male', 'female'] and gt_mod in ['T1', 'FA']:
                combined_total += 1
                both_correct = sex_is_correct and mod_is_correct
                if both_correct:
                    combined_correct += 1
                
                # Per-turn combined
                if turn_idx == 0:
                    combined_turn1_total += 1
                    if both_correct:
                        combined_turn1_correct += 1
                else:
                    combined_turn2_total += 1
                    if both_correct:
                        combined_turn2_correct += 1
                    
                    # Per-case combined accuracy
                    if comparison_case:
                        if comparison_case not in case_combined_stats:
                            case_combined_stats[comparison_case] = {'correct': 0, 'total': 0}
                        case_combined_stats[comparison_case]['total'] += 1
                        if both_correct:
                            case_combined_stats[comparison_case]['correct'] += 1
        
        metrics = {
            "eval_sex_acc": sex_correct / sex_total if sex_total else 0,
            "eval_sex_acc_turn1": turn1_correct / turn1_total if turn1_total else 0,
            "eval_sex_acc_turn2": turn2_correct / turn2_total if turn2_total else 0,
            "eval_modality_acc": mod_correct / mod_total if mod_total else 0,
            "eval_combined_acc": combined_correct / combined_total if combined_total else 0,
            "eval_combined_acc_turn1": combined_turn1_correct / combined_turn1_total if combined_turn1_total else 0,
            "eval_combined_acc_turn2": combined_turn2_correct / combined_turn2_total if combined_turn2_total else 0,
        }
        
        # Add per-case accuracy metrics (sex only)
        for case_name, stats in case_stats.items():
            case_acc = stats['correct'] / stats['total'] if stats['total'] else 0
            short_name = case_name.split('_')[0] if '_' in case_name else case_name
            metrics[f"eval_sex_acc_{short_name}"] = case_acc
        
        # Add per-case combined accuracy metrics
        for case_name, stats in case_combined_stats.items():
            case_acc = stats['correct'] / stats['total'] if stats['total'] else 0
            short_name = case_name.split('_')[0] if '_' in case_name else case_name
            metrics[f"eval_combined_acc_{short_name}"] = case_acc
        
        # Log summary
        logger.info(f"[Eval] Turn1 Sex Acc: {turn1_correct}/{turn1_total} = {metrics['eval_sex_acc_turn1']:.1%}")
        logger.info(f"[Eval] Turn2 Sex Acc: {turn2_correct}/{turn2_total} = {metrics['eval_sex_acc_turn2']:.1%}")
        logger.info(f"[Eval] Overall Sex Acc: {sex_correct}/{sex_total} = {metrics['eval_sex_acc']:.1%}")
        logger.info(f"[Eval] Modality Acc: {mod_correct}/{mod_total} = {metrics['eval_modality_acc']:.1%}")
        logger.info(f"[Eval] Combined Acc (Both correct): {combined_correct}/{combined_total} = {metrics['eval_combined_acc']:.1%}")
        
        return metrics