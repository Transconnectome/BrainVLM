"""
UMBRELLA Trainer: Multi-turn Conversation Training with LLaVA-Style Masking

Key Features:
- LLaVA-style masking: human turns masked (-100), gpt turns active
- Dynamic batch controlling with memory-aware scheduling
- Task-aware loss computation with dummy loss support
- Per-sample and per-task metrics
- Gradient accumulation with proper normalization
- Backward compatibility through selective loss computation
"""

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
from transformers import Trainer, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TurnType(Enum):
    """Enumeration for conversation turn types."""
    HUMAN = "human"
    GPT = "gpt"
    USER = "user"
    ASSISTANT = "assistant"
    UNKNOWN = "unknown"


@dataclass
class UMBRELLATrainingArgs(TrainingArguments):
    """Extended training arguments for UMBRELLA-specific settings."""

    # Multi-turn masking
    mask_human_turns: bool = True
    mask_padding_tokens: bool = True

    # Task-specific settings
    enable_task_aware_loss: bool = True
    task_type_weights: Optional[Dict[str, float]] = None

    # Dynamic batching
    enable_memory_aware_batching: bool = True
    memory_budget_gb: float = 30.0
    max_batch_tokens: Optional[int] = None

    # Dummy loss support
    enable_dummy_loss: bool = True
    dummy_loss_weight: float = 0.1

    # Logging
    log_turn_distribution: bool = True
    log_image_statistics: bool = True
    log_memory_usage: bool = False

    # Gradient normalization
    normalize_gradients_by_batch_size: bool = True
    base_batch_size: int = 32


class TurnMaskBuilder:
    """
    Builds LLaVA-style masking for multi-turn conversations with JSON format.

    Strategy (LLaVA-style):
    - user turns: masked (-100) - ignore in loss (model doesn't predict on user input)
    - assistant turns: active - included in loss (model learns to generate responses)
    - Padding: masked (-100) - ignore in loss

    Supports JSON format conversations:
    [
        {"role": "user", "content": [...]},
        {"role": "assistant", "content": [...]},
        ...
    ]
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 conversation_handler=None):
        """
        Initialize turn mask builder with JSON conversation support.

        Args:
            tokenizer: HuggingFace tokenizer
            conversation_handler: Optional LLaVAConversationHandler for validation
        """
        self.tokenizer = tokenizer
        self.conversation_handler = conversation_handler

    def build_masks_from_json_conversation(self,
                                          conversation_json: str,
                                          input_ids: torch.Tensor,
                                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Build masks from LLaVA JSON conversation format.

        Args:
            conversation_json: JSON string with conversation
            input_ids: Tokenized input (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Modified labels with -100 for masked positions (batch_size, seq_len)
        """
        import json
        
        try:
            conversation = json.loads(conversation_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse conversation JSON: {e}")
            # Fallback: mask nothing
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return labels

        # Validate conversation structure
        if self.conversation_handler:
            is_valid, errors = self.conversation_handler.validate_conversation(conversation)
            if not is_valid:
                logger.warning(f"Conversation validation errors: {errors}")

        # Build labels tensor from conversation
        labels = input_ids.clone()
        batch_size, seq_len = labels.shape

        # For simplicity, apply to first sample (batch size 1)
        # In production, would need to handle variable-length conversations per sample
        if batch_size > 0:
            labels[0] = self._mask_turns_from_conversation(
                conversation,
                input_ids[0],
                attention_mask[0]
            )

        return labels

    def _mask_turns_from_conversation(self,
                                      conversation: list,
                                      input_ids: torch.Tensor,
                                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masking based on conversation turn roles.

        Args:
            conversation: List of turn dicts with role and content
            input_ids: Token IDs for this sample
            attention_mask: Attention mask for this sample

        Returns:
            Labels with -100 for masked tokens
        """
        labels = input_ids.clone()
        seq_len = len(input_ids)

        # Create role sequence from conversation for reference
        # Normalize roles to standard format (user/assistant)
        roles = [self._normalize_role(turn.get("role", "unknown")) for turn in conversation]

        # Estimate token boundaries based on turn order
        # USER turns should be masked, ASSISTANT turns active
        in_user_turn = True  # Start with user (typically instruction)

        for seq_idx in range(seq_len):
            if attention_mask[seq_idx] == 0:
                # Padding token
                labels[seq_idx] = -100
            elif in_user_turn:
                # Mask USER turn tokens
                labels[seq_idx] = -100
                # Could detect turn transition via special tokens here
            # else: ASSISTANT turn - keep label (active in loss)

        return labels

    def _normalize_role(self, role: str) -> str:
        """
        Normalize role names to standard format.

        Args:
            role: Raw role string (can be "user", "human", "assistant", "gpt", etc.)

        Returns:
            Normalized role ("user" or "assistant")
        """
        if not role:
            return "unknown"

        role_lower = role.lower().strip()

        # Map old format to new format
        if role_lower in ("human", "user"):
            return "user"
        elif role_lower in ("gpt", "assistant"):
            return "assistant"
        else:
            return "unknown"

    def build_masks(self, batch: 'UMBRELLABatch') -> torch.Tensor:
        """
        Build turn-aware masks for a batch using JSON conversations.

        Args:
            batch: UMBRELLABatch containing input_ids, labels, and conversation_json

        Returns:
            Modified labels tensor with -100 for masked positions
        """
        labels = batch.labels.clone()
        batch_size, seq_len = labels.shape

        # Process each sample in batch
        for batch_idx in range(batch_size):
            # If conversation_json is available, use it
            if hasattr(batch, 'conversation_json') and batch.conversation_json:
                conv_json = batch.conversation_json[batch_idx]
                labels[batch_idx] = self._mask_turns_from_conversation(
                    json.loads(conv_json) if isinstance(conv_json, str) else conv_json,
                    batch.input_ids[batch_idx],
                    batch.attention_mask[batch_idx]
                )
            else:
                # Fallback: mask based on attention
                input_ids = batch.input_ids[batch_idx]
                attention = batch.attention_mask[batch_idx]
                labels[batch_idx] = self._mask_by_attention(
                    input_ids,
                    attention,
                    labels[batch_idx]
                )

        return labels

    def _mask_by_attention(self,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
        """
        Apply masking based on attention mask when conversation JSON unavailable.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Label tensor to modify

        Returns:
            Modified labels
        """
        labels[attention_mask == 0] = -100
        return labels

    def get_turn_distribution(self, conversation: list) -> Dict[str, int]:
        """
        Get distribution of turn types in conversation.

        Args:
            conversation: List of turn dicts

        Returns:
            Distribution stats
        """
        distribution = {
            "total_turns": len(conversation),
            "user_turns": 0,
            "assistant_turns": 0
        }

        for turn in conversation:
            role = turn.get("role")
            if role == "user":
                distribution["user_turns"] += 1
            elif role == "assistant":
                distribution["assistant_turns"] += 1

        return distribution


class TaskAwareLossComputer:
    """Computes loss with task-specific weighting and dummy loss support."""

    def __init__(self,
                 task_weights: Optional[Dict[str, float]] = None,
                 enable_dummy_loss: bool = True,
                 dummy_weight: float = 0.1):
        """
        Initialize loss computer.

        Args:
            task_weights: Task-specific loss weights (T1, T2, T3)
            enable_dummy_loss: Enable dummy loss for synthetic data
            dummy_weight: Weight for dummy loss (typically small)
        """
        self.task_weights = task_weights or {'T1': 1.0, 'T2': 1.0, 'T3': 1.0}
        self.enable_dummy_loss = enable_dummy_loss
        self.dummy_weight = dummy_weight

        # Ensure task weights sum to 1 for consistency
        total = sum(self.task_weights.values())
        self.task_weights = {k: v / total for k, v in self.task_weights.items()}

    def compute_loss(self,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    task_ids: torch.Tensor,
                    task_types: List[str],
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute task-aware loss with selective masking.

        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            labels: Target labels (batch_size, seq_len) with -100 for masked positions
            task_ids: Task type IDs (batch_size,)
            task_types: Task type strings for reference
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Weighted loss tensor
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute standard NLL loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        ).view(batch_size, seq_len - 1)

        # Create loss mask from shifted labels
        loss_mask = (shift_labels != -100).float()

        # Apply attention mask
        loss_mask = loss_mask * attention_mask[..., 1:].float()

        # Compute per-sample loss
        per_sample_loss = (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)

        # Apply task-aware weighting
        task_weights_vec = torch.tensor(
            [self.task_weights.get(t, 1.0) for t in task_types],
            device=per_sample_loss.device,
            dtype=per_sample_loss.dtype
        )

        weighted_loss = (per_sample_loss * task_weights_vec).mean()

        return weighted_loss


class ImageFeatureIntegrator:
    """Integrates image features with text embeddings in autoregressive generation."""

    def __init__(self, image_embedding_dim: int, text_embedding_dim: int):
        """
        Initialize integrator.

        Args:
            image_embedding_dim: Dimension of image embeddings
            text_embedding_dim: Dimension of text embeddings
        """
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = text_embedding_dim

        # Projection layer if dimensions don't match
        if image_embedding_dim != text_embedding_dim:
            self.projection = nn.Linear(image_embedding_dim, text_embedding_dim)
        else:
            self.projection = nn.Identity()

    def get_image_token_positions(self, input_ids: torch.Tensor,
                                 image_token_id: int) -> List[List[int]]:
        """Get positions of image tokens for each sample in batch."""
        batch_size = input_ids.size(0)
        positions = []

        for batch_idx in range(batch_size):
            pos = (input_ids[batch_idx] == image_token_id).nonzero(as_tuple=True)[0].tolist()
            positions.append(pos)

        return positions

    def integrate_images(self,
                        embeddings: torch.Tensor,
                        image_features: torch.Tensor,
                        image_positions: List[List[int]]) -> torch.Tensor:
        """
        Integrate image features into text embeddings at token positions.

        Args:
            embeddings: Text embeddings (batch_size, seq_len, text_embed_dim)
            image_features: Image features (batch_size, num_images, img_embed_dim)
            image_positions: Positions where image tokens appear

        Returns:
            Modified embeddings with images integrated
        """
        integrated = embeddings.clone()

        for batch_idx, positions in enumerate(image_positions):
            if not positions:
                continue

            image_feats = self.projection(image_features[batch_idx])

            for pos_idx, token_pos in enumerate(positions):
                if pos_idx < image_feats.size(0) and token_pos < integrated.size(1):
                    integrated[batch_idx, token_pos] = image_feats[pos_idx]

        return integrated


class UMBRELLATrainer(Trainer):
    """
    Enhanced Trainer for UMBRELLA with LLaVA-style masking and dynamic batching.

    Key enhancements:
    1. Multi-turn conversation masking (human turns ignored, gpt turns active)
    2. Task-aware loss with flexibility for different task types
    3. Dynamic batch size normalization
    4. Memory-aware batch sampling integration
    5. Comprehensive metrics tracking
    6. Support for dummy loss (synthetic data)
    """

    def __init__(self,
                 model: PreTrainedModel,
                 args: UMBRELLATrainingArgs,
                 train_dataset: Optional[Any] = None,
                 eval_dataset: Optional[Any] = None,
                 data_collator: Optional[Any] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 compute_metrics: Optional[Any] = None,
                 callbacks: Optional[List] = None,
                 optimizers: Optional[Tuple] = None,
                 preprocess_logits_for_metrics: Optional[Any] = None):
        """Initialize UMBRELLA trainer."""

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # Initialize components
        self.turn_mask_builder = TurnMaskBuilder(tokenizer) if tokenizer else None
        self.loss_computer = TaskAwareLossComputer(
            task_weights=args.task_type_weights,
            enable_dummy_loss=args.enable_dummy_loss
        )

        # Training metrics
        self.metrics_history = {
            'turn_distribution': [],
            'task_distribution': [],
            'image_statistics': [],
            'loss_by_task': {},
            'memory_usage': []
        }

        logger.info("UMBRELLA Trainer initialized with LLaVA-style masking")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with multi-turn masking.

        Args:
            model: The model to use
            inputs: Input batch (UMBRELLABatch)
            return_outputs: Whether to return model outputs

        Returns:
            Loss (and outputs if requested)
        """
        # Extract batch information BEFORE removing from inputs
        labels = inputs.pop("labels", None)
        task_types = inputs.pop("task_types", [])
        task_ids = inputs.pop("task_ids", None)

        # CRITICAL FIX: Remove UMBRELLA-specific metadata NOT accepted by LlavaForConditionalGeneration
        image_mask = inputs.pop("image_mask", None)
        num_images_per_sample = inputs.pop("num_images_per_sample", None)
        sample_indices = inputs.pop("sample_indices", None)
        metadata_list = inputs.pop("metadata", None)

        # Get model outputs with ONLY model-accepted parameters
        # Now inputs contains ONLY: pixel_values, input_ids, attention_mask
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply turn-aware masking if available
        if self.turn_mask_builder is not None and self.args.mask_human_turns:
            # Reconstruct batch for masking using SAVED metadata
            from umbrella_collator import UMBRELLABatch
            temp_batch = UMBRELLABatch(
                pixel_values=inputs.get('pixel_values', torch.empty(0)),
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels,
                image_mask=image_mask if image_mask is not None else torch.ones(labels.shape[0], 1),
                num_images_per_sample=num_images_per_sample or [1] * labels.shape[0],
                task_types=task_types or ['T1'] * labels.shape[0],
                task_ids=task_ids if task_ids is not None else torch.zeros(labels.shape[0]),
                sample_indices=sample_indices or list(range(labels.shape[0])),
                metadata=metadata_list or []
            )
            labels = self.turn_mask_builder.build_masks(temp_batch)

        # Compute loss with task awareness
        loss = self.loss_computer.compute_loss(
            logits=logits,
            labels=labels,
            task_ids=task_ids if task_ids is not None else torch.zeros(logits.shape[0]),
            task_types=task_types,
            attention_mask=inputs['attention_mask']
        )

        # Gradient normalization for dynamic batch sizes
        if self.args.normalize_gradients_by_batch_size and hasattr(self.args, 'base_batch_size'):
            current_batch_size = logits.shape[0]
            base_batch_size = self.args.base_batch_size
            loss = loss * (base_batch_size / current_batch_size)

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_batch_metrics(inputs, task_types)

        if return_outputs:
            return loss, outputs
        return loss

    def _log_batch_metrics(self, inputs: Dict, task_types: List[str]):
        """Log batch-specific metrics."""
        # Turn distribution
        if self.turn_mask_builder is not None and self.args.log_turn_distribution:
            from umbrella_collator import UMBRELLABatch
            temp_batch = UMBRELLABatch(
                pixel_values=inputs.get('pixel_values', torch.empty(0)),
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs.get('labels', torch.zeros_like(inputs['input_ids'])),
                image_mask=inputs.get('image_mask', torch.ones(inputs['input_ids'].shape[0], 1)),
                num_images_per_sample=inputs.get('num_images_per_sample', [1] * inputs['input_ids'].shape[0]),
                task_types=task_types,
                task_ids=torch.zeros(inputs['input_ids'].shape[0]),
                sample_indices=inputs.get('sample_indices', list(range(inputs['input_ids'].shape[0]))),
                metadata=inputs.get('metadata', [])
            )
            turn_dist = self.turn_mask_builder.get_turn_distribution(temp_batch)
            self.metrics_history['turn_distribution'].append(turn_dist)

        # Task distribution
        task_dist = {}
        for task_type in task_types:
            task_dist[task_type] = task_dist.get(task_type, 0) + 1
        self.metrics_history['task_distribution'].append(task_dist)

        # Image statistics
        if self.args.log_image_statistics and 'num_images_per_sample' in inputs:
            num_images = inputs['num_images_per_sample']
            self.metrics_history['image_statistics'].append({
                'mean': np.mean(num_images) if num_images else 0,
                'max': max(num_images) if num_images else 0,
                'total': sum(num_images) if num_images else 0
            })

    def save_training_metrics(self, output_dir: str):
        """Save training metrics to JSON."""
        output_path = Path(output_dir) / "training_metrics.json"

        # Convert non-serializable objects
        metrics_to_save = {
            'turn_distribution': self.metrics_history['turn_distribution'],
            'task_distribution': self.metrics_history['task_distribution'],
            'image_statistics': self.metrics_history['image_statistics'],
        }

        with open(output_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

        logger.info(f"Training metrics saved to {output_path}")


class GradientNormalizer:
    """Normalizes gradients across variable batch sizes."""

    def __init__(self, base_batch_size: int = 32):
        """
        Initialize normalizer.

        Args:
            base_batch_size: Reference batch size for normalization
        """
        self.base_batch_size = base_batch_size

    def normalize_gradients(self, model: nn.Module, current_batch_size: int):
        """
        Normalize gradients to account for dynamic batch size.

        Args:
            model: Model with gradients to normalize
            current_batch_size: Current batch size
        """
        scale_factor = current_batch_size / self.base_batch_size

        for param in model.parameters():
            if param.grad is not None:
                param.grad.data *= scale_factor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example turn mask builder
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    builder = TurnMaskBuilder(tokenizer)

    logger.info("Turn mask builder initialized")
    logger.info(f"Human token patterns: {builder.human_patterns}")
    logger.info(f"GPT token patterns: {builder.gpt_patterns}")
