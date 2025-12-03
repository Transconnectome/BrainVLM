"""
UMBRELLA Training Script with LLaVA-Next Format Support

This script implements complete UMBRELLA training pipeline with:
- LLaVA-Next conversation format support
- Custom 3D/4D brain MRI patch embedding
- Multi-turn conversation masking
- Directory-based dataset loading (multiple JSON files)
- Task-aware loss computation
- Memory-aware batching

Key Features:
1. LlavaForConditionalGeneration with custom PatchEmbed
2. Multi-turn masking: human turns masked, assistant turns active
3. Supports both file and directory inputs for train/eval data
4. Task filtering for specific conversation types
5. Memory-efficient training with gradient checkpointing

Usage:
    python main_umbrella_training_fixed.py \
        --config umbrella_llava_train.yaml \
        --train-data /path/to/train.json \
        --eval-data /path/to/eval.json \
        --modality T1
"""

import sys
import logging
import argparse
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration
)

# Import UMBRELLA components
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset.umbrella_dataset_fixed import UMBRELLADataset
from dataset.umbrella_collator import UMBRELLACollator, MemoryAwareUMBRELLACollator
from training.umbrella_trainer import UMBRELLATrainer, UMBRELLATrainingArgs
from model.patch_embed import PatchEmbed

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class UMBRELLATrainingConfig:
    """
    Unified configuration for UMBRELLA training.

    This configuration class consolidates all training settings and provides
    a factory method to convert to HuggingFace TrainingArguments.

    Loaded from YAML file or created programmatically.
    """

    # Model settings
    model_name: str = "llava-hf/llava-interleave-qwen-0.5b-hf"
    tokenizer_name: Optional[str] = None  # If None, use model_name

    # Training data (now supports directories)
    train_json_path: str = "./data/train.json"  # Can be file OR directory
    eval_json_path: Optional[str] = None  # Can be file OR directory

    # Task filtering
    task_filter: Optional[str] = None  # e.g., 'same_sex_comparison', 'different_sex_comparison'

    # Modality settings
    modality: str = "T1"  # Primary modality (T1, rsfMRI, etc.)
    img_size: List[int] = None  # Image dimensions [H, W, D] or [H, W, D, T]
    patch_size: List[int] = None  # Patch size for modality

    # Multi-modality settings (for custom PatchEmbed)
    T1_img_size: List[int] = None
    T1_patch_size: List[int] = None
    rsfMRI_img_size: List[int] = None
    rsfMRI_patch_size: List[int] = None

    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    num_epochs: int = 50
    max_seq_length: int = 2048
    max_images_per_sample: int = 10

    # Memory and performance
    enable_memory_aware_batching: bool = True
    memory_budget_gb: float = 30.0
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # or "fp16"

    # Multi-turn masking
    mask_human_turns: bool = True
    mask_padding_tokens: bool = True

    # Task-aware loss
    enable_task_aware_loss: bool = True
    task_type_weights: Optional[Dict[str, float]] = None

    # Dummy loss support
    enable_dummy_loss: bool = True
    dummy_loss_weight: float = 0.1

    # Logging and saving
    output_dir: str = "./hf_results/umbrella"
    logging_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    warmup_steps: int = 500

    # Advanced logging
    log_turn_distribution: bool = True
    log_image_statistics: bool = True
    log_memory_usage: bool = False

    # Gradient normalization
    normalize_gradients_by_batch_size: bool = True
    base_batch_size: int = 32

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "umbrella-training"
    wandb_api_key: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> 'UMBRELLATrainingConfig':
        """
        Create config from YAML file.

        Args:
            config_path: Path to umbrella_llava_train.yaml

        Returns:
            UMBRELLATrainingConfig instance
        """
        yaml_config = load_config(config_path)

        # Extract relevant settings
        dataset_config = yaml_config.get('dataset', {})
        model_config = yaml_config.get('model', {})
        trainer_config = yaml_config.get('trainer', {})

        # Determine modality (default to T1)
        modality = 'T1'  # TODO: Make this configurable
        modality_dataset_config = dataset_config.get(modality, {})

        # Get image size and patch size from config
        img_size = modality_dataset_config.get('img_size', [96, 96, 96])
        patch_size = model_config.get(modality, {}).get('patch_size', [16, 16, 16])

        # Get T1 and rsfMRI settings for custom PatchEmbed
        T1_config = dataset_config.get('T1', {})
        T1_model_config = model_config.get('T1', {})
        rsfMRI_config = dataset_config.get('rsfMRI', {})
        rsfMRI_model_config = model_config.get('rsfMRI', {})

        # Create config instance
        return cls(
            model_name=model_config.get('hf_name', 'llava-hf/llava-interleave-qwen-0.5b-hf'),
            modality=modality,
            img_size=img_size,
            patch_size=patch_size,
            T1_img_size=T1_config.get('img_size', [96, 96, 96]),
            T1_patch_size=T1_model_config.get('patch_size', [10, 10, 10]),
            rsfMRI_img_size=rsfMRI_config.get('img_size', [96, 96, 96, 24]),
            rsfMRI_patch_size=rsfMRI_model_config.get('patch_size', [16, 16, 16, 3]),
            batch_size=trainer_config.get('per_device_batch_size', 2),
            gradient_accumulation_steps=trainer_config.get('gradient_accumulation_steps', 1),
            learning_rate=trainer_config.get('learning_rate', 5e-5),
            num_epochs=trainer_config.get('max_epochs', 50),
            max_seq_length=2048,  # LLaVA default
            gradient_checkpointing=trainer_config.get('gradient_checkpointing', True),
            warmup_steps=trainer_config.get('warmup_steps', 500),
            output_dir=trainer_config.get('ckpt_dir', './hf_results/umbrella'),
            wandb_api_key=yaml_config.get('wandb', {}).get('API_KEY')
        )

    def to_training_args(self, eval_dataset_available: bool = False) -> UMBRELLATrainingArgs:
        """
        Convert config to UMBRELLATrainingArgs for HuggingFace Trainer.

        This factory method creates a properly configured UMBRELLATrainingArgs instance
        from the high-level configuration, ensuring all UMBRELLA-specific attributes
        are correctly set.

        Args:
            eval_dataset_available: Whether evaluation dataset is available

        Returns:
            UMBRELLATrainingArgs instance ready for UMBRELLATrainer
        """
        return UMBRELLATrainingArgs(
            # Standard HuggingFace TrainingArguments
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps if eval_dataset_available else None,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.mixed_precision == "fp16",
            bf16=self.mixed_precision == "bf16",
            save_strategy="steps",
            evaluation_strategy="steps" if eval_dataset_available else "no",
            logging_strategy="steps",
            report_to="wandb" if self.use_wandb else "none",
            load_best_model_at_end=eval_dataset_available,
            metric_for_best_model="loss" if eval_dataset_available else None,
            greater_is_better=False if eval_dataset_available else None,
            gradient_checkpointing=self.gradient_checkpointing,

            # UMBRELLA-specific arguments
            mask_human_turns=self.mask_human_turns,
            mask_padding_tokens=self.mask_padding_tokens,
            enable_task_aware_loss=self.enable_task_aware_loss,
            task_type_weights=self.task_type_weights,
            enable_memory_aware_batching=self.enable_memory_aware_batching,
            memory_budget_gb=self.memory_budget_gb,
            enable_dummy_loss=self.enable_dummy_loss,
            dummy_loss_weight=self.dummy_loss_weight,
            log_turn_distribution=self.log_turn_distribution,
            log_image_statistics=self.log_image_statistics,
            log_memory_usage=self.log_memory_usage,
            normalize_gradients_by_batch_size=self.normalize_gradients_by_batch_size,
            base_batch_size=self.base_batch_size,
        )


def create_llava_model_with_custom_patch_embed(config: UMBRELLATrainingConfig) -> LlavaForConditionalGeneration:
    """
    Create LlavaForConditionalGeneration model with custom 3D/4D brain MRI patch embedding.

    This function:
    1. Loads pre-trained LlavaForConditionalGeneration model
    2. Replaces default patch embedding with custom PatchEmbed for brain MRI
    3. Applies freezing strategy: ONLY custom patch embedding is trainable
    4. Enables gradient checkpointing for memory efficiency

    Freezing Strategy:
    - Vision Encoder (encoder layers, layernorms): FROZEN
    - Custom Patch Embedding (our 3D/4D MRI embedding): TRAINABLE
    - Multi-Modal Projector (LLaVA's projection): FROZEN
    - Language Model (entire LLM + lm_head): FROZEN

    Why this strategy?
    - Pre-trained vision encoder already understands visual features
    - Pre-trained language model already understands language
    - Only need to learn how to convert brain MRI patches to features
    - Custom patch embedding is the ONLY brain-specific component

    Args:
        config: Training configuration with model settings

    Returns:
        LlavaForConditionalGeneration model with custom patch embedding and proper freezing

    Raises:
        RuntimeError: If model structure is unexpected
    """
    logger.info("=" * 80)
    logger.info("INITIALIZING LlavaForConditionalGeneration WITH CUSTOM PATCH EMBED")
    logger.info("=" * 80)

    # Step 1: Load pre-trained LlavaForConditionalGeneration
    logger.info(f"Loading pre-trained model: {config.model_name}")
    model = LlavaForConditionalGeneration.from_pretrained(config.model_name)
    logger.info("  Pre-trained model loaded successfully")

    # Step 2: Create custom PatchEmbed for brain MRI
    logger.info("Creating custom PatchEmbed for brain MRI (3D/4D volumes)...")
    logger.info(f"  T1 image size: {config.T1_img_size}")
    logger.info(f"  T1 patch size: {config.T1_patch_size}")
    logger.info(f"  rsfMRI image size: {config.rsfMRI_img_size}")
    logger.info(f"  rsfMRI patch size: {config.rsfMRI_patch_size}")

    # Get embedding dimension from original model
    original_patch_embedding = model.vision_tower.vision_model.embeddings.patch_embedding
    embed_dim = int(original_patch_embedding.out_channels)
    logger.info(f"  Embedding dimension: {embed_dim}")

    # Initialize custom patch embedding
    patch_embed = PatchEmbed(
        T1_size=config.T1_img_size,
        T1_patch_size=config.T1_patch_size,
        rsfMRI_size=config.rsfMRI_img_size,
        rsfMRI_patch_size=config.rsfMRI_patch_size,
        embed_dim=embed_dim
    )
    logger.info("  Custom PatchEmbed created")

    # Step 3: Replace patch embedding
    logger.info("Replacing original patch_embedding with custom PatchEmbed...")
    model.vision_tower.vision_model.embeddings.patch_embedding = patch_embed
    logger.info("  Patch embedding replaced successfully")

    # Step 4: Apply freezing strategy
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING FREEZING STRATEGY")
    logger.info("=" * 80)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    logger.info("1. Froze all model parameters")

    # Unfreeze ONLY custom patch embedding
    for param in model.vision_tower.vision_model.embeddings.patch_embedding.parameters():
        param.requires_grad = True
    logger.info("2. Unfroze custom patch embedding (TRAINABLE)")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Trainable percentage: {100.0 * trainable_params / total_params:.4f}%")
    logger.info("=" * 80 + "\n")

    # Step 5: Enable gradient checkpointing if configured
    if config.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for memory efficiency...")
        model.gradient_checkpointing_enable()
        logger.info("  Gradient checkpointing enabled")

    logger.info("\n" + "=" * 80)
    logger.info("MODEL INITIALIZATION COMPLETE")
    logger.info("=" * 80 + "\n")

    return model


class UMBRELLATrainingPipeline:
    """Complete UMBRELLA training pipeline with config support."""

    def __init__(self, config: UMBRELLATrainingConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("=" * 80)
        logger.info("UMBRELLA Training Pipeline (UNIFIED CONFIG VERSION)")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Modality: {config.modality}")
        logger.info(f"Image size: {config.img_size}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        if config.task_filter:
            logger.info(f"Task filter: {config.task_filter}")

    def setup_model(self) -> Tuple[LlavaForConditionalGeneration, AutoTokenizer]:
        """
        Load and setup LLaVA model with custom patch embedding and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer_name = self.config.tokenizer_name or self.config.model_name

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add special tokens for LLaVA-Next format
        special_tokens = {
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>',  # LLaVA-Next turn markers
                '<image>',  # Generic image token
            ]
        }

        num_added = tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens")

        # Ensure tokenizer has required tokens
        if '<|im_start|>' not in tokenizer.get_vocab():
            logger.error("Tokenizer does not support <|im_start|> token!")
            logger.warning("Proceeding anyway, but training may fail")

        # Create model with custom patch embedding
        logger.info(f"Initializing model with custom patch embedding...")
        model = create_llava_model_with_custom_patch_embed(self.config)

        # Resize embeddings to accommodate new tokens
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings to {len(tokenizer)} tokens")

        return model, tokenizer

    def create_dataset(self, json_path: str, tokenizer, mode: str = 'train') -> UMBRELLADataset:
        """
        Create UMBRELLA dataset from JSON file OR directory.

        Args:
            json_path: Path to JSON file OR directory with JSON files
            tokenizer: Text tokenizer
            mode: 'train' or 'eval'

        Returns:
            UMBRELLADataset instance
        """
        path_obj = Path(json_path)

        if path_obj.is_file():
            logger.info(f"Creating {mode} dataset from file: {json_path}")
        elif path_obj.is_dir():
            logger.info(f"Creating {mode} dataset from directory: {json_path}")
        else:
            raise ValueError(f"Path is neither file nor directory: {json_path}")

        dataset = UMBRELLADataset(
            json_path=json_path,
            tokenizer=tokenizer,
            mode=mode,
            img_size=self.config.img_size,
            max_seq_length=self.config.max_seq_length,
            max_images=self.config.max_images_per_sample,
            augment=(mode == 'train'),
            modality=self.config.modality,
            task_filter=self.config.task_filter
        )

        logger.info(f"  Loaded {len(dataset)} samples")
        logger.info(f"  Image size: {dataset.img_size}")
        logger.info(f"  4D images: {dataset.is_4d}")

        return dataset

    def create_collator(self, tokenizer) -> UMBRELLACollator:
        """
        Create data collator.

        Args:
            tokenizer: Text tokenizer

        Returns:
            Collator instance
        """
        if self.config.enable_memory_aware_batching:
            collator = MemoryAwareUMBRELLACollator(
                tokenizer=tokenizer,
                img_size=self.config.img_size[0] if isinstance(self.config.img_size, list) else self.config.img_size,
                max_seq_length=self.config.max_seq_length,
                max_images=self.config.max_images_per_sample,
                memory_budget_gb=self.config.memory_budget_gb
            )
            logger.info("Using memory-aware collator")
        else:
            collator = UMBRELLACollator(
                tokenizer=tokenizer,
                img_size=self.config.img_size[0] if isinstance(self.config.img_size, list) else self.config.img_size,
                max_seq_length=self.config.max_seq_length,
                max_images=self.config.max_images_per_sample
            )
            logger.info("Using standard collator")

        return collator

    def train(self):
        """Run complete training pipeline."""
        try:
            # Setup W&B if enabled
            if self.config.use_wandb and self.config.wandb_api_key:
                import wandb
                wandb.login(key=self.config.wandb_api_key)
                wandb.init(project=self.config.wandb_project, config=vars(self.config))
                logger.info("Weights & Biases initialized")

            # Setup model and tokenizer
            model, tokenizer = self.setup_model()

            # Create datasets
            train_dataset = self.create_dataset(
                self.config.train_json_path,
                tokenizer,
                mode='train'
            )

            eval_dataset = None
            if self.config.eval_json_path:
                eval_dataset = self.create_dataset(
                    self.config.eval_json_path,
                    tokenizer,
                    mode='eval'
                )

            # Create collator
            collator = self.create_collator(tokenizer)

            # Create training arguments using factory method
            logger.info("\n" + "=" * 80)
            logger.info("CREATING TRAINING ARGUMENTS (UNIFIED CONFIG)")
            logger.info("=" * 80)
            training_args = self.config.to_training_args(eval_dataset_available=(eval_dataset is not None))
            logger.info("  Training arguments created successfully")
            logger.info(f"  Type: {type(training_args).__name__}")
            logger.info(f"  Has task_type_weights: {hasattr(training_args, 'task_type_weights')}")
            logger.info(f"  Has enable_task_aware_loss: {hasattr(training_args, 'enable_task_aware_loss')}")
            logger.info(f"  Has mask_human_turns: {hasattr(training_args, 'mask_human_turns')}")
            logger.info("=" * 80 + "\n")

            # Create UMBRELLA trainer with custom compute_loss that removes image_mask
            logger.info("Creating UMBRELLATrainer with unified config...")
            trainer = UMBRELLATrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                tokenizer=tokenizer,
            )
            logger.info("  UMBRELLATrainer created successfully")
            logger.info("  Using custom UMBRELLATrainer.compute_loss() to remove image_mask before model forward")

            logger.info("\n" + "=" * 80)
            logger.info("STARTING TRAINING")
            logger.info("=" * 80)

            # Train
            trainer.train()

            # Save final model
            logger.info("\nSaving final model...")
            final_output_dir = Path(self.config.output_dir) / "final_model"
            trainer.save_model(str(final_output_dir))
            tokenizer.save_pretrained(str(final_output_dir))

            logger.info("\n" + "=" * 80)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"Final model saved to: {final_output_dir}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UMBRELLA Training with LLaVA-Next Format and Unified Config"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to umbrella_llava_train.yaml config file'
    )
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training JSON file OR directory containing JSON files'
    )
    parser.add_argument(
        '--eval-data',
        type=str,
        help='Path to evaluation JSON file OR directory (optional)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='T1',
        choices=['T1', 'rsfMRI'],
        help='Modality to train on (T1 or rsfMRI)'
    )
    parser.add_argument(
        '--task-filter',
        type=str,
        help='Filter for specific task type (e.g., same_sex_comparison)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate from config'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config from YAML
    logger.info(f"Loading config from {args.config}")
    config = UMBRELLATrainingConfig.from_yaml(args.config)

    # Override with command-line arguments
    config.train_json_path = args.train_data
    config.eval_json_path = args.eval_data
    config.modality = args.modality
    config.task_filter = args.task_filter

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.no_wandb:
        config.use_wandb = False

    # Validate paths
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    if args.eval_data and not Path(args.eval_data).exists():
        raise FileNotFoundError(f"Evaluation data not found: {args.eval_data}")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Log configuration summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Training data: {config.train_json_path}")
    logger.info(f"  Type: {'Directory' if Path(config.train_json_path).is_dir() else 'File'}")
    if config.eval_json_path:
        logger.info(f"Eval data: {config.eval_json_path}")
        logger.info(f"  Type: {'Directory' if Path(config.eval_json_path).is_dir() else 'File'}")
    logger.info(f"Modality: {config.modality}")
    if config.task_filter:
        logger.info(f"Task filter: {config.task_filter}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Task-aware loss: {config.enable_task_aware_loss}")
    logger.info(f"Mask human turns: {config.mask_human_turns}")
    logger.info(f"Memory-aware batching: {config.enable_memory_aware_batching}")
    logger.info("=" * 80 + "\n")

    # Create and run pipeline
    pipeline = UMBRELLATrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()
