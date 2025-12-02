"""
UMBRELLA Integrated Training Script (FIXED VERSION)

Complete training pipeline with:
- Proper config loading from YAML
- LLaVA-Next tokenization format
- Variable-size image support (3D/4D)
- Multi-turn conversation training
- Correct import paths
- **NEW**: Directory-based data loading support

Key Updates:
1. Load from umbrella_llava_train.yaml
2. Support list-based image sizes from config
3. Proper dataset initialization with config parameters
4. Correct import paths for dataset and collator
5. LLaVA-Next model integration
6. **Directory and file-based data loading**
7. Task filtering support
"""

import torch
import logging
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import argparse
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

# Correct import paths
sys.path.append(str(Path(__file__).parent.parent))
from dataset.umbrella_dataset_fixed import UMBRELLADataset, create_umbrella_dataset_from_config
from dataset.umbrella_collator import UMBRELLACollator, MemoryAwareUMBRELLACollator

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
    Configuration for UMBRELLA training.

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

    # Masking
    mask_user_turns: bool = True

    # Logging and saving
    output_dir: str = "./hf_results/umbrella"
    logging_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    warmup_steps: int = 500

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

        # Get image size from config
        img_size = modality_dataset_config.get('img_size', [96, 96, 96])

        # Create config instance
        return cls(
            model_name=model_config.get('hf_name', 'llava-hf/llava-interleave-qwen-0.5b-hf'),
            modality=modality,
            img_size=img_size,
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


class UMBRELLATrainingPipeline:
    """Complete UMBRELLA training pipeline with config support."""

    def __init__(self, config: UMBRELLATrainingConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("=" * 80)
        logger.info("UMBRELLA Training Pipeline (FIXED VERSION - Directory Support)")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Modality: {config.modality}")
        logger.info(f"Image size: {config.img_size}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        if config.task_filter:
            logger.info(f"Task filter: {config.task_filter}")

    def setup_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load and setup LLaVA model and tokenizer."""
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

        logger.info(f"Loading model: {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.mixed_precision == "fp16" else torch.float32,
            device_map="auto",
            trust_remote_code=True  # Required for LLaVA models
        )

        # Resize embeddings to accommodate new tokens
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings to {len(tokenizer)} tokens")

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")

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

            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                eval_steps=self.config.eval_steps if eval_dataset else None,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                fp16=self.config.mixed_precision == "fp16",
                bf16=self.config.mixed_precision == "bf16",
                save_strategy="steps",
                evaluation_strategy="steps" if eval_dataset else "no",
                logging_strategy="steps",
                report_to="wandb" if self.config.use_wandb else "none",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="loss" if eval_dataset else None,
                greater_is_better=False if eval_dataset else None,
            )

            # Create trainer
            logger.info("Creating Trainer...")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                tokenizer=tokenizer,
            )

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
        description="UMBRELLA Training with LLaVA-Next Format and Directory Support"
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
        help='Filter samples by task type (e.g., "same_sex_comparison", "different_sex_comparison")'
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
    logger.info("=" * 80 + "\n")

    # Create and run pipeline
    pipeline = UMBRELLATrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()
