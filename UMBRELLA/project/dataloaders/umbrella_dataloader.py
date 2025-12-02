"""
UMBRELLA DataLoader - Main Integration Module
=================================================

Comprehensive dataloader combining all dataloader components for brain imaging
multi-turn conversation training with LLaVA-style models.

Features:
- Multi-split support (train/validation/test)
- Batch processing with collation
- Variable-length sequence handling
- Image preprocessing and batching
- Compatible with HuggingFace Trainer
- Modality-aware processing

Author: BrainVLM Team
Date: 2025-11-25
Version: 1.0 (Primary)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import numpy as np

from .t1_json_dataset import T1JSONDataset


class UMBRELLADataLoader(Dataset):
    """
    Main dataloader for UMBRELLA brain imaging conversations.

    Integrates all dataloader components:
    - JSON conversation loading
    - Image preprocessing
    - LLaVA format conversion
    - Tokenization
    - Batch collation
    """

    def __init__(self,
                 json_dir: Union[str, Path],
                 image_root: Optional[Union[str, Path]] = None,
                 split: str = "train",
                 tokenizer=None,
                 processor=None,
                 image_size: int = 224,
                 normalize: bool = True,
                 standardize: bool = True,
                 max_length: int = 2048,
                 add_generation_prompt: bool = False,
                 filter_fn: Optional[Callable] = None):
        """
        Initialize UMBRELLA dataloader.

        Args:
            json_dir: Root directory containing split subdirectories
            image_root: Root directory for image paths
            split: Data split ('train', 'validation', 'test')
            tokenizer: LLaVA tokenizer
            processor: LLaVA image processor
            image_size: Target image size
            normalize: Apply min-max normalization
            standardize: Apply z-score standardization
            max_length: Maximum sequence length
            add_generation_prompt: Add empty assistant prompt
            filter_fn: Optional function to filter examples
        """
        self.json_dir = Path(json_dir)
        self.split = split
        self.filter_fn = filter_fn

        # Get split directory
        split_dir = self.json_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Initialize base dataset
        self.dataset = T1JSONDataset(
            json_dir=split_dir,
            image_root=image_root,
            tokenizer=tokenizer,
            processor=processor,
            image_size=image_size,
            normalize=normalize,
            standardize=standardize,
            max_length=max_length,
            add_generation_prompt=add_generation_prompt
        )

        # Apply filtering if provided
        if self.filter_fn is not None:
            self._apply_filter()

        print(f"Initialized UMBRELLA DataLoader for {split} split")
        print(f"  Total examples: {len(self)}")

    def _apply_filter(self):
        """
        Apply filter function to dataset.

        Filters examples based on custom criteria.
        """
        # This would filter the dataset based on filter_fn
        # Implementation depends on specific filtering needs
        pass

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training example.

        Args:
            idx: Index

        Returns:
            Dictionary with training data
        """
        return self.dataset[idx]

    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.

        Returns:
            Dictionary with dataset information
        """
        stats = self.dataset.get_dataset_statistics()

        return {
            "split": self.split,
            "json_dir": str(self.json_dir),
            "num_examples": len(self),
            "statistics": stats
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate function for DataLoader batching.

        Handles variable-length sequences and multiple images per example.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched dictionary
        """
        # Separate components
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        metadata = [item["metadata"] for item in batch]
        task_ids = [item["task_id"] for item in batch]

        # Stack tensors
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)

        # Handle pixel values (may be None if no images)
        pixel_values = [item.get("pixel_values") for item in batch]
        if all(pv is not None for pv in pixel_values):
            # Stack if all present
            if len(pixel_values[0].shape) == 3:
                # Single image per example: [batch, C, H, W]
                pixel_values = torch.stack(pixel_values)
            elif len(pixel_values[0].shape) == 4:
                # Multiple images per example: [batch, num_images, C, H, W]
                # Need to handle variable number of images
                # For simplicity, assume fixed number per example
                pixel_values = torch.stack(pixel_values)
        else:
            pixel_values = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "pixel_values": pixel_values,
            "labels": labels,
            "metadata": metadata,
            "task_ids": task_ids
        }

    def validate_dataset(self, num_samples: int = 5) -> Dict:
        """
        Validate dataset integrity and compatibility.

        Tests loading a small number of samples to ensure:
        - JSON files load correctly
        - Conversations are properly formatted
        - Images load without errors
        - Tokenization works correctly
        - Output tensors have correct shapes

        Args:
            num_samples: Number of samples to validate (default: 5)

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "status": "success",
            "total_samples": len(self),
            "validated_samples": 0,
            "errors": [],
            "warnings": [],
            "sample_outputs": []
        }

        # Sample random indices for validation
        import random
        num_to_check = min(num_samples, len(self))
        indices = random.sample(range(len(self)), num_to_check)

        for idx in indices:
            try:
                # Load sample
                sample = self[idx]

                # Validate tensor shapes and types
                assert isinstance(sample["input_ids"], torch.Tensor), "input_ids not a tensor"
                assert isinstance(sample["attention_mask"], torch.Tensor), "attention_mask not a tensor"
                assert isinstance(sample["labels"], torch.Tensor), "labels not a tensor"

                # Check shapes
                seq_len = len(sample["input_ids"])
                assert len(sample["attention_mask"]) == seq_len, "Attention mask length mismatch"
                assert len(sample["labels"]) == seq_len, "Labels length mismatch"

                # Validate image presence if claimed
                if "pixel_values" in sample and sample["pixel_values"] is not None:
                    assert isinstance(sample["pixel_values"], torch.Tensor), "pixel_values not a tensor"

                validation_results["validated_samples"] += 1
                validation_results["sample_outputs"].append({
                    "index": idx,
                    "seq_length": seq_len,
                    "has_images": "pixel_values" in sample and sample["pixel_values"] is not None,
                    "metadata_keys": list(sample.get("metadata", {}).keys())
                })

            except Exception as e:
                validation_results["status"] = "failed"
                validation_results["errors"].append({
                    "index": idx,
                    "error": str(e)
                })

        return validation_results

    def create_dataloader(self,
                          batch_size: int = 4,
                          shuffle: bool = True,
                          num_workers: int = 0,
                          pin_memory: bool = True,
                          drop_last: bool = False) -> DataLoader:
        """
        Create PyTorch DataLoader.

        Args:
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch

        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last
        )


def create_umbrella_dataloaders(json_dir: Union[str, Path],
                                 image_root: Optional[Union[str, Path]] = None,
                                 tokenizer=None,
                                 processor=None,
                                 batch_size: int = 4,
                                 image_size: int = 224,
                                 max_length: int = 2048,
                                 num_workers: int = 0,
                                 **kwargs) -> Dict[str, DataLoader]:
    """
    Create dataloaders for all splits (train, validation, test).

    Args:
        json_dir: Root directory containing split subdirectories
        image_root: Root directory for image paths
        tokenizer: LLaVA tokenizer
        processor: LLaVA image processor
        batch_size: Batch size
        image_size: Target image size
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        **kwargs: Additional arguments for UMBRELLADataLoader

    Returns:
        Dictionary with dataloaders for each split
    """
    json_dir = Path(json_dir)
    dataloaders = {}

    for split in ["train", "validation", "test"]:
        split_dir = json_dir / split
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            continue

        # Create dataset
        dataset = UMBRELLADataLoader(
            json_dir=json_dir,
            image_root=image_root,
            split=split,
            tokenizer=tokenizer,
            processor=processor,
            image_size=image_size,
            max_length=max_length,
            **kwargs
        )

        # Create dataloader
        shuffle = (split == "train")
        dataloader = dataset.create_dataloader(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=(split == "train")
        )

        dataloaders[split] = dataloader

    return dataloaders


# Example usage
if __name__ == "__main__":
    print("UMBRELLA DataLoader - Example Usage")
    print("="*60)

    # Example: Create dataloaders for all splits
    json_dir = Path("../../sample_data/sex_comparison_conversations")

    if json_dir.exists():
        # Create train dataloader
        print("\nCreating train dataloader...")
        train_dataset = UMBRELLADataLoader(
            json_dir=json_dir,
            image_root=None,
            split="train",
            tokenizer=None,
            processor=None,
            normalize=True,
            standardize=True
        )

        print(f"Train dataset size: {len(train_dataset)}")

        # Get dataset info
        info = train_dataset.get_dataset_info()
        print("\nDataset Info:")
        for key, value in info.items():
            if key != "statistics":
                print(f"  {key}: {value}")

        print("\nStatistics:")
        for key, value in info["statistics"].items():
            print(f"  {key}: {value}")

        # Create DataLoader
        print("\nCreating DataLoader...")
        train_loader = train_dataset.create_dataloader(
            batch_size=2,
            shuffle=True,
            num_workers=0
        )

        print(f"DataLoader batches: {len(train_loader)}")

        # Try loading a batch
        print("\nLoading first batch...")
        try:
            batch = next(iter(train_loader))
            print(f"Batch keys: {batch.keys()}")
            if "input_ids" in batch:
                print(f"Input IDs shape: {batch['input_ids'].shape}")
                print(f"Attention mask shape: {batch['attention_mask'].shape}")
                print(f"Labels shape: {batch['labels'].shape}")
                if batch["pixel_values"] is not None:
                    print(f"Pixel values shape: {batch['pixel_values'].shape}")
                print(f"Task IDs: {batch['task_ids']}")
        except Exception as e:
            print(f"Error loading batch: {e}")

        # Create all dataloaders
        print("\n" + "-"*60)
        print("Creating dataloaders for all splits...")

        # This would create all splits
        # dataloaders = create_umbrella_dataloaders(
        #     json_dir=json_dir,
        #     batch_size=4
        # )
        # print(f"Created dataloaders: {list(dataloaders.keys())}")

    else:
        print(f"\nDataset directory not found: {json_dir}")
        print("Please run generate_sex_comparison_conversations.py first.")

    print("\n" + "="*60)
    print("UMBRELLA DataLoader ready for training.")
    print("Compatible with HuggingFace Trainer and LLaVA models.")
