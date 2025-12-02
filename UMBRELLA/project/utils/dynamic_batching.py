"""
Dynamic batching utilities for heterogeneous multi-task training.

Provides sampler, collator, and batch construction logic for online batch size control.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from torch.utils.data import Sampler, Dataset
import random

from memory_utils import MemoryPredictor

logger = logging.getLogger(__name__)


@dataclass
class HeterogeneousBatch:
    """Batch containing samples with variable number of images."""

    task_types: List[str] = field(default_factory=list)
    sample_indices: List[int] = field(default_factory=list)

    # Tensor data (padded to max in batch)
    pixel_values: torch.Tensor = None  # Shape: (batch_size, max_images, C, H, W, D)
    input_ids: torch.Tensor = None     # Shape: (batch_size, max_seq_len)
    attention_mask: torch.Tensor = None  # Shape: (batch_size, max_seq_len)
    labels: torch.Tensor = None        # Shape: (batch_size, max_seq_len)

    # Metadata
    image_mask: torch.Tensor = None    # Shape: (batch_size, max_images) - 1 if valid image, 0 if padded
    num_images_per_sample: List[int] = field(default_factory=list)  # Actual image count per sample

    # Additional metadata
    task_ids: torch.Tensor = None      # Task type as integer
    metadata: List[Dict[str, Any]] = field(default_factory=list)


class MemoryAwareBatchSampler(Sampler):
    """
    Sampler that constructs batches with online memory prediction.

    Uses MemoryPredictor to estimate memory requirements and constructs batches
    that fit within memory constraints while maintaining task diversity.
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 16,
                 max_memory_mb: float = 30000,  # 30GB
                 device: str = "cuda:0",
                 shuffle: bool = True,
                 task_diversity_weight: float = 0.5,
                 verbose: bool = False):
        """
        Initialize memory-aware batch sampler.

        Args:
            dataset: Dataset instance with get_sample_metadata() method
            batch_size: Target batch size (may be smaller based on memory)
            max_memory_mb: Maximum GPU memory budget in MB
            device: GPU device
            shuffle: Whether to shuffle samples
            task_diversity_weight: Weight for task diversity in batch construction (0-1)
            verbose: Log batch construction details
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_mb / 1024.0
        self.device = device
        self.shuffle = shuffle
        self.task_diversity_weight = task_diversity_weight
        self.verbose = verbose

        # Memory prediction
        self.memory_predictor = MemoryPredictor(device=device, verbose=verbose)

        # Precompute sample metadata for efficient batch construction
        self.sample_metadata = []
        self._precompute_metadata()

    def _precompute_metadata(self):
        """Precompute metadata for all samples (task type, num images, etc.)."""
        logger.info(f"Precomputing metadata for {len(self.dataset)} samples...")
        for idx in range(len(self.dataset)):
            metadata = self.dataset.get_sample_metadata(idx)
            self.sample_metadata.append(metadata)
        logger.info("Metadata precomputation complete")

    def _estimate_batch_memory(self, sample_indices: List[int]) -> float:
        """
        Estimate memory for a batch of samples.

        Args:
            sample_indices: List of sample indices in batch

        Returns:
            Estimated memory in GB
        """
        task_types = [self.sample_metadata[idx]['task_type'] for idx in sample_indices]
        return self.memory_predictor.predict_batch_memory(task_types, is_training=True)

    def _construct_batch(self, available_indices: List[int]) -> List[int]:
        """
        Construct a single batch using greedy bin-packing with task diversity.

        Args:
            available_indices: Pool of sample indices to choose from

        Returns:
            List of sample indices for this batch
        """
        if not available_indices:
            return []

        batch = []
        task_counts = {'T1': 0, 'T2': 0, 'T3': 0}
        used_indices = set()

        # Priority function: (memory_fit_score, task_diversity_score)
        def compute_priority(idx):
            """Compute priority for adding sample to batch."""
            metadata = self.sample_metadata[idx]
            task_type = metadata['task_type']

            # Memory fit: prefer samples that fit comfortably
            current_memory = self._estimate_batch_memory(batch)
            if current_memory >= self.max_memory_gb:
                return -float('inf')  # Doesn't fit

            new_memory = self._estimate_batch_memory(batch + [idx])
            if new_memory > self.max_memory_gb:
                memory_fit = -1.0
            else:
                # How much room is left after adding this sample
                memory_fit = (self.max_memory_gb - new_memory) / self.max_memory_gb

            # Task diversity: prefer underrepresented task types
            max_count = max(task_counts.values())
            current_count = task_counts[task_type]
            if max_count > 0:
                diversity = 1.0 - (current_count / max_count)
            else:
                diversity = 1.0

            # Combined priority
            priority = (1 - self.task_diversity_weight) * memory_fit + \
                      self.task_diversity_weight * diversity

            return priority

        # Greedily add samples to batch
        while len(batch) < self.batch_size and available_indices:
            # Find best sample to add
            valid_indices = [idx for idx in available_indices if idx not in used_indices]
            if not valid_indices:
                break

            priorities = [compute_priority(idx) for idx in valid_indices]
            best_idx_pos = np.argmax(priorities)
            best_priority = priorities[best_idx_pos]

            if best_priority <= -float('inf'):  # No more samples fit
                break

            # Add sample to batch
            sample_idx = valid_indices[best_idx_pos]
            batch.append(sample_idx)
            used_indices.add(sample_idx)

            # Update task counts
            task_type = self.sample_metadata[sample_idx]['task_type']
            task_counts[task_type] += 1

        if self.verbose and len(batch) > 0:
            memory = self._estimate_batch_memory(batch)
            logger.info(f"Batch {len(batch)}: memory={memory:.2f}GB, "
                       f"tasks={task_counts}, diversity={self.task_diversity_weight:.1f}")

        return batch

    def __iter__(self):
        """Generate batches."""
        # Create index list
        indices = list(range(len(self.dataset)))

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)

        # Construct batches
        batches = []
        remaining = indices.copy()

        while remaining:
            batch = self._construct_batch(remaining)
            if not batch:
                # Can't construct more batches, add remaining as single sample
                batch = remaining[:1]
                remaining = remaining[1:]
            else:
                remaining = [idx for idx in remaining if idx not in batch]

            batches.append(batch)

        # Flatten batches into single index list
        for batch in batches:
            for idx in batch:
                yield idx

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.dataset)


class HeterogeneousCollator:
    """
    Collate function for batches with variable number of images per sample.

    Pads batches to consistent shapes while tracking which positions are valid.
    """

    def __init__(self,
                 tokenizer=None,
                 img_size: int = 128,
                 max_seq_length: int = 2048):
        """
        Initialize collator.

        Args:
            tokenizer: Text tokenizer
            img_size: Image spatial dimensions (assuming square)
            max_seq_length: Maximum sequence length for text
        """
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict[str, Any]]) -> HeterogeneousBatch:
        """
        Collate batch of heterogeneous samples.

        Args:
            batch: List of sample dicts from dataset

        Returns:
            HeterogeneousBatch with padded tensors
        """
        if not batch:
            return HeterogeneousBatch()

        # Extract metadata
        task_types = [item.get('task_type', 'T1') for item in batch]
        sample_indices = [item.get('sample_index', -1) for item in batch]
        num_images_per_sample = [len(item.get('pixel_values', {}).get('T1', []))
                                if isinstance(item.get('pixel_values', {}).get('T1'), list)
                                else 1 for item in batch]

        max_images = max(num_images_per_sample) if num_images_per_sample else 1

        # Prepare pixel values (pad to max images in batch)
        batch_size = len(batch)
        pixel_values_padded = torch.zeros(
            batch_size, max_images, 1, self.img_size, self.img_size, 3,
            dtype=torch.float32
        )
        image_mask = torch.zeros(batch_size, max_images, dtype=torch.bool)

        for i, item in enumerate(batch):
            images = item.get('pixel_values', {}).get('T1', [])
            if not isinstance(images, list):
                images = [images]

            for j, img in enumerate(images):
                if img is not None:
                    # Ensure image has correct shape
                    if isinstance(img, torch.Tensor):
                        pixel_values_padded[i, j] = img[:1]
                    else:
                        pixel_values_padded[i, j] = torch.from_numpy(img[:1])

                    image_mask[i, j] = True

        # Prepare text data
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            ids = item.get('input_ids', torch.zeros(self.max_seq_length, dtype=torch.long))
            mask = item.get('attention_mask', torch.zeros(self.max_seq_length, dtype=torch.long))
            label = item.get('labels', torch.full((self.max_seq_length,), -100, dtype=torch.long))

            # Ensure correct length
            if len(ids) < self.max_seq_length:
                ids = torch.cat([ids, torch.zeros(self.max_seq_length - len(ids), dtype=torch.long)])
            else:
                ids = ids[:self.max_seq_length]

            if len(mask) < self.max_seq_length:
                mask = torch.cat([mask, torch.zeros(self.max_seq_length - len(mask), dtype=torch.long)])
            else:
                mask = mask[:self.max_seq_length]

            if len(label) < self.max_seq_length:
                label = torch.cat([label, torch.full((self.max_seq_length - len(label),), -100, dtype=torch.long)])
            else:
                label = label[:self.max_seq_length]

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(label)

        # Stack text data
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)

        # Create task ID tensor (map task type to integer)
        task_type_map = {'T1': 0, 'T2': 1, 'T3': 2}
        task_ids = torch.tensor([task_type_map.get(t, 0) for t in task_types], dtype=torch.long)

        # Prepare metadata
        metadata = [item.get('metadata', {}) for item in batch]

        return HeterogeneousBatch(
            task_types=task_types,
            sample_indices=sample_indices,
            pixel_values=pixel_values_padded,
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
            image_mask=image_mask,
            num_images_per_sample=num_images_per_sample,
            task_ids=task_ids,
            metadata=metadata
        )


class BatchConstructor:
    """Utility class for constructing batches with constraints."""

    @staticmethod
    def construct_balanced_batch(dataset: Dataset,
                                 batch_size: int,
                                 task_distribution: Dict[str, float]) -> List[int]:
        """
        Construct batch with target task distribution.

        Args:
            dataset: Dataset instance
            batch_size: Target batch size
            task_distribution: Dict mapping task type to target fraction

        Returns:
            List of sample indices
        """
        # Group samples by task type
        task_indices = {'T1': [], 'T2': [], 'T3': []}
        for idx in range(len(dataset)):
            metadata = dataset.get_sample_metadata(idx)
            task_type = metadata.get('task_type', 'T1')
            task_indices[task_type].append(idx)

        # Sample from each task type according to distribution
        batch = []
        for task_type, target_fraction in task_distribution.items():
            target_count = int(batch_size * target_fraction)
            available = task_indices[task_type]
            if available:
                sampled = np.random.choice(available, size=min(target_count, len(available)),
                                          replace=False)
                batch.extend(sampled.tolist())

        # Pad to batch size if needed
        all_indices = [idx for indices in task_indices.values() for idx in indices]
        while len(batch) < batch_size and all_indices:
            remaining = [idx for idx in all_indices if idx not in batch]
            if remaining:
                batch.append(np.random.choice(remaining))

        return batch[:batch_size]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # This would be used with a dataset that implements get_sample_metadata()
    # sampler = MemoryAwareBatchSampler(dataset, batch_size=16)
    # for batch_indices in sampler:
    #     print(f"Batch: {batch_indices}")

    # Collator example
    collator = HeterogeneousCollator()
    print("HeterogeneousCollator ready for use")
