"""
UMBRELLA Data Collator

Handles variable images per sample with proper padding and masking.

CRITICAL FIX (2025-12-03):
- UMBRELLABatch.keys(), values(), items(), __iter__() now filter out popped fields
- This prevents TypeError when model(**inputs) tries to pass image_mask=None
- Fields set to None via pop() are now excluded from dict unpacking
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)


@dataclass
class UMBRELLABatch:
    """
    Batch container for UMBRELLA training.

    Supports heterogeneous batches with variable images per sample.

    CRITICAL: Implements __len__() and __iter__() for HuggingFace Trainer compatibility.
    The Trainer's _prepare_inputs() method expects inputs to support len() check.

    CRITICAL FIX (2025-12-03):
    - keys(), values(), items(), __iter__() now filter out None fields
    - This ensures popped fields (e.g., image_mask) are not passed to model
    - Prevents TypeError: unexpected keyword argument 'image_mask'
    """

    # Core tensors
    pixel_values: torch.Tensor  # (batch, max_images, C, H, W, D) or (batch, max_images, C, H, W, D, T) for 4D
    input_ids: torch.Tensor     # (batch, max_seq_len)
    attention_mask: torch.Tensor  # (batch, max_seq_len)
    labels: torch.Tensor        # (batch, max_seq_len) - -100 for masked tokens, >=0 for active

    # UMBRELLA-specific metadata
    image_mask: torch.Tensor    # (batch, max_images) - 1 for valid images, 0 for padding
    num_images_per_sample: List[int]  # [num_imgs_sample_0, num_imgs_sample_1, ...]
    task_types: List[str]       # ['T1', 'T2', 'T3', ...]
    task_ids: torch.Tensor      # (batch,) - integer task IDs
    sample_indices: List[int]   # Original dataset indices for tracking
    metadata: List[Dict[str, Any]]  # Per-sample metadata (conversation history, etc.)

    def __len__(self):
        """
        Return batch size.

        CRITICAL: Required for HuggingFace Trainer's _prepare_inputs() method
        which checks `if len(inputs) == 0` before processing.
        """
        return self.input_ids.shape[0]

    def __iter__(self):
        """
        Make batch iterable for dict-like access patterns.

        Some HuggingFace utilities may iterate over inputs expecting dict behavior.
        This provides compatibility by yielding field names.

        CRITICAL FIX (2025-12-03): Filter out None fields (popped fields)
        to prevent passing them to model(**inputs).

        Yields:
            Field names that can be used for attribute access (excluding None fields)
        """
        # Define all possible field names
        all_fields = [
            'pixel_values',
            'input_ids',
            'attention_mask',
            'labels',
            'image_mask',
            'num_images_per_sample',
            'task_types',
            'task_ids',
            'sample_indices',
            'metadata'
        ]

        # Only yield fields that are NOT None (filter out popped fields)
        for field in all_fields:
            if hasattr(self, field) and getattr(self, field) is not None:
                yield field

    def __getitem__(self, key: str):
        """
        Enable dict-like access for HuggingFace compatibility.

        Args:
            key: Field name

        Returns:
            Field value

        Raises:
            KeyError: If field doesn't exist
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"UMBRELLABatch has no field '{key}'")

    def get(self, key: str, default=None):
        """
        Dict-like get method with default value support.

        CRITICAL FIX: Enables compute_loss() to use .get() method safely.

        Args:
            key: Field name
            default: Default value if field doesn't exist

        Returns:
            Field value or default
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def pop(self, key: str, default=None):
        """
        Dict-like pop method - removes and returns field value.

        CRITICAL FIX: Enables compute_loss() to use .pop() method.

        NOTE: This implementation does NOT actually remove the field from the
        dataclass (as dataclass fields are immutable), but instead sets the
        field to None after retrieval. This is sufficient for HuggingFace
        Trainer's compute_loss() pattern which extracts metadata once.

        IMPORTANT: keys(), values(), items(), __iter__() now filter out None fields,
        so popped fields will not be passed to model(**inputs).

        Args:
            key: Field name
            default: Default value if field doesn't exist

        Returns:
            Field value (before setting to None) or default
        """
        try:
            value = self.__getitem__(key)
            # Set field to None to simulate "removal"
            # This prevents accidental reuse and signals the field was "popped"
            if hasattr(self, key):
                object.__setattr__(self, key, None)
            return value
        except KeyError:
            return default

    def keys(self):
        """
        Return field names like a dict.

        CRITICAL FIX (2025-12-03): Filter out None fields (popped fields)
        to prevent passing them to model(**inputs).

        Returns:
            List of field names (excluding fields set to None)
        """
        all_fields = [
            'pixel_values',
            'input_ids',
            'attention_mask',
            'labels',
            'image_mask',
            'num_images_per_sample',
            'task_types',
            'task_ids',
            'sample_indices',
            'metadata'
        ]

        # Only return fields that are NOT None (filter out popped fields)
        return [field for field in all_fields
                if hasattr(self, field) and getattr(self, field) is not None]

    def values(self):
        """
        Return field values like a dict.

        CRITICAL FIX (2025-12-03): Only return values for non-None fields.

        Returns:
            List of field values (excluding None values from popped fields)
        """
        return [getattr(self, key) for key in self.keys()]

    def items(self):
        """
        Return (key, value) pairs like a dict.

        CRITICAL FIX (2025-12-03): Only return items for non-None fields.

        Returns:
            List of (key, value) tuples (excluding popped fields)
        """
        return [(key, getattr(self, key)) for key in self.keys()]


class UMBRELLACollator:
    """
    Collate function for UMBRELLA batches with variable images per sample.

    Handles:
    - Padding images to max in batch
    - Creating image masks for valid positions
    - Tokenizing and padding text
    - Task type encoding
    """

    # Task type to integer mapping
    TASK_TYPE_MAP = {'T1': 0, 'T2': 1, 'T3': 2}

    def __init__(self,
                 tokenizer=None,
                 max_images: int = 4,
                 pad_to_max_images: bool = False,
                 image_token_id: Optional[int] = None):
        """
        Initialize collator.

        Args:
            tokenizer: HuggingFace tokenizer for text processing
            max_images: Maximum number of images per sample
            pad_to_max_images: Whether to pad to max_images or max in batch
            image_token_id: Token ID for image placeholder (default: <image>)
        """
        self.tokenizer = tokenizer
        self.max_images = max_images
        self.pad_to_max_images = pad_to_max_images

        # Get image token ID
        if image_token_id is not None:
            self.image_token_id = image_token_id
        elif tokenizer is not None and hasattr(tokenizer, 'image_token_id'):
            self.image_token_id = tokenizer.image_token_id
        elif tokenizer is not None:
            # Fallback: try to get from vocab
            self.image_token_id = tokenizer.convert_tokens_to_ids('<image>')
        else:
            # Default fallback
            self.image_token_id = 32000  # LLaVA default

    def __call__(self, batch: List[Dict[str, Any]]) -> UMBRELLABatch:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dicts from UMBRELLADataset

        Returns:
            UMBRELLABatch with padded tensors
        """
        if not batch:
            return self._empty_batch()

        batch_size = len(batch)

        # Extract metadata
        task_types = [item.get('task_type', 'T1') for item in batch]
        sample_indices = [item.get('sample_index', -1) for item in batch]
        num_images_per_sample = [item.get('num_images', 1) for item in batch]

        # Determine image padding size
        if self.pad_to_max_images:
            max_imgs_in_batch = self.max_images
        else:
            max_imgs_in_batch = max(num_images_per_sample)

        # Prepare pixel values with proper padding
        pixel_values_list = []
        image_mask_list = []

        for i, item in enumerate(batch):
            imgs = item.get('pixel_values', torch.empty(0))  # (num_imgs, C, H, W, D)
            num_imgs = imgs.shape[0] if imgs.numel() > 0 else 0

            # Create image mask (1 for valid, 0 for padding)
            mask = torch.zeros(max_imgs_in_batch, dtype=torch.long)
            mask[:num_imgs] = 1
            image_mask_list.append(mask)

            # Pad images if needed
            if num_imgs < max_imgs_in_batch:
                # Get image shape (handle both 3D and 4D volumes)
                if imgs.numel() > 0:
                    img_shape = imgs.shape[1:]  # (C, H, W, D) or (C, H, W, D, T)
                else:
                    # Default to 3D fMRI shape if no images
                    img_shape = (1, 224, 224, 8)

                # Create padding zeros
                padding_shape = (max_imgs_in_batch - num_imgs,) + img_shape
                padding = torch.zeros(padding_shape)

                # Concatenate real images with padding
                if imgs.numel() > 0:
                    imgs = torch.cat([imgs, padding], dim=0)
                else:
                    imgs = padding

            pixel_values_list.append(imgs)

        # Stack pixel values (batch, max_imgs, C, H, W, D) or (batch, max_imgs, C, H, W, D, T)
        pixel_values = torch.stack(pixel_values_list, dim=0)
        image_mask = torch.stack(image_mask_list, dim=0)

        # Tokenize and pad text
        if self.tokenizer is not None:
            input_ids_list = []
            attention_mask_list = []
            labels_list = []

            for item in batch:
                input_ids_list.append(item.get('input_ids', torch.tensor([self.tokenizer.pad_token_id])))
                attention_mask_list.append(item.get('attention_mask', torch.tensor([0])))
                labels_list.append(item.get('labels', torch.tensor([-100])))

            # Pad sequences
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask_list, batch_first=True, padding_value=0
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=-100
            )
        else:
            # No tokenizer - use raw values
            input_ids = torch.stack([item.get('input_ids', torch.tensor([0])) for item in batch])
            attention_mask = torch.stack([item.get('attention_mask', torch.tensor([0])) for item in batch])
            labels = torch.stack([item.get('labels', torch.tensor([-100])) for item in batch])

        # Encode task types to integers
        task_ids = torch.tensor([self.TASK_TYPE_MAP.get(t, 0) for t in task_types], dtype=torch.long)

        # Extract metadata
        metadata = [item.get('metadata', {}) for item in batch]

        return UMBRELLABatch(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_mask=image_mask,
            num_images_per_sample=num_images_per_sample,
            task_types=task_types,
            task_ids=task_ids,
            sample_indices=sample_indices,
            metadata=metadata
        )

    def _empty_batch(self) -> UMBRELLABatch:
        """Return an empty batch for edge cases."""
        return UMBRELLABatch(
            pixel_values=torch.empty(0),
            input_ids=torch.empty(0, dtype=torch.long),
            attention_mask=torch.empty(0, dtype=torch.long),
            labels=torch.empty(0, dtype=torch.long),
            image_mask=torch.empty(0, dtype=torch.long),
            num_images_per_sample=[],
            task_types=[],
            task_ids=torch.empty(0, dtype=torch.long),
            sample_indices=[],
            metadata=[]
        )


class MemoryAwareCollator(UMBRELLACollator):
    """
    Memory-aware collator with batch size validation.

    Warns if batch exceeds memory budget but doesn't enforce limits
    (enforcement happens at sampler level).
    """

    def __init__(self,
                 tokenizer=None,
                 max_images: int = 4,
                 pad_to_max_images: bool = False,
                 image_token_id: Optional[int] = None,
                 memory_budget_gb: float = 8.0):
        """
        Initialize memory-aware collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_images: Maximum number of images per sample
            pad_to_max_images: Whether to pad to max_images
            image_token_id: Token ID for image placeholder
            memory_budget_gb: Memory budget in GB (for logging only)
        """
        super().__init__(tokenizer, max_images, pad_to_max_images, image_token_id)
        self.memory_budget_gb = memory_budget_gb

    def __call__(self, batch: List[Dict[str, Any]]) -> UMBRELLABatch:
        """
        Collate with memory validation.

        Logs warning if batch exceeds memory budget.
        """
        estimated_memory = self.estimate_batch_memory(batch)

        if estimated_memory > self.memory_budget_gb:
            logger.warning(
                f"Batch memory ({estimated_memory:.2f} GB) exceeds budget "
                f"({self.memory_budget_gb:.2f} GB). Consider adjusting batch size."
            )

        return super().__call__(batch)

    def estimate_batch_memory(self, batch: List[Dict[str, Any]]) -> float:
        """
        Estimate batch memory usage in GB.

        Args:
            batch: List of sample dicts

        Returns:
            Estimated memory in GB
        """
        if not batch:
            return 0.0

        batch_size = len(batch)
        num_images_per_sample = [item.get('num_images', 1) for item in batch]
        total_images = sum(num_images_per_sample)

        # Assume fMRI volumes: (1, 224, 224, 8) * 4 bytes (float32)
        bytes_per_image = 1 * 224 * 224 * 8 * 4

        # Estimate text memory (rough approximation)
        avg_text_len = 512  # tokens
        bytes_per_token = 4  # int32

        total_bytes = (
            total_images * bytes_per_image +  # Images
            batch_size * avg_text_len * bytes_per_token  # Text
        )

        return total_bytes / (1024 ** 3)  # Convert to GB
