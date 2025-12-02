"""
DataLoaders - LLaVA-Compatible Brain Imaging Data Loading
=========================================================

Complete dataloader package for brain imaging multi-turn conversations
with LLaVA-style vision-language models.

Modules:
- image_loader: Multi-modal brain image loading (sMRI, dMRI, fMRI)
- conversation_processor: JSON to LLaVA format conversion
- t1_json_dataset: PyTorch Dataset for JSON conversations
- umbrella_dataloader: Main integration module with batching

Author: BrainVLM Team
Date: 2025-11-27
Version: 1.0 (Primary)
"""

from .image_loader import ImageLoader, load_image, load_images_from_json
from .conversation_processor import (
    ConversationProcessor,
    format_conversation,
    process_json
)
from .t1_json_dataset import T1JSONDataset
from .umbrella_dataloader import (
    UMBRELLADataLoader,
    create_umbrella_dataloaders
)

__all__ = [
    # Image loading
    "ImageLoader",
    "load_image",
    "load_images_from_json",

    # Conversation processing
    "ConversationProcessor",
    "format_conversation",
    "process_json",

    # Datasets
    "T1JSONDataset",
    "UMBRELLADataLoader",

    # Utilities
    "create_umbrella_dataloaders"
]

__version__ = "1.0.0"
__author__ = "BrainVLM Team"
