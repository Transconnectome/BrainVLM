"""
T1 JSON Dataset - LLaVA-Compatible Dataloader
=================================================

PyTorch Dataset for loading JSON conversations with brain imaging data.
Compatible with LLaVA processors and training pipeline.

Key Features:
- Loads LLaVA JSON format with generic <image> tokens
- Processes conversations to LLaVA format
- Loads and preprocesses brain images
- Tokenizes with LLaVA processor
- Returns training-ready tensors

Author: BrainVLM Team
Date: 2025-11-25
Version: 1.0 (Primary)
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from .image_loader import ImageLoader
from .conversation_processor import ConversationProcessor


class T1JSONDataset(Dataset):
    """
    PyTorch Dataset for T1-weighted MRI conversations in LLaVA JSON format.

    Compatible with LLaVA processors and training pipelines.
    """

    def __init__(self,
                 json_dir: Union[str, Path],
                 image_root: Optional[Union[str, Path]] = None,
                 tokenizer=None,
                 processor=None,
                 image_size: int = 224,
                 normalize: bool = True,
                 standardize: bool = True,
                 max_length: int = 2048,
                 add_generation_prompt: bool = False):
        """
        Initialize dataset.

        Args:
            json_dir: Directory containing JSON conversation files
            image_root: Root directory for image paths (None = use absolute paths)
            tokenizer: LLaVA tokenizer
            processor: LLaVA image processor
            image_size: Target image size for vision encoder
            normalize: Apply min-max normalization to images
            standardize: Apply z-score standardization to images
            max_length: Maximum sequence length for tokenization
            add_generation_prompt: Add empty assistant prompt for generation
        """
        self.json_dir = Path(json_dir)
        self.image_root = Path(image_root) if image_root else None
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length

        # Initialize components
        self.image_loader = ImageLoader(
            normalize=normalize,
            standardize=standardize,
            target_shape=None  # Will handle resizing in processor
        )
        self.conversation_processor = ConversationProcessor(
            add_generation_prompt=add_generation_prompt
        )

        # Load JSON files
        self.json_files = self._load_json_files()

        print(f"Loaded {len(self.json_files)} JSON conversation files from {json_dir}")

    def _load_json_files(self) -> List[Path]:
        """
        Load all JSON files from directory.

        Returns:
            List of JSON file paths
        """
        json_files = []

        # Find all .json files (not .jsonl)
        for json_file in self.json_dir.glob("*.json"):
            if json_file.name.endswith("_conversations.jsonl"):
                continue  # Skip JSONL files
            json_files.append(json_file)

        return sorted(json_files)

    def load_json_file(self, json_path: Path) -> Dict:
        """
        Load and parse a JSON conversation file.

        Supports both v1 format (conversation key) and v2 format (conversations key).

        Args:
            json_path: Path to JSON file

        Returns:
            Parsed JSON data (normalized to v2 format structure)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Validate that we have either old or new format
        has_old_format = "conversation" in data
        has_new_format = "conversations" in data

        if not (has_old_format or has_new_format):
            raise ValueError(
                f"JSON file {json_path} missing 'conversation' or 'conversations' key. "
                f"Available keys: {list(data.keys())}"
            )

        # Normalize old format to new format for consistency
        if has_old_format and not has_new_format:
            # Rename old key to new key for uniform handling
            data["conversations"] = data.pop("conversation")

        # Validate core required fields
        required_fields = ["task_id", "conversations", "images", "metadata"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"JSON file {json_path} missing required field: {field}")

        return data

    def process_conversation(self, json_data: Dict) -> str:
        """
        Convert JSON conversations to LLaVA format.

        Handles both v1 (from/value) and v2 (role/content) conversation formats.

        Args:
            json_data: Parsed JSON conversation data

        Returns:
            Formatted LLaVA prompt string
        """
        conversations = json_data["conversations"]
        prompt = self.conversation_processor.format_conversation_for_llava(conversations)
        return prompt

    def load_images(self, json_data: Dict) -> List[np.ndarray]:
        """
        Load all images referenced in JSON.

        Args:
            json_data: Parsed JSON conversation data

        Returns:
            List of image arrays
        """
        images = []

        for img_meta in json_data["images"]:
            path = img_meta["path"]
            modality = img_meta["modality"]

            # Resolve path
            if self.image_root and not Path(path).is_absolute():
                path = self.image_root / path

            try:
                # Load image
                image_data = self.image_loader.load_image(path, modality)

                # Convert to 2D slice if needed (for vision encoder)
                if image_data.ndim == 3:
                    # Extract middle axial slice
                    image_data = self.image_loader.extract_slice(
                        image_data, axis=2, slice_idx=None
                    )

                images.append(image_data)

            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
                # Return zero placeholder
                images.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))

        return images

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.json_files)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training example.

        Args:
            idx: Index

        Returns:
            Dictionary with:
                - input_ids: Tokenized input
                - attention_mask: Attention mask
                - pixel_values: Processed images
                - labels: Labels for training
                - metadata: Original metadata
        """
        # Load JSON file
        json_path = self.json_files[idx]
        json_data = self.load_json_file(json_path)

        # Process conversation to text
        prompt = self.process_conversation(json_data)

        # Load images
        images = self.load_images(json_data)

        # Prepare for processor
        # Convert numpy arrays to PIL Images if processor expects PIL
        from PIL import Image as PILImage

        pil_images = []
        for img in images:
            # Normalize to 0-255 range for PIL
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Convert to RGB (duplicate channels for grayscale)
            img_rgb = np.stack([img, img, img], axis=-1)
            pil_img = PILImage.fromarray(img_rgb)
            pil_images.append(pil_img)

        # Tokenize and process with LLaVA processor
        if self.processor is not None and self.tokenizer is not None:
            # Use LLaVA processor for vision-language encoding
            encoding = self.processor(
                text=prompt,
                images=pil_images,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )

            # Create labels (same as input_ids for causal LM)
            labels = encoding["input_ids"].clone()

            # Optionally: mask user tokens in labels (train only on assistant)
            # This would require more sophisticated masking logic

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "pixel_values": encoding["pixel_values"].squeeze(0) if "pixel_values" in encoding else None,
                "labels": labels.squeeze(0),
                "metadata": json_data["metadata"],
                "task_id": json_data["task_id"]
            }

        else:
            # Fallback: manual tokenization
            if self.tokenizer is not None:
                encoding = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True
                )

                return {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "pixel_values": None,  # Would need separate processing
                    "labels": encoding["input_ids"].squeeze(0),
                    "metadata": json_data["metadata"],
                    "task_id": json_data["task_id"]
                }

            else:
                # No tokenizer provided, return raw data
                return {
                    "prompt": prompt,
                    "images": images,
                    "metadata": json_data["metadata"],
                    "task_id": json_data["task_id"]
                }

    def get_example_by_task_id(self, task_id: str) -> Optional[Dict]:
        """
        Get example by task ID.

        Args:
            task_id: Task ID to search for

        Returns:
            Example dictionary or None if not found
        """
        for idx, json_path in enumerate(self.json_files):
            json_data = self.load_json_file(json_path)
            if json_data["task_id"] == task_id:
                return self.__getitem__(idx)

        return None

    def get_dataset_statistics(self) -> Dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with statistics
        """
        total_samples = len(self)
        total_images = 0
        total_turns = 0
        task_types = {}

        for json_path in self.json_files:
            try:
                json_data = self.load_json_file(json_path)
                total_images += len(json_data["images"])
                total_turns += len(json_data["conversations"])

                task_type = json_data.get("task_type", "unknown")
                task_types[task_type] = task_types.get(task_type, 0) + 1

            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
                continue

        return {
            "total_samples": total_samples,
            "total_images": total_images,
            "total_turns": total_turns,
            "avg_images_per_sample": total_images / total_samples if total_samples > 0 else 0,
            "avg_turns_per_sample": total_turns / total_samples if total_samples > 0 else 0,
            "task_types": task_types
        }


# Example usage
if __name__ == "__main__":
    print("T1 JSON Dataset - Example Usage")
    print("="*60)

    # Example: Create dataset without tokenizer/processor
    json_dir = Path("../../sample_data/sex_comparison_conversations/samples")

    if json_dir.exists():
        dataset = T1JSONDataset(
            json_dir=json_dir,
            image_root=None,  # Use absolute paths from JSON
            tokenizer=None,
            processor=None,
            normalize=True,
            standardize=True
        )

        print(f"\nDataset size: {len(dataset)}")

        # Get first example
        print("\nLoading first example...")
        try:
            example = dataset[0]
            print(f"Task ID: {example['task_id']}")
            print(f"Prompt length: {len(example['prompt'])} characters")
            print(f"Number of images: {len(example['images'])}")
            print(f"Metadata: {example['metadata']}")
        except Exception as e:
            print(f"Error loading example: {e}")

        # Get statistics
        print("\nDataset statistics:")
        stats = dataset.get_dataset_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        print(f"\nSample directory not found: {json_dir}")
        print("Please run generate_sex_comparison_conversations.py first.")

    print("\n" + "="*60)
    print("Dataset ready for use with LLaVA processor.")
