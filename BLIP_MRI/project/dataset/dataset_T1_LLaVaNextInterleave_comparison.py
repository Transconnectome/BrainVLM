"""
Multi-turn Conversation Dataset for Comparison Tasks

Supports:
- Sex comparison (reference + query)
- Age comparison (reference + query)
- 2-turn conversation format
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from monai.data import NibabelReader
from monai.transforms import LoadImage, Randomizable, apply_transform, AddChannel, Compose, Resize, NormalizeIntensity, RandAxisFlip, ToTensor
from monai.utils import MAX_SEED, get_seed

from utils.utils import to_3tuple


class MultiTurnComparisonDataset(Dataset, Randomizable):
    """
    Multi-turn conversation dataset for comparison-based tasks

    Format:
    Turn 1: User shows reference image + label → Assistant acknowledges
    Turn 2: User shows query image + asks question → Assistant predicts

    Args:
        json_path: Path to JSON file with comparison tasks
        processor: HuggingFace processor
        img_size: Image size [H, W, D]
        mode: 'train' or 'eval'
    """

    def __init__(self,
                 json_path=None,
                 processor=None,
                 img_size=None,
                 mode='train'):

        self.json_path = json_path
        self.processor = processor
        self.tokenizer = processor.tokenizer if processor is not None else None
        self.img_size = img_size
        self.mode = mode

        # Load JSON data
        with open(json_path, 'r') as f:
            self.tasks = json.load(f)

        print(f"Loaded {len(self.tasks)} tasks from {json_path}")

        # Define image transform
        self.image_transform = self.define_augmentation(mode=mode)
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)

        self.set_random_state(seed=get_seed())
        self._seed = 0


    def define_augmentation(self, mode='train'):
        """Define image augmentation"""
        img_size = to_3tuple(self.img_size)
        if mode == 'train':
            transform = Compose([
                AddChannel(),
                Resize(img_size),
                RandAxisFlip(prob=0.5),
                NormalizeIntensity()
            ])
        elif mode == 'eval':
            transform = Compose([
                AddChannel(),
                Resize(img_size),
                NormalizeIntensity()
            ])
        return transform


    def randomize(self, data=None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype='uint32')


    def __transform_image__(self, image_file):
        """Load and transform a single image"""
        image = self.image_loader(image_file)
        if self.image_transform is not None:
            if isinstance(self.image_transform, Randomizable):
                self.image_transform.set_random_state(seed=self._seed)
            image = apply_transform(self.image_transform, image, map_items=False)
            image = torch.tensor(image)
        return image


    def __build_conversation_text__(self, task):
        """
        Build multi-turn conversation text from task

        Returns:
            full_text: Complete conversation in LLaVA-NeXT-Interleave format
            answer_start_pos: Position where assistant's final answer starts (for label masking)
        """

        conversations = task['conversations']

        # Build conversation following Qwen2 format
        full_text = ""

        for i, turn in enumerate(conversations):
            role = turn['role']
            content_list = turn['content']

            if role == 'user':
                full_text += "<|im_start|>user\n"
            elif role == 'assistant':
                full_text += "<|im_start|>assistant\n"

            # Process content (text + image tokens)
            for content_item in content_list:
                if content_item['type'] == 'text':
                    full_text += content_item['text']
                elif content_item['type'] == 'image':
                    full_text += "<image>"

            full_text += "<|im_end|>\n"

        return full_text

    def __preprocess_as_hf__(self, images, full_text):
        """
        Tokenize multi-turn conversation and apply instruction masking

        Args:
            images: List of [ref_image_tensor, query_image_tensor]
            full_text: Complete conversation text

        Returns:
            Dictionary with pixel_values, input_ids, attention_mask, labels
        """
        inputs = {}
        inputs['pixel_values'] = {}
        inputs['input_ids'] = {}
        inputs['attention_mask'] = {}
        inputs['labels'] = {}

        # ========== 핵심 수정! ==========
        # 두 이미지를 개별적으로 batch 차원 추가한 후 합치기
        # ref_image: [C, H, W, D] → [1, C, H, W, D]
        # query_image: [C, H, W, D] → [1, C, H, W, D]
        # 합치기: [2, C, H, W, D] → 이제 PatchEmbed가 batch=2로 처리

        processed_images = []
        for img in images:
            # Add batch dimension to each image
            processed_images.append(img.unsqueeze(0))  # [1, C, H, W, D]

        # Concatenate along batch dimension
        batched_images = torch.cat(processed_images, dim=0)  # [2, C, H, W, D]

        inputs['pixel_values']['T1'] = batched_images
        # ==================================

        # Tokenize full conversation
        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = full_encoding['input_ids'].squeeze(0)
        attention_mask = full_encoding['attention_mask'].squeeze(0)

        # Initialize labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Mask padding

        # Apply instruction masking: mask everything except the LAST assistant's response
        # We want to train only on the final answer, not on the intermediate "Understood" response

        # Find all assistant tokens
        assistant_pattern = "<|im_start|>assistant\n"
        assistant_tokens = self.tokenizer.encode(assistant_pattern, add_special_tokens=False)
        assistant_tensor = torch.tensor(assistant_tokens, device=input_ids.device)

        assistant_positions = []
        for i in range(len(input_ids) - len(assistant_tokens) + 1):
            if torch.equal(input_ids[i:i+len(assistant_tokens)], assistant_tensor):
                assistant_positions.append(i + len(assistant_tokens))

        if len(assistant_positions) >= 2:
            # Mask everything before the LAST assistant response
            last_assistant_pos = assistant_positions[-1]
            labels[:last_assistant_pos] = -100
        elif len(assistant_positions) == 1:
            # Only one assistant response (shouldn't happen in 2-turn, but handle it)
            labels[:assistant_positions[0]] = -100

        inputs['input_ids']['T1'] = input_ids
        inputs['attention_mask']['T1'] = attention_mask
        inputs['labels']['T1'] = labels

        return inputs


    def __len__(self) -> int:
        return len(self.tasks)


    def __getitem__(self, index: int):
        """
        Returns a multi-turn comparison sample

        Returns:
            Dictionary with:
            - pixel_values: Tensor [num_images, C, H, W, D] (dynamically determined from JSON)
            - input_ids, attention_mask, labels: Tokenized multi-turn conversation
            - modality: 'Comparison'
        """

        task = self.tasks[index]

        # Load ALL images dynamically (supports N references + 1 query)
        # JSON format: images = [ref1, ref2, ..., refN, query]
        images = []
        for img_info in task['images']:
            img_path = img_info['path']
            img_tensor = self.__transform_image__(img_path)
            images.append(img_tensor)

        # Build conversation text
        full_text = self.__build_conversation_text__(task)

        # Preprocess for model
        inputs = self.__preprocess_as_hf__(images=images, full_text=full_text)
        # Don't add 'modality' key - trainer extracts modality from dict keys (T1, rsfMRI, etc.)

        return inputs


class ComparisonDataModule:
    """
    Data module for comparison tasks (train/val/test splits)
    """

    def __init__(self,
                 train_json=None,
                 val_json=None,
                 test_json=None,
                 processor=None,
                 img_size=None):

        self.train_json = train_json
        self.val_json = val_json
        self.test_json = test_json
        self.processor = processor
        self.img_size = img_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.setup()


    def setup(self):
        """Create train/val/test datasets"""

        if self.train_json is not None:
            self.train_dataset = MultiTurnComparisonDataset(
                json_path=self.train_json,
                processor=self.processor,
                img_size=self.img_size,
                mode='train'
            )
            print(f"Train: {len(self.train_dataset)} tasks")

        if self.val_json is not None:
            self.val_dataset = MultiTurnComparisonDataset(
                json_path=self.val_json,
                processor=self.processor,
                img_size=self.img_size,
                mode='eval'
            )
            print(f"Val: {len(self.val_dataset)} tasks")

        if self.test_json is not None:
            self.test_dataset = MultiTurnComparisonDataset(
                json_path=self.test_json,
                processor=self.processor,
                img_size=self.img_size,
                mode='eval'
            )
            print(f"Test: {len(self.test_dataset)} tasks")

        return self.train_dataset, self.val_dataset, self.test_dataset
