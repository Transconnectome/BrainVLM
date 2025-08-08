import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union,  List, Iterator

import numpy as np

from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import itertools
import math


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/data/data_collator.py#L237
    """

    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class CustomDataCollatorWithPadding:
    """
    Modified from: 
    https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/data/data_collator.py#L237
    """

    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def _process_modality(self, modal_batch):
        padded = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            modal_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Handle label keys
        if "label" in padded:
            padded["labels"] = padded.pop("label")
        if "label_ids" in padded:
            padded["labels"] = padded.pop("label_ids")

        return padded

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        inputs shape:
        [
            [
                {
                    'pixel_values': {'T1': torch.tensor},
                    'input_ids': {'T1': torch.tensor},
                    'attention_mask': {'T1': torch.tensor},
                    'labels': {'T1': torch.tensor},
                },
                {
                    'pixel_values': {'T1': torch.tensor},
                    'input_ids': {'T1': torch.tensor},
                    'attention_mask': {'T1': torch.tensor},
                    'labels': {'T1': torch.tensor},
                },
            ],
            [
                {
                    'pixel_values': {'rsfMRI': torch.tensor},
                    'input_ids': {'rsfMRI': torch.tensor},
                    'attention_mask': {'rsfMRI': torch.tensor},
                    'labels': {'rsfMRI': torch.tensor},
                }, 
                {
                    'pixel_values': {'rsfMRI': torch.tensor},
                    'input_ids': {'rsfMRI': torch.tensor},
                    'attention_mask': {'rsfMRI': torch.tensor},
                    'labels': {'rsfMRI': torch.tensor},
                }
            ]
        ]
        
        
        intermediate data shape for padding
        {
            'T1':
                {
                'pixel_values': torch.tensor([torch.tensor]),
                'input_ids': torch.tensor([torch.tensor]),
                'attention_mask': torch.tensor([torch.tensor]),
                'labels': torch.tensor([torch.tensor])
                },

            'rsfMRI':
                {
                'pixel_values': torch.tensor([torch.tensor]),
                'input_ids': torch.tensor([torch.tensor]),
                'attention_mask': torch.tensor([torch.tensor]),
                'labels': torch.tensor([torch.tensor])
                },

        }

        

        """
        # Extract unique modalities
        modalities = list(set(
            modality for feature in features 
            for modality in feature['pixel_values'].keys()
        ))

        # Initialize batch structure
        batch = {
            modality: {
                key: [] for key in ['pixel_values', 'input_ids', 'attention_mask', 'labels']
            } for modality in modalities
        }

        # Collect features by modality
        for feature in features:
            modality = next(iter(feature['pixel_values'].keys()))
            for key in batch[modality].keys():
                batch[modality][key].append(feature[key][modality])

        # Apply padding for each modality
        return {
            modality: self._process_modality(batch[modality])
            for modality in modalities
        }


from torch.utils.data import Dataset
import random
from typing import List, Dict
import math

class InterleaveDataset(Dataset):
    def __init__(
        self,
        datasets: List[Dataset],
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.datasets = datasets
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.drop_last = drop_last
        
        # Calculate sizes and probabilities
        self.sizes = [len(dataset) for dataset in datasets]
        total_size = sum(self.sizes)
        self.probabilities = [size/total_size for size in self.sizes]
        
        # Track available indices per dataset
        self.available_indices: Dict[int, List[int]] = {}
        self.reset_indices()
        
        # Calculate total length considering drop_last
        if self.drop_last:
            # Round down to nearest multiple of num_datasets
            self.length = (total_size // len(datasets)) * len(datasets)
        else:
            self.length = total_size
    
    def reset_indices(self):
        """Reset available indices for all datasets"""
        self.available_indices = {
            i: list(range(size)) for i, size in enumerate(self.sizes)
        }
        if self.shuffle:
            for indices in self.available_indices.values():
                self.rng.shuffle(indices)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of range")
            
        available_datasets = [
            i for i, indices in self.available_indices.items() 
            if len(indices) > 0
        ]
        
        if not available_datasets:
            self.reset_indices()
            available_datasets = list(range(len(self.datasets)))
            
            if self.drop_last and idx >= self.__len__():
                raise IndexError("Dropped last incomplete batch")
        
        # Calculate normalized probabilities for available datasets
        probs = [self.probabilities[i] for i in available_datasets]
        sum_probs = sum(probs)
        norm_probs = [p/sum_probs for p in probs]
        
        # Select dataset and get sample
        dataset_idx = self.rng.choices(available_datasets, weights=norm_probs, k=1)[0]
        sample_idx = self.available_indices[dataset_idx].pop()
        
        return self.datasets[dataset_idx][sample_idx]


