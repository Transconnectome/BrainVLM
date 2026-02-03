"""
BrainVLM Dataset Module

This module provides dataset classes for loading and processing neuroimaging data
for the BrainVLM multimodal brain imaging model.

Classes:
    - BaseDataset classes (legacy): For backward compatibility
    - BasefMRIDataset: Abstract base class for JSON-based fMRI datasets
    - Dataset-specific fMRI classes: ABCD, UKB, HCP, HBN, ABIDE
    - T1JSONDataset: JSON-based T1/sMRI dataset

Usage:
    # New JSON-based fMRI datasets
    from project.dataset import ABCDfMRIDataset, create_fmri_dataset

    dataset = ABCDfMRIDataset(
        json_file='samples.json',
        data_root='/data/ABCD',
        tokenizer=tokenizer,
        sequence_length=20
    )

    # Or use factory function
    dataset = create_fmri_dataset(
        dataset_name='ABCD',
        json_file='samples.json',
        data_root='/data/ABCD',
        tokenizer=tokenizer
    )

    # T1/sMRI dataset
    from project.dataset import T1JSONDataset, create_t1_dataset

    dataset = T1JSONDataset(
        json_file='t1_samples.json',
        data_root='/data/T1',
        tokenizer=tokenizer,
        img_size=128
    )
"""

# Legacy imports for backward compatibility
from .dataset_rsfMRI import (
    BaseDataset,
    S1200,
    ABCD,
    UKB,
    HBN,
    ABIDE,
    Dummy
)

from .dataset_T1_LLaVa import (
    BaseDataset_T1,
    ABCD_T1,
    UKB_T1
)

# New JSON-based dataset imports
from .base_fmri_dataset import (
    BasefMRIDataset,
    RawfMRIDataset
)

from .fmri_datasets import (
    ABCDfMRIDataset,
    UKBfMRIDataset,
    UKBfMRIDatasetRaw,
    HCPfMRIDataset,
    HBNfMRIDataset,
    HBNfMRIDatasetRaw,
    ABIDEfMRIDataset,
    ABIDEfMRIDatasetRaw,
    create_fmri_dataset
)

from .t1_json_dataset import (
    T1JSONDataset,
    T1JSONDatasetRaw,
    create_t1_dataset
)

from .dmri_json_dataset import (
    dMRIJSONDataset,
    dMRIJSONDatasetRaw,
    create_dmri_dataset
)

# Utility imports
from .dataset_utils import (
    load_json,
    load_nifti,
    load_pt_frames,
    normalize_fmri,
    pad_tensor,
    format_conversation,
    tokenize_conversation,
    get_num_frames,
    resolve_path,
    DatasetConfig,
    DATASET_CONFIGS
)

# Data module imports
try:
    from .datamodule_rsfMRI import rsfMRIData
except ImportError:
    rsfMRIData = None

__all__ = [
    # Legacy classes
    'BaseDataset',
    'S1200',
    'ABCD',
    'UKB',
    'HBN',
    'ABIDE',
    'Dummy',
    'BaseDataset_T1',
    'ABCD_T1',
    'UKB_T1',
    'rsfMRIData',

    # New base classes
    'BasefMRIDataset',
    'RawfMRIDataset',

    # New fMRI dataset classes
    'ABCDfMRIDataset',
    'UKBfMRIDataset',
    'UKBfMRIDatasetRaw',
    'HCPfMRIDataset',
    'HBNfMRIDataset',
    'HBNfMRIDatasetRaw',
    'ABIDEfMRIDataset',
    'ABIDEfMRIDatasetRaw',
    'create_fmri_dataset',

    # T1/sMRI classes
    'T1JSONDataset',
    'T1JSONDatasetRaw',
    'create_t1_dataset',

    # dMRI classes
    'dMRIJSONDataset',
    'dMRIJSONDatasetRaw',
    'create_dmri_dataset',

    # Utilities
    'load_json',
    'load_nifti',
    'load_pt_frames',
    'normalize_fmri',
    'pad_tensor',
    'format_conversation',
    'tokenize_conversation',
    'get_num_frames',
    'resolve_path',
    'DatasetConfig',
    'DATASET_CONFIGS',
]
