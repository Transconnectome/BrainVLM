"""
Dataset-Specific fMRI Implementations for BrainVLM

Concrete implementations for ABCD, UKB, HCP, HBN, and ABIDE datasets.
Each class provides dataset-specific padding, TR, and normalization settings.
"""

import torch
from typing import Dict, Tuple, Any

from .base_fmri_dataset import BasefMRIDataset, RawfMRIDataset


class ABCDfMRIDataset(BasefMRIDataset):
    """
    ABCD (Adolescent Brain Cognitive Development) fMRI Dataset.

    Dataset characteristics:
        - Image shape: (96, 96, 95)
        - Padding: (0, 1, 0, 0, 0, 0) -> (96, 96, 96)
        - TR: 0.8s
        - Downsample ratio: 2.5
        - Background: First element of flattened tensor
    """

    def get_padding_config(self) -> Tuple[int, ...]:
        """
        Get ABCD-specific padding configuration.

        Returns:
            Padding tuple (0, 1, 0, 0, 0, 0) to convert (96, 96, 95) -> (96, 96, 96).
        """
        return (0, 1, 0, 0, 0, 0)

    def get_temporal_config(self) -> Dict[str, float]:
        """
        Get ABCD temporal configuration.

        Returns:
            Dict with TR=0.8s and downsample_ratio=2.5.
        """
        return {
            'tr': 0.8,
            'downsample_ratio': 2.5
        }

    def get_background_value(self, tensor: torch.Tensor) -> float:
        """
        Get background value from first element of flattened tensor.

        Args:
            tensor: Input tensor.

        Returns:
            First element as background value.
        """
        return tensor.flatten()[0].item()


class UKBfMRIDataset(BasefMRIDataset):
    """
    UK Biobank fMRI Dataset.

    Dataset characteristics:
        - Image shape: (88, 88, 64) - varies by acquisition
        - Padding: (3, 9, 0, 0, 10, 8) -> (96, 96, 96)
        - TR: ~0.735s (varies by acquisition)
        - Background: First element of flattened tensor
    """

    def get_padding_config(self) -> Tuple[int, ...]:
        """
        Get UKB-specific padding configuration.

        Returns:
            Padding tuple (3, 9, 0, 0, 10, 8) to standardize to (96, 96, 96).
        """
        return (3, 9, 0, 0, 10, 8)

    def get_temporal_config(self) -> Dict[str, float]:
        """
        Get UKB temporal configuration.

        Returns:
            Dict with TR=0.735s and downsample_ratio=1.0.
        """
        return {
            'tr': 0.735,
            'downsample_ratio': 1.0
        }

    def get_background_value(self, tensor: torch.Tensor) -> float:
        """
        Get background value from first element of flattened tensor.

        Args:
            tensor: Input tensor.

        Returns:
            First element as background value.
        """
        return tensor.flatten()[0].item()


class UKBfMRIDatasetRaw(RawfMRIDataset):
    """
    UK Biobank fMRI Dataset with raw output format.

    Returns dict with fmri_sequence, subject_name, target, etc.
    instead of HuggingFace formatted output.
    """

    def get_padding_config(self) -> Tuple[int, ...]:
        return (3, 9, 0, 0, 10, 8)

    def get_temporal_config(self) -> Dict[str, float]:
        return {'tr': 0.735, 'downsample_ratio': 1.0}

    def get_background_value(self, tensor: torch.Tensor) -> float:
        return tensor.flatten()[0].item()

    def get_study_name(self) -> str:
        return 'UKB'


class HCPfMRIDataset(BasefMRIDataset):
    """
    Human Connectome Project (HCP S1200) fMRI Dataset.

    Dataset characteristics:
        - Image shape: (91, 109, 91)
        - Padding: (3, 9, 0, 0, 10, 8) -> (96, 96, 96) approximately
        - TR: 0.72s
        - Background: First element of flattened tensor
    """

    def get_padding_config(self) -> Tuple[int, ...]:
        """
        Get HCP-specific padding configuration.

        Returns:
            Padding tuple (3, 9, 0, 0, 10, 8).
        """
        return (3, 9, 0, 0, 10, 8)

    def get_temporal_config(self) -> Dict[str, float]:
        """
        Get HCP temporal configuration.

        Returns:
            Dict with TR=0.72s and downsample_ratio=1.0.
        """
        return {
            'tr': 0.72,
            'downsample_ratio': 1.0
        }

    def get_background_value(self, tensor: torch.Tensor) -> float:
        """
        Get background value from first element of flattened tensor.

        Args:
            tensor: Input tensor.

        Returns:
            First element as background value.
        """
        return tensor.flatten()[0].item()


class HBNfMRIDataset(BasefMRIDataset):
    """
    Healthy Brain Network (HBN) fMRI Dataset.

    Dataset characteristics:
        - Rest image shape: (81, 95, 81)
        - Task image shape: (96, 96, 95)
        - Rest padding: (7, 8, 1, 0, 7, 8)[:,:,:,:96,:] -> (96, 96, 96)
        - Task padding: (0, 1, 0, 0, 0, 0) -> (96, 96, 96)
        - TR: varies by site
        - Background: First element of flattened tensor

    Note:
        This class handles both rest and task fMRI. Set `input_type` in kwargs
        to 'rest' or 'task' to select the appropriate padding.
    """

    def __init__(self, *args, input_type: str = 'rest', **kwargs):
        """
        Initialize HBN dataset.

        Args:
            input_type: Type of fMRI data ('rest' or 'task').
            *args, **kwargs: Passed to parent class.
        """
        self.input_type = input_type
        super().__init__(*args, **kwargs)

    def get_padding_config(self) -> Tuple[int, ...]:
        """
        Get HBN-specific padding configuration based on input type.

        Returns:
            Padding tuple for rest or task data.
        """
        if self.input_type == 'rest':
            # HBN rest image shape: (81, 95, 81)
            return (7, 8, 1, 0, 7, 8)
        else:
            # HBN task image shape: (96, 96, 95)
            return (0, 1, 0, 0, 0, 0)

    def get_temporal_config(self) -> Dict[str, float]:
        """
        Get HBN temporal configuration.

        Returns:
            Dict with TR (varies by site) and downsample_ratio=1.0.
        """
        return {
            'tr': 0.8,  # Varies by site, using common value
            'downsample_ratio': 1.0
        }

    def get_background_value(self, tensor: torch.Tensor) -> float:
        """
        Get background value from first element of flattened tensor.

        Args:
            tensor: Input tensor.

        Returns:
            First element as background value.
        """
        return tensor.flatten()[0].item()

    def load_and_process_fmri(
        self,
        subject_path: str,
        start_frame: int,
        sample_duration: int,
        num_frames: int
    ) -> torch.Tensor:
        """
        Load and preprocess HBN fMRI data with type-specific handling.

        Args:
            subject_path: Path to subject's frame directory.
            start_frame: Starting frame index.
            sample_duration: Number of frames to load.
            num_frames: Total available frames.

        Returns:
            Processed fMRI tensor of shape (1, H, W, D, T).
        """
        # Call parent method for basic loading
        y = super().load_and_process_fmri(
            subject_path, start_frame, sample_duration, num_frames
        )

        # For rest data, crop to 96 on the fourth dimension
        if self.input_type == 'rest':
            y = y[:, :, :, :96, :]

        return y


class HBNfMRIDatasetRaw(RawfMRIDataset):
    """
    HBN fMRI Dataset with raw output format.
    """

    def __init__(self, *args, input_type: str = 'rest', **kwargs):
        self.input_type = input_type
        super().__init__(*args, **kwargs)

    def get_padding_config(self) -> Tuple[int, ...]:
        if self.input_type == 'rest':
            return (7, 8, 1, 0, 7, 8)
        else:
            return (0, 1, 0, 0, 0, 0)

    def get_temporal_config(self) -> Dict[str, float]:
        return {'tr': 0.8, 'downsample_ratio': 1.0}

    def get_background_value(self, tensor: torch.Tensor) -> float:
        return tensor.flatten()[0].item()

    def get_study_name(self) -> str:
        return 'HBN'

    def load_and_process_fmri(
        self,
        subject_path: str,
        start_frame: int,
        sample_duration: int,
        num_frames: int
    ) -> torch.Tensor:
        y = super().load_and_process_fmri(
            subject_path, start_frame, sample_duration, num_frames
        )
        if self.input_type == 'rest':
            y = y[:, :, :, :96, :]
        return y


class ABIDEfMRIDataset(BasefMRIDataset):
    """
    ABIDE (Autism Brain Imaging Data Exchange) fMRI Dataset.

    Dataset characteristics:
        - Image shape: (97, 115, 97)
        - Padding: (0, -1, -10, -9, -1, 0) -> (96, 96, 96)
        - TR: varies by site (typically ~2.0s)
        - Background: First element of flattened tensor

    Note:
        ABIDE uses NEGATIVE padding values to crop the image.
    """

    def get_padding_config(self) -> Tuple[int, ...]:
        """
        Get ABIDE-specific padding configuration.

        Returns:
            Padding tuple (0, -1, -10, -9, -1, 0) to crop (97, 115, 97) -> (96, 96, 96).
        """
        return (0, -1, -10, -9, -1, 0)

    def get_temporal_config(self) -> Dict[str, float]:
        """
        Get ABIDE temporal configuration.

        Returns:
            Dict with TR=2.0s (varies by site) and downsample_ratio=1.0.
        """
        return {
            'tr': 2.0,  # Varies by site
            'downsample_ratio': 1.0
        }

    def get_background_value(self, tensor: torch.Tensor) -> float:
        """
        Get background value from first element of flattened tensor.

        Args:
            tensor: Input tensor.

        Returns:
            First element as background value.
        """
        return tensor.flatten()[0].item()


class ABIDEfMRIDatasetRaw(RawfMRIDataset):
    """
    ABIDE fMRI Dataset with raw output format.
    """

    def get_padding_config(self) -> Tuple[int, ...]:
        return (0, -1, -10, -9, -1, 0)

    def get_temporal_config(self) -> Dict[str, float]:
        return {'tr': 2.0, 'downsample_ratio': 1.0}

    def get_background_value(self, tensor: torch.Tensor) -> float:
        return tensor.flatten()[0].item()

    def get_study_name(self) -> str:
        return 'ABIDE'


# Convenience factory function
def create_fmri_dataset(
    dataset_name: str,
    json_file: str,
    data_root: str,
    output_format: str = 'hf',
    **kwargs
) -> BasefMRIDataset:
    """
    Factory function to create the appropriate fMRI dataset.

    Args:
        dataset_name: Name of dataset ('ABCD', 'UKB', 'HCP', 'HBN', 'ABIDE').
        json_file: Path to JSON file with sample definitions.
        data_root: Root directory for data files.
        output_format: Output format ('hf' for HuggingFace, 'raw' for raw).
        **kwargs: Additional arguments passed to dataset class.

    Returns:
        Instantiated dataset object.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    dataset_map = {
        'ABCD': ABCDfMRIDataset,
        'UKB': UKBfMRIDataset if output_format == 'hf' else UKBfMRIDatasetRaw,
        'HCP': HCPfMRIDataset,
        'S1200': HCPfMRIDataset,  # Alias for HCP
        'HBN': HBNfMRIDataset if output_format == 'hf' else HBNfMRIDatasetRaw,
        'ABIDE': ABIDEfMRIDataset if output_format == 'hf' else ABIDEfMRIDatasetRaw,
    }

    if dataset_name.upper() not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(dataset_map.keys())}"
        )

    dataset_class = dataset_map[dataset_name.upper()]
    return dataset_class(json_file=json_file, data_root=data_root, **kwargs)


# Export all classes
__all__ = [
    'ABCDfMRIDataset',
    'UKBfMRIDataset',
    'UKBfMRIDatasetRaw',
    'HCPfMRIDataset',
    'HBNfMRIDataset',
    'HBNfMRIDatasetRaw',
    'ABIDEfMRIDataset',
    'ABIDEfMRIDatasetRaw',
    'create_fmri_dataset',
]
