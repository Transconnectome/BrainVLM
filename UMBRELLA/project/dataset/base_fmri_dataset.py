"""
Base fMRI Dataset for BrainVLM

Abstract base class for JSON-based fMRI datasets with unified interface.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union

from .dataset_utils import (
    load_json,
    load_pt_frames,
    normalize_fmri,
    pad_tensor,
    format_conversation,
    tokenize_conversation,
    get_num_frames,
    resolve_path
)


class BasefMRIDataset(Dataset, ABC):
    """
    Abstract base class for JSON-based fMRI datasets.

    This class provides a unified interface for loading fMRI data from various
    datasets (ABCD, UKB, HCP, HBN, ABIDE) using JSON sample definitions.

    Subclasses must implement:
        - get_padding_config(): Returns padding tuple
        - get_temporal_config(): Returns TR and downsample ratio
        - get_background_value(): Returns padding value method

    Attributes:
        samples (List[Dict]): Loaded sample definitions from JSON.
        data_root (str): Root directory for data files.
        modality (str): Modality type ('fMRI').
        tokenizer: HuggingFace tokenizer for text processing.
        max_seq_length (int): Maximum token sequence length.
        sequence_length (int): Number of fMRI frames per sample.
        stride (int): Stride between samples.
        input_scaling_method (str): Normalization method.
        train (bool): Whether in training mode.
    """

    def __init__(
        self,
        json_file: str,
        data_root: str,
        modality: str = 'fMRI',
        tokenizer=None,
        image_processor=None,
        max_seq_length: int = 128,
        sequence_length: int = 20,
        stride_within_seq: int = 1,
        stride_between_seq: int = 1,
        input_scaling_method: str = 'znorm_zeroback',
        shuffle_time_sequence: bool = False,
        train: bool = True,
        add_context: bool = False,
        **kwargs
    ):
        """
        Initialize the base fMRI dataset.

        Args:
            json_file: Path to JSON file with sample definitions.
            data_root: Root directory for data files.
            modality: Modality type (default: 'fMRI').
            tokenizer: HuggingFace tokenizer for text processing.
            image_processor: Image processor (not used for fMRI).
            max_seq_length: Maximum token sequence length.
            sequence_length: Number of fMRI frames per sample.
            stride_within_seq: Frame skip factor within sequence.
            stride_between_seq: Sequence skip factor.
            input_scaling_method: Normalization method.
            shuffle_time_sequence: Whether to shuffle frame order.
            train: Whether in training mode.
            add_context: Whether to add demographic context.
            **kwargs: Additional arguments for subclasses.
        """
        super().__init__()

        self.json_file = json_file
        self.data_root = data_root
        self.modality = modality
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_length = max_seq_length
        self.sequence_length = sequence_length
        self.stride_within_seq = stride_within_seq
        self.stride_between_seq = stride_between_seq
        self.input_scaling_method = input_scaling_method
        self.shuffle_time_sequence = shuffle_time_sequence
        self.train = train
        self.add_context = add_context

        # Store additional kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Calculate sample duration and stride
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)

        # Load samples from JSON
        self.samples = load_json(json_file)

        # Build data index
        self.data = self._build_data_index()

    def _build_data_index(self) -> List[Tuple]:
        """
        Build data index from JSON samples.

        Returns:
            List of data tuples with format:
                (index, subject_id, subject_path, start_frame, sample_duration,
                 num_frames, conversations, metadata)
        """
        data = []

        for i, sample in enumerate(self.samples):
            subject_id = sample.get('subject_id', f'subject_{i}')
            conversations = sample.get('conversations', [])

            # Get fMRI path from modality_paths
            modality_paths = sample.get('modality_paths', {})
            fmri_key = None
            for key in modality_paths:
                if 'fmri' in key.lower() or 'rsfmri' in key.lower():
                    fmri_key = key
                    break

            if fmri_key is None:
                continue

            # Resolve path
            fmri_path = modality_paths[fmri_key]
            subject_path = resolve_path(fmri_path, self.data_root)

            # Get number of available frames
            num_frames = get_num_frames(subject_path)

            if num_frames == 0:
                continue

            # Calculate session duration
            session_duration = num_frames - self.sample_duration + 1

            if session_duration <= 0:
                # Use all available frames if not enough
                data_tuple = (
                    len(data),
                    subject_id,
                    subject_path,
                    0,
                    min(num_frames, self.sample_duration),
                    num_frames,
                    conversations,
                    sample.get('metadata', {})
                )
                data.append(data_tuple)
            else:
                # Create multiple samples from long sequences
                for start_frame in range(0, session_duration, self.stride):
                    data_tuple = (
                        len(data),
                        subject_id,
                        subject_path,
                        start_frame,
                        self.sample_duration,
                        num_frames,
                        conversations,
                        sample.get('metadata', {})
                    )
                    data.append(data_tuple)

        return data

    @abstractmethod
    def get_padding_config(self) -> Tuple[int, ...]:
        """
        Get padding configuration for this dataset.

        Returns:
            Tuple of padding values for torch.nn.functional.pad.
            Format: (left, right, top, bottom, front, back) for 3D padding.
        """
        raise NotImplementedError("Subclass must implement get_padding_config()")

    @abstractmethod
    def get_temporal_config(self) -> Dict[str, float]:
        """
        Get temporal configuration for this dataset.

        Returns:
            Dict with:
                - 'tr': Repetition time in seconds
                - 'downsample_ratio': Temporal downsampling ratio
        """
        raise NotImplementedError("Subclass must implement get_temporal_config()")

    @abstractmethod
    def get_background_value(self, tensor: torch.Tensor) -> float:
        """
        Get background value for padding.

        Args:
            tensor: Input tensor to determine background from.

        Returns:
            Background value for padding.
        """
        raise NotImplementedError("Subclass must implement get_background_value()")

    def load_and_process_fmri(
        self,
        subject_path: str,
        start_frame: int,
        sample_duration: int,
        num_frames: int
    ) -> torch.Tensor:
        """
        Load and preprocess fMRI data.

        Args:
            subject_path: Path to subject's frame directory.
            start_frame: Starting frame index.
            sample_duration: Number of frames to load.
            num_frames: Total available frames.

        Returns:
            Processed fMRI tensor of shape (1, H, W, D, T).
        """
        # Load frames
        y = load_pt_frames(
            subject_path=subject_path,
            start_frame=start_frame,
            num_frames=sample_duration,
            stride=self.stride_within_seq,
            shuffle=self.shuffle_time_sequence
        )

        # Apply normalization
        if self.input_scaling_method != 'none':
            stats_path = os.path.join(subject_path, 'global_stats.pt')
            y = normalize_fmri(y, stats_path, self.input_scaling_method)

        # Get background value before padding
        background_value = self.get_background_value(y)

        # Apply padding
        padding = self.get_padding_config()
        y = pad_tensor(y, padding, background_value)

        return y

    def process_text(
        self,
        conversations: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Process conversations into instruction and answer.

        Args:
            conversations: List of conversation turns.
            metadata: Optional metadata for context.

        Returns:
            Tuple of (instruction, answer) strings.
        """
        instruction, answer = format_conversation(
            conversations,
            include_image_token=True,
            modality=self.modality
        )

        # Add context if requested
        if self.add_context and metadata:
            context_parts = []
            if 'sex' in metadata:
                context_parts.append(f"Sex: {metadata['sex']}")
            if 'age' in metadata:
                context_parts.append(f"Age: {metadata['age']}")

            if context_parts:
                context_str = " ".join(context_parts)
                instruction = instruction.replace(
                    "ASSISTANT:",
                    f"{context_str} ASSISTANT:"
                )

        return instruction, answer

    def __preprocess_as_hf__(
        self,
        image: torch.Tensor,
        inst: str,
        answer: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Preprocess data for HuggingFace model input.

        Args:
            image: fMRI tensor.
            inst: Instruction text.
            answer: Answer text.

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels.
        """
        inputs = {
            'pixel_values': {},
            'input_ids': {},
            'attention_mask': {},
            'labels': {}
        }

        # Set image
        modality_key = 'rsfMRI' if 'fmri' in self.modality.lower() else self.modality
        inputs['pixel_values'][modality_key] = image

        # Tokenize text
        if self.tokenizer is not None:
            token_dict = tokenize_conversation(
                inst, answer, self.tokenizer, self.max_seq_length
            )
            inputs['input_ids'][modality_key] = token_dict['input_ids']
            inputs['attention_mask'][modality_key] = token_dict['attention_mask']
            inputs['labels'][modality_key] = token_dict['labels']

        return inputs

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Args:
            index: Sample index.

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels.
        """
        (_, subject_id, subject_path, start_frame,
         sample_duration, num_frames, conversations, metadata) = self.data[index]

        # Load and process fMRI
        image = self.load_and_process_fmri(
            subject_path, start_frame, sample_duration, num_frames
        )

        # Process text
        inst, answer = self.process_text(conversations, metadata)

        # Format for HuggingFace
        inputs = self.__preprocess_as_hf__(image, inst, answer)

        # Add metadata
        inputs['subject_id'] = subject_id
        inputs['metadata'] = metadata

        return inputs


class RawfMRIDataset(BasefMRIDataset):
    """
    Base class for fMRI datasets that return raw output format.

    Instead of HuggingFace formatted output, returns dict with:
        - fmri_sequence
        - subject_name
        - target
        - TR
        - sex
        - study_name
    """

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample by index in raw format.

        Args:
            index: Sample index.

        Returns:
            Dict with fmri_sequence, subject_name, target, TR, sex, study_name.
        """
        (_, subject_id, subject_path, start_frame,
         sample_duration, num_frames, conversations, metadata) = self.data[index]

        # Load and process fMRI
        y = self.load_and_process_fmri(
            subject_path, start_frame, sample_duration, num_frames
        )

        # Extract target from conversations
        target = None
        if conversations:
            for conv in conversations:
                if conv.get('from') in ['gpt', 'assistant']:
                    target = conv.get('value', '')
                    break

        return {
            'fmri_sequence': y,
            'subject_name': subject_id,
            'target': target,
            'TR': start_frame,
            'sex': metadata.get('sex', None),
            'study_name': self.get_study_name()
        }

    @abstractmethod
    def get_study_name(self) -> str:
        """Get the study name for this dataset."""
        raise NotImplementedError("Subclass must implement get_study_name()")
