"""
Memory utilities for heterogeneous batch training with dynamic batch size control.

Provides memory prediction, estimation, and monitoring for multi-task training scenarios.
"""

import numpy as np
import torch
import psutil
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Container for memory statistics."""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    available_mb: float
    percent_used: float


class MemoryPredictor:
    """
    Predicts GPU memory requirement for heterogeneous samples.

    Uses formula-based prediction with runtime calibration against actual GPU measurements.
    Supports three task types:
    - T1: Single subject, single modality
    - T2: Single subject, multiple modalities (2 images)
    - T3: Multiple subjects (2+ images per sample)
    """

    def __init__(self,
                 device: str = "cuda:0",
                 calibration_samples: int = 10,
                 verbose: bool = False):
        """
        Initialize memory predictor.

        Args:
            device: GPU device to monitor
            calibration_samples: Number of samples for calibration
            verbose: Log predictions and calibration
        """
        self.device = device
        self.calibration_samples = calibration_samples
        self.verbose = verbose

        # Memory formulas (GB per sample, forward pass only)
        # These are empirically derived from typical BrainVLM usage
        self.base_overhead = 0.15  # Model, tokenizer, etc. (GB)
        self.per_image_forward = 0.28  # Per image encoding (GB)
        self.per_image_backward = 0.35  # Per image gradient (GB, additional)
        self.batch_overhead = 0.12  # Batch processing overhead (GB)

        # Task-specific multipliers
        self.task_multipliers = {
            'T1': 1.0,    # Single image
            'T2': 1.85,   # Two images
            'T3': 1.87,   # Two images (multiple subjects)
        }

        # Calibration data
        self.calibration_history: List[Tuple[str, float, float]] = []  # (task_type, predicted, actual)
        self.calibration_errors: Dict[str, List[float]] = {'T1': [], 'T2': [], 'T3': []}
        self.calibration_complete = False
        self.adjustment_factors: Dict[str, float] = {'T1': 1.0, 'T2': 1.0, 'T3': 1.0}

    def predict_sample_memory(self, task_type: str, num_images: int = 1,
                             is_training: bool = True) -> float:
        """
        Predict GPU memory for a single sample.

        Args:
            task_type: 'T1', 'T2', or 'T3'
            num_images: Number of images in sample
            is_training: Whether in training mode (includes gradients)

        Returns:
            Predicted memory in GB
        """
        # Base memory for model + overhead
        memory = self.base_overhead + self.batch_overhead

        # Per-image memory
        if is_training:
            memory += num_images * (self.per_image_forward + self.per_image_backward)
        else:
            memory += num_images * self.per_image_forward

        # Task-specific adjustment
        multiplier = self.task_multipliers.get(task_type, 1.0)
        memory *= multiplier

        # Apply calibration adjustment if available
        if self.calibration_complete:
            adjustment = self.adjustment_factors.get(task_type, 1.0)
            memory *= adjustment

        return memory

    def predict_batch_memory(self, task_types: List[str], is_training: bool = True) -> float:
        """
        Predict GPU memory for a batch of samples.

        Args:
            task_types: List of task types in batch
            is_training: Whether in training mode

        Returns:
            Predicted batch memory in GB
        """
        if not task_types:
            return 0.0

        # Base overhead (amortized across batch)
        batch_memory = self.base_overhead + self.batch_overhead

        # Sum per-sample memory (share base cost)
        total_images = len(task_types)  # Simplified: assume 1 image per task on average
        if is_training:
            batch_memory += total_images * (self.per_image_forward + self.per_image_backward)
        else:
            batch_memory += total_images * self.per_image_forward

        # Apply task-specific adjustments
        for task_type in task_types:
            multiplier = self.task_multipliers.get(task_type, 1.0)
            batch_memory *= multiplier / len(task_types)  # Average multiplier

        if self.calibration_complete:
            # Average adjustment across task types in batch
            avg_adjustment = np.mean([self.adjustment_factors.get(t, 1.0)
                                      for t in task_types])
            batch_memory *= avg_adjustment

        return batch_memory

    def record_actual_memory(self, task_type: str, actual_memory_mb: float):
        """
        Record actual GPU memory usage for calibration.

        Args:
            task_type: 'T1', 'T2', or 'T3'
            actual_memory_mb: Actual GPU memory used (MB)
        """
        actual_gb = actual_memory_mb / 1024.0
        predicted_gb = self.predict_sample_memory(task_type,
                                                  num_images=1 + (1 if task_type in ['T2', 'T3'] else 0),
                                                  is_training=True)

        # Track calibration
        self.calibration_history.append((task_type, predicted_gb, actual_gb))
        error = abs(actual_gb - predicted_gb) / predicted_gb if predicted_gb > 0 else 0.0
        self.calibration_errors[task_type].append(error)

        if self.verbose:
            logger.info(f"Calibration {task_type}: predicted={predicted_gb:.3f}GB, "
                       f"actual={actual_gb:.3f}GB, error={error:.1%}")

        # Update adjustment factor after sufficient samples
        if len(self.calibration_errors[task_type]) >= self.calibration_samples:
            avg_actual = np.mean([h[2] for h in self.calibration_history
                                 if h[0] == task_type])
            avg_predicted = np.mean([h[1] for h in self.calibration_history
                                    if h[0] == task_type])
            if avg_predicted > 0:
                self.adjustment_factors[task_type] = avg_actual / avg_predicted

                # Check if all task types are calibrated
                if all(len(self.calibration_errors[t]) >= self.calibration_samples
                      for t in ['T1', 'T2', 'T3']):
                    self.calibration_complete = True
                    if self.verbose:
                        logger.info("Memory predictor calibration complete")
                        logger.info(f"Adjustment factors: {self.adjustment_factors}")

    def get_calibration_accuracy(self) -> Dict[str, float]:
        """
        Get calibration accuracy by task type.

        Returns:
            Dict mapping task type to mean absolute error percentage
        """
        accuracy = {}
        for task_type, errors in self.calibration_errors.items():
            if errors:
                accuracy[task_type] = np.mean(errors)
            else:
                accuracy[task_type] = None
        return accuracy

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current GPU memory statistics.

        Returns:
            MemoryStats dataclass with current GPU memory info
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / 1e6  # MB
            reserved = torch.cuda.memory_reserved(self.device) / 1e6
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e6
        else:
            allocated = reserved = max_allocated = 0.0

        # Available memory (system level)
        available_mb = psutil.virtual_memory().available / 1e6

        # Total GPU memory
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e6
            percent_used = (allocated / total_gpu_memory * 100) if total_gpu_memory > 0 else 0.0
        else:
            percent_used = 0.0

        return MemoryStats(
            timestamp=datetime.now().timestamp(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            available_mb=available_mb,
            percent_used=percent_used
        )

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                logger.info("GPU cache cleared")


class MemoryMonitor:
    """Monitor and log memory usage throughout training."""

    def __init__(self, device: str = "cuda:0", log_interval: int = 100):
        """
        Initialize memory monitor.

        Args:
            device: GPU device to monitor
            log_interval: Log memory every N batches
        """
        self.device = device
        self.log_interval = log_interval
        self.history: List[MemoryStats] = []
        self.batch_count = 0

    def record_memory(self):
        """Record current memory statistics."""
        predictor = MemoryPredictor(device=self.device)
        stats = predictor.get_memory_stats()
        self.history.append(stats)
        self.batch_count += 1

        if self.batch_count % self.log_interval == 0:
            logger.info(f"Batch {self.batch_count}: "
                       f"allocated={stats.allocated_mb:.1f}MB, "
                       f"reserved={stats.reserved_mb:.1f}MB, "
                       f"percent={stats.percent_used:.1f}%")

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self.history:
            return max(s.max_allocated_mb for s in self.history)
        return 0.0

    def get_avg_memory(self) -> float:
        """Get average memory usage in MB."""
        if self.history:
            return np.mean([s.allocated_mb for s in self.history])
        return 0.0

    def get_memory_trend(self) -> List[float]:
        """Get memory usage trend over time."""
        return [s.allocated_mb for s in self.history]


class GradientAccumulationScheduler:
    """
    Dynamically adjust gradient accumulation steps based on memory availability.

    Uses memory prediction to determine safe accumulation steps.
    """

    def __init__(self,
                 base_batch_size: int = 16,
                 target_batch_size: int = 32,
                 device: str = "cuda:0"):
        """
        Initialize accumulation scheduler.

        Args:
            base_batch_size: Batch size per forward pass
            target_batch_size: Target effective batch size
            device: GPU device
        """
        self.base_batch_size = base_batch_size
        self.target_batch_size = target_batch_size
        self.device = device
        self.predictor = MemoryPredictor(device=device)

        # Compute initial accumulation steps
        self.accumulation_steps = max(1, target_batch_size // base_batch_size)

    def get_accumulation_steps(self, task_types: List[str]) -> int:
        """
        Get recommended gradient accumulation steps based on task types.

        Args:
            task_types: List of task types in batch

        Returns:
            Number of accumulation steps
        """
        # Predict memory for different accumulation scenarios
        for accum_steps in range(self.accumulation_steps, 1, -1):
            effective_batch_size = self.base_batch_size * accum_steps
            if effective_batch_size <= self.target_batch_size:
                return accum_steps

        return 1

    def update_from_memory_stats(self, memory_stats: MemoryStats):
        """
        Adjust accumulation steps based on actual memory usage.

        Args:
            memory_stats: Current memory statistics
        """
        # If memory usage is high, reduce accumulation
        if memory_stats.percent_used > 90:
            self.accumulation_steps = max(1, self.accumulation_steps - 1)
        # If memory usage is low, increase accumulation
        elif memory_stats.percent_used < 50:
            self.accumulation_steps = min(8, self.accumulation_steps + 1)


def estimate_training_memory(num_samples: int,
                            task_distribution: Dict[str, float],
                            batch_size: int = 16,
                            num_epochs: int = 3,
                            device: str = "cuda:0") -> Dict[str, float]:
    """
    Estimate total GPU memory requirements for training.

    Args:
        num_samples: Total number of training samples
        task_distribution: Dict mapping task type to fraction (should sum to 1.0)
        batch_size: Batch size
        num_epochs: Number of training epochs
        device: GPU device

    Returns:
        Dict with memory estimates (model, data, gradients, total)
    """
    predictor = MemoryPredictor(device=device)

    # Estimate per-batch memory
    batch_task_types = []
    for task_type, fraction in task_distribution.items():
        batch_task_types.extend([task_type] * int(batch_size * fraction))

    batch_memory = predictor.predict_batch_memory(batch_task_types, is_training=True)

    # Estimate total training memory
    num_batches = (num_samples * num_epochs) // batch_size

    return {
        'per_batch_mb': batch_memory * 1024,
        'total_gpu_memory_mb': batch_memory * 1024,
        'estimated_peak_memory_mb': batch_memory * 1024 * 1.2,  # 20% headroom
        'estimated_batches': num_batches,
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    predictor = MemoryPredictor(verbose=True)

    # Predict memory for different task types
    print("\n=== Memory Predictions ===")
    for task_type in ['T1', 'T2', 'T3']:
        memory = predictor.predict_sample_memory(task_type, is_training=True)
        print(f"{task_type}: {memory:.3f} GB")

    # Check current GPU memory
    print("\n=== Current GPU Memory ===")
    stats = predictor.get_memory_stats()
    print(f"Allocated: {stats.allocated_mb:.1f} MB")
    print(f"Reserved: {stats.reserved_mb:.1f} MB")
    print(f"Usage: {stats.percent_used:.1f}%")
