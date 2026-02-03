"""
Memory safety utilities for preventing out-of-memory errors during training.

Provides OOM detection, prevention, and recovery mechanisms.
"""

import torch
import logging
import traceback
from typing import Optional, Callable, Any, Dict
from contextlib import contextmanager
import psutil

from memory_utils import MemoryPredictor, MemoryStats

logger = logging.getLogger(__name__)


class OOMGuardian:
    """
    Prevents and handles out-of-memory errors during training.

    Implements pre-flight memory checks, emergency cleanup, and batch size reduction.
    """

    def __init__(self,
                 device: str = "cuda:0",
                 memory_margin_percent: float = 0.10,
                 max_reduction_steps: int = 3,
                 verbose: bool = True):
        """
        Initialize OOMGuardian.

        Args:
            device: GPU device to monitor
            memory_margin_percent: Safety margin as fraction of total GPU memory (0-1)
            max_reduction_steps: Maximum number of batch size reductions
            verbose: Log actions
        """
        self.device = device
        self.memory_margin_percent = memory_margin_percent
        self.max_reduction_steps = max_reduction_steps
        self.verbose = verbose

        self.memory_predictor = MemoryPredictor(device=device, verbose=verbose)
        self.reduction_steps = 0
        self.oom_count = 0

    def preflight_check(self, predicted_memory_gb: float) -> bool:
        """
        Check if predicted memory fits within safety margins.

        Args:
            predicted_memory_gb: Predicted memory requirement in GB

        Returns:
            True if safe to proceed, False otherwise
        """
        stats = self.memory_predictor.get_memory_stats()

        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            safe_memory = total_gpu_memory * (1 - self.memory_margin_percent)
        else:
            safe_memory = float('inf')

        # Current + predicted should not exceed safe memory
        current_memory = stats.allocated_mb / 1024  # Convert MB to GB
        total_required = current_memory + predicted_memory_gb

        if total_required > safe_memory:
            if self.verbose:
                logger.warning(f"Preflight check failed: {total_required:.2f}GB > {safe_memory:.2f}GB "
                             f"(margin: {self.memory_margin_percent:.0%})")
            return False

        if self.verbose:
            logger.info(f"Preflight check passed: {total_required:.2f}GB <= {safe_memory:.2f}GB")
        return True

    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        if self.verbose:
            logger.warning("Executing emergency memory cleanup...")

        # Clear cache
        self.memory_predictor.clear_cache()

        # Force garbage collection
        import gc
        gc.collect()

        # Clear unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stats = self.memory_predictor.get_memory_stats()
        if self.verbose:
            logger.info(f"After cleanup: {stats.allocated_mb:.1f}MB allocated")

    @contextmanager
    def protected_forward(self, batch_size: int = None, task_type: str = None):
        """
        Context manager for protected forward passes.

        Monitors memory during forward and handles OOM gracefully.

        Args:
            batch_size: Optional batch size for prediction
            task_type: Optional task type for memory prediction

        Yields:
            Memory statistics before forward pass
        """
        if batch_size is not None and task_type is not None:
            predicted_memory = self.memory_predictor.predict_sample_memory(
                task_type, num_images=batch_size, is_training=True
            )

            if not self.preflight_check(predicted_memory):
                self.emergency_cleanup()
                if not self.preflight_check(predicted_memory):
                    raise RuntimeError(
                        f"Insufficient memory even after cleanup. "
                        f"Predicted: {predicted_memory:.2f}GB"
                    )

        # Record memory before forward
        stats_before = self.memory_predictor.get_memory_stats()

        try:
            yield stats_before
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.oom_count += 1
                logger.error(f"OOM Error (occurrence #{self.oom_count}): {str(e)}")
                self.emergency_cleanup()
                raise

    def reduce_batch_size(self, current_batch_size: int) -> int:
        """
        Reduce batch size to recover from OOM.

        Args:
            current_batch_size: Current batch size

        Returns:
            Reduced batch size (at least 1)
        """
        if self.reduction_steps >= self.max_reduction_steps:
            raise RuntimeError(
                f"Cannot reduce batch size further (reached max {self.max_reduction_steps} reductions)"
            )

        new_batch_size = max(1, current_batch_size // 2)
        self.reduction_steps += 1

        if self.verbose:
            logger.warning(f"Reducing batch size: {current_batch_size} → {new_batch_size} "
                         f"(reduction #{self.reduction_steps}/{self.max_reduction_steps})")

        return new_batch_size

    def estimate_safe_batch_size(self, task_type: str = 'T1') -> int:
        """
        Estimate maximum safe batch size for a task type.

        Args:
            task_type: Task type ('T1', 'T2', or 'T3')

        Returns:
            Estimated maximum batch size
        """
        if not torch.cuda.is_available():
            return 1

        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9  # GB
        safe_memory = total_memory * (1 - self.memory_margin_percent)

        # Binary search for maximum batch size
        low, high = 1, 256

        while low < high:
            mid = (low + high + 1) // 2
            memory_required = self.memory_predictor.predict_sample_memory(
                task_type, num_images=mid, is_training=True
            )

            if memory_required <= safe_memory:
                low = mid
            else:
                high = mid - 1

        if self.verbose:
            max_memory = self.memory_predictor.predict_sample_memory(
                task_type, num_images=low, is_training=True
            )
            logger.info(f"Safe batch size for {task_type}: {low} "
                       f"(uses {max_memory:.2f}GB of {safe_memory:.2f}GB)")

        return low

    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory status."""
        stats = self.memory_predictor.get_memory_stats()
        return {
            'allocated_mb': stats.allocated_mb,
            'reserved_mb': stats.reserved_mb,
            'percent_used': stats.percent_used,
            'available_mb': stats.available_mb,
        }

    def reset(self):
        """Reset guardian state (e.g., for new epoch)."""
        self.reduction_steps = 0


class MemorySafetyCallback:
    """
    Callback for HuggingFace Trainer that monitors memory safety.

    Can be integrated into training to automatically handle OOM scenarios.
    """

    def __init__(self,
                 device: str = "cuda:0",
                 memory_margin_percent: float = 0.10,
                 warning_threshold: float = 0.85,
                 log_interval: int = 100):
        """
        Initialize safety callback.

        Args:
            device: GPU device
            memory_margin_percent: Safety margin
            warning_threshold: Percentage at which to warn about memory (0-1)
            log_interval: Log memory stats every N steps
        """
        self.guardian = OOMGuardian(device=device,
                                   memory_margin_percent=memory_margin_percent,
                                   verbose=True)
        self.warning_threshold = warning_threshold
        self.log_interval = log_interval
        self.step_count = 0

    def on_step_end(self, args=None, state=None, control=None, **kwargs):
        """Called at end of each training step."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            status = self.guardian.get_memory_status()

            if status['percent_used'] > self.warning_threshold:
                logger.warning(f"High memory usage: {status['percent_used']:.1f}% "
                             f"({status['allocated_mb']:.0f}MB)")

            logger.info(f"Step {self.step_count}: Memory {status['percent_used']:.1f}% "
                       f"({status['allocated_mb']:.0f}MB used, "
                       f"{status['available_mb']:.0f}MB available)")

    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):
        """Called at end of each epoch."""
        logger.info(f"Epoch end memory status: {self.guardian.get_memory_status()}")
        self.guardian.reset()


class GradientCheckpointManager:
    """Manages gradient checkpointing for memory efficiency."""

    @staticmethod
    def enable_gradient_checkpointing(model):
        """
        Enable gradient checkpointing for supported models.

        Args:
            model: PyTorch model with gradient checkpointing support
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")

    @staticmethod
    def disable_gradient_checkpointing(model):
        """Disable gradient checkpointing."""
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled")


@contextmanager
def memory_safe_forward(predicted_memory_gb: float,
                       device: str = "cuda:0",
                       on_oom: Optional[Callable] = None):
    """
    Context manager for memory-safe forward passes.

    Args:
        predicted_memory_gb: Predicted memory requirement
        device: GPU device
        on_oom: Callback function if OOM occurs

    Yields:
        Memory statistics

    Raises:
        RuntimeError: If OOM occurs and cannot be recovered
    """
    guardian = OOMGuardian(device=device, verbose=True)

    with guardian.protected_forward():
        try:
            yield guardian.get_memory_status()
        except RuntimeError as e:
            if on_oom:
                on_oom(e)
            raise


def estimate_model_memory(model: torch.nn.Module,
                         batch_size: int = 1,
                         device: str = "cuda:0") -> float:
    """
    Estimate model memory usage.

    Args:
        model: PyTorch model
        batch_size: Batch size for estimation
        device: Device to estimate for

    Returns:
        Estimated memory in GB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size
    model_memory_gb = total_size / 1e9

    logger.info(f"Model memory: {model_memory_gb:.2f}GB "
               f"(params: {param_size/1e9:.2f}GB, buffers: {buffer_size/1e9:.2f}GB)")

    return model_memory_gb


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    guardian = OOMGuardian(verbose=True)

    # Check safe batch size
    print("\n=== Safe Batch Sizes ===")
    for task_type in ['T1', 'T2', 'T3']:
        safe_size = guardian.estimate_safe_batch_size(task_type)
        print(f"{task_type}: {safe_size}")

    # Get memory status
    print("\n=== Memory Status ===")
    status = guardian.get_memory_status()
    for key, value in status.items():
        print(f"{key}: {value:.1f}")
