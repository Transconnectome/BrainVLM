"""
Monitoring utilities for dynamic batching training.

Tracks metrics, memory usage, and performance across tasks and batches.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""
    batch_idx: int
    timestamp: float
    task_types: List[str] = field(default_factory=list)
    batch_size: int = 0
    loss: float = 0.0
    per_task_loss: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    throughput_samples_per_sec: float = 0.0
    gradient_norm: float = 0.0
    is_oom_recovery: bool = False


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    total_batches: int
    total_samples: int
    avg_loss: float = 0.0
    per_task_avg_loss: Dict[str, float] = field(default_factory=dict)
    per_task_samples: Dict[str, int] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    avg_throughput_samples_per_sec: float = 0.0
    oom_recoveries: int = 0
    epoch_time_seconds: float = 0.0


class DynamicBatchingMonitor:
    """
    Monitors heterogeneous dynamic batching training.

    Tracks batch metrics, task-specific performance, memory usage, and throughput.
    """

    def __init__(self,
                 output_dir: str = "./training_metrics",
                 save_interval: int = 100,
                 plot_interval: int = 500,
                 device: str = "cuda:0"):
        """
        Initialize monitor.

        Args:
            output_dir: Directory for saving metrics
            save_interval: Save metrics every N batches
            plot_interval: Generate plots every N batches
            device: GPU device
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.device = device

        # Metrics storage
        self.batch_metrics: List[BatchMetrics] = []
        self.epoch_metrics: List[EpochMetrics] = []

        # Summary statistics
        self.task_stats = {
            'T1': {'losses': [], 'samples': 0},
            'T2': {'losses': [], 'samples': 0},
            'T3': {'losses': [], 'samples': 0},
        }

        # Timing
        self.epoch_start_time: Optional[float] = None
        self.batch_start_times: Dict[int, float] = {}

        logger.info(f"DynamicBatchingMonitor initialized, output: {self.output_dir}")

    def start_epoch(self):
        """Mark start of epoch."""
        import time
        self.epoch_start_time = time.time()

    def record_batch(self, batch_metrics: BatchMetrics):
        """
        Record metrics for a batch.

        Args:
            batch_metrics: BatchMetrics dataclass
        """
        self.batch_metrics.append(batch_metrics)

        # Update task statistics
        for task_type in batch_metrics.task_types:
            if task_type in self.task_stats:
                self.task_stats[task_type]['losses'].append(batch_metrics.per_task_loss.get(task_type, 0.0))
                self.task_stats[task_type]['samples'] += 1

        # Save periodically
        if len(self.batch_metrics) % self.save_interval == 0:
            self.save_batch_metrics()

        # Plot periodically
        if len(self.batch_metrics) % self.plot_interval == 0:
            self.plot_training_curves()

    def end_epoch(self, total_samples: int, total_batches: int) -> EpochMetrics:
        """
        Mark end of epoch and compute epoch metrics.

        Args:
            total_samples: Total samples in epoch
            total_batches: Total batches in epoch

        Returns:
            EpochMetrics for this epoch
        """
        import time

        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        epoch_num = len(self.epoch_metrics)

        # Compute epoch statistics
        epoch_losses = [m.loss for m in self.batch_metrics[-total_batches:]]
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        per_task_avg_loss = {}
        per_task_samples = {}
        for task_type, stats in self.task_stats.items():
            if stats['losses']:
                per_task_avg_loss[task_type] = np.mean(stats['losses'])
            per_task_samples[task_type] = stats['samples']

        # Memory statistics
        recent_batches = self.batch_metrics[-min(1000, len(self.batch_metrics)):]
        memory_allocated = [m.memory_allocated_mb for m in recent_batches]
        peak_memory = max(memory_allocated) if memory_allocated else 0.0
        avg_memory = np.mean(memory_allocated) if memory_allocated else 0.0

        # Throughput
        throughputs = [m.throughput_samples_per_sec for m in recent_batches]
        avg_throughput = np.mean(throughputs) if throughputs else 0.0

        # OOM recoveries
        oom_count = sum(1 for m in self.batch_metrics[-total_batches:] if m.is_oom_recovery)

        epoch_metrics = EpochMetrics(
            epoch=epoch_num,
            total_batches=total_batches,
            total_samples=total_samples,
            avg_loss=avg_loss,
            per_task_avg_loss=per_task_avg_loss,
            per_task_samples=per_task_samples,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            avg_throughput_samples_per_sec=avg_throughput,
            oom_recoveries=oom_count,
            epoch_time_seconds=epoch_time
        )

        self.epoch_metrics.append(epoch_metrics)

        # Log epoch summary
        logger.info(f"Epoch {epoch_num} complete: loss={avg_loss:.4f}, "
                   f"memory={peak_memory:.0f}MB, throughput={avg_throughput:.1f} samples/s, "
                   f"OOM recoveries={oom_count}")

        return epoch_metrics

    def save_batch_metrics(self):
        """Save batch metrics to JSON file."""
        output_file = self.output_dir / "batch_metrics.json"

        metrics_dict = {
            'batches': [asdict(m) for m in self.batch_metrics],
            'summary': {
                'total_batches': len(self.batch_metrics),
                'timestamp': datetime.now().isoformat(),
            }
        }

        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        logger.info(f"Saved {len(self.batch_metrics)} batch metrics to {output_file}")

    def save_epoch_metrics(self):
        """Save epoch metrics to JSON file."""
        output_file = self.output_dir / "epoch_metrics.json"

        metrics_dict = {
            'epochs': [asdict(m) for m in self.epoch_metrics],
            'summary': {
                'total_epochs': len(self.epoch_metrics),
                'timestamp': datetime.now().isoformat(),
            }
        }

        with open(output_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        logger.info(f"Saved {len(self.epoch_metrics)} epoch metrics to {output_file}")

    def plot_training_curves(self):
        """Generate training curves plots."""
        if not self.batch_metrics:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics (Batch {len(self.batch_metrics)})', fontsize=16)

        # Loss curve
        batch_indices = [m.batch_idx for m in self.batch_metrics]
        losses = [m.loss for m in self.batch_metrics]

        axes[0, 0].plot(batch_indices, losses, alpha=0.7)
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Over Batches')
        axes[0, 0].grid(True, alpha=0.3)

        # Per-task loss
        for task_type in ['T1', 'T2', 'T3']:
            task_losses = [m.per_task_loss.get(task_type, 0.0)
                          for m in self.batch_metrics if task_type in m.task_types]
            if task_losses:
                axes[0, 1].plot(range(len(task_losses)), task_losses,
                               label=task_type, alpha=0.7)

        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Per-Task Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Memory usage
        memory = [m.memory_allocated_mb for m in self.batch_metrics]
        axes[1, 0].plot(batch_indices, memory, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].set_title('GPU Memory Usage')
        axes[1, 0].grid(True, alpha=0.3)

        # Throughput
        throughput = [m.throughput_samples_per_sec for m in self.batch_metrics]
        axes[1, 1].plot(batch_indices, throughput, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Samples/sec')
        axes[1, 1].set_title('Training Throughput')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / f"training_curves_batch{len(self.batch_metrics)}.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved plot to {plot_file}")

    def plot_task_comparison(self):
        """Generate task comparison plots."""
        if not self.epoch_metrics:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Task-Specific Performance', fontsize=16)

        epochs = [m.epoch for m in self.epoch_metrics]
        task_types = ['T1', 'T2', 'T3']

        # Loss by task
        for task_type in task_types:
            losses = [m.per_task_avg_loss.get(task_type, 0.0)
                     for m in self.epoch_metrics]
            axes[0].plot(epochs, losses, marker='o', label=task_type)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Average Loss')
        axes[0].set_title('Loss by Task Type')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Samples by task
        for task_type in task_types:
            samples = [m.per_task_samples.get(task_type, 0)
                      for m in self.epoch_metrics]
            axes[1].plot(epochs, samples, marker='o', label=task_type)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Samples Processed')
        axes[1].set_title('Samples by Task Type')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / "task_comparison.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved task comparison plot to {plot_file}")

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training report.

        Returns:
            Dict with training statistics
        """
        if not self.batch_metrics or not self.epoch_metrics:
            return {}

        # Overall statistics
        all_losses = [m.loss for m in self.batch_metrics]
        all_memory = [m.memory_allocated_mb for m in self.batch_metrics]
        all_throughput = [m.throughput_samples_per_sec for m in self.batch_metrics]

        report = {
            'summary': {
                'total_batches': len(self.batch_metrics),
                'total_epochs': len(self.epoch_metrics),
                'training_start_time': self.batch_metrics[0].timestamp if self.batch_metrics else None,
                'training_end_time': self.batch_metrics[-1].timestamp if self.batch_metrics else None,
            },
            'loss_statistics': {
                'min': float(np.min(all_losses)) if all_losses else 0.0,
                'max': float(np.max(all_losses)) if all_losses else 0.0,
                'mean': float(np.mean(all_losses)) if all_losses else 0.0,
                'std': float(np.std(all_losses)) if all_losses else 0.0,
            },
            'memory_statistics': {
                'min_mb': float(np.min(all_memory)) if all_memory else 0.0,
                'max_mb': float(np.max(all_memory)) if all_memory else 0.0,
                'mean_mb': float(np.mean(all_memory)) if all_memory else 0.0,
                'std_mb': float(np.std(all_memory)) if all_memory else 0.0,
            },
            'throughput_statistics': {
                'min_samples_per_sec': float(np.min(all_throughput)) if all_throughput else 0.0,
                'max_samples_per_sec': float(np.max(all_throughput)) if all_throughput else 0.0,
                'mean_samples_per_sec': float(np.mean(all_throughput)) if all_throughput else 0.0,
            },
            'per_task_statistics': self.task_stats,
            'epoch_metrics': [asdict(m) for m in self.epoch_metrics],
        }

        # Save report
        report_file = self.output_dir / "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved training report to {report_file}")

        return report

    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics of training so far."""
        if not self.batch_metrics:
            return {}

        losses = [m.loss for m in self.batch_metrics]
        memory = [m.memory_allocated_mb for m in self.batch_metrics]
        throughput = [m.throughput_samples_per_sec for m in self.batch_metrics]

        return {
            'avg_loss': float(np.mean(losses)),
            'avg_memory_mb': float(np.mean(memory)),
            'peak_memory_mb': float(np.max(memory)),
            'avg_throughput_samples_per_sec': float(np.mean(throughput)),
            'total_batches': len(self.batch_metrics),
            'total_epochs': len(self.epoch_metrics),
        }


class TrainingCallback:
    """Callback interface for monitoring integration."""

    def on_batch_end(self, batch_metrics: BatchMetrics):
        """Called at end of batch training."""
        pass

    def on_epoch_end(self, epoch_metrics: EpochMetrics):
        """Called at end of epoch."""
        pass

    def on_training_end(self, report: Dict[str, Any]):
        """Called at end of training."""
        pass


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    monitor = DynamicBatchingMonitor(output_dir="./test_metrics")

    # Simulate training
    monitor.start_epoch()
    for batch_idx in range(10):
        batch_metrics = BatchMetrics(
            batch_idx=batch_idx,
            timestamp=0.0,
            task_types=['T1', 'T2', 'T3'],
            batch_size=16,
            loss=0.5 - 0.01 * batch_idx,
            per_task_loss={'T1': 0.4, 'T2': 0.5, 'T3': 0.6},
            memory_allocated_mb=5000 + 100 * np.random.randn(),
            throughput_samples_per_sec=100 + 10 * np.random.randn(),
        )
        monitor.record_batch(batch_metrics)

    epoch_metrics = monitor.end_epoch(total_samples=160, total_batches=10)
    monitor.save_epoch_metrics()

    print(f"Epoch metrics: {asdict(epoch_metrics)}")
