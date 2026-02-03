"""
Comprehensive integration tests for dynamic batching system.

Tests all components: memory prediction, batching, safety, trainer, and monitoring.
"""

import unittest
import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from memory_utils import MemoryPredictor, MemoryMonitor, estimate_training_memory
from memory_safety import OOMGuardian, GradientCheckpointManager
from dynamic_batching import (
    MemoryAwareBatchSampler, HeterogeneousCollator, HeterogeneousBatch,
    BatchConstructor
)
from dynamic_trainer import (
    EffectiveBatchSizeNormalizer, GradNormBalancer
)
from dynamic_monitoring import DynamicBatchingMonitor, BatchMetrics, EpochMetrics


class TestMemoryPredictor(unittest.TestCase):
    """Test memory prediction system."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = MemoryPredictor(device="cpu", verbose=False)

    def test_single_image_prediction(self):
        """Test memory prediction for single image."""
        memory_t1 = self.predictor.predict_sample_memory('T1', num_images=1, is_training=True)
        self.assertGreater(memory_t1, 0.0)
        self.assertLess(memory_t1, 10.0)  # Should be < 10GB

    def test_multiple_images_prediction(self):
        """Test memory prediction increases with images."""
        memory_1 = self.predictor.predict_sample_memory('T1', num_images=1, is_training=True)
        memory_2 = self.predictor.predict_sample_memory('T1', num_images=2, is_training=True)

        self.assertGreater(memory_2, memory_1)

    def test_task_type_differences(self):
        """Test different task types have different memory requirements."""
        memory_t1 = self.predictor.predict_sample_memory('T1', is_training=True)
        memory_t2 = self.predictor.predict_sample_memory('T2', is_training=True)
        memory_t3 = self.predictor.predict_sample_memory('T3', is_training=True)

        # T2 and T3 should require more than T1
        self.assertLess(memory_t1, memory_t2)
        self.assertLess(memory_t1, memory_t3)

    def test_training_vs_inference(self):
        """Test training requires more memory than inference."""
        memory_train = self.predictor.predict_sample_memory('T1', is_training=True)
        memory_infer = self.predictor.predict_sample_memory('T1', is_training=False)

        self.assertGreater(memory_train, memory_infer)

    def test_batch_memory_prediction(self):
        """Test batch memory prediction."""
        batch_types = ['T1', 'T2', 'T3']
        batch_memory = self.predictor.predict_batch_memory(batch_types, is_training=True)

        self.assertGreater(batch_memory, 0.0)

    def test_calibration(self):
        """Test memory prediction calibration."""
        # Record some calibration data
        for i in range(5):
            self.predictor.record_actual_memory('T1', 1000.0)

        accuracy = self.predictor.get_calibration_accuracy()
        self.assertIn('T1', accuracy)
        self.assertIsNotNone(accuracy['T1'])


class SimpleMockDataset:
    """Simple mock dataset for testing."""

    def __init__(self, num_samples: int = 100, task_distribution: Dict[str, float] = None):
        """Initialize mock dataset."""
        if task_distribution is None:
            task_distribution = {'T1': 0.4, 'T2': 0.3, 'T3': 0.3}

        self.num_samples = num_samples
        self.task_distribution = task_distribution
        self.tasks = ['T1', 'T2', 'T3']

    def __len__(self):
        """Get dataset size."""
        return self.num_samples

    def __getitem__(self, idx):
        """Get sample."""
        task_type = self.get_sample_metadata(idx)['task_type']
        num_images = 1 + (1 if task_type in ['T2', 'T3'] else 0)

        return {
            'task_type': task_type,
            'pixel_values': {
                'T1': [torch.randn(1, 128, 128, 3) for _ in range(num_images)]
            },
            'input_ids': torch.randint(0, 32000, (2048,)),
            'attention_mask': torch.ones(2048),
            'labels': torch.ones(2048),
            'metadata': {'subject_id': f'sub-{idx:04d}'},
            'sample_index': idx,
        }

    def get_sample_metadata(self, idx):
        """Get sample metadata."""
        task_idx = idx % len(self.tasks)
        task_type = self.tasks[task_idx]
        return {'task_type': task_type, 'num_images': 1 + (1 if task_type != 'T1' else 0)}


class TestMemoryAwareBatchSampler(unittest.TestCase):
    """Test memory-aware batch sampler."""

    def setUp(self):
        """Set up test fixtures."""
        self.dataset = SimpleMockDataset(num_samples=100)
        self.sampler = MemoryAwareBatchSampler(
            dataset=self.dataset,
            batch_size=8,
            max_memory_mb=30000,
            device="cpu",
            shuffle=False,
            verbose=False
        )

    def test_sampler_creates_batches(self):
        """Test sampler generates batches."""
        batches = []
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) >= self.sampler.batch_size:
                batches.append(batch)
                batch = []

        if batch:
            batches.append(batch)

        self.assertGreater(len(batches), 0)

    def test_batch_size_respected(self):
        """Test batches respect size limits."""
        for batch in [list(range(i, min(i+8, len(self.dataset))))
                      for i in range(0, len(self.dataset), 8)]:
            if batch:
                memory = self.sampler._estimate_batch_memory(batch)
                self.assertLess(memory, self.sampler.max_memory_gb)

    def test_task_diversity(self):
        """Test batch diversity in task types."""
        batch = list(range(8))
        task_types = [self.dataset.get_sample_metadata(idx)['task_type']
                     for idx in batch]

        # Should have mix of tasks
        unique_tasks = set(task_types)
        self.assertGreater(len(unique_tasks), 1)


class TestHeterogeneousCollator(unittest.TestCase):
    """Test heterogeneous collator."""

    def setUp(self):
        """Set up test fixtures."""
        self.collator = HeterogeneousCollator(max_seq_length=2048)
        self.dataset = SimpleMockDataset(num_samples=10)

    def test_collator_handles_batch(self):
        """Test collator processes batches correctly."""
        samples = [self.dataset[i] for i in range(4)]
        batch = self.collator(samples)

        self.assertIsInstance(batch, HeterogeneousBatch)
        self.assertEqual(len(batch.task_types), 4)
        self.assertEqual(batch.pixel_values.shape[0], 4)

    def test_padding_to_max_images(self):
        """Test collator pads to maximum images in batch."""
        samples = [self.dataset[i] for i in range(4)]
        batch = self.collator(samples)

        # All samples should have same number of image positions
        self.assertEqual(batch.pixel_values.shape[1], max(batch.num_images_per_sample))

    def test_image_mask_created(self):
        """Test image mask properly indicates valid positions."""
        samples = [self.dataset[i] for i in range(4)]
        batch = self.collator(samples)

        # Image mask should match num_images
        for i, num_imgs in enumerate(batch.num_images_per_sample):
            valid_positions = batch.image_mask[i].sum().item()
            self.assertEqual(valid_positions, num_imgs)


class TestOOMGuardian(unittest.TestCase):
    """Test OOM prevention and recovery."""

    def setUp(self):
        """Set up test fixtures."""
        self.guardian = OOMGuardian(device="cpu", verbose=False)

    def test_preflight_check(self):
        """Test preflight memory check."""
        # Small memory requirement should pass
        result = self.guardian.preflight_check(0.1)
        self.assertTrue(result)

    def test_estimate_safe_batch_size(self):
        """Test safe batch size estimation."""
        for task_type in ['T1', 'T2', 'T3']:
            safe_size = self.guardian.estimate_safe_batch_size(task_type)
            self.assertGreater(safe_size, 0)

    def test_batch_size_reduction(self):
        """Test batch size reduction on OOM."""
        original_size = 32
        reduced_size = self.guardian.reduce_batch_size(original_size)

        self.assertLess(reduced_size, original_size)
        self.assertGreater(reduced_size, 0)

    def test_memory_status(self):
        """Test memory status reporting."""
        status = self.guardian.get_memory_status()

        self.assertIn('allocated_mb', status)
        self.assertIn('percent_used', status)
        self.assertGreaterEqual(status['percent_used'], 0)
        self.assertLessEqual(status['percent_used'], 100)


class TestEffectiveBatchSizeNormalizer(unittest.TestCase):
    """Test effective batch size normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = EffectiveBatchSizeNormalizer(
            base_batch_size=16,
            target_batch_size=32,
            base_lr=5e-5
        )

    def test_gradient_scale(self):
        """Test gradient scaling for different batch sizes."""
        scale_16 = self.normalizer.get_gradient_scale(16)
        scale_8 = self.normalizer.get_gradient_scale(8)

        self.assertGreater(scale_8, scale_16)

    def test_adjusted_learning_rate(self):
        """Test learning rate adjustment."""
        lr_16 = self.normalizer.get_adjusted_lr(16)
        lr_32 = self.normalizer.get_adjusted_lr(32)

        self.assertLess(lr_16, lr_32)

    def test_loss_scaling(self):
        """Test loss scaling for consistent effective batch size."""
        loss_scale_16 = self.normalizer.get_loss_scale(16)
        loss_scale_32 = self.normalizer.get_loss_scale(32)

        self.assertGreater(loss_scale_16, loss_scale_32)


class TestDynamicBatchingMonitor(unittest.TestCase):
    """Test monitoring system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = DynamicBatchingMonitor(
            output_dir=self.temp_dir,
            save_interval=5,
            plot_interval=10
        )

    def test_record_batch_metrics(self):
        """Test recording batch metrics."""
        batch_metrics = BatchMetrics(
            batch_idx=0,
            timestamp=0.0,
            task_types=['T1', 'T2'],
            batch_size=16,
            loss=0.5,
            per_task_loss={'T1': 0.4, 'T2': 0.6},
            memory_allocated_mb=5000,
            throughput_samples_per_sec=100
        )

        self.monitor.record_batch(batch_metrics)
        self.assertEqual(len(self.monitor.batch_metrics), 1)

    def test_epoch_metrics(self):
        """Test epoch metrics computation."""
        # Record some batches
        for i in range(10):
            batch_metrics = BatchMetrics(
                batch_idx=i,
                timestamp=float(i),
                task_types=['T1', 'T2', 'T3'],
                batch_size=16,
                loss=0.5 - 0.01*i,
                per_task_loss={'T1': 0.4, 'T2': 0.5, 'T3': 0.6},
                memory_allocated_mb=5000 + 100*np.random.randn(),
                throughput_samples_per_sec=100 + 10*np.random.randn()
            )
            self.monitor.record_batch(batch_metrics)

        epoch_metrics = self.monitor.end_epoch(total_samples=160, total_batches=10)

        self.assertIsInstance(epoch_metrics, EpochMetrics)
        self.assertEqual(epoch_metrics.epoch, 0)
        self.assertGreater(epoch_metrics.avg_loss, 0)

    def test_summary_statistics(self):
        """Test summary statistics generation."""
        for i in range(5):
            batch_metrics = BatchMetrics(
                batch_idx=i,
                timestamp=float(i),
                task_types=['T1'],
                batch_size=16,
                loss=0.5,
                per_task_loss={'T1': 0.5},
                memory_allocated_mb=5000,
                throughput_samples_per_sec=100
            )
            self.monitor.record_batch(batch_metrics)

        summary = self.monitor.get_summary_statistics()

        self.assertIn('avg_loss', summary)
        self.assertIn('peak_memory_mb', summary)
        self.assertIn('total_batches', summary)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)


class TestGradNormBalancer(unittest.TestCase):
    """Test GradNorm balancing for multi-task learning."""

    def setUp(self):
        """Set up test fixtures."""
        self.initial_weights = {'T1': 1.0, 'T2': 1.0, 'T3': 1.0}
        self.balancer = GradNormBalancer(self.initial_weights, alpha=1.5)

    def test_initial_weights(self):
        """Test initial loss weights."""
        weights = self.balancer.get_loss_weights()
        self.assertEqual(len(weights), 3)

    def test_weight_updates(self):
        """Test weight update mechanics."""
        # Create simple model
        model = torch.nn.Linear(10, 10)
        loss = model(torch.randn(5, 10)).sum()
        loss.backward()

        # Update weights (would normally use gradient norms)
        original_weights = self.balancer.get_loss_weights()
        self.balancer.update_loss_weights(model)
        new_weights = self.balancer.get_loss_weights()

        # Weights should have changed
        self.assertNotEqual(original_weights['T1'], new_weights['T1'])


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def test_full_pipeline_components(self):
        """Test all components work together."""
        # Create dataset
        dataset = SimpleMockDataset(num_samples=100)

        # Create sampler
        sampler = MemoryAwareBatchSampler(
            dataset=dataset,
            batch_size=8,
            max_memory_mb=30000,
            device="cpu",
            verbose=False
        )

        # Create collator
        collator = HeterogeneousCollator()

        # Get first batch
        batch_indices = list(range(8))
        samples = [dataset[idx] for idx in batch_indices]
        batch = collator(samples)

        # Verify batch structure
        self.assertIsInstance(batch, HeterogeneousBatch)
        self.assertEqual(len(batch.task_types), 8)
        self.assertIsNotNone(batch.pixel_values)
        self.assertIsNotNone(batch.input_ids)

    def test_memory_prediction_to_batching(self):
        """Test memory prediction flows into batch construction."""
        predictor = MemoryPredictor(device="cpu")
        dataset = SimpleMockDataset(num_samples=50)

        sampler = MemoryAwareBatchSampler(
            dataset=dataset,
            batch_size=8,
            device="cpu",
            verbose=False
        )

        # Get batch
        batch_indices = list(range(8))
        batch_memory = sampler._estimate_batch_memory(batch_indices)

        # Should have positive memory estimate
        self.assertGreater(batch_memory, 0.0)

    def test_monitoring_with_training_simulation(self):
        """Test monitoring during simulated training."""
        monitor = DynamicBatchingMonitor(
            output_dir=tempfile.mkdtemp(),
            save_interval=100,
            plot_interval=200
        )

        monitor.start_epoch()

        # Simulate 50 batches
        for batch_idx in range(50):
            batch_metrics = BatchMetrics(
                batch_idx=batch_idx,
                timestamp=float(batch_idx),
                task_types=['T1', 'T2', 'T3'],
                batch_size=16,
                loss=0.5 - 0.005*batch_idx,
                per_task_loss={
                    'T1': 0.4 - 0.003*batch_idx,
                    'T2': 0.5 - 0.005*batch_idx,
                    'T3': 0.6 - 0.007*batch_idx,
                },
                memory_allocated_mb=5000 + 100*np.sin(batch_idx/10),
                throughput_samples_per_sec=100 + 10*np.random.randn()
            )
            monitor.record_batch(batch_metrics)

        epoch_metrics = monitor.end_epoch(total_samples=800, total_batches=50)

        self.assertIsNotNone(epoch_metrics)
        self.assertEqual(epoch_metrics.epoch, 0)
        self.assertEqual(epoch_metrics.total_batches, 50)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryAwareBatchSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestHeterogeneousCollator))
    suite.addTests(loader.loadTestsFromTestCase(TestOOMGuardian))
    suite.addTests(loader.loadTestsFromTestCase(TestEffectiveBatchSizeNormalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicBatchingMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestGradNormBalancer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
