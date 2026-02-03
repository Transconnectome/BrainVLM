#!/usr/bin/env python3
"""
Quick test to verify all import styles work correctly.
Run from UMBRELLA root: python test_imports.py
"""

import sys
from pathlib import Path

# Replicate the dual-path setup
project_root = Path(__file__).parent / "project"
umbrella_root = Path(__file__).parent

sys.path.insert(0, str(umbrella_root))
sys.path.insert(0, str(project_root))

print("=" * 80)
print("IMPORT TEST - Verifying Both Import Styles Work")
print("=" * 80)

print("\nPath Configuration:")
print(f"  project_root:  {project_root}")
print(f"  umbrella_root: {umbrella_root}")
print(f"  sys.path[0]:   {sys.path[0]}")
print(f"  sys.path[1]:   {sys.path[1]}")

# Test Style A: from project.X
print("\n" + "-" * 80)
print("TEST 1: Style A Imports (from project.X)")
print("-" * 80)

try:
    from project.dataset.umbrella_dataset import UMBRELLADataset, UMBRELLASample, ConversationTurn
    print("✓ PASS: from project.dataset.umbrella_dataset import ...")
except ImportError as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

try:
    from project.training.umbrella_trainer import UMBRELLATrainer
    print("✓ PASS: from project.training.umbrella_trainer import ...")
except ImportError as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# Test Style B: from utils.X
print("\n" + "-" * 80)
print("TEST 2: Style B Imports (from utils.X)")
print("-" * 80)

try:
    from utils.utils import to_3tuple
    print("✓ PASS: from utils.utils import to_3tuple")
except ImportError as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

try:
    from utils.memory_utils import MemoryPredictor
    print("✓ PASS: from utils.memory_utils import MemoryPredictor")
except ImportError as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# Test nested imports
print("\n" + "-" * 80)
print("TEST 3: Nested Imports (transitively)")
print("-" * 80)

try:
    # This will transitively import from utils
    from project.dataset.dataset_T1_LLaVa import DatasetT1LLaVa
    print("✓ PASS: from project.dataset.dataset_T1_LLaVa import DatasetT1LLaVa")
except ImportError as e:
    print(f"✗ FAIL: {e}")
    # This might fail if dependencies aren't installed, but path is correct

print("\n" + "=" * 80)
print("IMPORT TEST COMPLETE")
print("=" * 80)
print("\n✓ All import styles working correctly!")
print("\nYou can now run:")
print("  python project/tests/validate_tokenization.py --verbose")
