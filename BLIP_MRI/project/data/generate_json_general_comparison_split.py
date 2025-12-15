"""
Generate JSON files for Comparison Tasks 

Supports:
- Categorical tasks: sex, diagnosis, etc.
- Numerical tasks: age, BMI, glucose, etc.

This script ensures complete separation:
- Inter-split: Train/Val/Test subjects do NOT overlap
- Intra-split: Within each split, Query and Reference pools do NOT overlap
"""

import os
import json
import pandas as pd
import glob
import numpy as np
from pathlib import Path
import random

def load_subjects_and_images(meta_path, img_dir, subject_id_col, target_col, study_sample='ABCD', max_subjects=None):
    """Load metadata and available images"""

    # Load metadata
    meta = pd.read_csv(meta_path)
    meta = meta[[subject_id_col, target_col]].dropna()

    # Load available images
    image_files = glob.glob(os.path.join(img_dir, '*.nii.gz'))
    image_dict = {}

    suffix_len = -7  # Remove '.nii.gz'

    for img_path in image_files:
        filename = os.path.basename(img_path)
        subject_id = filename[:suffix_len]
        image_dict[subject_id] = img_path

    if subject_id_col == 'subject_id':  # GARD
      # CSV의 subject_id 타입 확인
        if pd.api.types.is_integer_dtype(meta[subject_id_col]):
            # subject_id가 int면 image_dict의 key도 int로 변환
            # _brain 같은 suffix 제거
            image_dict_converted = {}
            for k, v in image_dict.items():
                # 숫자만 추출 (_brain 제거)
                k_clean = k.replace('_brain', '')
                try:
                    image_dict_converted[int(k_clean)] = v
                except ValueError:
                    continue  # 변환 실패하면 스킵
            image_dict = image_dict_converted

    # Filter subjects with both metadata and images
    meta = meta[meta[subject_id_col].isin(image_dict.keys())].reset_index(drop=True)

    # Limit number of subjects if specified
    if max_subjects is not None and len(meta) > max_subjects:
        print(f"Limiting to {max_subjects} subjects (from {len(meta)})")
        # Stratified sampling to maintain class balance
        if pd.api.types.is_numeric_dtype(meta[target_col]) and meta[target_col].nunique() <= 10:
            # Categorical: stratified by class
            samples_per_class = max_subjects // meta[target_col].nunique()
            meta = meta.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), samples_per_class), random_state=1234)
            ).reset_index(drop=True)
        else:
            # Numerical or large categorical: random sample
            meta = meta.sample(n=max_subjects, random_state=1234).reset_index(drop=True)

    # Remap sex values to 0/1 if target_col is 'sex'
    if 'sex' in target_col.lower():
        unique_values = meta[target_col].unique()
        if set(unique_values).issubset({1, 2}):
            print(f"Sex values are 1/2 format. Remapping: 1->0 (male), 2->1 (female)")
            meta[target_col] = meta[target_col] - 1
        elif set(unique_values).issubset({'M', 'F', 'Male', 'Female', 'male', 'female'}):
            print(f"Sex values are string format. Remapping: M/Male/male->0, F/Female/female->1")
            meta[target_col] = meta[target_col].map({
                'M': 0, 'Male': 0, 'male': 0,
                'F': 1, 'Female': 1, 'female': 1
            })
        elif not set(unique_values).issubset({0, 1}):
            print(f"[WARN] Sex values are unexpected format: {unique_values}")
            print(f"        Expected: 0/1, 1/2, or M/F variants")

    return meta, image_dict


def detect_task_type(meta, target_col):
    """
    Automatically detect if task is categorical or numerical

    Returns:
        'categorical' or 'numerical'
    """
    unique_values = meta[target_col].unique()

    # Check if all values are numeric
    if pd.api.types.is_numeric_dtype(meta[target_col]):
        # If small number of unique values (< 10), likely categorical
        if len(unique_values) <= 10:
            return 'categorical'
        else:
            return 'numerical'
    else:
        return 'categorical'


def parse_categorical_mapping(meta, target_col, mapping_str=None):
    """
    Parse categorical mapping from string or auto-detect

    Args:
        meta: DataFrame
        target_col: Target column name
        mapping_str: Optional mapping string like "male=0,female=1" or "1=male,2=female"

    Returns:
        value_to_label: dict mapping numeric values to string labels
        label_to_value: dict mapping string labels to numeric values
    """

    if mapping_str:
        # Parse user-provided mapping
        # Supports: "male=0,female=1" or "0=male,1=female"
        pairs = mapping_str.split(',')
        value_to_label = {}
        label_to_value = {}

        for pair in pairs:
            parts = pair.strip().split('=')
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip()
                # Try to determine which is numeric
                try:
                    num_val = int(key)
                    str_label = val
                except ValueError:
                    num_val = int(val)
                    str_label = key

                value_to_label[num_val] = str_label
                label_to_value[str_label] = num_val
    else:
        # Auto-detect from data
        unique_values = sorted(meta[target_col].unique())

        # Special handling for 'sex' column
        if 'sex' in target_col.lower():
            # Common sex encoding: 0/1 or 1/2
            if set(unique_values) == {0, 1}:
                value_to_label = {0: "male", 1: "female"}
                label_to_value = {"male": 0, "female": 1}
                print("  Detected sex column with 0/1 encoding (0=male, 1=female)")
            elif set(unique_values) == {1, 2}:
                # Will be remapped to 0/1 later
                value_to_label = {1: "male", 2: "female"}
                label_to_value = {"male": 1, "female": 2}
                print("  Detected sex column with 1/2 encoding (1=male, 2=female)")
            else:
                # Fallback for unexpected values
                value_to_label = {val: str(val) for val in unique_values}
                label_to_value = {str(val): val for val in unique_values}
        # Check if string values (use as-is)
        elif not pd.api.types.is_numeric_dtype(meta[target_col]):
            # String categorical values - use original values as labels
            value_to_label = {val: str(val) for val in unique_values}
            label_to_value = {str(val): val for val in unique_values}
            print(f"  Using original string values as labels: {list(unique_values)}")
        # Check if already 0-indexed integers
        elif set(unique_values) == set(range(len(unique_values))):
            # Use generic labels
            value_to_label = {i: f"class_{i}" for i in unique_values}
            label_to_value = {f"class_{i}": i for i in unique_values}
        # Check if 1-indexed
        elif set(unique_values) == set(range(1, len(unique_values) + 1)):
            # Remap to 0-indexed with generic labels
            value_to_label = {i: f"class_{i-1}" for i in unique_values}
            label_to_value = {f"class_{i-1}": i for i in unique_values}
        else:
            # Mixed values, use as-is
            value_to_label = {val: str(val) for val in unique_values}
            label_to_value = {str(val): val for val in unique_values}

    return value_to_label, label_to_value


def remap_categorical_values(meta, target_col, value_to_label):
    """
    Remap categorical values to 0-indexed if needed

    Returns:
        meta: DataFrame with remapped values
        value_to_label: Updated mapping
    """

    # Check if values need remapping
    unique_values = sorted(meta[target_col].unique())

    if set(unique_values) == set(value_to_label.keys()):
        # Already correct, but might need 0-indexing
        if min(unique_values) == 1:
            # 1-indexed → 0-indexed
            print(f"Remapping {target_col} from 1-indexed to 0-indexed:")
            for old_val in unique_values:
                new_val = old_val - 1
                label = value_to_label[old_val]
                print(f"  {old_val} ({label}) → {new_val}")
                meta[target_col] = meta[target_col].replace(old_val, new_val)

            # Update mapping
            new_value_to_label = {old_val - 1: label for old_val, label in value_to_label.items()}
            return meta, new_value_to_label

    return meta, value_to_label


def split_subjects_categorical(meta, subject_id_col, target_col, value_to_label,
                               train_ratio=0.7, val_ratio=0.15, seed=1234):
    """
    Split subjects for categorical tasks (stratified by class) with COMPLETE SEPARATION

    Returns a dictionary with 6 subject pools:
    - train_query, train_ref
    - val_query, val_ref
    - test_query, test_ref

    This ensures:
    1. Inter-split: Train/Val/Test subjects don't overlap
    2. Intra-split: Query and Reference pools don't overlap within each split
    """

    random.seed(seed)
    np.random.seed(seed)

    train_subjects = []
    val_subjects = []
    test_subjects = []

    # First split by class (stratified)
    for value, label in value_to_label.items():
        class_subjects = meta[meta[target_col] == value][subject_id_col].values.tolist()
        random.shuffle(class_subjects)

        n = len(class_subjects)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_subjects.extend(class_subjects[:n_train])
        val_subjects.extend(class_subjects[n_train:n_train+n_val])
        test_subjects.extend(class_subjects[n_train+n_val:])

        print(f"  {label} (value={value}): {n} subjects")
        print(f"    Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")

    # Further split each set into query and reference (50/50)
    def split_query_ref(subjects_list):
        """Split subjects into query and reference pools"""
        random.shuffle(subjects_list)
        n = len(subjects_list)
        query = subjects_list[:n//2]
        ref = subjects_list[n//2:]
        return query, ref

    train_query, train_ref = split_query_ref(train_subjects)
    val_query, val_ref = split_query_ref(val_subjects)
    test_query, test_ref = split_query_ref(test_subjects)

    print(f"\n  Query/Reference split:")
    print(f"    Train: Query={len(train_query)}, Ref={len(train_ref)}")
    print(f"    Val: Query={len(val_query)}, Ref={len(val_ref)}")
    print(f"    Test: Query={len(test_query)}, Ref={len(test_ref)}")

    return {
        'train_query': train_query,
        'train_ref': train_ref,
        'val_query': val_query,
        'val_ref': val_ref,
        'test_query': test_query,
        'test_ref': test_ref
    }


def split_subjects_numerical(meta, subject_id_col, target_col,
                             train_ratio=0.7, val_ratio=0.15, seed=1234):
    """
    Split subjects for numerical tasks (stratified by value bins) with COMPLETE SEPARATION

    Returns a dictionary with 6 subject pools:
    - train_query, train_ref
    - val_query, val_ref
    - test_query, test_ref

    This ensures:
    1. Inter-split: Train/Val/Test subjects don't overlap
    2. Intra-split: Query and Reference pools don't overlap within each split
    """

    random.seed(seed)
    np.random.seed(seed)

    # Bin values into quartiles for stratification
    meta['_bin'] = pd.qcut(meta[target_col], q=4, labels=False, duplicates='drop')

    train_subjects = []
    val_subjects = []
    test_subjects = []

    # First split each bin into train/val/test
    for bin_idx in sorted(meta['_bin'].unique()):
        bin_subjects = meta[meta['_bin'] == bin_idx][subject_id_col].values.tolist()
        random.shuffle(bin_subjects)

        n = len(bin_subjects)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_subjects.extend(bin_subjects[:n_train])
        val_subjects.extend(bin_subjects[n_train:n_train+n_val])
        test_subjects.extend(bin_subjects[n_train+n_val:])

        bin_meta = meta[meta['_bin'] == bin_idx]
        print(f"  Bin {bin_idx} ({bin_meta[target_col].min():.1f}-{bin_meta[target_col].max():.1f}): {n} subjects")
        print(f"    Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")

    meta = meta.drop('_bin', axis=1)

    # Further split each set into query and reference (50/50)
    def split_query_ref(subjects_list):
        """Split subjects into query and reference pools"""
        random.shuffle(subjects_list)
        n = len(subjects_list)
        query = subjects_list[:n//2]
        ref = subjects_list[n//2:]
        return query, ref

    train_query, train_ref = split_query_ref(train_subjects)
    val_query, val_ref = split_query_ref(val_subjects)
    test_query, test_ref = split_query_ref(test_subjects)

    print(f"\n  Query/Reference split:")
    print(f"    Train: Query={len(train_query)}, Ref={len(train_ref)}")
    print(f"    Val: Query={len(val_query)}, Ref={len(val_ref)}")
    print(f"    Test: Query={len(test_query)}, Ref={len(test_ref)}")

    return {
        'train_query': train_query,
        'train_ref': train_ref,
        'val_query': val_query,
        'val_ref': val_ref,
        'test_query': test_query,
        'test_ref': test_ref
    }


def generate_comparison_tasks_categorical(
    query_subjects,
    reference_subjects,
    meta,
    image_dict,
    subject_id_col,
    target_col,
    value_to_label,
    num_pairs_per_subject=5,
    same_class_ratio=0.5,
    seed=1234
):
    """Generate comparison tasks for categorical target"""

    random.seed(seed)

    query_meta = meta[meta[subject_id_col].isin(query_subjects)].reset_index(drop=True)

    # Group reference subjects by class
    ref_meta = meta[meta[subject_id_col].isin(reference_subjects)]
    ref_by_class = {}
    for value in value_to_label.keys():
        ref_by_class[value] = ref_meta[ref_meta[target_col] == value][subject_id_col].values.tolist()

    print(f"\nReference pool: {len(reference_subjects)} subjects")
    for value, label in value_to_label.items():
        print(f"  {label}: {len(ref_by_class[value])}")

    all_tasks = []

    for _, row in query_meta.iterrows():
        query_id = row[subject_id_col]
        query_value = int(row[target_col])
        query_label = value_to_label[query_value]
        query_img_path = image_dict[query_id]

        # Determine same-class vs different-class pairs
        num_same = int(num_pairs_per_subject * same_class_ratio)
        num_diff = num_pairs_per_subject - num_same

        # Sample same-class references
        same_pool = [s for s in ref_by_class[query_value] if s != query_id]
        if len(same_pool) >= num_same:
            same_refs = random.sample(same_pool, num_same)
        else:
            same_refs = same_pool

        # Sample different-class references
        diff_pool = []
        for value in value_to_label.keys():
            if value != query_value:
                diff_pool.extend(ref_by_class[value])

        if len(diff_pool) >= num_diff:
            diff_refs = random.sample(diff_pool, num_diff)
        else:
            diff_refs = diff_pool

        # Create tasks for same-class
        for ref_id in same_refs:
            ref_value = query_value
            ref_label = query_label
            ref_img_path = image_dict[ref_id]

            task = create_task_categorical(
                query_id, query_value, query_label, query_img_path,
                ref_id, ref_value, ref_label, ref_img_path,
                comparison_type='same',
                target_name=target_col
            )
            all_tasks.append(task)

        # Create tasks for different-class
        for ref_id in diff_refs:
            ref_value = int(meta[meta[subject_id_col] == ref_id][target_col].values[0])
            ref_label = value_to_label[ref_value]
            ref_img_path = image_dict[ref_id]

            task = create_task_categorical(
                query_id, query_value, query_label, query_img_path,
                ref_id, ref_value, ref_label, ref_img_path,
                comparison_type='different',
                target_name=target_col
            )
            all_tasks.append(task)

    print(f"Generated {len(all_tasks)} comparison tasks")
    return all_tasks

def generate_comparison_tasks_categorical(
        query_subjects,
        reference_subjects,
        meta,
        image_dict,
        subject_id_col,
        target_col,
        value_to_label,
        num_pairs_per_subject=5,
        same_class_ratio=0.5,
        seed=1234
    ):
        """Generate comparison tasks for categorical target"""

        random.seed(seed)

        query_meta = meta[meta[subject_id_col].isin(query_subjects)].reset_index(drop=True)

        # Group reference subjects by class
        ref_meta = meta[meta[subject_id_col].isin(reference_subjects)]
        ref_by_class = {}
        for value in value_to_label.keys():
            ref_by_class[value] = ref_meta[ref_meta[target_col] == value][subject_id_col].values.tolist()

        print(f"\nReference pool: {len(reference_subjects)} subjects")
        for value, label in value_to_label.items():
            print(f"  {label}: {len(ref_by_class[value])}")

        all_tasks = []

        for _, row in query_meta.iterrows():
            query_id = row[subject_id_col]
            # Convert to Python native types
            if isinstance(query_id, (np.integer, np.int64)):
                query_id = int(query_id)

            query_value = int(row[target_col])
            query_label = value_to_label[query_value]
            query_img_path = image_dict[query_id]

            # Determine same-class vs different-class pairs
            num_same = int(num_pairs_per_subject * same_class_ratio)
            num_diff = num_pairs_per_subject - num_same

            # Sample same-class references
            same_pool = [s for s in ref_by_class[query_value] if s != query_id]
            if len(same_pool) >= num_same:
                same_refs = random.sample(same_pool, num_same)
            else:
                same_refs = same_pool

            # Sample different-class references
            diff_pool = []
            for value in value_to_label.keys():
                if value != query_value:
                    diff_pool.extend(ref_by_class[value])

            if len(diff_pool) >= num_diff:
                diff_refs = random.sample(diff_pool, num_diff)
            else:
                diff_refs = diff_pool

            # Create tasks for same-class
            for ref_id in same_refs:
                # Convert to Python native types
                if isinstance(ref_id, (np.integer, np.int64)):
                    ref_id = int(ref_id)

                ref_value = query_value
                ref_label = query_label
                ref_img_path = image_dict[ref_id]

                task = create_task_categorical(
                    query_id, query_value, query_label, query_img_path,
                    ref_id, ref_value, ref_label, ref_img_path,
                    comparison_type='same',
                    target_name=target_col
                )
                all_tasks.append(task)

            # Create tasks for different-class
            for ref_id in diff_refs:
                # Convert to Python native types
                if isinstance(ref_id, (np.integer, np.int64)):
                    ref_id = int(ref_id)

                ref_value = int(meta[meta[subject_id_col] == ref_id][target_col].values[0])
                ref_label = value_to_label[ref_value]
                ref_img_path = image_dict[ref_id]

                task = create_task_categorical(
                    query_id, query_value, query_label, query_img_path,
                    ref_id, ref_value, ref_label, ref_img_path,
                    comparison_type='different',
                    target_name=target_col
                )
                all_tasks.append(task)

        print(f"Generated {len(all_tasks)} comparison tasks")
        return all_tasks
      
def generate_comparison_tasks_numerical(
        query_subjects,
        reference_subjects,
        meta,
        image_dict,
        subject_id_col,
        target_col,
        num_pairs_per_subject=6,
        seed=1234
    ):
        """Generate comparison tasks for numerical target (e.g., age)"""

        random.seed(seed)

        query_meta = meta[meta[subject_id_col].isin(query_subjects)].reset_index(drop=True)
        ref_meta = meta[meta[subject_id_col].isin(reference_subjects)]

        print(f"\nReference pool: {len(reference_subjects)} subjects")
        print(f"  {target_col} range: {ref_meta[target_col].min():.1f} - {ref_meta[target_col].max():.1f}")

        all_tasks = []

        for _, row in query_meta.iterrows():
            query_id = row[subject_id_col]
            # Convert to Python native types
            if isinstance(query_id, (np.integer, np.int64)):
                query_id = int(query_id)

            query_value = float(row[target_col])
            query_img_path = image_dict[query_id]

            # Sample references across different value ranges
            ref_pool = [s for s in reference_subjects if s != query_id]

            if len(ref_pool) >= num_pairs_per_subject:
                selected_refs = random.sample(ref_pool, num_pairs_per_subject)
            else:
                selected_refs = ref_pool

            for ref_id in selected_refs:
                # Convert to Python native types
                if isinstance(ref_id, (np.integer, np.int64)):
                    ref_id = int(ref_id)

                ref_value = float(meta[meta[subject_id_col] == ref_id][target_col].values[0])
                ref_img_path = image_dict[ref_id]

                task = create_task_numerical(
                    query_id, query_value, query_img_path,
                    ref_id, ref_value, ref_img_path,
                    target_name=target_col
                )
                all_tasks.append(task)

        print(f"Generated {len(all_tasks)} comparison tasks")
        return all_tasks


def create_task_categorical(query_id, query_value, query_label, query_img_path,
                            ref_id, ref_value, ref_label, ref_img_path,
                            comparison_type, target_name):
    """Create task for categorical target"""

    task_id = f"{query_id}_{comparison_type}_{target_name}_comparison"
    assistant_reasoning = (
            f"Based on comparison with the reference scan, this appears to be {query_label}."
        )

    task = {
        "task_id": task_id,
        "task_type": "T1",
        "subject_ids": [ref_id, query_id],
        "modalities": ["sMRI", "sMRI"],
        "images": [
            {"path": ref_img_path, "token": "<image>", "modality": "sMRI"},
            {"path": query_img_path, "token": "<image>", "modality": "sMRI"}
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is a T1-weighted brain MRI from a {ref_label} participant. This will serve as your reference scan."},
                    {"type": "image", "modality": "sMRI", "image_path": ref_img_path}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Understood. I've analyzed the reference {ref_label} brain scan."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Compare this brain scan with the reference. What is the {target_name}?"},
                    {"type": "image", "modality": "sMRI", "image_path": query_img_path}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_reasoning}]
            }
        ],
        "metadata": {
            "subject_id": query_id,
            "subject_label": query_label,
            "subject_label_numeric": int(query_value),
            "reference_id": ref_id,
            "reference_label": ref_label,
            "reference_label_numeric": int(ref_value),
            "comparison_type": comparison_type,
            "task": f"{target_name}_classification_via_comparison",
            "target_name": target_name,
            "task_type": "categorical"
        }
    }

    return task


def create_task_numerical(query_id, query_value, query_img_path,
                          ref_id, ref_value, ref_img_path, target_name):
    """Create task for numerical target"""

    task_id = f"{query_id}_{target_name}_comparison"

    assistant_reasoning = (
        f"I estimate this subject's {target_name} to be approximately {query_value:.1f}."
    )

    task = {
        "task_id": task_id,
        "task_type": "T1",
        "subject_ids": [ref_id, query_id],
        "modalities": ["sMRI", "sMRI"],
        "images": [
            {"path": ref_img_path, "token": "<image>", "modality": "sMRI"},
            {"path": query_img_path, "token": "<image>", "modality": "sMRI"}
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is a T1-weighted brain MRI from a participant with {target_name}: {ref_value:.1f}. This will serve as your reference scan."},
                    {"type": "image", "modality": "sMRI", "image_path": ref_img_path}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Understood. I've analyzed the reference brain scan ({target_name}: {ref_value:.1f})."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Compare this brain scan with the reference. What is the {target_name}?"},
                    {"type": "image", "modality": "sMRI", "image_path": query_img_path}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_reasoning}]
            }
        ],
        "metadata": {
            "subject_id": query_id,
            "subject_value": float(query_value),
            "reference_id": ref_id,
            "reference_value": float(ref_value),
            "task": f"{target_name}_regression_via_comparison",
            "target_name": target_name,
            "task_type": "numerical"
        }
    }

    return task


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate comparison task JSON with proper split (GENERAL: categorical or numerical)'
    )
    parser.add_argument('--study_sample', type=str, default='ABCD', choices=['ABCD', 'UKB', 'GARD'],
                        help='Study sample name')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to phenotype CSV file')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing MRI images')
    parser.add_argument('--target_col', type=str, required=True,
                        help='Target column name (e.g., sex, age, BMI)')
    parser.add_argument('--task_type', type=str, default='auto', choices=['auto', 'categorical', 'numerical'],
                        help='Task type (auto-detect if not specified)')
    parser.add_argument('--categorical_mapping', type=str, default=None,
                        help='Categorical mapping (e.g., "male=0,female=1" or "1=male,2=female")')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for JSON files')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Output file prefix (default: {study_sample}_{target_col}_comparison_tasks)')
    parser.add_argument('--subject_id_col', type=str, default=None,
                        help='Subject ID column name (default: subjectkey for ABCD, eid for UKB)')
    parser.add_argument('--num_pairs', type=int, default=5,
                        help='Number of comparison pairs per query subject')
    parser.add_argument('--same_class_ratio', type=float, default=0.5,
                        help='Ratio of same-class comparisons (categorical only)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--max_subjects', type=int, default=None, help='Maximum number of subjects to use (for quick testing)')

    args = parser.parse_args()

    # Set defaults
    # if args.subject_id_col is None:
    #     args.subject_id_col = 'subjectkey' if args.study_sample == 'ABCD' else 'eid'
    if args.subject_id_col is None:
        if args.study_sample == 'ABCD':
            args.subject_id_col = 'subjectkey'
        elif args.study_sample == 'UKB':
            args.subject_id_col = 'eid'
        elif args.study_sample == 'GARD':
            args.subject_id_col = 'subject_id'  # ← 이 부분 추가
            
        else:
            print("[WARN] Unknown study_sample. Please specify--subject_id_col manually.")
            args.subject_id_col = 'subject_id'

    if args.output_prefix is None:
        args.output_prefix = f"{args.study_sample}_{args.target_col}_comparison_tasks"

    print("=" * 70)
    print(f"GENERATING {args.target_col.upper()} COMPARISON TASKS WITH PROPER SPLIT")
    print("=" * 70)
    print(f"Study: {args.study_sample}")
    print(f"Target: {args.target_col}")
    print(f"Metadata: {args.meta_path}")
    print(f"Images: {args.img_dir}")
    print("=" * 70)

    # Load data
    print("\n[Step 1] Loading subjects and images...")
    meta, image_dict = load_subjects_and_images(
        args.meta_path, args.img_dir, args.subject_id_col, args.target_col, args.study_sample, args.max_subjects
    )
    print(f"Loaded {len(meta)} subjects with images")

    # Detect task type
    if args.task_type == 'auto':
        task_type = detect_task_type(meta, args.target_col)
        print(f"\n[Step 2] Auto-detected task type: {task_type}")
    else:
        task_type = args.task_type
        print(f"\n[Step 2] Task type: {task_type}")

    # Split subjects
    if task_type == 'categorical':
        value_to_label, label_to_value = parse_categorical_mapping(meta, args.target_col, args.categorical_mapping)
        print(f"\nCategorical mapping:")
        for value, label in sorted(value_to_label.items()):
            print(f"  {value} → {label}")

        meta, value_to_label = remap_categorical_values(meta, args.target_col, value_to_label)

        print("\n[Step 3] Splitting subjects (stratified by class) with COMPLETE SEPARATION...")
        splits = split_subjects_categorical(
            meta, args.subject_id_col, args.target_col, value_to_label,
            args.train_ratio, args.val_ratio, args.seed
        )
    else:
        print(f"\n{args.target_col} range: {meta[args.target_col].min():.1f} - {meta[args.target_col].max():.1f}")
        print("\n[Step 3] Splitting subjects (stratified by value bins) with COMPLETE SEPARATION...")
        splits = split_subjects_numerical(
            meta, args.subject_id_col, args.target_col,
            args.train_ratio, args.val_ratio, args.seed
        )

    print(f"\nTotal subjects:")
    print(f"  Train: {len(splits['train_query']) + len(splits['train_ref'])}")
    print(f"  Val: {len(splits['val_query']) + len(splits['val_ref'])}")
    print(f"  Test: {len(splits['test_query']) + len(splits['test_ref'])}")

    # Generate tasks
    os.makedirs(args.output_dir, exist_ok=True)

    if task_type == 'categorical':
        # Train: query from train_query, reference from train_ref (COMPLETE SEPARATION)
        print("\nGenerating TRAIN tasks (categorical)...")
        train_tasks = generate_comparison_tasks_categorical(
            splits['train_query'], splits['train_ref'], meta, image_dict,
            args.subject_id_col, args.target_col, value_to_label,
            args.num_pairs, args.same_class_ratio, args.seed
        )
        # Val: query from val_query, reference from val_ref (COMPLETE SEPARATION)
        print("\nGenerating VAL tasks (categorical)...")
        val_tasks = generate_comparison_tasks_categorical(
            splits['val_query'], splits['val_ref'], meta, image_dict,
            args.subject_id_col, args.target_col, value_to_label,
            args.num_pairs, args.same_class_ratio, args.seed + 1
        )
        # Test: query from test_query, reference from test_ref (COMPLETE SEPARATION)
        print("\nGenerating TEST tasks (categorical)...")
        test_tasks = generate_comparison_tasks_categorical(
            splits['test_query'], splits['test_ref'], meta, image_dict,
            args.subject_id_col, args.target_col, value_to_label,
            args.num_pairs, args.same_class_ratio, args.seed + 2
        )
    else:
        # Train: query from train_query, reference from train_ref (COMPLETE SEPARATION)
        print("\nGenerating TRAIN tasks (numerical)...")
        train_tasks = generate_comparison_tasks_numerical(
            splits['train_query'], splits['train_ref'], meta, image_dict,
            args.subject_id_col, args.target_col, args.num_pairs, args.seed
        )
        # Val: query from val_query, reference from val_ref (COMPLETE SEPARATION)
        print("\nGenerating VAL tasks (numerical)...")
        val_tasks = generate_comparison_tasks_numerical(
            splits['val_query'], splits['val_ref'], meta, image_dict,
            args.subject_id_col, args.target_col, args.num_pairs, args.seed + 1
        )
        # Test: query from test_query, reference from test_ref (COMPLETE SEPARATION)
        print("\nGenerating TEST tasks (numerical)...")
        test_tasks = generate_comparison_tasks_numerical(
            splits['test_query'], splits['test_ref'], meta, image_dict,
            args.subject_id_col, args.target_col, args.num_pairs, args.seed + 2
        )

    # Save
    train_path = os.path.join(args.output_dir, f"{args.output_prefix}_train.json")
    val_path = os.path.join(args.output_dir, f"{args.output_prefix}_val.json")
    test_path = os.path.join(args.output_dir, f"{args.output_prefix}_test.json")

    with open(train_path, 'w') as f:
        json.dump(train_tasks, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_tasks, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_tasks, f, indent=2)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Task type: {task_type}")
    print(f"Target: {args.target_col}")
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks: {len(val_tasks)}")
    print(f"Test tasks: {len(test_tasks)}")
    print(f"Total: {len(train_tasks) + len(val_tasks) + len(test_tasks)}")
    print("=" * 70)

    print("\nSample task:")
    print(json.dumps(train_tasks[0], indent=2))


if __name__ == '__main__':
    main()
