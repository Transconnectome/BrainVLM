"""
Generate JSON files for Sex Comparison Task (Multi-turn Conversation)

Task: Given a reference image with known sex, predict query image's sex through comparison
Format: 2-turn conversation
- Turn 1: User provides reference image + sex label -> Assistant acknowledges
- Turn 2: User provides query image + asks comparison -> Assistant predicts sex
"""

import os
import json
import pandas as pd
import glob
import numpy as np
from pathlib import Path
import random


def load_subjects_and_images(meta_path, img_dir, subject_id_col, sex_col, study_sample='ABCD'):
    """Load metadata and available images"""

    # Load metadata
    meta = pd.read_csv(meta_path)
    meta = meta[[subject_id_col, sex_col]].dropna()

    # Load available images
    image_files = glob.glob(os.path.join(img_dir, '*.nii.gz'))
    image_dict = {}

    # Determine suffix length based on study sample
    suffix_len = -7  # Remove '.nii.gz'

    for img_path in image_files:
        filename = os.path.basename(img_path)
        subject_id = filename[:suffix_len]
        image_dict[subject_id] = img_path

    # Filter subjects with both metadata and images
    meta = meta[meta[subject_id_col].isin(image_dict.keys())].reset_index(drop=True)

    # Remap sex values to 0/1 if needed
    unique_sex_values = meta[sex_col].unique()
    if set(unique_sex_values).issubset({1, 2}):
        meta[sex_col] = meta[sex_col] - 1

    return meta, image_dict


def split_subjects(meta, subject_id_col, sex_col, train_ratio=0.7, val_ratio=0.15, seed=1234):
    """
    Split subjects into train/val/test with COMPLETE SEPARATION

    Each split is further divided into query and reference pools (50/50)
    to ensure no subject appears as both query and reference.

    Returns:
        Dictionary with keys:
        - train_query, train_ref
        - val_query, val_ref
        - test_query, test_ref
    """

    random.seed(seed)
    np.random.seed(seed)

    # Separate by sex
    males = meta[meta[sex_col] == 0][subject_id_col].values.tolist()
    females = meta[meta[sex_col] == 1][subject_id_col].values.tolist()

    random.shuffle(males)
    random.shuffle(females)

    # Split males into train/val/test
    n_males = len(males)
    n_train_males = int(n_males * train_ratio)
    n_val_males = int(n_males * val_ratio)

    train_males = males[:n_train_males]
    val_males = males[n_train_males:n_train_males+n_val_males]
    test_males = males[n_train_males+n_val_males:]

    # Split females into train/val/test
    n_females = len(females)
    n_train_females = int(n_females * train_ratio)
    n_val_females = int(n_females * val_ratio)

    train_females = females[:n_train_females]
    val_females = females[n_train_females:n_train_females+n_val_females]
    test_females = females[n_train_females+n_val_females:]

    # Further split each set into query and reference (50/50)
    def split_query_ref(males_list, females_list):
        """Split into query and reference pools"""
        # Males
        n_m = len(males_list)
        query_males = males_list[:n_m//2]
        ref_males = males_list[n_m//2:]

        # Females
        n_f = len(females_list)
        query_females = females_list[:n_f//2]
        ref_females = females_list[n_f//2:]

        query = query_males + query_females
        ref = ref_males + ref_females

        return query, ref

    train_query, train_ref = split_query_ref(train_males, train_females)
    val_query, val_ref = split_query_ref(val_males, val_females)
    test_query, test_ref = split_query_ref(test_males, test_females)

    # Print summary
    print(f"Total subjects: {len(meta)}")
    print(f"  Males: {len(males)}, Females: {len(females)}")

    print(f"\nTrain: {len(train_males) + len(train_females)} total")
    print(f"  Query: {len(train_query)} (Males: {len([s for s in train_query if s in males])}, Females: {len([s for s in train_query if s in females])})")
    print(f"  Reference: {len(train_ref)} (Males: {len([s for s in train_ref if s in males])}, Females: {len([s for s in train_ref if s in females])})")

    print(f"\nVal: {len(val_males) + len(val_females)} total")
    print(f"  Query: {len(val_query)} (Males: {len([s for s in val_query if s in males])}, Females: {len([s for s in val_query if s in females])})")
    print(f"  Reference: {len(val_ref)} (Males: {len([s for s in val_ref if s in males])}, Females: {len([s for s in val_ref if s in females])})")

    print(f"\nTest: {len(test_males) + len(test_females)} total")
    print(f"  Query: {len(test_query)} (Males: {len([s for s in test_query if s in males])}, Females: {len([s for s in test_query if s in females])})")
    print(f"  Reference: {len(test_ref)} (Males: {len([s for s in test_ref if s in males])}, Females: {len([s for s in test_ref if s in females])})")

    return {
        'train_query': train_query,
        'train_ref': train_ref,
        'val_query': val_query,
        'val_ref': val_ref,
        'test_query': test_query,
        'test_ref': test_ref
    }


def generate_comparison_tasks(
    query_subjects,
    reference_subjects,
    meta,
    image_dict,
    subject_id_col,
    sex_col,
    num_pairs_per_subject=5,
    same_sex_ratio=0.5,
    seed=1234
):
    """
    Generate comparison tasks

    Args:
        query_subjects: List of subjects to use as queries
        reference_subjects: List of subjects to use as references
        meta: Full metadata DataFrame
        image_dict: Dict mapping subject_id to image path
        num_pairs_per_subject: Number of reference pairs per query
        same_sex_ratio: Ratio of same-sex vs different-sex comparisons
    """

    random.seed(seed)

    # Filter metadata to query subjects
    query_meta = meta[meta[subject_id_col].isin(query_subjects)].reset_index(drop=True)

    # Separate reference subjects by sex
    ref_meta = meta[meta[subject_id_col].isin(reference_subjects)]
    ref_males = ref_meta[ref_meta[sex_col] == 0][subject_id_col].values.tolist()
    ref_females = ref_meta[ref_meta[sex_col] == 1][subject_id_col].values.tolist()

    print(f"\nReference pool: {len(reference_subjects)} subjects")
    print(f"  Males: {len(ref_males)}, Females: {len(ref_females)}")

    all_tasks = []

    for _, row in query_meta.iterrows():
        query_id = row[subject_id_col]
        query_sex = int(row[sex_col])
        query_sex_label = 'male' if query_sex == 0 else 'female'
        query_img_path = image_dict[query_id]

        # Determine how many same-sex vs different-sex pairs
        num_same = int(num_pairs_per_subject * same_sex_ratio)
        num_diff = num_pairs_per_subject - num_same

        # Sample reference subjects (exclude query itself if in reference pool)
        if query_sex == 0:  # Query is male
            same_pool = [s for s in ref_males if s != query_id]
            diff_pool = ref_females
        else:  # Query is female
            same_pool = [s for s in ref_females if s != query_id]
            diff_pool = ref_males

        # Sample same-sex references
        if len(same_pool) >= num_same:
            same_refs = random.sample(same_pool, num_same)
        else:
            same_refs = same_pool
            if len(same_refs) < num_same:
                print(f"Warning: Query {query_id} has only {len(same_refs)} same-sex references (requested {num_same})")

        # Sample different-sex references
        if len(diff_pool) >= num_diff:
            diff_refs = random.sample(diff_pool, num_diff)
        else:
            diff_refs = diff_pool
            if len(diff_refs) < num_diff:
                print(f"Warning: Query {query_id} has only {len(diff_refs)} different-sex references (requested {num_diff})")

        # Create tasks for same-sex comparisons
        for ref_id in same_refs:
            ref_sex = query_sex
            ref_sex_label = query_sex_label
            ref_img_path = image_dict[ref_id]

            task = create_task(
                query_id=query_id,
                query_sex=query_sex,
                query_sex_label=query_sex_label,
                query_img_path=query_img_path,
                ref_id=ref_id,
                ref_sex=ref_sex,
                ref_sex_label=ref_sex_label,
                ref_img_path=ref_img_path,
                comparison_type='same'
            )
            all_tasks.append(task)

        # Create tasks for different-sex comparisons
        for ref_id in diff_refs:
            ref_sex = 1 - query_sex
            ref_sex_label = 'female' if ref_sex == 1 else 'male'
            ref_img_path = image_dict[ref_id]

            task = create_task(
                query_id=query_id,
                query_sex=query_sex,
                query_sex_label=query_sex_label,
                query_img_path=query_img_path,
                ref_id=ref_id,
                ref_sex=ref_sex,
                ref_sex_label=ref_sex_label,
                ref_img_path=ref_img_path,
                comparison_type='different'
            )
            all_tasks.append(task)

    print(f"Generated {len(all_tasks)} comparison tasks")

    return all_tasks


def create_task(query_id, query_sex, query_sex_label, query_img_path,
                ref_id, ref_sex, ref_sex_label, ref_img_path, comparison_type):
    """Create a single comparison task in JSON format"""

    task_id = f"{query_id}_{comparison_type}_sex_comparison"

    # Generate assistant responses based on comparison
    if comparison_type == 'same':
        assistant_reasoning = (
            f"Based on comparison with the reference scan, this appears to be a {query_sex_label} subject. "
            f"Structural similarities include comparable gray matter volumes and white matter distribution patterns "
            f"typical of {query_sex_label} brain anatomy."
        )
    else:
        assistant_reasoning = (
            f"Based on comparison with the reference scan, this appears to be a {query_sex_label} subject. "
            f"Despite being compared with a {ref_sex_label} reference, I observe distinct structural differences "
            f"in gray matter distribution and white matter patterns characteristic of {query_sex_label} brain anatomy."
        )

    task = {
        "task_id": task_id,
        "task_type": "T1",
        "subject_ids": [ref_id, query_id],
        "modalities": ["sMRI", "sMRI"],
        "images": [
            {
                "path": ref_img_path,
                "token": "<image>",
                "modality": "sMRI"
            },
            {
                "path": query_img_path,
                "token": "<image>",
                "modality": "sMRI"
            }
        ],
        "conversations": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is a T1-weighted brain MRI from a {ref_sex_label} participant. This will serve as your reference scan."
                    },
                    {
                        "type": "image",
                        "modality": "sMRI",
                        "image_path": ref_img_path
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Understood. I've analyzed the reference {ref_sex_label} brain scan."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare this brain scan with the reference. What is the likely biological sex of this subject?"
                    },
                    {
                        "type": "image",
                        "modality": "sMRI",
                        "image_path": query_img_path
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_reasoning
                    }
                ]
            }
        ],
        "metadata": {
            "subject_id": query_id,
            "subject_label": query_sex_label,
            "subject_label_numeric": query_sex,
            "reference_id": ref_id,
            "reference_label": ref_sex_label,
            "reference_label_numeric": ref_sex,
            "comparison_type": comparison_type,
            "task": "sex_classification_via_comparison"
        }
    }

    return task


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate sex comparison task JSON with proper train/val/test split (NO DATA LEAKAGE)'
    )
    parser.add_argument('--study_sample', type=str, default='ABCD', choices=['ABCD', 'UKB'],
                        help='Study sample name')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to phenotype CSV file')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing MRI images')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for JSON files')
    parser.add_argument('--output_prefix', type=str, default='ABCD_sex_comparison_tasks',
                        help='Output file prefix')
    parser.add_argument('--subject_id_col', type=str, default=None,
                        help='Subject ID column name (default: subjectkey for ABCD, eid for UKB)')
    parser.add_argument('--sex_col', type=str, default='sex',
                        help='Sex column name')
    parser.add_argument('--num_pairs', type=int, default=5,
                        help='Number of comparison pairs per query subject')
    parser.add_argument('--same_sex_ratio', type=float, default=0.5,
                        help='Ratio of same-sex comparisons')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')

    args = parser.parse_args()

    # Set default subject ID column if not specified
    if args.subject_id_col is None:
        if args.study_sample == 'ABCD':
            args.subject_id_col = 'subjectkey'
        elif args.study_sample == 'UKB':
            args.subject_id_col = 'eid'

    print("=" * 70)
    print("GENERATING SEX COMPARISON TASKS WITH PROPER SPLIT")
    print("=" * 70)
    print(f"Study: {args.study_sample}")
    print(f"Metadata: {args.meta_path}")
    print(f"Images: {args.img_dir}")
    print(f"Output: {args.output_dir}/{args.output_prefix}_*.json")
    print("=" * 70)

    # Load subjects and images
    print("\n Loading subjects and images...")
    meta, image_dict = load_subjects_and_images(
        meta_path=args.meta_path,
        img_dir=args.img_dir,
        subject_id_col=args.subject_id_col,
        sex_col=args.sex_col,
        study_sample=args.study_sample
    )

    # Split subjects into train/val/test with query/reference separation
    print("\nSplitting subjects with COMPLETE SEPARATION...")
    splits = split_subjects(
        meta=meta,
        subject_id_col=args.subject_id_col,
        sex_col=args.sex_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Generate tasks for each split
    os.makedirs(args.output_dir, exist_ok=True)

    # Train: query from train_query, reference from train_ref (NO OVERLAP!)
    print("\nGenerating TRAIN tasks...")
    train_tasks = generate_comparison_tasks(
        query_subjects=splits['train_query'],
        reference_subjects=splits['train_ref'],
        meta=meta,
        image_dict=image_dict,
        subject_id_col=args.subject_id_col,
        sex_col=args.sex_col,
        num_pairs_per_subject=args.num_pairs,
        same_sex_ratio=args.same_sex_ratio,
        seed=args.seed
    )

    train_path = os.path.join(args.output_dir, f"{args.output_prefix}_train.json")
    with open(train_path, 'w') as f:
        json.dump(train_tasks, f, indent=2)
    print(f"✓ Saved: {train_path}")

    # Val: query from val_query, reference from val_ref (NO OVERLAP!)
    print("\nGenerating VAL tasks...")
    val_tasks = generate_comparison_tasks(
        query_subjects=splits['val_query'],
        reference_subjects=splits['val_ref'],
        meta=meta,
        image_dict=image_dict,
        subject_id_col=args.subject_id_col,
        sex_col=args.sex_col,
        num_pairs_per_subject=args.num_pairs,
        same_sex_ratio=args.same_sex_ratio,
        seed=args.seed + 1
    )

    val_path = os.path.join(args.output_dir, f"{args.output_prefix}_val.json")
    with open(val_path, 'w') as f:
        json.dump(val_tasks, f, indent=2)
    print(f"Saved: {val_path}")

    # Test: query from test_query, reference from test_ref (NO OVERLAP!)
    print("\nGenerating TEST tasks...")
    test_tasks = generate_comparison_tasks(
        query_subjects=splits['test_query'],
        reference_subjects=splits['test_ref'],
        meta=meta,
        image_dict=image_dict,
        subject_id_col=args.subject_id_col,
        sex_col=args.sex_col,
        num_pairs_per_subject=args.num_pairs,
        same_sex_ratio=args.same_sex_ratio,
        seed=args.seed + 2
    )

    test_path = os.path.join(args.output_dir, f"{args.output_prefix}_test.json")
    with open(test_path, 'w') as f:
        json.dump(test_tasks, f, indent=2)
    print(f"Saved: {test_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks: {len(val_tasks)}")
    print(f"Test tasks: {len(test_tasks)}")
    print(f"Total tasks: {len(train_tasks) + len(val_tasks) + len(test_tasks)}")
    print("=" * 70)

    # Print sample task
    print("\nSample TRAIN task:")
    print(json.dumps(train_tasks[0], indent=2))


if __name__ == '__main__':
    main()
