#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Separate Modality Datasets Generator (Single-Turn) for UMBRELLA_gemini
======================================================================
Creates T1 only, FA only, and T1+FA JSONL datasets from subjects with BOTH T1 and FA images.
Uses single-turn format matching UMBRELLA_gemini style.
All 3 datasets use the SAME subjects with the SAME train/val/test split for fair comparison.

Usage:
    # First, create file lists on Perlmutter:
    ls /pscratch/sd/h/heehaw/data/1.ABCD/2.3.nii_gz_T1/*.nii.gz | xargs -n1 basename | sed 's/.nii.gz//' > ./data/t1_subjects.txt
    ls /pscratch/sd/h/heehaw/data/1.ABCD/3.1.1.FA_unwarpped_nii_cleaned/*.nii.gz | xargs -n1 basename | sed 's/.nii.gz//' > ./data/fa_subjects.txt

    # Then run:
    python generate_separate_modality_datasets.py \
        --meta_csv /pscratch/sd/h/heehaw/data/1.ABCD/ABCD_subjectkey_sex.csv \
        --t1_subjects ./data/t1_subjects.txt \
        --fa_subjects ./data/fa_subjects.txt \
        --t1_dir /pscratch/sd/h/heehaw/data/1.ABCD/2.3.nii_gz_T1 \
        --fa_dir /pscratch/sd/h/heehaw/data/1.ABCD/3.1.1.FA_unwarpped_nii_cleaned \
        --output_dir ./data/separate_modality_datasets
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def normalize_sex(sex_value):
    """Normalize sex value to 'male' or 'female'."""
    if pd.isna(sex_value):
        return None
    if isinstance(sex_value, (int, float)):
        if sex_value == 1:
            return 'male'
        elif sex_value == 2:
            return 'female'
    elif isinstance(sex_value, str):
        sex_lower = sex_value.lower().strip()
        if sex_lower in ['m', 'male', '1']:
            return 'male'
        elif sex_lower in ['f', 'female', '2']:
            return 'female'
    return None


def load_subject_list(filepath):
    """Load subject IDs from a text file (one per line)."""
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def create_t1_sample(subject_id, t1_path, sex_label):
    """Create T1-only single-turn sample."""
    return {
        "task_id": "{0}_T1_sex_classification".format(subject_id),
        "task_type": "T1",
        "subject_ids": [subject_id],
        "modalities": ["T1"],
        "images": [
            {
                "path": t1_path,
                "token": "<image>",
                "modality": "T1"
            }
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "image", "modality": "T1", "image_path": t1_path},
                    {"type": "text", "text": "Analyze this T1-weighted brain MRI scan. Determine the biological sex of the subject."}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": "{0}.".format(sex_label)}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": sex_label,
            "task": "sex_classification"
        }
    }


def create_fa_sample(subject_id, fa_path, sex_label):
    """Create FA-only single-turn sample."""
    return {
        "task_id": "{0}_FA_sex_classification".format(subject_id),
        "task_type": "FA",
        "subject_ids": [subject_id],
        "modalities": ["FA"],
        "images": [
            {
                "path": fa_path,
                "token": "<image>",
                "modality": "FA"
            }
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "image", "modality": "FA", "image_path": fa_path},
                    {"type": "text", "text": "Analyze this FA (Fractional Anisotropy) brain map. Determine the biological sex of the subject."}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": "{0}.".format(sex_label)}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": sex_label,
            "task": "sex_classification"
        }
    }


def create_t1_fa_sample(subject_id, t1_path, fa_path, sex_label):
    """Create T1+FA single-turn sample."""
    return {
        "task_id": "{0}_T1_FA_sex".format(subject_id),
        "task_type": "T1_FA_classification",
        "subject_ids": [subject_id],
        "modalities": ["T1", "FA"],
        "images": [
            {
                "path": t1_path,
                "modality": "T1"
            },
            {
                "path": fa_path,
                "modality": "FA"
            }
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Using both the structural T1 MRI and diffusion FA map, classify the subject's biological sex."}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": "{0}.".format(sex_label)}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": sex_label,
            "task": "sex_classification",
            "modality_type": "T1_FA"
        }
    }


def save_jsonl(samples, output_path):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print("  Saved {0} samples to {1}".format(len(samples), output_path))


def main():
    parser = argparse.ArgumentParser(description='Generate T1 only and FA only datasets (single-turn)')
    parser.add_argument('--meta_csv', type=str, required=True)
    parser.add_argument('--t1_subjects', type=str, required=True)
    parser.add_argument('--fa_subjects', type=str, required=True)
    parser.add_argument('--t1_dir', type=str, required=True)
    parser.add_argument('--fa_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--subject_col', type=str, default='subjectkey')
    parser.add_argument('--sex_col', type=str, default='sex')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("T1 ONLY & FA ONLY DATASETS GENERATOR (Single-Turn)")
    print("=" * 70)
    
    # Load subject lists
    print("\n[1/5] Loading subject lists...")
    t1_subjects = load_subject_list(args.t1_subjects)
    fa_subjects = load_subject_list(args.fa_subjects)
    common_subjects = t1_subjects & fa_subjects
    
    print("  T1 subjects: {0}".format(len(t1_subjects)))
    print("  FA subjects: {0}".format(len(fa_subjects)))
    print("  Common (T1+FA both): {0} <- Using these for fair comparison".format(len(common_subjects)))
    
    # Load metadata
    print("\n[2/5] Loading metadata from: {0}".format(args.meta_csv))
    df = pd.read_csv(args.meta_csv)
    print("  Total subjects in CSV: {0}".format(len(df)))
    
    # Filter to subjects with BOTH modalities
    df = df[df[args.subject_col].astype(str).isin(common_subjects)]
    print("  Subjects with both T1+FA: {0}".format(len(df)))
    
    # Filter out subjects without valid sex label
    df['sex_normalized'] = df[args.sex_col].apply(normalize_sex)
    df = df[df['sex_normalized'].notna()]
    print("  Subjects with valid sex label: {0}".format(len(df)))
    
    # Create output directories
    output_base = Path(args.output_dir)
    t1_only_dir = output_base / "T1_only"
    fa_only_dir = output_base / "FA_only"
    t1_fa_dir = output_base / "T1_FA"
    
    for d in [t1_only_dir, fa_only_dir, t1_fa_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Split subjects
    print("\n[3/5] Splitting subjects (same split for fair comparison)...")
    train_df, temp_df = train_test_split(
        df, train_size=args.train_ratio, random_state=args.seed,
        stratify=df['sex_normalized']
    )
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio_adjusted, random_state=args.seed,
        stratify=temp_df['sex_normalized']
    )
    
    print("  Train subjects: {0}".format(len(train_df)))
    print("  Val subjects: {0}".format(len(val_df)))
    print("  Test subjects: {0}".format(len(test_df)))
    
    # Generate samples
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    print("\n[4/5] Generating samples...")
    
    for split_name, split_df in splits.items():
        print("\n  --- {0} set ({1} subjects) ---".format(split_name.upper(), len(split_df)))
        
        t1_only_samples = []
        fa_only_samples = []
        t1_fa_samples = []
        
        for _, row in split_df.iterrows():
            subject_id = str(row[args.subject_col])
            sex_label = row['sex_normalized']
            
            t1_path = "{0}/{1}.nii.gz".format(args.t1_dir, subject_id)
            fa_path = "{0}/{1}.nii.gz".format(args.fa_dir, subject_id)
            
            t1_only_samples.append(create_t1_sample(subject_id, t1_path, sex_label))
            fa_only_samples.append(create_fa_sample(subject_id, fa_path, sex_label))
            t1_fa_samples.append(create_t1_fa_sample(subject_id, t1_path, fa_path, sex_label))
        
        # Shuffle
        np.random.seed(args.seed)
        np.random.shuffle(t1_only_samples)
        np.random.seed(args.seed)
        np.random.shuffle(fa_only_samples)
        np.random.seed(args.seed)
        np.random.shuffle(t1_fa_samples)
        
        # Save
        save_jsonl(t1_only_samples, t1_only_dir / "{0}_conversations.jsonl".format(split_name))
        save_jsonl(fa_only_samples, fa_only_dir / "{0}_conversations.jsonl".format(split_name))
        save_jsonl(t1_fa_samples, t1_fa_dir / "{0}_conversations.jsonl".format(split_name))
    
    # Summary
    print("\n" + "=" * 70)
    print("[5/5] COMPLETE!")
    print("=" * 70)
    print("\nOutput directories:")
    print("  T1 only: {0}".format(t1_only_dir))
    print("  FA only: {0}".format(fa_only_dir))
    print("  T1+FA:   {0}".format(t1_fa_dir))
    
    total_subjects = len(df)
    print("\nDataset: {0} subjects (same subjects in all 3 datasets)".format(total_subjects))
    
    sex_counts = df['sex_normalized'].value_counts()
    print("\nSex distribution:")
    for sex, count in sex_counts.items():
        print("  {0}: {1} ({2:.1f}%)".format(sex, count, 100.0 * count / total_subjects))


if __name__ == "__main__":
    main()
