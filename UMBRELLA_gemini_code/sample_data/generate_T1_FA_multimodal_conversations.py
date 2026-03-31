#!/usr/bin/env python3
"""
T1+FA Multimodal JSONL Generator for UMBRELLA

Generates JSONL files for T1_FA late fusion training.
Each sample contains two images: T1 and FA from the same subject.

Usage:
    python generate_T1_FA_multimodal_conversations.py \
        --meta_csv /path/to/metadata.csv \
        --t1_dir /path/to/T1_images \
        --fa_dir /path/to/FA_images \
        --output_dir ./t1_fa_conversations \
        --task sex_classification

Output JSONL format:
{
    "task_id": "sub-001_T1_FA_sex",
    "task_type": "T1_FA_classification",
    "images": [
        {"path": "/path/to/T1.nii.gz", "modality": "T1"},
        {"path": "/path/to/FA.nii.gz", "modality": "FA"}
    ],
    "conversations": [...],
    "metadata": {...}
}
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Prompt templates
PROMPTS = {
    'sex_classification': {
        'question': [
            "Based on the T1-weighted MRI and FA map, what is the biological sex of this subject?",
            "Analyze these brain scans (T1 and FA). Estimate the biological sex.",
            "Using both the structural T1 MRI and diffusion FA map, classify the subject's biological sex.",
        ],
        'answer_male': ["male."],
        'answer_female': ["female."],
    },
    'age_prediction': {
        'question': [
            "Based on the T1-weighted MRI and FA map, estimate the age of this subject.",
            "Analyze these brain scans (T1 and FA). What is the approximate age?",
        ],
        'answer_template': ["{age} years old."],
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate T1+FA Multimodal JSONL')
    parser.add_argument('--meta_csv', type=str, required=True,
                        help='Path to metadata CSV (must have subject_id, sex columns)')
    parser.add_argument('--t1_dir', type=str, required=True,
                        help='Directory containing T1 images')
    parser.add_argument('--fa_dir', type=str, required=True,
                        help='Directory containing FA images')
    parser.add_argument('--output_dir', type=str, default='./t1_fa_conversations',
                        help='Output directory for JSONL files')
    parser.add_argument('--task', type=str, default='sex_classification',
                        choices=['sex_classification', 'age_prediction'],
                        help='Task type')
    parser.add_argument('--t1_suffix', type=str, default='.nii.gz',
                        help='T1 file suffix (e.g., .nii.gz, _T1.nii.gz)')
    parser.add_argument('--fa_suffix', type=str, default='.nii.gz',
                        help='FA file suffix (e.g., .nii.gz, _FA.nii.gz)')
    parser.add_argument('--subject_col', type=str, default='subject_id',
                        help='Column name for subject ID')
    parser.add_argument('--sex_col', type=str, default='sex',
                        help='Column name for sex (1=male, 2=female or M/F)')
    parser.add_argument('--age_col', type=str, default='age',
                        help='Column name for age')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-exists', action='store_true',
                        help='Only include subjects where both T1 and FA images exist')
    return parser.parse_args()


def get_image_paths(subject_id: str, t1_dir: str, fa_dir: str,
                    t1_suffix: str, fa_suffix: str) -> tuple:
    """Get T1 and FA image paths for a subject."""
    t1_path = Path(t1_dir) / f"{subject_id}{t1_suffix}"
    fa_path = Path(fa_dir) / f"{subject_id}{fa_suffix}"
    return str(t1_path), str(fa_path)


def check_images_exist(t1_path: str, fa_path: str) -> bool:
    """Check if both T1 and FA images exist."""
    return Path(t1_path).exists() and Path(fa_path).exists()


def normalize_sex(sex_value) -> Optional[str]:
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


def create_sex_classification_sample(
    subject_id: str,
    sex: str,
    t1_path: str,
    fa_path: str
) -> Dict:
    """Create a T1_FA sex classification sample."""

    question = np.random.choice(PROMPTS['sex_classification']['question'])
    if sex == 'male':
        answer = np.random.choice(PROMPTS['sex_classification']['answer_male'])
    else:
        answer = np.random.choice(PROMPTS['sex_classification']['answer_female'])

    return {
        "task_id": f"{subject_id}_T1_FA_sex",
        "task_type": "T1_FA_classification",
        "subject_ids": [subject_id],
        "modalities": ["T1", "FA"],
        "images": [
            {"path": t1_path, "modality": "T1"},
            {"path": fa_path, "modality": "FA"}
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "image"},  # T1
                    {"type": "image"},  # FA
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": sex,
            "task": "sex_classification",
            "modality_type": "T1_FA"
        }
    }


def create_age_prediction_sample(
    subject_id: str,
    age: float,
    t1_path: str,
    fa_path: str
) -> Dict:
    """Create a T1_FA age prediction sample."""

    question = np.random.choice(PROMPTS['age_prediction']['question'])
    answer_template = np.random.choice(PROMPTS['age_prediction']['answer_template'])
    answer = answer_template.format(age=int(round(age)))

    return {
        "task_id": f"{subject_id}_T1_FA_age",
        "task_type": "T1_FA_regression",
        "subject_ids": [subject_id],
        "modalities": ["T1", "FA"],
        "images": [
            {"path": t1_path, "modality": "T1"},
            {"path": fa_path, "modality": "FA"}
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "image"},  # T1
                    {"type": "image"},  # FA
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ],
        "metadata": {
            "subject_id": subject_id,
            "subject_label": age,
            "task": "age_prediction",
            "modality_type": "T1_FA"
        }
    }


def generate_samples(
    df: pd.DataFrame,
    t1_dir: str,
    fa_dir: str,
    t1_suffix: str,
    fa_suffix: str,
    task: str,
    subject_col: str,
    sex_col: str,
    age_col: str,
    check_exists: bool = False
) -> List[Dict]:
    """Generate samples from dataframe."""

    samples = []
    skipped = 0

    for _, row in df.iterrows():
        subject_id = str(row[subject_col])
        t1_path, fa_path = get_image_paths(subject_id, t1_dir, fa_dir, t1_suffix, fa_suffix)

        # Optionally check if images exist
        if check_exists and not check_images_exist(t1_path, fa_path):
            skipped += 1
            continue

        if task == 'sex_classification':
            sex = normalize_sex(row.get(sex_col))
            if sex is None:
                skipped += 1
                continue
            sample = create_sex_classification_sample(subject_id, sex, t1_path, fa_path)

        elif task == 'age_prediction':
            age = row.get(age_col)
            if pd.isna(age):
                skipped += 1
                continue
            sample = create_age_prediction_sample(subject_id, float(age), t1_path, fa_path)

        samples.append(sample)

    if skipped > 0:
        print(f"  Skipped {skipped} subjects (missing data or images)")

    return samples


def save_jsonl(samples: List[Dict], output_path: Path):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"  Saved {len(samples)} samples to {output_path}")


def main():
    args = parse_args()

    print("="*60)
    print("T1+FA MULTIMODAL JSONL GENERATOR")
    print("="*60)

    # Load metadata
    print(f"\nLoading metadata from: {args.meta_csv}")
    df = pd.read_csv(args.meta_csv)
    print(f"  Total subjects: {len(df)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    print(f"\nSplitting data (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})")

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        random_state=args.seed,
        stratify=df[args.sex_col] if args.task == 'sex_classification' else None
    )

    # Second split: val vs test
    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=args.seed,
        stratify=temp_df[args.sex_col] if args.task == 'sex_classification' else None
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Generate samples for each split
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    for split_name, split_df in splits.items():
        print(f"\nGenerating {split_name} samples...")
        samples = generate_samples(
            split_df,
            args.t1_dir,
            args.fa_dir,
            args.t1_suffix,
            args.fa_suffix,
            args.task,
            args.subject_col,
            args.sex_col,
            args.age_col,
            check_exists=args.check_exists
        )

        # Save JSONL
        output_path = output_dir / f"{split_name}_conversations.jsonl"
        save_jsonl(samples, output_path)

    # Print summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Task: {args.task}")
    print(f"Modality: T1_FA (late fusion)")
    print("\nFiles created:")
    print(f"  - train_conversations.jsonl")
    print(f"  - val_conversations.jsonl")
    print(f"  - test_conversations.jsonl")

    print("\n" + "="*60)
    print("SAMPLE FORMAT")
    print("="*60)
    print("""
{
    "task_id": "sub-001_T1_FA_sex",
    "task_type": "T1_FA_classification",
    "images": [
        {"path": "/path/to/T1.nii.gz", "modality": "T1"},
        {"path": "/path/to/FA.nii.gz", "modality": "FA"}
    ],
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Based on the T1 and FA..."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "male."}]
        }
    ],
    "metadata": {
        "subject_id": "sub-001",
        "subject_label": "male",
        "task": "sex_classification",
        "modality_type": "T1_FA"
    }
}
    """)


if __name__ == "__main__":
    main()
