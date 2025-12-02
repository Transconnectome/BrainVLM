#!/usr/bin/env python3
"""
Sex-Based sMRI Comparison Dataset Creator
==========================================

This script creates balanced train/validation/test splits from the ABCD dataset
for sex-based comparison learning tasks.

Dataset: ABCD Study (Adolescent Brain Cognitive Development)
Task: Sex classification via structural MRI comparison
Strategy: Balanced stratified sampling (50% male, 50% female per split)

Author: BrainVLM Team
Date: 2025-11-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuration
METADATA_PATH = "ABCD_phenotype_total.csv"
OUTPUT_DIR = Path("sex_comparison_splits")
RANDOM_SEED = 42

# Dataset sizes (balanced 50M/50F per split)
TRAIN_SIZE = 100  # 50 males + 50 females
VAL_SIZE = 100    # 50 males + 50 females
TEST_SIZE = 100   # 50 males + 50 females

# Image path template
IMAGE_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256"
IMAGE_TEMPLATE = "{subject_id}.nii.gz"


def load_metadata(path):
    """Load ABCD phenotype metadata."""
    print(f"Loading metadata from: {path}")
    df = pd.read_csv(path)
    print(f"Total subjects: {len(df)}")

    # Extract relevant columns
    df = df[['subjectkey', 'sex']].copy()
    df.columns = ['subject_id', 'sex']

    # Remove any missing values
    df = df.dropna()

    # Sex encoding: 1=male, 2=female
    print(f"\nSex distribution:")
    print(f"  Males (sex=1): {(df['sex'] == 1).sum()}")
    print(f"  Females (sex=2): {(df['sex'] == 2).sum()}")

    return df


def stratified_sample(df, n_per_sex, random_state):
    """
    Sample exactly n_per_sex from each sex category.

    Args:
        df: DataFrame with 'subject_id' and 'sex' columns
        n_per_sex: Number of subjects to sample per sex
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with balanced sample
    """
    males = df[df['sex'] == 1].sample(n=n_per_sex, random_state=random_state)
    females = df[df['sex'] == 2].sample(n=n_per_sex, random_state=random_state)

    # Combine and shuffle
    sampled = pd.concat([males, females], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sampled


def create_balanced_splits(df, train_size, val_size, test_size, random_seed):
    """
    Create balanced train/val/test splits with equal sex distribution.

    Args:
        df: Full dataset DataFrame
        train_size: Total training samples (will be split 50/50 by sex)
        val_size: Total validation samples (will be split 50/50 by sex)
        test_size: Total test samples (will be split 50/50 by sex)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with train/val/test DataFrames
    """
    # Each split needs equal males and females
    n_train_per_sex = train_size // 2
    n_val_per_sex = val_size // 2
    n_test_per_sex = test_size // 2

    print(f"\nSampling strategy:")
    print(f"  Train: {n_train_per_sex} males + {n_train_per_sex} females = {train_size} total")
    print(f"  Val:   {n_val_per_sex} males + {n_val_per_sex} females = {val_size} total")
    print(f"  Test:  {n_test_per_sex} males + {n_test_per_sex} females = {test_size} total")

    # Separate by sex
    males = df[df['sex'] == 1].copy()
    females = df[df['sex'] == 2].copy()

    # Sample males
    male_train = males.sample(n=n_train_per_sex, random_state=random_seed)
    males_remaining = males.drop(male_train.index)

    male_val = males_remaining.sample(n=n_val_per_sex, random_state=random_seed)
    males_remaining = males_remaining.drop(male_val.index)

    male_test = males_remaining.sample(n=n_test_per_sex, random_state=random_seed)

    # Sample females
    female_train = females.sample(n=n_train_per_sex, random_state=random_seed)
    females_remaining = females.drop(female_train.index)

    female_val = females_remaining.sample(n=n_val_per_sex, random_state=random_seed)
    females_remaining = females_remaining.drop(female_val.index)

    female_test = females_remaining.sample(n=n_test_per_sex, random_state=random_seed)

    # Combine and shuffle each split
    train_df = pd.concat([male_train, female_train], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_df['split'] = 'train'

    val_df = pd.concat([male_val, female_val], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df['split'] = 'validation'

    test_df = pd.concat([male_test, female_test], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df['split'] = 'test'

    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }


def verify_balance(splits):
    """Verify sex balance in all splits."""
    print("\n" + "="*60)
    print("SPLIT VERIFICATION")
    print("="*60)

    for split_name, df in splits.items():
        males = (df['sex'] == 1).sum()
        females = (df['sex'] == 2).sum()
        total = len(df)

        print(f"\n{split_name.upper()}:")
        print(f"  Total: {total}")
        print(f"  Males (sex=1): {males} ({males/total*100:.1f}%)")
        print(f"  Females (sex=2): {females} ({females/total*100:.1f}%)")
        print(f"  Balance check: {'✓ PASS' if males == females else '✗ FAIL'}")


def save_splits(splits, output_dir):
    """Save splits to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*60)
    print("SAVING SPLITS")
    print("="*60)

    for split_name, df in splits.items():
        output_path = output_dir / f"{split_name}_subjects.csv"
        df.to_csv(output_path, index=False)
        print(f"  {split_name}: {output_path} ({len(df)} subjects)")

    # Also save combined metadata
    combined = pd.concat(splits.values(), ignore_index=True)
    combined_path = output_dir / "all_subjects_metadata.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  combined: {combined_path} ({len(combined)} subjects)")

    return output_dir


def create_pairing_metadata(splits, output_dir):
    """
    Create pairing metadata for comparison-based learning.

    For each subject, we'll create pairs with:
    - 1 reference subject of same sex
    - 1 reference subject of different sex

    This enables both within-sex and between-sex comparisons.
    """
    print(f"\n" + "="*60)
    print("CREATING PAIRING METADATA")
    print("="*60)

    for split_name, df in splits.items():
        pairs = []

        # Separate by sex
        males = df[df['sex'] == 1]['subject_id'].tolist()
        females = df[df['sex'] == 2]['subject_id'].tolist()

        for idx, row in df.iterrows():
            subject_id = row['subject_id']
            sex = row['sex']

            # Create same-sex comparison
            same_sex_pool = males if sex == 1 else females
            same_sex_pool = [s for s in same_sex_pool if s != subject_id]
            if same_sex_pool:
                same_sex_ref = np.random.choice(same_sex_pool)
                pairs.append({
                    'subject_id': subject_id,
                    'sex': sex,
                    'reference_id': same_sex_ref,
                    'reference_sex': sex,
                    'comparison_type': 'same_sex',
                    'split': split_name
                })

            # Create different-sex comparison
            diff_sex_pool = females if sex == 1 else males
            if diff_sex_pool:
                diff_sex_ref = np.random.choice(diff_sex_pool)
                pairs.append({
                    'subject_id': subject_id,
                    'sex': sex,
                    'reference_id': diff_sex_ref,
                    'reference_sex': 2 if sex == 1 else 1,
                    'comparison_type': 'different_sex',
                    'split': split_name
                })

        # Save pairing metadata
        pairs_df = pd.DataFrame(pairs)
        pairs_path = output_dir / f"{split_name}_pairs.csv"
        pairs_df.to_csv(pairs_path, index=False)
        print(f"  {split_name}: {pairs_path} ({len(pairs_df)} pairs)")

    print("\nPairing strategy:")
    print("  - Each subject has 1 same-sex comparison")
    print("  - Each subject has 1 different-sex comparison")
    print("  - This enables both within-sex and between-sex learning")


def main():
    """Main execution function."""
    print("="*60)
    print("SEX-BASED sMRI COMPARISON DATASET CREATOR")
    print("="*60)

    # Load metadata
    df = load_metadata(METADATA_PATH)

    # Create balanced splits
    splits = create_balanced_splits(
        df,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED
    )

    # Verify balance
    verify_balance(splits)

    # Save splits
    output_dir = save_splits(splits, OUTPUT_DIR)

    # Create pairing metadata
    create_pairing_metadata(splits, output_dir)

    # Final summary
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Total subjects sampled: {TRAIN_SIZE + VAL_SIZE + TEST_SIZE}")
    print(f"  Train: {TRAIN_SIZE} (50M/50F)")
    print(f"  Validation: {VAL_SIZE} (50M/50F)")
    print(f"  Test: {TEST_SIZE} (50M/50F)")
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Image format: {IMAGE_TEMPLATE}")
    print("\nNext step: Run generate_sex_comparison_conversations.py")
    print("="*60)


if __name__ == "__main__":
    main()
