#!/usr/bin/env python3
"""
Any-to-Any Comparison Dataset Creator
=====================================

This script creates stratified train/validation/test splits balancing 
both 'sex' and 'BMI_sds' (using SD bins).

Additionally, it extracts 20 Anchors from the TRAIN split:
 - 5 BMI Bins in total
 - For each Bin, it selects 2 Male and 2 Female subjects.
 - Total = 20 Anchors (10 Male, 10 Female).

All subjects in all splits are paired with all 20 anchors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
METADATA_PATH = "ABCD_phenotype_total_revised.csv"
OUTPUT_DIR = Path("anytoany_comparison_splits_100subjects")
RANDOM_SEED = 42

# Configurable Split Sizes 
TRAIN_SIZE = 100  
VAL_SIZE = 100     
TEST_SIZE = 100    

SPLIT_SIZES = {
    'train': TRAIN_SIZE,
    'validation': VAL_SIZE,
    'test': TEST_SIZE
}

def load_and_clean_data(path):
    p = Path(path)
    if not p.exists():
        p = Path("ABCD_phenotype_total.csv")
    df = pd.read_csv(p)
    if 'subjectkey' in df.columns:
        df = df.rename(columns={'subjectkey': 'subject_id'})
    if 'BMI_sds' not in df.columns or 'sex' not in df.columns:
        raise ValueError("Missing 'BMI_sds' or 'sex' in metadata")
    df = df.dropna(subset=['subject_id', 'BMI_sds', 'sex'])
    df['sex_label'] = df['sex'].apply(lambda x: 'male' if int(x) == 1 else 'female')
    
    # 1. Create BMI bins globally
    mu = df['BMI_sds'].mean()
    sigma = df['BMI_sds'].std()
    bins_edges = [-np.inf, mu - 1.5 * sigma, mu - 0.5 * sigma, mu + 0.5 * sigma, mu + 1.5 * sigma, np.inf]
    labels = ['bin1', 'bin2', 'bin3', 'bin4', 'bin5']
    df['bmi_bin'] = pd.cut(df['BMI_sds'], bins=bins_edges, labels=labels, right=False)
    
    # Stratum for balanced splitting
    df['stratum'] = df['sex_label'] + "_" + df['bmi_bin'].astype(str)
    return df

def sample_stratified_combined(df, size, seed):
    available_counts = df['stratum'].value_counts().to_dict()
    strata = list(available_counts.keys())
    allocated = {s: 0 for s in strata}
    remaining_size = size
    sorted_strata = sorted(strata, key=lambda s: available_counts[s])
    
    for i, s in enumerate(sorted_strata):
        remaining_strata = len(sorted_strata) - i
        fair_share = remaining_size // remaining_strata
        take = min(fair_share, available_counts[s])
        allocated[s] = take
        remaining_size -= take
        
    while remaining_size > 0:
        capable = [s for s in strata if available_counts[s] > allocated[s]]
        if not capable:
            raise ValueError(f"Not enough data to sample {size}.")
        capable.sort(key=lambda s: available_counts[s] - allocated[s], reverse=True)
        for s in capable:
            if remaining_size == 0:
                break
            allocated[s] += 1
            remaining_size -= 1
            
    sampled_dfs = []
    for s in strata:
        if allocated[s] > 0:
            s_df = df[df['stratum'] == s]
            sampled_dfs.append(s_df.sample(n=allocated[s], random_state=seed))
            
    if not sampled_dfs:
        return pd.DataFrame(columns=df.columns), df.copy()
        
    sampled_combined_raw = pd.concat(sampled_dfs)
    sampled_combined = sampled_combined_raw.sample(frac=1, random_state=seed).reset_index(drop=True)
    remaining = df.drop(sampled_combined_raw.index)
        
    return sampled_combined, remaining

def extract_anchors(train_df):
    """Pick 4 anchors per bin (2 male, 2 female)."""
    print("\n[Selecting Anchors from TRAIN split]")
    anchors = []
    
    bins = sorted(train_df['bmi_bin'].unique())
    for b in bins:
        bin_df = train_df[train_df['bmi_bin'] == b]
        males = bin_df[bin_df['sex_label'] == 'male']
        females = bin_df[bin_df['sex_label'] == 'female']
        
        n_males = min(2, len(males))
        n_females = min(2, len(females))
        
        if n_males > 0:
            anchors.append(males.sample(n=n_males, random_state=RANDOM_SEED))
        if n_females > 0:
            anchors.append(females.sample(n=n_females, random_state=RANDOM_SEED))
            
        print(f"  {b}: Selected {n_males} males, {n_females} females.")
        
    anchors_df = pd.concat(anchors).reset_index(drop=True)
    print(f"Total Anchors Selected: {len(anchors_df)}\n")
    return anchors_df

def create_pairs_for_split(split_name, split_df, anchors_df):
    """Pair every subject in the split with every anchor."""
    print(f"Creating pairs for {split_name.upper()} split...")
    pairs = []
    
    anchor_records = anchors_df.to_dict('records')
    
    for _, row in split_df.iterrows():
        subject_id = row['subject_id']
        bmi_sds = row['BMI_sds']
        sex_label = row['sex_label']
        
        for anchor in anchor_records:
            anchor_id = anchor['subject_id']
            if subject_id == anchor_id:
                continue # Skip self pairing
                
            is_same_sex = (sex_label == anchor['sex_label'])
            
            pairs.append({
                'subject_id': subject_id,
                'BMI_sds': bmi_sds,
                'sex_label': sex_label,
                'reference_id': anchor_id,
                'reference_BMI_sds': anchor['BMI_sds'],
                'reference_sex': anchor['sex_label'],
                'comparison_type': 'same-sex' if is_same_sex else 'different-sex',
                'anchor_bin': anchor['bmi_bin'],
                'split': split_name
            })
            
    pairs_df = pd.DataFrame(pairs)
    print(f"  Created {len(pairs_df)} pairs for {split_name}.")
    return pairs_df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DATASET CREATOR: ANY-TO-ANY + COMPARISON")
    print("="*60)
    
    df = load_and_clean_data(METADATA_PATH)
    
    # Sampling Splits
    pool = df.copy()
    test_df, pool = sample_stratified_combined(pool, TEST_SIZE, RANDOM_SEED)
    val_df, pool = sample_stratified_combined(pool, VAL_SIZE, RANDOM_SEED)
    train_df, pool = sample_stratified_combined(pool, TRAIN_SIZE, RANDOM_SEED)
    
    # Extract Anchors strictly from Train
    anchors_df = extract_anchors(train_df)
    anchors_csv = OUTPUT_DIR / "anchor_subjects.csv"
    anchors_df.to_csv(anchors_csv, index=False)
    
    # Generate Pairings
    splits_dfs = {'train': train_df, 'validation': val_df, 'test': test_df}
    for name, s_df in splits_dfs.items():
        pairs_df = create_pairs_for_split(name, s_df, anchors_df)
        pairs_csv = OUTPUT_DIR / f"{name}_pairs.csv"
        pairs_df.to_csv(pairs_csv, index=False)
        
    print("\n" + "="*60)
    print("DATASET PARING COMPLETE")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
