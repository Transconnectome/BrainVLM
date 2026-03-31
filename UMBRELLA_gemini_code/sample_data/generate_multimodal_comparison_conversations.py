#!/usr/bin/env python3
"""
Multi-Turn Multimodal Comparison Conversation Generator
========================================================

Generates JSONL files for T1/FA sex classification with comparison learning.

Structure:
- Turn 1: Reference image → Sex prediction
- Turn 2: Comparison image → Sex prediction (5 comparison cases)

Comparison Cases (configurable via --reference_modality):

T1 as reference (Case 1-5):
1. T1 vs T1, Same sex
2. T1 vs T1, Different sex
3. T1 vs FA, Same subject
4. T1 vs FA, Different subject, Same sex
5. T1 vs FA, Different subject, Different sex

FA as reference (Case 6-10):
6. FA vs FA, Same sex
7. FA vs FA, Different sex
8. FA vs T1, Same subject
9. FA vs T1, Different subject, Same sex
10. FA vs T1, Different subject, Different sex

Options:
--reference_modality T1    → Case 1-5 only
--reference_modality FA    → Case 6-10 only
--reference_modality both  → All 10 cases

Data split: 100 subjects each for train/val/test
Loss: Computed on ALL turn answers

Author: UMBRELLA Team
Date: 2026-03-04
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Image directories
T1_IMG_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256"
FA_IMG_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/3.1.1.FA_unwarpped_nii_cleaned"
META_CSV = "/pscratch/sd/h/heehaw/data/1.ABCD/ABCD_phenotype_total.csv"

# Prompt templates
PROMPTS = {
    'turn1_question': [
        "Analyze this brain scan and estimate the biological sex of the subject.",
    ],
    'turn2_question': [
        "Now compare this second brain scan with the previous one. What is the biological sex of this subject?",
    ],
    'answer_male': ["male."],
    'answer_female': ["female."],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Multimodal Comparison JSONL')
    parser.add_argument('--meta_csv', type=str, default=META_CSV,
                        help='Path to metadata CSV')
    parser.add_argument('--t1_dir', type=str, default=T1_IMG_DIR,
                        help='Directory containing T1 images')
    parser.add_argument('--fa_dir', type=str, default=FA_IMG_DIR,
                        help='Directory containing FA images')
    parser.add_argument('--output_dir', type=str, 
                        default='./multimodal_comparison_conversations',
                        help='Output directory for JSONL files')
    parser.add_argument('--subject_col', type=str, default='subjectkey',
                        help='Column name for subject ID')
    parser.add_argument('--sex_col', type=str, default='sex',
                        help='Column name for sex (M/F or 1/2)')
    parser.add_argument('--n_subjects_per_split', type=int, default=100,
                        help='Number of subjects per split (train/val/test)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check_exists', action='store_true',
                        help='Only include subjects where both T1 and FA exist')
    parser.add_argument('--reference_modality', type=str, default='T1',
                        choices=['T1', 'FA', 'both'],
                        help='Reference modality: T1 (case1-5), FA (case6-10), both (case1-10)')
    return parser.parse_args()


def get_image_path(subject_id: str, modality: str, t1_dir: str, fa_dir: str) -> str:
    """Get image path for a subject and modality."""
    if modality == "T1":
        return f"{t1_dir}/{subject_id}.nii.gz"
    else:  # FA
        return f"{fa_dir}/{subject_id}.nii.gz"


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


def get_answer(sex: str) -> str:
    """Get answer text for sex."""
    if sex == 'male':
        return np.random.choice(PROMPTS['answer_male'])
    else:
        return np.random.choice(PROMPTS['answer_female'])


def create_comparison_sample(
    ref_subject: str,
    ref_sex: str,
    ref_modality: str,
    comp_subject: str,
    comp_sex: str,
    comp_modality: str,
    comparison_case: str,
    t1_dir: str,
    fa_dir: str
) -> Dict:
    """
    Create a multi-turn comparison sample.
    
    Turn 1: Reference image → sex prediction
    Turn 2: Comparison image → sex prediction
    """
    
    ref_path = get_image_path(ref_subject, ref_modality, t1_dir, fa_dir)
    comp_path = get_image_path(comp_subject, comp_modality, t1_dir, fa_dir)
    
    turn1_question = np.random.choice(PROMPTS['turn1_question'])
    turn1_answer = get_answer(ref_sex)
    turn2_question = np.random.choice(PROMPTS['turn2_question'])
    turn2_answer = get_answer(comp_sex)
    
    return {
        "task_id": f"{ref_subject}_{comparison_case}",
        "task_type": "multimodal_comparison",
        "comparison_case": comparison_case,
        "subject_ids": [ref_subject, comp_subject],
        "modalities": [ref_modality, comp_modality],
        "images": [
            {"path": ref_path, "modality": ref_modality},
            {"path": comp_path, "modality": comp_modality}
        ],
        "conversations": [
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": turn1_question}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": turn1_answer}
                ]
            },
            {
                "role": ROLE_USER,
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": turn2_question}
                ]
            },
            {
                "role": ROLE_ASSISTANT,
                "content": [
                    {"type": "text", "text": turn2_answer}
                ]
            }
        ],
        "metadata": {
            "reference_subject": ref_subject,
            "reference_sex": ref_sex,
            "reference_modality": ref_modality,
            "comparison_subject": comp_subject,
            "comparison_sex": comp_sex,
            "comparison_modality": comp_modality,
            "comparison_case": comparison_case,
            "task": "sex_classification_comparison"
        }
    }


def generate_comparison_samples_for_subject(
    ref_subject: str,
    ref_sex: str,
    subjects_by_sex: Dict[str, List[str]],
    t1_dir: str,
    fa_dir: str,
    reference_modality: str = "T1",
    seed: int = None
) -> List[Dict]:
    """
    Generate 5 comparison samples for a reference subject.
    
    T1 as reference (Case 1-5):
    1. T1 vs T1, Same sex
    2. T1 vs T1, Different sex
    3. T1 vs FA, Same subject
    4. T1 vs FA, Different subject, Same sex
    5. T1 vs FA, Different subject, Different sex
    
    FA as reference (Case 6-10):
    6. FA vs FA, Same sex
    7. FA vs FA, Different sex
    8. FA vs T1, Same subject
    9. FA vs T1, Different subject, Same sex
    10. FA vs T1, Different subject, Different sex
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    ref_modality = reference_modality
    other_modality = "FA" if ref_modality == "T1" else "T1"
    
    # Case numbering: T1 reference = 1-5, FA reference = 6-10
    case_offset = 0 if ref_modality == "T1" else 5
    
    opposite_sex = "female" if ref_sex == "male" else "male"
    
    # Get candidate subjects (excluding reference)
    same_sex_subjects = [s for s in subjects_by_sex[ref_sex] if s != ref_subject]
    diff_sex_subjects = subjects_by_sex[opposite_sex]
    
    if len(same_sex_subjects) < 2 or len(diff_sex_subjects) < 2:
        return []
    
    # Case 1/6: Same modality + Same sex
    comp_subject_1 = np.random.choice(same_sex_subjects)
    samples.append(create_comparison_sample(
        ref_subject, ref_sex, ref_modality,
        comp_subject_1, ref_sex, ref_modality,
        f"case{1 + case_offset}_same_mod_same_sex",
        t1_dir, fa_dir
    ))
    
    # Case 2/7: Same modality + Different sex
    comp_subject_2 = np.random.choice(diff_sex_subjects)
    samples.append(create_comparison_sample(
        ref_subject, ref_sex, ref_modality,
        comp_subject_2, opposite_sex, ref_modality,
        f"case{2 + case_offset}_same_mod_diff_sex",
        t1_dir, fa_dir
    ))
    
    # Case 3/8: Different modality + Same subject (same sex obviously)
    samples.append(create_comparison_sample(
        ref_subject, ref_sex, ref_modality,
        ref_subject, ref_sex, other_modality,
        f"case{3 + case_offset}_diff_mod_same_subj",
        t1_dir, fa_dir
    ))
    
    # Case 4/9: Different modality + Different subject + Same sex
    comp_subject_4 = np.random.choice([s for s in same_sex_subjects if s != comp_subject_1])
    samples.append(create_comparison_sample(
        ref_subject, ref_sex, ref_modality,
        comp_subject_4, ref_sex, other_modality,
        f"case{4 + case_offset}_diff_mod_diff_subj_same_sex",
        t1_dir, fa_dir
    ))
    
    # Case 5/10: Different modality + Different subject + Different sex
    comp_subject_5 = np.random.choice([s for s in diff_sex_subjects if s != comp_subject_2])
    samples.append(create_comparison_sample(
        ref_subject, ref_sex, ref_modality,
        comp_subject_5, opposite_sex, other_modality,
        f"case{5 + case_offset}_diff_mod_diff_subj_diff_sex",
        t1_dir, fa_dir
    ))
    
    return samples


def check_both_images_exist(subject_id: str, t1_dir: str, fa_dir: str) -> bool:
    """Check if both T1 and FA images exist for a subject."""
    t1_path = Path(t1_dir) / f"{subject_id}.nii.gz"
    fa_path = Path(fa_dir) / f"{subject_id}.nii.gz"
    return t1_path.exists() and fa_path.exists()


def save_jsonl(samples: List[Dict], output_path: Path):
    """Save samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"  Saved {len(samples)} samples to {output_path}")


def main():
    args = parse_args()
    
    print("="*70)
    print("MULTI-TURN MULTIMODAL COMPARISON CONVERSATION GENERATOR")
    print("="*70)
    
    # Calculate cases per subject based on reference modality
    cases_per_subject = 10 if args.reference_modality == 'both' else 5
    
    print(f"\nConfiguration:")
    print(f"  - Reference modality: {args.reference_modality}")
    print(f"  - Subjects per split: {args.n_subjects_per_split}")
    print(f"  - Cases per subject: {cases_per_subject}")
    print(f"  - Total samples per split: {args.n_subjects_per_split * cases_per_subject}")
    print(f"  - T1 dir: {args.t1_dir}")
    print(f"  - FA dir: {args.fa_dir}")
    
    # Load metadata
    print(f"\nLoading metadata from: {args.meta_csv}")
    df = pd.read_csv(args.meta_csv)
    print(f"  Total rows: {len(df)}")
    
    # Normalize sex and filter valid subjects
    df['sex_normalized'] = df[args.sex_col].apply(normalize_sex)
    df = df[df['sex_normalized'].notna()].copy()
    print(f"  Valid sex labels: {len(df)}")
    
    # Optionally filter by image existence
    if args.check_exists:
        print(f"\nChecking image existence...")
        valid_mask = df[args.subject_col].apply(
            lambda x: check_both_images_exist(x, args.t1_dir, args.fa_dir)
        )
        df = df[valid_mask].copy()
        print(f"  Subjects with both T1 and FA: {len(df)}")
    
    # Get sex distribution
    male_subjects = df[df['sex_normalized'] == 'male'][args.subject_col].tolist()
    female_subjects = df[df['sex_normalized'] == 'female'][args.subject_col].tolist()
    print(f"\nSex distribution:")
    print(f"  Male: {len(male_subjects)}")
    print(f"  Female: {len(female_subjects)}")
    
    # Shuffle and split subjects
    np.random.seed(args.seed)
    np.random.shuffle(male_subjects)
    np.random.shuffle(female_subjects)
    
    # Calculate subjects per sex per split (balanced)
    n_per_sex = args.n_subjects_per_split // 2
    total_needed_per_sex = n_per_sex * 3  # train + val + test
    
    if len(male_subjects) < total_needed_per_sex or len(female_subjects) < total_needed_per_sex:
        print(f"\nWarning: Not enough subjects. Need {total_needed_per_sex} per sex.")
        print(f"  Available: Male={len(male_subjects)}, Female={len(female_subjects)}")
        # Adjust
        n_per_sex = min(len(male_subjects), len(female_subjects)) // 3
        print(f"  Adjusted to {n_per_sex} per sex per split")
    
    # Create splits
    splits = {
        'train': {
            'male': male_subjects[0:n_per_sex],
            'female': female_subjects[0:n_per_sex]
        },
        'val': {
            'male': male_subjects[n_per_sex:2*n_per_sex],
            'female': female_subjects[n_per_sex:2*n_per_sex]
        },
        'test': {
            'male': male_subjects[2*n_per_sex:3*n_per_sex],
            'female': female_subjects[2*n_per_sex:3*n_per_sex]
        }
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build subject pool for comparison (all subjects)
    all_subjects_by_sex = {
        'male': male_subjects[:3*n_per_sex],
        'female': female_subjects[:3*n_per_sex]
    }
    
    # Generate samples for each split
    print("\n" + "="*70)
    print("GENERATING SAMPLES")
    print("="*70)
    
    for split_name, split_subjects in splits.items():
        print(f"\n{split_name.upper()} split:")
        
        split_samples = []
        ref_subjects = split_subjects['male'] + split_subjects['female']
        
        # Get sex for each subject
        subject_sex_map = {}
        for s in split_subjects['male']:
            subject_sex_map[s] = 'male'
        for s in split_subjects['female']:
            subject_sex_map[s] = 'female'
        
        for i, ref_subject in enumerate(ref_subjects):
            ref_sex = subject_sex_map[ref_subject]
            
            # Determine which reference modalities to use
            if args.reference_modality == 'both':
                ref_modalities = ['T1', 'FA']
            else:
                ref_modalities = [args.reference_modality]
            
            for ref_mod in ref_modalities:
                samples = generate_comparison_samples_for_subject(
                    ref_subject, ref_sex,
                    all_subjects_by_sex,
                    args.t1_dir, args.fa_dir,
                    reference_modality=ref_mod,
                    seed=args.seed + i + (0 if ref_mod == 'T1' else 1000)
                )
                split_samples.extend(samples)
        
        print(f"  Reference subjects: {len(ref_subjects)}")
        print(f"  Total samples: {len(split_samples)}")
        
        # Count by case
        case_counts = defaultdict(int)
        for s in split_samples:
            case_counts[s['comparison_case']] += 1
        print("  Samples by case:")
        for case, count in sorted(case_counts.items()):
            print(f"    {case}: {count}")
        
        # Save
        output_path = output_dir / f"{split_name}_conversations.jsonl"
        save_jsonl(split_samples, output_path)
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Reference modality: {args.reference_modality}")
    print(f"\nFiles created:")
    samples_per_split = n_per_sex * 2 * cases_per_subject
    print(f"  - train_conversations.jsonl ({samples_per_split} samples)")
    print(f"  - val_conversations.jsonl ({samples_per_split} samples)")
    print(f"  - test_conversations.jsonl ({samples_per_split} samples)")
    
    print("\n" + "="*70)
    print("SAMPLE FORMAT")
    print("="*70)
    print("""
{
    "task_id": "NDARINV001_case1_same_mod_same_sex",
    "task_type": "multimodal_comparison",
    "comparison_case": "case1_same_mod_same_sex",
    "images": [
        {"path": "/path/to/T1.nii.gz", "modality": "T1"},
        {"path": "/path/to/compare.nii.gz", "modality": "T1"}
    ],
    "conversations": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "male."}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "male."}]}
    ],
    "metadata": {
        "reference_subject": "...",
        "reference_sex": "male",
        "reference_modality": "T1",
        "comparison_subject": "...",
        "comparison_sex": "male", 
        "comparison_modality": "T1",
        "comparison_case": "case1_same_mod_same_sex"
    }
}

10 COMPARISON CASES:

T1 as reference (--reference_modality T1):
1. case1_same_mod_same_sex: T1 vs T1, same sex
2. case2_same_mod_diff_sex: T1 vs T1, different sex
3. case3_diff_mod_same_subj: T1 vs FA, same subject
4. case4_diff_mod_diff_subj_same_sex: T1 vs FA, different subject, same sex
5. case5_diff_mod_diff_subj_diff_sex: T1 vs FA, different subject, different sex

FA as reference (--reference_modality FA):
6. case6_same_mod_same_sex: FA vs FA, same sex
7. case7_same_mod_diff_sex: FA vs FA, different sex
8. case8_diff_mod_same_subj: FA vs T1, same subject
9. case9_diff_mod_diff_subj_same_sex: FA vs T1, different subject, same sex
10. case10_diff_mod_diff_subj_diff_sex: FA vs T1, different subject, different sex

Use --reference_modality both to generate all 10 cases.
    """)


if __name__ == "__main__":
    main()
