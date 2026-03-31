#!/usr/bin/env python3
"""
Modality + Sex Prediction Conversation Generator
=================================================

Generates JSONL files where the model predicts BOTH modality AND sex.

Structure:
- Turn 1: Reference image → "modality, sex" prediction (e.g., "T1, male")
- Turn 2: Comparison image → "modality, sex" prediction (e.g., "FA, female")

Answer format: "{modality}, {sex}" (e.g., "T1, male", "FA, female")

Evaluation: BOTH modality AND sex must be correct.

Options:
--reference_modality T1    → Case 1-5 only (T1 as reference)
--reference_modality FA    → Case 6-10 only (FA as reference)
--reference_modality both  → All 10 cases

Author: UMBRELLA Team
Date: 2026-03-07
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
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
        "Analyze this brain scan. What is the imaging modality and the biological sex of the subject?",
        "Look at this brain MRI. Identify the scan type (T1 or FA) and predict the subject's sex.",
        "Examine this brain scan and determine both the modality and biological sex.",
    ],
    'turn2_question': [
        "Now analyze this second brain scan. What is its modality and the subject's sex?",
        "Compare this second scan. What type of scan is it and what is the subject's sex?",
        "Look at this second brain image. Identify the modality and sex.",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Modality+Sex Prediction JSONL')
    parser.add_argument('--meta_csv', type=str, default=META_CSV)
    parser.add_argument('--t1_dir', type=str, default=T1_IMG_DIR)
    parser.add_argument('--fa_dir', type=str, default=FA_IMG_DIR)
    parser.add_argument('--output_dir', type=str, 
                        default='./modality_sex_prediction')
    parser.add_argument('--subject_col', type=str, default='subjectkey')
    parser.add_argument('--sex_col', type=str, default='sex')
    parser.add_argument('--n_subjects_per_split', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check_exists', action='store_true')
    parser.add_argument('--reference_modality', type=str, default='both',
                        choices=['T1', 'FA', 'both'])
    return parser.parse_args()


def get_image_path(subject_id: str, modality: str, t1_dir: str, fa_dir: str) -> str:
    if modality == "T1":
        return f"{t1_dir}/{subject_id}.nii.gz"
    else:
        return f"{fa_dir}/{subject_id}.nii.gz"


def normalize_sex(sex_value) -> Optional[str]:
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


def get_answer(modality: str, sex: str) -> str:
    """Get combined answer: 'modality, sex' format."""
    return f"{modality}, {sex}."


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
    """Create a multi-turn sample with modality+sex prediction."""
    
    ref_path = get_image_path(ref_subject, ref_modality, t1_dir, fa_dir)
    comp_path = get_image_path(comp_subject, comp_modality, t1_dir, fa_dir)
    
    turn1_question = np.random.choice(PROMPTS['turn1_question'])
    turn1_answer = get_answer(ref_modality, ref_sex)
    turn2_question = np.random.choice(PROMPTS['turn2_question'])
    turn2_answer = get_answer(comp_modality, comp_sex)
    
    return {
        "task_id": f"{ref_subject}_{comparison_case}",
        "task_type": "modality_sex_prediction",
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
            "task": "modality_sex_prediction"
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
    """Generate 5 comparison samples for a reference subject."""
    
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    ref_modality = reference_modality
    other_modality = "FA" if ref_modality == "T1" else "T1"
    
    case_offset = 0 if ref_modality == "T1" else 5
    opposite_sex = "female" if ref_sex == "male" else "male"
    
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
    
    # Case 3/8: Different modality + Same subject
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
    t1_path = Path(t1_dir) / f"{subject_id}.nii.gz"
    fa_path = Path(fa_dir) / f"{subject_id}.nii.gz"
    return t1_path.exists() and fa_path.exists()


def save_jsonl(samples: List[Dict], output_path: Path):
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"  Saved {len(samples)} samples to {output_path}")


def main():
    args = parse_args()
    
    print("="*70)
    print("MODALITY + SEX PREDICTION CONVERSATION GENERATOR")
    print("="*70)
    
    cases_per_subject = 10 if args.reference_modality == 'both' else 5
    
    print(f"\nConfiguration:")
    print(f"  - Task: Predict BOTH modality AND sex")
    print(f"  - Answer format: 'modality, sex' (e.g., 'T1, male')")
    print(f"  - Reference modality: {args.reference_modality}")
    print(f"  - Subjects per split: {args.n_subjects_per_split}")
    print(f"  - Cases per subject: {cases_per_subject}")
    print(f"  - Total samples per split: {args.n_subjects_per_split * cases_per_subject}")
    
    # Load metadata
    print(f"\nLoading metadata from: {args.meta_csv}")
    df = pd.read_csv(args.meta_csv)
    print(f"  Total rows: {len(df)}")
    
    df['sex_normalized'] = df[args.sex_col].apply(normalize_sex)
    df = df[df['sex_normalized'].notna()].copy()
    print(f"  Valid sex labels: {len(df)}")
    
    if args.check_exists:
        print(f"\nChecking image existence...")
        valid_mask = df[args.subject_col].apply(
            lambda x: check_both_images_exist(x, args.t1_dir, args.fa_dir)
        )
        df = df[valid_mask].copy()
        print(f"  Subjects with both T1 and FA: {len(df)}")
    
    male_subjects = df[df['sex_normalized'] == 'male'][args.subject_col].tolist()
    female_subjects = df[df['sex_normalized'] == 'female'][args.subject_col].tolist()
    print(f"\nSex distribution: Male={len(male_subjects)}, Female={len(female_subjects)}")
    
    np.random.seed(args.seed)
    np.random.shuffle(male_subjects)
    np.random.shuffle(female_subjects)
    
    n_per_sex = args.n_subjects_per_split // 2
    
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
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING SAMPLES")
    print("="*70)
    
    for split_name, split_subjects in splits.items():
        print(f"\n{split_name.upper()} split:")
        
        split_subjects_by_sex = {
            'male': split_subjects['male'],
            'female': split_subjects['female']
        }
        
        split_samples = []
        ref_subjects = split_subjects['male'] + split_subjects['female']
        
        subject_sex_map = {}
        for s in split_subjects['male']:
            subject_sex_map[s] = 'male'
        for s in split_subjects['female']:
            subject_sex_map[s] = 'female'
        
        for i, ref_subject in enumerate(ref_subjects):
            ref_sex = subject_sex_map[ref_subject]
            
            if args.reference_modality == 'both':
                ref_modalities = ['T1', 'FA']
            else:
                ref_modalities = [args.reference_modality]
            
            for ref_mod in ref_modalities:
                samples = generate_comparison_samples_for_subject(
                    ref_subject, ref_sex,
                    split_subjects_by_sex,
                    args.t1_dir, args.fa_dir,
                    reference_modality=ref_mod,
                    seed=args.seed + i + (0 if ref_mod == 'T1' else 1000)
                )
                split_samples.extend(samples)
        
        print(f"  Reference subjects: {len(ref_subjects)}")
        print(f"  Total samples: {len(split_samples)}")
        
        case_counts = defaultdict(int)
        for s in split_samples:
            case_counts[s['comparison_case']] += 1
        print("  Samples by case:")
        for case, count in sorted(case_counts.items()):
            print(f"    {case}: {count}")
        
        output_path = output_dir / f"{split_name}_conversations.jsonl"
        save_jsonl(split_samples, output_path)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nAnswer format: 'modality, sex'")
    print(f"Examples: 'T1, male', 'FA, female'")
    print(f"\nEvaluation: BOTH modality AND sex must be correct!")


if __name__ == "__main__":
    main()
