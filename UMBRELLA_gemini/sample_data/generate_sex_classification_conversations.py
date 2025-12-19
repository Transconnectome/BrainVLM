#!/usr/bin/env python3
"""
Sex Classification Conversation Generator (Single Turn)
=====================================================

Generates single-turn conversation JSON files for simple sex classification.
Format: User presents image -> Assistant predicts sex.

Dataset: ABCD Study
Task: Sex Classification (T1)

Structure:
    User: <image>\nAnalyze this brain MRI scan. Determine the biological sex of the subject.
    Assistant: [male/female]

Author: BrainVLM Team (Modified for Single Subject)
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configuration
SPLITS_DIR = Path("sex_classification_splits_100subjects")
OUTPUT_DIR = Path("sex_classification_conversations_100subjects")
IMAGE_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256"
IMAGE_TEMPLATE = "{subject_id}.nii.gz"

# Roles
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Task configuration
TASK_TYPE = "T1" # Classification Task Type
MODALITY = 'sMRI'

def create_single_turn_conversation(subject_row) -> Dict:
    """Create a single conversation entry for sex classification."""
    
    subject_id = subject_row['subject_id']
    sex_label = subject_row['sex'] # 'male' or 'female'
    sex_label = 'male' if int(sex_label) == 1 else 'female'
    
    # Construct Image Path
    image_path = f"{IMAGE_DIR}/{IMAGE_TEMPLATE.format(subject_id=subject_id)}"
    
    # 1. Define Task ID
    task_id = f"{subject_id}_sex_classification"
    
    # 2. Define Conversation Content
    # Prompt variations can be added here if needed
    question_text = "Analyze this T1-weighted brain MRI scan. Determine the biological sex of the subject."
    
    # Answer text (Ground Truth)
    answer_text = f"{sex_label}." # e.g., "male." or "female."
    
    # 3. Construct Conversation Object (LLaVA-Next format compatible)
    # Note: <image> token handling is done by the Tokenizer/Dataset class usually,
    # but here we include it in the content structure as per your previous pipeline.
    
    conversation = [
        {
            "role": ROLE_USER,
            "content": [
                {
                    "type": "image",
                    "modality": MODALITY,
                    "image_path": image_path
                },
                {
                    "type": "text",
                    "text": question_text
                }
            ]
        },
        {
            "role": ROLE_ASSISTANT,
            "content": [
                {
                    "type": "text",
                    "text": answer_text
                }
            ]
        }
    ]
    
    # 4. Create Entry
    entry = {
        "task_id": task_id,
        "task_type": TASK_TYPE,
        "subject_ids": [subject_id], # List for consistency
        "modalities": [MODALITY],
        "images": [
            {
                "path": image_path,
                "token": "<image>",
                "modality": MODALITY
            }
        ],
        "conversations": conversation,
        "metadata": {
            "subject_id": subject_id,
            "subject_label": sex_label,
            "task": "sex_classification"
        }
    }
    
    return entry

def save_conversations(split_name: str, conversations: List[Dict], output_dir: Path):
    """Save conversations to JSONL and individual JSONs."""
    
    # Setup directories
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main JSONL file (used for training/eval)
    jsonl_path = output_dir / f"{split_name}_conversations.jsonl" # Root level or split folder? Usually split folder for clean org.
    # User's previous structure seemed to have one main file per split context, let's follow standard:
    # Saving to output_dir/train_conversations.jsonl
    
    target_file = output_dir / f"{split_name}_conversations.jsonl"
    
    print(f"  Saving {len(conversations)} conversations to {target_file}...")
    
    with open(target_file, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
            
    # Optional: Save individual JSONs for inspection (first 5 only to save space/time)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for i, conv in enumerate(conversations[:5]):
        sample_path = sample_dir / f"sample_{split_name}_{i:03d}.json"
        with open(sample_path, 'w') as f:
            json.dump(conv, f, indent=2)

def generate_conversations_for_split(split_name: str, df: pd.DataFrame) -> List[Dict]:
    """Generate conversation list for a dataframe."""
    conversations = []
    
    print(f"Generating prompts for {len(df)} subjects in {split_name}...")
    
    for _, row in df.iterrows():
        conv = create_single_turn_conversation(row)
        conversations.append(conv)
        
    return conversations

def main():
    # Load Splits
    if not SPLITS_DIR.exists():
        print(f"Error: Split directory {SPLITS_DIR} does not exist.")
        print("Run 'create_sex_classification_dataset.py' first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        csv_path = SPLITS_DIR / f"{split}_subjects.csv"
        
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
            
        print(f"\nProcessing {split.upper()} split:")
        df = pd.read_csv(csv_path)
        
        conversations = generate_conversations_for_split(split, df)
        save_conversations(split, conversations, OUTPUT_DIR)
        
    print("\n" + "="*60)
    print("CONVERSATION GENERATION COMPLETE (Single Turn)")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()