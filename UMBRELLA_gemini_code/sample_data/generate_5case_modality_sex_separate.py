#!/usr/bin/env python3
import json, argparse, pandas as pd, numpy as np, os
from pathlib import Path
from collections import defaultdict

T1_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer_256"
FA_DIR = "/pscratch/sd/h/heehaw/data/1.ABCD/3.1.1.FA_unwarpped_nii_cleaned"
META_CSV = "/pscratch/sd/h/heehaw/data/1.ABCD/ABCD_phenotype_total.csv"

def normalize_sex(v):
    if pd.isna(v): return None
    if isinstance(v, (int,float)): return 'male' if v==1 else ('female' if v==2 else None)
    if isinstance(v, str):
        s = v.lower().strip()
        if s in ['m','male','1']: return 'male'
        if s in ['f','female','2']: return 'female'
    return None

def check_images_exist(subj, t1_dir, fa_dir):
    return os.path.exists(f"{t1_dir}/{subj}.nii.gz") and os.path.exists(f"{fa_dir}/{subj}.nii.gz")

def create_sample(ref_subj, ref_sex, ref_mod, comp_subj, comp_sex, comp_mod, case, t1_dir, fa_dir):
    ref_path = f"{t1_dir if ref_mod=='T1' else fa_dir}/{ref_subj}.nii.gz"
    comp_path = f"{t1_dir if comp_mod=='T1' else fa_dir}/{comp_subj}.nii.gz"
    return {
        "task_id": f"{ref_subj}_{case}",
        "task_type": "modality_sex_separate",
        "comparison_case": case,
        "images": [{"path": ref_path, "modality": ref_mod}, {"path": comp_path, "modality": comp_mod}],
        "conversations": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is the imaging modality of this brain scan?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{ref_mod}."}]},
            {"role": "user", "content": [{"type": "text", "text": "What is the biological sex of this subject?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{ref_sex}."}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is the imaging modality of this brain scan?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{comp_mod}."}]},
            {"role": "user", "content": [{"type": "text", "text": "What is the biological sex of this subject?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{comp_sex}."}]}
        ],
        "metadata": {"reference_subject": ref_subj, "reference_sex": ref_sex, "reference_modality": ref_mod,
                     "comparison_subject": comp_subj, "comparison_sex": comp_sex, "comparison_modality": comp_mod,
                     "comparison_case": case}
    }

def gen_samples(ref_subj, ref_sex, subj_by_sex, t1_dir, fa_dir, seed):
    np.random.seed(seed)
    opp_sex = "female" if ref_sex=="male" else "male"
    same = [s for s in subj_by_sex[ref_sex] if s!=ref_subj]
    diff = subj_by_sex[opp_sex]
    if len(same)<2 or len(diff)<2: return []
    return [
        create_sample(ref_subj, ref_sex, "T1", np.random.choice(same), ref_sex, "T1", "case1_same_mod_same_sex", t1_dir, fa_dir),
        create_sample(ref_subj, ref_sex, "T1", np.random.choice(diff), opp_sex, "T1", "case2_same_mod_diff_sex", t1_dir, fa_dir),
        create_sample(ref_subj, ref_sex, "T1", ref_subj, ref_sex, "FA", "case3_diff_mod_same_subj", t1_dir, fa_dir),
        create_sample(ref_subj, ref_sex, "T1", np.random.choice(same), ref_sex, "FA", "case4_diff_mod_diff_subj_same_sex", t1_dir, fa_dir),
        create_sample(ref_subj, ref_sex, "T1", np.random.choice(diff), opp_sex, "FA", "case5_diff_mod_diff_subj_diff_sex", t1_dir, fa_dir),
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./data/5case_modality_sex_separate')
    parser.add_argument('--n_subjects_per_split', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t1_dir', default=T1_DIR)
    parser.add_argument('--fa_dir', default=FA_DIR)
    args = parser.parse_args()
    
    print("Loading metadata...")
    df = pd.read_csv(META_CSV)
    df['sex'] = df['sex'].apply(normalize_sex)
    df = df[df['sex'].notna()]
    print(f"Total subjects with sex label: {len(df)}")
    
    print("Checking image existence (T1 + FA)...")
    valid = []
    for subj in df['subjectkey'].tolist():
        if check_images_exist(subj, args.t1_dir, args.fa_dir):
            valid.append(subj)
    df = df[df['subjectkey'].isin(valid)]
    print(f"Subjects with BOTH T1 and FA: {len(df)}")
    
    males = df[df['sex']=='male']['subjectkey'].tolist()
    females = df[df['sex']=='female']['subjectkey'].tolist()
    print(f"Male: {len(males)}, Female: {len(females)}")
    
    np.random.seed(args.seed)
    np.random.shuffle(males)
    np.random.shuffle(females)
    
    n = args.n_subjects_per_split // 2
    splits = {
        'train': {'male': males[0:n], 'female': females[0:n]},
        'val': {'male': males[n:2*n], 'female': females[n:2*n]},
        'test': {'male': males[2*n:3*n], 'female': females[2*n:3*n]}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for split, subjs in splits.items():
        samples = []
        refs = subjs['male'] + subjs['female']
        sex_map = {s:'male' for s in subjs['male']}
        sex_map.update({s:'female' for s in subjs['female']})
        split_subj = {'male': list(subjs['male']), 'female': list(subjs['female'])}
        for i, ref in enumerate(refs):
            samples.extend(gen_samples(ref, sex_map[ref], split_subj, args.t1_dir, args.fa_dir, args.seed+i))
        
        out = Path(args.output_dir) / f"{split}_conversations.jsonl"
        with open(out, 'w') as f:
            for s in samples: f.write(json.dumps(s)+'\n')
        
        cases = defaultdict(int)
        for s in samples: cases[s['comparison_case']] += 1
        print(f"{split}: {len(samples)} samples")
        for c,cnt in sorted(cases.items()): print(f"  {c}: {cnt}")

if __name__ == "__main__":
    main()
