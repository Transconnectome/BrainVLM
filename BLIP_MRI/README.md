# LLaVA-NeXT-Interleave for Brain MRI Comparison Tasks

Multi-image comparison framework using LLaVA-NeXT-Interleave for brain MRI analysis.

---

## Overview

**Architecture:** LLaVA-NeXT-Interleave (Qwen-0.5b)
**Task:** Reference-augmented comparison 
**Format:** Multi-turn conversation with interleaved images

**Example:**
```
Turn 1 (Reference):
User: "Here is a brain scan from a male participant. <image>"
Assistant: "Understood. I've analyzed the reference scan."

Turn 2 (Query):
User: "Compare this scan with the reference. <image> What is the sex?"
Assistant: "Based on the comparison, the subject is male."
```

---

## 1. Data Preparation

### Generate Comparison JSON

#### Sex(or any categorical attributes) Comparison (Categorical)

```bash
python generate_json_comparison_split_general.py \
  --study_sample ABCD \
  --meta_path /path/to/phenotype.csv \
  --img_dir /path/to/images \
  --target_col sex # (can be other categorical attributes)\
  --task_type categorical \
  --output_dir ./ \
  --num_pairs 3 \
  --seed 1234
```

**Output:**
- `data/ABCD_sex_comparison_tasks_train.json`
- `data/ABCD_sex_comparison_tasks_val.json`
- `data/ABCD_sex_comparison_tasks_test.json`

#### BMI_sds(or any numerical attributes) Regression (Numerical)

```bash
python generate_json_comparison_split_general.py \
  --study_sample ABCD \
  --meta_path /path/to/phenotype.csv \
  --img_dir /path/to/images \
  --target_col BMI_sds # (can be other numerical attributes) \
  --task_type numerical \
  --output_dir ./ \
  --num_pairs 3 \ 
  --seed 1234
```

**Output:**
- `data/ABCD_BMI_sds_comparison_tasks_train.json`
- `data/ABCD_BMI_sds_comparison_tasks_val.json`
- `data/ABCD_BMI_sds_comparison_tasks_test.json`

**Key Parameters:**
- `--num_pairs`: Number of references per query subject

---

## 2. Data Split Logic

**Complete Separation:**
- **Inter-split:** Train/Val/Test subjects do NOT overlap
- **Intra-split:** Query and Reference pools do NOT overlap within each split

**Example (1000 subjects, 70/15/15 split):**
```
Train: 700 subjects
  ├─ Query: 350 subjects
  └─ Reference: 350 subjects (different from query!)

Val: 150 subjects
  ├─ Query: 75 subjects
  └─ Reference: 75 subjects

Test: 150 subjects
  ├─ Query: 75 subjects
  └─ Reference: 75 subjects
```

**Why?** Test subjects NEVER appear in training (even as references) for true generalization test

---

## 3. Training

### Configure

Edit `config/Brain_LLaVa_train_Deepspeed_joint_multiturn_comparison.yaml`:

```yaml
dataset:
    target_col: "sex"  # or "age", "diagnosis", etc.

    train_json:
        - "./data/ABCD_sex_comparison_tasks_train.json"
    val_json:
        - "./data/ABCD_sex_comparison_tasks_val.json"
    test_json:
        - "./data/ABCD_sex_comparison_tasks_test.json"

```

## 4. Troubleshooting

**Error: File not found**
- Check image paths in JSON match actual files

**Error: Image token mismatch**
- Ensure using updated `Trainer_LLaVA_Next_interleave.py`

**Low metrics**
- Check data split: Are train/val/test balanced?
- Review generation logs for prediction quality

