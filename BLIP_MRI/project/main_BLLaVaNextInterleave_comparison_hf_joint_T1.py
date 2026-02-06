"""
Main training script for Multi-turn Comparison Tasks with LLaVA-NeXT-Interleave

Supports:
- Sex comparison (reference + query → predict sex)
- Numerical values comparison (reference + query → predict age, bmi, ...)
- 2-turn conversation format
"""

import datetime
import hashlib
from omegaconf import OmegaConf
from omegaconf import ListConfig

import torch
import transformers
from transformers import Trainer, TrainingArguments
from utils.Trainer_LLaVaNextInterleave_comparison import CustomTrainer
from utils.Trainer_LLaVaNextInterleave_comparison import compute_metrics_with_tokenizer, preprocess_logits_for_metrics

from utils.data import CustomDataCollatorWithPadding
from dataset.dataset_T1_LLaVaNextInterleave_comparison import ComparisonDataModule

import os
import wandb

import warnings
warnings.filterwarnings('ignore')

def __main__():
    ### setting huggingface verbose
    transformers.logging.set_verbosity_info()

    ### make experiment ID
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]

    config = OmegaConf.load("./config/Brain_LLaVa_train_Deepspeed_joint_multiturn_comparison.yaml")

    ### setting logger
    wandb.login(key=config.wandb.API_KEY)
    os.environ['WANDB_PROJECT'] = "BLIP_sMRI_LLaVA_Next_Interleave_MultiTurn_Comparison"
    os.environ["WANDB_RUN_ID"] = f'{hash_key}'

    ### setting seed
    transformers.set_seed(config.seed)

    ### setting processor and tokenizer for LLaVA-NeXT-Interleave
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-0.5b-hf", trust_remote_code=True)
    tokenizer = processor.tokenizer

    ### Load comparison task datasets from multiple sources
    from utils.data import InterleaveDataset

    train_datasets = []
    eval_datasets = []
    test_datasets = []


    # Support multiple JSON files (e.g., ABCD, UKB, etc.)
    if isinstance(config.dataset.train_json, (list, ListConfig)):
        train_json_list = list(config.dataset.train_json)
        val_json_list = list(config.dataset.val_json)
        test_json_list = list(config.dataset.test_json)
    else:
        train_json_list = [config.dataset.train_json]
        val_json_list = [config.dataset.val_json]
        test_json_list = [config.dataset.test_json]

    for train_json, val_json, test_json in zip(train_json_list, val_json_list, test_json_list):
        data_module = ComparisonDataModule(
            train_json=train_json,
            val_json=val_json,
            test_json=test_json,
            processor=processor,
            img_size=config.dataset.img_size
        )

        if data_module.train_dataset is not None:
            train_datasets.append(data_module.train_dataset)
        if data_module.val_dataset is not None:
            eval_datasets.append(data_module.val_dataset)
        if data_module.test_dataset is not None:
            test_datasets.append(data_module.test_dataset)

    # Concatenate all datasets
    if len(train_datasets) > 1:
        train_dataset = InterleaveDataset(train_datasets, shuffle=True, seed=config.seed)
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = None

    if len(eval_datasets) > 1:
        eval_dataset = InterleaveDataset(eval_datasets, shuffle=False, seed=config.seed)
    elif len(eval_datasets) == 1:
        eval_dataset = eval_datasets[0]
    else:
        eval_dataset = None

    if len(test_datasets) > 1:
        test_dataset = InterleaveDataset(test_datasets, shuffle=False, seed=config.seed)
    elif len(test_datasets) == 1:
        test_dataset = test_datasets[0]
    else:
        test_dataset = None

    #### setting model - LLaVA-NeXT-Interleave
    # from model.Bblip_t5 import PatchEmbed
    from model.Bblip_t5_interleave import PatchEmbedInterleave
    from transformers import LlavaForConditionalGeneration

    # Load LLaVA-NeXT-Interleave model (Qwen-based)
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-interleave-qwen-0.5b-hf",
        # torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        # trust_remote_code=True,
        # attn_implementation="eager"
    )

    patch_embed = PatchEmbedInterleave(
                    T1_size=config.dataset.img_size,
                    T1_patch_size=config.model.patch_size,
                    in_chans=1,
                    embed_dim=int(model.vision_tower.vision_model.embeddings.patch_embedding.out_channels)
                )

    # # Replace vision encoder's patch embedding layer for 3D brain MRI
    # patch_embed = PatchEmbedInterleave(
    #         T1_size=config.dataset.img_size,
    #         T1_patch_size=config.model.patch_size,
    #         rsfMRI_size=[96, 96, 96, 24],  # Placeholder (not used)
    #         rsfMRI_patch_size=[16, 16, 16, 3],  # Placeholder (not used)
    #         in_chans=1,
    #         embed_dim=int(model.vision_tower.vision_model.embeddings.patch_embedding.out_channels))

    setattr(model.vision_tower.vision_model, "embeddings", patch_embed)

    # Freeze vision encoder except embeddings
    for name, param in model.vision_tower.vision_model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        if 'pre_layernorm' in name:
            param.requires_grad = False
        if 'post_layernorm' in name:
            param.requires_grad = False
        if 'embeddings' in name:
            param.requires_grad = True

    # Freeze multi-modal projector
    for name, param in model.named_parameters():
        if 'multi_modal_projector' in name:
            param.requires_grad = False

    # Freeze language model
    for name, param in model.named_parameters():
        if 'model.layers' in name:  # Qwen2 uses model.layers
            param.requires_grad = False
        if 'lm_head' in name:
            param.requires_grad = False

    # set gradient checkpointing
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        # basic settings
        output_dir=f'./hf_results/{os.environ["WANDB_RUN_ID"]}',
        do_train=True,
        do_eval=True,
        remove_unused_columns=False,
        # training
        num_train_epochs=config.trainer.max_epochs,
        learning_rate=config.trainer.learning_rate,
        warmup_steps=config.trainer.warmup_steps,
        weight_decay=config.trainer.weight_decay,
        per_device_train_batch_size=config.trainer.per_device_batch_size,
        per_device_eval_batch_size=config.trainer.per_device_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        # # arguments for reducing memory
        # bf16=True,
        # bf16_full_eval=True,
        # for evaluation and loggings
        report_to = 'wandb',
        logging_dir=f'./hf_logs/{os.environ["WANDB_RUN_ID"]}',
        logging_steps=config.trainer.logging_steps,
        eval_strategy="steps",
        eval_steps=1000,
        eval_accumulation_steps=1,
        save_steps=1000,
        disable_tqdm=False,
        # checkpoint saving
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True
    )

    # Determine task type for metrics
    # Support both single target and multi-target (list)
    target_col = config.dataset.get('target_col', None)
    task_type = config.dataset.get('task_type', 'categorical')

    if target_col:
        # target_col can be string or list
        if isinstance(target_col, list):
            targets = target_col
            print(f"[INFO] Multi-task mode with targets: {targets}")
        else:
            targets = [target_col]
            print(f"[INFO] Single-task mode with target: {target_col}")
    else:
        print(f"[WARN] No target_col or task_type specified")


    # Use existing compute_metrics - it already handles long reasoning text!
    # The key difference is that multi-turn generates longer responses,
    # but the extraction logic (regex search for 'male'/'female' or numbers) is the same
    trainer = CustomTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics_with_tokenizer(tokenizer=tokenizer, targets=targets),
        data_collator = CustomDataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=512
        ),
        model_optimization_type = 'joint',
    )

    # training
    trainer.train()

    # test
    if test_dataset is not None:
        trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')


if __name__ == '__main__':
    __main__()


"""
##TODO

1. Generate JSON files:
   python generate_json_general_comparison_split.py

2. Update config with correct paths

3. Train:
   python main_MultiTurn_Comparison.py
"""
