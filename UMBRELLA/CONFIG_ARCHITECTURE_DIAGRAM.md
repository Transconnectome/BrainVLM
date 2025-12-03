# UMBRELLA Configuration Architecture - Visual Reference

## Before Consolidation (Broken)

```
┌─────────────────────────────────────────────────────────────┐
│                    main() Entry Point                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Load UMBRELLATrainingConfig from YAML               │
│                                                             │
│  Attributes: model_name, batch_size, learning_rate,         │
│              img_size, modality, etc. (32 attributes)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│      Create vanilla TrainingArguments (WRONG!)              │
│                                                             │
│  training_args = TrainingArguments(                         │
│      output_dir=config.output_dir,                          │
│      num_train_epochs=config.num_epochs,                    │
│      ...                                                    │
│  )                                                          │
│                                                             │
│  Missing: task_type_weights, mask_human_turns,              │
│           enable_task_aware_loss, etc. (13 attributes)      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Create UMBRELLATrainer                           │
│                                                             │
│  trainer = UMBRELLATrainer(                                 │
│      model=model,                                           │
│      args=training_args  ← Wrong type!                      │
│  )                                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         UMBRELLATrainer.__init__()                          │
│                                                             │
│  self.loss_computer = TaskAwareLossComputer(                │
│      task_weights=args.task_type_weights  ← AttributeError! │
│  )                                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ❌ TRAINING FAILS


┌─────────────────────────────────────────────────────────────┐
│           UMBRELLATrainingArgs (DEFINED BUT UNUSED)         │
│                                                             │
│  @dataclass                                                 │
│  class UMBRELLATrainingArgs(TrainingArguments):             │
│      task_type_weights: Optional[Dict[str, float]] = None   │
│      enable_task_aware_loss: bool = True                    │
│      mask_human_turns: bool = True                          │
│      ...  (13 UMBRELLA-specific attributes)                 │
└─────────────────────────────────────────────────────────────┘
           ↑
           │
     Never instantiated!
```

---

## After Consolidation (Fixed)

```
┌─────────────────────────────────────────────────────────────┐
│                    main() Entry Point                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Load UMBRELLATrainingConfig from YAML (ENHANCED)           │
│                                                             │
│  Original Attributes (32):                                  │
│    • model_name, tokenizer_name                             │
│    • train_json_path, eval_json_path, task_filter          │
│    • modality, img_size, patch_size                         │
│    • T1_img_size, T1_patch_size                             │
│    • rsfMRI_img_size, rsfMRI_patch_size                     │
│    • batch_size, learning_rate, num_epochs                  │
│    • max_seq_length, max_images_per_sample                  │
│    • gradient_accumulation_steps, warmup_steps              │
│    • enable_memory_aware_batching, memory_budget_gb         │
│    • gradient_checkpointing, mixed_precision                │
│    • output_dir, logging_steps, save_steps, eval_steps      │
│    • save_total_limit, use_wandb, wandb_project             │
│                                                             │
│  NEW Attributes (13):                                       │
│    ✓ mask_human_turns, mask_padding_tokens                  │
│    ✓ enable_task_aware_loss, task_type_weights              │
│    ✓ enable_dummy_loss, dummy_loss_weight                   │
│    ✓ log_turn_distribution, log_image_statistics            │
│    ✓ log_memory_usage                                       │
│    ✓ normalize_gradients_by_batch_size, base_batch_size     │
│                                                             │
│  Total: 45 attributes (Single Source of Truth)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Factory Method: config.to_training_args()           │
│                                                             │
│  def to_training_args(self,                                 │
│                       eval_dataset_available: bool = False  │
│                      ) -> UMBRELLATrainingArgs:             │
│                                                             │
│      return UMBRELLATrainingArgs(                           │
│          # Standard HuggingFace args (20+)                  │
│          output_dir=self.output_dir,                        │
│          num_train_epochs=self.num_epochs,                  │
│          per_device_train_batch_size=self.batch_size,       │
│          learning_rate=self.learning_rate,                  │
│          ...                                                │
│                                                             │
│          # UMBRELLA-specific args (13)                      │
│          mask_human_turns=self.mask_human_turns,            │
│          task_type_weights=self.task_type_weights,          │
│          enable_task_aware_loss=self.enable_task_aware_loss,│
│          enable_dummy_loss=self.enable_dummy_loss,          │
│          ...                                                │
│      )                                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│      UMBRELLATrainingArgs (PROPERLY INSTANTIATED)           │
│                                                             │
│  @dataclass                                                 │
│  class UMBRELLATrainingArgs(TrainingArguments):             │
│      # Inherits 100+ TrainingArguments attributes           │
│      # + 13 UMBRELLA-specific attributes                    │
│                                                             │
│  Instance created with ALL attributes:                      │
│    ✓ task_type_weights = None                              │
│    ✓ enable_task_aware_loss = True                         │
│    ✓ mask_human_turns = True                               │
│    ✓ enable_dummy_loss = True                              │
│    ✓ ... all other attributes                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Create UMBRELLATrainer (CORRECT TYPE)            │
│                                                             │
│  trainer = UMBRELLATrainer(                                 │
│      model=model,                                           │
│      args=training_args  ← Correct UMBRELLATrainingArgs!    │
│      train_dataset=train_dataset,                           │
│      eval_dataset=eval_dataset,                             │
│      data_collator=collator,                                │
│      tokenizer=tokenizer                                    │
│  )                                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         UMBRELLATrainer.__init__()                          │
│                                                             │
│  self.turn_mask_builder = TurnMaskBuilder(tokenizer)        │
│                                                             │
│  self.loss_computer = TaskAwareLossComputer(                │
│      task_weights=args.task_type_weights  ← ✓ Works!       │
│      enable_dummy_loss=args.enable_dummy_loss  ← ✓ Works!  │
│  )                                                          │
│                                                             │
│  All attributes accessible:                                 │
│    ✓ args.mask_human_turns                                 │
│    ✓ args.enable_task_aware_loss                           │
│    ✓ args.log_turn_distribution                            │
│    ✓ args.normalize_gradients_by_batch_size                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ✓ TRAINING SUCCEEDS
```

---

## Key Design Decisions

### 1. Single Source of Truth
```
UMBRELLATrainingConfig owns ALL configuration
        │
        ├─ High-level settings (model, data, training)
        ├─ UMBRELLA-specific settings (masking, loss, logging)
        └─ Conversion logic (to_training_args factory method)
```

### 2. Clean Separation of Concerns
```
UMBRELLATrainingConfig
    ↓ (what user configures)

to_training_args()
    ↓ (conversion bridge)

UMBRELLATrainingArgs
    ↓ (what trainer receives)

UMBRELLATrainer
    ↓ (what executes training)
```

### 3. Type Safety
```
config: UMBRELLATrainingConfig
    ↓
training_args: UMBRELLATrainingArgs  ← Type-safe conversion
    ↓
trainer.__init__(args: UMBRELLATrainingArgs)  ← Type matches
```

---

## Attribute Flow Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                   YAML Configuration File                     │
│                                                               │
│  dataset:                                                     │
│    T1:                                                        │
│      img_size: [96, 96, 96]                                   │
│  model:                                                       │
│    hf_name: "llava-hf/llava-interleave-qwen-0.5b-hf"          │
│  trainer:                                                     │
│    per_device_batch_size: 2                                   │
│    learning_rate: 5e-5                                        │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│          UMBRELLATrainingConfig.from_yaml()                   │
│                                                               │
│  Parses YAML and creates config instance                     │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              UMBRELLATrainingConfig Instance                  │
│                                                               │
│  config.model_name = "llava-hf/..."                           │
│  config.batch_size = 2                                        │
│  config.learning_rate = 5e-5                                  │
│  config.img_size = [96, 96, 96]                               │
│  config.mask_human_turns = True                               │
│  config.task_type_weights = None                              │
│  ... (45 total attributes)                                    │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│          config.to_training_args() Factory Method             │
│                                                               │
│  Maps config attributes to training args attributes           │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│           UMBRELLATrainingArgs Instance                       │
│                                                               │
│  Standard HuggingFace TrainingArguments:                      │
│    args.output_dir = config.output_dir                        │
│    args.num_train_epochs = config.num_epochs                  │
│    args.per_device_train_batch_size = config.batch_size       │
│    args.learning_rate = config.learning_rate                  │
│    args.fp16 = (config.mixed_precision == "fp16")             │
│    args.bf16 = (config.mixed_precision == "bf16")             │
│    ... (20+ standard attributes)                              │
│                                                               │
│  UMBRELLA-specific Extensions:                                │
│    args.mask_human_turns = config.mask_human_turns            │
│    args.task_type_weights = config.task_type_weights          │
│    args.enable_task_aware_loss = config.enable_task_aware_loss│
│    args.enable_dummy_loss = config.enable_dummy_loss          │
│    args.log_turn_distribution = config.log_turn_distribution  │
│    ... (13 UMBRELLA attributes)                               │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                  UMBRELLATrainer.__init__()                   │
│                                                               │
│  Accesses args attributes:                                    │
│    self.loss_computer = TaskAwareLossComputer(                │
│        task_weights=args.task_type_weights  ✓                 │
│    )                                                          │
│                                                               │
│    if args.mask_human_turns:  ✓                              │
│        self.turn_mask_builder = TurnMaskBuilder(...)          │
│                                                               │
│    if args.log_turn_distribution:  ✓                         │
│        self.metrics_history['turn_distribution'] = []         │
└───────────────────────────────────────────────────────────────┘
```

---

## Code Comparison

### Before: Manual Mapping (22 lines)

```python
training_args = TrainingArguments(
    output_dir=self.config.output_dir,
    num_train_epochs=self.config.num_epochs,
    per_device_train_batch_size=self.config.batch_size,
    per_device_eval_batch_size=self.config.batch_size,
    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
    learning_rate=self.config.learning_rate,
    warmup_steps=self.config.warmup_steps,
    logging_steps=self.config.logging_steps,
    eval_steps=self.config.eval_steps if eval_dataset else None,
    save_steps=self.config.save_steps,
    save_total_limit=self.config.save_total_limit,
    fp16=self.config.mixed_precision == "fp16",
    bf16=self.config.mixed_precision == "bf16",
    save_strategy="steps",
    evaluation_strategy="steps" if eval_dataset else "no",
    logging_strategy="steps",
    report_to="wandb" if self.config.use_wandb else "none",
    load_best_model_at_end=True if eval_dataset else False,
    metric_for_best_model="loss" if eval_dataset else None,
    greater_is_better=False if eval_dataset else None,
)
# Missing: task_type_weights, mask_human_turns, etc. → AttributeError
```

### After: Factory Method (1 line)

```python
training_args = self.config.to_training_args(
    eval_dataset_available=(eval_dataset is not None)
)
# All attributes present ✓
```

---

## Summary

**Problem**: Two parallel config classes, no bridge, AttributeError

**Solution**: Unified config + factory method + type safety

**Result**: Single line conversion, zero errors, maintainable code

**Key Achievement**: `config.to_training_args()` - clean, simple, correct
