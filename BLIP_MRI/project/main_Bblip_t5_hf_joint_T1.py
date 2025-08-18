import datetime
import hashlib
from omegaconf import OmegaConf

import torch
import transformers
from transformers import Trainer, TrainingArguments
from utils.Trainer import CustomTrainer
from utils.Trainer import compute_metrics_with_tokenizer, preprocess_logits_for_metrics


from dataset.dataset_T1 import ABCD_T1, UKB_T1
from dataset.datamodule_rsfMRI import rsfMRIData
from utils.data import CustomDataCollatorWithPadding, InterleaveDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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

    #config = OmegaConf.load("/pscratch/sd/h/heehaw/BLIP_MRI/project/config/Brain_blip_t5_train_DeepSpeed_joint.yaml") 
    config = OmegaConf.load("/pscratch/sd/h/heehaw/BLIP_MRI/project/config/Brain_blip_t5_train_DeepSpeed_joint_T1.yaml") 

    ### setting logger 
    wandb.login(key=config.wandb.API_KEY)
    os.environ['WANDB_PROJECT'] = "BLIP_sMRI"
    os.environ["WANDB_RUN_ID"] = f'{hash_key}'


    ### setting seed 
    #pl.seed_everything(config.seed)
    transformers.set_seed(config.seed)
    

    ### setting tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_name, use_fast=False)     # transformers <=4.30.2, "use_fast" argument should be set to False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    
    ### setting T1 dataset
    T1_train_datasets = [] 
    T1_eval_datasets = [] 
    T1_test_datasets = [] 
    for study_sample, img_dir, meta_dir in zip(config.dataset.T1.study_sample, config.dataset.T1.img_dir, config.dataset.T1.meta_dir):
        if study_sample == "ABCD": 
            ABCD_dataset = ABCD_T1(tokenizer=tokenizer,config_dataset=config.dataset, img_dir=img_dir, meta_dir=meta_dir)
            # for train dataset
            T1_train_datasets.append(ABCD_dataset.train_dataset)
            # for val/test dataset 
            T1_eval_datasets.append(ABCD_dataset.val_dataset)
            T1_test_datasets.append(ABCD_dataset.test_dataset)
        elif study_sample == 'UKB': 
            UKB_dataset = UKB_T1(tokenizer=tokenizer, config_dataset=config.dataset, img_dir=img_dir, meta_dir=meta_dir)
            # for train dataset
            T1_train_datasets.append(UKB_dataset.train_dataset)
            # for val/test dataset 
            T1_eval_datasets.append(UKB_dataset.val_dataset)
            T1_test_datasets.append(UKB_dataset.test_dataset)
    
    ### setting rsfMRI dataset
    rsfMRI_train_datasets = [] 
    rsfMRI_eval_datasets = [] 
    rsfMRI_test_datasets = [] 
    for study_sample, img_dir in zip(config.dataset.rsfMRI.study_sample, config.dataset.rsfMRI.img_dir):
        rsfMRI_dataset = rsfMRIData(
                                    dataset_name=study_sample, 
                                    image_path=img_dir, 
                                    target=config.dataset.rsfMRI.target, 
                                    config=config.dataset,
                                    tokenizer=tokenizer
                                    )
        # for train dataset
        rsfMRI_train_datasets.append(rsfMRI_dataset.train_dataset)
        # for val/test dataset 
        rsfMRI_eval_datasets.append(rsfMRI_dataset.val_dataset)
        rsfMRI_test_datasets.append(rsfMRI_dataset.test_dataset)


    ### Concatenating all modality dataset 
    All_modality_train_datasets = T1_train_datasets + rsfMRI_train_datasets
    All_modality_eval_datasets = T1_eval_datasets + rsfMRI_eval_datasets
    All_modality_test_datasets = T1_test_datasets + rsfMRI_test_datasets
    
    ### Building Dataset and setting batch sampler
    train_datasets = InterleaveDataset(All_modality_train_datasets, shuffle=True, seed=config.seed)
    eval_datasets = InterleaveDataset(All_modality_eval_datasets, shuffle=False, seed=config.seed)
    test_datasets = InterleaveDataset(All_modality_test_datasets, shuffle=False, seed=config.seed)


    #### setting model 
    #TODO: Change the source code of model!!!!!
    from model.Bblip_t5 import PatchEmbed
    from transformers import AutoTokenizer, Blip2ForConditionalGeneration
    model = Blip2ForConditionalGeneration.from_pretrained(config.model.hf_name)
    patch_embed = PatchEmbed(
            T1_size=config.dataset.T1.img_size, 
            T1_patch_size=config.model.T1.patch_size, 
            rsfMRI_size=config.dataset.rsfMRI.img_size, 
            rsfMRI_patch_size=config.model.rsfMRI.patch_size,
            in_chans=1, 
            embed_dim=int(model.vision_model.embeddings.patch_embedding.out_channels))
    setattr(model.vision_model, "embeddings", patch_embed)
    for name, param in model.vision_model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        if 'post_layernorm' in name:
            param.requires_grad = False
        if 'embeddings' in name:
            param.requires_grad = True


    # freeze Qformer
    for name, param in model.named_parameters():
        if 'qformer' in name:
            param.requires_grad = False
        if 'language_projection' in name:
            param.requires_grad = False
        if 'query_tokens' in name:
            param.requires_grad = False


    # freeze query token 
    for name, param in model.named_parameters():
        if 'language_model' in name:
            param.requires_grad = False
    #model.cuda()
    
    # set gradient checkpointing
    model.gradient_checkpointing_enable() 
    
    training_args = TrainingArguments(
        # basic settings
        run_name=os.environ["WANDB_RUN_ID"],
        output_dir=f'./hf_results/{os.environ["WANDB_RUN_ID"]}',          # output directory
        do_train=True, 
        do_eval=True,
        #eval_on_start=True,
        remove_unused_columns=False, # This shoud set as False so that input dictionary can have 'modality' as one of keys 
        # training
        num_train_epochs=config.trainer.max_epochs,              # total number of training epochs
        learning_rate=config.trainer.learning_rate,
        warmup_steps=config.trainer.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=config.trainer.weight_decay,               # strength of weight decay
        per_device_train_batch_size=config.trainer.per_device_batch_size,  # batch size per device during training
        per_device_eval_batch_size=config.trainer.per_device_batch_size,   # batch size for evaluation
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps, 
        # arguments for reducing memory (+ deepspeed zero3 offload)
        bf16=True,      # when using DDP deactivate this line
        bf16_full_eval=True,        # when using DDP deactivate this line
        gradient_checkpointing=config.trainer.gradient_checkpointing,
        # for evaluation and loggings
        report_to = 'wandb',
        logging_dir=f'./hf_logs/{os.environ["WANDB_RUN_ID"]}',            # directory for storing logs
        logging_steps=config.trainer.logging_steps,
        evaluation_strategy="epoch",    # In transformers==4.28.1 version, this argument is named as "evaluation_strategy" but in recent version (transformers==4.48.x) it is named as "eval_strategy"
        eval_accumulation_steps=1, # You should set this as 1 for avoiding memory leakage during evaluation
        disable_tqdm=False, 
        # checkpoint saving
        save_strategy="epoch",
        save_total_limit=3, 
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,    # It enable save checkpoint after evaluation (e.g., after epoch)


    )
    

    trainer = CustomTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer, 
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        #data_collator = CustomDataCollatorWithPadding(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,    # It shoud be used for preventing memory leakage during evaluation (link: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13)
        compute_metrics=compute_metrics_with_tokenizer(tokenizer=tokenizer),
        data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer,
                                                      padding=True,
                                                      max_length=128),
        model_optimization_type = 'joint',
    )




    # training 
    trainer.train()

    # test 
    trainer.evaluate(eval_dataset=test_datasets, metric_key_prefix='test')
    


if __name__ == '__main__': 
    __main__()


"""
##TODO 

1. multi-gpu 세팅에서 돌아가는지 확인해보기 => deepspeed 런처 쓰면 잘됨 
2. 학습이 잘되는지 확인하기 => 잘됨
4. blip2_flat_t5_xxl 학습하기 => deepspeed 런처 쓰면 잘됨
3. data module에서 batch size 지우고 configuration 파일에서 컨트롤 하도록 만들기
6. ['zero_optimization.reduce_bucket_size', 'zero_optimization.stage3_prefetch_bucket_size', 'zero_optimization.stage3_param_persistence_threshold'] => 얘네들 일일히 직접 입력할 필요 없게 만들기 (link에서 'hidden_size' 검색하고 참고: https://huggingface.co/docs/transformers/deepspeed) => 현재 hidden_size는 1408
7. checkpoint resume 에러 해결 
"""
