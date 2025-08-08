import datetime
import hashlib
from omegaconf import OmegaConf

import torch
import transformers
from transformers import Trainer, TrainingArguments
from utils.Trainer import CustomTrainer
from utils.Trainer import compute_metrics_with_tokenizer, preprocess_logits_for_metrics


from dataset.dataset_T1 import ABCD_T1, UKB_T1

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

    config = OmegaConf.load("./config/Brain_blip_t5_train_DeepSpeed.yaml") 

    ### setting logger 
    wandb.login(key=config.wandb.API_KEY)
    os.environ['WANDB_PROJECT'] = "BLIP_sMRI"
    os.environ["WANDB_RUN_ID"] = f'{hash_key}'


    ### setting seed 
    pl.seed_everything(config.seed)
    transformers.set_seed(config.seed)
    

    ### setting tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_name)
    
    
    ### setting dataset
    T1_train_datasets = [] 
    T1_val_datasets = [] 
    T1_test_datasets = [] 
    for study_sample, img_dir, meta_dir in zip(config.dataset.T1.study_sample, config.dataset.T1.img_dir, config.dataset.T1.meta_dir):
        if study_sample == "ABCD": 
            ABCD_dataset = ABCD_T1(tokenizer=tokenizer,config_dataset=config.dataset, img_dir=img_dir, meta_dir=meta_dir)
            # for train dataset
            T1_train_datasets.append(ABCD_dataset.train_dataset)
            # for val/test dataset 
            T1_val_datasets.append(ABCD_dataset.val_dataset)
            T1_test_datasets.append(ABCD_dataset.test_dataset)
        elif study_sample == 'UKB': 
            UKB_dataset = UKB_T1(tokenizer=tokenizer, config_dataset=config.dataset, img_dir=img_dir, meta_dir=meta_dir)
            # for train dataset
            T1_train_datasets.append(UKB_dataset.train_dataset)
            num_train_samples = UKB_dataset.train_dataset.__len__()

            # for val/test dataset 
            T1_val_datasets.append(UKB_dataset.val_dataset)
            T1_test_datasets.append(UKB_dataset.test_dataset)

    train_datasets = torch.utils.data.ConcatDataset(T1_train_datasets)
    val_datasets = torch.utils.data.ConcatDataset(T1_val_datasets)
    test_datasets = torch.utils.data.ConcatDataset(T1_test_datasets)

   

    #### setting model 
    from model.Bblip_t5 import PatchEmbed
    from transformers import AutoTokenizer, Blip2ForConditionalGeneration
    model = Blip2ForConditionalGeneration.from_pretrained(config.model.hf_name)
    patch_embed = PatchEmbed(
            T1_size=config.dataset.T1.img_size, 
            T1_patch_size=config.model.T1.patch_size, 
            in_chans=1, 
            embed_dim=int(model.vision_model.embeddings.patch_embedding.out_channels))
    setattr(model.vision_model, "embeddings", patch_embed)
    
    
    # load safetensors and generate state dict from checkpoint 
    state_dict = {}
    for filename in os.listdir(config.trainer.ckpt_dir):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(config.trainer.ckpt_dir, filename)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    
    # apply state dict (generated from safetensors) to pre-configured model 
    model.load_state_dict(state_dict, strict=False) 
    
    training_args = TrainingArguments(
        # basic settings
        run_name=os.environ["WANDB_RUN_ID"],
        output_dir=f'./hf_results/{os.environ["WANDB_RUN_ID"]}',          # output directory
        remove_unused_columns=False, # This shoud set as False so that input dictionary can have 'modality' as one of keys 
        # evaluation
        per_device_eval_batch_size=config.trainer.per_device_batch_size,   # batch size for evaluation
        #gradient_accumulation_steps=config.trainer.gradient_accumulation_steps, 
        # arguments for reducing memory (+ deepspeed zero3 offload)
        # for evaluation and loggings
        report_to = 'wandb',
        logging_dir=f'./hf_logs/{os.environ["WANDB_RUN_ID"]}',            # directory for storing logs
        logging_steps=config.trainer.logging_steps,
        eval_strategy="epoch",
        eval_accumulation_steps=1, # You should set this as 1 for avoiding memory leakage during evaluation
        disable_tqdm=False, 
    )
    

    trainer = CustomTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer, 
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        #data_collator = CustomDataCollatorWithPadding(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,    # It shoud be used for preventing memory leakage during evaluation (link: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13)
        compute_metrics=compute_metrics_with_tokenizer(tokenizer=tokenizer)
    )




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
4. Trainer argument configuration이랑 deepspeed configuration 정리하기. 
5. 코드에서 필요없는 argument 전부 빼기 
6. ['zero_optimization.reduce_bucket_size', 'zero_optimization.stage3_prefetch_bucket_size', 'zero_optimization.stage3_param_persistence_threshold'] => 얘네들 일일히 직접 입력할 필요 없게 만들기 (link에서 'hidden_size' 검색하고 참고: https://huggingface.co/docs/transformers/deepspeed) => 현재 hidden_size는 1408
7. Trainer 커스텀해서 중간 중간에 accuracy 뽑도록 만들어보기
"""