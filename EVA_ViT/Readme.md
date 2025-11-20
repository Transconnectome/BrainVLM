### Set conda environments 
```
conda activate swift
```

### Train ViT from scratch 
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex 
```


### Finetuning Pretrained ViT (Download from internet)
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --use_pretrained_weight
```

### Finetuning from your saved checkpoint
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --pretrained_model /path/to/your/checkpoint.pt
```

### Finetuning Pretrained ViT with additional projector layers
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --use_pretrained_weight --use_projector
```
