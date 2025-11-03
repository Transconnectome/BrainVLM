### Set conda environments 
```
conda activate swift
```

### Train ViT from scratch 
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --img_size 96 96 96 --patch_size 8
```


### Finetuning Pretrained ViT (w/ freeze encoder)
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --img_size 96 96 96 --patch_size 8 --use_pretrained_weight --freeze_encoder
```

### Finetuning Pretrained ViT (w/o freeze encoder)
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --img_size 96 96 96 --patch_size 8 --use_pretrained_weight
```


### Finetuning Pretrained ViT (w/o freeze encoder) with additional projector layers  
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --img_size 96 96 96 --patch_size 8 --use_pretrained_weight --use_projector
```

### Finetuning Pretrained ViT (w/ freeze encoder) with additional projector layers  
```
torchrun --standalone --nnodes=1 --nproc_per_node=1  main_EVA_ViT.py --model evavit_giant_patch16_3D --optim AdamW --lr 1e-4 --epoch 200 --exp_name test --batch_size 16 --accumulation_steps 16  --study_sample GARD_T1 --cat_target sex --img_size 96 96 96 --patch_size 8 --use_pretrained_weight --use_projector --freeze_encoder
```

