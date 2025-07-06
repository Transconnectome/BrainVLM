import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler 


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

import model.model_EvaViT as EvaViT

## ======= load module ======= ##

from util.utils import CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, saving_outputs, set_random_seed, load_imagenet_pretrained_weight
from util.optimizers import LAMB, LARS 
from util.lr_sched import CosineAnnealingWarmUpRestarts
from util.loss_functions  import loss_forward, mixup_loss, calculating_eval_metrics
from util.augmentation import mixup_data


import time
from tqdm import tqdm
import copy


def inference(net, partition, num_classes, args): 
    testloader = torch.utils.data.DataLoader(partition['test'],
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=16)

    net.eval()

    
    losses = []
    if args.mixup:
        loss_fn = mixup_loss(num_classes)
    else:
        loss_fn = loss_forward(num_classes)

    eval_metrics = calculating_eval_metrics(num_classes=num_classes, is_DDP=False)    
    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader),0):
            #images = images.to(f'cuda:{net.device_ids[0]}')
            images, labels = data 
            labels = labels.cuda()
            images = images.cuda()
            if args.mixup:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels)
                with torch.cuda.amp.autocast():
                    pred = net(mixed_images)
                    loss = loss_fn(pred, labels_a, labels_b, lam)
            else:
                with torch.cuda.amp.autocast():
                    pred = net(images)
                    loss = loss_fn(pred, labels)
            losses.append(loss.item())
            eval_metrics.store(pred, labels)
                           
    return net, np.mean(losses), eval_metrics.get_result() 


def inference_engine(partition, num_classes, checkpoint_dir, args): #in_channels,out_dim

    # setting network 
    if args.model.find('evavit_') != -1:
        net = EvaViT.__dict__[args.model](img_size = args.img_size[0], num_classes=num_classes, use_lora=args.use_lora)
    elif args.model.find('evavit2_') != -1:
        #net = EvaViT.__dict__[args.model](img_size = args.img_size[0], num_classes=num_classes, use_projector=True, use_lora=args.use_lora)
        net = EvaViT.__dict__[args.model](img_size = args.img_size[0], patch_size=14,num_classes=num_classes, use_lora=args.use_lora)

    # loading the best model 
    net = checkpoint_load(net, checkpoint_dir, optimizer=None, scheduler=None, scaler=None, mode='inference')
    print('Inference start from the best model.')

    # attach network to cuda device. This line should come before wrapping the model with DDP 
    net.cuda()

    # do inference 
    result = {}
    net, test_loss, test_performance = inference(net, partition, num_classes, args)

    result['test_loss'] = test_loss
    result.update(test_performance)

    return vars(args), result
        

## ==================================== ##