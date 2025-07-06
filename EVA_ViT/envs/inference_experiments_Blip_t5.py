import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler 


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

from model.model_Blip_t5 import Brain_BLIP

## ======= load module ======= ##

from util.utils import checkpoint_load



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
    accs = []

   
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            #images = images.to(f'cuda:{net.device_ids[0]}')
            images, labels = data 
            images = images.cuda()

            labels_txt = labels.tolist()
            labels_txt = ['male' if sex == 0 else 'female' for sex in labels_txt]
            prompt = "You are a neurologist and now you are analyzing T1-weighted MRI images. Question: If You are a neurologist and now you are analyzing T1-weighted MRI images, will this subject be classified as male of female? Answer: "
            inst = [prompt for _ in range(len(labels_txt))]
            
            batch = {} 
            batch['image'] = images 
            batch['text_input'] = inst 
            batch['text_output'] = labels_txt    

            if args.use_amp: 
                with torch.cuda.amp.autocast():
                    loss, loss_dict, pred = net(batch)
                losses.append(loss.item())
                acc = accuracy_score(labels.tolist(), pred)
                accs.append(acc)
            else: 
                loss, loss_dict, pred = net(batch)
                acc = accuracy_score(labels.tolist(), pred)
                accs.append(acc)
    return net, np.mean(losses), np.mean(accs)                           


def inference_engine(partition, num_classes, checkpoint_dir, args): #in_channels,out_dim
    # for hugging face tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    
    # loading model
    if args.model == 'blip2_t5xl':
        net = Brain_BLIP(model_arch="blip2_t5",
                        model_type="pretrain_flant5xl",
                        img_size=args.img_size[0],
                        lora_vit=False,
                        lora_llm=False,
                        )
    else: 
        raise ValueError("Only 'blip2_t5xl' is implemented")


    # loading the best model 
    net = checkpoint_load(net, checkpoint_dir, optimizer=None, scheduler=None, scaler=None, mode='inference')
    print('Inference start from the best model.')

    # attach network to cuda device. This line should come before wrapping the model with DDP 
    net.cuda()

    # do inference 
    result = {}
    net, test_loss, test_performance = inference(net, partition, num_classes, args)

    result['test_loss'] = test_loss
    result['test_acc'] = test_performance

    return vars(args), result
        

## ==================================== ##