import numpy as np
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler 
import numpy as np 
import loralib as lora


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score

## ======= load module ======= ##



from util.utils import CLIreporter, save_exp_result, set_checkpoint_dir, checkpoint_save, checkpoint_load, saving_outputs, set_random_seed, load_imagenet_pretrained_weight
from util.optimizers import LAMB, LARS 
from util.lr_sched import CosineAnnealingWarmUpRestarts
from util.loss_functions  import loss_forward, mixup_loss, calculating_eval_metrics
from util.augmentation import mixup_data


import time
from tqdm import tqdm
import copy


def model_train(net, partition, optimizer, scaler, epoch, num_classes, args):
    train_sampler = DistributedSampler(partition['train'], shuffle=True)    

    trainloader = torch.utils.data.DataLoader(partition['train'],
                                             batch_size=args.batch_size,
                                             sampler=train_sampler, 
                                             shuffle=False,
                                             drop_last = False,
                                             num_workers=8)

    net.train()
    trainloader.sampler.set_epoch(epoch)

    losses = []
    accs = []

    
    optimizer.zero_grad()
    for i, data in enumerate(trainloader,0):
        ################################################
        ################################################
        #################### TO DO #####################
        # loss_fn = cross entropy or MSE 
        # change pred, target, mask = net(images) => latent, cls_tokens = net(images)
        images, labels = data 
        images = images.cuda()

        labels_txt = labels.tolist()
        ### for sex classification
        labels_txt = ['male' if sex == 0 else 'female' for sex in labels_txt]
        prompt = "You are a neurologist and now you are analyzing T1-weighted MRI images. Question:  Analyze the image and estimate sex of subject from this image. Answer with 'male' or 'female'. Answer: "
        ### for regression
        #labels_txt = ["{:.1f}".format(value) for value in labels_txt]
        #prompt = "You are a neurologist and now you are analyzing T1-weighted MRI images. Question: Analyze the image and estimate the BMI standard deviation score of the child. Answer with number only without any additional text. Answer: "
        ### for weight classification 
        #prompt = "You are a neurologist and now you are analyzing T1-weighted MRI images. Question: Analyze the image and estimate the weight status of the children. Answer with 'underweight', 'normal', or 'overweight'. Answer: "
        #labels_txt = labels_txt
        inst = [prompt for _ in range(len(labels_txt))]

        
        batch = {} 
        batch['image'] = images 
        batch['text_input'] = inst 
        batch['text_output'] = labels_txt        
        
        """
        if loss is calculated inside the model class, the output from the model forward method would be [loss] * number of devices. In other words, len(net(images)) == n_gpus
        """
        #loss, pred, mask  = net(images)
        
        if args.use_amp: 
            with torch.cuda.amp.autocast():
                loss, loss_dict, pred = net(batch)
                print(pred)
            
            losses.append(loss.item())
            
            try:
                ### for sex classification
                acc = accuracy_score(labels.tolist(), pred)
                ### for regression
                #acc = nn.functional.mse_loss(torch.tensor(pred), torch.tensor([float(value) for value in batch['text_output']])).item()
                #acc = r2_score(torch.tensor([float(value) for value in batch['text_output']]), torch.tensor(pred))
                
            except: 
                acc = np.nan
                #print("Answers still contain additional text.")
            accs.append(acc)
            
            

            assert args.accumulation_steps >= 1
            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()   # pytorch 1.xx 
            #scaler.scale(loss).sum().backward()    # pytorch 2.xx
            if  (i + 1) % args.accumulation_steps == 0:
                # gradient clipping 
                if args.gradient_clipping == True:  
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, error_if_nonfinite=False)   # max_norm=1 from https://arxiv.org/pdf/2010.11929.pdf
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()       
                #print(f"GT:{batch['text_output'][:4]}\nPRED{pred[:4]}")
        else: 
            loss, loss_dict, pred = net(batch)        
            
            losses.append(loss.item())

            try:
                ### for sex classification
                #acc = accuracy_score(labels.tolist(), pred)
                ### for age regression
                #acc = nn.functional.mse_loss(torch.tensor(pred), torch.tensor([float(value) for value in batch['text_output']])).item()
                acc = r2_score(torch.tensor([float(value) for value in batch['text_output']]), torch.tensor(pred))
            except: 
                acc = np.nan
                #print("Answers still contain additional text.")
            accs.append(acc)

            assert args.accumulation_steps >= 1
            loss = loss / args.accumulation_steps
            loss.backward()   # pytorch 1.xx 
            #scaler.scale(loss).sum().backward()    # pytorch 2.xx
            if  (i + 1) % args.accumulation_steps == 0:
                # gradient clipping 
                if args.gradient_clipping == True:  
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, error_if_nonfinite=False)   # max_norm=1 from https://arxiv.org/pdf/2010.11929.pdf
                optimizer.step()
                optimizer.zero_grad()      

            

    return net, np.mean(losses), np.nanmean(accs)
        

def model_validation(net, partition, epoch, num_classes, args):
    val_sampler = DistributedSampler(partition['val'])  
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            sampler=val_sampler, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=8)

    net.eval()
    valloader.sampler.set_epoch(epoch)
    
    losses = []
    accs = []

   
    with torch.no_grad():
        for i, data in enumerate(valloader,0):
            #images = images.to(f'cuda:{net.device_ids[0]}')
            images, labels = data 
            images = images.cuda()

            labels_txt = labels.tolist()
            ### for sex classification
            labels_txt = ['male' if sex == 0 else 'female' for sex in labels_txt]
            prompt = "You are a neurologist and now you are analyzing T1-weighted MRI images. Question:  Analyze the image and estimate sex of subject from this image. Answer with 'male' or 'female'. Answer: "
            ### for regression
            #labels_txt = ["{:.1f}".format(value) for value in labels_txt]
            #prompt = "You are a neurologist and now you are analyzing T1-weighted MRI images. Question: Analyze the image and estimate the BMI standard deviation score of the child. Answer with number only without any additional text. Answer: "
            ### for weight classification 
            #prompt = "You are a neurologist and now you are analyzing T1-weighted MRI images. Question: Analyze the image and estimate the weight status of the children. Answer with 'underweight', 'normal', or 'overweight'. Answer: "
            #labels_txt = labels_txt
            
            inst = [prompt for _ in range(len(labels_txt))]

            batch = {} 
            batch['image'] = images 
            batch['text_input'] = inst 
            batch['text_output'] = labels_txt    

            if args.use_amp: 
                with torch.cuda.amp.autocast():
                    loss, loss_dict, pred = net(batch)
                
                losses.append(loss.item())
                
                try:
                    ### for sex classification
                    acc = accuracy_score(labels.tolist(), pred)
                    ### for regression
                    #acc = nn.functional.mse_loss(torch.tensor(pred), torch.tensor([float(value) for value in batch['text_output']])).item()
                    #acc = r2_score(torch.tensor([float(value) for value in batch['text_output']]), torch.tensor(pred))
                except: 
                    acc = np.nan
                #    print("Answers still contain additional text.")
                accs.append(acc)
            else: 
                loss, loss_dict, pred = net(batch)
                
                try:
                    ### for sex classification
                    #acc = accuracy_score(labels.tolist(), pred)
                    ### for age regression
                    #acc = nn.functional.mse_loss(torch.tensor(pred), torch.tensor([float(value) for value in batch['text_output']])).item()
                    acc = r2_score(torch.tensor([float(value) for value in batch['text_output']]), torch.tensor(pred))
                except: 
                    acc = np.nan
                accs.append(acc)
            #if i == 0:
            #    print(f"GT:{batch['text_output'][:4]}\nPRED{pred[:4]}")
    return net, np.mean(losses), np.nanmean(accs)


def train_experiment(partition, num_classes, save_dir, args): #in_channels,out_dim
    if args.model == 'blip2_opt2.7b':
        from model.model_Blip_opt import Brain_BLIP
        net = Brain_BLIP(model_arch="blip2_opt",
                        model_type="pretrain_opt2.7b",
                        img_size=args.img_size[0],
                        lora_vit=False,
                        lora_llm=False,
                        )
    elif args.model == 'blip2_opt6.7b':
        from model.model_Blip_opt import Brain_BLIP
        net = Brain_BLIP(model_arch="blip2_opt",
                        model_type="pretrain_opt6.7b",
                        img_size=args.img_size[0],
                        lora_vit=False,
                        lora_llm=False,
                        )

    else: 
        raise ValueError("Only 'blip2_t5xl' is implemented")
    checkpoint_dir = set_checkpoint_dir(save_dir=save_dir, args=args)

    # setting optimizer 
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=0,weight_decay=args.weight_decay)
    elif args.optim == 'LARS':
        optimizer = LARS(net.parameters(), lr=0, momentum=0.9)
    elif args.optim == 'LAMB':
        optimizer = LAMB(net.parameters(), lr=0, weight_decay=args.weight_decay)        
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=0, weight_decay=args.weight_decay,betas=(0.9, 0.95))
    else:
        raise ValueError('In-valid optimizer choice')

    # setting learning rate scheduler 
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', patience=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.5)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0, epochs=args.epoch)
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.epoch, T_mult=2, eta_max=args.lr,  T_up=args.warmup_epoch, gamma=0.5)
    scheduler = None
    

    # setting AMP gradient scaler 
    scaler = torch.cuda.amp.GradScaler()
    

    if args.resume == False:
        last_epoch = 0
        previous_performance = None
    elif args.resume == True:  # loading last checkpoint 
        if args.checkpoint_dir != None:
            net, optimizer, scheduler, last_epoch, optimizer.param_groups[0]['lr'], scaler, previous_performance = checkpoint_load(net, checkpoint_dir, optimizer, scheduler, scaler, mode='pretrain')
            print('Training start from epoch {} and learning rate {}.'.format(last_epoch, optimizer.param_groups[0]['lr']))
        else: 
            raise ValueError('IF YOU WANT TO RESUME TRAINING FROM PREVIOUS STATE, YOU SHOULD SET THE FILE PATH AS AN OPTION. PLZ CHECK --checkpoint_dir OPTION')
        
    # attach network to cuda device. This line should come before wrapping the model with DDP 
    net.cuda()

    # setting DataParallel
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.gpu], find_unused_parameters=True)
    # pytorch 2.0
    #torch._dynamo.config.suppress_errors = True
    #net = torch.compile(net)
    
    # attach optimizer to cuda device.
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # for hugging face tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
   
    # setting for results' data frame
    train_losses = []
    val_losses = []
    if previous_performance is not None: 
        val_losses = [previous_performance[i]for i in range(len(previous_performance))]

    # training
    for epoch in tqdm(range(last_epoch, last_epoch + args.epoch)):
        ts = time.time()
        net, train_loss, train_performance = model_train(net, partition, optimizer, scaler, epoch, num_classes, args)
        net, val_loss, val_performance = model_validation(net, partition, epoch, num_classes, args)
        
        #scheduler.step(loss)
        if scheduler is not None:
            scheduler.step()
        te = time.time()

        # visualize the result and saving the checkpoint
        # saving model. When use DDP, if you do not indicate device ids, the number of saved checkpoint would be the same as the number of process.
        if args.gpu == 0:
            # store result per epoch 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('Epoch {}. Train Loss: {:2.2f}. Validation Loss: {:2.2f}. \n Training Prediction Performance: {}. \n Validation Prediction Performance: {}. \n Current learning rate {}. Took {:2.2f} sec'.format(epoch+1, train_loss, val_loss, train_performance, val_performance, optimizer.param_groups[0]['lr'],te-ts))

            if epoch == 0: 
                checkpoint_save(net, optimizer, checkpoint_dir, epoch, scheduler, scaler, args, val_losses,mode='pretrain')
            else:
                if val_loss <= val_losses[np.argmax(val_losses[:-1])]: 
                    checkpoint_save(net, optimizer, checkpoint_dir, epoch, scheduler, scaler, args, val_losses,mode='pretrain')

            
        torch.cuda.empty_cache()
            

    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['validation_losses'] = val_losses

    return vars(args), result, checkpoint_dir
        

## ==================================== ##