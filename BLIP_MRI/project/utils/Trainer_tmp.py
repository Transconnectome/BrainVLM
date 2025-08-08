import torch 
from transformers import Trainer
from sklearn.metrics import balanced_accuracy_score, f1_score
from dataclasses import dataclass



def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


@torch.no_grad()
def compute_metrics_with_tokenizer(tokenizer):
    def compute_metrics(pred):
        """
        tokenizer in this function is the input of the huggingface 'Trainer' argumet 'tokenizer' (Not the TrainingArguments)
        """

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = int(32001)
        pred_ids = pred.predictions[0]

        label_txt = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label = [0 if sex == 'male' else 1 for sex in label_txt]
        pred = [0 if sex == 'male' else 1 for sex in pred_txt]
        acc = balanced_accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')

        return {'accuracy': acc,
                'f1': f1,}
    return compute_metrics


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set static graph for DDP
        self._static_graph_set = False

    def _ensure_set_static_graph(self, model):
        if not self._static_graph_set and self.is_in_train:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model._set_static_graph()
            self._static_graph_set = True
    
    def _compute_modality_loss(self, model, inputs, labels=None):
        #"""Helper function to compute loss for a single modality"""
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = unwrapped_model.base_model.model._get_name() if _is_peft_model(unwrapped_model) else unwrapped_model._get_name()
            loss = self.label_smoother(outputs, labels, shift_labels=model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(f"Model did not return loss. Got keys: {','.join(outputs.keys())}")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return loss, outputs



    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        
        # Check gradients after backward
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item()}")
        

        return loss






    def compute_loss(self, model, inputs, return_outputs=False):
        self._ensure_set_static_graph(model)
        total_loss = 0.
        outputs = None
        modalities = list(inputs.keys())
        #base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        #def get_dummy_loss():
            #return sum(param.sum() * 0
                    #for param in base_model.vision_model.embeddings.parameters())

        # Cache parameter states to avoid reuse
        #with torch.no_grad():
        #    model_params = {name: param.clone() for name, param in model.named_parameters()}

        if len(modalities) == 1:
            modality = modalities[0]
            inputs_single = inputs[modality]
            skip_modality = 'rsfMRI' if modality == 'T1' else 'T1'
            
            # Simplified dummy loss computation
            #base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            #with torch.set_grad_enabled(False):
            #    dummy_params = [param for name, param in base_model.vision_model.embeddings.named_parameters() 
            #                  if skip_modality in name]
            #    dummy_loss = sum(param.detach().sum() * 0. for param in dummy_params)
            #base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            #dummy_loss = sum(param.sum() * 0. 
            #               for name, param in base_model.vision_model.embeddings.named_parameters() 
            #               if skip_modality in name)
            #dummy_loss = torch.zeros(1, device=model.device, requires_grad=True)
            dummy_loss = 0.
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                for name, param in model.module.vision_model.embeddings.named_parameters(): 
                    if f"{skip_modality}" in name: 
                        dummy_loss += param.sum()*0.
            else: 
                for name, param in model.vision_model.embeddings.named_parameters(): 
                    if f"{skip_modality}" in name: 
                        dummy_loss += param.sum()*0.

            labels = inputs_single.pop("labels") if self.label_smoother and "labels" in inputs_single else None
            loss, outputs = self._compute_modality_loss(model, inputs_single, labels)
            total_loss = dummy_loss + loss
        else:
            for modality in modalities:
                inputs_modality = inputs[modality]
                labels = inputs_modality.pop("labels") if self.label_smoother and "labels" in inputs_modality else None
                loss, outputs = self._compute_modality_loss(model, inputs_modality, labels)
                total_loss += loss / len(modalities)

        return (total_loss, outputs) if return_outputs else total_loss


class CustomTrainer_tmp(Trainer):
    """
    Modified based on https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L294
    """
    def _compute_modality_loss(self, model, inputs, labels=None):
        #"""Helper function to compute loss for a single modality"""
        outputs = model(**inputs)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            model_name = unwrapped_model.base_model.model._get_name() if _is_peft_model(unwrapped_model) else unwrapped_model._get_name()
            loss = self.label_smoother(outputs, labels, shift_labels=model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(f"Model did not return loss. Got keys: {','.join(outputs.keys())}")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        return loss, outputs
    def compute_loss(self, model, inputs, return_outputs=False):
        #"""
        #How the loss is computed by Trainer. By default, all models return the loss in the first element.

        #Subclass and override for custom behavior.

        #inputs = 
        #{
        #    'T1':
        #        {
        #        'pixel_values': torch.tensor([torch.tensor]),
        #        'input_ids': torch.tensor([torch.tensor]),
        #        'attention_mask': torch.tensor([torch.tensor]),
        #        'labels': torch.tensor([torch.tensor])
        #        },

        #    'rsfMRI':
        #        {
        #        'pixel_values': torch.tensor([torch.tensor]),
        #        'input_ids': torch.tensor([torch.tensor]),
        #        'attention_mask': torch.tensor([torch.tensor]),
        #        'labels': torch.tensor([torch.tensor])
        #        },

        #}
        
        #"""

        total_loss = 0.
        outputs = None
        modalities = list(inputs.keys())

        if len(modalities) == 1:
            modality = modalities[0]
            inputs_single = inputs[modality]
            skip_modality = 'rsfMRI' if modality == 'T1' else 'T1'
            
            #"""
            # Handle dummy loss
            dummy_loss = sum(param.sum() * 0. for name, param in 
                           (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) 
                            else model).vision_model.embeddings.named_parameters() 
                           if skip_modality in name)
            #"""
            dummy_loss = 0.
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                for name, param in model.module.vision_model.embeddings.named_parameters():
                    if f"{skip_modality}" in name:
                        dummy_loss += param.sum()*0.
                        #print(modality, skip_modality, name)
                #for name, param in model.module.language_model.named_parameters(): 
                #    dummy_loss += param.sum()*0.
            else:
                for name, param in model.vision_model.embeddings.named_parameters():
                    if f"{skip_modality}" in name:
                        dummy_loss += param.sum()*0.
                #for name, param in model.language_model.named_parameters():
                #    dummy_loss += param.sum()*0.

            # Handle labels
            labels = inputs_single.pop("labels") if self.label_smoother and "labels" in inputs_single else None
            loss, outputs = self._compute_modality_loss(model, inputs_single, labels)
            total_loss = dummy_loss + loss
        else:
            for modality in modalities:
                inputs_modality = inputs[modality]
                labels = inputs_modality.pop("labels") if self.label_smoother and "labels" in inputs_modality else None
                loss, outputs = self._compute_modality_loss(model, inputs_modality, labels)
                total_loss += loss    

        return (total_loss, outputs) if return_outputs else total_loss
    
    
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        #Check gradients after backward
        for name, param in model.named_parameters():
            if 'T1' in name:
                print(f"{name} grad norm: {param.grad.norm().item()}")


        return loss

