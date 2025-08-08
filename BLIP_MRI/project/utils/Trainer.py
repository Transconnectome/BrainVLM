import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import datasets

from transformers import Trainer
from transformers.trainer_utils import has_length, seed_worker
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_datasets_available,
    is_sagemaker_mp_enabled,
    
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    nested_detach,
    IterableDatasetShard,

)

from sklearn.metrics import balanced_accuracy_score, f1_score
from dataclasses import dataclass


from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union



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
    """
    Modified based on https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L294
    """

    def __init__(self, model_optimization_type='sequential', *args, **kwargs):
        # Set static graph for DDP
        super().__init__(*args, **kwargs)
        self._static_graph_set = False
        self.model_optimization_type= model_optimization_type
        


    def _ensure_set_static_graph(self, model):
        if not self._static_graph_set and self.is_in_train:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model._set_static_graph()
            self._static_graph_set = True


    def repack_inputs_except_for_pixel_values(self, inputs, modalities): 
        """
        inputs = 
        {
            'T1':
                {
                'pixel_values': torch.tensor([torch.tensor]),
                'input_ids': torch.tensor([torch.tensor]),
                'attention_mask': torch.tensor([torch.tensor]),
                'labels': torch.tensor([torch.tensor])
                },

            'rsfMRI':
                {
                'pixel_values': torch.tensor([torch.tensor]),
                'input_ids': torch.tensor([torch.tensor]),
                'attention_mask': torch.tensor([torch.tensor]),
                'labels': torch.tensor([torch.tensor])
                },

        }

        outputs = 
        {
            'pixel_values': {
                            'T1': torch.tensor([torch.tensor]),
                            'rsfMRI':torch.tensor([torch.tensor]),
                            }
            'input_ids': torch.tensor([torch.tensor]),
            'attention_mask': torch.tensor([torch.tensor]),
            'labels': torch.tensor([torch.tensor]),

        }
        """
        assert len(modalities) > 1

        outputs = {} 
        outputs['pixel_values'] = {}
        outputs['input_ids'] = []
        outputs['attention_mask'] = []
        outputs['labels'] = [] 

        for modality in modalities: 
            modality_data = inputs[modality]
            #print(modality_data)
            outputs['pixel_values'][modality] = modality_data['pixel_values']
            outputs['input_ids'].append(modality_data['input_ids'])
            outputs['attention_mask'].append(modality_data['attention_mask'])
            outputs['labels'].append(modality_data['labels'])
        
        outputs['input_ids'] = torch.cat(outputs['input_ids'], dim=0)
        outputs['attention_mask'] = torch.cat(outputs['attention_mask'], dim=0)
        outputs['labels'] = torch.cat(outputs['labels'], dim=0)

        return outputs





    def _compute_modality_loss(self, model, inputs, labels=None):
        """Helper function to compute loss for a single modality"""
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
        #TODO 
        #현재 방식의 코드에서는 태생적으로 순차적으로 두개의 모달리티로부터 각각 loss를 얻어서 합한 loss로 최적화할 수가 없다. 
        #왜냐하면 한개의 모달리티로부터 Loss를 얻기 위해서는 patch layer를 제외한 나머지 layer들을 전부 거쳐야하는데, 이렇게 하고 나면 거쳐간 layer들을 업데이트하지 않은 상태에서 두번째 모달리티의 데이터가 이런 layer들을 거치게 되면서 backward()에서 에러가 발생한다. 
        #그런데 흥미로운 점은 x-instruct-BLIP 페이퍼에서는 다양한 모달리티로부터 얻은 Loss들을 joint optimization하지 않아도 multi-modal network를 학습할 수 있음을 보였다. 
        #다만, OneLLM은 애초에 라우팅하는 것을 특장점으로 삼았기 때문에 joint optimization을 한다 
        # joint optimization을 위해서는 BLIP2의 원래 코드를 짜고, 그 코드 위에다가 weight를 얹는 방식으로 진행해야할 것 같다.

        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        inputs = 
        {
            'T1':
                {
                'pixel_values': torch.tensor([torch.tensor]),
                'input_ids': torch.tensor([torch.tensor]),
                'attention_mask': torch.tensor([torch.tensor]),
                'labels': torch.tensor([torch.tensor])
                },

            'rsfMRI':
                {
                'pixel_values': torch.tensor([torch.tensor]),
                'input_ids': torch.tensor([torch.tensor]),
                'attention_mask': torch.tensor([torch.tensor]),
                'labels': torch.tensor([torch.tensor])
                },

        }
        
        """
        self._ensure_set_static_graph(model)
        total_loss = 0.
        outputs = None
        modalities = list(inputs.keys())

        #assert len(modalities) == 1

        if len(modalities) == 1:    # in case of sequential model optimization
            modality = modalities[0]
            inputs_single = inputs[modality].copy()
            skip_modality = 'rsfMRI' if modality == 'T1' else 'T1'
            
            # Handle dummy loss
            dummy_loss = 0.
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                for name, param in model.module.vision_model.embeddings.named_parameters(): 
                    if f"{skip_modality}" in name: 
                        dummy_loss += param.sum()*0.

            else: 
                for name, param in model.vision_model.embeddings.named_parameters(): 
                    if f"{skip_modality}" in name: 
                        dummy_loss += param.sum()*0.  
            """
            dummy_loss = sum(param.sum() * 0. for name, param in 
                           (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) 
                            else model).vision_model.embeddings.named_parameters() 
                           if skip_modality in name)
            """
            
            # Handle labels
            labels = inputs_single.pop("labels") if self.label_smoother and "labels" in inputs_single else None
            loss, outputs = self._compute_modality_loss(model, inputs_single, labels)
            total_loss = dummy_loss + loss

        
        if len(modalities) == 2:    # in case of joint model optimization
            #print(modalities)
            inputs = self.repack_inputs_except_for_pixel_values(inputs, modalities)
            labels = inputs.pop("labels") if self.label_smoother and "labels" in inputs else None
            loss, outputs = self._compute_modality_loss(model, inputs, labels)
            total_loss += loss
        
        return (total_loss, outputs) if return_outputs else total_loss
   

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        # Log gradients at logging steps
        modalities = list(inputs.keys())
        if len(modalities) == 1:
            if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
                grad_norms = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if modalities[0] in name:
                            if 'bias' in name: 
                                continue
                            else:
                                grad_norms[f"grad/{name}"] = param.grad.norm().item()
        
        else: 
            if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
                grad_norms = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if 'bias' in name: 
                                continue
                        else:
                            grad_norms[f"grad/{name}"] = param.grad.norm().item()

        # Log to loggers through trainer's log() method
        self.log(grad_norms)


        """
        # Check gradients after backward
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm().item()}")
        """

        return loss
    

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        # Custom Part
    


        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
        
