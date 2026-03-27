import os
import json
import numpy as np

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
    if isinstance(logits, tuple):
        logits = logits[0]

    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

# @torch.no_grad()
def compute_metrics_with_tokenizer(tokenizer, targets):
    """
    Automatically compute metrics based on target types.
    Categorical: sex -> accuracy, f1
    Numerical: age, bmi, glucose -> MAE, RMSE
    """
    import re
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = {}
        total_samples = len(decoded_preds)

        # Check if single-task or multi-task
        is_single_task = len(targets) == 1

        # Determine task type
        task_name = targets[0] if is_single_task else None
        is_sex_task = task_name and 'sex' in task_name.lower()

        # Check if numerical (simple heuristic: common numerical task names)
        numerical_keywords = ['age', 'bmi', 'glucose', 'weight', 'height', 'score']
        is_numerical_task = task_name and any(kw in task_name.lower() for kw in numerical_keywords)

        if is_single_task and is_sex_task:
            # Sex classification task
            pred_genders = []
            true_genders = []

            for pred in decoded_preds:
                pred_clean = pred.lower().strip()
                if re.search(r'\bfemale\b', pred_clean):
                    pred_genders.append(1)
                elif re.search(r'\bmale\b', pred_clean):
                    pred_genders.append(0)
                else:
                    pred_genders.append(-1)

            for label in decoded_labels:
                label_clean = label.lower().strip()
                if re.search(r'\bfemale\b', label_clean):
                    true_genders.append(1)
                elif re.search(r'\bmale\b', label_clean):
                    true_genders.append(0)
                else:
                    true_genders.append(-1)

            # Valid pairs (only for metrics on valid predictions)
            valid_pairs = [(p, t) for p, t in zip(pred_genders, true_genders) if p != -1 and t != -1]

            if valid_pairs:
                valid_preds, valid_trues = zip(*valid_pairs)
                valid_accuracy = balanced_accuracy_score(valid_trues, valid_preds)
                valid_f1 = f1_score(valid_trues, valid_preds, average='macro')
            else:
                valid_accuracy = 0.0
                valid_f1 = 0.0

            # Overall metrics (treat invalid as wrong answer)
            overall_preds = []
            overall_trues = []

            for p, t in zip(pred_genders, true_genders):
                if t != -1:  # Only when ground truth is valid
                    overall_trues.append(t)
                    if p == -1:
                        # Treat invalid as wrong (flip the answer)
                        overall_preds.append(1 - t)
                    else:
                        overall_preds.append(p)

            if overall_preds:
                overall_accuracy = balanced_accuracy_score(overall_trues, overall_preds)
                overall_f1 = f1_score(overall_trues, overall_preds, average='macro')
            else:
                overall_accuracy = 0.0
                overall_f1 = 0.0

            invalid_predictions = pred_genders.count(-1)
            response_rate = (total_samples - invalid_predictions) / total_samples if total_samples > 0 else 0

            metrics = {
                'accuracy': valid_accuracy,
                'f1': valid_f1,
                'overall_accuracy': overall_accuracy,
                'overall_f1': overall_f1,
                'response_rate': response_rate,
                'valid_samples': len(valid_pairs),
                'total_samples': total_samples,
                'invalid_predictions': invalid_predictions
            }

        elif is_single_task and is_numerical_task:
            # Single numerical task (e.g., age, bmi, glucose)
            task_name = targets[0]
            pred_values = []
            true_values = []

            for pred in decoded_preds:
                pred_clean = pred.strip()
                # Extract first number
                match = re.search(r'(\d+\.?\d*)', pred_clean)
                if match:
                    pred_values.append(float(match.group(1)))
                else:
                    pred_values.append(-1)

            for label in decoded_labels:
                label_clean = label.strip()
                match = re.search(r'(\d+\.?\d*)', label_clean)
                if match:
                    true_values.append(float(match.group(1)))
                else:
                    true_values.append(-1)

            # Valid pairs
            valid_pairs = [(p, t) for p, t in zip(pred_values, true_values) if p != -1 and t != -1]

            if valid_pairs:
                valid_preds, valid_trues = zip(*valid_pairs)
                mae = mean_absolute_error(valid_trues, valid_preds)
                rmse = np.sqrt(mean_squared_error(valid_trues, valid_preds))
            else:
                mae = 0.0
                rmse = 0.0

            invalid_predictions = pred_values.count(-1)
            response_rate = (total_samples - invalid_predictions) / total_samples if total_samples > 0 else 0

            metrics = {
                f'{task_name}_mae': mae,
                f'{task_name}_rmse': rmse,
                'response_rate': response_rate,
                'valid_samples': len(valid_pairs),
                'total_samples': total_samples,
                'invalid_predictions': invalid_predictions
            }

        elif is_single_task:
            # Other categorical tasks (not sex)
            # Extract unique labels from ground truth
            print(f"[INFO] Generic categorical task detected: {task_name}")
            print(f"[INFO] Extracting labels from ground truth...")

            # First pass: collect all possible labels from ground truth
            all_labels = set()
            for label in decoded_labels:
                label_clean = label.lower().strip()
                # Try to extract label after common patterns
                # Pattern 1: "appears to be X"
                match = re.search(r'appears to be\s+(\w+)', label_clean)
                if match:
                    all_labels.add(match.group(1))
                # Pattern 2: "is X"
                elif re.search(r'\bis\s+(\w+)', label_clean):
                    match = re.search(r'\bis\s+(\w+)', label_clean)
                    all_labels.add(match.group(1))
                # Pattern 3: just the label itself (e.g., "control", "patient")
                else:
                    words = label_clean.split()
                    if len(words) > 0:
                        all_labels.add(words[-1])  # Take last word as label

            # Create label to idx mapping
            label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
            print(f"[INFO] Detected labels: {label_to_idx}")

            pred_values = []
            true_values = []

            for pred in decoded_preds:
                pred_clean = pred.lower().strip()
                found = False
                for label_text in label_to_idx.keys():
                    if re.search(rf'\b{label_text}\b', pred_clean):
                        pred_values.append(label_to_idx[label_text])
                        found = True
                        break
                if not found:
                    pred_values.append(-1)

            for label in decoded_labels:
                label_clean = label.lower().strip()
                found = False
                for label_text in label_to_idx.keys():
                    if re.search(rf'\b{label_text}\b', label_clean):
                        true_values.append(label_to_idx[label_text])
                        found = True
                        break
                if not found:
                    true_values.append(-1)

            # Valid pairs
            valid_pairs = [(p, t) for p, t in zip(pred_values, true_values) if p != -1 and t != -1]

            if valid_pairs:
                valid_preds, valid_trues = zip(*valid_pairs)
                accuracy = balanced_accuracy_score(valid_trues, valid_preds)
                f1 = f1_score(valid_trues, valid_preds, average='macro')
            else:
                accuracy = 0.0
                f1 = 0.0

            invalid_predictions = pred_values.count(-1)
            response_rate = (total_samples - invalid_predictions) / total_samples if total_samples > 0 else 0

            metrics = {
                'accuracy': accuracy,
                'f1': f1,
                'response_rate': response_rate,
                'valid_samples': len(valid_pairs),
                'total_samples': total_samples,
                'invalid_predictions': invalid_predictions
            }

        # else:
        #     # Multi-task
        #     for task_name in targets:
        #         if task_name in categorical_tasks:
        #             # Categorical task
        #             pattern = rf'{task_name}:\s*(\w+)'
        #             pred_values = []
        #             true_values = []

        #             for pred in decoded_preds:
        #                 pred_clean = pred.lower().strip()
        #                 match = re.search(pattern, pred_clean)
        #                 if match:
        #                     label_text = match.group(1)
        #                     if label_text in categorical_tasks[task_name]:
        #                         pred_values.append(categorical_tasks[task_name][label_text])
        #                     else:
        #                         pred_values.append(-1)
        #                 else:
        #                     pred_values.append(-1)

        #             for label in decoded_labels:
        #                 label_clean = label.lower().strip()
        #                 match = re.search(pattern, label_clean)
        #                 if match:
        #                     label_text = match.group(1)
        #                     if label_text in categorical_tasks[task_name]:
        #                         true_values.append(categorical_tasks[task_name][label_text])
        #                     else:
        #                         true_values.append(-1)
        #                 else:
        #                     true_values.append(-1)

        #             valid_pairs = [(p, t) for p, t in zip(pred_values, true_values) if p != -1 and t != -1]

        #             if valid_pairs:
        #                 valid_preds, valid_trues = zip(*valid_pairs)
        #                 accuracy = balanced_accuracy_score(valid_trues, valid_preds)
        #                 f1 = f1_score(valid_trues, valid_preds, average='macro')
        #             else:
        #                 accuracy = 0.0
        #                 f1 = 0.0

        #             metrics[f'{task_name}_accuracy'] = accuracy
        #             metrics[f'{task_name}_f1'] = f1

        #         else:
        #             # Numerical task
        #             pattern = rf'{task_name}:\s*(\d+\.?\d*)'
        #             pred_values = []
        #             true_values = []

        #             for pred in decoded_preds:
        #                 pred_clean = pred.strip()
        #                 match = re.search(pattern, pred_clean)
        #                 if match:
        #                     pred_values.append(float(match.group(1)))
        #                 else:
        #                     pred_values.append(-1)

        #             for label in decoded_labels:
        #                 label_clean = label.strip()
        #                 match = re.search(pattern, label_clean)
        #                 if match:
        #                     true_values.append(float(match.group(1)))
        #                 else:
        #                     true_values.append(-1)

        #             valid_pairs = [(p, t) for p, t in zip(pred_values, true_values) if p != -1 and t != -1]

        #             if valid_pairs:
        #                 valid_preds, valid_trues = zip(*valid_pairs)
        #                 mae = mean_absolute_error(valid_trues, valid_preds)
        #                 rmse = np.sqrt(mean_squared_error(valid_trues, valid_preds))
        #             else:
        #                 mae = 0.0
        #                 rmse = 0.0

        #             metrics[f'{task_name}_mae'] = mae
        #             metrics[f'{task_name}_rmse'] = rmse

        #     # Overall response rate
        #     all_pred_values = []
        #     for pred in decoded_preds:
        #         valid = True
        #         for task_name in targets:
        #             if task_name in categorical_tasks:
        #                 pattern = rf'{task_name}:\s*(\w+)'
        #             else:
        #                 pattern = rf'{task_name}:\s*(\d+\.?\d*)'
        #             if not re.search(pattern, pred.lower()):
        #                 valid = False
        #                 break
        #         all_pred_values.append(1 if valid else -1)

        #     invalid_predictions = all_pred_values.count(-1)
        #     response_rate = (total_samples - invalid_predictions) / total_samples if total_samples > 0 else 0

        #     metrics['response_rate'] = response_rate
        #     metrics['total_samples'] = total_samples
        #     metrics['invalid_predictions'] = invalid_predictions

        return metrics

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


    def _compute_dummy_gradient(self, model, active_modality):
        """Compute dummy gradient for inactive modality parameters."""
        skip_modality = 'rsfMRI' if active_modality == 'T1' else 'T1'

        # Get embeddings module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            base_model = model.module
        else:
            base_model = model

        embeddings = (base_model.vision_tower.vision_model.embeddings
                    if hasattr(base_model, 'vision_tower')
                    else base_model.vision_model.embeddings) # vision_tower is for LLaVA

        # Compute dummy loss
        dummy_loss = 0.
        for name, param in embeddings.named_parameters():
            if skip_modality in name:
                dummy_loss += param.sum() * 0.

        return dummy_loss


    def _compute_loss_with_labels(self, model, inputs):
        """Compute loss handling both label_smoother and direct cases."""
        # Extract labels if using label smoother
        if self.label_smoother and "labels" in inputs:
            labels = inputs.pop("labels")
            return self._compute_modality_loss(model, inputs, labels)

        outputs = model(**inputs)

        # Extract loss from various output formats
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            loss = outputs[0]
        else:
            raise ValueError(f"Model did not return a loss. Output type: {type(outputs)}")

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

        if len(modalities) == 1:
            # Single modality: add dummy gradient for stability
            modality = modalities[0]
            inputs_single = inputs[modality].copy()

            # Dummy loss for unused modality parameters
            dummy_loss = self._compute_dummy_gradient(model, modality)

            # Compute actual loss
            loss, outputs = self._compute_loss_with_labels(model, inputs_single)
            total_loss = dummy_loss + loss

        else:  # len(modalities) >= 2
            # Multiple modalities: repack and compute
            inputs_repacked = self.repack_inputs_except_for_pixel_values(inputs, modalities)
            loss, outputs = self._compute_loss_with_labels(model, inputs_repacked)
            total_loss = loss

        return (total_loss, outputs) if return_outputs else total_loss


    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        # generation result
        if self.state.global_step % 50 == 0 and self.state.global_step > 0:
            self.log_generated_result(model, inputs, mode="training")

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

        modalities = list(inputs.keys())
        if len(modalities) == 1 and modalities[0] in ['T1', 'rsfMRI']:
            inputs = inputs[modalities[0]]
        elif len(modalities) > 1:
            inputs = self.repack_inputs_except_for_pixel_values(inputs, modalities)

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
                        if len(modalities) == 1 and modalities[0] in ['T1', 'rsfMRI']: # do we need this logic
                            wrapped_inputs = {modalities[0]: inputs}
                            loss, outputs = self.compute_loss(model, wrapped_inputs, return_outputs=True)
                        else:
                            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

                    if loss is not None:
                        if isinstance(loss, torch.Tensor):
                            loss = loss.mean().detach()
                        else:
                            loss = torch.tensor(loss)

                    if isinstance(outputs, dict):
                        # LLaVA
                        logits = outputs.get('logits', None)
                        if logits is None:
                            # fallback
                            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                            if len(logits) == 1:
                                logits = logits[0]
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs[1:] if len(outputs) > 1 else None
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', None)
                        if logits is None:
                            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs

                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0 and hasattr(outputs, '__getitem__'):
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        if logits is not None:
            logits = nested_detach(logits)
            if isinstance(logits, (tuple, list)) and len(logits) == 1:
                logits = logits[0]

        # Log generated result during evaluation (first sample of each eval)
        if not prediction_loss_only and not hasattr(self, '_eval_generation_logged'):
            self._eval_generation_logged = True
            self.log_generated_result(model, inputs, mode="evaluation")

        return (loss, logits, labels)


    def log_generated_result(self, model, inputs, mode="training"):
        """
        Log generated result during training or evaluation

        Args:
            model: The model to use for generation
            inputs: Input dictionary (wrapped or unwrapped)
            mode: "training" or "evaluation"
        """
        actual_model = model.module if hasattr(model, 'module') else model

        # Only set eval mode for training (already in eval during evaluation)
        if mode == "training":
            actual_model.eval()

        with torch.no_grad():
            try:
                # Handle input format (different for training vs evaluation)
                if 'pixel_values' in inputs and 'input_ids' in inputs:
                    sample_input = inputs
                else:
                    # Still wrapped in modality key (typical for training)
                    modality_keys = [k for k in inputs.keys() if k in ['T1', 'rsfMRI']]
                    if modality_keys:
                        sample_input = inputs[modality_keys[0]]
                    else:
                        sample_input = inputs

                # Get first sample from batch
                input_ids = sample_input['input_ids'][0]

                # Handle pixel_values (supports both single-image and multi-image)
                pixel_values = sample_input['pixel_values']
                if len(pixel_values.shape) == 6:
                    # Multi-image: [batch, num_images, C, D, H, W] -> take first batch
                    pixel_values_sample = pixel_values[0:1]  # [1, num_images, C, D, H, W]
                elif len(pixel_values.shape) == 5:
                    # Single-image: [batch, C, D, H, W] -> take first batch
                    pixel_values_sample = pixel_values[0:1]  # [1, C, D, H, W]
                else:
                    print(f"[WARN] Unexpected pixel_values shape: {pixel_values.shape}")
                    return

                # Search for LAST assistant token (multi-turn: we want to generate the final answer)
                # Conversation structure:
                # Turn 1: user <image> → assistant (acknowledgment)
                # Turn 2: user <image> → assistant (answer) ← We generate this!
                assistant_variants = ["<|im_start|>assistant\n", "<|im_start|>assistant"]
                assistant_positions = []

                for variant in assistant_variants:
                    assistant_tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                    for i in range(len(input_ids) - len(assistant_tokens)):
                        if torch.equal(input_ids[i:i+len(assistant_tokens)],
                                    torch.tensor(assistant_tokens, device=input_ids.device)):
                            assistant_positions.append(i + len(assistant_tokens))

                if len(assistant_positions) == 0:
                    print(f"[WARN] Assistant token not found in {mode} input")
                    return

                # Use LAST assistant position (for multi-turn, this is the final answer)
                last_assistant_pos = assistant_positions[-1]
                prompt_ids = input_ids[:last_assistant_pos].unsqueeze(0)

                # Generate
                generated_ids = actual_model.generate(
                    pixel_values=pixel_values_sample,  # Use prepared pixel_values
                    input_ids=prompt_ids,
                    max_new_tokens=250,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                generated_only = generated_ids[0][len(prompt_ids[0]):]
                generated_text = self.tokenizer.decode(generated_only, skip_special_tokens=True)

                # Build result dictionary
                result = {
                    "type": mode,
                    "step": self.state.global_step,
                    "epoch": float(self.state.epoch) if hasattr(self.state, 'epoch') else 0,
                    "generated_text": generated_text,
                }

                # Add ground truth for evaluation mode
                if mode == "evaluation":
                    labels = sample_input.get('labels', None)
                    if labels is not None:
                        labels_clean = labels[0].clone()
                        labels_clean[labels_clean == -100] = self.tokenizer.pad_token_id
                        ground_truth = self.tokenizer.decode(labels_clean, skip_special_tokens=True)
                    else:
                        ground_truth = "N/A"
                    result["ground_truth"] = ground_truth

                # Save to JSON
                json_file = "generation_logs.json"
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        logs = json.load(f)
                else:
                    logs = []

                logs.append(result)

                with open(json_file, 'w') as f:
                    json.dump(logs, f, indent=2, ensure_ascii=False)

                # Print output
                prefix = "[TRAIN]" if mode == "training" else "[EVAL]"
                if mode == "evaluation":
                    print("\n" + "="*80)
                    print(f"{prefix} Step: {self.state.global_step}, Epoch: {result['epoch']}")
                    print(f"{prefix} Generated: {generated_text}")
                    print(f"{prefix} Ground Truth: {result.get('ground_truth', 'N/A')}")
                    print("="*80 + "\n")
                else:
                    print(f"{prefix} Step: {self.state.global_step}")
                    print(f"{prefix} Generated: {generated_text}")

            except Exception as e:
                print(f"[ERROR] {mode.capitalize()} generation failed: {e}")
                import traceback
                traceback.print_exc()

        # Restore train mode only if we changed it
        if mode == "training":
            actual_model.train()

    def evaluation_loop(self, *args, **kwargs):
        """Override to reset generation flag at start of each evaluation"""
        # Reset flag so we log generation once per eval
        if hasattr(self, '_eval_generation_logged'):
            delattr(self, '_eval_generation_logged')

        return super().evaluation_loop(*args, **kwargs)
