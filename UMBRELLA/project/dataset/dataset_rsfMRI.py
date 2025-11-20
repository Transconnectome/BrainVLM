# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride_between_seq = 1
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)
        self.quest_template, self.ans_template = self.make_question_answer_template()
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        
        y = []
        if self.shuffle_time_sequence: # shuffle whole sequences
            load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
        else:
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
        # if self.with_voxel_norm:
        #     load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
        for fname in load_fnames:
            img_path = os.path.join(subject_path, fname)
            y_i = torch.load(img_path, weights_only=False).unsqueeze(0)
            y.append(y_i)
        y = torch.cat(y, dim=4)
            
        if self.input_scaling_method == 'none':
            #print('Assume that normalization already done and global_stats.pt does not exist (preprocessing v1)')
            pass
        else:
            stats_path = os.path.join(subject_path,'global_stats.pt')
            stats_dict = torch.load(stats_path, weights_only=False) # ex) {'valid_voxels': 172349844, 'global_mean': tensor(7895.4902), 'global_std': tensor(5594.5850), 'global_max': tensor(37244.4766)}
            if self.input_scaling_method == 'minmax':
                y = y / stats_dict['global_max'] # assume that min value is zero and in background  
            elif self.input_scaling_method == 'znorm_zeroback':
                background = y==0
                y = (y - stats_dict['global_mean']) / stats_dict['global_std']
                y[background] = 0
            elif self.input_scaling_method == 'znorm_minback':
                background = y==0
                y = (y - stats_dict['global_mean']) / stats_dict['global_std']
            return y

    def make_question_answer_template(self): 
        quest = 'Question: You are a neurologist and now you are analyzing T1-weighted MRI images.'
        ans = 'Answer:'
        return quest, ans 


    def __preprocess_as_hf__(self, image, inst, answer):
        inputs = {} 
        inputs['pixel_values'] = {}
        inputs['input_ids'] = {}
        inputs['attention_mask'] = {}
        inputs['labels'] = {}

        inputs['pixel_values']['rsfMRI'] = image
        inputs_txt = self.tokenizer(inst, padding=True, return_tensors='pt')
        inputs['input_ids']['rsfMRI'] = inputs_txt['input_ids'].squeeze()
        inputs['attention_mask']['rsfMRI'] = inputs_txt['attention_mask'].squeeze()
        #inputs.update(inputs_txt)

        output_tokens = self.tokenizer(answer, padding=True, return_tensors='pt')
        output_tokens.input_ids = output_tokens.input_ids.squeeze()
        targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.tokenizer.pad_token_id, -100
            )
        inputs['labels']['rsfMRI'] = targets
        #inputs['modality'] = torch.tensor([1]) # 0: T1, 1: rsfMRI
        return inputs
    
    
    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")




class S1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.sample_duration, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __transform_image__(self, image):
        background_value = image.flatten()[0]
        image = image.permute(0,4,1,2,3) 
        image = torch.nn.functional.pad(image, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
        image = image.permute(0,2,3,4,1) 
        return image 

    def __transform_text__(self, label, add_context=False, sex=None, age=None):
        if self.target == "sex":
            assert isinstance(label, str) is True
            assert label in ["male", "female"]
            inst = f"{self.quest_template} Estimate sex of subject from this image. {self.ans_template} "
            answer = f"{label}"
        elif self.target == "age": 
            inst = f"{self.quest_template} Estimate age of subject from this image. {self.ans_template} "
            answer = f'{label}'
        return inst, answer

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()
        image = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
        image = self.__transform_image__(image)

        if self.add_context:
            inst, answer = self.__transform_text__(label=target, add_context=True, sex=None, age=None)
        else: 
            inst, answer = self.__transform_text__(label=target, add_context=False)
        # preprocess as huggingface input
        inputs = self.__preprocess_as_hf__(image=image, inst=inst, answer=answer)
        return inputs



class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex)
                data.append(data_tuple)
                        
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __transform_image__(self, image):
        background_value = image.flatten()[0]
        image = image.permute(0,4,1,2,3) 
        image = torch.nn.functional.pad(image, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data 
        image = image.permute(0,2,3,4,1) 
        return image 

    def __transform_text__(self, label, add_context=False, sex=None, age=None):
        if self.target == "sex":
            assert isinstance(label, str) is False
            if int(label) == 0: 
                inst = f"{self.quest_template} Estimate sex of subject from this image. {self.ans_template} "
                answer = f'male'
            elif int(label) == 1: 
                inst = f"{self.quest_template} Estimate sex of subject from this image. {self.ans_template} "
                answer = f'female'
        elif self.target == "age": 
            inst = f"{self.quest_template} Estimate age of subject from this image. {self.ans_template} "
            answer = f'{round(label//12)}' # In ABCD dataset, age is recorded as months 
        return inst, answer

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()
        image = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)
        image = self.__transform_image__(image)

        if self.add_context:
            inst, answer = self.__transform_text__(label=target, add_context=True, sex=None, age=None)
        else: 
            inst, answer = self.__transform_text__(label=target, add_context=False)
        # preprocess as huggingface input
        inputs = self.__preprocess_as_hf__(image=image, inst=inst, answer=answer)
        return inputs
    

        

class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject20227 = str(subject_name)+'_20227_2_0'
            subject_path = os.path.join(img_root, subject20227)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3) 
        y = torch.nn.functional.pad(y, (3, 9, 0, 0, 10, 8), value=background_value) # adjust this padding level according to your data 
        y = y.permute(0,2,3,4,1) 
        return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex,
                        "study_name": 'UKB'
                } 
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()
        
        return {
                "fmri_sequence": y,
                "subject_name": subj,
                "target": target,
                "TR": 0,
                "sex": sex,
                "study_name": 'Dummy'
            } 

class ABIDE(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []

        for i, subject in enumerate(subject_dict):
            sex,target,site_id,data_type = subject_dict[subject]

            #subject_path = os.path.join(root, data_type, 'sub-'+subjecta)
            subject_path = os.path.join(root,'img',data_type, subject)

            #num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            num_frames = len(glob.glob(os.path.join(subject_path,'frame_*')))
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)

        assert self.input_scaling_method == 'none'
        print('Assume that normalization already done and global_stats.pt does not exist (preprocessing v1)')

        # train dataset
        # for regression tasks
        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        y = self.load_sequence(subject_path, start_frame, sequence_length,num_frames)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        y = torch.nn.functional.pad(y, (0, -1, -10, -9, -1, 0), value=background_value) # (97,115,97) -> (96, 96, 96)
        y = y.permute(0,2,3,4,1)

        return {
            "fmri_sequence": y,
            "subject_name": subject,
            "target": target,
            "TR": start_frame,
            "sex": sex,
            "study_name": 'ABIDE'
        } 
    
class HBN(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            
            subject_path = os.path.join(img_root, subject_name)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std >>> 데이터마다 다름? > 안잘라도 됨?
            session_duration = num_frames - self.sample_duration + 1 #### >>>> 이건 뭘까

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.sample_duration, num_frames, target, sex)
                data.append(data_tuple)
                        
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1) #### 이건 왜 하는거지?

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

        background_value = y.flatten()[0]
        y = y.permute(0,4,1,2,3)
        if self.input_type == 'rest':
            # HBN rest image shape: 81, 95, 81
            y = torch.nn.functional.pad(y, (7, 8, 1, 0, 7, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data

        elif self.input_type == 'task':
            # ABCD task image shape: 96, 96, 95
            # background value = 0
            # minmax scaled in brain (0~1)
            y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
        y = y.permute(0,2,3,4,1)

        return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
                "study_name": 'HBN'
            } 


