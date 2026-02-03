import os
import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from dataset.dataset_rsfMRI import S1200, ABCD, UKB, Dummy, HBN, ABIDE
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

class rsfMRIData:
    def __init__(self, dataset_name=None, image_path=None, target=None, config=None, tokenizer=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_path = image_path 
        self.target = target[0]
        self.config = config
        self.dataset_params = {
                "add_context": self.config.add_context,
                "root": image_path,
                "target": self.target,
                "sequence_length": self.config.rsfMRI.sequence_length,
                "stride_between_seq": self.config.rsfMRI.stride_between_seq,
                "stride_within_seq": self.config.rsfMRI.stride_within_seq,
                "shuffle_time_sequence": self.config.rsfMRI.shuffle_time_sequence,
                "input_scaling_method" : self.config.rsfMRI.input_scaling_method,
                "label_scaling_method" : self.config.rsfMRI.label_scaling_method,
                "dtype":'float16'
                }
        self.dataset_params.update({"tokenizer":tokenizer}) 
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup()
        

        
    def define_split_file_path(self, dataset_name, dataset_split_num):
        # generate splits folder
        split_dir_path = f'./data/splits/{dataset_name}/'
        os.makedirs(split_dir_path, exist_ok=True)
        split_file_path = os.path.join(split_dir_path, f"split_fixed_{dataset_split_num}.txt")
        return split_file_path
        
    def get_dataset(self, dataset_name):
        if dataset_name == "Dummy":
            return Dummy
        elif dataset_name == "HBN":
            return HBN
        elif dataset_name == "S1200":
            return S1200
        elif dataset_name == "ABCD":
            return ABCD
        elif 'UKB' in dataset_name:
            return UKB
        elif dataset_name == 'ABIDE':
            return ABIDE
        else:
            raise NotImplementedError
    

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        #subj_idx = np.array([str(x[0]) for x in subj_list])
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        # print(S)
        print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx
    
    def save_split(self, sets_dict, split_file_path):
        with open(split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")
    
    def determine_split_stratified(self, S, idx, split_file_path):
        print('making stratified split')
        #S = np.unique([x[1] for x in index_l]) #len(np.unique([x[1] for x in index_l]))
        site_dict = {x:S[x][idx] for x in S} # index 2: site_id, idex 3: data type (ABIDE1/ABIDE2)
        site_ids = np.array(list(site_dict.values()))
        #print('site_ids:',site_ids)
        #print('S:',S)
        #subjects = list(S.keys())
        
        #remove sites that has only one valid samples
        one_value_sites=[]
        values, counts = np.unique(site_ids, return_counts=True)
        # Print the value counts
        for value, count in zip(values, counts):
            # print(f"{value}: {count}") # 20,40 has one level
            if count == 1:
                one_value_sites.append(value)
                
        filtered_site_dict = {subj:site for subj,site in site_dict.items() if site not in one_value_sites}
        filtered_subjects = np.array(list(filtered_site_dict.keys()))
        filtered_site_ids = np.array(list(filtered_site_dict.values()))
        
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=1-self.config.train_size-self.config.val_size, random_state=self.config.rsfMRI.dataset_split_num)
        trainval_indices, test_indices = next(strat_split.split(filtered_subjects, filtered_site_ids)) # 0.
        
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=self.config.val_size, random_state=self.config.rsfMRI.dataset_split_num)
        train_indices, valid_indices = next(strat_split.split(filtered_subjects[trainval_indices], filtered_site_ids[trainval_indices]))
        S_train, S_val, S_test = filtered_subjects[trainval_indices][train_indices], filtered_subjects[trainval_indices][valid_indices], filtered_subjects[test_indices]
        
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test}, split_file_path)
        return S_train, S_val, S_test
    
    def determine_split_randomly(self, S, split_file_path):
        S = list(S.keys())
        S_train = int(len(S) * self.config.train_size)
        S_val = int(len(S) * self.config.val_size)
        S_train = np.random.choice(S, S_train, replace=False)
        remaining = np.setdiff1d(S, S_train) # np.setdiff1d(np.arange(S), S_train)
        S_val = np.random.choice(remaining, S_val, replace=False)
        S_test = np.setdiff1d(S, np.concatenate([S_train, S_val])) # np.setdiff1d(np.arange(S), np.concatenate([S_train, S_val]))
        # train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(S_train, S_val, S_test, self.subject_list)
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test}, split_file_path)
        return S_train, S_val, S_test
    
    def load_split(self, split_file_path):
        subject_order = open(split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]
        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self, dataset_name, image_path):
        # output: {'subj1':[target1,target2],'subj2':[target1,target2]...}
        img_root = os.path.join(image_path, 'img')
        final_dict = dict()
        if dataset_name == "S1200":
            subject_list = os.listdir(img_root)
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_gender.csv"))
            meta_data_residual = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_precise_age.csv"))
            meta_data_all = pd.read_csv(os.path.join(image_path, "metadata", "HCP_1200_all.csv"))
            if self.target == 'sex': task_name = 'Gender'
            elif self.target == 'age': task_name = 'age'
            elif self.target == 'int_total': task_name = 'CogTotalComp_AgeAdj'
            else: raise NotImplementedError()

            if self.target == 'sex':
                meta_task = meta_data[['Subject',task_name]].dropna()
            elif self.target == 'age':
                meta_task = meta_data_residual[['subject',task_name,'sex']].dropna()
                #rename column subject to Subject
                meta_task = meta_task.rename(columns={'subject': 'Subject'})
            elif self.target == 'int_total':
                meta_task = meta_data[['Subject',task_name,'Gender']].dropna()  
            
            for subject in subject_list:
                if int(subject) in meta_task['Subject'].values:
                    if self.target == 'sex':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        target = "male" if target == "M" else "female"
                        sex = target
                    elif self.target == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["sex"].values[0]
                        sex = "male" if sex == "M" else "female"
                    elif self.target == 'int_total':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["Gender"].values[0]
                        sex = "male" if sex == "M" else "female"
                    final_dict[subject]=[sex,target]

        elif dataset_name == "HBN": 
            subject_list = os.listdir(img_root)
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "HBN_metadata_231212.csv"))
            if self.target == 'sex': task_name = 'Sex'
            elif self.target == 'age': task_name = 'Age'
            elif self.target == 'Dx.ndd_HC': task_name = 'Dx.ndd_HC'
            elif self.target == 'Dx.all_HC': task_name = 'Dx.all_HC'
            elif self.target == 'Dx.adhd_HC': task_name = 'Dx.adhd_HC'
            elif self.target == 'Dx.asd_HC': task_name = 'Dx.asd_HC'
            elif self.target == 'Dx.adhd_asd': task_name = 'Dx.adhd_asd'
            else: raise ValueError('downstream task not supported')
           
            if self.target == 'Sex':
                meta_task = meta_data[['SUBJECT_ID',task_name]].dropna()
            else:
                meta_task = meta_data[['SUBJECT_ID',task_name,'Sex']].dropna() # 왜 이렇게 하는거지?: sex를 왜 포함?
            
            for subject in subject_list:
                if subject in meta_task['SUBJECT_ID'].values:
                    target = meta_task[meta_task["SUBJECT_ID"]==subject][task_name].values[0]
                    sex = meta_task[meta_task["SUBJECT_ID"]==subject]["Sex"].values[0]
                    final_dict[subject]=[sex,target]

        elif dataset_name == "ABCD":
            subject_list = [subj for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "ABCD_phenotype_total.csv"))
            if self.target == 'sex': task_name = 'sex'
            elif self.target == 'age': task_name = 'age'
            elif self.target == 'int_total': task_name = 'nihtbx_totalcomp_uncorrected'
            else: raise ValueError('downstream task not supported')
           
            if self.target == 'sex':
                meta_task = meta_data[['subjectkey',task_name]].dropna()
            else:
                meta_task = meta_data[['subjectkey',task_name,'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subjectkey'].values:
                    target = meta_task[meta_task["subjectkey"]==subject][task_name].values[0]
                    sex = meta_task[meta_task["subjectkey"]==subject]["sex"].values[0]
                    final_dict[subject]=[sex,target]
            
        elif "UKB" in dataset_name:
            if self.target == 'sex': task_name = 'sex'
            elif self.target == 'age': task_name = 'age'
            elif self.target == 'int_fluid' : task_name = 'fluid'
            elif self.target == 'depression_current' : task_name = 'Depressed.Current'
            else: raise ValueError('downstream task not supported')
                
            meta_data = pd.read_csv(os.path.join(image_path, "metadata", "UKB_phenotype_depression_included.csv"))
            if task_name == 'sex':
                meta_task = meta_data[['eid',task_name]].dropna()
            else:
                meta_task = meta_data[['eid',task_name,'sex']].dropna()

            for subject in os.listdir(img_root):
                if subject.endswith('20227_2_0') and (int(subject[:7]) in meta_task['eid'].values):
                    target = meta_task[meta_task["eid"]==int(subject[:7])][task_name].values[0]
                    sex = meta_task[meta_task["eid"]==int(subject[:7])]["sex"].values[0]
                    final_dict[str(subject[:7])] = [sex,target]
                else:
                    continue 
        elif dataset_name == "ABIDE":
            if self.target == 'sex': task_name = 'SEX'
            elif self.target == 'age': task_name = 'AGE_AT_SCAN'
            elif self.target == 'ASD': task_name = 'DX_GROUP'
            else: raise NotImplementedError()
            
            
            abide1=pd.read_csv(os.path.join(image_path, "metadata", "ABIDE1_pheno.csv"))
            abide2=pd.read_csv(os.path.join(image_path, "metadata", "ABIDE2_pheno_total.csv"),encoding= 'unicode_escape')
            total=pd.concat([abide1,abide2])
            # only leave version2
            meta_data = total.loc[~total.duplicated('SUB_ID',keep='last'),:].reset_index(drop=True)

            #img_abide1 = os.path.join(image_path, 'ABIDE1')
            #img_abide2 = os.path.join(image_path, 'ABIDE2')
            img_abide1 = os.path.join(image_path, 'img', 'ABIDE1')
            img_abide2 = os.path.join(image_path, 'img', 'ABIDE2')

            subj_list1 = os.listdir(img_abide1)
            subj_list2 = os.listdir(img_abide2)

            subj_list1 = [subj for subj in subj_list1 if subj not in subj_list2]

            #img_root = os.path.join(self.hparams.image_path, 'ABIDE1')
            img_root = os.path.join(image_path, 'ABIDE1')
            

            if self.target  == 'sex':
                meta_task = meta_data[['SUB_ID',task_name,'SITE_ID']].dropna()
                meta_task = meta_task.rename(columns={'SUB_ID': 'Subject'})
            elif self.target  == 'age':
                meta_task = meta_data[['SUB_ID',task_name,'SEX','SITE_ID']].dropna()
                #rename column subject to Subject
                meta_task = meta_task.rename(columns={'SUB_ID': 'Subject'})
            elif self.target  == 'ASD':
                meta_task = meta_data[['SUB_ID',task_name,'SEX','SITE_ID']].dropna()
                meta_task = meta_task.rename(columns={'SUB_ID': 'Subject'})

            le = LabelEncoder()
            meta_task['SITE_ID'] = le.fit_transform(meta_task['SITE_ID'])
            
            for i, subject in enumerate(subj_list1):
                #subject = subject[4:]
                if int(subject) in meta_task['Subject'].values:
                    if self.target == 'sex':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0] -1
                        sex = target-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.target == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.target == 'ASD':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]-1
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0] 
                    final_dict[subject] = [sex, target, site_id, 'ABIDE1']
            for i, subject in enumerate(subj_list2):
                #subject = subject[4:]
                if int(subject) in meta_task['Subject'].values:
                    if self.target == 'sex':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0] -1
                        sex = target-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.target == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0]
                    elif self.target == 'ASD':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]-1
                        sex = meta_task[meta_task["Subject"]==int(subject)]["SEX"].values[0]-1
                        site_id = meta_task[meta_task["Subject"]==int(subject)]["SITE_ID"].values[0] 
                    final_dict[subject] = [sex, target, site_id, 'ABIDE2']
        return final_dict

    def _undersampling(self, dataset:dict, num_samples=None, binary=True, randomize=True):
        assert binary == True
        keys = list(dataset.keys())
        ctrl_keys = []
        case_keys = [] 
        for _key in keys: 
            if dataset[_key][1] == 0: 
                ctrl_keys.append(_key)
            elif dataset[_key][1] == 1: 
                case_keys.append(_key)
                
        if num_samples is not None: 
            num_ctrl = int(num_samples / 2)
            num_case = int(num_samples / 2)
        else: 
            if len(ctrl_keys) >= len(case_keys): 
                num_ctrl = len(case_keys)
                num_case = len(case_keys)
            elif len(ctrl_keys) < len(case_keys):
                num_ctrl = len(ctrl_keys)
                num_case = len(ctrl_keys)
        sampled_ctrl_keys = random.sample(ctrl_keys, num_ctrl)
        sampled_case_keys = random.sample(case_keys, num_case)
        sampled_keys = sampled_ctrl_keys + sampled_case_keys
        undersampled_dataset = {key: dataset[key] for key in sampled_keys}
                
        if randomize: 
            undersampled_keys = list(undersampled_dataset.keys())
            randomized_undersampled_keys = random.sample(undersampled_keys, len(undersampled_keys))
            undersampled_dataset = {key: undersampled_dataset[key] for key in randomized_undersampled_keys}
        return undersampled_dataset, num_ctrl, num_case  


    def setup(self, stage=None):
        # this function will be called at each devices        
        ###
        split_file_path = self.define_split_file_path(self.dataset_name, self.config.rsfMRI.dataset_split_num)
        Dataset = self.get_dataset(self.dataset_name)
        subject_dict = self.make_subject_dict(self.dataset_name, self.image_path)

        if os.path.exists(split_file_path):
            train_names, val_names, test_names = self.load_split(split_file_path)
        elif self.dataset_name == 'ABIDE':
            #stratified split for ABIDE dataset
            idx = 2 # idx = 2 for site_id
            train_names, val_names, test_names = self.determine_split_stratified(subject_dict, idx, split_file_path)
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict, split_file_path)


        #train_dict = {key: subject_dict[key] for key in train_names[:10] if key in subject_dict}
        #val_dict = {key: subject_dict[key] for key in val_names[:10] if key in subject_dict}
        #test_dict = {key: subject_dict[key] for key in test_names[:10] if key in subject_dict}
        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}


        if self.config.rsfMRI.balanced_training_samples: 
            if self.config.rsfMRI.limit_training_samples:
                train_dict, train_num_ctrl, train_num_case = self._undersampling(dataset=train_dict, num_samples=self.config.rsfMRI.limit_training_samples)
            else: 
                train_dict, train_num_ctrl, train_num_case = self._undersampling(dataset=train_dict)
            print(f"Train/ctrl: {train_num_ctrl}, Train/case: {train_num_case}")
        else: 
            if self.config.rsfMRI.limit_training_samples:
                keys = list(train_dict.keys())
                if len(keys) > self.config.rsfMRI.limit_training_samples:
                    sampled_keys = random.sample(keys, self.config.rsfMRI.limit_training_samples)
                else:
                    sampled_keys = keys
                train_dict = {key: train_dict[key] for key in sampled_keys}


        if self.config.rsfMRI.balanced_validation_samples: 
            if self.config.rsfMRI.limit_validation_samples:
                val_dict, val_num_ctrl, val_num_case = self._undersampling(dataset=val_dict, num_samples=self.config.rsfMRI.limit_validation_samples)
            else: 
                val_dict, val_num_ctrl, val_num_case = self._undersampling(dataset=val_dict)
            print(f"Val/ctrl: {train_num_ctrl}, Val/case: {train_num_case}")
        else: 
            if self.config.rsfMRI.limit_validation_samples:
                keys = list(val_dict.keys())
                if len(keys) > self.config.rsfMRI.limit_validation_samples:
                    sampled_keys = random.sample(keys, self.config.rsfMRI.limit_validation_samples)
                else:
                    sampled_keys = keys
                val_dict = {key: val_dict[key] for key in sampled_keys}

        if self.config.rsfMRI.balanced_test_samples: 
            if self.config.rsfMRI.limit_test_samples:
                test_dict, test_num_ctrl, test_num_case = self._undersampling(dataset=test_dict, num_samples=self.config.rsfMRI.limit_test_samples)
            else: 
                test_dict, test_num_ctrl, test_num_case = self._undersampling(dataset=test_dict)
            print(f"Test/ctrl: {train_num_ctrl}, Test/case: {train_num_case}")
        else:
            if self.config.rsfMRI.limit_test_samples:
                keys = list(test_dict.keys())
                if len(keys) > self.config.rsfMRI.limit_test_samples:
                    sampled_keys = random.sample(keys, self.config.rsfMRI.limit_test_samples)
                else:
                    sampled_keys = keys
                test_dict = {key: test_dict[key] for key in sampled_keys}


        train_dataset = Dataset(**self.dataset_params,subject_dict=train_dict,use_augmentations=False, train=True)
        # load train mean/std of target labels to val/test dataloader
        val_dataset = Dataset(**self.dataset_params,subject_dict=val_dict,use_augmentations=False,train=False)
        test_dataset = Dataset(**self.dataset_params,subject_dict=test_dict,use_augmentations=False,train=False)
        print(f"{self.dataset_name} (rsfMRI)\n    Train: {len(train_dict)}\n    Val: {len(val_dict)}\n    Test: {len(test_dict)}")
        return train_dataset, val_dataset, test_dataset

