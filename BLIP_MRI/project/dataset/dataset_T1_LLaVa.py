import os 
import numpy as np 
import pandas as pd 
import torch
import glob

from torch.utils.data import Dataset, DataLoader
#import lightning as L

from monai.data import NibabelReader
from monai.transforms import LoadImage, Randomizable, apply_transform, AddChannel, Compose, RandRotate90, Resize, NormalizeIntensity, Flip, ToTensor, RandAxisFlip, RandAffine
from monai.utils import MAX_SEED, get_seed

from utils.utils import to_3tuple

class BaseDataset_T1(Dataset, Randomizable): 
    def __init__(self, 
                 mode=None,
                 tokenizer=None,
                 img_size=None,
                 image_files=None, 
                 label=None, 
                 label_names=None, 
                 add_context=False, 
                 sex_text=None, 
                 age_text=None,
                 ):
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.image_files = image_files
        self.image_transform = self.define_image_augmentation(mode=mode)
        self.label = label
        self.label_names = label_names
        self.add_context = add_context
        if self.add_context: 
            assert sex_text is not None and age_text is not None
        self.sex_text = sex_text
        self.age_text = age_text
        self.quest_template, self.ans_template = self.make_question_answer_template(label_names=label_names)
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)    # use default reader of LoadImage
        self.set_random_state(seed=get_seed())
        self._seed = 0


    def define_image_augmentation(self, mode='train'):
        img_size = to_3tuple(self.img_size)
        if mode == 'train':
            transform = Compose([AddChannel(),
                               Resize(img_size),
                               RandAxisFlip(prob=0.5),
                               NormalizeIntensity()])
        elif mode == 'eval': 
            transform = Compose([AddChannel(),
                             Resize(img_size),
                             NormalizeIntensity()])
        return transform
    

    def make_question_answer_template(self, label_names=None): 
        quest = "USER: <image>\nYou are a neurologist and now you are analyzing T1-weighted MRI images."
        ans = 'ASSISTANT: '
        return quest, ans 
        

    def randomize(self, data=None) -> None: 
        self._seed = self.R.randint(MAX_SEED, dtype='uint32')


    def __transform_image__(self, image_file):
        image = self.image_loader(image_file)
        if self.image_transform is not None: 
            if isinstance(self.image_transform, Randomizable): 
                self.image_transform.set_random_state(seed=self._seed)
            image = apply_transform(self.image_transform, image, map_items=False)
            image = torch.tensor(image)
        return image
    
    def __transform_text__(self, label, add_context=False, sex=None, age=None):
        if len(self.label_names) == 1 and 'sex' in self.label_names:
            if int(label) == 1: 
                inst = f"{self.quest_template} Estimate sex of subject from this image. {self.ans_template} "
                answer = f'male'
            elif int(label) == 2: 
                inst = f"{self.quest_template} Estimate sex of subject from this image. {self.ans_template} "
                answer = f'female'
        elif len(self.label_names) == 1 and 'age' in self.label_names: 
            inst = f"{self.quest_template} Estimate age of subject from this image."
            answer = f'{self.ans_template} {label}'
        return inst, answer


    def __preprocess_as_hf__(self, image, inst, answer):
        inputs = {} 
        inputs['pixel_values'] = image
        inputs_txt = self.tokenizer(text=inst+answer, padding=True, return_tensors='pt')
        inputs['input_ids'] = inputs_txt['input_ids'].squeeze()
        inputs['attention_mask'] = inputs_txt['attention_mask'].squeeze()
        inputs['labels'] = inputs['input_ids'].clone()
        return inputs

    def __len__(self) -> int: 
        return len(self.image_files)
    
    def __getitem__(self, index:int): 
        """
        output: {"image": torch.tensor,"inst": str, "answer": str}
        """
        #assert np.array(self.label).shape[-1] == 1, "Multitask version is not implemented now. Now only support for single task version " 
        self.label = np.squeeze(self.label)

        image = self.__transform_image__(image_file=self.image_files[index])
        if self.add_context:
            inst, answer = self.__transform_text__(label=self.label[index], add_context=True, sex=self.sex_text[index], age=self.age_text[index])
        else: 
            inst, answer = self.__transform_text__(label=self.label[index], add_context=False)
        # preprocess as huggingface input
        inputs = self.__preprocess_as_hf__(image=image, inst=inst, answer=answer)
        return inputs 


class ABCD_T1: 
    def __init__(self, tokenizer=None, config_dataset=None, img_dir=None, meta_dir=None):
        self.tokenizer = tokenizer
        self.config_dataset = config_dataset
        self.img_dir = img_dir
        self.meta_dir = meta_dir
        self.subject_id_col = 'subjectkey'
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup()
        print(f"ABCD\n    Train: {self.train_dataset.__len__()}\n    Val: {self.val_dataset.__len__()}\n    Test: {self.test_dataset.__len__()}")

    def loading_metadata(self, meta_dir, study_sample='ABCD', subject_id_col='subjectkey'):
        targets = self.config_dataset.T1.target
        targets = [x for x in targets if x != None] # remove None in list
        col_list = targets + [subject_id_col]

        ### get subject ID and target variables
        meta_data = pd.read_csv(meta_dir)
        meta_data = meta_data.loc[:,col_list]
        meta_data = meta_data.sort_values(by=subject_id_col)
        meta_data = meta_data.dropna(axis = 0)
        meta_data = meta_data.reset_index(drop=True) # removing subject have NA values in sex

        return meta_data, targets
    

    def loading_images(self, image_dir, study_sample='ABCD', subject_id_col='subjectkey'):
        # getting each image files directory
        if study_sample.find('ABCD') != -1:
            image_files = glob.glob(os.path.join(image_dir,'*.npy'))
            #if study_sample.find('_T1') != -1:
            #    image_files = glob.glob(os.path.join(image_dir,'*.npy'))
            #else:
            #    image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
        image_files = sorted(image_files)
        #image_files = image_files[:100]
        print("Loading ABCD image file names as list is completed")

        # pre-process subject ids with image files
        if study_sample.find('ABCD') != -1:
            suffix_len = -4
            #if study_sample.find('_T1') != -1:
            #    suffix_len = -4
            #else:
            #    suffix_len = -7
        
        subj = []
        for i in range(len(image_files)):
                subject_id = os.path.split(image_files[i])[-1]
                subj.append(str(subject_id[:suffix_len]))
        
        image_list = pd.DataFrame({subject_id_col:subj, 'image_files': image_files})
        return image_list 
    


    def get_dataset(self, img_dir=None, meta_dir=None, subject_id_col=None): 
        ## loading meta data 
        if meta_dir is None: 
            raise ValueError("YOU SHOULD SPECIFY A DIRECTORY OF META DATA IN a '/config/*.yaml' FILE")
        elif meta_dir.find('.csv') != -1:
            meta_data, targets = self.loading_metadata(meta_dir=self.meta_dir, study_sample="ABCD", subject_id_col=subject_id_col)
        else: 
            NotImplementedError("This code need only meta data file of '.csv'format.")
        
        ## loading image data as pd.DataFrame (containing directory of each image files)
        image_data = self.loading_images(image_dir=self.img_dir, study_sample="ABCD", subject_id_col=subject_id_col)

        ## merging two pd.DataFrame, each of them are meta data and image data 
        meta_data[subject_id_col] = meta_data[subject_id_col].astype(str)
        image_data[subject_id_col] = image_data[subject_id_col].astype(str)
        dataset = pd.merge(meta_data, image_data, how='inner', on=subject_id_col)
        dataset = dataset.sort_values(by=subject_id_col)

        ## randomly assign subjects into train/val/test split
        #total_subj = 50
        total_subj = len(dataset)
        shuffle_idx = np.arange(total_subj)
        np.random.shuffle(shuffle_idx)

        assert self.config_dataset.train_size + self.config_dataset.val_size + self.config_dataset.test_size == 1
        train_subj = int(self.config_dataset.train_size * total_subj)
        val_subj = int(self.config_dataset.val_size * total_subj)
        test_subj = total_subj - train_subj - val_subj

        ## split image 
        train_images = [os.path.join(img_dir, img) for img in dataset['image_files'].values[shuffle_idx[:train_subj]]]
        val_images = [os.path.join(img_dir, img) for img in dataset['image_files'].values[shuffle_idx[train_subj:train_subj+val_subj]]]
        test_images = [os.path.join(img_dir, img) for img in dataset['image_files'].values[shuffle_idx[train_subj+val_subj:]]]


        ## split label 
        train_label = dataset[targets].values[shuffle_idx[:train_subj]]
        val_label = dataset[targets].values[shuffle_idx[train_subj:train_subj+val_subj]]
        test_label = dataset[targets].values[shuffle_idx[train_subj+val_subj:]]
        
        if self.config_dataset.add_context: 
            """
            ## split sex 
            train_sex = meta_data['sex'].values[shuffle_idx[:train_subj]]
            val_sex = meta_data['sex'].values[shuffle_idx[train_subj:train_subj+val_subj]]
            test_sex = meta_data['sex'].values[shuffle_idx[train_subj+val_subj:]]

            ## split age 
            train_age = meta_data['age'].values[shuffle_idx[:train_subj]]
            val_age = meta_data['age'].values[shuffle_idx[train_subj:train_subj+val_subj]]
            test_age = meta_data['age'].values[shuffle_idx[train_subj+val_subj:]]
            
            ## prepare dataset    
            train_dataset = Text_Image_Dataset(image_files=train_images, image_transform=self.define_image_augmentation(mode='train'),label=train_label,  add_context=True, sex_text=train_sex, age_text=train_age)
            val_dataset = Text_Image_Dataset(image_files=val_images, image_transform=self.define_image_augmentation(mode='eval'), label=val_label,  add_context=True, sex_text=val_sex, age_text=val_age)
            test_dataset = Text_Image_Dataset(image_files=test_images, image_transform=self.define_image_augmentation(mode='eval'), label=test_label,  add_context=True, sex_text=test_sex, age_text=test_age)
            """
        else: 
            ## prepare dataset    
            train_dataset = BaseDataset_T1(mode='train',tokenizer=self.tokenizer, img_size=self.config_dataset.img_size, image_files=train_images, label=train_label,  label_names=targets, add_context=False, sex_text=None, age_text=None)
            val_dataset = BaseDataset_T1(mode='eval', tokenizer=self.tokenizer, img_size=self.config_dataset.img_size, image_files=val_images, label=val_label,  label_names=targets, add_context=False, sex_text=None, age_text=None)
            test_dataset = BaseDataset_T1(mode='eval', tokenizer=self.tokenizer, img_size=self.config_dataset.img_size, image_files=test_images, label=test_label,  label_names=targets, add_context=False, sex_text=None, age_text=None)
        return train_dataset, val_dataset, test_dataset

    
    def setup(self, stage=None): 
        train_dataset, val_dataset, test_dataset = self.get_dataset(img_dir=self.img_dir, meta_dir=self.meta_dir, subject_id_col=self.subject_id_col)
        return train_dataset, val_dataset, test_dataset




class UKB_T1: 
    def __init__(self, tokenizer=None, config_dataset=None, img_dir=None, meta_dir=None):
        self.tokenizer = tokenizer
        self.config_dataset = config_dataset
        self.img_dir = img_dir
        self.meta_dir = meta_dir
        self.subject_id_col = 'eid'
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup()
        print(f"UKB\n    Train: {self.train_dataset.__len__()}\n    Val: {self.val_dataset.__len__()}\n    Test: {self.test_dataset.__len__()}")

    def loading_metadata(self, meta_dir, study_sample='UKB', subject_id_col='eid'):
        targets = self.config_dataset.T1.target
        targets = [x for x in targets if x != None] # remove None in list
        col_list = targets + [subject_id_col]

        ### get subject ID and target variables
        meta_data = pd.read_csv(meta_dir)
        meta_data = meta_data.loc[:,col_list]
        meta_data = meta_data.sort_values(by=subject_id_col)
        meta_data = meta_data.dropna(axis = 0)
        meta_data = meta_data.reset_index(drop=True) # removing subject have NA values in sex

        return meta_data, targets
    

    def loading_images(self, image_dir, study_sample='UKB', subject_id_col='eid'):
        # getting each image files directory
        if study_sample.find('UKB') != -1:
            image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))

        image_files = sorted(image_files)
        #image_files = image_files[:100]
        print("Loading UKB image file names as list is completed")

        # pre-process subject ids with image files
        if study_sample.find('UKB') != -1:
            suffix_len = -7
        
        subj = []
        for i in range(len(image_files)):
                subject_id = os.path.split(image_files[i])[-1]
                subj.append(str(subject_id[:suffix_len]))
        
        image_list = pd.DataFrame({subject_id_col:subj, 'image_files': image_files})
        
        return image_list 
    


    def get_dataset(self, img_dir=None, meta_dir=None, subject_id_col=None): 
        ## loading meta data 
        if meta_dir is None: 
            raise ValueError("YOU SHOULD SPECIFY A DIRECTORY OF META DATA IN a '/config/*.yaml' FILE")
        elif meta_dir.find('.csv') != -1:
            meta_data, targets = self.loading_metadata(meta_dir=self.meta_dir, study_sample="UKB", subject_id_col=subject_id_col)
        else: 
            NotImplementedError("This code need only meta data file of '.csv'format.")
        
        ## loading image data as pd.DataFrame (containing directory of each image files)
        image_data = self.loading_images(image_dir=self.img_dir, study_sample="UKB", subject_id_col=subject_id_col)

        ## merging two pd.DataFrame, each of them are meta data and image data 
        meta_data[subject_id_col] = meta_data[subject_id_col].astype(str)
        image_data[subject_id_col] = image_data[subject_id_col].astype(str)
        dataset = pd.merge(meta_data, image_data, how='inner', on=subject_id_col)
        dataset = dataset.sort_values(by=subject_id_col)

        ## randomly assign subjects into train/val/test split
        #total_subj = 50
        total_subj = len(dataset)
        shuffle_idx = np.arange(total_subj)
        np.random.shuffle(shuffle_idx)

        assert self.config_dataset.train_size + self.config_dataset.val_size + self.config_dataset.test_size == 1
        train_subj = int(self.config_dataset.train_size * total_subj)
        val_subj = int(self.config_dataset.val_size * total_subj)
        test_subj = total_subj - train_subj - val_subj

        ## split image 
        train_images = [os.path.join(img_dir, img) for img in dataset['image_files'].values[shuffle_idx[:train_subj]]]
        val_images = [os.path.join(img_dir, img) for img in dataset['image_files'].values[shuffle_idx[train_subj:train_subj+val_subj]]]
        test_images = [os.path.join(img_dir, img) for img in dataset['image_files'].values[shuffle_idx[train_subj+val_subj:]]]


        ## split label 
        train_label = dataset[targets].values[shuffle_idx[:train_subj]]
        val_label = dataset[targets].values[shuffle_idx[train_subj:train_subj+val_subj]]
        test_label = dataset[targets].values[shuffle_idx[train_subj+val_subj:]]
        
        if self.config_dataset.add_context: 
            """
            ## split sex 
            train_sex = meta_data['sex'].values[shuffle_idx[:train_subj]]
            val_sex = meta_data['sex'].values[shuffle_idx[train_subj:train_subj+val_subj]]
            test_sex = meta_data['sex'].values[shuffle_idx[train_subj+val_subj:]]

            ## split age 
            train_age = meta_data['age'].values[shuffle_idx[:train_subj]]
            val_age = meta_data['age'].values[shuffle_idx[train_subj:train_subj+val_subj]]
            test_age = meta_data['age'].values[shuffle_idx[train_subj+val_subj:]]
            
            ## prepare dataset    
            train_dataset = Text_Image_Dataset(image_files=train_images, image_transform=self.define_image_augmentation(mode='train'),label=train_label,  add_context=True, sex_text=train_sex, age_text=train_age)
            val_dataset = Text_Image_Dataset(image_files=val_images, image_transform=self.define_image_augmentation(mode='eval'), label=val_label,  add_context=True, sex_text=val_sex, age_text=val_age)
            test_dataset = Text_Image_Dataset(image_files=test_images, image_transform=self.define_image_augmentation(mode='eval'), label=test_label,  add_context=True, sex_text=test_sex, age_text=test_age)
            """
        else: 
            ## prepare dataset    
            train_dataset = BaseDataset_T1(mode='train',tokenizer=self.tokenizer, img_size=self.config_dataset.img_size, image_files=train_images, label=train_label,  label_names=targets, add_context=False, sex_text=None, age_text=None)
            val_dataset = BaseDataset_T1(mode='eval', tokenizer=self.tokenizer, img_size=self.config_dataset.img_size, image_files=val_images, label=val_label,  label_names=targets, add_context=False, sex_text=None, age_text=None)
            test_dataset = BaseDataset_T1(mode='eval', tokenizer=self.tokenizer, img_size=self.config_dataset.img_size, image_files=test_images, label=test_label,  label_names=targets, add_context=False, sex_text=None, age_text=None)
        return train_dataset, val_dataset, test_dataset

    
    def setup(self, stage=None): 
        train_dataset, val_dataset, test_dataset = self.get_dataset(img_dir=self.img_dir, meta_dir=self.meta_dir, subject_id_col=self.subject_id_col)
        return train_dataset, val_dataset, test_dataset



