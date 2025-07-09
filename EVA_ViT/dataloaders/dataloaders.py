import os
from os import listdir
from os.path import isfile, join
import glob


from util.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, NormalizeIntensity, Flip, ToTensor, RandSpatialCrop, ScaleIntensity, RandAxisFlip, RandCoarseDropout
from monai.data import ImageDataset


import nibabel as nib
from scipy.ndimage import zoom


def offline_resample(
    input_dir: str,
    output_dir: str,
    target_zooms=(2.0,2.0,2.0),
    order: int = 1,
    suffix: str = "_2mm",
):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        return  # 이미 리샘플된 파일이 있으면 건너뜀
    os.makedirs(output_dir, exist_ok=True)
    for path in glob.glob(os.path.join(input_dir, "*.nii*")):
        img = nib.load(path)
        data = img.get_fdata()
        orig_zooms = img.header.get_zooms()[:3]
        factors = tuple(o/t for o,t in zip(orig_zooms, target_zooms))
        down = zoom(data, factors, order=order)

        hdr = img.header.copy()
        hdr.set_zooms((*target_zooms, hdr.get_zooms()[3]))
        base, ext = os.path.splitext(os.path.basename(path))
        if ext == ".gz":
            base, _ = os.path.splitext(base)
            ext = ".nii.gz"
        out_path = os.path.join(output_dir, f"{base}{suffix}{ext}")
        nib.save(nib.Nifti1Image(down.astype(np.float32),
                                 img.affine, hdr), out_path)


def check_study_sample(study_sample):
    if study_sample == 'UKB':
        image_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'
        #image_dir = '/pscratch/sd/h/heehaw/data/2.UKB/1.sMRI_fs_cropped'
        #phenotype_dir = '/pscratch/sd/h/heehaw/data/2.UKB/UKB_phenotype.csv'
    elif study_sample == 'UKB_MNI':
        image_dir = '/scratch/connectome/3DCNN/data/2.UKB/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/2.UKB/2.demo_qc/UKB_phenotype.csv'    
    elif study_sample == 'ABCD_T1':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/5.demo_qc/ABCD_phenotype_total.csv'  
        #image_dir = "/grand/NeuroX/T1_data/1.ABCD/2.sMRI_freesurfer"
        #phenotype_dir = "/grand/NeuroX/T1_data/1.ABCD/ABCD_phenotype_total.csv"
        #image_dir = "/pscratch/sd/h/heehaw/data/1.ABCD/2.sMRI_freesurfer"
        #phenotype_dir = "/pscratch/sd/h/heehaw/data/1.ABCD/ABCD_phenotype_total.csv" 
    elif study_sample == 'ABCD_MNI':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/5.demo_qc/ABCD_phenotype_total.csv'  
    elif study_sample == 'ABCD_FA':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/3.2.FA_warpped_nii'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/5.demo_qc/ABCD_phenotype_total.csv'  
    elif study_sample == 'GARD_T1':
        image_dir = '/scratch/connectome/3DCNN/data/3.GARD/2.T1_GARD_ALL'
        phenotype_dir = '/scratch/connectome/3DCNN/data/3.GARD/1.demo_qc/GARD_demographics_neuroimaging.csv'   
    return image_dir, phenotype_dir 



def loading_images(image_dir, args, study_sample='UKB'):
    if study_sample.find('UKB') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample.find('ABCD') != -1:
        if study_sample.find('_T1') != -1:
            image_files = glob.glob(os.path.join(image_dir,'*.npy'))
        else:
            image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample.find('GARD') != -1:
        if study_sample.find('_T1') != -1:
            image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    image_files = sorted(image_files)
    #image_files = image_files[:100]
    print("Loading image file names as list is completed")
    return image_files

def loading_phenotype(phenotype_dir, args, study_sample='UKB'):
    if study_sample.find('UKB') != -1:
        subject_id_col = 'eid'
    elif study_sample.find('ABCD') != -1:
        subject_id_col = 'subjectkey'
    elif study_sample.find('GARD') != -1:
        subject_id_col = 'subject_id'

    targets = args.cat_target + args.num_target
    col_list = targets + [subject_id_col]

    ### get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    subject_data = subject_data.loc[:,col_list]
    #subject_data = subject_data.sort_values(by=subject_id_col)
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex

    ### preprocessing categorical variables and numerical variables
    if args.cat_target:
        subject_data = preprocessing_cat(subject_data, args)
        num_classes = int(subject_data[args.cat_target].nunique().values)
    if args.num_target:
        #subject_data = preprocessing_num(subject_data, args)
        num_classes = 1 
    
    return subject_data, targets, num_classes


## combine categorical + numeric
def combining_image_target(subject_data, image_files, target_list, study_sample='UKB'):
    if study_sample.find('UKB') != -1:
        subject_id_col = 'eid'
        suffix_len = -7
    elif study_sample.find('ABCD') != -1:
        subject_id_col = 'subjectkey'
        if study_sample.find('_T1') != -1:
            suffix_len = -4
        else:
            suffix_len = -7
    elif study_sample.find('GARD') != -1:
        subject_id_col = 'subject_id'
        suffix_len = -7
        

    imageFiles_labels = []
    
    
    subj = []
    if type(subject_data[subject_id_col][0]) == np.str_ or type(subject_data[subject_id_col][0]) == str:
        for i in range(len(image_files)):
            subject_id = os.path.split(image_files[i])[-1]
            subj.append(str(subject_id[:suffix_len]))
    elif type(subject_data[subject_id_col][0]) == np.int_ or type(subject_data[subject_id_col][0]) == int:
        for i in range(len(image_files)):
            subject_id = os.path.split(image_files[i])[-1]
            subj.append(int(subject_id[:suffix_len]))

    image_list = pd.DataFrame({subject_id_col:subj, 'image_files': image_files})
    subject_data = pd.merge(subject_data, image_list, how='inner', on=subject_id_col)
    #subject_data = subject_data.sort_values(by=subject_id_col)
    """
    assert len(target_list) == 1  
    for i in tqdm(range(len(subject_data))):
        imageFile_label = (subject_data['image_files'][i], subject_data[target_list[0]][i])
        imageFiles_labels.append(imageFile_label)
    """

    for i in tqdm(range(len(subject_data))):
        target = {} 
        for j, t in enumerate(target_list): 
            target[t] = subject_data[t][i]
        imageFile_label = (subject_data['image_files'][i], target)
        imageFiles_labels.append(imageFile_label)
        
    return imageFiles_labels



def partition_dataset_pretrain(imageFiles,args):
    train_transform = Compose([AddChannel(),
                               Resize(tuple(args.img_size)),
                               RandAxisFlip(prob=0.5),
                               NormalizeIntensity(),
                               ToTensor()])

    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.img_size)),
                             NormalizeIntensity(),
                             ToTensor()])

    # number of total / train,val, test
    num_total = len(imageFiles)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)
    

    # image for MAE training and linear classifier training (linear classifier is trained during linear evaluation protocol) 
    images_train = imageFiles[:num_train]

    # image for validation set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_val = imageFiles[num_train:num_train+num_val]

    # image for test set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    #images_test = imageFiles[num_train+num_val:]

    print("Training Sample: {}".format(len(images_train)))

    train_set = ImageDataset(image_files=images_train,transform=train_transform) 
    val_set = ImageDataset(image_files=images_val,transform=val_transform)
    #test_set = ImageDataset(image_files=images_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    #partition['test'] = test_set

    return partition
## ====================================== ##



def partition_dataset_finetuning(imageFiles_labels, args):
    """train_set for training simCLR
        finetuning_set for finetuning simCLR for prediction task 
        tests_set for evaluating simCLR for prediction task"""
    #if args.shuffle_data: 
    #    random.shuffle(imageFiles_labels)

    images = []
    labels = []

    for imageFile_label in imageFiles_labels:
        image, label = imageFile_label
        images.append(image)
        labels.append(label)


    # sexs = [lbl['sex'] if isinstance(lbl, dict) else lbl for lbl in labels]
    # print("Before undersampling sex distribution:")
    # print(pd.Series(sexs).value_counts())

    if args.undersample:
        keys = args.cat_target + args.num_target
        tuple_labels = [
            tuple(lbl[k] for k in keys)
            for lbl in labels
        ]

        #undersampling
        df = pd.DataFrame({
            'image':       images,
            'tuple_label': tuple_labels,
            'orig_label':  labels
        })

        # 3) Find the minimum class count
        min_count = df['tuple_label'].value_counts().min()

        # 4) Sample min_count from each tuple label
        df_balanced = pd.concat([
            grp.sample(min_count, random_state=getattr(args, 'seed', 42))
            for _, grp in df.groupby('tuple_label')
        ]).sample(frac=1, random_state=getattr(args, 'seed', 42))

        # 5) Restore to original lists
        images = df_balanced['image'].tolist()
        labels = df_balanced['orig_label'].tolist()


    # number of total / train,val, test
    num_total = len(images)
    num_train = int(num_total*args.train_size)
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)
        

    # image and label for SSL training and linear classifier training (linear classifier is trained during linear evaluation protocol) 
    images_train = images[:num_train]
    labels_train = labels[:num_train]

    # image for validation set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]


    ratio = 0.3
    patch_size = (8, 8, 8)
    num_patches = (args.img_size[0] // patch_size[0]) + (args.img_size[1] // patch_size[1]) + (args.img_size[2] // patch_size[2])

    train_transform = Compose([AddChannel(),
                               Resize(tuple(args.img_size)),
                               RandAxisFlip(prob=0.5),
                               NormalizeIntensity(),
    #                           RandCoarseDropout(holes=int(num_patches * ratio), spatial_size=patch_size, dropout_holes=True, fill_value=0,prob=0.3),
                               ToTensor()])

    val_transform = Compose([AddChannel(),
                             Resize(tuple(args.img_size)),
                             NormalizeIntensity(),
                             ToTensor()])

    # image for test set during fine tuning (exactly saying linear classifier training during linear evaluation protocol)
    images_test = images[num_train+num_val:]
    labels_test = labels[num_train+num_val:]

   
    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=val_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set



    #case_control_count(labels_train, 'train', args)
    #case_control_count(labels_val, 'validation', args)
    #case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##
