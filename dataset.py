
# import pathlib
# import numpy as np
# import nibabel as nib

# import torch
# from torch.utils.data import Dataset

# from tools import normalize

# class Dataset(Dataset):
#     """
#     Dataset class for converting the data into batches.
#     The data.Dataset class is a pyTorch class which help
#     in speeding up  this process with effective parallelization
#     """
#     'Characterizes a dataset for PyTorch'

#     def __init__(self, atlas_filename, target_filename, state, path, modality):
#         'Initialization'
#         self.atlas_filename = atlas_filename
#         self.target_filename = target_filename
#         self.state = state
#         self.path = path
#         self.modality = modality
#         self.atlas_num = len(self.atlas_filename)

#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.target_filename)

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # atlas img and atlas
#         if self.modality == 'ct' or self.modality == 'mr':
#             atlas_img_ct = [0] * int(self.atlas_num) 
#             atlas_img_mr = [0] * int(self.atlas_num) 
#             atlas_label_ct = [0] * int(self.atlas_num)
#             atlas_label_mr = [0] * int(self.atlas_num)
#             atlas_ID_ct = [0] * int(self.atlas_num)
#             atlas_ID_mr = [0] * int(self.atlas_num)
#         else :
#             atlas_img_ct = [0] * int(self.atlas_num/2) 
#             atlas_img_mr = [0] * int(self.atlas_num/2) 
#             atlas_label_ct = [0] * int(self.atlas_num/2)
#             atlas_label_mr = [0] * int(self.atlas_num/2)
#             atlas_ID_ct = [0] * int(self.atlas_num/2)
#             atlas_ID_mr = [0] * int(self.atlas_num/2)
        
#         target_ID = self.target_filename[index]
#         j_ct = 0
#         j_mr = 0
#         for i in range(self.atlas_num):
#             if self.atlas_filename[i].split('_')[0] == 'MR':
#                 atlas_img_path = pathlib.Path(self.path,'train',self.atlas_filename[i], 'image.npy')
#                 atlas_ID_mr[j_mr] = self.atlas_filename[i]
#                 atlas_img_mr[j_mr] = np.load(atlas_img_path)
#                 atlas_img_mr[j_mr] = atlas_img_mr[j_mr][np.newaxis,...]
#                 atlas_img_mr[j_mr] = torch.Tensor(atlas_img_mr[j_mr])
#                 atlas_img_mr[j_mr] = normalize(atlas_img_mr[j_mr],'win_maxmin')
                
#                 atlas_label_path = pathlib.Path(self.path,'train',self.atlas_filename[i], 'seg.npy')
#                 atlas_label_mr[j_mr] = np.load(atlas_label_path)[ np.newaxis,...]
#                 # print("MR", atlas_label_mr[j_mr].max(), atlas_label_mr[j_mr].min())
#                 atlas_label_mr[j_mr] = np.round(atlas_label_mr[j_mr])
#                 atlas_label_mr[j_mr] = torch.Tensor(atlas_label_mr[j_mr])
                
#                 j_mr += 1
                
#             if self.atlas_filename[i].split('_')[0] == 'CT':
#                 atlas_img_path = pathlib.Path(self.path,'train',self.atlas_filename[i], 'procimg.nii.gz')
#                 atlas_ID_ct[j_ct] = self.atlas_filename[i]
#                 atlas_img_ct[j_ct] = nib.load(atlas_img_path).get_fdata()
#                 atlas_img_ct[j_ct] = atlas_img_ct[j_ct][np.newaxis,...]
#                 atlas_img_ct[j_ct] = torch.Tensor(atlas_img_ct[j_ct])
#                 atlas_img_ct[j_ct] = normalize(atlas_img_ct[j_ct],'win')
                
#                 atlas_label_path = pathlib.Path(self.path,'train',self.atlas_filename[i], 'seg.nii.gz')
#                 atlas_label_ct[j_ct] = nib.load(atlas_label_path).get_fdata()[ np.newaxis,...]
#                 # print("CT", atlas_label_ct[j_ct].max(), atlas_label_ct[j_ct].min())
#                 atlas_label_ct[j_ct] = np.round(atlas_label_ct[j_ct])
#                 atlas_label_ct[j_ct] = torch.Tensor(atlas_label_ct[j_ct])
                
#                 j_ct += 1

#         # target
        
#         if target_ID.split('_')[0] == 'MR':
#             target_img_path = pathlib.Path(self.path,self.state,target_ID, 'image.npy')
#             target_img = np.load(target_img_path)
#             target_img = target_img[np.newaxis,...] #[1,96,96,64]
#             target_img = torch.Tensor(target_img)
#             target_img = normalize(target_img,'win_maxmin') 
        
#         if target_ID.split('_')[0] == 'CT':
#             target_img_path = pathlib.Path(self.path,self.state,target_ID, 'procimg.nii.gz')
#             nii = nib.load(target_img_path)
#             target_img = nii.get_fdata()
#             target_img = target_img[np.newaxis,...] #[1,96,96,64]
#             target_img = torch.Tensor(target_img)
#             target_img = normalize(target_img,'win')

#         sample = {'atlas_img_ct':atlas_img_ct, 'atlas_img_mr':atlas_img_mr, 'atlas_label_ct':atlas_label_ct, 'atlas_label_mr':atlas_label_mr, 'target_img':target_img, 'target_ID':target_ID}
#         return sample

import pathlib
import numpy as np
import nibabel as nib
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset

from tools import *

class SegDataset(Dataset):
    '''
    
    '''
    
    def __init__(self, atlas_list, target_filename, state, path):
        self.atlas_list = atlas_list
        self.target_filename = target_filename
        self.path = path
        self.state = state
        self.atlas_img_ct, self.atlas_img_mr, self.atlas_ct_meta, self.atlas_mr_meta, self.atlas_label_ct, self.atlas_label_mr = [], [], [], [], [], []
        for atlas in atlas_list:
            atlas_img_path = pathlib.Path(self.path, "train", atlas, "procimg.nii.gz")
            atlas_img, header, affine, spacing = read_nii(atlas_img_path)
            atlas_img = torch.Tensor(atlas_img[np.newaxis, ...])
            
            atlas_label_path = pathlib.Path(self.path, "train", atlas, "seg.nii.gz")
            atlas_label, _, _, _ = read_nii(atlas_label_path)
            atlas_label = torch.Tensor(np.round(atlas_label[np.newaxis, ...]))
            if atlas.split('_')[0] == 'CT':
                atlas_img = normalize(atlas_img, "win")
                save_nii(atlas_img, header, affine, "./procimgct.nii.ga")
                self.atlas_img_ct.append(atlas_img)
                self.atlas_label_ct.append(atlas_label)
                self.atlas_ct_meta.append([header, affine, spacing])
            else:
                atlas_img = normalize(atlas_img, "win_maxmin")
                save_nii(atlas_img, header, affine, "./procimgct.nii.ga")
                self.atlas_img_mr.append(atlas_img)
                self.atlas_label_mr.append(atlas_label)
                self.atlas_mr_meta.append([header, affine, spacing])
        # self.atlas_img_ct = np.asarray(self.atlas_img_ct, np.float32)
        # print(self.atlas_img_ct.shape)
        # self.atlas_img_mr = np.asarray(self.atlas_img_mr, np.float32)
        # self.atlas_label_ct = np.asarray(self.atlas_label_ct, np.float32)
        # self.atlas_label_mr = np.asarray(self.atlas_label_mr, np.float32)
        
    def __len__(self):
        return len(self.target_filename)
    
    def __getitem__(self, index):
        target_ID = self.target_filename[index]
        target_img_path = pathlib.Path(self.path, self.state, target_ID, 'procimg.nii.gz')
        target_img, header, affine, spacing = read_nii(target_img_path)
        target_img = torch.Tensor(target_img[np.newaxis, ...])
        target_img = normalize(target_img, "win_maxmin") if target_ID.split('_')[0] == 'MR' \
                else normalize(target_img, "win")
        if self.state == "train":
            sample = {'atlas_img_ct': self.atlas_img_ct, 'atlas_img_mr': self.atlas_img_mr, 'atlas_label_ct': self.atlas_label_ct, 'atlas_label_mr': self.atlas_label_mr, 'target_img': target_img, 'target_ID': target_ID}
        else:
            target_label_path = pathlib.Path(self.path, self.state, target_ID, "seg.nii.gz")
            target_label, header, affine, spacing = read_nii(target_label_path)
            target_label = torch.Tensor(np.round(target_label[np.newaxis, ...]))
            # print(target_label_path)
            sample = {'atlas_img_ct': self.atlas_img_ct, 'atlas_img_mr': self.atlas_img_mr,  'atlas_label_ct': self.atlas_label_ct, 'atlas_label_mr': self.atlas_label_mr, 'target_img': target_img, 'target_ID': target_ID, 'target_label': target_label, 'target_path': str(target_img_path)}
        return sample
