# torchio - data augmentation Reference: https://blog.promedius.ai/torchioreul-iyonghan-3d-segmentation/
import copy
import time
import pprint

import torch
import torchio
import torchio as tio
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random

sns.set_style("whitegrid", {'axes.grid' : False})
# config InlineBackend.figure_format = 'retina'
torch.manual_seed(14041931)

print('TorchIO version:', tio.__version__)
print('Last run on', time.ctime())


import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_mridataset
from PIL import Image
import random
from glob import glob
import nibabel as nib
import numpy as np
from torchvision.transforms import functional as F # torchvision.transforms.functional.to_tensor(pic)

##################### helee ######################
import torch
import numpy as np
import torchio as tio
from pathlib import Path
from torchio.transforms import HistogramStandardization
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation, 
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

################################### delete ##########################################
# training_batch_size = 4
# validation_batch_size = 2 * training_batch_size

# training_loader = torch.utils.data.DataLoader(dataset = training_set, batch_size = training_batch_size, shuffle = True,
#                                               num_workers=0)

# validation_loader = torch.utils.data.DataLoader(dataset = validation_set, batch_size = validation_batch_size,
#                                                 num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CHANNELS_DIMENSION = 1      #???
# SPATIAL_DIMENSIONS = 2,3,4  #???
################################### delete ##########################################

def prepare_batch(batch, device):
  inputs = batch['MRI'][tio.DATA].to(device)
  foreground = batch['LABEL'][tio.DATA].type(torch.float32).to(device)
  background = 1 - foreground
  targets = torch.cat((background, foreground), dim = CHANNELS_DIMENSION)
  return inputs, targets


def get_dice_score(output, target, epsilon = 1e-9):
  p0 = output
  g0 = target
  p1 = 1 - p0
  g1 = 1 - g0
  tp = (p0 * g0).sum(dim = SPATIAL_DIMENSIONS)
  fp = (p0 * g1).sum(dim = SPATIAL_DIMENSIONS)
  fn = (p1 * g0).sum(dim = SPATIAL_DIMENSIONS)
  num = 2 * tp
  denom = 2 * tp + fp + fn + epsilon
  dice_score = num / denom

  return dice_score

def get_dice_loss(output, target):
  return 1 - get_dice_score(output, target)

# def forward(model, inputs):
#   with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category = UserWarning)
#     logits = model(inputs)
#   return logits

# def get_model_and_optimizer(device):
#   model = UNet(
#       in_channels = 1, 
#       out_classes = 2,
#       dimensions = 3,
#       num_encoding_blocks = 4,
#       out_channels_first_layer = 32,
#       normalization = 'batch',
#       padding = True,
#       activation = 'PReLU',
#   ).to(device)
#   optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)
#   return model, optimizer

##################################################

# affine_transform = tio.RandomAffine()
# transformed_tensor = affine_transform(tensor)
# type(transformed_tensor)
# <class 'torch.Tensor'>
# array = np.random.rand(1, 256, 256, 159)
# transformed_array = affine_transform(array)
# type(transformed_array)
# <class 'numpy.ndarray'>
# subject = tio.datasets.Colin27()
# transformed_subject = affine_transform(subject)
# transformed_subject

##################### helee ######################
def histogram_standardization3D(t1_paths=['/home/haneollee/myworkspace/cyclegan/datasets/mri_nii/*/DSC.nii.gz'], t2_paths=['/home/haneollee/myworkspace/cyclegan/datasets/mri_nii/*/DCE.nii.gz]']):
    from glob import glob
    # t1_paths = glob(t1_paths) #이게 왜 에러가 날까?
    # t2_paths = glob(t2_paths)
    print(t1_paths)
    print(t2_paths)
    t1_landmarks_path = Path('t1_landmarks.npy')
    t2_landmarks_path = Path('t2_landmarks.npy')
    t1_landmarks = (
        t1_landmarks_path
        if t1_landmarks_path.is_file()
        else HistogramStandardization.train(t1_paths)
    )
    torch.save(t1_landmarks, t1_landmarks_path)

    t2_landmarks = (
        t2_landmarks_path
        if t2_landmarks_path.is_file()
        else HistogramStandardization.train(t2_paths)
    )
    torch.save(t2_landmarks, t2_landmarks_path)

    landmarks_dict = {
        't1': t1_landmarks,
        't2': t2_landmarks,
    }
    transform = HistogramStandardization(landmarks_dict)

    return transform
##################################################

def nii_to_numpy(path):
    img = nib.load(path)
    img = np.array(img.dataobj)
    return img

def load_numpy(path):
    img = np.load(path)
    return img


class MricycleganDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot) # os.path.join(opt.dataroot, opt.phase + 'A')   # # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot)  # create a path '/path/to/data/trainB'

        self.ref_size_A = torch.zeros(opt.crop_size,opt.crop_size,opt.input_nc) #$#$
        self.ref_size_B = torch.zeros(opt.crop_size,opt.crop_size,opt.output_nc)

        if self.opt.dummy_mode == False:
            if opt.input_type == '.npy':                
                self.A_paths = sorted(make_mridataset(self.dir_A, opt.max_dataset_size, '*/*/input_DSC_full.npy', input_type=opt.input_type))
                self.B_paths = sorted(make_mridataset(self.dir_B, opt.max_dataset_size, '*/*/label_old.npy', input_type=opt.input_type))
            elif opt.input_type =='.nii.gz':
                self.A_paths = sorted(make_mridataset(self.dir_A, opt.max_dataset_size, '*/*/input_DSC_full.nii.gz', input_type=opt.input_type)) 
                self.B_paths = sorted(make_mridataset(self.dir_B, opt.max_dataset_size, '*/*/label_ktrans_fixed_full.nii.gz', input_type=opt.input_type))
            else:
                print('Error! opt.input_type .. must be in [.npy , .nii.gz]')
            
    ################################## subtract fail data (192,192,40,60 이 아닌 case들은 제거한다.) ##################################

            # python train_MRI.py --dataroot /home/haneollee/myworkspace/cyclegan/datasets/mri_nii --dataset_mode mricyclegan --name mricyclegan --model mricyclegan --input_nc 60 --output_nc 60

            print(f'self.A_paths[:5]:{self.A_paths[:5]}')
            print(f'self.B_paths[:5]:{self.B_paths[:5]}')

            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B
            btoA = self.opt.direction == 'BtoA'
            input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
            output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
            # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
            # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))


    ######################## helee : training, validation별로 분기문 만들어야 할 듯함 ########################

            self.z_selection = 20

            self.training_transform = tio.Compose([
                tio.ToCanonical(),
                tio.RescaleIntensity((-1, 1)),
                # tio.Resample(4),
                # tio.CropOrPad((192,192,60)),
                # tio.RandomMotion(p=0.2),
                # tio.RandomBiasField(p=0.3),
                # tio.RandomNoise(p=0.5),
                # tio.RandomFlip(axes=(0,)),
                # tio.RandomAffine(),
                ZNormalization(),
            ])

            self.validation_transform = tio.Compose([
                tio.ToCanonical(),
                tio.RescaleIntensity((-1, 1)),
                # tio.Resample(4),
                # tio.CropOrPad((192,192,60)),
                ZNormalization(),
            ])

        else:
            print('dummy_mode == True setting!\n\n')
            self.A_size = 20  # get the size of dataset A
            self.B_size = 20
            # A = torch.rand(1, 256, 256, 60)
            # B = torch.rand(1, 256, 256, 1)


    ##################################################### visualize ####################################################


    #################################################################################################################################
    ###################################################### normalize function 넣기 ##################################################


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.dummy_mode == False:
            print(f'self.A_paths[index % self.A_size]{self.A_paths[index % self.A_size]}')
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]

            print(A_path)
            print(B_path)

            # 1-1 load np file
                # 1-2 convert to nii file
            A_img = tio.ScalarImage(A_path)
            B_img = tio.ScalarImage(B_path)
            
            print(f'type(A_img):{type(A_img)}')
            print(f'type(B_img):{type(B_img)}')

            A_img_augmented = self.training_transform(A_img)         # apply the transform
            B_img_augmented = self.training_transform(B_img)

            print(f'type(A_img_augmented):{type(A_img_augmented)}')

            print(f'type(B_img_augmented):{type(B_img_augmented)}')

    ##################################### nii.gz version #####################################
            if self.opt.input_type == '.nii.gz':
                A = A_img_augmented.data
                B = B_img_augmented.data

    ################ 4D인경우!! ################
                # print(random.randint(0,59))
                # self.z_selection = 39
                # A = A_img_augmented.data[:,:,:,self.z_selection]
                # B = B_img_augmented.data[:,:,:,self.z_selection]
                # print(self.z_selection)
                # print(f'A_img_augmented.data: {A_img_augmented.data}')
    ################ 4D인경우!! ################
            elif self.opt.input_type == '.npy': # /*/*/input_DSC_full.npy , 192, 192, 60 불러올 것!
                # self.z_selection = random.randint(0,59)
                A = A_img_augmented.data
                B = B_img_augmented.data

            print(type(A))
            print(f'A.shape: {A.shape}')           # (1,256,256,60)
            print(A.squeeze().shape) # (256,256,60)

            print(f'B.shape:{B.shape}')        # (1,256,256,1)
            print(B.squeeze(0).shape) # (256,256,1)

            print(f'A.transpose(2,3).transpose(1,2).shape:{A.transpose(2,3).transpose(1,2).shape}')


        
        else:
            A = torch.rand(1, 256, 256, 60)
            B = torch.rand(1, 256, 256, 1)
            A_paths=glob('./datasets_dummy/A_*.npy')
            # [random.randint(0,199)]
            B_paths=glob('./datasets_dummy/B_*.npy')
            # [random.randint(0,199)]
            print(len(A_paths))
            print(len(B_paths))
            A_path = A_paths[random.randint(0,3)]
            B_path = B_paths[random.randint(0,3)]


        ### squeeze 하는 게 맞나? (Yes!)
        A=A.transpose(2,3).transpose(1,2).squeeze()
        B=B.transpose(2,3).transpose(1,2).squeeze(0)
        
        if A.shape != self.ref_size_A or B.shape!= self.ref_size_B:
         
            with open(os.path.join(self.dir_A,'mricyclegan_error.txt'), 'a') as f:
                f.write(f' A_path: {A.shape}\n B_path: {B.shape} \n\n')

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.opt.dummy_mode == True:
            self.A_size = 200
            self.B_size = 200

        return max(self.A_size, self.B_size)
