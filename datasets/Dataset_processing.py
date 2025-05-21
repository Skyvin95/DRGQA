import os
import scipy
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from resize_center import img_resize
from random_crop_same import resize_crop
from skimage import io, color
from skimage.feature import local_binary_pattern

transform = transforms.ToTensor() 
data_path = '/mnt/mdisk/Datasets/LEISD'
matdata = scipy.io.loadmat('/mnt/mdisk/Datasets/LEISD/MOS.mat')
data = {}
data['Enhance_image'] = matdata['img_path'].flatten()    # enhanced images
data['Raw_name'] = matdata['Raw_name'].flatten()         # degraded images
data['MOS'] = matdata['MOS'].flatten()

# LBP_params
r = 1
n = 8 * r

class Dataset(Dataset):
    def __init__(self):
        self.raw_imgs = data['Raw_name']
        self.enhance_imgs = data['Enhance_image']
        self.MOS = data['MOS']
        self.transform = transform

    def __len__(self):
        return len(self.enhance_imgs)

    def __getitem__(self, idx):
        raw_img = Image.open(os.path.join(data_path, self.raw_imgs[idx][0].replace('\\', '/'))).convert('RGB')
        enhance_img = Image.open(os.path.join(data_path, self.enhance_imgs[idx][0].replace('\\', '/'))).convert('RGB')
        # raw_img_resized = img_resize(raw_img)
        # enhance_img_resized = img_resize(enhance_img)
        raw_img_resized, enhance_img_resized = resize_crop(raw_img, enhance_img)
        raw_img = self.transform(color.rgb2lab(np.array(raw_img_resized)))
        enhance_img = self.transform(color.rgb2lab(np.array(enhance_img_resized)))

        # texture residual map
        raw_img_G = cv2.cvtColor(np.asarray(raw_img_resized), cv2.COLOR_RGB2GRAY)
        raw_lbp = local_binary_pattern(raw_img_G, n, r)
        enhance_img_G = cv2.cvtColor(np.asarray(enhance_img_resized), cv2.COLOR_RGB2GRAY)
        enhance_lbp = local_binary_pattern(enhance_img_G, n, r)
        residual_tex = abs(enhance_lbp - raw_lbp)
        residual_tex = self.transform(residual_tex).cuda()

        # color residual map
        raw_img_YCbCr = cv2.cvtColor(np.array(raw_img_resized), cv2.COLOR_RGB2YCrCb)
        raw_Y, raw_Cr, raw_Cb = cv2.split(raw_img_YCbCr)
        raw_CBR = raw_Cb + raw_Cr
        enhance_img_YCbCr = cv2.cvtColor(np.array(enhance_img_resized), cv2.COLOR_RGB2YCrCb)
        enhance_Y, enhance_Cr, enhance_Cb = cv2.split(enhance_img_YCbCr)
        enhance_CBR = enhance_Cb + enhance_Cr
        residual_color = abs(enhance_CBR - raw_CBR)
        residual_color = self.transform(residual_color).cuda()

        mos = self.MOS[idx]

        return raw_img, enhance_img, residual_tex, residual_color, mos