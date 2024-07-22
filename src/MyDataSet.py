# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/4/25 15:07
# @Function:

import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset_path, img_channels, img_height, img_width):
        mat = sio.loadmat(dataset_path)
        data = mat['HT']
        data = data.astype('float32')
        data = np.reshape(data, (len(data), img_channels, img_height, img_width))
        data_tensor = torch.from_numpy(data)

        self.imgs = data_tensor
        self.labels = data_tensor

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]