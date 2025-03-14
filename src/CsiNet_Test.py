# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/4/25 15:03
# @Function:

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from enumber import *
from MyDataSet import MyDataset
from CsiNet_Train import test
from Eval_tool import db_conversion

output_list = []

batch_size = 128
lr = 0.001
momentum = 0.9
num_epochs = 1
img_channels = 2
img_height = 32
img_width = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for envir in Environment:

    if envir.name == 'indoor':
        x_test = MyDataset(dataset_path='../data/DATA_Htestin.mat', img_channels=img_channels, img_height=img_height,
                           img_width=img_width)
    elif envir.name == 'outdoor':
        x_test = MyDataset(dataset_path='../data/DATA_Htestout.mat', img_channels=img_channels, img_height=img_height,
                           img_width=img_width)
    test_dataloader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)

    for encoded_dim in Digit:
        file_path = "../model/CsiNet/CsiNet_{}_{}.pth".format(envir.name, encoded_dim.value)
        if os.path.isfile(file_path):
            print("Model exist, try to load model!")
            model = torch.load(file_path)
            print("Load model successfully!")
        else:
            print("model no exist! Check again!")

        criterion = nn.MSELoss()
        criterion = criterion.to(device)

        mse, rho = test(model, test_dataloader, criterion, device, img_height, img_width)


        output_list.append("CsiNet_{}_{}: \n\t mse:{:.2f}dB\n\trho:{:.2f}".format(envir.name, encoded_dim.value, db_conversion(mse), rho))

print('\n'.join(output_list))








