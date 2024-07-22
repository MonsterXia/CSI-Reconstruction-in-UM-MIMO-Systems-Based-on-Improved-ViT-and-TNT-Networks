# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/5/6 19:55
# @Function:

import os
import torch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader
from enumber import *
from MyDataSet import MyDataset
from Transformer_ViT_Train import test
from Eval_tool import db_conversion
from fvcore.nn import FlopCountAnalysis

warnings.filterwarnings("ignore")

output_list = []
flops_list = []

batch_size = 128
lr = 0.001
momentum = 0.9
num_epochs = 10
img_channels = 2
img_height = 32
img_width = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_tensor = torch.rand(1, 2, 32, 32)
input_tensor = input_tensor.to(device)

for envir in Environment:

    if envir.name == 'indoor':
        x_test = MyDataset(dataset_path='../data/DATA_Htestin.mat', img_channels=img_channels, img_height=img_height,
                           img_width=img_width)
    elif envir.name == 'outdoor':
        x_test = MyDataset(dataset_path='../data/DATA_Htestout.mat', img_channels=img_channels, img_height=img_height,
                           img_width=img_width)
    test_dataloader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)

    for encoded_dim in Digit:
        file_path = "../model/TNT/TNT_{}_{}.pth".format(envir.name, encoded_dim.value)
        if os.path.isfile(file_path):
            print("Model exist, try to load model!")
            model = torch.load(file_path)
            model = model.to(device)
            print("Load model successfully!")
        else:
            print("model no exist! Check again!")

        criterion = nn.MSELoss()
        criterion = criterion.to(device)

        mse, rho = test(model, test_dataloader, criterion, device, img_height, img_width)

        output_list.append("TNT_{}_{}: \n\tmse:{:.2f}dB\n\trho:{:.4f}".format(envir.name, encoded_dim.value,
                                                                              db_conversion(mse), rho))

        flops = FlopCountAnalysis(model, (input_tensor,))
        flops_list.append("TNT_{}_{}'s FLOPs={}".format(envir.name, encoded_dim.value, flops.total()))


print('\n'.join(output_list))
print('\n'.join(flops_list))