# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/5/2 13:37
# @Function:

from Test import test
from Eval_tool import db_conversion
import os

import torch.optim as optim
from torch.utils.data import DataLoader

from MyDataSet import MyDataset
import warnings

from TNT import *

warnings.filterwarnings("ignore")

output_list = []

batch_size = 128
lr = 0.001
num_epochs = 3
img_channels = 2
img_height = 32
img_width = 32
embedding_dim = 16
num_heads = 2
num_layers = 3
patch_size = 4
dropout = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()
criterion = criterion.to(device)

envir = 'indoor'
print("Reading the data...")
if envir == 'indoor':
    x_test = MyDataset(dataset_path='../data/DATA_Htestin.mat', img_channels=img_channels, img_height=img_height,
                       img_width=img_width)
elif envir == 'outdoor':
    x_test = MyDataset(dataset_path='../data/DATA_Htestout.mat', img_channels=img_channels, img_height=img_height,
                       img_width=img_width)
print("Read the data successfully")

print("Creating the dataloader...")
test_dataloader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)
print("Create the dataloader successfully")

encoded_dim = embedding_dim
file_path = "../model/TNT/TNT_{}_{}.pth".format(envir, encoded_dim)
if os.path.isfile(file_path):
    print("Model TNT_{}_{} exist, try to load model!".format(envir, encoded_dim))
    model = torch.load(file_path)
    print("Load model successfully!")
else:
    print("model no exist! Check again!")
model = model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=lr)

mse, rho = test(model, test_dataloader, criterion, device, img_height, img_width)

output_list.append("TNT_{}_{}: \n\tmse:{:.2f}dB\n\trho:{:.2f}".format(envir, encoded_dim, db_conversion(mse), rho))

print('\n'.join(output_list))
