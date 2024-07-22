# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/4/18 16:05
# @Function:

import os
import copy
import time
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CsiNet import CsiNet
from MyDataSet import MyDataset
from Eval_tool import *
from enumber import *


warnings.filterwarnings("ignore")

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def test(model, test_loader, criterion, device, img_height, img_width):
    model.eval()

    running_loss = 0.0
    total_rho = 0
    total_samples = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            rho, mse = params_calculator(outputs, labels, img_height, img_width, device)

            batch_size = labels.shape[0]
            total_rho += torch.sum(rho)
            total_samples += batch_size
        average_rho = total_rho / total_samples

    return running_loss / len(test_loader), average_rho


def main():
    writer = SummaryWriter("../result/logs/CsiNet")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    lr = 0.001
    num_epochs = 1000
    img_channels = 2
    img_height = 32
    img_width = 32


    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    for envir in Environment:
        if envir.name == 'indoor':
            x_train = MyDataset(dataset_path='../data/DATA_Htrainin.mat', img_channels=img_channels, img_height=img_height,
                                img_width=img_width)
            x_val = MyDataset(dataset_path='../data/DATA_Hvalin.mat', img_channels=img_channels, img_height=img_height,
                              img_width=img_width)
        elif envir.name == 'outdoor':
            x_train = MyDataset(dataset_path='../data/DATA_Htrainout.mat', img_channels=img_channels, img_height=img_height,
                                img_width=img_width)
            x_val = MyDataset(dataset_path='../data/DATA_Hvalout.mat', img_channels=img_channels, img_height=img_height,
                              img_width=img_width)

        train_dataloader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=x_val, batch_size=batch_size, shuffle=False)

        for encoded_dim in Digit:
            file_path = "../model/CsiNet/CsiNet_{}_{}.pth".format(envir.name, encoded_dim.value)
            if os.path.isfile(file_path):
                print("Model CsiNet_{}_{} exist, try to load model!".format(envir.name, encoded_dim.value))
                model = torch.load(file_path)
                print("Load model successfully!")
                _, rho = test(model, val_dataloader, criterion, device, img_height, img_width)
                best_rho = rho
                print("Model's rho : {}".format(best_rho))
            else:
                model = CsiNet(img_channels, img_height, img_width, encoded_dim.value)
                print("Create model successfully!")
                best_rho = 0
            model = model.to(device)
            best_model = copy.deepcopy(model)

            optimizer = optim.Adam(params=model.parameters(), lr=lr)

            for epoch in range(num_epochs):
                start_time = time.time()

                train_loss = train(model, train_dataloader, criterion, optimizer, device)
                test_loss, rho = test(model, val_dataloader, criterion, device, img_height, img_width)

                end_time = time.time()
                epoch_time = end_time - start_time

                if rho >= best_rho:
                    best_model = copy.deepcopy(model)
                    best_rho = rho
                    print("Model Updates, best_rho={}".format(best_rho))
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Epoch Time: {:.4f}s'.format(epoch + 1,
                                                                                                         num_epochs,
                                                                                                         train_loss,
                                                                                                         test_loss,
                                                                                                         epoch_time))

                writer.add_scalar("CsiNet_{}_{}'s value_loss".format(envir.name, encoded_dim.value), test_loss, epoch)
                writer.add_scalar("CsiNet_{}_{}'s rho".format(envir.name, encoded_dim.value), rho, epoch)
            torch.save(best_model, file_path)

    writer.close()


if __name__ == '__main__':
    main()
