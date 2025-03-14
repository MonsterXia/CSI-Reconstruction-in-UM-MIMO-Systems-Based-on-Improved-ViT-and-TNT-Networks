# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/5/2 17:13
# @Function:


import os
import copy
import time
import warnings
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from TNT import *
from enumber import *
from Eval_tool import *
from MyDataSet import MyDataset


warnings.filterwarnings("ignore")


def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, ncols=128)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        progress_bar.update()
        progress_bar.set_postfix({"train_loss": loss.item()})
    progress_bar.close()
    return running_loss / len(train_loader)


def test(model, test_loader, criterion, device, img_height, img_width):
    model.eval()

    running_loss = 0.0
    total_rho = 0
    total_samples = 0
    progress_bar = tqdm(test_loader, ncols=128)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            rho, mse = params_calculator(outputs, labels, img_height, img_width, device)

            batch_size = labels.shape[0]
            total_rho += torch.sum(rho)
            total_samples += batch_size
            average_rho = total_rho / total_samples
            progress_bar.update()
            progress_bar.set_postfix({"average_rho": average_rho})

    progress_bar.close()
    return running_loss / len(test_loader), average_rho


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("../result/logs/TNT")
    output_list = []

    batch_size = 128
    lr = 0.001
    gamma = 0.1
    num_epochs = 200
    img_channels = 2
    img_height = 32
    img_width = 32
    num_heads = 2
    num_layers = 3
    patch_size = 4
    dropout = 0.1

    # loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    # environment = ['indoor', 'outdoor']
    for envir in Environment:
        # Read the data
        print("Reading the data...")
        if envir.name == 'indoor':
            x_train = MyDataset(dataset_path='../data/DATA_Htrainin.mat', img_channels=img_channels,
                                img_height=img_height, img_width=img_width)
            print("...")
            x_val = MyDataset(dataset_path='../data/DATA_Hvalin.mat', img_channels=img_channels, img_height=img_height,
                              img_width=img_width)
        elif envir.name == 'outdoor':
            x_train = MyDataset(dataset_path='../data/DATA_Htrainout.mat', img_channels=img_channels,
                                img_height=img_height, img_width=img_width)
            print("...")
            x_val = MyDataset(dataset_path='../data/DATA_Hvalout.mat', img_channels=img_channels, img_height=img_height,
                              img_width=img_width)
        print("Read the data successfully")

        # Create the dataloader
        print("Creating the dataloader...")
        train_dataloader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
        print("...")
        val_dataloader = DataLoader(dataset=x_val, batch_size=batch_size, shuffle=False)
        print("Create the dataloader successfully")

        # digit = [16, 32, 64, 128, 512]
        for encoded_dim in Digit:
            # Create/Read the model
            file_path = "../model/TNT/TNT_{}_{}.pth".format(envir.name, encoded_dim.value)
            if os.path.isfile(file_path):
                print("Model TNT_{}_{} exist, try to load model!".format(envir.name, encoded_dim.value))
                model = torch.load(file_path)
                print("Load model successfully!")
                _, rho = test(model, val_dataloader, criterion, device, img_height, img_width)
                best_rho = rho
                print("Model's rho : {}".format(best_rho))
            else:
                model = ImageTransformer(img_size=img_height, patch_size=patch_size, in_channels=img_channels,
                                         embedding_dim=encoded_dim.value, num_heads=num_heads, num_layers=num_layers,
                                         dropout=dropout)
                print("Create model successfully!")
                best_rho = 0
            model = model.to(device)
            best_model = copy.deepcopy(model)

            # optimizer
            optimizer = optim.Adam(params=model.parameters(), lr=lr)

            # scheduler
            milestones = [int(0.5*num_epochs), int(0.75*num_epochs)]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

            for epoch in range(num_epochs):
                start_time = time.time()

                train_loss = train(model, train_dataloader, criterion, optimizer, scheduler, device)
                test_loss, rho = test(model, val_dataloader, criterion, device, img_height, img_width)

                end_time = time.time()
                epoch_time = end_time - start_time

                # best model update
                if rho >= best_rho:
                    best_model = copy.deepcopy(model)
                    best_rho = rho
                    print("Model Updates, best_rho={}".format(best_rho))

                writer.add_scalar("TNT_{}_{}'s value_loss".format(envir.name, encoded_dim.value), test_loss, epoch)
                writer.add_scalar("TNT_{}_{}'s rho".format(envir.name, encoded_dim.value), rho, epoch)

                output_list.append(
                    'Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Epoch Time: {:.4f}s'.format(epoch + 1,
                                                                                                       num_epochs,
                                                                                                       train_loss,
                                                                                                       test_loss,
                                                                                                       epoch_time))
            torch.save(best_model, file_path)

    writer.close()
    print('\n'.join(output_list))


if __name__ == '__main__':
    main()
