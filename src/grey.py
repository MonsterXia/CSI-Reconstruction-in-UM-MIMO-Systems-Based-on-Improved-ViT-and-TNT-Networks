# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/6/4 16:33
# @Function:

import os
import torch
import warnings
import matplotlib.pyplot as plt
from MyDataSet import MyDataset
from matplotlib.gridspec import GridSpec
from torchvision.transforms import Normalize


warnings.filterwarnings("ignore")


# Get one from the dataset to generate image, use index to select CSI data, default 0
def getdata(envir, device, img_channels=2, img_height=32, img_width=32, index=0):
    if envir == 'indoor':
        x_test = MyDataset(dataset_path='../data/DATA_Htestin.mat', img_channels=img_channels, img_height=img_height,
                           img_width=img_width)
    elif envir == 'outdoor':
        x_test = MyDataset(dataset_path='../data/DATA_Htestout.mat', img_channels=img_channels, img_height=img_height,
                           img_width=img_width)

    data = x_test[index]
    input = data[0]

    input = input.unsqueeze(0)
    input = input.to(device)

    return input


def getmodel(envir, encoded_dim, model_type, device):
    file_path = "../model/{}/{}_{}_{}.pth".format(model_type, model_type, envir, encoded_dim)

    if os.path.isfile(file_path):
        print("Model exist, try to load model!")
        model = torch.load(file_path)
        model = model.to(device)
        print("Load model successfully!")
    else:
        print("model no exist! Check again!")

    model = model.to(device)
    return model


def show_img(input, model1, model2, model3, result_root):
    output1 = model1(input)
    output2 = model2(input)
    output3 = model3(input)

    input = input.squeeze().to("cpu")
    output1 = output1.squeeze().to("cpu")
    output2 = output2.squeeze().to("cpu")
    output3 = output3.squeeze().to("cpu")

    real_input = input[0]
    imag_input = input[1]
    abs_input = real_input ** 2 + imag_input ** 2
    real_output1 = output1[0]
    imag_output1 = output1[1]
    abs_output1 = real_output1 ** 2 + imag_output1 ** 2
    real_output2 = output2[0]
    imag_output2 = output2[1]
    abs_output2 = real_output2 ** 2 + imag_output2 ** 2
    real_output3 = output3[0]
    imag_output3 = output3[1]
    abs_output3 = real_output3 ** 2 + imag_output3 ** 2

    real_input = real_input.unsqueeze(0)
    imag_input = imag_input.unsqueeze(0)
    abs_input = abs_input.unsqueeze(0)
    real_output1 = real_output1.unsqueeze(0)
    imag_output1 = imag_output1.unsqueeze(0)
    abs_output1 = abs_output1.unsqueeze(0)
    real_output2 = real_output2.unsqueeze(0)
    imag_output2 = imag_output2.unsqueeze(0)
    abs_output2 = abs_output2.unsqueeze(0)
    real_output3 = real_output3.unsqueeze(0)
    imag_output3 = imag_output3.unsqueeze(0)
    abs_output3 = abs_output3.unsqueeze(0)

    normalize = Normalize(mean=[0.5], std=[0.5])

    real_gray_input = normalize((real_input * 255).type(torch.float32)).type(torch.uint8)
    imag_gray_input = normalize((imag_input * 255).type(torch.float32)).type(torch.uint8)
    abs_gray_input = normalize((abs_input * 255).type(torch.float32)).type(torch.uint8)
    real_gray_output1 = normalize((real_output1 * 255).type(torch.float32)).type(torch.uint8)
    imag_gray_output1 = normalize((imag_output1 * 255).type(torch.float32)).type(torch.uint8)
    abs_gray_output1 = normalize((abs_output1 * 255).type(torch.float32)).type(torch.uint8)
    real_gray_output2 = normalize((real_output2 * 255).type(torch.float32)).type(torch.uint8)
    imag_gray_output2 = normalize((imag_output2 * 255).type(torch.float32)).type(torch.uint8)
    abs_gray_output2 = normalize((abs_output2 * 255).type(torch.float32)).type(torch.uint8)
    real_gray_output3 = normalize((real_output3 * 255).type(torch.float32)).type(torch.uint8)
    imag_gray_output3 = normalize((imag_output3 * 255).type(torch.float32)).type(torch.uint8)
    abs_gray_output3 = normalize((abs_output3 * 255).type(torch.float32)).type(torch.uint8)

    gs = GridSpec(4, 3, figure=plt.figure(figsize=(10, 6)))

    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(real_gray_input.squeeze().numpy(), cmap='gray')
    ax1.set_title('Real Input')
    ax1.axis('off')

    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(imag_gray_input.squeeze().numpy(), cmap='gray')
    ax2.set_title('Imaginary Input')
    ax2.axis('off')

    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(abs_gray_input.squeeze().numpy(), cmap='gray')
    ax3.set_title('Magnitude Input')
    ax3.axis('off')

    ax4 = plt.subplot(gs[1, 0])
    ax4.imshow(real_gray_output1.squeeze().numpy(), cmap='gray')
    ax4.set_title('Real Output of Csi-Net')
    ax4.axis('off')

    ax5 = plt.subplot(gs[1, 1])
    ax5.imshow(imag_gray_output1.squeeze().numpy(), cmap='gray')
    ax5.set_title('Imaginary Output of Csi-Net')
    ax5.axis('off')

    ax6 = plt.subplot(gs[1, 2])
    ax6.imshow(abs_gray_output1.squeeze().numpy(), cmap='gray')
    ax6.set_title('Magnitude Output of Csi-Net')
    ax6.axis('off')

    ax7 = plt.subplot(gs[2, 0])
    ax7.imshow(real_gray_output2.squeeze().numpy(), cmap='gray')
    ax7.set_title('Real Output of Csi-ViT')
    ax7.axis('off')

    ax8 = plt.subplot(gs[2, 1])
    ax8.imshow(imag_gray_output2.squeeze().numpy(), cmap='gray')
    ax8.set_title('Imaginary Output of Csi-ViT')
    ax8.axis('off')

    ax9 = plt.subplot(gs[2, 2])
    ax9.imshow(abs_gray_output2.squeeze().numpy(), cmap='gray')
    ax9.set_title('Magnitude Output of Csi-ViT')
    ax9.axis('off')

    ax10 = plt.subplot(gs[3, 0])
    ax10.imshow(real_gray_output3.squeeze().numpy(), cmap='gray')
    ax10.set_title('Real Output of Csi-TNT')
    ax10.axis('off')

    ax11 = plt.subplot(gs[3, 1])
    ax11.imshow(imag_gray_output3.squeeze().numpy(), cmap='gray')
    ax11.set_title('Imaginary Output of Csi-TNT')
    ax11.axis('off')

    ax12 = plt.subplot(gs[3, 2])
    ax12.imshow(abs_gray_output3.squeeze().numpy(), cmap='gray')
    ax12.set_title('Magnitude Output of TNT')
    ax12.axis('off')

    plt.tight_layout()
    plt.savefig(result_root)
    plt.show()


img_channels = 2
img_height = 32
img_width = 32

envir = 'outdoor'
encoded_dim = 64
model_type = 'TNT'

img_save_root = r"../result/output/output.png"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input = getdata(envir=envir, device=device, img_channels=img_channels, img_height=img_height, img_width=img_width)

TNT_model = getmodel(envir=envir, encoded_dim=encoded_dim, model_type="TNT", device=device)
Transformer_ViT_model = getmodel(envir=envir, encoded_dim=encoded_dim, model_type="Transformer_ViT", device=device)
CsiNet_model = getmodel(envir=envir, encoded_dim=encoded_dim, model_type="CsiNet", device=device)

show_img(input=input, model1=CsiNet_model, model2=Transformer_ViT_model, model3=TNT_model, result_root= img_save_root)

