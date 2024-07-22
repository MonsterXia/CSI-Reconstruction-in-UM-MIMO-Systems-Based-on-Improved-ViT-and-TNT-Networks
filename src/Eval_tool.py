# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/4/26 18:37
# @Function:

import torch

def channels_mix(img, img_height, img_width):
    img_real = torch.reshape(img[:, 0, :, :], (len(img), -1))
    img_imag = torch.reshape(img[:, 1, :, :], (len(img), -1))
    img_c = img_real - 0.5 + 1j * (img_imag - 0.5)
    return img_c


def fre2time(img, img_height, img_width, device):
    img_c = channels_mix(img, img_height, img_width)
    img_f = torch.reshape(img_c, (len(img_c), img_height, img_width))
    padding = torch.zeros(len(img_c), img_height, 257 - img_width)
    padding = padding.to(device)
    img_fft = torch.fft.fft(torch.cat((img_f, padding), dim=2), axis=2)
    img_fft = img_fft[:, :, 0:125]
    return img_fft


def params_calculator(origin, predict, img_height, img_width, device):
    origin_c = channels_mix(origin, img_height, img_width)
    predict_c = channels_mix(predict, img_height, img_width)

    origin_t = fre2time(origin, img_height, img_width, device)
    predict_t = fre2time(predict, img_height, img_width, device)

    n1 = torch.sqrt(torch.sum(torch.conj(origin_t) * origin_t, dim=1)).to(torch.float64)
    n2 = torch.sqrt(torch.sum(torch.conj(predict_t) * predict_t, dim=1)).to(torch.float64)
    aa = torch.abs(torch.sum(torch.conj(origin_t) * predict_t, dim=1))
    rho = torch.mean(aa / (n1 * n2), dim=1)

    predict_t = torch.reshape(predict_t, (len(predict_t), -1))
    origin_t = torch.reshape(origin_t, (len(origin_t), -1))

    power = torch.sum(abs(origin_c) ** 2, dim=1)
    power_d = torch.sum(abs(predict_c) ** 2, dim=1)
    mse = torch.sum(abs(origin_c - predict_c) ** 2, dim=1)
    return rho, mse


def db_conversion(x):
    ref = torch.tensor(1.0)
    return 10.0 * torch.log10(x / ref)
