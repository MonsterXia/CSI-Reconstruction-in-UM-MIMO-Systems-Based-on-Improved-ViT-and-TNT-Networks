# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/5/20 1:59
# @Function:

import os
import warnings

from Eval_tool import *
from enumber import *


warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for envir in Environment:
    for encoded_dim in Digit:
        file_path = "../model/TNT/TNT_{}_{}.pth".format(envir.name, encoded_dim.value)
        if os.path.isfile(file_path):
            print("Load model TNT_{}_{}!".format(envir.name, encoded_dim.value))
            model = torch.load(file_path)
            model = model.to(device)
            print("Turn device successfully!")
        else:
            print("model no exist! Check again!")

        torch.save(model, file_path)
        print("Save!")

