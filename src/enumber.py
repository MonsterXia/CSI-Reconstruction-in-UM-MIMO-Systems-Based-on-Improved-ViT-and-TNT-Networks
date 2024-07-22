# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/4/26 20:46
# @Function:

import enum


class Environment(enum.Enum):
    indoor = 0
    outdoor = 1


class Digit(enum.Enum):
    sixteen = 16
    thirty_two = 32
    sixty_four = 64
    one_twenty_eight = 128
    five_twelve = 512
