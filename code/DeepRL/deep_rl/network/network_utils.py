#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import *


class BaseNet:
    def __init__(self):
        pass


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    #nn.init.normal_(layer.weight, mean=0, std=1.0)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def layer_random(layer, w_scale=1.0):
    #nn.init.orthogonal_(layer.weight.data)
    nn.init.normal_(layer.weight, mean=0, std=1.0)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
