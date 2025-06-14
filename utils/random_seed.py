#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


SEED = 126