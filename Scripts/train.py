import numpy as np
import pandas as pd
import os
import torch
import torchvision as tv
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import SequentialSampler, BatchSampler
from torch._six import int_classes as _int_classes

from fastai import *
from fastai.vision import *

from matplotlib import patches, patheffects
import matplotlib.pyplot as plt
import PIL
import scipy

import math
import pdb
import glob
import time
import copy
from pathlib import Path

from tqdm import tqdm, tnrange, tqdm_notebook
from time import sleep

from definitions import *

np.random.seed(42)
torch.random.manual_seed(42)

torch.set_default_tensor_type('torch.FloatTensor')

h = 128
w = 160
bs = 8
bptt = 7
lr = 1e-3
epoch = 60

if type == '__main__':
    ransforms = tv.transforms.Compose([tv.transforms.Resize((h, w)), tv.transforms.ToTensor(),
                                       tv.transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    train_ds = Image_Dataset(train_f, transforms)
    val_ds = Image_Dataset(val_f, transforms)

    train_sampler = SequenceBatchSampler(SequentialSampler(train_ds), bs)  # 2,4,8,16,32
    val_sampler = SequenceBatchSampler(SequentialSampler(val_ds), 7, val=True)  # 2,7,14,17

    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=0)

    m = CRNN().cuda()

    train_loss, val_loss = train(100, lr, encode=True)

    train_loss, val_loss = train(60, lr)

    train_loss, val_loss = train(150, lr / 10)
