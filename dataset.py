import os
import itertools
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


class ref_dataset(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 rf_transform=None,
                 real=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.rf_transform = rf_transform
        self.real = real
        if real:
            self.ids = sorted(os.listdir(root))
        else:
            self.ids = sorted(os.listdir(os.path.join(root, 'I')))

    def __getitem__(self, index):
        img = self.ids[index]
        if self.real:
            input = Image.open(os.path.join(self.root, img)).convert('RGB')
            if self.transform is not None:
                input = self.transform(input)
            return input
        else:
            input = Image.open(os.path.join(self.root, 'I', img)).convert('RGB')
            target = Image.open(os.path.join(self.root, 'B', img)).convert('RGB')
            target_rf = Image.open(os.path.join(self.root, 'R', img)).convert('RGB')
            if self.transform is not None:
                input = self.transform(input)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.rf_transform is not None:
                target_rf = self.rf_transform(target_rf)
            return input, target, target_rf

    def __len__(self):
        return len(self.ids)
