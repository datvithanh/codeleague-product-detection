import os

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from utils import load_image

class TrainingDataset(Dataset):
    def __init__(self, file_path):
        self.root = file_path
        self.table = pd.read_csv(os.path.join(file_path))

        X = self.table['image'].tolist()
        Y = self.table['label'].tolist()

        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = load_image(self.X[index], angle = 0)
        y = int(self.Y[index])
        return x, y

    def __len__(self):
        return len(self.Y)


def LoadDataset(split, data_path, n_jobs=8, train_set='train', dev_set='val', batch_size=8, cuda = True):
    if split == 'train':
        shuffle = True
        dataset_file = train_set + '.csv'
    else:
        shuffle = False
        dataset_file = dev_set + '.csv'

    ds = TrainingDataset(os.path.join(data_path, dataset_file))

    return  DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=cuda)
