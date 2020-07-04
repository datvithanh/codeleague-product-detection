import os

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from utils import load_image, RandomErasing, Cutout, Normaliztion, RandomHorizontalFlip


class TrainingDataset(Dataset):
    def __init__(self, file_path, transform):
        self.root = file_path
        self.table = pd.read_csv(os.path.join(file_path))

        X = self.table['image'].tolist()
        Y = self.table['label'].tolist()

        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x = load_image(self.X[index].replace('/home/datvt/hust/shopee-codeleague/train', '/home2/htthanh/DatVT/code-league/train'))
        y = int(self.Y[index])

        x = self.transform(x)
        return x.copy(), y

    def __len__(self):
        return len(self.Y)


def LoadDataset(split, data_path, n_jobs=2, train_set='train', dev_set='val', batch_size=8, cuda = True):
    if split == 'train':
        shuffle = True
        dataset_file = train_set + '.csv'
        transform = transforms.Compose([RandomErasing(), RandomHorizontalFlip(), Cutout(), Normaliztion()])
        # transform = transforms.Compose([RandomErasing(), RandomHorizontalFlip(), Cutout(), transforms.Resize(400), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        shuffle = False
        dataset_file = dev_set + '.csv'
        transform = transforms.Compose([Normaliztion()])
        # transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    ds = TrainingDataset(os.path.join(data_path, dataset_file), transform)

    return  DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=cuda)

