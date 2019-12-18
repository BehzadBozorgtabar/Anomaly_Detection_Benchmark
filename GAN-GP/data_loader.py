import torch
import os
import random
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from dataset import DatasetFromFolder



def load_data_list(data_dir):
     path = os.path.join(data_dir, '', '*')
     file_list = glob(path)
     return file_list

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


def return_loader(crop_size, batch_size, mode='train'):
    """Return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    shuffle = False

    if mode == 'train':
        shuffle = True


    dataroot='/data/author'
    train_ds = ImageFolder(os.path.join(dataroot, 'train'), transform)
    valid_ds = ImageFolder(os.path.join(dataroot, 'test'), transform)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return Data(train_dl, valid_dl)
