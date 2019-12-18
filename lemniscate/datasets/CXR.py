import os
from glob import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


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


class MyCXR(ImageFolder):

    def __init__(self, augmentation, *args, **kwargs):
        super(MyCXR, self).__init__(*args, **kwargs)
        self.augmentation = augmentation

    def __len__(self):
        return super(MyCXR, self).__len__() * self.augmentation

    def __getitem__(self, idx):
        upd_idx = idx // self.augmentation
        img, annot = super(MyCXR, self).__getitem__(upd_idx)
        return img, annot, idx


def return_loader(crop_size, batch_size):
    """Return data loader."""

    transform_train = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_val = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataroot='data/author'
    train_ds = MyCXR(2, os.path.join(dataroot, 'train'), transform_train)
    valid_ds = MyCXR(1, os.path.join(dataroot, 'test'), transform_val)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)