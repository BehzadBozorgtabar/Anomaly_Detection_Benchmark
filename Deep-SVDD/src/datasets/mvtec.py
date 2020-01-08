from PIL import Image
from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os


class MVTEC_Dataset(TorchvisionDataset):

    def __init__(self, root: str, isize=224, object=0):
        super().__init__(root)

        objects = {0 : 'bottle', 1 : 'cable', 2 : 'capsule', 3 : 'carpet', 4 : 'grid', 5 : 'hazelnut', 6 : 'leather',
                   7 : 'metal_nut', 8 : 'pill', 9 : 'screw', 10 : 'tile', 11 : 'toothbrush', 12 : 'transistor', 13 : 'wood', 14 : 'zipper'}
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([0])
        self.outlier_classes = tuple([1])

        transform_train = transforms.Compose([transforms.Resize(isize),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        transform_test = transforms.Compose([transforms.Resize(isize),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        self.train_set = MyMVTEC(2, os.path.join(self.root, objects[object], 'train'), transform_train)

        self.test_set = MyMVTEC(1, os.path.join(self.root, objects[object], 'test'), transform_test)


class MyMVTEC(ImageFolder):

    def __init__(self, augmentation, *args, **kwargs):
        super(MyMVTEC, self).__init__(*args, **kwargs)
        self.augmentation = augmentation

    def __len__(self):
        return super(MyMVTEC, self).__len__() * self.augmentation

    def __getitem__(self, idx):
        upd_idx = idx // self.augmentation
        img, annot = super(MyMVTEC, self).__getitem__(upd_idx)
        return img, annot, idx
