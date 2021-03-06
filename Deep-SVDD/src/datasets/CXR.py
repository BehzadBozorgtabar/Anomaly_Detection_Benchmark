from PIL import Image
from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

class CXR_Dataset(TorchvisionDataset):

    def __init__(self, root: str, isize=28):
        super().__init__(root)

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

        self.train_set = MyCXR(2, os.path.join(self.root, 'train'), transform_train)

        self.test_set = MyCXR(1, os.path.join(self.root, 'test'), transform_test)
        
        
class MyCXR(ImageFolder):

    def __init__(self, augmentation,*args, **kwargs):
        super(MyCXR, self).__init__(*args, **kwargs)
        self.augmentation = augmentation
        
    def __len__(self):
        return super(MyCXR, self).__len__()*self.augmentation
        
    def __getitem__(self, idx):
        upd_idx = idx//self.augmentation
        img, annot = super(MyCXR, self).__getitem__(upd_idx)
        return img, annot, idx
