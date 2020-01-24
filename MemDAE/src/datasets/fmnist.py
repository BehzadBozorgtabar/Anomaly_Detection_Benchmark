from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
 
import torchvision.transforms as transforms


class FMNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-2.6812391, 24.85431),
                   (-2.5778584, 10.849295),
                   (-2.8081715, 19.133549),
                   (-1.9533653, 18.65673),
                   (-2.6103857, 19.166685),
                   (-1.2358522, 28.463093),
                   (-3.1003108, 24.196823),
                   (-1.081444, 16.878813),
                   (-3.6560965, 11.350284),
                   (-1.3859291, 11.42665)]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],[min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyFMNIST(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyFMNIST(root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)


class MyFMNIST(FashionMNIST):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyFMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
