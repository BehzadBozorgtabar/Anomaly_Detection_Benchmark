from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .CXR import CXR_Dataset
from .fmnist import FMNIST_Dataset
from .mvtec import MVTEC_Dataset

def load_dataset(dataset_name, data_path, normal_class, isize):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'CXR_author', 'fmnist', 'mvtec')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)
        
    if dataset_name == 'CXR_author':
        dataset = CXR_Dataset(root=data_path, isize=isize)
        
    if dataset_name == 'fmnist':
        dataset = FMNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'mvtec':
        dataset = MVTEC_Dataset(root=data_path, object=normal_class, isize=isize)

    return dataset
