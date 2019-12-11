from .mnist_LeNet import MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU_Autoencoder
from .resnet18 import ResNet18Autoencoder
from .fmnist_LeNet import FMNIST_LeNet_Autoencoder

def build_network(net_name, rep_dim, k_gmm):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'CXR_resnet18', 'fmnist_LeNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder(rep_dim, k_gmm)

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder(rep_dim, k_gmm)

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder(rep_dim, k_gmm)
        
    if net_name == 'CXR_resnet18':
        ae_net = ResNet18Autoencoder(rep_dim, k_gmm)
        
    if net_name == 'fmnist_LeNet':
        ae_net = FMNIST_LeNet_Autoencoder(rep_dim, k_gmm)

    return ae_net