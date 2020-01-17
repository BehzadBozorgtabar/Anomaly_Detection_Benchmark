import click
import random
import torch
import numpy as np

from utils.config import Config
from optim.trainer import Solver
from networks.main import build_network
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'CXR_author', 'fmnist']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'CXR_resnet18','fmnist_LeNet', 'unet']))
@click.argument('xp_path')
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=1, help='Set seed. If -1, use randomization.')
@click.option('--batch_size', type=int, default=32, help='Batch size for mini-batch training.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--isize', type=int, default=256,
              help='Specify the image input size.')
@click.option('--rep_dim', type=int, default=200,
              help='Specify the latent vector size.')
@click.option('--k', type=int, default=100,
              help='Specify the number of closest neighbours to consider for metric calculation.')
@click.option('--npc_temperature', default=0.1, type=float,
                        help='temperature parameter for softmax')
@click.option('--npc_momentum', default=0.5, type=float,
                        help='momentum for non-parametric updates')
@click.option('--ans_select_rate', default=0.1, type=float,
                        help='ANs select rate at each round')
@click.option('--ans_size', default=10, type=int,
                        help='ANs size discarding the anchor')


def main(dataset_name, net_name, xp_path, data_path, load_model, device, seed,
         batch_size, n_jobs_dataloader, normal_class, isize, rep_dim, k, npc_temperature,
                npc_momentum, ans_select_rate, ans_size):

    cfg = Config(locals().copy())
    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'

    dataset = load_dataset(dataset_name, data_path, normal_class, isize)
    network = build_network(net_name, rep_dim)
    lr, n_epochs, weight_decay, w_rec, w_contrast = 0,0,0,0,0
    trainer = Solver(dataset, network, lr, n_epochs, batch_size, rep_dim, k, weight_decay, device, n_jobs_dataloader,
                     w_rec, w_contrast, npc_temperature, npc_momentum, ans_select_rate, ans_size, cfg)
    trainer.load_model(load_model)
    auc_score = trainer.test()
    print("AUC score = %.5f" % (auc_score))

if __name__ == '__main__':
    main()
