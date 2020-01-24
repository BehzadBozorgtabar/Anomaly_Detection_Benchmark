import click
import random
import torch
import numpy as np

from deepSVDD import DeepSVDD
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'CXR_author', 'fmnist']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'resnet18','fmnist_LeNet']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--isize', type=int, default=28,
              help='Specify the image input size.')
@click.option('--rep_dim', type=int, default=100,
              help='Specify the latent vector size.')
              
              
def main(dataset_name, net_name, xp_path, data_path, load_model, objective, nu, device, seed,
         batch_size, n_jobs_dataloader, normal_class, isize, rep_dim):

    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, isize)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(objective, nu)
    deep_SVDD.set_network(net_name, rep_dim)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    deep_SVDD.load_model(model_path=load_model)
    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader, batch_size=batch_size)
    print(deep_SVDD.results['test_auc'])

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results_test.json')


if __name__ == '__main__':
    main()
