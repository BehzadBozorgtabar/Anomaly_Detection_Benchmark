
import click
import torch
import logging
import random
import numpy as np
import os

from utils.config import Config
from optim.trainer import Solver
from networks.main import build_network
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', default='CXR_author')
@click.argument('net_name', default='CXR_resnet18')
@click.argument('xp_path', default='/media/bozorgta/Elements/SVDD/Anomaly_Detection_Benchmark/EDFAD/log')
@click.argument('data_path', default='/media/bozorgta/Elements/Anomaly/ganomaly/data/NIH_Chest')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--isize', type=int, default=28,
              help='Specify the image input size.')
@click.option('--rep_dim', type=int, default=100,
              help='Specify the latent vector size.')
@click.option('--k', type=int, default=1,
              help='Specify the number of closest neighbours to consider for metric calculation.')
@click.option('--w_rec', type=float, default=50,
              help='Specify the weight of reconstruction loss.')
@click.option('--w_feat', type=float, default=1,
              help='Specify the weight of feature consistency loss')

def main(dataset_name, net_name, xp_path, data_path, load_model, device, seed,
         lr, n_epochs, batch_size, weight_decay, n_jobs_dataloader, normal_class, isize, rep_dim, k, w_rec, w_feat):

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    if not os.path.exists(xp_path):
        os.mkdir(xp_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Log training details
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])
    logger.info('Training rep_dim: %d' % cfg.settings['rep_dim'])
    logger.info('Training k: %d' % cfg.settings['k'])
    logger.info('Training reconstruction loss weight: %d' % cfg.settings['w_rec'])
    logger.info('Training feature consistency loss weight: %d' % cfg.settings['w_feat'])

    dataset = load_dataset(dataset_name, data_path, normal_class, isize)
    network = build_network(net_name, rep_dim)
    trainer = Solver(dataset, network, k, lr, n_epochs, batch_size, rep_dim, k, weight_decay, device, n_jobs_dataloader, w_rec, w_feat, cfg)

    trainer.train()

if __name__ == '__main__':
    main()
