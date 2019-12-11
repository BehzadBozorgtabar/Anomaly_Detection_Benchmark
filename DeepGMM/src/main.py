import os
import argparse
from deepGMM_trainer import Solver
from datasets.main import load_dataset
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.mkdir(config.log_path)

    data_loader = load_dataset(config.dataset, config.data_path, config.normal_class, config.isize)
    
    # Solver
    solver = Solver(data_loader, vars(config))
    solver.train()

    return solver
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--k_gmm', type=int, default=4)
    parser.add_argument('--rep_dim', type=int, default=32)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--normal_class', type=int, default=0)
    parser.add_argument('--isize', type=int, default=28)
    parser.add_argument('--n_jobs_dataloader', type=int, default=0)

    # Misc
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--network', type=str, default='mnist_LeNet')
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--log_path', type=str, default='../log/mnist')

    config = parser.parse_args()
 
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
