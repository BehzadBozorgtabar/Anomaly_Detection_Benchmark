import os
import argparse
from torch.backends import cudnn
from solver import Solver
from data_loader import return_loader
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--c_dim', type=int, default=2)
    parser.add_argument('--img_crop_size', type=int, default=128)
    parser.add_argument('--g_first_dim', type=int, default=64)
    parser.add_argument('--d_first_dim', type=int, default=64)
    parser.add_argument('--enc_repeat_num', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--enc_lr', type=float, default=0.0001)
    parser.add_argument('--dec_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_id', type=float, default=10)
    parser.add_argument('--lambda_bi', type=float, default=10)
    parser.add_argument('--lambda_ssim', type=float, default=20)
    parser.add_argument('--lambda_f', type=float, default=1)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_epochs', type=int, default=650)
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=60000)
    parser.add_argument('--trained_model', type=str, default='')

    # Test settings
    parser.add_argument('--test_model', type=str, default='650_200')

    # Set mode (train or test)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    # Path to save models and logs
    parser.add_argument('--log_path', type=str, default='/media/bozorgta/Elements/Anomaly/master/log')
    parser.add_argument('--model_save_path', type=str, default='/media/bozorgta/Elements/Anomaly/master/Chex_Models')
    parser.add_argument('--sample_path', type=str, default='/media/bozorgta/Elements/Anomaly/master/samples')
    parser.add_argument('--result_path', type=str, default='/media/bozorgta/Elements/Anomaly/master/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=250)
    parser.add_argument('--model_save_step', type=int, default=200)

    config = parser.parse_args()
    print(config)
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    img_data_loader = return_loader(config.img_crop_size, config.batch_size, config.mode)
    # Solver
    solver = Solver(img_data_loader,config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()

