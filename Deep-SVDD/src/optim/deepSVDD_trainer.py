from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from utils.visualization.plot_images_grid import plot_images_grid

import logging
import time
import torch
import torch.optim as optim
import numpy as np

from tqdm import tqdm


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, pre_training_epochs: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader, pre_training_epochs)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, deepSVDD, cfg, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        best_score = 0
        
        if self.c is None:
            self.c = self.init_center_c(train_loader, net)
            
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            for data in tqdm(train_loader):
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                features, rec_images = net(inputs)
                dist = torch.sum((features - self.c) ** 2, dim=1)
                rec_loss = torch.mean(torch.sum(torch.abs(rec_images - inputs), dim=tuple(range(1, rec_images.dim()))))
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = cfg.settings['w_svdd']*(self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))) + cfg.settings['w_rec']*rec_loss
                else:
                    loss = cfg.settings['w_svdd']*torch.mean(dist) + cfg.settings['w_rec']*rec_loss
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
                        
            # Test model
            deepSVDD.test(dataset, device=self.device, n_jobs_dataloader=self.n_jobs_dataloader)
            if self.test_auc > best_score:
                # Save results, model, and configuration
                best_score = self.test_auc
                deepSVDD.R = float(self.R.cpu().data.numpy())  # get float
                deepSVDD.c = self.c.cpu().data.numpy().tolist()  # get list
                deepSVDD.save_results(export_json=cfg.settings['xp_path'] + '/results.json')
                deepSVDD.save_model(export_model=cfg.settings['xp_path'] + '/model.tar')
                cfg.save_config(export_json=cfg.settings['xp_path'] + '/config.json')
                
                if cfg.settings['dataset_name'] in ('mnist', 'cifar10'):
                
                    # Plot most anomalous and most normal (within-class) test samples
                    indices, labels, scores = zip(*deepSVDD.results['test_scores'])
                    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
                    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

                    if cfg.settings['dataset_name'] == 'mnist':
                        X_normals = dataset.test_set.data[idx_sorted[:32], ...].unsqueeze(1)
                        X_outliers = dataset.test_set.data[idx_sorted[-32:], ...].unsqueeze(1)

                    if cfg.settings['dataset_name'] == 'cifar10':
                        X_normals = torch.tensor(np.transpose(dataset.test_set.data[idx_sorted[:32], ...], (0, 3, 1, 2)))
                        X_outliers = torch.tensor(np.transpose(dataset.test_set.data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

                    plot_images_grid(X_normals, export_img=cfg.settings['xp_path'] + '/normals', title='Most normal examples', padding=2)
                    plot_images_grid(X_outliers, export_img=cfg.settings['xp_path'] + '/outliers', title='Most anomalous examples', padding=2)

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net
        
    def pretrain(self, deepSVDD, cfg, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        net.train()
        best_score = 0
        for epoch in range(self.pre_training_epochs):

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            
            for data in tqdm(train_loader):
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                _, rec_images = net(inputs)
                loss = torch.mean(torch.sum(torch.abs(rec_images - inputs), dim=tuple(range(1, rec_images.dim()))))
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.pre_training_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in tqdm(test_loader):
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                features, _ = net(inputs)
                dist = torch.sum((features - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in tqdm(train_loader):
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                features, _ = net(inputs)
                n_samples += features.shape[0]
                c += torch.sum(features, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
