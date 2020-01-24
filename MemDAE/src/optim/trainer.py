from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import math

class Solver(BaseTrainer):

    def __init__(self, dataset : BaseADDataset, network : BaseNet, k : int, lr: float, n_epochs: int, batch_size: int, rep_dim:int, K : int,
                 weight_decay: float, device: str, n_jobs_dataloader: int, w_rec : float, w_feat : float, cfg):

        super().__init__(lr, n_epochs, batch_size, rep_dim, K, weight_decay, device, n_jobs_dataloader, w_rec, w_feat)
        self.ae_net = network.to(self.device)
        self.train_loader, self.test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.rec_loss = torch.nn.L1Loss()
        self.feat_loss = torch.nn.MSELoss()
        self.best_score = 0
        self.min_loss = 1000
        self.k = k
        self.cfg = cfg
        self.logger = logging.getLogger()
        self.memory = torch.randn(size=(len(self.train_loader.dataset), self.rep_dim)).to(self.device)

    def train(self):

        for e in range(1, self.n_epochs+1):
            loss = 0

            self.ae_net.train()
            for inputs, _, indexes in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()

                latent1, rec_images = self.ae_net(inputs)
                rec_loss = self.w_rec*self.rec_loss(inputs, rec_images)
                rec_loss.backward()

                loss += rec_loss.item()*inputs.shape[0]
                self.optimizer.step()

                self.memory[indexes] = latent1

            score, test_loss = self.test()
            if test_loss < self.min_loss:
                self.min_loss = test_loss
                torch.save({'state_dict' : self.ae_net.state_dict()}, self.cfg.settings['xp_path'] + '/model.tar')
            self.logger.info("Epoch %d/%d :Train  Loss = %f | Test Loss = %f | AUC = %.4f " % (e, self.n_epochs, loss/len(self.train_loader.dataset), test_loss, score))

    def test(self, load_memory=False):

        idx_label_score = []
        self.ae_net.eval()
        with torch.no_grad():
            if load_memory:
                for inputs, _, indexes in tqdm(self.train_loader):
                    inputs = inputs.to(self.device)

                    latent1, rec_images = self.ae_net(inputs)
                    self.memory[indexes] = latent1

            loss = 0
            trainLabels = torch.LongTensor(self.train_loader.dataset.targets).to(self.device)
            trainfeatures = self.memory[trainLabels==0].t()
            for data in tqdm(self.test_loader):
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                latent1, rec_images = self.ae_net(inputs)

                dist = torch.acos(torch.mm(latent1, trainfeatures)) / math.pi

                scores, _ = dist.topk(self.K, dim=1, largest=False, sorted=True)
                scores = torch.mean(scores, dim=1)
                loss += self.rec_loss(inputs, rec_images).item()*inputs.shape[0]

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_scores = idx_label_score
        test_loss = loss / len(self.test_loader.dataset)
        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        labels = np.array(labels)
        scores = np.array(scores)


        return roc_auc_score(labels, scores), test_loss

    def load_model(self, model_path):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)
        self.ae_net.load_state_dict(model_dict['state_dict'])
