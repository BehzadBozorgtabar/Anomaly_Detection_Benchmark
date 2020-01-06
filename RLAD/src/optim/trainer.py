from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

from lib import protocols
from lib.non_parametric_classifier import NonParametricClassifier
from lib.criterion import Criterion
from lib.ans_discovery import ANsDiscovery
from lib.utils import AverageMeter, time_progress, adjust_learning_rate

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import math

from tqdm import tqdm


class Solver(BaseTrainer):

    def __init__(self, dataset : BaseADDataset, network : BaseNet, k : int, lr: float, n_epochs: int, batch_size: int, rep_dim:int, K : int,
                weight_decay: float, device: str, n_jobs_dataloader: int, w_rec : float, npc_temperature,
                npc_momentum, ans_select_rate, ans_size, cfg):

        super().__init__(lr, n_epochs, batch_size, rep_dim, K, weight_decay, device, n_jobs_dataloader, w_rec)
        self.ae_net = network.to(self.device)
        self.train_loader, self.test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        ntrain = len(self.train_loader.dataset)
        self.npc = NonParametricClassifier(rep_dim, ntrain, npc_temperature, npc_momentum).to(device)
        self.ANs_discovery = ANsDiscovery(ntrain, ans_select_rate, ans_size, device).to(device)
        self.criterion = Criterion().to(device)

        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.best_score = 0
        self.k = k
        self.cfg = cfg
        self.logger = logging.getLogger()
        #self.memory = torch.randn(size=(len(self.train_loader.dataset), 512))
        self.memory = torch.randn(size=(len(self.train_loader.dataset), self.rep_dim)).to(self.device)

    def train(self):

        for r in range(5):

            if r > 0:
                self.ANs_discovery.update(r, self.npc)

            for e in range(1, self.n_epochs + 1):
                loss = 0

                self.ae_net.train()
                # tracking variables
                train_loss = AverageMeter()

                # switch the model to train mode
                self.ae_net.train()
                # adjust learning rate
                #adjust_learning_rate(self.optimizer, lr)

                for batch_idx, (inputs, _, indexes) in enumerate(tqdm(self.train_loader)):
                    inputs, indexes = inputs.to(self.device), indexes.to(self.device)
                    self.optimizer.zero_grad()

                    latent, rec_images = self.ae_net(inputs)
                    outputs = self.npc(latent, indexes)
                    loss = self.criterion(outputs, indexes, self.ANs_discovery) / len(inputs)

                    loss.backward()
                    train_loss.update(loss.item() * len(inputs), inputs.size(0))

                    self.optimizer.step()

                print('Round: {round}/5 Epoch: {epoch}/{tot_epochs}'
                        'LR: {learning_rate:.5f} '
                        'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                    round=r, epoch=e, tot_epochs=self.n_epochs,
                    elps_iters=batch_idx, learning_rate=self.lr, train_loss=train_loss))

                score = self.test()
                if score > self.best_score:
                    self.best_score = score
                    torch.save({'state_dict' : self.ae_net.state_dict()}, self.cfg.settings['xp_path'] + '/model.tar')
                self.logger.info("Epoch %d/%d : Loss = %f | AUC = %.4f | BEST AUC = %.4f" % (e, self.n_epochs, loss/len(self.train_loader.dataset), score, self.best_score))

        #ANs_num = self.ANs_discovery.anchor_indexes.shape[0]
        """
        for e in range(1, self.n_epochs+1):
            loss = 0

            self.ae_net.train()
            i = 0
            for inputs, _, _ in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                self.optimizer.zero_grad()

                latent1, rec_images, latent2 = self.ae_net(inputs)
                rec_loss = self.w_rec*self.rec_loss(inputs, rec_images) + self.w_feat*self.feat_loss(latent2, latent1)
                rec_loss.backward()

                loss += rec_loss.item()*inputs.shape[0]
                self.optimizer.step()

                self.memory[i*self.batch_size : min((i+1)*self.batch_size, len(self.train_loader.dataset))] = latent1
                i+=1
        """

    def test(self):

        idx_label_score = []
        self.ae_net.eval()
        trainFeatures = self.npc.memory.t()
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                latent1, rec_images = self.ae_net(inputs)

                dist = torch.acos(torch.mm(latent1, trainFeatures)) / math.pi

                scores, _ = dist.topk(self.K, dim=1, largest=False, sorted=True)
                scores = torch.mean(scores, dim=1)

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        return roc_auc_score(labels, scores)