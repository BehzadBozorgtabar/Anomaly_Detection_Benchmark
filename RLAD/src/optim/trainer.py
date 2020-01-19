from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from lib import protocols
from lib.non_parametric_classifier import NonParametricClassifier
from lib.criterion import Criterion
from lib.ans_discovery import ANsDiscovery
from lib.utils import AverageMeter, time_progress, adjust_learning_rate, traverse

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import math

from tqdm import tqdm


class Solver(BaseTrainer):

    def __init__(self, dataset : BaseADDataset, network, lr: float, n_epochs: int, batch_size: int, rep_dim:int, K : int,
                weight_decay: float, device: str, n_jobs_dataloader: int, w_rec : float, w_contrast, npc_temperature,
                npc_momentum, ans_select_rate, ans_size, cfg):

        super().__init__(lr, n_epochs, batch_size, rep_dim, K, weight_decay, device, n_jobs_dataloader, w_rec, w_contrast)
        self.ae_net = network.to(self.device)

        self.train_loader, self.test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        ntrain = len(self.train_loader.dataset)

        self.npc = NonParametricClassifier(rep_dim, ntrain, npc_temperature, npc_momentum).to(device)
        self.ANs_discovery = ANsDiscovery(ntrain, ans_select_rate, ans_size, device).to(device)
        self.criterion = Criterion().to(device)
        self.recfn = nn.L1Loss()

        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.best_score = 0
        self.cfg = cfg
        self.ans_select_rate = ans_select_rate
        self.logger = logging.getLogger()

    def train(self):

        for r in range(0, (int) (1 / self.ans_select_rate) + 1):

            if r > 0:
                self.ANs_discovery.update(r, self.npc)

            for e in range(1, self.n_epochs + 1):

                # tracking variables
                train_loss = 0

                # switch the model to train mode
                self.ae_net.train()

                for batch_idx, (inputs, _, indexes) in enumerate(tqdm(self.train_loader)):
                    inputs, indexes = inputs.to(self.device), indexes.to(self.device)
                    self.optimizer.zero_grad()

                    latent, rec_images = self.ae_net(inputs)
                    #latent, _ = self.ae_net(inputs)
                    outputs = self.npc(latent, indexes)
                    rec_loss = self.recfn(rec_images, inputs)
                    loss = self.w_contrast*(self.criterion(outputs, indexes, self.ANs_discovery)) + self.w_rec*rec_loss

                    loss.backward()
                    train_loss += loss.item() * len(inputs)

                    self.optimizer.step()

                print('Round: {round}/{nbr_round} Epoch: {epoch}/{tot_epochs}' 
                        'LR: {learning_rate:.5f} '
                        'Loss: {train_loss:.4f}'.format(
                    round=r, nbr_round = (int) (1 / self.ans_select_rate),epoch=e, tot_epochs=self.n_epochs,
                    elps_iters=batch_idx, learning_rate=self.lr, train_loss=train_loss/len(self.train_loader.dataset)))

                score = self.test()
                if score > self.best_score:
                    self.best_score = score
                    torch.save({'model_state_dict' : self.ae_net.state_dict(),
                                'memory_state_dict' : self.npc.state_dict(),
                                'ans_discovery_state_dict' : self.ANs_discovery.state_dict()}
                               , self.cfg.settings['xp_path'] + '/model.tar')
                self.logger.info("Epoch %d/%d : AUC = %.4f | BEST AUC = %.4f" % (e, self.n_epochs, score, self.best_score))
            torch.save({'model_state_dict' : self.ae_net.state_dict(),
                        'memory_state_dict' : self.npc.state_dict(),
                        'ans_discovery_state_dict' : self.ANs_discovery.state_dict()}
                       ,self.cfg.settings['xp_path'] + '/model_round' + str(r) + '.tar')

    def test(self):

        idx_label_score = []
        self.ae_net.eval()

        trainFeatures = self.npc.memory
        trainLabels = torch.LongTensor(self.train_loader.dataset.targets).to(self.device)
        trainFeatures = trainFeatures[trainLabels==0].t()

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                latent1, rec_images = self.ae_net(inputs)

                cossims = torch.acos(torch.mm(latent1, trainFeatures)) / math.pi

                cossims, _ = cossims.topk(self.K, dim=1, largest=False, sorted=True)
                cossims_scores = torch.mean(cossims, dim=1)
                scores = cossims_scores

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        labels = np.array(labels)
        scores = np.array(scores)

        return roc_auc_score(labels, scores)

    def load_model(self, path):
        dic = torch.load(path)
        self.ae_net.load_state_dict(dic['model_state_dict'])
        self.npc.load_state_dict(dic['memory_state_dict'])
        #self.ANs_discovery.load_state_dict(dic['ans_discovery_state_dict'])