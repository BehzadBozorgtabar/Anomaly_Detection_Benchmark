import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
import matplotlib.pyplot as plt
import IPython
from networks.main import build_network
from tqdm import tqdm

class Solver(object):
    DEFAULTS = {}   
    def __init__(self, data_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.train_loader, self.test_loader = data_loader.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dagmm = build_network(self.network, self.rep_dim, self.k_gmm).to(self.device)
        self.best_auc_score = 0
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

    def reset_grad(self):
        self.dagmm.zero_grad()

    def train(self):
        iters_per_epoch = len(self.train_loader)

        # Start training
        iter_ctr = 0
        start_time = time.time()

        for e in range(self.num_epochs):
            print("Epoch {}".format(e+1))
            loss = {}
            loss['total_loss'] = 0
            loss['sample_energy'] = 0
            loss['recon_error'] = 0
            loss['cov_diag'] = 0
            
            for input_data, _, _ in tqdm(self.train_loader):
                iter_ctr += 1
                start = time.time()

                input_data = input_data.to(self.device)
                total_loss,sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
                # Logging
                loss['total_loss'] += total_loss.data.item() * input_data.shape[0]
                loss['sample_energy'] += sample_energy.item() * input_data.shape[0]
                loss['recon_error'] += recon_error.item() * input_data.shape[0]
                loss['cov_diag'] += cov_diag.item() * input_data.shape[0]

            for key in loss:
                loss[key] = loss[key] / len(self.train_loader.dataset)
            print(loss)

            auc_score = self.test()
            
            if auc_score >= self.best_auc_score:
                self.best_auc_score = auc_score
                torch.save(self.dagmm.state_dict(), os.path.join(self.log_path, '{}_dagmm.pth'.format(e+1)))
            print("AUC : {:0.4f}, MAX_AUC : {:0.4f}".format(auc_score, self.best_auc_score))

    def dagmm_step(self, input_data):
        self.dagmm.train()
        z, dec, gamma = self.dagmm(input_data)

        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag, self.device)

        self.reset_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()

        return total_loss,sample_energy, recon_error, cov_diag

    def test(self):
        print("======================TEST MODE======================")
        self.dagmm.eval()

        with torch.no_grad():
            test_energy = []
            test_labels = []
            test_z = []
            for input_data, labels, _ in tqdm(self.test_loader):
                input_data = input_data.to(self.device)
                z, dec, gamma = self.dagmm(input_data)
                sample_energy, cov_diag = self.dagmm.compute_energy(z, self.device, size_average=False)
                test_energy.append(sample_energy.data.cpu().numpy())
                test_z.append(z.data.cpu().numpy())
                test_labels.append(labels.numpy())


            test_energy = np.concatenate(test_energy,axis=0)
            test_z = np.concatenate(test_z,axis=0)
            test_labels = np.concatenate(test_labels,axis=0)

            gt = test_labels.astype(int)

            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(gt,test_energy)
        
            return auc_score    
