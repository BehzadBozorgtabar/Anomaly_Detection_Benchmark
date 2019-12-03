import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import os
import time
import datetime
import utils
from model import Discriminator
from model import Generator
from model import evaluate
from PIL import Image
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import pytorch_ssim




class Solver(object):

    def __init__(self, img_data_loader, config):
        # Data loader
        self.img_data_loader = img_data_loader

        # Model parameters
        self.c_dim = config.c_dim
        self.num_layers=config.num_layers
        self.g_first_dim = config.g_first_dim
        self.d_first_dim = config.d_first_dim
        self.enc_repeat_num = config.enc_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat
        self.img_crop_size= config.img_crop_size

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_id = config.lambda_id
        self.lambda_bi = config.lambda_bi
        self.lambda_ssim = config.lambda_ssim
        self.lambda_f = config.lambda_f
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.enc_lr
        self.enc_lr = config.enc_lr
        self.dec_lr = config.dec_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.is_training= config.mode
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.trained_model = config.trained_model

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Set tensorboard
        self.build_model()
        self.use_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load_trained_model()

    def build_model(self):
        # Define encoder-decoder (generator) and a discriminator
        self.G= Generator(self.g_first_dim, self.enc_repeat_num)
        self.D = Discriminator(self.img_crop_size, self.d_first_dim, self.d_repeat_num)
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def load_trained_model(self):

        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.trained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.trained_model))))
        print('loaded models (step: {})..!'.format(self.trained_model))

    def use_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)


    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def to_var2(self, x, volatile=False):
        return Variable(x, volatile=volatile)

    def calculate_accuracy(self, x, y):
        _, predicted = torch.max(x, dim=1)
        correct = (predicted == y).float()
        accuracy = torch.mean(correct) * 100.0
        return accuracy

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def train(self):
        """Train anomaly detection model"""
        self.data_loader = self.img_data_loader
        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader.train)

        fixed_x = []
        for i, (images, labels) in enumerate(self.data_loader.train):
            fixed_x.append(images)
            if i == 0:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)


        # Learning rate for decaying
        d_lr = self.d_lr
        g_lr= self.g_lr

        # Start with trained model
        if self.trained_model:
            start = int(self.trained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (real_x, real_label) in enumerate(self.data_loader.train):
                rand_idx = torch.randperm(real_label.size(0))
                # Convert tensor to variable
                real_x = self.to_var(real_x)


                #================== Train Discriminator ================== #
                # Compute loss with real images
                out_src = self.D(real_x)
                d_loss_real = - torch.mean(out_src)

                fake_x, _, _= self.G(real_x)
                fake_x = Variable(fake_x.data)

                out_src = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Discriminator losses
                d_loss = d_loss_real + d_loss_fake
                self.reset()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)


                # Gradient penalty loss
                d_loss = self.lambda_gp * d_loss_gp
                self.reset()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # ================== Train Encoder-Decoder networks ================== #
                if (i+1) % self.d_train_repeat == 0:


                    fake_x, enc_feat, rec_feat= self.G(real_x)
                    out_src =self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)


                    g_loss_rec_x = torch.mean(torch.abs(real_x - fake_x))

                    g_loss_ssim = (0.5 * (1 - self.ssim_loss(real_x, fake_x))).clamp(0, 1)

                    g_loss_feature = torch.mean(torch.pow((enc_feat-rec_feat), 2))


                    g_loss = g_loss_fake +self.lambda_f * g_loss_feature+ +self.lambda_bi * g_loss_rec_x + self.lambda_ssim * g_loss_ssim
                    self.reset()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging Generator losses
                    loss['G/loss_feature'] = g_loss_feature.item()
                    loss['G/loss_image'] = g_loss_rec_x.item()
                    loss['G/loss_ssim'] = g_loss_ssim.item()
                    loss['G/loss_fake'] = g_loss_fake.item()


                # Print out log
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)


                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)


                # Reconstructed images
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    #for fixed_c in fixed_c_list:
                    sample_result, _, _ = self.G(fixed_x)
                    fake_image_list.append(sample_result)
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Generated images and saved into {}..!'.format(self.sample_path))


                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e + 1, i + 1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))


            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))



    def test(self):
        """Computing AUC for the unseen test set """
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.device = torch.device("cuda:0")
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()


        data_loader = self.img_data_loader
        # Create big error tensor for the test set.
        self.an_scores = torch.zeros(size=(len(data_loader.valid.dataset),), dtype=torch.float32,
                                     device=self.device)
        self.gt_labels = torch.zeros(size=(len(data_loader.valid.dataset),), dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i, (real_x, org_c) in enumerate(data_loader.valid):
                real_x = self.to_var(real_x, volatile=True)
                fake_x, enc_feat, rec_feat = self.G(real_x)
                error=torch.mean(torch.pow((enc_feat - rec_feat), 2),dim=(1,2,3))+torch.mean(torch.abs(real_x - fake_x),dim=(1,2,3))
                self.an_scores[i * self.batch_size: i * self.batch_size + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i * self.batch_size: i * self.batch_size + error.size(0)] = org_c.reshape(error.size(0))
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
            torch.max(self.an_scores) - torch.min(self.an_scores))
            self.an_scores = self.an_scores.detach()
            self.an_scores = self.an_scores.cpu().numpy()
            self.gt_labels = self.gt_labels.detach()
            self.gt_labels = self.gt_labels.cpu().numpy()
            auc = evaluate(self.gt_labels, self.an_scores, metric='roc')

            #target_c_list = []
            #for j in range(self.c_dim):
            #    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
            #    target_c_list.append(self.to_var(target_c, volatile=True))

            # Target image generation
            #fake_image_list = [real_x]
            #for target_c in target_c_list:
            #    enc_feat = self.Enc(real_x)
            #    rec_feat = self.Enc(fake_x)
            #    sample_result = self.Dec(enc_feat)
            #    fake_image_list.append(sample_result)
            #fake_images = torch.cat(fake_image_list, dim=3)
            #save_path = os.path.join(self.result_path, '{}_fake.png'.format(i + 1))
            #save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            #print('Translated test images and saved into "{}"..!'.format(save_path))


