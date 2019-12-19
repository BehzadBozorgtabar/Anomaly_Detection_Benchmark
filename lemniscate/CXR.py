'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse
import time

import models
from datasets.CXR import return_loader

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import kNN

parser = argparse.ArgumentParser(description='PyTorch CXR Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--workers', help='Number of workers to load the data', default=0, type=int)
parser.add_argument('--K', help='Number of points to consider to calculate the metric', default=1,type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_auc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
loaders = return_loader(224, 16)

ndata = len(loaders.train.dataset)

print('==> Building model..')
net = models.__dict__['resnet18'](low_dim=args.low_dim)
# define leminiscate
if args.nce_k > 0:
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
else:
    lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

# Model
if args.test_only or len(args.resume) > 0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.resume)
    net.load_state_dict(checkpoint['net'])
    lemniscate = checkpoint['lemniscate']
    best_auc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# define loss function
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    criterion = nn.CrossEntropyLoss()

net.to(device)
lemniscate.to(device)
criterion.to(device)

if args.test_only:
    acc = kNN(0, net, lemniscate, loaders.train, loaders.valid, 200, args.nce_t, 1)
    sys.exit(0)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    #Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch - 80) // 40))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(loaders.train):
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        optimizer.zero_grad()

        features = net(inputs)
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs, indexes)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
            epoch, batch_idx, len(loaders.train), batch_time=batch_time, data_time=data_time, train_loss=train_loss))

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    auc = kNN(epoch, net, lemniscate, loaders.train, loaders.valid, args.K, args.nce_t, 0, device)

    if auc > best_auc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'lemniscate': lemniscate,
            'auc': auc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_auc = auc

    print('auc_score {:.2f} | best auc score: {:.2f}'.format(auc*100, best_auc * 100))

#acc = kNN(0, net, lemniscate, loaders.train, loaders.valid, 200, args.nce_t, 1)
#print('last accuracy: {:.2f}'.format(acc * 100))