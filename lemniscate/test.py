import torch
import time
import datasets
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, device='cuda'):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).to(device)
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).to(device)
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.to(device)
            batchSize = inputs.size(0)
            features = net(inputs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).to(device)
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    end = time.time()
    idx_label_score = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(tqdm(testloader)):
            end = time.time()
            targets = targets.to(device)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = (torch.acos(torch.mm(features, trainFeatures))+1)/2

            scores, _ = dist.topk(K, dim=1, largest=False, sorted=True)
            scores = torch.mean(scores, dim=1)

            idx_label_score += list(zip(indexes.cpu().data.numpy().tolist(),
                                        targets.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        test_auc = roc_auc_score(labels, scores)

    return test_auc

