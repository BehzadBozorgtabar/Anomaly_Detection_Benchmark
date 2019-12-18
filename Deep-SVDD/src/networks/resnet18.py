import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from base.base_net import BaseNet

class ResNet18(nn.Module):

    def __init__(self, rep_dim):
        super(ResNet18, self).__init__()
        
        model = models.resnet18(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(in_features=512, out_features=rep_dim, bias=True)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)
        classifier = self.avgpool(feature)
        classifier = classifier.view(-1, 512)
        classifier = self.fc(classifier)
        return feature, classifier
        
        
class UPBlock(nn.Module):

    def __init__(self, ich, och, upsample=False):
        super(UPBlock, self).__init__()
        
        self.upsample = upsample
        self.upsampleLayer = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                        nn.Conv2d(ich, och, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) if upsample else  None
                                        
        self.conv1 = nn.Sequential(nn.Conv2d(ich, och, 3, 1, 1, bias=False),
                nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.relu = nn.ReLU(True)           
        self.conv2 = nn.Conv2d(och, och, 3, 1, 1, bias=False)
        self.bnrm = nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                    
    def forward(self, x):
        if self.upsample:
            identity = self.upsampleLayer(x)
            out = identity
        else:
            identity = x
            out = self.conv1(x)
            
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bnrm(out)
        
        out += identity
        out = self.relu(out)
        return out
        
        
class ResNetDecoderSymmetric(nn.Module):

    def __init__(self):
        super(ResNetDecoderSymmetric, self).__init__()
        
        self.decoder = nn.Sequential(UPBlock(512, 512),
                                    UPBlock(512, 512),
                                    UPBlock(512, 256, True),
                                    UPBlock(256, 256),
                                    UPBlock(256, 128, True),
                                    UPBlock(128, 128),
                                    UPBlock(128, 64, True),
                                    UPBlock(64, 64))
                                    
        self.output = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                        nn.Conv2d(64, 3, kernel_size=1, bias=False),
                        nn.Tanh())                               
                                    
    def forward(self, x):
        out = self.decoder(x)
        out = self.output(out)
        return out

class ResNetDecoder(nn.Module):

    def __init__(self):
        super(ResNetDecoder, self).__init__()
        
        
        self.deconv = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2, 0),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(256, 128, 2, 2, 0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(128, 64, 2, 2, 0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64, 32, 2, 2, 0),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 1, 3, 1, 1),
                                    nn.Tanh())
                                    
    def forward(self, x):
        return self.deconv(x)
                                    
class ResNet18Autoencoder(BaseNet):

    def __init__(self, rep_dim):
        super().__init__()
        self.encoder = ResNet18(rep_dim)
        self.decoder = ResNetDecoderSymmetric()
        self.rep_dim = rep_dim
        
    def forward(self, x):
        feature, classifier = self.encoder(x)
        rec_image = self.decoder(feature)
        return classifier, rec_image
        
          
#model = ResNet18()
#deconv = ResNetDecoderSymmetric()
#from torchsummary import summary
#print(summary(model, (1, 256, 256)))
#print(summary(deconv, (512, 16, 16)))
