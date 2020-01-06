import torch
import torch.nn as nn


class Double_conv(nn.Module):
    """
    Apply a double convolution on the input x
    args:
        - ich: Number of channel of the input
        - och: Number of channel of the output
    """

    def __init__(self, ich, och):
        super(Double_conv, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(ich, och, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(och),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(och, och, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(och),
                                nn.LeakyReLU(0.1))
                                
    def forward(self, x):
        x = self.conv(x)
        return x
        
        
class Down(nn.Module):

    """
    Apply an encoding convolution, divides the image size by two
    args:
        - ich: Number of channel of the input
        - och: Number of channel of the output
    """
    def __init__(self, ich, och):
        super(Down, self).__init__()
        
        self.pool = nn.MaxPool2d(2)
        self.conv = Double_conv(ich, och)
        
    def forward(self,x):
        x = self.pool(x)
        x = self.conv(x)
        return x
        
class Up(nn.Module):

    """
    Apply a decoding deconvolution, multiplies the image size by two
    args:
        - ich: Number of channel of the input
        - och: Number of channel of the output
    """
    def __init__(self, ich, och):
        super(Up, self).__init__()
        
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                nn.Conv2d(ich, ich // 2, kernel_size=1, bias=False),
                                nn.BatchNorm2d(ich // 2))
        #self.conv = Double_conv(och + ich // 2, och)
        self.conv = Double_conv(ich // 2, och)
        
    def forward(self, enc_layer, dec_layer):
        dec = self.up(dec_layer)
        #x = torch.cat([enc_layer, dec], dim=1)
        #x = self.conv(x)
        x = self.conv(dec)
        return x        

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class UNET(nn.Module):

    """
    Implements a unet network with convolutional encoding and deconvolutional decoding.
    The forward method implements skip connections between encoder and decoder to keep consistency in the features
    arg:
        n_classes: The number of channel output
    """
    def __init__(self, rep_dim):
        super(UNET, self).__init__()

        self.input = Double_conv(3, 16)
        self.e1 = Down(16, 32)
        self.e2 = Down(32, 64)
        self.e3 = Down(64, 128)
        self.e4 = Down(128, 256)
        self.e5 = Down(256, 512)
        self.e6 = Down(512, 1024)
        self.feat = nn.Conv2d(1024, rep_dim, 4, bias=False)
        self.l2norm = Normalize()

        self.defeat = nn.ConvTranspose2d(rep_dim, 1024, 4, bias=False)
        self.d1 = Up(1024, 512)
        self.d2 = Up(512, 256)
        self.d3 = Up(256, 128)
        self.d4 = Up(128, 64)
        self.d5 = Up(64, 32)
        self.d6 = Up(32, 16)
        
        self.final = nn.Conv2d(16, 3, 1, 1, 0, bias=False)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        e1 = self.input(x)
        e2 = self.e1(e1)
        e3 = self.e2(e2)
        e4 = self.e3(e3)
        e5 = self.e4(e4)
        e6 = self.e5(e5)
        e7 = self.e6(e6)

        feature = self.l2norm(self.feat(e7))
        defeature = self.defeat(feature)
        x = self.d1(e6, defeature)
        x = self.d2(e5, x)
        x = self.d3(e4, x)
        x = self.d4(e3, x)
        x = self.d5(e2, x)
        x = self.d6(e1, x)
        x = self.final(x)
        x = self.activation(x)
        
        return feature.squeeze(), x
