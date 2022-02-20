import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .spectral_norm import spectral_norm as _spectral_norm
from .NonLocalBlock import NONLocalBlock3D

class Discriminator(nn.Module):

    def __init__(self, in_channels=3, use_spectral_norm=True, init_weights=True, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(spectral_norm(nn.Conv3d(in_channels, conv_dim, kernel_size=(3,4,4), stride=(1, 2, 2),
                                              padding=(1, 2, 2)), True))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=(3,4,4), stride=(1, 2, 2),
                                              padding=(1, 2, 2)), True))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=(3,4,4), stride=(1, 2, 2),
                                              padding=(1, 2, 2)), True))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=(3,4,4), stride=(1, 2, 2),
                                                  padding=(1, 2, 2)), True))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv3d(curr_dim, 1, kernel_size=(3,4,4)))
        self.last = nn.Sequential(*last)

    def forward(self, x):

        x = torch.transpose(x, 1, 2)

        out = self.l1(x) 
        out = self.l2(out) 
        out = self.l3(out) 
        out=self.l4(out) 
        out=self.last(out)

        out = torch.transpose(out, 1, 2)

        return out

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module