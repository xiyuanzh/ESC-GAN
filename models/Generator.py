import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .NonLocalBlock import NONLocalBlock3D
from matplotlib import pyplot as plt


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1),
                 padding=(0,0,0), dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # https://nv-adlr.github.io/publication/partialconv-inpainting
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, pad=1, di=1, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()

        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=(3,5,5), stride=(1,2,2),
                                    padding=(pad,2,2), bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=(3,7,7), stride=(1,2,2),
                                    padding=(pad,3,3), bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=(3,3,3), stride=(1,2,2),
                                    padding=(pad,pad,pad), dilation=di, bias=conv_bias)
        elif sample == 'none-3':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=(3,3,3), stride=(1,1,1),
                                    padding=(pad,pad,pad), dilation=di, bias=conv_bias)
        elif sample == 'none-5':
            self.conv = PartialConv(in_ch, out_ch, kernel_size=(3,5,5), stride=(1,1,1),
                                    padding=(pad,2,2), dilation=di, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm3d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        _, ch, _, _, _ = h.size()
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class P3DConvUNet(nn.Module):
    def __init__(self, layer_size=4, input_channels=1, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv(input_channels, 64, bn=True, sample='down-3')
        self.enc_2 = PCBActiv(64, 128, bn=True, sample='down-3')
        self.enc_3 = PCBActiv(128, 256, bn=True)
        self.enc_4 = PCBActiv(256, 512, bn=True)
        self.nonlocalblock1 = NONLocalBlock3D(in_channels=512, out_channels=512, sub_sample=True)

        self.enc_scale_1 = PCBActiv(input_channels, 64, bn=True, sample='down-7')
        self.enc_scale_2 = PCBActiv(64, 128, bn=True, sample='down-5')
        self.enc_scale_3 = PCBActiv(128, 256, bn=True)
        self.enc_scale_4 = PCBActiv(256, 512, bn=True)
        self.nonlocalblock2 = NONLocalBlock3D(in_channels=512, out_channels=512, sub_sample=True)


        self.dec_4 = PCBActiv(512 + 256, 256, bn=True, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, bn=True, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64, 64, bn=True, activ='leaky')
        self.dec_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=True, activ=None, conv_bias=True)
        
        self.dec_scale_4 = PCBActiv(512 + 256, 256, bn=True, activ='leaky')
        self.dec_scale_3 = PCBActiv(256 + 128, 128, bn=True, activ='leaky')
        self.dec_scale_2 = PCBActiv(128 + 64, 64, bn=True, activ='leaky')
        self.dec_scale_1 = PCBActiv(64 + input_channels, input_channels,
                              bn=True, activ=None, conv_bias=True)
       
        self.dec_merge = nn.Conv3d(input_channels*2, input_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, input, input_mask, vis=False):
        b, t, c, height, weight = input.size()
        input = input.permute(0, 2, 1, 3, 4)
        input_mask = input_mask.permute(0, 2, 1, 3, 4) #"1" means valid in partial conv

        ######################################## scale 1 ########################################
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        h, attn = self.nonlocalblock1(h)

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            if (i == 2 or i == 1):
                h = F.interpolate(h, scale_factor=(1,2,2), mode=self.upsampling_mode)
                h_mask = F.interpolate(h_mask, scale_factor=(1,2,2), mode='nearest')
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
    
        ######################################## scale 2 ########################################
        h_scale_dict = {}  # for the output of enc_N
        h_scale_mask_dict = {}  # for the output of enc_N

        h_scale_dict['h_0'], h_scale_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_scale_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_scale_dict[h_key], h_scale_mask_dict[h_key] = getattr(self, l_key)(
                h_scale_dict[h_key_prev], h_scale_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h_scale, h_scale_mask = h_scale_dict[h_key], h_scale_mask_dict[h_key]

        h_scale, attn_scale = self.nonlocalblock2(h_scale)

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            if (i == 2 or i == 1):
                h_scale = F.interpolate(h_scale, scale_factor=(1,2,2), mode=self.upsampling_mode)
                h_scale_mask = F.interpolate(h_scale_mask, scale_factor=(1,2,2), mode='nearest')
            h_scale = torch.cat([h_scale, h_scale_dict[enc_h_key]], dim=1)
            h_scale_mask = torch.cat([h_scale_mask, h_scale_mask_dict[enc_h_key]], dim=1)
            h_scale, h_scale_mask = getattr(self, dec_l_key)(h_scale, h_scale_mask)
            
        h = self.dec_merge(torch.cat((h,h_scale),dim=1))
      
        return h.permute(0,2,1,3,4).reshape(b*t,c,height,weight), attn, attn_scale
