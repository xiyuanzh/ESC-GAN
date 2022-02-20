import torch
from torch import nn
import math
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module): 
    
    def __init__(self,in_dim, out_dim, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim
        
        self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        
        if(sub_sample):
            self.key_conv = nn.Sequential(nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1),
                                nn.MaxPool3d(kernel_size=(1, 2, 2)))
            self.value_conv = nn.Sequential(nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1),
                                nn.MaxPool3d(kernel_size=(1, 2, 2)))
        else:
            self.key_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
            self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.W = nn.Sequential(nn.Conv3d(in_channels = in_dim , out_channels = out_dim, kernel_size= 1),
                               nn.BatchNorm3d(out_dim))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.W_in = nn.Sequential(nn.Conv3d(in_channels = in_dim , out_channels = out_dim, kernel_size= 1),
                              nn.BatchNorm3d(out_dim))

        self.softmax  = nn.Softmax(dim=-1) #

        self.method = 'disentangle'

    def forward(self,x,st=True):

        m_batchsize,C,t,width,height = x.size()
        if(st == True):
            proj_query  = self.query_conv(x).permute(0,2,3,4,1).reshape(m_batchsize,-1,self.chanel_in//8) # (b,c//8,t,w,h) -> (b,t*w*h,c//8)
            proj_key =  self.key_conv(x).reshape(m_batchsize,self.chanel_in//8,-1) # (b,c//8,t,w/2,h/2) -> (b,c//8,t*w*h/4)
            proj_value = self.value_conv(x).reshape(m_batchsize,self.chanel_in,-1) # (b,c,t,w/2,h/2) -> (b,c,t*w*h/4)
        else:
            proj_query  = self.query_conv(x).permute(0,2,3,4,1).reshape(m_batchsize*t,-1,self.chanel_in//8) # (b,c//8,t,w,h) -> (b*t,w*h,c//8)
            proj_key =  self.key_conv(x).permute(0,2,1,3,4).reshape(m_batchsize*t,self.chanel_in//8,-1) # (b,c//8,t,w/2,h/2) -> (b*t,c//8,w*h/4)
            proj_value = self.value_conv(x).permute(0,2,1,3,4).reshape(m_batchsize*t,self.chanel_in,-1) # (b,c,t,w/2,h/2) -> (b*t,c,w*h/4)

        if self.method == 'regular':
            energy =  torch.bmm(proj_query,proj_key) / math.sqrt(proj_query.size(1)) # (b*t,w*h,w*h/4)
        elif self.method == 'disentangle':
            u_query = proj_query.mean(1).unsqueeze(1).expand(-1,proj_query.size(1),-1)
            u_key = proj_key.mean(2).unsqueeze(2).expand(-1,-1,proj_key.size(2))
            energy = torch.bmm(proj_query-u_query, proj_key-u_key) + torch.bmm(u_query,proj_key)
            energy /= math.sqrt(proj_query.size(1))
        elif self.method == 'abs':
            M1 = proj_query.unsqueeze(1).expand(-1,proj_query.size(1),-1,-1)
            M2 = proj_query.unsqueeze(2).expand(-1,-1,proj_query.size(1),-1)
            norm_diff = torch.abs(torch.norm(M1 - M2, dim=3))
            smoothing = torch.topk(norm_diff, k=36, largest=False, dim=2)[0][:,:,-1].unsqueeze(2).expand(-1,-1,proj_query.size(1))
            selected_idx = (norm_diff < smoothing)
            norm_diff[selected_idx] = smoothing[selected_idx]
            energy = torch.log(torch.abs( 1 / norm_diff) / math.sqrt(proj_query.size(1)))

        attention = self.softmax(energy) # (b*t,w*h,w*h/4) 

        out = torch.bmm(proj_value,attention.permute(0,2,1)) # (b*t,c,w*h)
        out = out.reshape(m_batchsize,t,C,width,height).permute(0,2,1,3,4)
        
        out = self.W(out) + self.W_in(x)

        return out, attention
     
class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, out_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              out_dim=out_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
