import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def total_variation_loss_mask(image, mask):
    # https://github.com/jacobaustin123/pytorch-inpainting-partial-conv/blob/master/loss.py
    dilated = F.conv2d(1 - mask, torch.ones((1, image.size(1), 3, 3)).cuda(), padding=1)
    dilated[dilated != 0] = 1

    eroded = F.conv2d(mask, torch.ones((1, image.size(1), 3, 3)).cuda(), padding=1)
    eroded[eroded != 0] = 1
    eroded = 1 - eroded

    mask = dilated - eroded # slightly more than the 1-pixel region in the paper, but okay

    # adapted from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
    loss = (torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]) * mask[:,:,:,:-1]).mean() + \
        (torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]) * mask[:,:,:-1,:]).mean()

    return loss

class ESCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, train_mask, test_mask, valid_mask, fake, data):
        b, t, c, h, w = data.size()
        data = data.view(b*t, c, h, w)
        valid_mask = valid_mask.view(b*t, c, h, w)
        test_mask = test_mask.view(b*t, c, h, w)
        output = fake * valid_mask * (1 - test_mask)
        gt = data * valid_mask * (1 - test_mask)

        loss_dict = {}
        output_comp = output * train_mask + gt * (1-train_mask) + \
                      fake * (1-valid_mask) + fake * test_mask - fake * test_mask * (1-valid_mask)

        supervised_criterion = nn.MSELoss().cuda()
        loss_dict['hole'] = supervised_criterion(output * train_mask, gt * train_mask)

        mask = valid_mask*(train_mask + test_mask)
        mask[mask != 0] = 1
        loss_dict['tv'] = total_variation_loss_mask(output_comp, mask)

        loss_value = loss_dict['hole'] + 0.1*loss_dict['tv']

        return loss_value
