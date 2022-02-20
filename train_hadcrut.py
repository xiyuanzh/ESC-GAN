import argparse
import datetime
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from models.Discriminator import Discriminator
from models.Generator import P3DConvUNet
from loss import ESCLoss

from data import Hadcrut

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--seq', type=int, default=60, help='length of the sequence')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='./log/', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='hadcrut', help='tags for the current run')
parser.add_argument('--model_path', default='./model/', help='path to save the model')
opt = parser.parse_args()

# Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = f"{opt.run_tag}_{date}" if opt.run_tag != '' else date
log_dir_name = os.path.join(opt.logdir, run_name)
writer = SummaryWriter(log_dir_name)
writer.add_text('Options', str(opt), 0)
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")
device = torch.device("cuda:0" if opt.cuda else "cpu")

dataset = Hadcrut(opt.seq)
print("dataset:", len(dataset))
assert dataset

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False)

netG = P3DConvUNet(input_channels=1).to(device)
netD = Discriminator(use_spectral_norm=True, in_channels=1).to(device)
act = nn.ReLU()

optimizerG = optim.Adam(netG.parameters(), lr=5e-3)
optimizerD = optim.Adam(netD.parameters(), lr=5e-3)

supervised_criterion = nn.MSELoss().to(device)
adv_criterion = nn.ReLU()
overall_all = ESCLoss().to(device)

for epoch in range(opt.epochs):

    totalG, totalD, cnt, totalG_adv = 0, 0, 0, 0
    for i, tuple in enumerate(train_dataloader, 0):

        cnt += 1
        niter = epoch * len(train_dataloader) + i
        data = tuple[0].permute(0, 1, 4, 2, 3).to(device)
        b, t, c, h, w = data.size()
        valid_mask = tuple[1].permute(0, 1, 4, 2, 3).to(device)
        test_mask = tuple[2].permute(0, 1, 4, 2, 3).to(device)

        train_mask = np.zeros(h * w)
        p = random.uniform(0.01, 0.5)
        idx = np.random.choice(np.arange(0, h * w), int(h * w * p), replace=False)
        train_mask[idx] = 1
        train_mask = torch.from_numpy(train_mask.reshape((h, w))).to(device).float()

        input = data * (1 - test_mask) * (1 - train_mask)

        gen_loss, dis_loss, hole_loss, non_hole_loss = 0, 0, 0, 0

        fake, attn, attn_scale = netG(input, (1 - test_mask) * valid_mask * (1 - train_mask)) #vis for the first batch
        gen_loss += overall_all(train_mask, test_mask, valid_mask, fake, data)

        real_vid_feat = netD((data * (1 - test_mask)))
        comp_data = fake.view(b,t,c,h,w) * (1 - test_mask) * valid_mask
        fake_vid_feat = netD(comp_data.detach())

        dis_real_loss = act(1 - real_vid_feat).mean()
        dis_fake_loss = act(1 + fake_vid_feat).mean()
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        optimizerD.zero_grad()
        dis_loss.backward()
        optimizerD.step()
        totalD += dis_loss.item()

        if (niter % 5 == 0):
            gen_vid_feat = netD(comp_data)
            gan_loss = -(gen_vid_feat).mean()
            gen_loss += 0.001 * gan_loss
            totalG_adv += gan_loss

        optimizerG.zero_grad()
        gen_loss.backward()
        optimizerG.step()
        totalG += gen_loss.item()
    
    print("totalG: %.4f totalD: %.4f totalGadv: %.4f" %
          (totalG / cnt, totalD / cnt, totalG_adv / cnt))

    with torch.no_grad():

        val_mse, zero_mse, mean_mse, val_mae, cnt_mse = 0, 0, 0, 0, 0
        for i, tuple in enumerate(val_dataloader, 0):

            data = tuple[0].permute(0, 1, 4, 2, 3).to(device)
            b, t, c, h, w = data.size()
            valid_mask = tuple[1].permute(0, 1, 4, 2, 3).to(device)
            test_mask = tuple[2].permute(0, 1, 4, 2, 3).to(device)
            input = data * (1 - test_mask)

            cnt_mse += (valid_mask * test_mask).sum()

            fake, _, _ = netG(input, (1 - test_mask) * valid_mask)
            val_mse += ((fake.view(b, t, c, h, w) * valid_mask * test_mask - data * valid_mask * test_mask) ** 2).sum()
            val_mae += (torch.abs(fake.view(b, t, c, h, w) * valid_mask * test_mask - data * valid_mask * test_mask)).sum()

            zero_mse += ((input * valid_mask * test_mask - data * valid_mask * test_mask) ** 2).sum()

            filled_mean = input.view(b * t * c, h * w).clone().detach().cpu().numpy()
            filled_mean[filled_mean == 0] = np.nan
            for cur in range(b * t * c):
                filled_mean[cur][filled_mean[cur] != filled_mean[cur]] = np.nanmean(filled_mean[cur])
            filled_mean = torch.from_numpy(filled_mean).view(b, t, c, h, w).to(device)
            mean_mse += ((filled_mean * valid_mask * test_mask - data * valid_mask * test_mask) ** 2).sum()

    print('[%d/%d] Ours MSE: %.4f Zero MSE: %.4f Mean MSE: %.4f Our MAE: %.4f' %
          (epoch, opt.epochs, val_mse / cnt_mse, zero_mse / cnt_mse, mean_mse / cnt_mse, val_mae / cnt_mse))
    writer.add_scalar('val MSE', val_mse / cnt_mse, epoch)
    writer.add_scalar('val MAE', val_mae / cnt_mse, epoch)

    if (epoch % 100 == 0):
        torch.save(netG.state_dict(), opt.model_path + opt.run_tag + str(epoch))