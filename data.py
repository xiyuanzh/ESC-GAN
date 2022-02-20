from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import torch

class Prep(Dataset):

    def __init__(self, seq_len):

        dset = xr.open_dataset("./data/CMAP.nc")
        temp_nc = dset.variables['precip'][:492] #up to year 2019
        timespan = len(temp_nc)
        lat = len(temp_nc[0])
        lon = len(temp_nc[0][0])
        temp_arr = temp_nc.values

        mean, var = self.z_norm(temp_arr)
        temp_arr = (temp_arr - self.mean) / self.var

        self.valid_mask = (temp_arr == temp_arr) + 0

        temp_arr[temp_arr != temp_arr] = 0 #set nan to zero
        self.temp = torch.from_numpy(temp_arr).unsqueeze(3)
        self.temp = torch.stack(list(torch.split(self.temp, seq_len))).float()

        self.valid_mask = torch.stack(list(torch.split(torch.from_numpy(self.valid_mask), seq_len))).unsqueeze(4).float()

        test_mask = np.load('./data/cmap_mask.npy').reshape((lat, lon, 1))
        self.test_mask = torch.from_numpy(test_mask).repeat(len(self.temp), seq_len, 1, 1, 1).float()

    def __len__(self):
        return len(self.temp)

    def __getitem__(self, idx):
        return self.temp[idx], self.valid_mask[idx], self.test_mask[idx]

    def z_norm(self, x):
        self.mean = np.nanmean(x)
        self.var = np.nanstd(x)
        return self.mean, self.var

    def de_z_norm(self, x):
        return x * self.var + self.mean


class Hadcrut(Dataset):

    def __init__(self, seq_len):

        dset = xr.open_dataset("./data/HadCRUT.nc")
        temp_nc = dset.variables['temperature_anomaly'][:2040]  # up to year 2019
        timespan = len(temp_nc)
        lat = len(temp_nc[0])
        lon = len(temp_nc[0][0])
        temp_arr = temp_nc.values

        mean, var = self.z_norm(temp_arr)
        temp_arr = (temp_arr - self.mean) / self.var

        self.valid_mask = (temp_arr == temp_arr) + 0

        temp_arr[temp_arr != temp_arr] = 0 #set nan to zero
        self.temp = torch.from_numpy(temp_arr).unsqueeze(3)
        self.temp = torch.stack(list(torch.split(self.temp, seq_len))).float()

        self.valid_mask = torch.stack(list(torch.split(torch.from_numpy(self.valid_mask), seq_len))).unsqueeze(4).float()

        test_mask = np.load('./data/hadcrut_mask.npy').reshape((lat, lon, 1))
        self.test_mask = torch.from_numpy(test_mask).repeat(len(self.temp), seq_len, 1, 1, 1).float()

    def __len__(self):
        return len(self.temp)

    def __getitem__(self, idx):
        return self.temp[idx], self.valid_mask[idx], self.test_mask[idx]

    def z_norm(self, x):
        self.mean = np.nanmean(x)
        self.var = np.nanstd(x)
        return self.mean, self.var

    def de_z_norm(self, x):
        return x * self.var + self.mean