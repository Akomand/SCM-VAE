import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim


class ConvEncoder(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        # init 128*128
        # 64x64x32, 32x32x64, 16x16x64, 8x8x64, 4x4x256, 4x4x3 (want
        # init 96*96
        self.conv1 = torch.nn.Conv2d(3, 32, 4, 2, 1)  # 48*48
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False)  # 24*24
        self.conv3 = torch.nn.Conv2d(64, 1, 4, 2, 1, bias=False)
        # self.conv4 = torch.nn.Conv2d(128, 1, 1, 1, 0) # 54*44

        self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.convm = torch.nn.Conv2d(1, 1, 4, 2, 1)
        self.convv = torch.nn.Conv2d(1, 1, 4, 2, 1)
        self.mean_layer = nn.Sequential(
            torch.nn.Linear(8 * 8, 128)
        )  # 12*12
        self.var_layer = nn.Sequential(
            torch.nn.Linear(8 * 8, 128)
        )
        # self.fc1 = torch.nn.Linear(6*6*128, 512)
        self.conv6 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1), # 4x4x256
            nn.ReLU(True),
            nn.Conv2d(256, 64, 4, 2, 1) # 2x2x64
        )

    def encode(self, x):
        x = self.LReLU(self.conv1(x))
        x = self.LReLU(self.conv2(x))
        x = self.LReLU(self.conv3(x))
        # x = self.LReLU(self.conv4(x))
        # print(x.size())
        hm = self.convm(x)
        # print(hm.size())
        hm = hm.view(-1, 8 * 8)
        hv = self.convv(x)
        hv = hv.view(-1, 8 * 8)
        mu, var = self.mean_layer(hm), self.var_layer(hv)
        var = F.softplus(var) + 1e-8
        # var = torch.reshape(var, [-1, 16, 16])
        # print(mu.size())
        return mu, var

    def encode_simple(self, x):
        x = self.conv6(x)
        x = x.reshape(x.shape[0], 256)
        # print(x.shape)
        # exit(0)
        m, v = ut.gaussian_parameters(x, dim=1)
        # print(m.size())
        return m, v


class ConvDecoder(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()

        self.net6 = nn.Sequential(
            nn.Conv2d(32, 128, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def decode_sep(self, x):
        return None

    def decode(self, z):
        z = z.view(-1, 32, 1, 1)
        z = self.net6(z)
        return z


class ConvDec(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        self.concept = 4
        self.z1_dim = 32
        self.z_dim = 128
        self.net1 = ConvDecoder()
        self.net2 = ConvDecoder()
        self.net3 = ConvDecoder()
        self.net4 = ConvDecoder()
        self.net5 = nn.Sequential(
            nn.Linear(16, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024)
        )
        self.net6 = nn.Sequential(
            nn.Conv2d(16, 128, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def decode_sep(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)

        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1.decode(zy1)
        # print(rx1.size())
        rx2 = self.net2.decode(zy2)
        rx3 = self.net3.decode(zy3)
        rx4 = self.net4.decode(zy4)
        z = (rx1 + rx2 + rx3 + rx4) / 4
        return z, z, z, z, z

    def decode(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim, 1, 1)
        z = self.net6(z)
        # print(z.size())

        return z

