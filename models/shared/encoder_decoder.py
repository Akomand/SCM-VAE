import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
import sys
sys.path.append('../')

from models.shared import utils as ut

class Encoder(nn.Module):
    def __init__(self, z_dim, channel=4, y_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.channel = channel
        self.fc1 = nn.Linear(self.channel * 96 * 96, 300)
        self.fc2 = nn.Linear(300 + y_dim, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 2 * z_dim)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.net = nn.Sequential(
            nn.Linear(self.channel * 96 * 96, 900),
            nn.ELU(),
            nn.Linear(900, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )

    def conditional_encode(self, x, l):
        x = x.view(-1, self.channel * 96 * 96)
        x = F.elu(self.fc1(x))
        l = l.view(-1, 4)
        x = F.elu(self.fc2(torch.cat([x, l], dim=1)))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        m, v = ut.gaussian_parameters(x, dim=1)
        return m, v

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        # print(len(xy))
        xy = xy.view(-1, self.channel * 96 * 96)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        # print(self.z_dim,m.size(),v.size())
        return m, v


class Decoder_DAG(nn.Module):
    def __init__(self, z_dim, concept, z1_dim, channel=4, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept
        self.y_dim = y_dim
        self.channel = channel
        # print(self.channel)
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net5 = nn.Sequential(
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )

        self.net6 = nn.Sequential(
            nn.Linear(z_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )

    def decode_condition(self, z, u):
        # z = z.view(-1,3*4)
        z = z.view(-1, 4 * 4)
        z1, z2, z3, z4 = torch.split(z, self.z_dim // 4, dim=1)
        # print(z1.shape)
        # exit(0)
        # print(u[:,0].reshape(1,u.size()[0]).size())
        rx1 = self.net1(
            torch.transpose(torch.cat((torch.transpose(z1, 1, 0), u[:, 0].reshape(1, u.size()[0])), dim=0), 1, 0))
        rx2 = self.net2(
            torch.transpose(torch.cat((torch.transpose(z2, 1, 0), u[:, 1].reshape(1, u.size()[0])), dim=0), 1, 0))
        rx3 = self.net3(
            torch.transpose(torch.cat((torch.transpose(z3, 1, 0), u[:, 2].reshape(1, u.size()[0])), dim=0), 1, 0))
        rx4 = self.net4(
            torch.transpose(torch.cat((torch.transpose(z4, 1, 0), u[:, 2].reshape(1, u.size()[0])), dim=0), 1, 0))
        temp = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        # print(temp.shape)
        # exit(0)
        # h = self.net6(torch.cat((rx1, rx2, rx3, rx4), dim=1))

        h = (rx1 + rx2 + rx3 + rx4) / 4

        return h

    def decode_mix(self, z):
        z = z.permute(0, 2, 1)
        z = torch.sum(z, dim=2, out=None)
        # print(z.contiguous().size())
        z = z.contiguous()
        h = self.net1(z)
        return h

    def decode_union(self, z, u, y=None):

        z = z.view(-1, self.concept * self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5((rx1 + rx2 + rx3 + rx4) / 4)
        return h, h, h, h, h

    def decode(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        h = self.net6(z)
        return h, h, h, h, h

    def decode_sep(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)

        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
            elif self.concept == 3:
                zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
        else:
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
            elif self.concept == 3:
                zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept == 4:
            rx4 = self.net4(zy4)
            h = (rx1 + rx2 + rx3 + rx4) / self.concept
        elif self.concept == 3:
            h = (rx1 + rx2 + rx3) / self.concept

        return h, h, h, h, h

    def decode_cat(self, z, u, y=None):
        z = z.view(-1, 4 * 4)
        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2, zy3, zy4 = torch.split(zy, 1, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5(torch.cat((rx1, rx2, rx3, rx4), dim=1))
        return h
