import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class MaskLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z1_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ELU(),
            nn.Linear(32, z_dim),
        )
        self.net_g = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ELU(),
            nn.Linear(32, 4),
        )

    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def masked_sep(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def g(self, z, i=None):
        rx = self.net_g(z)

        return rx

    def mix(self, z):
        zy = z.view(-1, self.concept * self.z1_dim)
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
            h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        elif self.concept == 3:
            h = torch.cat((rx1, rx2, rx3), dim=1)
        # print(h.size())
        return h



class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features, i=False, bias=False):
        super(nn.Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features, out_features)

        self.a[0, 2:4] = 1
        self.a[1, 2:4] = 1

        # self.a[0, 1], self.a[1, 2], self.a[3, 2] = 1, 1, 1
        # self.a[0, 2], self.a[1, 3], self.a[2, 3] = 1, 1, 1

        # self.A = nn.Parameter(self.a)
        self.A = self.a.to(device)

        self.b = torch.eye(out_features)
        self.b = self.b
        self.B = self.b.to(device)
        # self.B = nn.Parameter(self.b)

        self.I = torch.eye((out_features)).to(device)
        # self.I = nn.Parameter(torch.eye(out_features))
        # self.I.requires_grad = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def mask_z(self, x, i):
        self.B = self.A
        x = torch.mul((self.B + self.I)[:, i].reshape(4, 1).clone(), x.clone())

        return x

    def mask_z_learn(self, x, i):
        self.B = self.A

        # x = F.linear(x.clone(), (self.B + self.I)[:, i].reshape(4, 1).clone(), self.bias)
        x = torch.mul((self.B + self.I)[:, i].reshape(4, 1).clone(), x.clone())

        return x


    def mask_u(self, x):
        self.B = self.A
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x

    def inv_cal(self, x, v):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
        x = F.linear(x, self.I - self.A, self.bias)

        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x, v

    def calculate_dag(self, x, v):

        if x.dim() > 2:
            x = x.permute(0, 2, 1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias)

        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x, v

    def calculate_cov(self, x, v):
        # print(self.A)
        v = ut.vector_expand(v)
        # x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        # print(v)
        return x, v

    def calculate_gaussian_ini(self, x, v):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
        x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
        v = F.linear(v, torch.mul(torch.inverse(self.I - self.A), torch.inverse(self.I - self.A)), self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
            v = v.permute(0, 2, 1).contiguous()
        return x, v

    # def encode_
    def forward(self, x):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)

        x = torch.matmul(x, torch.inverse(self.I - self.A.t()).t())

        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()

        return x

    def calculate_gaussian(self, x, v):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
            v = v.permute(0, 2, 1).contiguous()
        return x, v