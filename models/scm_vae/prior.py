import numpy as np
import os
import shutil
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils import data
import torch.utils.data as Data
from torch.distributions.multivariate_normal import MultivariateNormal
from PIL import Image



class CausalPrior():
    """
    Definition of the Structural Causal Prior in SCM-VAE

    Parameters
    ----------
    scale: numpy array [float]
           Normalizing for each causal variable
    label: numpy array [float]
           Label vector of ground-truth causal variables for given batch of images
    dim: int
         Dimension of the latent variable
    A: numpy array [int]
       Binary adjacency matrix consisting of true causal structure between variables
    """
    def __int__(self):
        super().__init__()
        self.scale = scale
        self.label = label
        self.dim = dim
        self.A = A

    # Conditional Structure Prior
    def structural_caual_prior(self, scale, label, dim,
                                   A):  # CHANGE TO WHEN WE ARE INTERVENING ON 3RD OR 4TH CONCEPT TO HARDCODE IT
        mean = torch.ones(label.size()[0], label.size()[1], dim)
        var = torch.ones(label.size()[0], label.size()[1], dim)
        I = torch.eye(4).to(device)
        for i in range(label.size()[0]):
            for j in range(label.size()[1]):
                inp = A.to(device).t() + I

                num_parents = torch.count_nonzero((inp), dim=1)
                norm_label = (label[i].to(device) - torch.tensor(scale[:, 0]).to(device)) / (
                    torch.tensor(scale[:, 1]).to(device))

                mul = torch.matmul((inp).float(), norm_label.float().to(device))[j]
                mul = mul / num_parents[j]  # averaging

                mean[i][j] = torch.ones(dim) * mul.item()

                var[i][j] = torch.ones(dim) * 1
        return mean, var