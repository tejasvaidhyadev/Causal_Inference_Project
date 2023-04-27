import torch
import torch.nn as nn

# Define the MMD regularizer
# This module need to be recheck for futhur development


class MMDLoss(nn.Module):
    def __init__(self, kernel_bandwidth):
        super(MMDLoss, self).__init__()
        self.kernel_bandwidth = kernel_bandwidth

    def forward(self, f0, f1):
        n = f0.shape[0]
        m = f1.shape[0]
        kxx = torch.exp(-torch.sum((f0.unsqueeze(1) - f0.unsqueeze(0)) ** 2, dim=2) / (2 * self.kernel_bandwidth ** 2)).mean()
        kyy = torch.exp(-torch.sum((f1.unsqueeze(1) - f1.unsqueeze(0)) ** 2, dim=2) / (2 * self.kernel_bandwidth ** 2)).mean()
        kxy = torch.exp(-torch.sum((f0.unsqueeze(1) - f1.unsqueeze(0)) ** 2, dim=2) / (2 * self.kernel_bandwidth ** 2)).mean()
        return kxx + kyy - 2 * kxy
