import torch
import torch.nn as nn
import logging
import os

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MMDLoss(nn.Module):
    def __init__(self, kernel_bandwidth=None):
        super(MMDLoss, self).__init__()
        if kernel_bandwidth is None:
            # Create log-spaced regularization coefficients between 0 and 128
            kernel_bandwidth = np.logspace(0, np.log10(128), num=10)
        self.kernel_bandwidth = torch.tensor(kernel_bandwidth, dtype=torch.float32)

    def gaussian_rbf_kernel(self, x, y, bandwidth):
        # Compute the pairwise squared distances between the samples in x and y
        xx = torch.sum(x**2, dim=1, keepdim=True)
        yy = torch.sum(y**2, dim=1, keepdim=True)
        xy = torch.matmul(x, y.t())
        squared_distances = xx - 2 * xy + yy.t()

        # Compute the Gaussian RBF kernel using the squared distances
        kernel = torch.exp(-squared_distances / (2 * bandwidth**2))
        return kernel

    def forward(self, source_samples, target_samples):
        # Compute the log probabilities of the source and target samples
        source_log_probs = torch.log_softmax(source_samples, dim=1)
        target_log_probs = torch.log_softmax(target_samples, dim=1)

        # Compute the MMD using the Gaussian RBF kernel for each bandwidth
        mmds = []
        for bandwidth in self.kernel_bandwidth:
            xx_kernel = self.gaussian_rbf_kernel(source_log_probs, source_log_probs, bandwidth)
            yy_kernel = self.gaussian_rbf_kernel(target_log_probs, target_log_probs, bandwidth)
            xy_kernel = self.gaussian_rbf_kernel(source_log_probs, target_log_probs, bandwidth)

            mmd = torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2 * torch.mean(xy_kernel)
            mmds.append(mmd)

        # Average the MMD values computed for each coefficient
        avg_mmd = torch.mean(torch.stack(mmds))
        return avg_mmd

    
# class MMDLoss(nn.Module):
#     def __init__(self, kernel_bandwidth=10.0):
#         super(MMDLoss, self).__init__()
#         self.kernel_bandwidth = kernel_bandwidth

#     def gaussian_rbf_kernel(self, x, y, bandwidth):
#         # Compute the pairwise squared distances between the samples in x and y
#         xx = torch.sum(x**2, dim=1, keepdim=True)
#         yy = torch.sum(y**2, dim=1, keepdim=True)
#         xy = torch.matmul(x, y.t())
#         squared_distances = xx - 2 * xy + yy.t()

#         # Compute the Gaussian RBF kernel using the squared distances
#         kernel = torch.exp(-squared_distances / (2 * bandwidth**2))
#         return kernel

#     def forward(self, source_samples, target_samples):
#         # Compute the log probabilities of the source and target samples
#         source_log_probs = torch.log_softmax(source_samples, dim=1)
#         target_log_probs = torch.log_softmax(target_samples, dim=1)

#         # Compute the MMD using the Gaussian RBF kernel
#         xx_kernel = self.gaussian_rbf_kernel(source_log_probs, source_log_probs, self.kernel_bandwidth)
#         yy_kernel = self.gaussian_rbf_kernel(target_log_probs, target_log_probs, self.kernel_bandwidth)
#         xy_kernel = self.gaussian_rbf_kernel(source_log_probs, target_log_probs, self.kernel_bandwidth)

#         mmd = torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2 * torch.mean(xy_kernel)
#         return mmd


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)



