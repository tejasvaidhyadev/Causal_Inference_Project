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

def calculate_label_flips(original_preds, perturbed_preds):
    assert len(original_preds) == len(perturbed_preds)
    flips = sum(original_preds[i] != perturbed_preds[i] for i in range(len(original_preds)))
    return flips / len(original_preds)

def compute_marginal_regularization(model, mmd_loss, pooled_output, Z):
    f_X_z0 = pooled_output[Z == 0]
    f_X_z0 = model.classifier(f_X_z0)

    f_X_z1 = pooled_output[Z == 1]
    f_X_z1 = model.classifier(f_X_z1)
    return mmd_loss(f_X_z0, f_X_z1)

def compute_conditional_regularization(model, mmd_loss, pooled_output, Z, Y, y_value):
    f_X_z0_y = pooled_output[(Z == 0) & (Y == y_value)]    
    f_X_z0_y = model.classifier(f_X_z0_y)
    f_X_z1_y = pooled_output[(Z == 1) & (Y == y_value)]
    f_X_z1_y = model.classifier(f_X_z1_y)

    return mmd_loss(f_X_z0_y, f_X_z1_y)

        
def compute_regularization_term(model, mmd_loss, pooled_output, Z, labels, is_marginal_reg):
    """
    Compute the regularization term based on the specified type (marginal or conditional).

    Parameters:
    - mmd_loss: The MMD loss function.
    - pooled_output: The pooled output from the model.
    - Z: The latent variable.
    - labels: The labels for the data.
    - is_marginal_reg: A boolean flag indicating whether to compute the marginal regularization term.

    Returns:
    - reg_term: The computed regularization term.
    """

    # Compute the marginal regularization term
    if is_marginal_reg:
        marginal_reg = compute_marginal_regularization(model, mmd_loss, pooled_output, Z)
        reg_term = marginal_reg
    else:
        # Compute the conditional regularization terms
        cond_reg_y0 = compute_conditional_regularization(model, mmd_loss, pooled_output, Z, labels, 0)

        # Compute the conditional regularization terms
        cond_reg_y1 = compute_conditional_regularization(model, mmd_loss, pooled_output, Z, labels, 1)
        
        reg_term = cond_reg_y0 + cond_reg_y1

    return reg_term

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

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)