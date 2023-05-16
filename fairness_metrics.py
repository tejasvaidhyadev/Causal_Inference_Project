import torch

def demographic_parity(y_hat, group):
    """
    Calculates demographic parity for a binary classification task.
    
    Parameters:
        y_hat (torch.Tensor): Predicted labels of the samples.
        group (torch.Tensor): Group membership of the samples, where 1 represents the protected group and 0 represents the non-protected group.
    
    Returns:
        float: The difference between the proportion of positive predictions in the non-protected group and the protected group.
    """
    non_protected_mask = (group == 0)
    protected_mask = (group == 1)
    
    non_protected_positives = torch.sum(torch.logical_and(y_hat == 1, non_protected_mask))
    non_protected_negatives = torch.sum(torch.logical_and(y_hat == 0, non_protected_mask))
    
    protected_positives = torch.sum(torch.logical_and(y_hat == 1, protected_mask))
    protected_negatives = torch.sum(torch.logical_and(y_hat == 0, protected_mask))
    
    non_protected_positive_rate = non_protected_positives / (non_protected_positives + non_protected_negatives)
    protected_positive_rate = protected_positives / (protected_positives + protected_negatives)
    
    return non_protected_positive_rate - protected_positive_rate


def equalized_odds(y, y_hat, a):
    """
    Calculates equalized odds for a binary classification task.
    
    Parameters:
        y (torch.Tensor): True labels of the samples
        y_hat (torch.Tensor): Predicted labels of the samples.
        group (torch.Tensor): Group membership of the samples, where 1 represents the protected group and 0 represents the non-protected group.
    
    Returns:
        float: The difference between the proportion of positive predictions in the non-protected group and the protected group.
    """
    p_0 = torch.sum(torch.floor(y[a == 0]*0.5 + y_hat[a == 0]*0.5))/torch.sum(y[a == 0])
    p_1 = torch.sum(torch.floor(y[a == 1]*0.5 + y_hat[a == 1]*0.5))/torch.sum(y[a == 1])
    eo = p_0 - p_1
    return(eo)


    