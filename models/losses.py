import torch 
import torch.nn as nn 
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.

        Parameters:
        - alpha: Weighting factor for the positive class (storm), can be set to [0,1] depending on class imbalance.
        - gamma: Focusing parameter to adjust the weighting of hard examples.
        - reduction: Specifies the reduction to apply to the output ('none', 'mean', or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for the storm class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for focal loss.
        
        Parameters:
        - inputs: Predicted probabilities for the positive class (storm) (after sigmoid for binary classification).
        - targets: Ground truth binary labels (0 for non-storm, 1 for storm).
        
        Returns:
        - Calculated focal loss.
        """
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Get p_t
        pt = torch.where(targets == 1, inputs, 1 - inputs)  # p_t is the predicted prob for the correct class

        # Apply focal loss components
        focal_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha) * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss