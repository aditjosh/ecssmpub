import torch
import torch.nn as nn

class DirectionalBCELoss(nn.Module):
    def __init__(self, alpha=100.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_rets_masked, target_rets_masked):
        # Convert true target return into soft 0 or 1 labels
        target_labels = torch.sigmoid(self.alpha * target_rets_masked)
        
        # Evaluate Binary Cross-Entropy against model predictions
        loss = self.bce(pred_rets_masked * self.alpha, target_labels)
        return loss