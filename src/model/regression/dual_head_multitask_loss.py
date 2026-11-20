import torch
import torch.nn as nn
import torch.nn.functional as F
from model.constants import Constants
import sys

class DualHeadMultiTaskLoss(nn.Module):
    """
    Loss function optimizing Direction Classification (BCEWithLogits)
    and Return Magnitude Regression (Smooth L1) independently.
    """
    def __init__(self, noise_margin: float = 1e-4):
        super().__init__()
        self.noise_margin = noise_margin
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.magnitude_loss = nn.SmoothL1Loss(reduction='none')
        self.mse = nn.MSELoss()

    def forward(
        self, 
        pred_dir_logits, pred_mags, pred_rets, pred_vols, pred_log_liqs, mask,
        dir_labels, target_rets_multi, target_vols_multi, target_log_l_multi,
        mean_physics_loss, lambda_data, lambda_phys, discount_weights=None
    ):
        # Noise Margin Filter: Only calculate BCE on steps where return magnitude > noise_margin
        significant_mask = mask & (torch.abs(target_rets_multi) > self.noise_margin)

        target_magnitudes = torch.abs(target_rets_multi)
        pred_magnitudes = torch.abs(pred_rets)         # B x SEQ_LEN x k
        
        # Binary Cross-Entropy Loss on Direction Logits
        bce_raw = self.bce_loss(pred_dir_logits, dir_labels)

        # Smooth L1 Loss on Return Magnitude
        mag_raw = self.magnitude_loss(pred_mags, target_magnitudes)
        ##B, H, step_k = pred_magnitudes.shape
        #mag_raw = self.magnitude_loss(pred_magnitudes[significant_mask], target_magnitudes[significant_mask])

        # Apply discount weights
        #if discount_weights is not None:
        #    bce_raw = bce_raw * discount_weights
        #    mag_raw = mag_raw * discount_weights[:,:,::step_k]
        #    pred_vols = pred_vols * discount_weights[:,:,::step_k]
        #    target_vols_3d = target_vols_3d * discount_weights
        #    pred_log_liqs = pred_log_liqs * discount_weights[:,:,::step_k]
        #    target_log_l_3d = target_log_l_3d * discount_weights
        
        loss_dir_bce = torch.mean(bce_raw[significant_mask])
        loss_mag = torch.mean(mag_raw[significant_mask])
        loss_v_mse = self.mse(pred_vols[significant_mask], target_vols_multi[significant_mask])
        loss_l_mse = self.mse(pred_log_liqs[significant_mask], target_log_l_multi[significant_mask])

        # Total Return Data Loss
        #total_loss_r = Constants.LAMBDA1 * loss_mag + Constants.LAMBDA2 * loss_dir_bce

        # Combined loss with decoupled physics term
        total_loss_r =  Constants.LAMBDA1 * loss_mag + Constants.LAMBDA2 * loss_dir_bce + Constants.LAMBDA4 * loss_v_mse + Constants.LAMBDA5 * loss_l_mse
        joint_loss = lambda_data * total_loss_r + (lambda_phys * mean_physics_loss)

        return joint_loss, total_loss_r, loss_dir_bce, loss_mag