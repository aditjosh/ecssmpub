import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from training.ec_ssm_train import build_denoised_targets

class LastSignBaseline:
    """
    Persistence heuristic: forecasts that future horizons t+1...t+k 
    will follow the sign of the current step's realized return r_t.
    """
    def __init__(self, forecast_horizon_k: int):
        self.k = forecast_horizon_k

    def predict_logits(self, current_returns: torch.Tensor) -> torch.Tensor:
        """
        current_returns: (B, T)
        Returns pseudo-logits of shape (B, T, k)
        """
        # Sign at step t broadcast across all k forecast horizons
        sign_t = torch.sign(current_returns).unsqueeze(-1).expand(-1, -1, self.k)
        return sign_t

class LogisticBaseline(nn.Module):
    """
    Multi-horizon Logistic Regression operating strictly on instantaneous 
    step-t features (no recurrent state memory).
    """
    def __init__(self, input_dim: int, forecast_horizon_k: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, forecast_horizon_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim) -> logits: (B, T, k)
        return self.linear(x)

class GRUDirectionalBaseline(nn.Module):
    """
    Standard GRU sequence baseline predicting multi-horizon directional logits.
    """
    def __init__(self, input_dim: int, hidden_dim: int, forecast_horizon_k: int, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_dim, forecast_horizon_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim) -> logits: (B, T, k)
        h_seq, _ = self.gru(x)
        return self.head(h_seq)

class PlainMambaBaseline(nn.Module):
    """
    Plain Mamba / Selective State-Space baseline trained purely with BCE 
    (no AMM physics constraints, no contemporaneous head, no magnitude decomposition).
    """
    def __init__(self, input_dim: int, hidden_dim: int, forecast_horizon_k: int):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        
        # Selective SSM / Gated Recurrent State parameters
        self.A_log = nn.Parameter(torch.randn(hidden_dim))
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)
        self.C_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dir_head = nn.Linear(hidden_dim, forecast_horizon_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device
        H = self.in_proj.out_features

        u = F.silu(self.in_proj(x))
        gate = F.silu(self.gate_proj(u))
        dt = F.softplus(self.dt_proj(u))
        B_t = self.B_proj(u)
        C_t = self.C_proj(u)
        A = -torch.exp(self.A_log)  # (H,)

        h = torch.zeros(B, H, device=device)
        outputs = []

        for t in range(T):
            dA = torch.exp(dt[:, t] * A.unsqueeze(0))
            dB = dt[:, t] * B_t[:, t]
            h = dA * h + dB * u[:, t]
            y_t = (C_t[:, t] * h) * gate[:, t]
            outputs.append(y_t)

        h_seq = self.norm(torch.stack(outputs, dim=1))
        return self.dir_head(h_seq)

def extract_step_features(batch: dict, device: str) -> torch.Tensor:
    """
    Concatenates event_types, log-stabilized modify_seq, stable_swap_seq, 
    and log(current_prices) into a unified (B, T, D) input tensor.
    """
    ev = batch['event_types'].to(device).float().unsqueeze(-1)  # (B, T, 1)
    mod = batch['modify_seq'].to(device).clone()                # (B, T, 4)
    mod[:, :, 2] = torch.sign(mod[:, :, 2]) * torch.log(torch.abs(mod[:, :, 2]) + 1.0)
    swap = batch['stable_swap_seq'].to(device)                  # (B, T, 3)
    log_p = torch.log(torch.clamp(batch['current_prices'].to(device), min=1e-6)).unsqueeze(-1)
    
    return torch.cat([ev, mod, swap, log_p], dim=-1)            # (B, T, 9)

def train_neural_baseline(
    baseline_model: nn.Module,
    train_loader,
    forecast_horizon_k: int,
    epochs: int = 15,
    lr: float = 1e-3,
    device: str = 'cuda'
):
    """
    Trains Logistic, GRU, or Plain-Mamba baselines using the same 
    swap-masked and fee-thresholded BCE objective.
    """
    baseline_model.to(device)
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=1e-4)
    bce_fn = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        baseline_model.train()
        last_log_S = None
        for batch in train_loader:
            x = extract_step_features(batch, device)
            event_types = batch['event_types'].to(device)
            log_sqrt_prices = batch['log_sqrt_prices'].to(device)
            swap_volumes = batch['swap_volumes'].to(device)

            if last_log_S is not None:
                step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, forecast_horizon_k)
                last_log_S = log_sqrt_prices[-1,:]
            else:
                last_log_S = log_sqrt_prices[None, 0]
                step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, forecast_horizon_k)
                last_log_S = log_sqrt_prices[None, -1]

            valid_dir_mask_3d = valid_dir_mask_3d.to(device)
            dir_labels_3d = dir_labels_3d.to(device)

            B, T, _ = x.shape
            swap_mask = (event_types == 1).unsqueeze(-1)
            valid_horizon = torch.ones((B, T, forecast_horizon_k), dtype=torch.bool, device=device)
            for j in range(forecast_horizon_k):
                valid_horizon[:, -(j + 1):, j] = False

            mask = swap_mask & valid_horizon & valid_dir_mask_3d
            if not mask.any():
                continue

            optimizer.zero_grad()
            pred_logits = baseline_model(x)
            loss = bce_fn(pred_logits, dir_labels_3d)[mask].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
            optimizer.step()

    return baseline_model

