import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np
import pandas as pd
import sqlite3
import time
import random
from torch.utils.data import DataLoader, IterableDataset, random_split, Dataset, BatchSampler, SequentialSampler, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------
import sys
import pathlib
# Add the root directory of your project to the Python path
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ---------------

from model.regression.ec_ssm import EventConditionedSSMCell, ECVolatilityPINN
from events_dataset import EventsSequenceDataset, create_events_sequence_dataloaders, assign_train_validation_pools
from model.constants import Constants

## Track the best score and count bad epochs
class EarlyStopper:
    def __init__(self, patience=4, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_metric = float('inf')
        self.early_stop = False

    def __call__(self, val_metric):
        # Checks if validation metric improved
        if val_metric < self.min_val_metric - self.min_delta:
            self.min_val_metric = val_metric
            self.counter = 0  # Reset on improvement
        elif self.counter >= self.patience:
            self.early_stop = True
        else:
            self.counter += 1  # Increment if no improvement

def exec_db_query(db_file, query):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(query, conn)
    return df
    
def save_model_state_dict(model, model_path):
    torch.save(model.state_dict(), model_path)

def signed_log_transform(x):
    """Maps massive positive/negative EVM integers safely to a small float space."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_ensemble_member(member_id: int, hidden_dim: int = 64, forecast_horizon_k: int=10, forecast_horizon_k_step_size: int = 1,
        base_seed: int = 42, init_weights: str = 'zeros'):
    # Generate a unique, deterministic seed per ensemble member
    member_seed = base_seed + (member_id * 1000)
    seed_all(member_seed)
    
    # Instantiate model - parameters will be randomly initialized differently
    model = ECVolatilityPINN(hidden_dim=hidden_dim, forecast_horizon_k=forecast_horizon_k, forecast_horizon_k_step_size=forecast_horizon_k_step_size, 
            init_weights=init_weights)
    return model, member_seed

def configure_optimizers(model, lr=1e-3):
    # Extract dir_logits_head parameters
    dir_params = list(model.ec_cell.dir_logits_head.parameters())
    dir_param_ids = set(map(id, dir_params))
    
    # Extract all other parameters
    other_params = [p for p in model.ec_cell.parameters() if id(p) not in dir_param_ids]
    
    optimizer = torch.optim.AdamW([
        {
            'params': dir_params, 
            'lr': lr * 3.0,          # 3x learning rate boost (e.g. 1.5e-3)
            'weight_decay': 0.0      # Zero weight decay prevents weight atrophy to 0.0
        },
        {
            'params': other_params, 
            'lr': lr, 
            'weight_decay': 1e-4
        }
    ]) 
    return optimizer

def build_denoised_targets(log_S: torch.Tensor, last_log_S: float, swap_volumes: torch.Tensor,
    forecast_horizon_k: int = 10, ema_alpha: float = 0.3):
    """
    Constructs denoised multi-horizon cumulative returns and directional labels
    resistant to AMM bid-ask bounce.
    
    Parameters:
    - log_S: 1D array of raw log(sqrt(P)) values of length T.
    - last_log_S: Previous sequence's final log(sqrt(P)).
    - swap_volumes: 1D array of USD/token trade sizes at each step.
    - forecast_horizon_k: Number of future steps k.
    - fee_threshold: Minimum log-return magnitude to count as a valid directional move.
    - ema_alpha: Volume-weighted smoothing decay for micro-tick bounce suppression.
    
    Returns:
    - step_returns: (T,) 1-step log returns (for physics / auxiliary MSE).
    - cum_target_rets_3d: (T, k) Cumulative log returns R_{t -> t+j+1}.
    - dir_labels_3d: (T, k) Binary direction labels (1.0 = Up, 0.0 = Down).
    - valid_dir_mask_3d: (T, k) Boolean mask excluding bid-ask bounce noise inside +/- fee_threshold.
    """
    # Convert torch tensors to numpy if needed
    if isinstance(log_S, torch.Tensor):
        log_S = log_S.detach().cpu().numpy()
    if isinstance(last_log_S, torch.Tensor):
        last_log_S = last_log_S.detach().cpu().numpy()
    if isinstance(swap_volumes, torch.Tensor):
        swap_volumes = swap_volumes.detach().cpu().numpy()

    log_S = np.asarray(log_S, dtype=np.float64)
    swap_volumes = np.asarray(swap_volumes, dtype=np.float64)

    # Handle trailing singleton dimension e.g. (B, T, 1) -> (B, T)
    if log_S.ndim == 3 and log_S.shape[-1] == 1:
        log_S = log_S.squeeze(-1)
    if swap_volumes.ndim == 3 and swap_volumes.shape[-1] == 1:
        swap_volumes = swap_volumes.squeeze(-1)

    is_1d = (log_S.ndim == 1)
    if is_1d:
        log_S = log_S[None, :]                                  # (1, T)
        swap_volumes = swap_volumes[None, :]                    # (1, T)
    
    B, T = log_S.shape

    # Normalize last_log_S to shape (B, 1) regardless of input shape
    if last_log_S is None:
        last_log_S = log_S[:, 0:1]
    else:
        last_arr = np.asarray(last_log_S, dtype=np.float64)
        if last_arr.size == 1:
            # Scalar or (1, 1): broadcast across batch
            last_log_S = np.full((B, 1), last_arr.item(), dtype=np.float64)
        elif last_arr.shape == (B,) or last_arr.shape == (B, 1):
            # Per-batch boundary value
            last_log_S = last_arr.reshape(B, 1)
        elif last_arr.ndim == 2 and last_arr.shape[0] == B:
            # Full previous sequence (B, T_prev): take its final step
            last_log_S = last_arr[:, -1:]
        elif last_arr.size == T:
            # 1D sequence of length T (e.g. previous window): take its final element
            last_log_S = np.full((B, 1), last_arr.reshape(-1)[-1], dtype=np.float64)
        else:
            raise ValueError(
                f"Incompatible last_log_S shape {last_arr.shape} for log_S shape {log_S.shape}"
            )

    # Raw 1-step log returns along time axis (axis=1): shape (B, T)
    full_log_S = np.concatenate([last_log_S, log_S], axis=1)       # (B, T + 1)
    step_returns = 2.0 * np.diff(full_log_S, axis=1)               # (B, T)
    
    # Volume-Weighted EMA of log-price to filter single-tick bid-ask bounce
    log_P = 2.0 * log_S
    smoothed_log_P = np.zeros_like(log_P)
    smoothed_log_P[:, 0] = log_P[:, 0]
    
    # Compute mean volume per sequence strictly over active swaps (> 0)
    active_swaps = swap_volumes > 0
    active_counts = np.maximum(active_swaps.sum(axis=1, keepdims=True), 1)
    mean_vol = np.where(active_swaps, swap_volumes, 0.0).sum(axis=1, keepdims=True) / active_counts
    vol_norm = swap_volumes / (mean_vol + 1e-8)

    for t in range(1, T):
        # Zero-volume events get alpha=0 so fair value stays unchanged
        raw_alpha = np.clip(ema_alpha * vol_norm[:, t], 0.05, 0.95)
        dynamic_alpha = np.where(swap_volumes[:, t] > 0, raw_alpha, 0.0)
        smoothed_log_P[:, t] = (
            dynamic_alpha * log_P[:, t] + (1.0 - dynamic_alpha) * smoothed_log_P[:, t - 1]
        )
        
    # Build Multi-Horizon Cumulative Targets (T, k)
    cum_target_rets_3d = np.zeros((int(B), int(T), int(forecast_horizon_k)), dtype=np.float32)
    dir_labels_3d = np.zeros((int(B), int(T), int(forecast_horizon_k)), dtype=np.float32)
    valid_dir_mask_3d = np.zeros((int(B), int(T), int(forecast_horizon_k)), dtype=bool)
    
    for j in range(forecast_horizon_k):
        horizon_step = j + 1
        if T <= horizon_step:
            continue
            
        # Cumulative return from current execution log_P[t] to future smoothed fair value smoothed_log_P[t + j + 1]
        # Eliminates endpoint bid-ask bounce at t + j + 1
        # Slice along time dimension (axis 1): shape (B, T - horizon_step)
        cum_ret = smoothed_log_P[:, horizon_step:] - log_P[:, :-horizon_step]

        cum_target_rets_3d[:, :-horizon_step, j] = cum_ret
        dir_labels_3d[:, :-horizon_step, j] = (cum_ret > 0.0).astype(np.float32)
        valid_dir_mask_3d[:, :-horizon_step, j] = np.abs(cum_ret)

    if is_1d:
        step_returns = step_returns[0]
        cum_target_rets_3d = cum_target_rets_3d[0]
        dir_labels_3d = dir_labels_3d[0]
        valid_dir_mask_3d = valid_dir_mask_3d[0]
        
    return (
        torch.tensor(step_returns, dtype=torch.float32, device=log_S.device),
        torch.tensor(cum_target_rets_3d, dtype=torch.float32, device=log_S.device),
        torch.tensor(dir_labels_3d, dtype=torch.float32, device=log_S.device),
        torch.tensor(valid_dir_mask_3d, dtype=torch.bool, device=log_S.device)
    )

def train_time_stream(train_dataloader, num_epochs, forecast_horizon_k=10, forecast_horizon_k_step_size=1, model_ensemble_id=1, init_weights = 'zeros', ablation=False):
    # Model
    device = torch.device("cpu")
    early_stopper = EarlyStopper(patience=4)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # for Apple Silicon GPUs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')
    model, member_seed = build_ensemble_member(member_id=model_ensemble_id, 
            hidden_dim=Constants.HIDDEN_DIM, forecast_horizon_k=forecast_horizon_k, forecast_horizon_k_step_size=forecast_horizon_k_step_size, 
            base_seed=42, init_weights=init_weights)
    model = model.to(device)

    # Training Loop
    #optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = configure_optimizers(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    model.train()
    print(f'Training for {num_epochs} epochs...')
    for epoch in range(1, num_epochs + 1):
        current_lambda_data = Constants.LAMBDA_DATA
        last_log_S = None

        if ablation:
            ## Ablation: current_lambda_phys = 0
            current_lambda_phys = 0
        else:
            current_lambda_phys = Constants.LAMBDA_PHYS * min(1.0, epoch / 5.0)
            
        for batch_idx, batch in enumerate(train_dataloader):
            start_time = time.perf_counter()

            block_ts_utc = batch['block_ts_utc']          # Shape: B x SEQ_LEN
            event_types = batch['event_types']
            mod_liquidity_features = batch['modify_seq']
            stable_swap_features = batch['stable_swap_seq']
            raw_swap_features = batch['raw_swap_seq']
            current_prices = batch['current_prices']
            log_sqrt_prices = batch['log_sqrt_prices']
            swap_volumes = batch['swap_volumes']
            target_returns = batch['target_returns']
            target_vols = batch['target_vols'] 
            target_log_l = batch['target_log_l']

            if last_log_S is not None:
                step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, forecast_horizon_k)
                last_log_S = log_sqrt_prices[-1,:]
            else:
                last_log_S = log_sqrt_prices[None, 0]
                step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, forecast_horizon_k)
                last_log_S = log_sqrt_prices[None, -1]

            event_type_tensor = event_types.to(device)
            mod_liquidity_features = mod_liquidity_features.to(device)
            stable_swap_features = stable_swap_features.to(device)
            raw_swap_features = raw_swap_features.to(device)
            current_prices = current_prices.to(device)
            target_returns = target_returns.to(device)
            target_vols = target_vols.to(device)
            target_log_l = target_log_l.to(device)
            dir_labels_3d = dir_labels_3d.to(device)
            valid_dir_mask_3d = valid_dir_mask_3d.to(device)

            # Enable gradient tracking now that data are batched and moved
            stable_swap_features.requires_grad_(True)
            
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            joint_loss, total_data_loss, loss_directional, mean_physics_loss = model(
                event_type_tensor,
                mod_liquidity_features, 
                stable_swap_features,
                raw_swap_features,
                current_prices,
                dir_labels=dir_labels_3d,
                valid_dir_mask=valid_dir_mask_3d,
                target_returns=target_returns,
                target_vols=target_vols,
                target_log_l=target_log_l,
                lambda_data=current_lambda_data,
                lambda_phys=current_lambda_phys
            )

            # Skip anomalous update bursts entirely
            if torch.isnan(joint_loss) or torch.isinf(joint_loss):
                print("Warning: Corrupted loss matrix detected. Skipping training step optimization cycle.")
                continue

            # Backward pass optimization
            joint_loss.backward()

            # Final protective safety net against gradient explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_time = time.perf_counter() - start_time
            #print(f"Batch processed in {batch_time:.2f} seconds\n")

            if batch_idx == 0:
                print(f"--- EPOCH {epoch} BATCH 0 DIAGNOSTICS ---")
                print(f"Total data Loss   : {total_data_loss.item():.6f}")
                print(f"Physics Loss: {mean_physics_loss.item():.6f}")
                print(f"Joint Loss  : {joint_loss.item():.6f}")
                #print(f"Directional Hit Rate - Returns (%) : {directional_rets_hit_rate.item():.3f}")
                print(f"Directional Loss : {loss_directional.item():.3f}")

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Joint Loss: {joint_loss.item():.4f}")

            if early_stopper(joint_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()
        print(f"--- EPOCH {epoch} END ---")
    return model

###### Evaluation
def evaluate_price_accuracy(model, dataloader, device='cuda', evaluate_swaps_only=True):
    """
    Evaluates multi-horizon return and price prediction accuracy.
    
    Parameters:
    - model: Trained model instance.
    - dataloader: DataLoader yielding sequential evaluation batches.
    - device: 'cuda' or 'cpu'.
    - evaluate_swaps_only: bool, if True filters evaluation exclusively to active swap event steps.
    
    Returns:
    - metrics_df: pd.DataFrame summarizing performance per forecast horizon step j in [1 ... k].
    - global_summary: dict containing aggregate multi-horizon metrics.
    """
    model.eval() # Set model to evaluation mode (freezes dropout/batchnorm)
    model.to(device)
    k = model.forecast_horizon_k
    step = model.forecast_horizon_k_step_size

    # Per-horizon storage lists
    all_pred_rets = [[] for _ in range(k)]
    all_pred_dirs = [[] for _ in range(k)]
    all_target_rets = [[] for _ in range(k)]
    all_cum_target_rets = [[] for _ in range(k)]  # Cumulative target returns sum_{m=1}^{j+1} r_{t+m}
    all_pred_prices = [[] for _ in range(k)]
    all_target_prices = [[] for _ in range(k)]
    all_ofm_targets = [[] for _ in range(k)]
    all_dir_labels_3d = [[] for _ in range(k)]

    last_log_S = None
    
    # 1. Gather all predictions across the validation/test set safely
    with torch.no_grad(): # Disable autograd engine to save massive memory
        for batch in dataloader:
            print("--- DATASET PIPELINE SANITY CHECK ---")
            print("Sample Target Returns (First 5):", batch['target_returns'][0, :5].cpu().numpy())
            print("Sample Target Volatility Mean  :", batch['target_vols'].mean().item())
            print("Sample Target Volatility Max   :", batch['target_vols'].max().item())
            break

        for batch_idx, batch in enumerate(dataloader):
            event_types = batch['event_types']
            modify_seq = batch['modify_seq']
            stable_swap_seq = batch['stable_swap_seq']
            raw_swap_seq = batch['raw_swap_seq']
            current_prices = batch['current_prices']
            log_sqrt_prices = batch['log_sqrt_prices']
            swap_volumes = batch['swap_volumes']
            target_returns = batch['target_returns']
            target_vols = batch['target_vols']
            target_log_l = batch['target_log_l']

            if last_log_S is not None:
                step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, k)
                last_log_S = log_sqrt_prices[-1,:]
            else:
                last_log_S = log_sqrt_prices[None, 0]
                step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, k)
                last_log_S = log_sqrt_prices[None, -1]

            event_type = event_types.to(device, non_blocking=True)
            modify_seq = modify_seq.to(device, non_blocking=True)
            stable_swap_seq = stable_swap_seq.to(device, non_blocking=True)
            current_prices = current_prices.to(device, non_blocking=True)
            target_returns = target_returns.to(device, non_blocking=True)
            dir_labels_3d = dir_labels_3d.to(device)
            #valid_dir_mask_3d = valid_dir_mask_3d.to(device)

            # Keep a clean copy of the raw inputs for the physics evaluation
            raw_swap_features = raw_swap_seq
            
            # Get model log predictions
            # Forward pass -> pred_rets shape: (B, T, k)
            pred_rets, pred_dir_logits, pred_vols, pred_log_liqs = model(
                event_type,
                modify_seq, 
                stable_swap_seq,
                raw_swap_seq,
                current_prices
            )

            B, T, _ = pred_rets.shape

            # Build 3D multi-horizon targets: target_3d[b, t, j] = target_returns[b, t + j + 1]
            target_rets_3d = model._build_multi_horizon_targets(target_returns)

            # Compute realized 1-step order flow direction (+1 for buy pressure, -1 for sell pressure)
            raw_dx = raw_swap_seq[:, :, 0]  # Token 0 in (sell)
            raw_dy = raw_swap_seq[:, :, 1]  # Token 1 in (buy)
            order_flow_direction = torch.sign(raw_dy - raw_dx)  # Shape: (B, T)
            order_flow_direction = order_flow_direction.to(device, non_blocking=True)

            # Build multi-horizon order flow targets (t+1 ... t+k)
            target_ofm_3d = model._build_multi_horizon_targets(order_flow_direction)

            # Event masking
            if evaluate_swaps_only:
                step_mask = (event_type == 1) # Shape: (B, T)
            else:
                step_mask = torch.ones_like(event_type, dtype=torch.bool)

            # Process each forecast horizon step j in [0 ... k-1] (representing t + j + 1)
            for j in range(0, k, step):
                # Mask out boundary steps where sequence lookahead exceeds T
                valid_boundary = torch.ones((B, T), dtype=torch.bool, device=device)
                valid_boundary[:, -(j + 1):] = False

                # Compute cumulative target return over subsequent (j + 1) steps: sum_{m=0}^{j} target_rets_3d[:, :, m]
                cum_target_rets_j = torch.sum(target_rets_3d[:, :, :j+1], dim=-1)  # Shape: (B, T)

                # Finite Number Validation Mask (Eliminates NaNs and Inf at input level)
                valid_targets = ~torch.isnan(target_rets_3d[:, :, :j]).any(dim=-1) & \
                                ~torch.isinf(target_rets_3d[:, :, :j]).any(dim=-1)
                valid_preds = ~torch.isnan(pred_rets[:, :, j]) & ~torch.isinf(pred_rets[:, :, j])
                valid_prices = ~torch.isnan(current_prices) & ~torch.isinf(current_prices) & (current_prices > 0)

                combined_mask = step_mask & valid_boundary & valid_targets & valid_prices
                
                if not combined_mask.any():
                    continue
                    
                #combined_mask = combined_mask_device.to('cpu')
                p_ret_j = pred_rets[:, :, j][combined_mask].cpu().numpy()
                logits_j = pred_dir_logits[:, :, j][combined_mask].cpu().numpy()

                t_ret_j = target_rets_3d[:, :, j][combined_mask].cpu().numpy()
                t_cum_ret_j = cum_target_rets_j[combined_mask].cpu().numpy()
                ofm_targets_j = target_ofm_3d[:, :, j][combined_mask].cpu().numpy()
                
                all_pred_rets[j].extend(p_ret_j)
                all_pred_dirs[j].extend(logits_j)
                all_target_rets[j].extend(t_ret_j)
                all_cum_target_rets[j].extend(t_cum_ret_j)
                all_ofm_targets[j].extend(ofm_targets_j)

                # --- Cumulative Price Trajectory Calculation ---
                # Cumulative predicted return from step t+1 up to step t+j+1
                cum_pred_ret_j = torch.sum(pred_rets[:, :, :j+1], dim=-1)       # Shape: (B, T)

                # CLAMP cumulative log-returns to [-5.0, 5.0] to strictly prevent exp() overflow (Inf)
                cum_pred_ret_clamped = torch.clamp(cum_pred_ret_j[combined_mask], min=-5.0, max=5.0)
                cum_target_ret_clamped = torch.clamp(cum_target_rets_j[combined_mask], min=-5.0, max=5.0)

                # Reconstruct predicted price vs actual target price
                p_curr = current_prices[combined_mask]
                pred_price_j = (p_curr * torch.exp(cum_pred_ret_clamped)).cpu().numpy()
                target_price_j = (p_curr * torch.exp(cum_target_ret_clamped)).cpu().numpy()
                
                all_pred_prices[j].extend(pred_price_j)
                all_target_prices[j].extend(target_price_j)

                dir_labels_3d_j = dir_labels_3d[:, :, j][combined_mask].cpu().numpy()
                all_dir_labels_3d[j].extend(dir_labels_3d_j)

    # Compute Horizon-Specific Metrics
    horizon_metrics = []

    for j in range(0, k, step):
        pred_rets = np.array(all_pred_rets[j])
        pred_rets_dir = np.array(all_pred_dirs[j])
        target_rets = np.array(all_target_rets[j])
        cum_targets = np.array(all_cum_target_rets[j])
        p_prices = np.array(all_pred_prices[j])
        t_prices = np.array(all_target_prices[j])
        ofm_targets = np.array(all_ofm_targets[j])
        dir_labels = np.array(all_dir_labels_3d[j])

        valid_idx = np.isfinite(pred_rets) & np.isfinite(target_rets) & np.isfinite(cum_targets) & \
                    np.isfinite(p_prices) & np.isfinite(t_prices)

        pred_rets = pred_rets[valid_idx]
        pred_rets_dir = pred_rets_dir[valid_idx]
        target_rets = target_rets[valid_idx]
        cum_targets = cum_targets[valid_idx]
        p_prices = p_prices[valid_idx]
        t_prices = t_prices[valid_idx]
        
        if len(pred_rets) == 0:
            continue
            
        # Return MAE & RMSE
        mae = np.mean(np.abs(pred_rets - target_rets))
        rmse = np.sqrt(np.mean((pred_rets - target_rets) ** 2))
        
        # Directional Hit Rate (Strict Sign Comparison on Step Returns)
        # Exclude exact 0.0 targets if any exist
        non_zero_targets = dir_labels != 0.0
        if np.any(non_zero_targets):
            pred_direction = (np.sign(pred_rets_dir) > 0.0)
            pred_direction = np.array(pred_direction, dtype=float)
            target_direction = dir_labels
            hit_rate = np.mean(pred_direction == target_direction) * 100.0
        else:
            hit_rate = 50.0
        
        # Single-step Information Coefficient (Pearson Correlation) between prediction made at time 't' pred(t) 
        # and the corresponding returns at t+j+1 target(t+j+1)
        # all_pred_rets[j] holds every prediction made at time t across the dataset for horizon j, 
        # and all_target_rets[j] holds the corresponding realized returns at t + j + 1.
        if np.std(pred_rets) > 1e-9 and np.std(target_rets) > 1e-9:
            ic, _ = stats.pearsonr(pred_rets, target_rets)
        else:
            ic = 0.0

        # Cumulative Return IC: Corr(pred_r_{t+j+1|t}, sum_{m=1}^{j+1} target_r_{t+m})
        if np.std(pred_rets) > 1e-9 and np.std(cum_targets) > 1e-9:
            cum_ic, _ = stats.pearsonr(pred_rets, cum_targets)
        else:
            cum_ic = 0.0

        # Order Flow Momentum IC: Correlation between predicted return and subsequent order flow sign
        if np.std(pred_rets) > 1e-9 and np.std(ofm_targets) > 1e-9:
            ic_ofm, _ = stats.pearsonr(pred_rets, ofm_targets)
            pred_direction = np.sign(pred_rets_dir)
            target_direction = ofm_targets
            # Order Flow Directional Hit Rate (%)
            ofm_hit_rate = np.mean(pred_direction == target_direction) * 100.0
        else:
            ic_ofm = 0.0
            ofm_hit_rate = 50.0
            
        epsilon = 1e-8
        # Price MAPE (%)
        price_mape = np.mean(np.abs((p_prices - t_prices) / t_prices + epsilon)) * 100.0

        # Rets MAPE
        mape_rets = np.mean(np.abs((pred_rets - target_rets) / (target_rets + epsilon))) * 100
        
        horizon_metrics.append({
            'Horizon (Step)': j + 1,
            'Return MAE': mae,
            'Return RMSE': rmse,
            'Directional Hit Rate (%)': hit_rate,
            'Step IC': ic,
            'OFM IC': ic_ofm,
            'OFM Hit Rate (%)': ofm_hit_rate,
            'Cumulative Return IC': cum_ic,
            'Price MAPE (%)': price_mape,
            'Returns MAPE (%)': mape_rets,
            'Num Swap Events': len(target_rets)
        })

    metrics_df = pd.DataFrame(horizon_metrics)

    if metrics_df.empty:
        print("ERROR!! No metrics calculated.")
        return None, None
    
    # Global Summary Metrics (Averaged across all k horizons)
    global_summary = {
        'Mean_Step_IC': metrics_df['Step IC'].mean() if not metrics_df.empty else 0.0,
        'Mean_OFM_IC': metrics_df['OFM IC'].mean() if not metrics_df.empty else 0.0,
        'Mean_Cumulative_Return_IC': metrics_df['Cumulative Return IC'].mean() if not metrics_df.empty else 0.0,
        'Mean_Directional_Hit_Rate': metrics_df['Directional Hit Rate (%)'].mean(),
        'Mean_OFM_Hit_Rate': metrics_df['OFM Hit Rate (%)'].mean(),
        'Mean_Price_MAPE': metrics_df['Price MAPE (%)'].mean(),
        'Mean_Returns_MAPE': metrics_df['Returns MAPE (%)'].mean(),
        'Forecast_Horizon_K': k,
        'Total Swap Events Evaluated': metrics_df['Num Swap Events'].sum()
    }
    
    return metrics_df, global_summary

def eval_model(model, val_dataloader):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    metrics_df, summary = evaluate_price_accuracy(
        model=model, 
        dataloader=val_dataloader, 
        device=device,
        evaluate_swaps_only=True
    )
    
    print("\n================ Forecast Performance by Horizon ================")
    print(metrics_df.to_string(index=False))
    print("\n================ Global Model Summary ================")
    for metric_name, val in summary.items():
        print(f"{metric_name}: {val:.4f}")

### Data trend
def test_data_trend(dataloader):
    all_directions = []
    
    for batch_idx, batch in enumerate(dataloader):
        event_types = batch['event_types']
        current_prices = batch['current_prices']
        target_returns = batch['target_returns']
        target_vols = batch['target_vols']
        target_log_l = batch['target_log_l']

        # Mask evaluations to only care about active swap events
        mask = (event_types == 1.0)
            
        target_signs = np.sign(np.asarray(target_returns[mask]))
        target_signs = [v.item() for v in target_signs]
        all_directions.extend(target_signs)
    
    total_count = len(all_directions)
    proportions = {}

    mask_minus_ones = list(filter(lambda x: x == -1.0, all_directions))
    if isinstance(mask_minus_ones, list):
        proportions['-1'] = np.abs(np.sum(mask_minus_ones)) / total_count
    mask_zeros = list(filter(lambda x: x == 0.0, all_directions))
    if isinstance(mask_zeros, list) > 0:
        proportions['0'] = np.sum(mask_zeros) / total_count
    mask_ones = list(filter(lambda x: x == 1.0, all_directions))
    if isinstance(mask_ones, list)  > 0:
        proportions['1'] = np.sum(mask_ones) / total_count

    print(f"Total Swap Events: {len(all_directions)}")
    print(f"Target Returns Signs Proportions: {proportions}")

def analyze_direction_flip_frequency(dataloader, forecast_horizon_k):
    """
    Computes the sign-change (flip) frequency, persistence rate, and class balance
    of `dir_labels_3d` along the time axis for each forecast horizon k.

    Returns:
    - DataFrame summarizing Flip Rate (%), Persistence (%), and Up-Label Ratio (%) per horizon.
    """
    k = forecast_horizon_k

    # Per-horizon storage lists
    all_dir_labels_3d = []
    all_valid_dir_mask_3d = []

    last_log_S = None
    
    # Gather all predictions across the validation/test set safely
    for batch_idx, batch in enumerate(dataloader):
        event_types = batch['event_types']
        modify_seq = batch['modify_seq']
        swap_volumes = batch['swap_volumes']
        current_prices = batch['current_prices']
        log_sqrt_prices = batch['log_sqrt_prices']
        target_returns = batch['target_returns']
        target_vols = batch['target_vols']
        target_log_l = batch['target_log_l']

        if last_log_S is not None:
            step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, k)
            last_log_S = log_sqrt_prices[-1,:]
        else:
            last_log_S = log_sqrt_prices[None, 0]
            step_returns, cum_target_rets_3d, dir_labels_3d, valid_dir_mask_3d = build_denoised_targets(log_sqrt_prices, last_log_S, swap_volumes, k)
            last_log_S = log_sqrt_prices[None, -1]

        all_dir_labels_3d.append(dir_labels_3d)
        all_valid_dir_mask_3d.append(valid_dir_mask_3d)

    # Concat to single 3d arrays
    dir_labels_3d = torch.concatenate(all_dir_labels_3d, axis=0)
    valid_dir_mask_3d = torch.concatenate(all_valid_dir_mask_3d, axis=0)

    if isinstance(dir_labels_3d, torch.Tensor):
        labels = dir_labels_3d.detach().cpu().numpy()
    else:
        labels = np.asarray(dir_labels_3d)

    if labels.ndim == 2:
        labels = labels[None, :, :]  # (1, T, k)

    B, T, k = labels.shape

    if valid_dir_mask_3d is not None:
        if isinstance(valid_dir_mask_3d, torch.Tensor):
            mask = valid_dir_mask_3d.detach().cpu().numpy().astype(bool)
        else:
            mask = np.asarray(valid_dir_mask_3d, dtype=bool)
        if mask.ndim == 2:
            mask = mask[None, :, :]
    else:
        # Default: exclude trailing boundary zeros at the end of each sequence
        mask = np.ones((B, T, k), dtype=bool)
        for j in range(k):
            mask[:, -(j + 1):, j] = False

    # Consecutive transitions along time axis (axis=1): shape (B, T - 1, k)
    flips = (labels[:, 1:, :] != labels[:, :-1, :])
    valid_transitions = mask[:, 1:, :] & mask[:, :-1, :]

    summary = []
    for j in range(k):
        valid_t = valid_transitions[:, :, j]
        n_transitions = int(valid_t.sum())

        if n_transitions > 0:
            flip_rate = float(flips[:, :, j][valid_t].mean() * 100.0)
            # Average run length (number of consecutive steps before a sign change)
            avg_run_length = 100.0 / max(flip_rate, 1e-6)
        else:
            flip_rate = 0.0
            avg_run_length = float('nan')

        valid_single = mask[:, :, j]
        up_ratio = float(labels[:, :, j][valid_single].mean() * 100.0) if valid_single.any() else 50.0

        summary.append({
            "Horizon (k)": j + 1,
            "Sign Flip Rate (%)": round(flip_rate, 2),
            "Persistence (%)": round(100.0 - flip_rate, 2),
            "Avg Streak (steps)": round(avg_run_length, 2),
            "Up-Label Ratio (%)": round(up_ratio, 2),
            "Valid Transitions": n_transitions
        })

    summary_df = pd.DataFrame(summary)

    # Global Summary Metrics (Averaged across all k horizons)
    global_summary = {
        'Mean Sign Flip Rate (%)': f"{summary_df['Sign Flip Rate (%)'].mean().item():.6f}" if not summary_df.empty else 0.0,
        'Mean Persistence (%)': f"{summary_df['Persistence (%)'].mean().item():.6f}" if not summary_df.empty else 0.0,
        'Mean Avg Streak (steps)': f"{summary_df['Avg Streak (steps)'].mean().item():.6f}",
        'Mean Up-Label Ratio (%)': f"{summary_df['Up-Label Ratio (%)'].mean().item():.6f}",
        'Total Valid Transitions': f"{summary_df['Valid Transitions'].sum().item():.6f}"
    }
    print(global_summary)

    return summary_df


# --- Usage ---
if __name__ == "__main__":
    MODEL_NUMBER = 1
    MODEL_TAG = ''
    NUM_EPOCHS = 20
    SEQ_LEN = 500
    FORECAST_HORIZON = 60
    FORECAST_HORIZON_STEP_SIZE = 10
    init_weights = 'zeros'     # 'rand','zeros' 
    is_ablation = True
    TOKENS_TAG = 'eth_usdt'
    DEC0 = Constants.WETH_DECIMAL_SCALE
    DEC1 = Constants.USDT_DECIMAL_SCALE
    db_file = f"../analysis/sqldb/{TOKENS_TAG}/{TOKENS_TAG}_univ4_timeline.db"

    train_pools, validation_pools = assign_train_validation_pools(TOKENS_TAG)
    train_dataloader, val_dataloader = create_events_sequence_dataloaders(
        db_file, table_name="timeline", seq_len=SEQ_LEN, vol_window_len=30, train_pools=train_pools, 
        validation_pools=validation_pools, fetch_limit=100000, test_prop=0.2, dec0=DEC0, dec1=DEC1)
    
    # Train
    model = train_time_stream(train_dataloader, num_epochs=NUM_EPOCHS, model_ensemble_id=MODEL_NUMBER, 
        forecast_horizon_k=FORECAST_HORIZON, forecast_horizon_k_step_size=FORECAST_HORIZON_STEP_SIZE, 
        init_weights=init_weights, ablation=is_ablation)
    ## Save the model's state dictionary
    #torch.save(model.state_dict(), f"trained/nov26/{TOKENS_TAG}/model_{MODEL_NUMBER}{MODEL_TAG}_128_ws{SEQ_LEN}_k{FORECAST_HORIZON}_weights.pth")

    # Evaluate
    eval_model(model, val_dataloader)

    # Data trends
    test_data_trend(val_dataloader)

    # How frequently dir_labels_3d flips direction (between 1.0 and 0.0) along the time dimension
    print('====== Training dataloader ======')
    analyze_direction_flip_frequency(train_dataloader, FORECAST_HORIZON)
    print('====== Validation dataloader ======')
    analyze_direction_flip_frequency(val_dataloader, FORECAST_HORIZON)

