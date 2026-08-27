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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR

# ---------------
import sys
import pathlib
# Add the root directory of your project to the Python path
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
# ---------------

from model.regression.ec_vol_ssm import EventConditionedVolatilitySSMCell, ECVolatilityPINN
from time_sliding_window_vol_dataset import create_sliding_window_sequence_dataloaders, create_sliding_window_sequence_dataloaders_for_ensemble

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

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_ensemble_member(member_id: int, hidden_dim: int = 64, base_seed: int = 42):
    # Generate a unique, deterministic seed per ensemble member
    member_seed = base_seed + (member_id * 1000)
    seed_everything(member_seed)
    
    # Instantiate model - parameters will be randomly initialized differently
    model = ECVolatilityPINN(hidden_dim=hidden_dim)
    return model, member_seed

def train_time_stream(train_dataloader, num_epochs, model_ensemble_id=1):
    # Model
    device = torch.device("cpu")
    HIDDEN_DIM = 128
    LAMBDA_PHYS = 0.5
    LAMBDA_DATA = 1.0
    early_stopper = EarlyStopper(patience=4)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # for Apple Silicon GPUs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')
    model, member_seed = build_ensemble_member(member_id=model_ensemble_id, hidden_dim=HIDDEN_DIM, base_seed=42)
    model = model.to(device)

    # Training Loop
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    model.train()
    print(f'Training for {num_epochs} epochs...')
    for epoch in range(num_epochs + 1):
        current_lambda_data = LAMBDA_DATA
        current_lambda_phys = LAMBDA_PHYS * min(1.0, epoch / 5.0)
        for batch_idx, batch in enumerate(train_dataloader):
            start_time = time.perf_counter()

            block_ts_utc = batch['block_ts_utc']
            event_types = batch['event_types']
            mod_liquidity_features = batch['modify_seq']
            stable_swap_features = batch['stable_swap_seq']
            raw_swap_features = batch['raw_swap_seq']
            current_prices = batch['current_prices']
            target_returns = batch['target_returns']
            target_vols = batch['target_vols'] 
            target_log_l = batch['target_log_l']

            event_type_tensor = event_types.to(device, non_blocking=True)
            mod_liquidity_features = mod_liquidity_features.to(device, non_blocking=True)
            stable_swap_features = stable_swap_features.to(device, non_blocking=True)
            raw_swap_features = raw_swap_features.to(device, non_blocking=True)
            current_prices = current_prices.to(device, non_blocking=True)
            target_returns = target_returns.to(device, non_blocking=True)
            target_vols = target_vols.to(device, non_blocking=True)
            target_log_l = target_log_l.to(device, non_blocking=True)

            # Enable gradient tracking now that data are batched and moved
            stable_swap_features.requires_grad_(True)

            # Forward pass
            joint_loss, total_data_loss, mean_physics_loss = model(
                event_type_tensor,
                mod_liquidity_features, 
                stable_swap_features,
                raw_swap_features,
                current_prices,
                target_returns,
                target_vols,
                target_log_l,
                current_lambda_data,
                current_lambda_phys
            )

            #if epoch < 2:
            #    lambda_phys_current = 0.0  # Let the model learn raw prices first
            #elif epoch < 4:
            #    lambda_phys_current = lambda_phys * 0.1  # Introduce physics softly
            #else:
            #    lambda_phys_current = lambda_phys  # Apply full physics constraints

            #lambda_phys_current = LAMBDA_PHYS

            # Skip anomalous update bursts entirely
            if torch.isnan(joint_loss) or torch.isinf(joint_loss):
                print("Warning: Corrupted loss matrix detected. Skipping training step optimization cycle.")
                continue

            # Backward pass optimization
            optimizer.zero_grad(set_to_none=True)
            joint_loss.backward()

            vol_weight_grad = model.mamba_cell.predict_log_volatility[-1].weight.grad
            vol_bias_grad = model.mamba_cell.predict_log_volatility[-1].bias.grad

            #print(f"Volatility Head Weight Grad Norm: {vol_weight_grad.norm().item():.8f}")
            #print(f"Volatility Head Bias Grad Norm:   {vol_bias_grad.norm().item():.8f}")

            # Final protective safety net against gradient explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_time = time.perf_counter() - start_time
            #print(f"Batch processed in {batch_time:.2f} seconds\n")

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {joint_loss.item():.4f}")

            if batch_idx == 0:
                print(f"--- EPOCH {epoch} BATCH 0 DIAGNOSTICS ---")
                print(f"Total data Loss   : {total_data_loss.item():.6f}")
                print(f"Physics Loss: {mean_physics_loss.item():.6f}")
                print(f"Joint Loss  : {joint_loss.item():.6f}")

        if early_stopper(joint_loss):
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()
    return model

###### Evaluation
def evaluate_price_accuracy(model, dataloader, device, decimal0=6, decimal1=18):
    """
    Args:
        decimal0: Decimals of Token 0 (e.g., USDC = 6)
        decimal1: Decimals of Token 1 (e.g., WETH = 18)
    """
    model.eval() # Set model to evaluation mode (freezes dropout/batchnorm)
    
    all_raw_preds = []
    all_raw_targets = []
    all_log_preds = []
    all_log_targets = []
    all_correct_directions = []
    all_pred_rets = []
    all_target_rets = []
    
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
            mod_liquidity_features = batch['modify_seq']
            stable_swap_features = batch['stable_swap_seq']
            raw_swap_features = batch['raw_swap_seq']
            current_prices = batch['current_prices']
            target_returns = batch['target_returns']
            target_vols = batch['target_vols']
            target_log_l = batch['target_log_l']

            event_type_tensor = event_types.to(device, non_blocking=True)
            mod_liquidity_features = mod_liquidity_features.to(device, non_blocking=True)
            stable_swap_features = stable_swap_features.to(device, non_blocking=True)
            current_prices = current_prices.to(device, non_blocking=True)
            #target_returns = target_returns.to(device, non_blocking=True)
            #target_vols = target_vols.to(device, non_blocking=True)
            #target_log_l = target_log_l.to(device, non_blocking=True)

            # Keep a clean copy of the raw inputs for the physics evaluation
            raw_swap_features = raw_swap_features
            
            # Get model log predictions
            pred_rets, pred_vols, pred_log_liqs = model(
                event_type_tensor,
                mod_liquidity_features, 
                stable_swap_features,
                raw_swap_features,
                current_prices
            )

            pred_vols = np.asarray(pred_vols.cpu(), dtype=np.float64)
            target_vols = np.asarray(target_vols.cpu(), dtype=np.float64)

            #pred_log_liqs = np.asarray(pred_log_liqs.cpu(), dtype=np.float64)

            #print(f"Shape:              {target_log_prices_ground_truth.shape}")
            #print(f"Total Elements:     {target_log_prices_ground_truth.size}")
            #print(f"Minimum Value:      {target_log_prices_ground_truth.min()}")
            #print(f"Maximum Value:      {target_log_prices_ground_truth.max()}")
            #print(f"Arithmetic Mean:    {target_log_prices_ground_truth.mean()}")
            #print(f"Median Value:       {np.median(target_log_prices_ground_truth)}")
            #print(f"Variance:           {target_log_prices_ground_truth.var():.2f}")
            #print(f"Standard Deviation: {target_log_prices_ground_truth.std():.2f}")
            #print(f"25th Percentile:    {np.percentile(target_log_prices_ground_truth, 25)}")
            #print(f"75th Percentile:    {np.percentile(target_log_prices_ground_truth, 75)}")
            
            # Mask evaluations to only care about active swap events
            mask = (event_type_tensor == 1.0)
            mask = mask.cpu().numpy()
            
            if mask.sum() > 0:
                # Hard-limit the log predictions before exponentiating 
                # This prevents exp(pred) from hitting 'inf'
                #pred_log_capped = np.clip(pred_log_prices, -50.0, 50.0)
                #target_log_capped = target_log_prices_ground_truth

                # Exponential back-transformation out of Log-Space
                #pred_raw_prices = np.exp(pred_log_prices)
                #target_raw_prices_ground_truth = np.exp(target_log_capped)
                
                # Extract only the valid swap steps and move to CPU
                all_raw_preds.extend(pred_vols[mask])
                all_raw_targets.extend(target_vols[mask])

                all_pred_rets.extend(pred_rets[mask].detach().cpu())
                all_target_rets.extend(target_returns[mask].detach().cpu())

                predicted_signs = torch.sign(pred_rets[mask]).to(device, non_blocking=True)
                target_signs = torch.sign(target_returns[mask]).to(device, non_blocking=True)

                correct_directions = (predicted_signs == target_signs).float().mean().item()
                #print(f"Current Epoch Directional Accuracy: {correct_directions * 100:.2f}%")
                all_correct_directions.append(correct_directions)

    pred_vols_raw = np.array(all_raw_preds)
    target_vols_raw = np.array(all_raw_targets)

    all_pred_rets = np.array(all_pred_rets)
    all_target_rets = np.array(all_target_rets)

    # 2. Convert raw EVM sqrtPriceX96 vectors into human-readable pool prices
    # (Price of Token 0 denominated in terms of Token 1)
    #decimal_adjustment = 10 ** (decimal0 - decimal1)

    # Force the arrays to be completely flat 1D vectors
    #preds_human = ((pred_prices_raw) * decimal_adjustment).flatten()
    #targets_human = ((target_prices_raw) * decimal_adjustment).flatten()

    print("target_vols_raw contains NaN:", np.isnan(target_vols_raw).any())
    print("pred_vols_raw contains Inf:", np.isinf(pred_vols_raw).any())

    # Clean predictions and true labels simultaneously
    valid_mask = np.isfinite(target_vols_raw) & np.isfinite(pred_vols_raw)

    target_vols = target_vols_raw[valid_mask]
    pred_vols = pred_vols_raw[valid_mask]

    # Calculate Core Financial Accuracy Metrics
    # Mean Absolute Error in asset pricing unit (e.g., true price off by $4.50)
    mae_usd = mean_absolute_error(target_vols, pred_vols)
    mae_rets = mean_absolute_error(all_target_rets[valid_mask], all_pred_rets[valid_mask])
    
    # Root Mean Squared Error (Punishes larger outliers heavier)
    diff = target_vols - pred_vols
    #diff_capped = np.clip(diff, -1e18, 1e18) # 1e18 squared is 1e36 (safely within float32/64 limits)
    #rmse_usd = np.sqrt(np.mean(diff_capped ** 2))
    rmse_usd = np.sqrt(np.mean(diff ** 2))
    
    # Mean Absolute Percentage Error (Most informative metric, e.g., 0.02 = 2% average error)
    # Calculate MAPE safely (avoid division by zero or division by tiny targets)
    # Ensure targets are not zero (add a tiny epsilon 1e-8 to the denominator)
    epsilon = 1e-8
    mape = np.mean(np.abs(target_vols - pred_vols) / (target_vols + epsilon)) * 100
    mape_rets = np.mean(np.abs(all_target_rets[valid_mask] - all_pred_rets[valid_mask]) / (all_target_rets[valid_mask] + epsilon)) * 100

    print("="*40)
    print("      ACCURACY REPORT      ")
    print("="*40)
    print(f"Total Swap Events Evaluated: {len(target_vols)}")
    print(f"Mean Absolute Error (MAE):   {mae_usd:.4f}")
    print(f"Mean absolute error (returns): {mae_rets:.4f}")
    print(f"Root Mean Squared Error:     {rmse_usd:.4f}")
    print(f"MAPE: {mape:.3f}%")
    print(f"MAPE (returns): {mape_rets:.3f}%")
    print(f"Directional accuracy: {np.mean(all_correct_directions)}")
    print("="*40)

    print(f"Mean of target_vols: {np.mean(target_vols)} Variance: {np.var(target_vols)} Median: {np.median(target_vols)}")
    print(f"Mean of pred_vols: {np.mean(pred_vols)} Variance: {np.var(pred_vols)} Median: {np.median(pred_vols)}")
    
    return target_vols, pred_vols

def eval_model(model, val_dataloader):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    target_vols, pred_vols = evaluate_price_accuracy(model, val_dataloader, device)
    print("\nSample Track (Actual vs Predicted for validation set):")

    rng = np.random.default_rng()
    # Generate random integers from 0 to 100 (101 is excluded)
    integers = rng.integers(low=0, high=len(target_vols), size=10)
    for i in integers:
        diff = pred_vols[i] - target_vols[i]
        pct = (diff / target_vols[i]) * 100
        print(f"True: {target_vols[i]:.4f} | Pred: {pred_vols[i]} | Delta: {diff:.6f} ({pct:.4f}%)")

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
            
        target_signs = np.sign(target_returns[mask])
        target_signs = [v.item() for v in target_signs.detach().cpu().numpy()]
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


# --- Example Usage ---
if __name__ == "__main__":
    db_file = "../analysis/sqldb/usdc_eth/usdc_eth_univ4_timeline.db"
    #train_dataloader, val_dataloader = create_sliding_window_sequence_dataloaders(
    #    db_file, table_name="timeline", window_size=100, vol_window=30, train_chunk_size=5000, val_chunk_size=5000)

    MODEL_NUMBER = 5
    WINDOW_SIZE = 100
    train_dataloader, val_dataloader = create_sliding_window_sequence_dataloaders_for_ensemble(
        db_file, table_name="timeline", window_size=WINDOW_SIZE, vol_window=30, train_chunk_size=6000, 
        val_chunk_size=5000, member_seed=MODEL_NUMBER * 100)
    
    # Train
    model = train_time_stream(train_dataloader, num_epochs=2, model_ensemble_id=MODEL_NUMBER)
    ## Save the model's state dictionary
    torch.save(model.state_dict(), f"trained/model_{MODEL_NUMBER}_128_ws{WINDOW_SIZE}_weights.pth")

    # Evaluate
    eval_model(model, val_dataloader)

    # Data trends
    test_data_trend(val_dataloader)

    