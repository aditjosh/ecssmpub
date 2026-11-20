import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from model.regression.directional_bce_loss import DirectionalBCELoss
from model.regression.multitask_loss import MultiTaskLoss
from model.regression.dual_head_multitask_loss import DualHeadMultiTaskLoss
from model.constants import Constants

class EventConditionedSSMCell(nn.Module):
    def __init__(self, hidden_dim: int, init_weights: str = 'zeros', forecast_horizon_k: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon_k = forecast_horizon_k

        # --- Net1: Modify Liquidity State Layers ---
        # Input features: [tick_upper, tick_lower, delta_liquidity]
        self.modify_proj = nn.Linear(3, hidden_dim)
        self.modify_ssm_A = nn.Linear(3, hidden_dim)
        self.modify_ssm_B = nn.Linear(3, hidden_dim)

        # HEAD 1: Forecast Directional Log-Return (mu) -> Can be positive or negative
        self.forecast_return = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, forecast_horizon_k)
        )
        
        # HEAD 2: Forecast Volatility (sigma) -> Must be strictly positive
        self.forecast_log_volatility = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, forecast_horizon_k),
            nn.Softplus()
        )
        
        # HEAD 3: Forecast Liquidity (as before)
        self.forecast_log_liq = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, forecast_horizon_k)
        )

        # Dedicated Direction Classification Head (Logits)
        self.dir_logits_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon_k)
        )
        
        # Expected Return Magnitude Head (Non-negative)
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon_k),
            nn.Softplus()
        )

        # --- Net2: Swap Layers & Predictors ---
        # Input features: [amount0, amount1]
        # Predictor combines previous hidden state (HS1) + current swap inputs
        self.swap_update_proj = nn.Linear(forecast_horizon_k + 2, hidden_dim)
        self.swap_ssm_A = nn.Linear(forecast_horizon_k + 2, hidden_dim)
        self.swap_ssm_B = nn.Linear(forecast_horizon_k + 2, hidden_dim)        

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Learnable horizon step embeddings (1 to k)
        self.horizon_embed = nn.Embedding(forecast_horizon_k, hidden_dim)

        # Direct linear bottleneck shortcut for residual stability
        self.shortcut = nn.Linear(hidden_dim, forecast_horizon_k)

        self._init_cells(init_weights=init_weights)
        
    def _init_cells(self, init_weights: str = 'rand'):
        # Initialize volatility bias to a realistic daily standard deviation (e.g., 2% -> 0.02)
        if init_weights == 'rand':
            beta = 0.01  # usdc/eth=1.0, eth/usdt=0.01, wbtc/usdc=0.1
            nn.init.normal_(self.forecast_log_volatility[-2].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.forecast_log_volatility[-2].bias, beta * random.randint(1, 5))
            nn.init.normal_(self.forecast_return[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.forecast_return[-1].bias, beta * random.randint(1, 5)) 
            nn.init.normal_(self.forecast_log_liq[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.forecast_log_liq[-1].bias, beta * random.randint(1, 5))
            nn.init.xavier_uniform_(self.dir_logits_head[-1].weight)
            nn.init.normal_(self.dir_logits_head[-1].bias, beta * random.randint(1, 5))
        else:
            nn.init.normal_(self.forecast_return[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.forecast_return[-1].bias)
            nn.init.normal_(self.forecast_log_volatility[-2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.forecast_log_volatility[-2].bias)
            nn.init.normal_(self.forecast_log_liq[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.forecast_log_liq[-1].bias)
            nn.init.xavier_uniform_(self.dir_logits_head[-1].weight)
            nn.init.zeros_(self.dir_logits_head[-1].bias)


    def forward(self, event_type, modify_in, swap_in, hs):
        """
        Processes a single step in the sequence.
        Args:
            event_type: Tensor (B, 1) -> 0 for modifyLiquidity, 1 for swap
            modify_in:  Tensor (B, 3) -> [tick_upper, tick_lower, delta_liquidity]
            swap_in:    Tensor (B, 2) -> [amount0, amount1]
            hs_t:       Tensor (B, t, hidden_dim) -> Previous hidden state HS
        """
        B, H = hs.shape

        # --- PATHWAY A: modifyLiquidity ---
        mod_x = self.modify_proj(modify_in)
        mod_A = torch.sigmoid(self.modify_ssm_A(modify_in))
        mod_B = torch.sigmoid(self.modify_ssm_B(modify_in))
        hs_modify = mod_A * hs + mod_B * mod_x          # B, T, H

        # --- PATHWAY B: swap ---
        swap_context = torch.cat([hs, swap_in], dim=-1)   # B, H + 2
        
        # Output volatility (constrained to be strictly positive using ELU)
        pred_log_vol = self.forecast_log_volatility(swap_context)      # B, k
        pred_vol = torch.exp(pred_log_vol)
        
        # Output Liquidity
        pred_log_liq = self.forecast_log_liq(swap_context)              # B, k

        # Output directionality
        pred_dir_logits = self.dir_logits_head(swap_context)    
        pred_mags = self.mag_head(swap_context)

        is_modify = (event_type == 0).float()
        is_swap = (event_type == 1).float()
        
        # Feedback Loop -> Feed predicted liquidity back into state update
        # Clamp log liquidity feedback to a realistic, safe regime
        # Stops the model from passing micro-fractions back into the state machine
        # 10^4 to 10^30 in raw liquidity corresponds to 9.2 to 69.0 in log space
        safe_feedback_log_liq = torch.clamp(pred_log_liq, min=9.0, max=70.0)
        
        swap_feedback_in = torch.cat([swap_in, safe_feedback_log_liq], dim=-1)
        swap_x = self.swap_update_proj(swap_feedback_in)
        swap_A = torch.sigmoid(self.swap_ssm_A(swap_feedback_in))
        swap_B = torch.sigmoid(self.swap_ssm_B(swap_feedback_in))
        hs_swap = swap_A * hs + swap_B * swap_x
        
        # --- Conditional State Transition & Routing ---
        next_hs = is_modify * hs_modify + is_swap * hs_swap
        ## Clamping prevents sequence history from scaling infinitely
        next_hs = torch.clamp(next_hs, min=-10.0, max=10.0)
        
        # Zero out predictions for modifyLiquidity steps
        return next_hs, is_swap * pred_dir_logits, is_swap * pred_mags, is_swap * pred_vol, is_swap * pred_log_liq

class ECVolatilityPINN(nn.Module):
    def __init__(self, hidden_dim: int, init_weights: str = 'zeros', noise_margin: float = 1e-4,
        forecast_horizon_k: int = 10, forecast_horizon_k_step_size: int = 1, baseline_vol: float = 0.002):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon_k = forecast_horizon_k
        self.forecast_horizon_k_step_size = forecast_horizon_k_step_size
        self.ec_cell = EventConditionedSSMCell(hidden_dim, init_weights=init_weights)
        self.loss_fn = DualHeadMultiTaskLoss()
        self.dropout = nn.Dropout(p=0.2)

        # SEPARATE 0-Step Contemporaneous Impact Head (EXCLUSIVELY FOR PINN PHYSICS)
        # Prevents physics constraints at step 't' from corrupting forward horizon 't+1...t+k' forecasts
        self.contemporaneous_phys_head = nn.Linear(hidden_dim, 1)

        self.loss_fn = DualHeadMultiTaskLoss(
            noise_margin=noise_margin
        )

    def _build_multi_horizon_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Transforms 2D targets (B, T) to 3D multi-horizon target tensor (B, T, k).
        target_3d[b, t, j] = targets[b, t + j + 1]
        """
        if targets.dim() == 3:
            return targets  # Already shaped as (B, T, k)
        
        B, T = targets.shape
        k = self.forecast_horizon_k
        step = self.forecast_horizon_k_step_size
        targets_3d = torch.zeros(B, T, k, device=targets.device, dtype=targets.dtype)

        for j in range(0, k, step):
            # Shift targets left by j + 1 steps
            shifted = torch.roll(targets, shifts=-(j + 1), dims=1)
            # Mask out sequence boundary wrapping at the end
            shifted[:, -(j + 1):] = 0.0
            targets_3d[:, :, j] = shifted
        return targets_3d

    def _expand_array_dim(self, src_array, t_d1, t_d2):
            D0, D1, D2 = src_array.shape
            expanded_pred_rets = torch.zeros((D0, t_d1, t_d2), dtype=src_array.dtype)
            rets_j = 0
            step = self.forecast_horizon_k_step_size
            for j in range(0, t_d2, step):
                expanded_pred_rets[:,:,j] = src_array[:,:,rets_j]
                rets_j += 1
            return expanded_pred_rets

    def forward(self, event_types: torch.Tensor, modify_seq: torch.Tensor, stable_swap_seq: torch.Tensor, 
        raw_swap_seq: torch.Tensor, current_prices: torch.Tensor, dir_labels: torch.Tensor = None,
        valid_dir_mask: torch.Tensor = None, target_returns: torch.Tensor = None, 
        target_vols: torch.Tensor = None, target_log_l: torch.Tensor = None, lambda_data=1.0, lambda_phys=0.1):

        batch_size, seq_len = event_types.size()
        device = event_types.device
        k = self.forecast_horizon_k
        step = self.forecast_horizon_k_step_size
        horizon_dim_k = k // step     # E.g. 10//1 = 10, 51//10 = 5
        
        hs = torch.zeros(batch_size, self.hidden_dim, device=device)        # B x H
        return_history, vol_history, liquidity_history, target_return_history, mag_history, dir_logits_history = [], [], [], [], [], []
        physics_loss_accumulator = torch.tensor(0.0, device=device)
        swap_count = 0
        
         # --- Pre-transform inputs to prevent linear layer explosion ---
        # Ticks are naturally small, but delta_liquidity can be massive.
        mod_seq_stable = modify_seq.clone()
        mod_seq_stable[:, :, 2] = torch.sign(modify_seq[:, :, 2]) * torch.log(torch.abs(modify_seq[:, :, 2]) + 1.0)

        # For each input in seq_len, generate k-length forecast
        for t in range(seq_len):
            ev_t = event_types[:, t:t+1].to(device)
            mod_t = mod_seq_stable[:, t].to(device)
            swap_t = stable_swap_seq[:, t].to(device)
            raw_swap_t = raw_swap_seq[:, t].to(device)
            p_t = current_prices[:, t:t+1].to(device)
            hs_prev = hs.clone()
            hs, pred_dir_logits_k, pred_mag_k, pred_vol_k, pred_log_l_k = self.ec_cell(ev_t, mod_t, swap_t, hs)

            pred_ret_k = torch.tanh(pred_dir_logits_k / 2.0) * pred_mag_k
            # Contemporaneous 0-step prediction for physics alignment
            pred_contemp_ret_t = self.contemporaneous_phys_head(hs_prev)

            return_history.append(pred_ret_k)
            vol_history.append(pred_vol_k)
            liquidity_history.append(pred_log_l_k)
            mag_history.append(pred_mag_k)
            dir_logits_history.append(pred_dir_logits_k)
            
            is_swap_step = (ev_t == 1)
            if is_swap_step.any() and lambda_phys != 0.0:
                # --- AMM PRICE IMPACT PHYSICS ---
                raw_dx = raw_swap_t[:, 0:1] # amount0
                raw_dy = raw_swap_t[:, 1:2] # amount1

                # Extract 1-step predictions for physical constraint alignment
                pred_ret_1step = pred_ret_k[:, 0:1]
                pred_vol_1step = pred_vol_k[:, 0:1]
                pred_log_l_1step = pred_log_l_k[:, 0:1]
                
                predicted_liq = torch.exp(pred_log_l_1step)
                sqrt_p = torch.sqrt(torch.clamp(p_t, min=1e-5))

                # Sign correction 
                # Swapping token 1 in (dy > 0) increases pool price (+).
                # Swapping token 0 in (dx > 0) decreases pool price (-).
                # Enforce signed direction explicitly via torch.sign:
                signed_dy = torch.abs(raw_dy)
                signed_dx = torch.abs(raw_dx)

                # Determine active side from raw_swap features or stable_swap_seq direction
                is_dy_in = (raw_dy > 0) | (stable_swap_seq[:, t, 0:1] > 0)

                # Physical Log-Return Calculation:
                # Compute exact physical sqrt_price shift
                # dy increases pool sqrt_price, dx decreases it
                delta_sqrt_p = (signed_dy / (predicted_liq + 1e-5))
                next_sqrt_p = torch.clamp(sqrt_p + delta_sqrt_p, min=1e-5)

                # Convert the physical move into LOG-RETURN space
                # This puts the physics constraint on the EXACT same numerical scale as target_returns!
                physical_return = torch.where(
                    is_dy_in,
                    2.0 * (torch.log(next_sqrt_p) - torch.log(sqrt_p)),   # Positive price movement
                    -2.0 * (torch.log(next_sqrt_p) - torch.log(sqrt_p))   # Negative price movement
                )

                # Target physical log-volatility: ln(|physical_log_return|)
                physical_vol = Constants.OMEGA * (torch.abs(physical_return) + 1e-5)

                # Calculate step_phys_loss (Now natively in log-space!)
                step_ret_phys_loss = torch.pow(pred_ret_1step - physical_return, 2) * is_swap_step.float() # MSE
                step_vol_phys_loss = torch.pow(pred_vol_1step - physical_vol, 2) * is_swap_step.float()
                total_step_phys_loss = (Constants.LAMBDA6 * step_ret_phys_loss + Constants.LAMBDA7 * step_vol_phys_loss)
                total_step_phys_loss = torch.where(torch.isnan(total_step_phys_loss), torch.zeros_like(total_step_phys_loss), total_step_phys_loss)
                physics_loss_accumulator += total_step_phys_loss.mean()
                swap_count += 1

        pred_rets = torch.stack(return_history, dim=1)  
        # Shift the values across k horizon by filling in the values between [0,0,i] and [0,0,i+1] with the
        # values in [0,0,i]. i iterates till < k.               
        pred_rets = self._expand_array_dim(pred_rets, seq_len, k).to(device)       
        pred_vols = torch.stack(vol_history, dim=1)
        pred_vols = self._expand_array_dim(pred_vols, seq_len, k).to(device)
        pred_log_liqs = torch.stack(liquidity_history, dim=1)
        pred_log_liqs = self._expand_array_dim(pred_log_liqs, seq_len, k).to(device)
        pred_mags = torch.stack(mag_history, dim=1)
        pred_mags = self._expand_array_dim(pred_mags, seq_len, k).to(device)
        pred_dir_logits = torch.stack(dir_logits_history, dim=1) # [B, T, k]
        pred_dir_logits = self._expand_array_dim(pred_dir_logits, seq_len, k).to(device)
        
        # Multi-Horizon Data Loss Computation
        if target_returns is not None and target_vols is not None and target_log_l is not None:
            # Construct 3D Multi-Horizon Targets (B, seq_len, k)
            target_rets_multi = self._build_multi_horizon_targets(target_returns)
            target_vols_multi = self._build_multi_horizon_targets(target_vols)
            target_log_l_multi = self._build_multi_horizon_targets(target_log_l)

            # swap mask
            swap_mask = (event_types == 1).unsqueeze(-1)         # Shape: (B, seq_len, 1)

            # Validity mask to ignore padding at sequence boundary (t + j >= seq_len)
            valid_horizon = torch.ones(batch_size, seq_len, k, device=device, dtype=torch.bool)
            for j in range(0, k, step):
                valid_horizon[:, -(j + 1):, j] = False

            swap_horizon_mask = (swap_mask & valid_horizon).expand(-1, -1, k)      # B x SEQ_LEN x k
            full_mask = swap_horizon_mask & valid_dir_mask
            mean_physics_loss = physics_loss_accumulator / max(swap_count, 1)

            # Construct Horizon Discount Weights: [1, 1, k] -> gamma^0, gamma^1 ... gamma^(k-1)
            discount_weights = (Constants.GAMMA_D ** torch.arange(k, device=device)).view(1, 1, k)

            # Compute balanced multi-task loss
            joint_loss, total_data_loss, loss_dir, loss_mag = self.loss_fn(
                pred_dir_logits, pred_mags, pred_rets, pred_vols, pred_log_liqs, full_mask,
                dir_labels, target_rets_multi, target_vols_multi, target_log_l_multi, 
                mean_physics_loss, lambda_data, lambda_phys, discount_weights
            )
            return joint_loss, total_data_loss, loss_dir, mean_physics_loss
            
        return pred_rets, pred_dir_logits, pred_vols, pred_log_liqs
