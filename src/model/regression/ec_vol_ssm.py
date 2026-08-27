import torch
import torch.nn as nn
from model.regression.directional_bce_loss import DirectionalBCELoss

class EventConditionedVolatilitySSMCell(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # --- Net1: Modify Liquidity State Layers ---
        # Input features: [tick_upper, tick_lower, delta_liquidity]
        self.modify_proj = nn.Linear(3, hidden_dim)
        self.modify_ssm_A = nn.Linear(3, hidden_dim)
        self.modify_ssm_B = nn.Linear(3, hidden_dim)

        # HEAD 1: Predicts Directional Log-Return (mu) -> Can be positive or negative
        self.predict_return = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            #nn.Dropout(p=0.4),
            nn.Linear(hidden_dim // 2, 1)
            #nn.Linear(hidden_dim, 1)
        )
        
        # HEAD 2: Predicts Volatility (sigma) -> Must be strictly positive
        self.predict_log_volatility = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
            #nn.Linear(hidden_dim, 1)
        )
        
        # HEAD 3: Predicts Liquidity (as before)
        self.predict_log_liq = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            #nn.Dropout(p=0.4),
            nn.Linear(hidden_dim // 2, 1)
            #nn.Linear(hidden_dim, 1)
        )

        # --- Net2: Swap Layers & Predictors ---
        # Input features: [amount0, amount1]
        # Predictor combines previous hidden state (HS1) + current swap inputs
        self.swap_update_proj = nn.Linear(3, hidden_dim)
        self.swap_ssm_A = nn.Linear(3, hidden_dim)
        self.swap_ssm_B = nn.Linear(3, hidden_dim)

        # Initialize volatility bias to a realistic daily standard deviation (e.g., 2% -> 0.02)
        nn.init.zeros_(self.predict_log_volatility[-1].weight)
        nn.init.constant_(self.predict_log_volatility[-1].bias, 2.0)

    def forward(self, event_type, modify_in, swap_in, hs):
        """
        Processes a single step in the sequence.
        Args:
            event_type: Tensor (B, 1) -> 0 for modifyLiquidity, 1 for swap
            modify_in:  Tensor (B, 3) -> [tick_upper, tick_lower, delta_liquidity]
            swap_in:    Tensor (B, 2) -> [amount0, amount1]
            hs:         Tensor (B, hidden_dim) -> Previous hidden state HS
        """
        is_modify = (event_type == 0).float()
        is_swap = (event_type == 1).float()
        
        # --- PATHWAY A: modifyLiquidity ---
        mod_x = self.modify_proj(modify_in)
        mod_A = torch.sigmoid(self.modify_ssm_A(modify_in))
        mod_B = torch.sigmoid(self.modify_ssm_B(modify_in))
        hs_modify = mod_A * hs + mod_B * mod_x
        
        # --- PATHWAY B: swap ---
        swap_context = torch.cat([hs, swap_in], dim=-1)

        # 1. Output directional return (no restricting activation function needed)
        pred_return = self.predict_return(swap_context)
        
        # 2. Output volatility (constrained to be strictly positive using ELU)
        pred_log_vol = self.predict_log_volatility(swap_context)
        #pred_vol = torch.nn.functional.elu(raw_vol, alpha=1.0) + 1.0 + 1e-5
        # Softplus is strictly positive, smooth, and has active non-zero gradients
        # Adding 1e-4 prevents absolute zero
        #pred_vol = torch.nn.functional.softplus(raw_vol) + 1e-4
        pred_vol = torch.exp(pred_log_vol)
        
        # 3. Output Liquidity
        pred_log_liq = self.predict_log_liq(swap_context)

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
        # Clamping or using a Tanh prevents sequence history from scaling infinitely
        next_hs = torch.clamp(next_hs, min=-10.0, max=10.0)
        
        # Zero out predictions for modifyLiquidity steps
        return next_hs, is_swap * pred_return, is_swap * pred_vol, is_swap * pred_log_vol, is_swap * pred_log_liq


class ECVolatilityPINN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mamba_cell = EventConditionedVolatilitySSMCell(hidden_dim)
        ##self.mse = nn.MSELoss()
        ## Huber Loss acts quadratically for small errors, but linearly for large outliers
        self.loss_fn = nn.SmoothL1Loss(beta=1e-3)
        ##self.loss_fn = nn.MSELoss()
        self.direction_loss_fn = DirectionalBCELoss(alpha=50.0)

    def forward(self, event_types, modify_seq, stable_swap_seq, raw_swap_seq, 
                current_prices, target_returns=None, target_vols=None, target_log_l=None,
                lambda_data=1.0, lambda_phys=0.1):
        """
        Args:
            current_prices: Tensor of shape (B, seq_len) containing pool price (P) at step t
        """
        batch_size, seq_len = event_types.size()
        device = event_types.device
        
        hs1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return_history, vol_history, liquidity_history, log_vol_history = [], [], [], []
        physics_loss_accumulator = torch.tensor(0.0, device=device)
        swap_count = 0
        
         # --- Pre-transform inputs to prevent linear layer explosion ---
        # Ticks are naturally small, but delta_liquidity (index 2) is massive.
        mod_seq_stable = modify_seq.clone()
        mod_seq_stable[:, :, 2] = torch.sign(modify_seq[:, :, 2]) * torch.log(torch.abs(modify_seq[:, :, 2]) + 1.0)

        for t in range(seq_len):
            ev_t = event_types[:, t:t+1].to(device)
            mod_t = mod_seq_stable[:, t].to(device)
            swap_t = stable_swap_seq[:, t].to(device)
            raw_swap_t = raw_swap_seq[:, t].to(device)
            p_t = current_prices[:, t:t+1].to(device)
            
            hs1, pred_ret, pred_vol, pred_log_vol, pred_log_l = self.mamba_cell(ev_t, mod_t, swap_t, hs1)
            
            return_history.append(pred_ret)
            vol_history.append(pred_vol)
            log_vol_history.append(pred_log_vol)
            liquidity_history.append(pred_log_l)
            
            is_swap_step = (ev_t == 1)
            if is_swap_step.any():
                # --- AMM PRICE IMPACT PHYSICS ---
                raw_dx = raw_swap_t[:, 0:1] # amount0
                raw_dy = raw_swap_t[:, 1:2] # amount1
                
                predicted_liq = torch.exp(pred_log_l)
                sqrt_p = torch.sqrt(torch.clamp(p_t, min=1e-5))

                # Physical Log-Return Calculation:
                # Compute exact physical sqrt_price shift
                # dy increases pool sqrt_price, dx decreases it
                delta_sqrt_p = (raw_dy / (predicted_liq + 1e-5)) - (raw_dx * sqrt_p * sqrt_p / (predicted_liq + 1e-5))
                next_sqrt_p = torch.clamp(sqrt_p + delta_sqrt_p, min=1e-5)

                # Convert the physical move into LOG-RETURN space
                # This puts the physics constraint on the EXACT same numerical scale as target_returns!
                physical_log_return = 2.0 * (torch.log(next_sqrt_p) - torch.log(sqrt_p))
                # Multiply by 100.0 to match the scaled target_vols space!
                physical_log_return = torch.abs(physical_log_return) * 100.0

                # Target physical log-volatility: ln(|physical_log_return|)
                physical_log_vol = torch.log(torch.abs(physical_log_return) + 1e-5)
                
                # Calculate step_phys_loss (Now natively in log-space!)
                step_direction_phys_loss = torch.pow(pred_ret - physical_log_return, 2)
                #step_vol_phys_loss = torch.pow(pred_vol - physical_vol_target, 2)
                step_vol_phys_loss = torch.pow(pred_log_vol - physical_log_vol, 2)
                
                LAMBDA6 = 1.0
                step_phys_loss = (step_direction_phys_loss + LAMBDA6 * step_vol_phys_loss) * is_swap_step.float()
                physics_loss_accumulator += step_phys_loss.mean()
                swap_count += 1

        pred_rets = torch.stack(return_history, dim=1).squeeze(-1)
        pred_vols = torch.stack(vol_history, dim=1).squeeze(-1)
        pred_log_vols = torch.stack(log_vol_history, dim=1).squeeze(-1)
        pred_log_liqs = torch.stack(liquidity_history, dim=1).squeeze(-1)
        
        if target_returns is not None and target_vols is not None and target_log_l is not None:
            # swap mask
            mask = (event_types == 1)
            
            # Align output shapes safely
            pred_rets_masked = pred_rets[mask].view(-1)
            target_rets_masked = target_returns[mask].view(-1)
            
            pred_log_vols_masked = pred_log_vols[mask].view(-1)
            target_log_vols_masked = torch.log(target_vols[mask].view(-1) + 1e-5)

            # Multi-Task Data Loss
            loss_r_magnitude = self.loss_fn(pred_rets_masked, target_rets_masked)
            # Directionality Loss (Sign Penalty)
            loss_r_direction = self.direction_loss_fn(pred_rets_masked, target_rets_masked)
            # Combine return losses (give direction loss a weight factor like 0.1)
            LAMBDA1 = 100.0
            LAMBDA2 = 0.1
            loss_r_data = (LAMBDA1 * loss_r_magnitude) + (LAMBDA2 * loss_r_direction)
            #loss_r_data = loss_r_magnitude

            loss_v_data = self.loss_fn(pred_log_vols_masked, target_log_vols_masked)
            loss_l_data = self.loss_fn(pred_log_liqs[mask].view(-1), target_log_l[mask].view(-1))
            
            # Multiply volatility loss by a multiplier (e.g., 10.0 or 50.0)
            # to give its gradients equal "voting power" in the optimizer
            LAMBDA3 = 10.0
            total_data_loss = loss_r_data + (LAMBDA3 * loss_v_data) + loss_l_data
            
            if torch.isnan(total_data_loss):
                total_data_loss = torch.tensor(0.0, device=device, requires_grad=True)

            ## Physics loss
            mean_physics_loss = physics_loss_accumulator / max(swap_count, 1)

            if torch.isnan(mean_physics_loss) or torch.isinf(mean_physics_loss):
                print("Warning: Corrupted mean_physics_loss detected.")
                physics_loss_accumulator = 0.0
                mean_physics_loss = 0.0

            #print("total_data_loss:", total_data_loss)
            #print("mean_physics_loss:", mean_physics_loss)

            #print(mean_physics_loss)
            joint_loss = (lambda_data * total_data_loss) + (lambda_phys * mean_physics_loss)
            
            return joint_loss, total_data_loss, mean_physics_loss
            
        return pred_rets, pred_vols, pred_log_liqs