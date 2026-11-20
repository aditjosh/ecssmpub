import sqlite3
import numpy as np
import pandas as pd
import torch
import random
import sys, math
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import SequentialSampler, BatchSampler, RandomSampler
from sklearn.model_selection import GroupShuffleSplit

class EventsSequenceDataset(IterableDataset):
    """
    Reads records from database.
    """
    def __init__(self, db_path: str, table_name: str='timeline', pool_ids: list[str]=[], seq_len: int=1000, 
        vol_window_len: int=30, fetch_limit: int=10000, dec0: int=18, dec1: int=18):
        """
        Dataset for AMM event sequences. Volatility is calculated using a sliding window.
        Extracts contiguous chunks of length `seq_len` from continuous market logs.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.vol_window_len = vol_window_len
        self.db_fetch_batch = seq_len
        self.seq_len = seq_len
        self.fetch_limit = fetch_limit
        self.event_type_map = {"mod_liquidity": float(0.0), "swap": float(1.0)}
        self.pool_ids = pool_ids
        self.dec0 = dec0
        self.dec1 = dec1

        # Calculate total available pool events
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(self.pool_ids))
        cursor.execute(f"SELECT COUNT(id) FROM {self.table_name} WHERE pool_id IN ({placeholders})", self.pool_ids)
        self.total_records = cursor.fetchone()[0]
        conn.close()
    
    def __len__(self):
        return self.total_records

    def _get_query(self, allowed_pools: list[str]):
        """Construct SQL parameter query for allowed pool_ids."""
        placeholders = ",".join(["?"] * len(allowed_pools))
        if self.fetch_limit:
            query = f"""
                SELECT id AS id, block_ts_utc AS block_ts_utc, event_type AS event_type, COALESCE(sqrt_price_x96, '0.0') AS sqrt_price_x96, 
                COALESCE(liquidity, '0.0') AS liquidity, COALESCE(tick_upper, '0.0') AS tick_upper, COALESCE(tick_lower, '0.0') AS tick_lower, 
                COALESCE(delta_liquidity, '0.0') AS delta_liquidity, COALESCE(amount0, '0.0') AS amount0, COALESCE(amount1, '0.0') AS amount1 
                FROM {self.table_name}
                WHERE pool_id IN ({placeholders})
                ORDER BY id ASC
                LIMIT {self.fetch_limit};
        """
        else:
            query = f"""
                SELECT id AS id, block_ts_utc AS block_ts_utc, event_type AS event_type, COALESCE(sqrt_price_x96, '0.0') AS sqrt_price_x96, 
                COALESCE(liquidity, '0.0') AS liquidity, COALESCE(tick_upper, '0.0') AS tick_upper, COALESCE(tick_lower, '0.0') AS tick_lower, 
                COALESCE(delta_liquidity, '0.0') AS delta_liquidity, COALESCE(amount0, '0.0') AS amount0, COALESCE(amount1, '0.0') AS amount1 
                FROM {self.table_name}
                WHERE pool_id IN ({placeholders})
                ORDER BY id ASC;
            """
        
        return query, allowed_pools

    def compute_swap_volumes(self, amount0: np.ndarray, amount1: np.ndarray, log_S: np.ndarray,
        event_types: np.ndarray = None, dec0: int = 18, dec1: int = 18):
        """
        Computes unified swap volume in Token 1 (quote) units from raw token swap deltas.

        Parameters:
        - amount0: 1D array of Token 0 delta amounts (signed or unsigned).
        - amount1: 1D array of Token 1 delta amounts (signed or unsigned).
        - log_S: 1D array of log(sqrt(P)), where P = token1 / token0 (decimal-adjusted).
        - event_types: Optional 1D array where 1 = Swap. If provided, non-swaps get 0 volume.
        - dec0: Decimal places for Token 0 (e.g., 18 for WETH, 8 for WBTC). Set to 0 if already normalized.
        - dec1: Decimal places for Token 1 (e.g., 6 for USDC/USDT). Set to 0 if already normalized.

        Returns:
        - swap_volumes: 1D array of positive trade volumes denominated in Token 1 units.
        """
        # 1. Adjust raw integer amounts by token decimals (skip if already decimal-adjusted)
        adj_amt0 = np.abs(amount0.astype(np.float64)) / (10.0 ** dec0)
        adj_amt1 = np.abs(amount1.astype(np.float64)) / (10.0 ** dec1)

        # 2. Compute instantaneous pool price P = S^2 = exp(2 * log_S)
        # Clamp log_S to avoid floating-point overflow in exp()
        price_t = np.exp(np.clip(2.0 * log_S.astype(np.float64), -30.0, 30.0))

        # 3. Convert Token 0 leg into Token 1 units and take the two-leg mean
        vol_in_token1 = 0.5 * (adj_amt0 * price_t + adj_amt1)

        # 4. Zero out non-swap events (e.g., mints/burns) so they don't update the EMA
        if event_types is not None:
            vol_in_token1 = np.where(event_types == 1, vol_in_token1, 0.0)

        return vol_in_token1.astype(np.float32)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Split target pool_ids across DataLoader workers if num_workers > 1
        if worker_info is not None and len(self.pool_ids) > 1:
            worker_pools = np.array_split(self.pool_ids, worker_info.num_workers)[worker_info.id].tolist()
        else:
            worker_pools = self.pool_ids

        if not worker_pools:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Historical memory elements to preserve continuity across db_fetch_batch pagination
        #history_buffer = []
        #returns_history = []
        count_fetched = 0

        # Iterate per pool to preserves continuous sequence per batch
        for pool_id in worker_pools:
            query, params = self._get_query([pool_id])
            cursor.execute(query, params)

            # Historical memory elements to preserve continuity across db_fetch_batch pagination
            history_buffer = []
            last_log_S = None
            returns_history = []

            #while True:
            for rows in pd.read_sql_query(query, conn, params=params, chunksize=self.db_fetch_batch,
                dtype={"id": str, "block_ts_utc": str, "event_type": str, "sqrt_price_x96": str, "liquidity": np.float32,
                "tick_upper": np.float32, "tick_lower": np.float32, "delta_liquidity": np.float32, "amount0": np.float32,
                "amount1": np.float32}):
                ## Low-overhead raw tuple extraction 
                #rows = cursor.fetchmany(self.db_fetch_batch)
                #if not rows:
                #    break

                # Convert raw tuples to isolated NumPy structural column vectors
                # This completely avoids heavy pandas overhead
                #column_types = np.dtype([
                #    ('block_ts_utc', 'S200'),    # The string column
                #    ('event_type', 'S100'),      # An integer column
                #    ('sqrt_price_x96', 'f8'),    # A decimal/float column
                #    ('liquidity', 'f4'),
                #    ('tick_upper', 'f4'),
                #    ('tick_lower', 'f4'),
                #    ('delta_liquidity', 'f4'),
                #    ('amount0', 'f4'),
                #    ('amount1', 'f4'),
                #])

                raw_matrix = rows[rows.iloc[:, 3] != 'None']

                if raw_matrix.empty:
                    continue
                
                idx = raw_matrix['id'].to_numpy()
                block_ts_utc = raw_matrix['block_ts_utc'].to_numpy()
                event_types_raw = raw_matrix['event_type'].to_numpy()
                event_types = []
                for v in event_types_raw:
                    if v == 'mod_liquidity':
                        event_types.append(0.0)
                    else:
                        event_types.append(1.0)
                sqrt_price_x96 = raw_matrix['sqrt_price_x96'].to_numpy(dtype=np.float64)
                target_liq = raw_matrix['liquidity'].to_numpy()
                tick_u = raw_matrix['tick_upper'].to_numpy()
                tick_l = raw_matrix['tick_lower'].to_numpy()
                delta_liq = raw_matrix['delta_liquidity'].to_numpy()
                amt0 = raw_matrix['amount0'].to_numpy()
                amt1 = raw_matrix['amount1'].to_numpy()

                # --- Vectorized Physics Math Transformations ---
                current_prices = (sqrt_price_x96 / (2**96)) ** 2
                log_S = np.log(sqrt_price_x96)

                # Compute sequential log returns across batches flawlessly
                if last_log_S is not None:
                    padded_log_S = np.concatenate([[last_log_S], log_S])
                    target_returns = 2.0 * np.diff(padded_log_S)              # Log return = ln(P_t / P_(t-1))
                else:
                    try:
                        target_returns = 2.0 * np.diff(np.concatenate([[log_S[0]], log_S]))
                    except IndexError:
                        print('Index error')
                        sys.exit()
                    
                last_log_S = log_S[-1]

                # ---- Swap volumes ------
                swap_volumes = self.compute_swap_volumes(amt0, amt1, log_S, event_types, self.dec0, self.dec1)

                # --- Continuous Rolling Volatility Vectorization ---
                # Maintain a strict historical list of returns for the rolling window calculation
                all_returns = returns_history + target_returns.tolist()

                target_vols = []
                
                # Calculate rolling standard deviations efficiently
                # Uses a manual fallback loop for the boundary window to preserve execution speed
                start_idx = len(returns_history)
                for i in range(start_idx, len(all_returns)):
                    window_slice = all_returns[max(0, i - self.vol_window_len + 1) : i + 1]
                    if len(window_slice) > 1:
                        target_vols.append(np.std(window_slice, ddof=1))
                    else:
                        target_vols.append(0.02) # Baseline standard deviation warmup

                # Cache the tail end of the returns for the upcoming cursor fetch cycle
                returns_history = all_returns[-(self.vol_window_len - 1):] if len(all_returns) >= self.vol_window_len else all_returns
                target_vols = np.array(target_vols, dtype=np.float32)

                # Scale raw volatility decimals up to percentage space (0.015 -> 1.5)
                target_vols_scaled = target_vols * 100.0
                #target_vols_scaled = target_vols
                
                # --- Model Metric Normalization Processing ---
                target_log_l = np.log(target_liq + 1e-5).astype(np.float32)
                stable_amount0 = (np.sign(amt0) * np.log1p(np.abs(amt0))).astype(np.float32)
                stable_amount1 = (np.sign(amt1) * np.log1p(np.abs(amt1))).astype(np.float32)
                
                # --- Sequence Reconstruction Queue ---
                # Pack processed calculations back into sequence records
                for t in range(len(raw_matrix)):
                    history_buffer.append({
                        "id": idx[t],
                        "block_ts_utc": block_ts_utc[t],
                        "event_types": event_types[t],
                        "modify_seq": np.array([tick_l[t], tick_u[t], delta_liq[t]], dtype=np.float32),
                        "stable_swap_seq": np.array([stable_amount0[t], stable_amount1[t]], dtype=np.float32),
                        "swap_volumes": swap_volumes[t].astype(np.float32),
                        "raw_swap_seq": np.array([amt0[t], amt1[t]], dtype=np.float32),
                        "current_prices": current_prices[t].astype(np.float32),
                        "log_sqrt_prices": log_S[t].astype(np.float32),
                        "target_returns": target_returns[t].astype(np.float32),
                        "target_vols": target_vols_scaled[t],
                        "target_log_l": target_log_l[t]
                    })

                    # The moment our FIFO rolling queue matches our Mamba window, we slice and yield
                    if len(history_buffer) == self.seq_len:
                        yield {
                            "id": np.array([x["id"] for x in history_buffer], dtype=str),
                            "block_ts_utc": np.array([x["block_ts_utc"] for x in history_buffer], dtype=str),
                            "event_types": np.array([x["event_types"] for x in history_buffer], dtype=np.int64),
                            "modify_seq": np.stack([x["modify_seq"] for x in history_buffer], axis=0),
                            "stable_swap_seq": np.stack([x["stable_swap_seq"] for x in history_buffer], axis=0),
                            "swap_volumes": np.array([x["swap_volumes"] for x in history_buffer], dtype=np.float32),
                            "raw_swap_seq": np.stack([x["raw_swap_seq"] for x in history_buffer], axis=0),
                            "current_prices": np.array([x["current_prices"] for x in history_buffer], dtype=np.float32),
                            "log_sqrt_prices": np.array([x["log_sqrt_prices"] for x in history_buffer], dtype=np.float32),
                            "target_returns": np.array([x["target_returns"] for x in history_buffer], dtype=np.float32),
                            "target_vols": np.array([x["target_vols"] for x in history_buffer], dtype=np.float32),
                            "target_log_l": np.array([x["target_log_l"] for x in history_buffer], dtype=np.float32),
                        }
                        # Clear out the old elements to free memory space
                        count_fetched += len(history_buffer)
                        history_buffer = []

            # No need to fetch from remaining pools if reached the fetch_limit
            if self.fetch_limit and count_fetched >= self.fetch_limit:
                break
        conn.close()

def pinn_sequence_collate(batch):
    """
    Collates dynamic dictionaries directly into the structured matrix 
    dimensions required by HardenedPINNDirectionalMamba.
    """
    return {
        "id": np.array([item["id"] for item in batch], dtype=str),
        "block_ts_utc": np.array([item["block_ts_utc"] for item in batch], dtype=str),
        "event_types": torch.from_numpy(np.array([item['event_types'] for item in batch], dtype=np.int64)),
        "modify_seq": torch.from_numpy(np.array([item['modify_seq'] for item in batch], dtype=np.float32)),
        "stable_swap_seq": torch.from_numpy(np.array([item['stable_swap_seq'] for item in batch], dtype=np.float32)),
        "swap_volumes": torch.from_numpy(np.array([item['swap_volumes'] for item in batch], dtype=np.float32)),
        "raw_swap_seq": torch.from_numpy(np.array([item['raw_swap_seq'] for item in batch], dtype=np.float32)),
        "current_prices": torch.from_numpy(np.array([item['current_prices'] for item in batch], dtype=np.float32)),
        "log_sqrt_prices": torch.from_numpy(np.array([item['log_sqrt_prices'] for item in batch], dtype=np.float32)),
        "target_returns": torch.from_numpy(np.array([item['target_returns'] for item in batch], dtype=np.float32)),
        "target_vols": torch.from_numpy(np.array([item['target_vols'] for item in batch], dtype=np.float32)),
        "target_log_l": torch.from_numpy(np.array([item['target_log_l'] for item in batch], dtype=np.float32)),
    }

##############
##############

def split_pools(db_path, test_size):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT pool_id FROM {table_name}")
    unique_pools = [str(row[0]) for row in cursor.fetchall()]
    conn.close()

    # Group split pool_ids exclusively into train and validation sets
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(unique_pools, groups=unique_pools))
    train_pools = np.array(unique_pools)[train_idx].tolist()
    val_pools = np.array(unique_pools)[val_idx].tolist()
    return train_pools, val_pools

def create_events_sequence_dataloaders(db_path: str, table_name: str="timeline", seq_len: int=1000, 
    vol_window_len: int=30, train_pools = [], validation_pools = [], fetch_limit: int=None, 
    test_prop: float=0.2, dec0: int = 18, dec1: int = 18, random_state: int = 42):

    if len(train_pools) == 0 and len(validation_pools) == 0:
        # Don't split by pool_ids abruptly when few pools have orders of magintude more data!
        train_pools, validation_pools = split_pools(db_path, test_size)
    
    # Instantiate IterableDatasets with restricted pool lists
    BATCH_SIZE = 4
    num_workers = 1
    train_loader, val_loader = None, None
    if len(train_pools) >= 0:
        train_dataset = EventsSequenceDataset(db_path=db_path, table_name=table_name, dec0=dec0, dec1=dec1,
            pool_ids=train_pools, fetch_limit=int((1.0 - test_prop) * fetch_limit), seq_len=seq_len, vol_window_len=vol_window_len)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, # Total tensors per batch: 8 batches x 2,300 elements = 18,400 steps
            collate_fn=pinn_sequence_collate,  # Use our clean collate function
            num_workers=num_workers,       # Spreads the 1M rows over 4 parallel database streamers
            pin_memory=True,
            drop_last=True
        )
    
    if len(validation_pools) >= 0:
        val_dataset = EventsSequenceDataset(db_path=db_path, table_name=table_name, dec0=dec0, dec1=dec1,
            pool_ids=validation_pools, fetch_limit=int(test_prop * fetch_limit), seq_len=seq_len, vol_window_len=vol_window_len)
        val_loader = DataLoader(
            val_dataset, 
            collate_fn=pinn_sequence_collate,  # Use our clean collate function
            batch_size=BATCH_SIZE, 
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

    return train_loader, val_loader


# Manually assigned training and validation pools accounting for each pool size.
def assign_train_validation_pools(tokens_tag):
    if tokens_tag == 'eth_usdt':
        train_pools = [
            '0x90078845bceb849b171873cfbc92db8540e9c803ff57d9d21b1215ec158e79b3',
            '0x903abdd60327e732c5e25f66310681200d0f5cda61683764b2d5434a0d985f2c'
        ]
        validation_pools = [
            '0xf2df4f75505a4e1e5bfb6e36b3be82c2e03d5d8f8f9c6b7caa1cbd059ec5c2ee',
            '0x1dc6f99e8bf15e8b62c8e7eb03b67291e6bc5accf2970a55a7068a7fc0b33a46',
            '0x7892abd5b96b71630a7fafa79f6242d785b02b4c4329c394b2954a1b6361a042'
        ]
    elif tokens_tag == "usdc_eth":
        train_pools = [
            '0xe500210c7ea6bfd9f69dce044b09ef384ec2b34832f132baec3b418208e3a657',
            '0xb9fd8b8b6580b80432037a379071b6fb22bbceff6bafbd57390889c01d8757b9',
            '0xdd8dd509e58ec98631b800dd6ba86ee569c517ffbd615853ed5ab815bbc48ccb',
            '0xde033c66aad5552aa960e6546da5661ecb79c99e52a66962c338938080d8111e',
            '0xf5add258e96048c4b55f8d4829eef24fad82074eb3ff9518841bd88701149a53',
            '0xf92118dfd8a825604e56784d4387aea38559cc78e8fab5d395533cb2e37ce353'
        ]
        validation_pools = [
            '0x4f88f7c99022eace4740c6898f59ce6a2e798a1e64ce54589720b7153eb224a7',
            '0x55cd2060d9ad3c05fff7061604a332a29adcd0aa557efa63973f26d00553ec5f',
            '0x5a5c7cab5f55c7ea020e97d4fa6dd5d99270e56ce76afa61d8cbddec0af92060'
        ]
    elif tokens_tag == "wbtc_usdc":
        train_pools = [
            '0xaa2e4169ac3a6ada5e46836017d7c43887b8e5642d9267918ea9895d90af138d',
            '0x3ea74c37fbb79dfcd6d760870f0f4e00cf4c3960b3259d0d43f211c0547394c1'
        ]
        validation_pools = [
            '0xb98437c7ba28c6590dd4e1cc46aa89eed181f97108e5b6221730d41347bc817f',
            '0xbcc7f8dc40d8ab941271c66916190fed73f540690741ee41bb9bdb26cc6f587c'
        ]

    return train_pools, validation_pools

def test_dataloader():
    SEQ_LEN = 100
    NUM_EPOCHS = 1
    TOKENS_TAG = 'usdc_eth'
    db_file = f"../analysis/sqldb/{TOKENS_TAG}/{TOKENS_TAG}_univ4_timeline.db"

    train_pools, validation_pools = assign_train_validation_pools(TOKENS_TAG)

    train_dataloader, val_dataloader = create_events_sequence_dataloaders(
        db_file, table_name="timeline", seq_len=SEQ_LEN, vol_window_len=30, train_pools=train_pools, 
        validation_pools=validation_pools, fetch_limit=1000, test_prop=0.2, dec0=dec0, dec1=dec1,)

    for epoch in range(1, NUM_EPOCHS + 1):
        item_count_t = 0
        item_count_v = 0
        for batch in train_dataloader:
            print("--- TRAIN DATASET PIPELINE SANITY CHECK ---")
            print("Sample Target Returns (First 5):", batch['target_returns'][0, :5].cpu().numpy())
            print("Sample Target Volatility Mean  :", batch['target_vols'].mean().item())
            print("Sample Target Volatility Max   :", batch['target_vols'].max().item())
            print("Sample ID :", batch['id'][0, :5])
            item_count_t += math.prod(batch['target_returns'].shape)
        print("Total items in train batches: ", item_count_t)

        for batch in val_dataloader:
            print("--- VALIDATION DATASET PIPELINE SANITY CHECK ---")
            #print("Sample Target Returns (First 5):", batch['target_returns'][0, :5].cpu().numpy())
            #print("Sample Target Volatility Mean  :", batch['target_vols'].mean().item())
            #print("Sample Target Volatility Max   :", batch['target_vols'].max().item())
            print("Sample ID :", batch['id'][0, :5])
            item_count_v += math.prod(batch['target_returns'].shape)
        print("Total items in validation batches: ", item_count_v)

# --- Example Usage ---
if __name__ == "__main__":
    test_dataloader()
