import sqlite3
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler, RandomSampler

class SlidingWindowVolatilityStreamer(IterableDataset):
    def __init__(self, db_path, table_name='timeline', split_type=0, window_size=2300, vol_window=30, db_fetch_batch=5000):
        """
        Args:
            db_path (str): Path to the SQLite database.
            window_size (int): The sequence window length for Mamba (default 2300).
            vol_window (int): Rolling window size for standard deviation of log returns.
            chunk_size (int): Size of chronological database records to process at once.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.window_size = window_size
        self.vol_window = vol_window
        self.db_fetch_batch = db_fetch_batch
        self.split_type = split_type
        self.event_type_map = {"mod_liquidity": float(0.0), "swap": float(1.0)}

        # Calculate total available pool events
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(id) FROM {self.table_name} WHERE split = {self.split_type}")
        self.total_records = cursor.fetchone()[0]
        conn.close()

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch a large chunk chronologically
        query = f"""
                    SELECT block_ts_utc AS block_ts_utc, event_type AS event_type, COALESCE(sqrt_price_x96, '0.0') AS sqrt_price_x96, 
                    COALESCE(liquidity, '0.0') AS target_liquidity, COALESCE(tick_upper, '0.0') AS tick_upper, COALESCE(tick_lower, '0.0') AS tick_lower, 
                    COALESCE(delta_liquidity, '0.0') AS delta_liquidity, COALESCE(amount0, '0.0') AS amount0, COALESCE(amount1, '0.0') AS amount1 
                    FROM {self.table_name}
                    WHERE split = {self.split_type}
                    ORDER BY id ASC
        """
        query_val = f"""
                    SELECT block_ts_utc AS block_ts_utc, event_type AS event_type, COALESCE(sqrt_price_x96, '0.0') AS sqrt_price_x96, 
                    COALESCE(liquidity, '0.0') AS target_liquidity, COALESCE(tick_upper, '0.0') AS tick_upper, COALESCE(tick_lower, '0.0') AS tick_lower, 
                    COALESCE(delta_liquidity, '0.0') AS delta_liquidity, COALESCE(amount0, '0.0') AS amount0, COALESCE(amount1, '0.0') AS amount1 
                    FROM {self.table_name}
                    WHERE split = {self.split_type}
                    ORDER BY id ASC
        """

        ## GROUP BY strftime('%Y-%m-%d %H:00:00', block_ts_utc)
        query = query if (self.split_type == 0) else query_val
        cursor.execute(query)

        # Historical memory elements to preserve continuity across db_fetch_batch pagination
        history_buffer = []
        last_log_S = None
        returns_history = []

        while True:
            # Low-overhead raw tuple extraction 
            rows = cursor.fetchmany(self.db_fetch_batch)
            if not rows:
                break

            # Convert raw tuples to isolated NumPy structural column vectors
            # This completely avoids heavy pandas overhead
            column_types = np.dtype([
                ('block_ts_utc', 'S200'),  # The string column
                ('event_type', 'S100'),  # An integer column
                ('sqrt_price_x96', 'f8'),    # A decimal/float column
                ('liquidity', 'f4'),
                ('tick_upper', 'f4'),
                ('tick_lower', 'f4'),
                ('delta_liquidity', 'f4'),
                ('amount0', 'f4'),
                ('amount1', 'f4'),
            ])

            raw_matrix = np.array(rows, dtype=column_types)
            
            block_ts_utc = raw_matrix['block_ts_utc']
            event_types_raw = raw_matrix['event_type']
            event_types = []
            for v in event_types_raw:
                if v == 'mod_liquidity':
                    event_types.append(0.0)
                else:
                    event_types.append(1.0)
            sqrt_price_x96 = raw_matrix['sqrt_price_x96']
            target_liq = raw_matrix['liquidity']
            tick_u = raw_matrix['tick_upper']
            tick_l = raw_matrix['tick_lower']
            delta_liq = raw_matrix['delta_liquidity']
            amt0 = raw_matrix['amount0']
            amt1 = raw_matrix['amount1']

            # --- Vectorized Physics Math Transformations ---
            current_prices = (sqrt_price_x96 / (2**96)) ** 2
            log_S = np.log(sqrt_price_x96)

            # Compute sequential log returns across batches flawlessly
            if last_log_S is not None:
                padded_log_S = np.concatenate([[last_log_S], log_S])
                target_returns = 2.0 * np.diff(padded_log_S)
            else:
                target_returns = 2.0 * np.diff(np.concatenate([[log_S[0]], log_S]))
                
            last_log_S = log_S[-1]

            # --- Continuous Rolling Volatility Vectorization ---
            # Maintain a strict historical list of returns for the rolling window calculation
            all_returns = returns_history + target_returns.tolist()

            target_vols = []
            
            # Calculate rolling standard deviations efficiently
            # Uses a manual fallback loop for the boundary window to preserve execution speed
            start_idx = len(returns_history)
            for i in range(start_idx, len(all_returns)):
                window_slice = all_returns[max(0, i - self.vol_window + 1) : i + 1]
                if len(window_slice) > 1:
                    target_vols.append(np.std(window_slice, ddof=1))
                else:
                    target_vols.append(0.02) # Baseline standard deviation warmup

            # Cache the tail end of the returns for the upcoming cursor fetch cycle
            returns_history = all_returns[-(self.vol_window - 1):] if len(all_returns) >= self.vol_window else all_returns
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
            for t in range(len(rows)):
                history_buffer.append({
                    "block_ts_utc": block_ts_utc[t],
                    "event_types": event_types[t],
                    "modify_seq": np.array([tick_l[t], tick_u[t], delta_liq[t]], dtype=np.float32),
                    "stable_swap_seq": np.array([stable_amount0[t], stable_amount1[t]], dtype=np.float32),
                    "raw_swap_seq": np.array([amt0[t], amt1[t]], dtype=np.float32),
                    "current_prices": current_prices[t].astype(np.float32),
                    "target_returns": target_returns[t].astype(np.float32),
                    "target_vols": target_vols_scaled[t],
                    "target_log_l": target_log_l[t]
                })

                # The moment our FIFO rolling queue matches our Mamba window, we slice and yield
                if len(history_buffer) == self.window_size:
                    yield {
                        "block_ts_utc": np.array([x["block_ts_utc"] for x in history_buffer], dtype=str),
                        "event_types": np.array([x["event_types"] for x in history_buffer], dtype=np.int64),
                        "modify_seq": np.stack([x["modify_seq"] for x in history_buffer], axis=0),
                        "stable_swap_seq": np.stack([x["stable_swap_seq"] for x in history_buffer], axis=0),
                        "raw_swap_seq": np.stack([x["raw_swap_seq"] for x in history_buffer], axis=0),
                        "current_prices": np.array([x["current_prices"] for x in history_buffer], dtype=np.float32),
                        "target_returns": np.array([x["target_returns"] for x in history_buffer], dtype=np.float32),
                        "target_vols": np.array([x["target_vols"] for x in history_buffer], dtype=np.float32),
                        "target_log_l": np.array([x["target_log_l"] for x in history_buffer], dtype=np.float32),
                    }
                    # Clear out the old elements to free memory space
                    history_buffer = []
        conn.close()

def pinn_sequence_collate(batch):
    """
    Collates dynamic dictionaries directly into the structured matrix 
    dimensions required by HardenedPINNDirectionalMamba.
    """
    return {
        "block_ts_utc": np.array([item["block_ts_utc"] for item in batch], dtype=str),
        "event_types": torch.from_numpy(np.array([item['event_types'] for item in batch], dtype=np.int64)),
        "modify_seq": torch.from_numpy(np.array([item['modify_seq'] for item in batch], dtype=np.float32)),
        "stable_swap_seq": torch.from_numpy(np.array([item['stable_swap_seq'] for item in batch], dtype=np.float32)),
        "raw_swap_seq": torch.from_numpy(np.array([item['raw_swap_seq'] for item in batch], dtype=np.float32)),
        "current_prices": torch.from_numpy(np.array([item['current_prices'] for item in batch], dtype=np.float32)),
        "target_returns": torch.from_numpy(np.array([item['target_returns'] for item in batch], dtype=np.float32)),
        "target_vols": torch.from_numpy(np.array([item['target_vols'] for item in batch], dtype=np.float32)),
        "target_log_l": torch.from_numpy(np.array([item['target_log_l'] for item in batch], dtype=np.float32)),
    }

def create_sliding_window_sequence_dataloaders(db_path, table_name="timeline", window_size=2300, vol_window=30, train_chunk_size=10000, val_chunk_size=10000):
    train_dataset = SlidingWindowVolatilityStreamer(db_path=db_path, table_name=table_name, split_type=0, window_size=window_size, vol_window=vol_window, db_fetch_batch=train_chunk_size)
    val_dataset = SlidingWindowVolatilityStreamer(db_path=db_path, table_name=table_name, split_type=1, window_size=window_size, vol_window=vol_window, db_fetch_batch=val_chunk_size)
    BATCH_SIZE = 4

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, # Total tensors per batch: 8 batches x 2,300 elements = 18,400 steps
        collate_fn=pinn_sequence_collate,  # Use our clean collate function
        num_workers=0,       # Spreads the 1M rows over 4 parallel database streamers
    )

    val_loader = DataLoader(
        val_dataset, 
        collate_fn=pinn_sequence_collate,  # Use our clean collate function
        batch_size=BATCH_SIZE, 
        num_workers=0, 
    )

    #dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
    return train_loader, val_loader


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, source_dataset, buffer_size=10000):
        super().__init__()
        self.source_dataset = source_dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        # Obtain the iterator from the underlying stream
        source_iter = iter(self.source_dataset)
        buffer = []

        # Fill the buffer up to buffer_size
        try:
            for _ in range(self.buffer_size):
                buffer.append(next(source_iter))
        except StopIteration:
            # If dataset is smaller than buffer_size, safely catch it
            pass

        # Yield a random item and stream in a new one
        for item in source_iter:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            buffer[idx] = item  # Replace the yielded item with a fresh one

        # Flush the remaining items in the buffer when the stream ends
        random.shuffle(buffer)
        for item in buffer:
            yield item

def get_bootstrap_subset(dataset, fraction: float = 0.85):
    num_samples = int(len(dataset) * fraction)
    # Random indices without replacement per member
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    return Subset(dataset, indices)

def create_sliding_window_sequence_dataloaders_for_ensemble(db_path, table_name="timeline", window_size=2300, vol_window=30, train_chunk_size=10000, val_chunk_size=10000, member_seed=10):
    train_dataset = SlidingWindowVolatilityStreamer(db_path=db_path, table_name=table_name, split_type=0, window_size=window_size, vol_window=vol_window, db_fetch_batch=train_chunk_size)
    val_dataset = SlidingWindowVolatilityStreamer(db_path=db_path, table_name=table_name, split_type=1, window_size=window_size, vol_window=vol_window, db_fetch_batch=val_chunk_size)
    BATCH_SIZE = 4

    # Option 1: Shuffle the dataset
    train_dataset_mod = ShuffleIterableDataset(train_dataset, buffer_size=window_size)

    # Option 2: Train on 80–90% random sub-chunks of time-series sequences
    #train_dataset_mod = get_bootstrap_subset(train_dataset, fraction=0.90)
    
    g = torch.Generator()
    g.manual_seed(member_seed)

    train_loader = DataLoader(
        train_dataset_mod, 
        batch_size=BATCH_SIZE, # Total tensors per batch: 8 batches x 2,300 elements = 18,400 steps
        collate_fn=pinn_sequence_collate,  # Use our clean collate function
        num_workers=0,       # Spreads the 1M rows over 4 parallel database streamers
        generator=g
    )

    val_loader = DataLoader(
        val_dataset, 
        collate_fn=pinn_sequence_collate,  # Use our clean collate function
        batch_size=BATCH_SIZE, 
        num_workers=0, 
    )

    return train_loader, val_loader
