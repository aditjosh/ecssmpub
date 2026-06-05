import math
from web3 import Web3
import json
import sqlite3
import time
import constants
import requests
from tqdm import tqdm
from eth_abi import decode, encode
from eth_utils import to_normalized_address, event_abi_to_log_topic, decode_hex
from collections import defaultdict
import datetime
from decimal import Decimal

# Setup Connection to Archive Node
RPC_URL = constants.RPC_URL
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise ConnectionError("Failed to connect to the Ethereum node.")

# Canonical Addresses
POOL_MANAGER_ADDR = constants.UNI_POOL_MANAGER.lower()
POSITION_MANAGER_ADDR = constants.UNI_POSITION_MANAGER.lower()
CURRENCY_1_ADDR = constants.DAI_CONTRACT_ADDR.lower()
CURRENCY_1_DECIMAL_SCALE = constants.DAI_DECIMAL_SCALE
CURRENCY_2_ADDR = constants.USDT_CONTRACT_ADDR.lower()
CURRENCY_2_DECIMAL_SCALE = constants.USDT_DECIMAL_SCALE
DSCALE = CURRENCY_1_DECIMAL_SCALE - CURRENCY_2_DECIMAL_SCALE
C0, C1 = (CURRENCY_1_ADDR, CURRENCY_2_ADDR) if int(CURRENCY_1_ADDR, 16) < int(CURRENCY_2_ADDR, 16) else (CURRENCY_2_ADDR, CURRENCY_1_ADDR)

# Minimal ABIs for Event Decoding
POOL_MANAGER_ABI = constants.POOL_MANAGER_ABI
POSITION_MANAGER_ABI = constants.POSITION_MANAGER_ABI

# Instantiate Contracts
PM_CONTRACT = w3.eth.contract(address=Web3.to_checksum_address(POOL_MANAGER_ADDR), abi=POOL_MANAGER_ABI)
#POS_CONTRACT = w3.eth.contract(address=POSITION_MANAGER_ADDR, abi=POSITION_MANAGER_ABI)

# --- Helper Math Functions ---
def tick_to_price(tick):
    # For factor_decimal_diff see https://support.uniswap.org/hc/en-us/articles/7423663459597-Why-does-the-price-input-automatically-round
    return (1.0001 ** tick)

def calculate_amounts(sqrt_price_x96, tick_lower, tick_upper, liquidity):
    sqrt_p = sqrt_price_x96 / (2 ** 96)
    sqrt_p_l = math.sqrt(tick_to_price(tick_lower))
    sqrt_p_u = math.sqrt(tick_to_price(tick_upper))
    
    if liquidity == 0:
        return 0.0, 0.0

    if sqrt_p < sqrt_p_l:
        amount0 = liquidity * (1 / sqrt_p_l - 1 / sqrt_p_u)
        amount1 = 0.0
    elif sqrt_p > sqrt_p_u:
        amount0 = 0.0
        amount1 = liquidity * (sqrt_p_u - sqrt_p_l)
    else:
        amount0 = liquidity * (1 / sqrt_p - 1 / sqrt_p_u)
        amount1 = liquidity * (sqrt_p - sqrt_p_l)
        
    # Scale for token decimals
    return amount0 / CURRENCY_1_DECIMAL_SCALE, amount1 / CURRENCY_2_DECIMAL_SCALE

# --- Database
class DataExtraction:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()

    def init_database_pool_ids(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # 1. Pool Registry Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pools (
                pool_id TEXT PRIMARY KEY,
                currency1 TEXT,
                currency2 TEXT,
                static_base_fee INTEGER,
                is_dynamic INTEGER,
                start_block INTEGER,
                end_block INTEGER
            )
        """)
        self.conn.commit()

    def init_database_fees_liquidity(self):
        """
        Initializes the SQLite database tables and creates the indexing strategy.
        """
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        # Historical Fees and Prices Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fee_and_price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pool_id TEXT,
                tx_hash TEXT,
                block_number INTEGER,
                sender TEXT,
                recipient TEXT,
                amount0 TEXT,
                amount1 TEXT,
                fee_pips INTEGER,
                sqrt_price_x96 TEXT, -- Kept as text to safely store uint160 size
                liquidity TEXT,
                tick TEXT,
                gas_fee_wei TEXT, 
                gas_fee_eth TEXT,
                FOREIGN KEY(pool_id) REFERENCES pools(pool_id)
            )
        """)
    
        # Consolidated Liquidity Positions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS liquidity_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pool_id TEXT,
                owner_address TEXT,
                tx_hash TEXT,
                block_number INTEGER, 
                block_ts TEXT,
                block_ts_utc TEXT,
                tick_lower INTEGER,
                tick_upper INTEGER,
                delta_liquidity TEXT, -- Kept as text to support wide int256 states
                token1_balance REAL,
                token2_balance REAL,
                gas_fee_wei TEXT, 
                gas_fee_eth TEXT,
                sqrt_price_x96 TEXT,
                FOREIGN KEY(pool_id) REFERENCES pools(pool_id)
            )
        """)

        # --- Performance Indexing Strategy ---
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_lookup ON fee_and_price_history(pool_id, block_number);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_position_owner ON liquidity_positions(pool_id, owner_address);")
        self.conn.commit()

    def save_pool_metadata(self, insert_tuple):
        """
        Saves or updates pool configuration details.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO pools (pool_id, currency1, currency2, static_base_fee, is_dynamic, start_block, end_block)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, insert_tuple)
        self.conn.commit()
        print("Pool metadata written to SQLite database.")

    def save_fees_and_prices(self, insert_data):
        """
        Extracts updates from swap logs and writes them to the database in bulk.
        """
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT INTO fee_and_price_history (pool_id, tx_hash, block_number, sender, recipient, amount0,
                        amount1, fee_pips, sqrt_price_x96, liquidity, tick, gas_fee_wei, gas_fee_eth)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_data)
        self.conn.commit()
        print(f"Successfully inserted {len(insert_data)} history rows into fee_and_price_history.")
        

    def save_positions_and_amounts(self, insert_data):
        """
        Extracts raw ledger events, computes financial balances, and stores positions.
        """
        cursor = self.conn.cursor()
        # Clear old calculations to ensure our snapshot stays accurate
        #cursor.execute("DELETE FROM liquidity_positions WHERE pool_id = ?", (pool_id,))
        
        cursor.executemany("""
            INSERT INTO liquidity_positions (pool_id, owner_address, tx_hash, block_number, block_ts, block_ts_utc, 
                            tick_lower, tick_upper, delta_liquidity, token1_balance, token2_balance, gas_fee_wei, 
                            gas_fee_eth, sqrt_price_x96)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_data)
        self.conn.commit()
        print(f"Successfully synchronized {len(insert_data)} records into liquidity_positions.")

    def select_pool_ids(self, db_path):
        this_conn = sqlite3.connect(db_path)
        cursor = this_conn.cursor()
        SELECT_QUERY = """SELECT pool_id, is_dynamic, start_block, end_block
                        FROM pools;
                        """
        cursor.execute(SELECT_QUERY)
        ## list of tuples
        list_rows = cursor.fetchall()
        return list_rows

# --- DATA CURATION ENGINE ---
def curate_pool_metadata(data_extractor, start_block, end_block, chunk_size=50000):
    """
    Finds the PoolId and base static fee for the currency1/currency2 pair.
    """
    print("Discovering Pool ID and static configurations...")
    # Pad addresses to 32 bytes for topic matching
    topic_c0 = "0x" + C0[2:].zfill(64).lower()
    topic_c1 = "0x" + C1[2:].zfill(64).lower()
    
    logs = None

    pools_data = []
    unique_pool_ids = set()
    for current_start in tqdm(range(start_block, end_block, chunk_size), desc="Scanning pools metadata"):
        current_end = min(current_start + chunk_size, end_block)

        try:
            logs = w3.eth.get_logs({
                "fromBlock": current_start,
                "toBlock": current_end,
                "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
                "topics": [
                    w3.keccak(text="Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)").to_0x_hex(),
                    None,
                    topic_c0,
                    topic_c1
                ]
            })
    
            if not logs:
                print("\nNo initialization log found for this pair in the provided block range.")
            else:
                for log in logs:
                    decoded = PM_CONTRACT.events.Initialize().process_log(log)
                    pool_id = "0x" + str(decoded['args']['id'].hex())
                    static_fee = decoded['args']['fee']
                    clean_fee = static_fee & 0x7FFFFF
                    is_dynamic = bool(static_fee & 0x800000)
                    print(f"-> Pool ID Found for currency1/currency2: {pool_id}")

                    if not pool_id in unique_pool_ids:
                        unique_pool_ids.add(pool_id)
                        pools_data.append((pool_id, CURRENCY_1_ADDR, CURRENCY_2_ADDR, clean_fee, is_dynamic, current_start, current_end))

        except Exception as e:
            print(f"\ncurate_pool_metadata: Timeout or error at block {current_start}: {e}. Retrying with smaller chunk...")
            # If a specific chunk fails, you could recursively call this with chunk_size // 2
            time.sleep(1)
            continue
    
    for insert_data in pools_data:
        data_extractor.save_pool_metadata(insert_data)
    return pools_data

def curate_fees_and_prices(data_extractor, pool_id, start_block, end_block, chunk_size=50000):
    """
    (2) Tracks Static and Dynamic Fees along with pool prices from Swap logs
    """
    fee_timeline = []
    for current_start in tqdm(range(start_block, end_block, chunk_size), desc=f"Scanning fees and prices for pool id: {pool_id}"):
        current_end = min(current_start + chunk_size, end_block)

        try:
            swap_signature_hash = w3.keccak(text="Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)").to_0x_hex()
            logs = w3.eth.get_logs({
                "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
                "fromBlock": current_start, # Update to your starting block
                "toBlock": current_end,
                "topics": [
                    swap_signature_hash,
                    pool_id # Topic 1 filters by indexed PoolId
                ]
            })
            
            if len(logs) == 0:
                #print(f"No swap logs found for pool id: {pool_id}!!")
                continue
            else:
                ##print(f"\nFound swap logs for pool id: {pool_id}")
                for log in logs:
                    decoded_log = PM_CONTRACT.events.Swap().process_log(log)
                    args = decoded_log['args']
                    #sender = args['sender'] if 'sender' in args else None
                    block_number = decoded_log['blockNumber']
                    # The 24th bit flags dynamic behavior, mask it to get the active fee
                    current_fee = args['fee'] & 0x7FFFFF

                    # === Fetch Transaction Receipt for Gas Info ===
                    tx_hash = log['transactionHash'].hex()
                    receipt = w3.eth.get_transaction_receipt(tx_hash)
                    tx = w3.eth.get_transaction(tx_hash)
                    gas_used = receipt['gasUsed']
                    # Handle both legacy and EIP-1559 transactions
                    if 'gasPrice' in tx and tx['gasPrice'] is not None:
                        gas_price_wei = tx['gasPrice']
                    else:
                        # EIP-1559
                        gas_price_wei = receipt.get('effectiveGasPrice') or tx.get('maxFeePerGas', 0)

                    gas_fee_wei = gas_used * gas_price_wei
                    gas_fee_eth = Web3.from_wei(gas_fee_wei, 'ether')

                    sender = "0x" + str(log['topics'][1].hex()[-40:])
                    recipient = "0x" + str(log['topics'][2].hex()[-40:])

                    # Decode un-indexed parameters
                    # Non-indexed parameter types to pass to Web3.py
                    unindexed_types = ['int256', 'int256', 'uint160', 'uint128', 'int24']
                    decoded_data = decode(unindexed_types, log['data'])
                    amount0, amount1, sqrtPriceX96, liquidity, tick = decoded_data
                    
                    # Explicitly pass block_identifier to pinpoint the exact state at that specific block
                    #try:
                    #    pool_id_bytes = Web3.to_bytes(hexstr=pool_id)
                    #    slot0 = PM_CONTRACT.functions.getSlot0(pool_id_bytes).call(block_identifier=block_number)
                    #    active_liquidity = PM_CONTRACT.functions.getLiquidity(pool_id_bytes).call(block_identifier=block_number)
                    #    manager_sqrt_price = str(slot0[0])
                    #    manager_tick = slot0[1]
                    #except Exception as e:
                    #    if "execution reverted" in str(e):
                    #        print(f"Pool didn't exist yet at block {block_number}. Skipping.")
                    #        continue  # Move to the next block/event
                    
                    #print(f"Found decoded log for pool id: {pool_id}")
                    fee_timeline.append((
                        pool_id,  # 0
                        tx_hash,
                        block_number,  # 2
                        sender,
                        recipient,
                        str(amount0),
                        str(amount1),
                        current_fee,
                        str(sqrtPriceX96),  # 8
                        str(liquidity),
                        str(tick),
                        str(gas_fee_wei),
                        str(gas_fee_eth),
                        #str(active_liquidity) if active_liquidity is not None else None,
                        #str(manager_sqrt_price),
                        #manager_tick
                    ))
        except Exception as e:
                print(f"\ncurate_fees_and_prices: Timeout or error at block {current_start}: {e}. Retrying with a pause...")
                # If a specific chunk fails, you could recursively call this with chunk_size // 2
                time.sleep(1)
                continue

    if len(fee_timeline) > 0:
        data_extractor.save_fees_and_prices(fee_timeline)
        print(f"Processed {len(fee_timeline)} swap/fee entries in pool id: {pool_id}")
    else:
        print(f"No swap/fee entries in pool id: {pool_id}")
    return fee_timeline

def curate_positions_and_amounts(data_extractor, pool_id, latest_price, start_block, end_block, chunk_size=50000):
    """
    (1) & (3) Tracks individual liquidity positions and computes token values
    """
    pos_amounts = []
    #print("Curating liquidity positions and calculating active balances...")
    for current_start in tqdm(range(start_block, end_block, chunk_size), desc="Curating positions and amounts"):
        current_end = min(current_start + chunk_size, end_block)

        pool_id_topic = "0x" + pool_id[2:].zfill(64).lower() if pool_id.startswith("0x") else "0x" + pool_id.zfill(64).lower()
        modify_signature_hash = w3.keccak(text="ModifyLiquidity(bytes32,address,int24,int24,int256,bytes32)").to_0x_hex()
        
        logs = w3.eth.get_logs({
            "fromBlock": current_start,
            "toBlock": current_end,
            "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
            "topics": [
                modify_signature_hash, # Topic 0
                pool_id_topic          # Topic 1 (indexed 'id' field)
                #None
            ]
        })
    
        # Track book-kept active liquidity per unique position key
        # Key: (sender, tickLower, tickUpper)
        active_positions = {}
    
        for log in logs:
            decoded_log = PM_CONTRACT.events.ModifyLiquidity().process_log(log)
            args = decoded_log['args']
            tx_hash = log['transactionHash'].hex() 
            block_number = log['blockNumber']
            block_ts = log['blockTimestamp']
            timestamp_int = int(block_ts, 16)
            #block_ts_utc = datetime.datetime.utcfromtimestamp(timestamp_int).isoformat()
            block_ts_utc = datetime.datetime.fromtimestamp(timestamp_int, datetime.UTC).isoformat()
            sender = args['sender']
            pos_key = (sender, tx_hash, block_number, block_ts, block_ts_utc, args['tickLower'], args['tickUpper'])
            
            if pos_key not in active_positions:
                active_positions[pos_key] = 0
                
            active_positions[pos_key] += args['liquidityDelta']
        
        # Resolve ownership from PositionManager if the sender is the peripheral contract
        for (sender, tx_hash, block_number, block_ts, block_ts_utc, tick_l, tick_u), active_liq in active_positions.items():
            if active_liq <= 0:
                continue # Skip fully drained or empty positions
            
            owner = sender
            # If wrapped by the canonical PositionManager, trace ownership via ERC721 logic
            if sender == POSITION_MANAGER_ADDR:
                # Trace back transfers to find who owns this specific interaction position
                # (In production, map transaction hash logs to the specific Transfer event)
                owner = f"PositionManager Wrapped Asset (Lookup wallet via ERC721 TokenId)"

            # Explicitly pass block_identifier to pinpoint the exact state at that specific block
            #pool_id_bytes = Web3.to_bytes(hexstr=pool_id)
            #slot0 = PM_CONTRACT.functions.getSlot0(pool_id_bytes).call(block_identifier=block_number)
            #active_liquidity = PM_CONTRACT.functions.getLiquidity(pool_id_bytes).call(block_identifier=block_number)
            #manager_sqrt_price = str(slot0[0])
            #manager_tick = slot0[1]

            # Compute underlying token balances
            curr1_bal, curr2_bal = calculate_amounts(float(latest_price), tick_l, tick_u, active_liq)

            # === Fetch Transaction Receipt for Gas Info ===
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            tx = w3.eth.get_transaction(tx_hash)
            gas_used = receipt['gasUsed']
            # Handle both legacy and EIP-1559 transactions
            if 'gasPrice' in tx and tx['gasPrice'] is not None:
                gas_price_wei = tx['gasPrice']
            else:
                # EIP-1559
                gas_price_wei = receipt.get('effectiveGasPrice') or tx.get('maxFeePerGas', 0)

            gas_fee_wei = gas_used * gas_price_wei
            gas_fee_eth = Web3.from_wei(gas_fee_wei, 'ether')
        
            pos_amounts.append((
                pool_id,
                owner,
                tx_hash,
                block_number, 
                block_ts,
                block_ts_utc,
                tick_l,
                tick_u,
                str(active_liq),
                curr1_bal,
                curr2_bal,
                str(gas_fee_wei),
                str(gas_fee_eth),
                str(latest_price),
                #str(active_liquidity) if active_liquidity is not None else None,
                #str(manager_sqrt_price),
                #manager_tick
            ))
        
    data_extractor.save_positions_and_amounts(pos_amounts)

def curate_positions_and_amounts_at_block(data_extractor, pool_id, block_number, latest_price):
    """
    (1) & (3) Tracks individual liquidity positions and computes token values
    """
    pos_amounts = []
    #print("Curating liquidity positions and calculating active balances...")
    pool_id_topic = "0x" + pool_id[2:].zfill(64).lower() if pool_id.startswith("0x") else "0x" + pool_id.zfill(64).lower()
    modify_signature_hash = w3.keccak(text="ModifyLiquidity(bytes32,address,int24,int24,int256,bytes32)").to_0x_hex()
        
    logs = w3.eth.get_logs({
        "fromBlock": block_number,
        "toBlock": block_number,
        "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
        "topics": [
            modify_signature_hash, # Topic 0
            pool_id_topic          # Topic 1 (indexed 'id' field)
            #None
        ]
    })
    
    # Track book-kept active liquidity per unique position key
    # Key: (sender, tickLower, tickUpper)
    active_positions = {}

    for log in logs:
        decoded_log = PM_CONTRACT.events.ModifyLiquidity().process_log(log)
        args = decoded_log['args']
        tx_hash = log['transactionHash'].hex() 
        block_number = log['blockNumber']
        block_ts = log['blockTimestamp']
        timestamp_int = int(block_ts, 16)
        #block_ts_utc = datetime.datetime.utcfromtimestamp(timestamp_int).isoformat()
        block_ts_utc = datetime.datetime.fromtimestamp(timestamp_int, datetime.UTC).isoformat()
        sender = args['sender']
        pos_key = (sender, tx_hash, block_number, block_ts, block_ts_utc, args['tickLower'], args['tickUpper'])
        
        if pos_key not in active_positions:
            active_positions[pos_key] = 0
            
        active_positions[pos_key] += args['liquidityDelta']

    # Resolve ownership from PositionManager if the sender is the peripheral contract
    for (sender, tx_hash, block_number, block_ts, block_ts_utc, tick_l, tick_u), active_liq in active_positions.items():
        if active_liq <= 0:
            continue # Skip fully drained or empty positions
        
        owner = sender
        # If wrapped by the canonical PositionManager, trace ownership via ERC721 logic
        if sender == POSITION_MANAGER_ADDR:
            # Trace back transfers to find who owns this specific interaction position
            # (In production, map transaction hash logs to the specific Transfer event)
            owner = f"PositionManager Wrapped Asset (Lookup wallet via ERC721 TokenId)"

        # Compute underlying token balances
        curr1_bal, curr2_bal = calculate_amounts(float(latest_price), tick_l, tick_u, active_liq)

        # === Fetch Transaction Receipt for Gas Info ===
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        tx = w3.eth.get_transaction(tx_hash)
        gas_used = receipt['gasUsed']
        # Handle both legacy and EIP-1559 transactions
        if 'gasPrice' in tx and tx['gasPrice'] is not None:
            gas_price_wei = tx['gasPrice']
        else:
            # EIP-1559
            gas_price_wei = receipt.get('effectiveGasPrice') or tx.get('maxFeePerGas', 0)

        gas_fee_wei = gas_used * gas_price_wei
        gas_fee_eth = Web3.from_wei(gas_fee_wei, 'ether')

        # Explicitly pass block_identifier to pinpoint the exact state at that specific block
        #pool_id_bytes = Web3.to_bytes(hexstr=pool_id)
        #slot0 = PM_CONTRACT.functions.getSlot0(pool_id_bytes).call(block_identifier=block_number)
        #active_liquidity = PM_CONTRACT.functions.getLiquidity(pool_id_bytes).call(block_identifier=block_number)
        #manager_sqrt_price = str(slot0[0])
        #manager_tick = slot0[1]

        pos_amounts.append((
            pool_id,
            owner,
            tx_hash,
            block_number, 
            block_ts,
            block_ts_utc,
            tick_l,
            tick_u,
            str(active_liq),
            curr1_bal,
            curr2_bal,
            str(gas_fee_wei),
            str(gas_fee_eth),
            str(latest_price),
            #str(active_liquidity) if active_liquidity is not None else None,
            #str(manager_sqrt_price),
            #manager_tick
        ))
        
        if len(pos_amounts) > 0:
            data_extractor.save_positions_and_amounts(pos_amounts)

#### View states
def call_json_rpc(cp_w3, pool_id, function_name, params=None):
    #ALCHEMY_KEY = "jRqSpoOhZAkGhTIjefdr-"
    #RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/" + ALCHEMY_KEY
    RPC_URL = constants.RPC_URL

    POOL_MANAGER = constants.UNI_POOL_MANAGER  # mainnet
    state_view_addr = Web3.to_checksum_address(constants.UNI_V4_STATE_VIEW_CONTRACT)   # mainnet
    ## Use the StateView contract (a read-only helper deployed by Uniswap).
    #state_view = cp_w3.eth.contract(address=state_view_addr, abi=constants.STATEVIEW_ABI)
    data = None
    if function_name == "getSlot0":
        selector = cp_w3.keccak(text="getSlot0(bytes32)")[:4].hex()
        data = "0x" + selector + pool_id[2:].zfill(64)
    elif function_name == "getTickInfo":
        param_types = ["bytes32", "int24"]
        encoded_params = cp_w3.codec.encode(param_types, params)
        selector = cp_w3.keccak(text=f"getTickInfo({','.join(param_types)})")[:4]
        data = "0x" + (selector + encoded_params).hex()
    elif function_name == "getTickBitmap":
        param_types = ["bytes32", "int16"]
        encoded_params = cp_w3.codec.encode(param_types, params)
        selector = cp_w3.keccak(text=f"getTickBitmap({','.join(param_types)})")[:4]
        data = "0x" + (selector + encoded_params).hex()

    params = [{"to": state_view_addr, "data": data}, "latest"]

    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": params,
        "id": 1
    }
    response = requests.post(RPC_URL, json=payload).json()
    return response

def check_pool_exists(cp_w3, pool_id):
    try:
        response = call_json_rpc(cp_w3, pool_id, "getSlot0")
        result = response['result']
        if result and len(result) > 2:
            decoded = cp_w3.codec.decode(
                ['uint160', 'int24', 'uint24', 'uint24'], 
                bytes.fromhex(result[2:])
            )
            sqrt_price, tick, protocol_fee, lp_fee = decoded

            print("Pool exists!")
            print("sqrtPriceX96:", sqrt_price)
            print("Tick:", tick)
            print("Protocol fee: ", protocol_fee)
            print("Liquidity:", lp_fee)
        return result
    except Exception as e:  # typically reverts or returns zero values if not initialized
        print("Pool does NOT exist or not initialized:", e)

def get_tick_info(cp_w3, pool_id, params):
    try:
        response = call_json_rpc(cp_w3, pool_id, "getTickInfo", params)
        result = response['result']
        if result and len(result) > 2:
            decoded = cp_w3.codec.decode(
                ['uint128', 'int128', 'uint256', 'uint256'], 
                bytes.fromhex(result[2:])
            )
            liquidity_gross, liquidity_net, fg0, fg1 = decoded

            print("liquidity_gross:", liquidity_gross)
            print("liquidity_net:", liquidity_net)
            print("feeGrowthOutside0X128: ", fg0)
            print("feeGrowthOutside1X128:", fg1)
        
    except Exception as e:  # typically reverts or returns zero values if not initialized
        print("Pool does NOT exist or not initialized:", e)

def get_tick_bitmap(cp_w3, pool_id, params):
    try:
        response = call_json_rpc(cp_w3, pool_id, "getTickBitmap", params)
        result = response['result']
        if result and len(result) > 2:
            decoded = cp_w3.codec.decode(
                ['uint256'], 
                bytes.fromhex(result[2:])
            )
            tick = decoded
            #print("Tick:", tick)
        return result
    except Exception as e:  # typically reverts or returns zero values if not initialized
        print("Pool does NOT exist or not initialized:", e)

def call_state_view(cp_w3, function_name, param_types, params):
    selector = w3.keccak(text=f"{function_name}({','.join(param_types)})")[:4]
    encoded_params = cp_w3.codec.encode(param_types, params)
    data = "0x" + (selector + encoded_params).hex()
    state_view_addr = Web3.to_checksum_address(constants.UNI_V4_STATE_VIEW_CONTRACT)
    
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": state_view_addr, "data": data}, "latest"],
        "id": 1
    }
    response = requests.post(RPC_URL, json=payload).json()
    return response.get("result")

def get_initialized_ticks(cp_w3, pool_id: str, current_tick: int, words_left: int = 200, words_right: int = 200):
    initialized = []
    min_word = (current_tick >> 8) - words_left
    max_word = (current_tick >> 8) + words_right
    
    print(f"Scanning bitmap from word {min_word} to {max_word} ...")
    
    for word_pos in range(min_word, max_word + 1):
        params = [pool_id, word_pos]
        response = call_json_rpc(cp_w3, pool_id, "getTickBitmap", params)
        result = response['result']
        if result == "0x" or int(result, 16) == 0:
            continue
            
        bitmap = int(result, 16)
        
        for bit in range(256):
            if bitmap & (1 << bit):
                tick = (word_pos << 8) + bit
                # Fetch tick info
                response = call_json_rpc(cp_w3, pool_id, "getTickInfo", [pool_id, tick])
                tick_result = response['result']
                decoded = w3.codec.decode(
                    ['uint128', 'int128', 'uint256', 'uint256'],
                    bytes.fromhex(tick_result[2:])
                )
                liquidity_gross, liquidity_net, fg0, fg1 = decoded
                
                initialized.append({
                    "tick": tick,
                    "liquidityGross": liquidity_gross,
                    "liquidityNet": liquidity_net,
                    "feeGrowthOutside0X128": fg0,
                    "feeGrowthOutside1X128": fg1
                })
    
    return initialized

def find_liquidity_removal_time(pool_id: str, from_block, to_block, chunk_size=2000):
    """Find the most recent transaction that removed liquidity to zero"""
    cp_w3 = Web3(Web3.HTTPProvider(RPC_URL))

    # ModifyLiquidity event signature
    event_signature = "ModifyLiquidity(bytes32,address,int24,int24,int256,bytes32)"
    topic0 = w3.keccak(text=event_signature)

    removal_tx = None
    for current_start in tqdm(range(from_block, to_block, chunk_size), desc="Scanning for liq removal"):
        current_end = min(current_start + chunk_size, to_block)
        logs = w3.eth.get_logs({
            "fromBlock": current_start,
            "toBlock": current_end,
            "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
            "topics": [topic0, pool_id]   # topic0 + indexed PoolId
            ##"topics": [topic0, None]   # topic0 + indexed PoolId
        })

        if removal_tx:
            break
        
        for log in sorted(logs, key=lambda x: x['blockNumber'], reverse=True):
            data = log['data'][2:] if isinstance(log['data'], str) else log['data'].hex()
            if len(data) < 64*5:  # 5 parameters × 32 bytes
                continue

            # Decode event
            decoded = cp_w3.codec.decode(
                ['address', 'int24', 'int24', 'int256', 'bytes32'],
                bytes.fromhex(data)
            )
            sender, tickLower, tickUpper, liquidity_delta, salt = decoded
            
            print(f"Liquidity Delta: {liquidity_delta}")
            if liquidity_delta < 0:  # Liquidity was removed
                block = cp_w3.eth.get_block(log['blockNumber'])
                timestamp = block['timestamp']
                
                # Optional: Check current liquidity to confirm it went to zero
                current_liq = int(call_state_view("getLiquidity", ["bytes32"], [pool_id]), 16)
                
                print(f"   Liquidity removal detected!")
                print(f"   Block     : {log['blockNumber']}")
                #print(f"   Time      : {datetime.datetime.utcfromtimestamp(timestamp)} UTC")
                print(f"   Time      : {datetime.datetime.fromtimestamp(timestamp, datetime.UTC).isoformat()} UTC")
                print(f"   Delta     : {liquidity_delta}")
                print(f"   Sender    : {sender}")
                print(f"   Current Liquidity: {current_liq}")
                
                removal_tx = log['transactionHash'].hex()
                break  # Most recent one
    
    return removal_tx

def get_liquidity_providers(pool_id: str, from_block, to_block, chunk_size=2000):
    cp_w3 = Web3(Web3.HTTPProvider(RPC_URL))
    
    event_sig = "ModifyLiquidity(bytes32,address,int24,int24,int256,bytes32)"
    topic0 = cp_w3.keccak(text=event_sig)

    active = {}
    positions = {}
    for current_start in tqdm(range(from_block, to_block, chunk_size), desc="Scanning blocks"):
        current_end = min(current_start + chunk_size, to_block)
    
        logs = cp_w3.eth.get_logs({
            "fromBlock": current_start,
            "toBlock": current_end,
            "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
            "topics": [topic0, pool_id]
        })
                
        for log in logs:
            data = log['data'][2:] if isinstance(log['data'], str) else log['data'].hex()
            if len(data) < 64*5:  # 5 parameters × 32 bytes
                continue

            # Decode event
            decoded = cp_w3.codec.decode(
                ['address', 'int24', 'int24', 'int256', 'bytes32'],
                bytes.fromhex(data)
            )

            owner, tick_lower, tick_upper, delta, salt = decoded
            key = (owner, tick_lower, tick_upper, salt)
            salt_hex = salt.hex()

            if key not in positions:
                positions[key] = {
                    "owner": owner,
                    "tickLower": tick_lower,
                    "tickUpper": tick_upper,
                    "salt": salt_hex,
                    "liquidity": 0,
                    "last_block": None,
                    "last_tx": None,
                    "last_timestamp": None
                }
            positions[key]["liquidity"] += delta
            positions[key]["last_block"] = log['blockNumber']
            positions[key]["last_tx"] = log['transactionHash'].hex()
        
        # Filter active positions
        active = {k: v for k, v in positions.items() if v['liquidity']}
        
        print(f"Found {len(active)} active liquidity positions:")
        for pos in sorted(active.values(), key=lambda x: x["liquidity"], reverse=True):
            print(f"Wallet     : {pos['owner']}")
            print(f"Range      : [{pos['tickLower']}, {pos['tickUpper']}]")
            print(f"Liquidity  : {pos['liquidity']:,}")
            print(f"Last Block : {pos['last_block']}")
            print(f"Last Tx    : {pos['last_tx']}")
    
    return active

# --- EXECUTION RUNNER ---
def curate_positions(from_block, to_block, curr1_str, curr2_str, pool_id = None, pool_ids_db_file = None):
    # Example Blocks covering initialization/activity (Adjust for your target timeline)
    START_BLOCK =  from_block
    END_BLOCK = to_block

    # 25044360: considered latest
    # 24430360: 87 days ago
    # 23760360: 181 days ago
    # 23120360: 270 days ago
    # 22480360: 360 days ago
    # 21835360: 450 days ago
    # 21191360: 540 days ago
    if C0 == CURRENCY_1_ADDR:
        CURR1_CURR2 = f"{curr1_str}_{curr2_str}"
    else:
        CURR1_CURR2 = f"{curr2_str}_{curr1_str}"
    
    # Single pool_id specified
    if pool_id:
        pools_data = [[pool_id]]
        output_file_name = f'sqldb/{CURR1_CURR2}/swaps_{pool_id[:5]}_univ4_{CURR1_CURR2}_{START_BLOCK}_to_{END_BLOCK}.db'
        data_extractor = DataExtraction(output_file_name)
        data_extractor.init_database_fees_liquidity()
    else:
        output_file_name = f'sqldb/{CURR1_CURR2}/swaps_univ4_{CURR1_CURR2}_{START_BLOCK}_to_{END_BLOCK}.db'
        print(f"Output file name: {output_file_name}")
        data_extractor = DataExtraction(output_file_name)
        data_extractor.init_database_fees_liquidity()
        # Pool id file specified 
        if pool_ids_db_file:
            print("Found pools_ids file")
            pools_data = data_extractor.select_pool_ids(pool_ids_db_file)
        else:
            pools_data = curate_pool_metadata(data_extractor, start_block=START_BLOCK, end_block=END_BLOCK, chunk_size=1000)
        
    print(f"Number of pools: {len(pools_data)}")

    try:
        for pool in pools_data:
            pool_id = pool[0]

            # Gather Fee Configurations & Timeline
            swap_data = curate_fees_and_prices(data_extractor, pool_id, start_block=START_BLOCK, end_block=END_BLOCK, chunk_size=1000)
            
            # Extract modify liquidity events from the blocks corresponding to the swap events
            for swap in swap_data:
                block_number = swap[2]
                current_sqrt_price = swap[8]  # fallback template
            
                # Track Active Positions & calculate token allocations
                positions = curate_positions_and_amounts_at_block(data_extractor, pool_id, block_number=block_number, latest_price=current_sqrt_price)
        
            # Extract all modify liquidity events in the block range
            #positions = curate_positions_and_amounts(data_extractor, pool_id, start_block=START_BLOCK, end_block=END_BLOCK, chunk_size=1000)
        print("\nPipeline Execution Finished successfully.")
            
    except Exception as e:
        print(f"curate_positions: Error executing data extraction: {e}")

    finally:
        data_extractor.close()

def curate_pool_ids(from_block, to_block, curr1_str, curr2_str, chunk_size=1000):
    cp_w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if C0 == CURRENCY_1_ADDR:
        CURR1_CURR2 = f"{curr1_str}_{curr2_str}"
    else:
        CURR1_CURR2 = f"{curr2_str}_{curr1_str}"

    print(CURR1_CURR2)

    db_file_name = f'sqldb/{CURR1_CURR2}/pool_ids_univ4_{CURR1_CURR2}_{from_block}_to_{to_block}.db'
    
    try:
        data_extractor = DataExtraction(db_file_name)
        data_extractor.init_database_pool_ids()

        # Identify the Pool Id
        pools_data = curate_pool_metadata(data_extractor, start_block=from_block, end_block=to_block, chunk_size=chunk_size)
        print(f"Number of pools: {len(pools_data)}")
    except Exception as e:
        print(f"curate_pool_ids: Error executing data extraction: {e}")
    finally:
        data_extractor.close()


def decode_ticks_liquidity():
    ##pool_id = "0xf6e8088529094bc485561fa2a03e3d19c9a60f5d99a997e8fe16ab4ca2db277a"   #### works!
    cp_w3 = Web3(Web3.HTTPProvider(RPC_URL))
    pool_id = "0xa138ea2e4e2dd14d6de99cde73303640db3044b362ec3606ba8c18d91cac0c3f"
    slot0_result = check_pool_exists(cp_w3, pool_id)

    ##get_tick_info(cp_w3, pool_id, [pool_id, 0])

    ##word_pos = -198000 >> 8
    ##word = get_tick_bitmap(cp_w3, pool_id, [pool_id, word_pos])

    sqrt_price, tick, protocol_fee, lp_fee = cp_w3.codec.decode(
        ['uint160', 'int24', 'uint24', 'uint24'],
        bytes.fromhex(slot0_result[2:])
    )

    # Scan initialized ticks
    SCAN_WORDS_LEFT = 500   # ~76,800 ticks left
    SCAN_WORDS_RIGHT = 500  # ~76,800 ticks right
    ticks = get_initialized_ticks(cp_w3, pool_id, tick, SCAN_WORDS_LEFT, SCAN_WORDS_RIGHT)
    print(f"\nFound {len(ticks)} initialized ticks:")
    for t in sorted(ticks, key=lambda x: x["tick"]):
        print(f"  Tick {t['tick']:8d} | Gross {t['liquidityGross']:>20,} | Net {t['liquidityNet']:>20,}")

def is_pool_inactive(pool_id: str):
    cp_w3 = Web3()
    # Get Slot0
    slot0_result = call_state_view(cp_w3, "getSlot0", ["bytes32"], [pool_id])
    if slot0_result == "0x" or len(slot0_result) < 10:
        return True, "Not initialized"
    
    decoded = cp_w3.codec.decode(
        ['uint160', 'int24', 'uint24', 'uint24'],
        bytes.fromhex(slot0_result[2:])
    )
    sqrt_price, current_tick, _, _ = decoded
    
    if sqrt_price == 0:
        return True, "Not initialized (sqrtPriceX96 = 0)"
    
    # Get total liquidity
    liq_result = call_state_view(cp_w3, "getLiquidity", ["bytes32"], [pool_id])
    liquidity = int(liq_result, 16)
    
    if liquidity == 0:
        return True, "Zero active liquidity"
    
    # Optional: Check if there is any liquidity near current tick
    word_pos = current_tick >> 8
    bitmap_result = call_state_view(cp_w3, "getTickBitmap", ["bytes32", "int16"], [pool_id, word_pos])
    bitmap = int(bitmap_result, 16)
    
    if bitmap == 0:
        return False, "Has liquidity but no initialized ticks in current word (potentially low activity)"
    
    return False, f"Active (Liquidity: {liquidity})"


if __name__ == "__main__":
    ##status, msg = is_pool_inactive(pool_id)
    ##print(msg)

    # 25044360: considered latest
    # 24430360: 87 days ago
    # 23760360: 181 days ago
    # 23120360: 270 days ago
    # 22480360: 360 days ago
    # 21835360: 450 days ago
    # 21191360: 540 days ago

    to_block = 21835360 - 1
    from_block = 21191360
    #removal_tx = find_liquidity_removal_time(pool_id, from_block, to_block)
    #print(f"Liquidity removal transaction: {removal_tx}")
    ##active_positions = get_liquidity_providers(pool_id, from_block, to_block, chunk_size=2000)

    ## Curate liquidity ticks, swaps, etc.
    curr1_str = "dai" 
    curr2_str = "usdt"
    curate_positions(from_block, to_block, curr1_str, curr2_str, pool_ids_db_file = f"sqldb/dai_usdt/pool_ids_univ4_dai_usdt_21191360_to_25044360.db")

    
    #pool_id = "0xf6e8088529094bc485561fa2a03e3d19c9a60f5d99a997e8fe16ab4ca2db277a"
    ##curate_positions(from_block, to_block, curr1_str, curr2_str, pool_id = "0xf6e8088529094bc485561fa2a03e3d19c9a60f5d99a997e8fe16ab4ca2db277a")
    

    ## Curate pool ids only
    ##curate_pool_ids(from_block, to_block, curr1_str, curr2_str, chunk_size=2000)