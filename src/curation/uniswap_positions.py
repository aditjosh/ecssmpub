from web3 import Web3
import json
import sqlite3
import time
from datetime import datetime
import constants
from tqdm import tqdm
from datetime import datetime
from eth_abi import decode

# Configuration
RPC_URL = constants.RPC_URL
w3 = Web3(Web3.HTTPProvider(RPC_URL))

POOL_MANAGER_ADDR = constants.UNI_POOL_MANAGER.lower()
POSITION_MANAGER_ADDR = constants.UNI_POSITION_MANAGER.lower()
UNI_V4_START_BLOCK = constants.UNI_V4_START_BLOCK
#START_BLOCK = 24693836

CURRENCY_1_ADDR = constants.DAI_CONTRACT_ADDR.lower()
CURRENCY_2_ADDR = constants.USDT_CONTRACT_ADDR.lower()
curr1_curr2 = "dai_usdt"

# Event Topics
INIT_SIG = w3.keccak(text="Initialize(bytes32,address,address,uint24,int24,address)").to_0x_hex()
MOD_SIG = w3.keccak(text="ModifyLiquidity(bytes32,address,int24,int24,int256,bytes32)").to_0x_hex()
TRANSFER_SIG = w3.keccak(text="Transfer(address,address,uint256)").to_0x_hex()
SWAP_SIG = w3.keccak(text="Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)").to_0x_hex()
ZERO_ADDR = "0x0000000000000000000000000000000000000000"

# Standardize currency order (v4 requires currency0 < currency1)
C0, C1 = (CURRENCY_1_ADDR, CURRENCY_2_ADDR) if int(CURRENCY_1_ADDR, 16) < int(CURRENCY_2_ADDR, 16) else (CURRENCY_2_ADDR, CURRENCY_1_ADDR)

# Timestamp Cache to minimize RPC calls
BLOCK_CACHE = {}

# Event Signatures
INIT_SIG = w3.keccak(text="Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)").to_0x_hex()

# Table names
TABLE_CURR1_CURR2_V4_POOLS = f"{curr1_curr2}_v4_pools"
TABLE_CURR1_CURR2_V4_POOL_FEES = f"{curr1_curr2}_v4_pool_fees"
TABLE_CURR1_CURR2_SYNC_PROGRESS = f"{curr1_curr2}_sync_progress"
CURR1_CURR2_V4_POOL_DISCOVERY_KEY = f"{curr1_curr2}_pool_discovery_key"
TABLE_CURR1_CURR2_LIFETIMES = f"v4_{curr1_curr2}_lifetimes"

def setup_db(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    # Stores the PoolIds we care about
    c.execute('CREATE TABLE IF NOT EXISTS ' + TABLE_CURR1_CURR2_V4_POOLS + ' (pool_id TEXT PRIMARY KEY)')
    # Stores how far we've scanned for pool initialization
    c.execute('CREATE TABLE IF NOT EXISTS ' + TABLE_CURR1_CURR2_SYNC_PROGRESS + ' (key TEXT PRIMARY KEY, last_block INTEGER)')
    c.execute('INSERT OR IGNORE INTO ' + TABLE_CURR1_CURR2_SYNC_PROGRESS + ' VALUES ("' + CURR1_CURR2_V4_POOL_DISCOVERY_KEY + '", ?)', (UNI_V4_START_BLOCK,))
    conn.commit()
    return conn

def get_timestamp(block_num):
    """Fetches and caches block timestamps."""
    if block_num == "Active": return int(time.time())
    if block_num not in BLOCK_CACHE:
        BLOCK_CACHE[block_num] = w3.eth.get_block(block_num)['timestamp']
    return BLOCK_CACHE[block_num]

def decode_log_data(hex_str):
    # The log_data contains everything not marked 'indexed': true
    # Non-indexed fields in ModifyLiquidity ABI: https://github.com/Uniswap/v4-core/blob/main/src/interfaces/IPoolManager.sol
    non_indexed_types = ["int24", "int24", "int256", "bytes32"]
    decoded_data = decode(non_indexed_types, bytes.fromhex(hex_str))
    tick_lower = decoded_data[0]
    tick_upper = decoded_data[1]
    liquidity_delta = decoded_data[2]
    salt = decoded_data[3]
    return tick_lower, tick_upper, liquidity_delta, salt


def decode_int24(hex_str):
    """Decodes a hex word into a signed 24-bit integer."""
    # Take the last 6 hex chars (24 bits)
    val = int(hex_str[-6:], 16)
    # Handle negative values (two's complement)
    if val & 0x800000:
        val -= 0x1000000
    return val

def sync_currency_swap_liq_pools(conn, end_block, chunk_size=50000):
    c = conn.cursor()
    c.execute('SELECT last_block FROM ' + TABLE_CURR1_CURR2_SYNC_PROGRESS + ' WHERE key="' + CURR1_CURR2_V4_POOL_DISCOVERY_KEY + '"')
    start_block = c.fetchone()[0] 
    
    if start_block >= end_block:
        return
    
    print(f"Syncing Pool Discovery from {start_block} to {end_block}...")
    
    # Topics for Initialize(poolId, currency0, currency1, ...)
    topics = [
        INIT_SIG, 
        None, 
        "0x" + C0[2:].lower().zfill(64), 
        "0x" + C1[2:].lower().zfill(64)
    ]

    for current_start in tqdm(range(start_block, end_block, chunk_size), desc="Scanning for Pools"):
        current_end = min(current_start + chunk_size, end_block)
        try:
            logs = w3.eth.get_logs({
                "fromBlock": current_start,
                "toBlock": current_end,
                "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
                "topics": topics
            })
            
            for log in logs:
                pool_id = log['topics'][1].to_0x_hex()
                c.execute('INSERT OR IGNORE INTO ' + TABLE_CURR1_CURR2_V4_POOLS + ' VALUES (?)', (pool_id,))
            
            # Update progress after each successful chunk
            c.execute('UPDATE ' + TABLE_CURR1_CURR2_SYNC_PROGRESS + ' SET last_block = ? WHERE key="' + CURR1_CURR2_V4_POOL_DISCOVERY_KEY + '"', (current_end,))
            conn.commit()
            
        except Exception as e:
            print(f"\nTimeout or error at block {current_start}: {e}. Retrying with smaller chunk...")
            # If a specific chunk fails, you could recursively call this with chunk_size // 2
            time.sleep(1)
            continue

## Log extraction
### Tables
def create_lifetime_table(cursor):
    """Creates a table and saves the lifetime data."""
    cursor.execute('''DROP TABLE IF EXISTS ''' + TABLE_CURR1_CURR2_LIFETIMES)
    cursor.execute('''CREATE TABLE IF NOT EXISTS ''' + TABLE_CURR1_CURR2_LIFETIMES + ''' (
            position_id TEXT PRIMARY KEY,
            pool_id TEXT,
            tick_lower INT, 
            tick_upper INT,
            liquidity_delta INTEGER,
            sender TEXT,
            born_block INTEGER,
            died_block TEXT,
            born_timestamp INTEGER,
            died_timestamp INTEGER,
            lifetime_seconds TEXT,
            born_date TEXT,
            died_date TEXT,
            status TEXT)''')
    return cursor

def create_table_historical_swap_fee(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS ''' + TABLE_CURR1_CURR2_V4_POOL_FEES + ''' (
             pool_id TEXT, block_number INT, fee_pips INT, 
             volume_usd REAL, PRIMARY KEY (pool_id, block_number))''')
    return cursor

### Events
def sync_liquidity_events(pool_ids, start_block, end_block, chunk_size):
    positions = {} # salt -> {poolId, mint_block, burn_block}
    print(f"Scanning blocks {start_block} to {end_block} for position activity...")
    for i in tqdm(range(start_block, end_block, chunk_size), desc="Scanning Liquidity Events"):
        to_b = min(i + chunk_size, end_block)
        
        # We look for any interaction with our target pools
        for pid in pool_ids:
            mod_logs = w3.eth.get_logs({
                "fromBlock": i, 
                "toBlock": to_b,
                "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
                "topics": [MOD_SIG, pid]
            })
            for log in mod_logs:
                #salt = log['topics'][3].hex()
                data_hex = log['data'].hex()

                if data_hex.startswith('0x'): 
                    print('hex')
                    data_hex = data_hex[2:]
                # 
                # Word 0: sender (chars 0-32)
                # Word 1: tickLower (chars 32-64)
                # Word 2: tickUpper (chars 64-96)
                # Word 3: liquidityDelta (chars 96-128)
                # Word 3: salt (chars 192-256)

                tick_lower, tick_upper, liquidity_delta, salt = decode_log_data(data_hex)

                sender = w3.to_checksum_address(f"0x{log['topics'][1].hex()[-40:]}")
            
                #tick_lower = decode_int24(data_hex[0:64])
                #tick_upper = decode_int24(data_hex[64:128])
                #salt = "0x" + data_hex[192:256] # 96 bytes offset = 192 hex chars
                salt = "0x" + str(salt.hex())
                positions[salt] = {
                    "poolId": pid, 
                    "first_seen": log['blockNumber'],
                    "tick_lower": tick_lower,
                    "tick_upper": tick_upper,
                    "liquidity_delta": liquidity_delta,
                    "sender": sender
                }

    # 3. Resolve lifetimes for every discovered Position (TokenId)
    print(f"Resolving lifetimes for {len(positions)} unique positions...")
    results = []
    for tid, data in tqdm(positions.items(), desc="Tracing positions"):
        # info contains {'pool_id': ..., 'first_seen': ...}
        tid_hex = tid.zfill(64)
        position_id = int(tid_hex, 16)
        pid = data['poolId']
        tick_upper = data['tick_upper']
        tick_lower = data['tick_lower']
        liquidity_delta = str(data['liquidity_delta'])
        sender = data['sender']
        
        # We only need to search from the first time we saw this token
        # to the current block. 
        start_search = data['first_seen']
        latest = end_block
        
        # CHUNKED TRANSFER SEARCH
        # If the gap is huge, we still chunk to avoid RPC limits
        step = 50000 
        died_block = "Active"
        
        for chunk_start in range(start_search, latest, step):
            chunk_end = min(chunk_start + step, latest)
            
            try:
                t_logs = w3.eth.get_logs({
                    "fromBlock": chunk_start,
                    "toBlock": chunk_end,
                    "address": Web3.to_checksum_address(POSITION_MANAGER_ADDR),
                    "topics": [TRANSFER_SIG, None, None, tid_hex]
                })
                
                for t in t_logs:
                    # topic2 is 'to'. If 'to' is zero address, it's a burn.
                    to_addr = "0x" + t['topics'][2].hex()[-40:]
                    if to_addr == ZERO_ADDR:
                        died_block = t['blockNumber']
                        break # Found the death, stop searching chunks
                
                if died_block != "Active": break
                
            except Exception as e:
                print(f"Transfer scan failed for {tid_hex} at {chunk_start}: {e}")
                continue

        # Save to DB (Calculate timestamps only for successful finds to save API calls)
        b_ts = get_timestamp(start_search)
        d_ts = get_timestamp(died_block) if isinstance(died_block, int) else int(time.time())
        lifetime_blocks = (d_ts - b_ts) if isinstance(d_ts, int) else "N/A"
        b_ts_date = datetime.fromtimestamp(b_ts).isoformat()
        d_ts_date = datetime.fromtimestamp(d_ts).isoformat()

        status = "Closed" if isinstance(died_block, int) else "Active"
        results.append((str(position_id), pid, tick_lower, tick_upper, liquidity_delta, sender, start_search, str(died_block), b_ts, d_ts, lifetime_blocks, b_ts_date, d_ts_date, status))
    return results

def sync_fees(pool_ids, start_block, end_block, chunk_size):
    results = [] # salt -> {poolId, mint_block, burn_block}
    print(f"Scanning blocks {start_block} to {end_block} for position activity...")
    for i in tqdm(range(start_block, end_block, chunk_size), desc="Scanning Liquidity Events for fees"):
        to_b = min(i + chunk_size, end_block)
        for pid in pool_ids:
            logs = w3.eth.get_logs({
                "fromBlock": i,
                "toBlock": to_b,
                "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
                "topics": [SWAP_SIG, pid]
            })
            for log in logs:
                # Swap data: amount0 (int128), amount1 (int128), sqrtPrice (uint160), 
                # liquidity (uint128), tick (int24), fee (uint24)
                data = log['data'].hex()
                # The fee is the last 32-byte word (64 hex chars)
                fee_pips = int(data[-64:], 16)
                
                # Save block and fee to DB
                results.append((pid, log['blockNumber'], fee_pips, 0)) # Volume calculation optional
    return results

def process_block_range(db_name, start_block, end_block, pool_ids, chunk_size=2000):
    # Chunked scan for Position activity (ModifyLiquidity)
    liquidity_events = sync_liquidity_events(pool_ids, start_block, end_block, chunk_size)
    historical_fees = sync_fees(pool_ids, start_block, end_block, chunk_size)

    #print(positions)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor = create_lifetime_table(cursor)
    cursor = create_table_historical_swap_fee(cursor)
   
    # 4. Save to Database
    cursor.executemany('INSERT INTO ' + TABLE_CURR1_CURR2_LIFETIMES + ' VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', liquidity_events)
    cursor.executemany('INSERT OR IGNORE INTO ' + TABLE_CURR1_CURR2_V4_POOL_FEES + ' VALUES (?, ?, ?, ?)', historical_fees)
    conn.commit()
    conn.close()
    print("Database sync complete.")

def heath_check(w3, past_block):
    # Test querying the node
    try:
        multicall_addr = Web3.to_checksum_address(constants.MULTICALL_ADDR.lower())
        code = w3.eth.get_code(multicall_addr, block_identifier=past_block)
        if len(code) <= 2:
            print(f"FAILED: No contract code at block {past_block}. Your node is likely not an Archive node.")
        else:
            print(f"SUCCESS: Node is an Archive node. Code found at block {past_block}.")
    except Exception as e:
        print(f"ERROR: {e}. The node rejected the historical query.")

def topic_debugger(w3):
    print("--- Uniswap v4 Topic Debugger ---")
    
    # 1. The signature I gave you (Missing 'int24' at the end)
    old_sig = "Initialize(bytes32,address,address,uint24,int24,address,uint160)"
    old_hash = w3.keccak(text=old_sig).to_0x_hex()
    
    # 2. The correct signature for v4 Mainnet
    correct_sig = "Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)"
    correct_hash = w3.keccak(text=correct_sig).to_0x_hex()
    
    print(f"Old Hash (Incorrect): {old_hash}")
    print(f"Correct Hash:         {correct_hash}")

    # 3. Quick test: Does the correct hash exist in the last 50,000 blocks?
    latest = w3.eth.block_number
    try:
        logs = w3.eth.get_logs({
            "fromBlock": latest - 500,
            "toBlock": "latest",
            "address": Web3.to_checksum_address(POOL_MANAGER_ADDR),
            "topics": [correct_hash]
        })
        print(f"Status: Found {len(logs)} pool initializations with the correct hash!")
    except Exception as e:
        print(f"Status: Search failed. Check RPC or POOL_MANAGER address. Error: {e}")


if __name__ == "__main__":
    last_n_days = 180
    
    w3 = Web3(Web3.HTTPProvider(constants.RPC_URL))
    topic_debugger(w3)
    latest_block =  24430360 - 1  # 24695825. w3.eth.block_number

    # 25044360: considered latest
    # 24430360: 87 days ago
    # 23760360: 181 days ago
    # 23120360: 270 days ago
    # 22480360: 360 days ago
    # 21835360: 450 days ago
    # 21191360: 540 days ago
    past_block = 23760360
    #past_block = latest_block - (last_n_days * 7100)
    #past_block = latest_block - 20000
    print(f"Latest block is {latest_block}")

    db_name = f'sqldb/{curr1_curr2}/sql_univ4_{curr1_curr2}_{past_block}_to_{latest_block}.db'
    db_conn = setup_db(db_name)
    
    heath_check(w3, past_block)

    # Discover all pools in safe increments
    sync_currency_swap_liq_pools(db_conn, latest_block, chunk_size=2000)

    # Now fetch the PoolIds from the DB to proceed with position tracking
    cursor = db_conn.cursor()
    cursor.execute('SELECT pool_id FROM ' + TABLE_CURR1_CURR2_V4_POOLS)
    all_pool_ids = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(all_pool_ids)} pools to monitor.")

    process_block_range(db_name, past_block, latest_block, all_pool_ids, chunk_size=500)