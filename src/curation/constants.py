ETH_ILK_NAME = "ETH-A"                           # Collateral type
SQLDB_NAME = "maker_vault_collateral_eth-a.db"
RPC_URL = "http://eth-fullarch-validator-1.virtual.crc.nd.edu:11500"
MULTICALL_ADDR = "0xca11bde05977b3631167028862be2a173976ca11"
VAT_ADDR = "0x35D1b3F3D7966A1DFe207aa4514C12a259A0492B"
CDP_MANAGER_ADDR = "0x5ef30b9986345249bc32d8928B7ee64DE9435E39"
DAI_USD_FEED = "0xAed0c38402a5d19df6E4c03F4E2DceD6e29c1ee9" # Chainlink Mainnet
ETH_USD_FEED = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"

UNI_POOL_MANAGER = "0x000000000004444c5dc75cB358380D2e3dE08A90"
UNI_POSITION_MANAGER = "0xbd216513d74c8cf14cf4747e6aaa6420ff64ee9e"
UNI_V4_START_BLOCK = 21688329
UNI_V4_STATE_VIEW_CONTRACT = "0x7ffe42c4a5deea5b0fec41c94c136cf115597227"
DAI_CONTRACT_ADDR = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
USDT_CONTRACT_ADDR = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
USDC_CONTRACT_ADDR = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH_CONTRACT_ADDR = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
WBTC_CONTRACT_ADDR = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
ATH_CONTRACT_ADDR = "0xbe0ed4138121ecfc5c0e56b40517da27e6c5226b"
CB_BTC_CONTRACT_ADDR = "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf"
APXUSD_CONTRACT_ADDR = "0x98a878b1cd98131b271883b390f68d2c90674665"
AZTEC_CONTRACT_ADDR = "0xa27ec0006e59f245217ff08cd52a7e8b169e62d2"

DAI_DECIMAL_SCALE = 1e18
ATH_DECIMAL_SCALE = 1e18
APXUSD_DECIMAL_SCALE = 1e18
AZTEC_DECIMAL_SCALE = 1e18
WETH_DECIMAL_SCALE = 1e18
WBTC_DECIMAL_SCALE = 1e8
CB_BTC_DECIMAL_SCALE = 1e8
USDT_DECIMAL_SCALE = 1e6
USDC_DECIMAL_SCALE = 1e6


# ABIs
MULTICALL_ABI = '[{"inputs":[{"components":[{"internalType":"address","name":"target","type":"address"},{"internalType":"bool","name":"allowFailure","type":"bool"},{"internalType":"bytes","name":"callData","type":"bytes"}],"name":"calls","type":"tuple[]"}],"name":"aggregate3","outputs":[{"components":[{"internalType":"bool","name":"success","type":"bool"},{"internalType":"bytes","name":"returnData","type":"bytes"}],"name":"returnData","type":"tuple[]"}],"stateMutability":"payable","type":"function"}]'
VAT_ABI = '[{"constant":true,"inputs":[{"name":"","type":"bytes32"},{"name":"","type":"address"}],"name":"urns","outputs":[{"name":"ink","type":"uint256"},{"name":"art","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[{"name":"","type":"bytes32"}],"name":"ilks","outputs":[{"name":"Art","type":"uint256"},{"name":"rate","type":"uint256"},{"name":"spot","type":"uint256"},{"name":"line","type":"uint256"},{"name":"dust","type":"uint256"}],"type":"function"}]'
MANAGER_ABI = '[{"constant":true,"inputs":[],"name":"cdpi","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[{"name":"","type":"uint256"}],"name":"urns","outputs":[{"name":"","type":"address"}],"type":"function"},{"constant":true,"inputs":[{"name":"","type":"uint256"}],"name":"ilks","outputs":[{"name":"","type":"bytes32"}],"type":"function"}]'
CHAINLINK_ABI = '[{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"}]'

# Minimal ABIs for Event Decoding
POOL_MANAGER_ABI = [
    {
        "anonymous": False,
        "name": "Initialize",
        "type": "event",
        "inputs": [
            {"indexed": True, "name": "id", "type": "bytes32"},
            {"indexed": True, "name": "currency0", "type": "address"},
            {"indexed": True, "name": "currency1", "type": "address"},
            {"indexed": False, "name": "fee", "type": "uint24"},
            {"indexed": False, "name": "tickSpacing", "type": "int24"},
            {"indexed": False, "name": "hooks", "type": "address"},
            {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "name": "tick", "type": "int24"}
        ]
    },
    {
        "anonymous": False,
        "name": "ModifyLiquidity",
        "type": "event",
        "inputs": [
            {"indexed": True, "name": "id", "type": "bytes32"},
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": False, "name": "tickLower", "type": "int24"},
            {"indexed": False, "name": "tickUpper", "type": "int24"},
            {"indexed": False, "name": "liquidityDelta", "type": "int256"},
            {"indexed": False, "name": "salt", "type": "bytes32"}
        ]
    },
    {
        "anonymous": False,
        "name": "Swap",
        "type": "event",
        "inputs": [
            {"indexed": True, "name": "id", "type": "bytes32"},
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": False, "name": "amount0", "type": "int128"},
            {"indexed": False, "name": "amount1", "type": "int128"},
            {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "name": "liquidity", "type": "uint128"},
            {"indexed": False, "name": "tick", "type": "int24"},
            {"indexed": False, "name": "fee", "type": "uint24"}
        ]
    },
    {
        "constant": True,
        "inputs": [{"name": "id", "type": "bytes32"}],
        "name": "getSlot0",
        "outputs": [
            {"name": "sqrtPriceX96", "type": "uint160"},
            {"name": "tick", "type": "int24"},
            {"name": "protocolFee", "type": "uint24"},
            {"name": "lpFee", "type": "uint24"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "id", "type": "bytes32"}],
        "name": "getLiquidity",
        "outputs": [{"name": "liquidity", "type": "uint128"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
]

POSITION_MANAGER_ABI = [
    {
        "anonymous": False,
        "name": "Transfer",
        "type": "event",
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": True, "name": "tokenId", "type": "uint256"}
        ]
    }
]

SWAP_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "internalType": "PoolId", "name": "id", "type": "bytes32"},
        {"indexed": False, "internalType": "address", "name": "sender", "type": "address"},
        {"indexed": False, "internalType": "int256", "name": "amount0", "type": "int256"},
        {"indexed": False, "internalType": "int256", "name": "amount1", "type": "int256"},
        {"indexed": False, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
        {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"},
        {"indexed": False, "internalType": "uint24", "name": "fee", "type": "uint24"}
    ],
    "name": "Swap",
    "type": "event"
}

UNISWAP_V4_STATEVIEW_ABI = STATEVIEW_ABI = [
    {
        "inputs": [{"internalType": "PoolId", "name": "poolId", "type": "bytes32"}],
        "name": "getSlot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint24", "name": "protocolFee", "type": "uint24"},
            {"internalType": "uint24", "name": "lpFee", "type": "uint24"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "PoolId", "name": "poolId", "type": "bytes32"}],
        "name": "getLiquidity",
        "outputs": [{"internalType": "uint128", "name": "liquidity", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "PoolId", "name": "poolId", "type": "bytes32"},
            {"internalType": "int24", "name": "tick", "type": "int24"}
        ],
        "name": "getTickInfo",
        "outputs": [
            {"internalType": "uint128", "name": "liquidityGross", "type": "uint128"},
            {"internalType": "int128", "name": "liquidityNet", "type": "int128"},
            {"internalType": "uint256", "name": "feeGrowthOutside0X128", "type": "uint256"},
            {"internalType": "uint256", "name": "feeGrowthOutside1X128", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "PoolId", "name": "poolId", "type": "bytes32"},
            {"internalType": "int24", "name": "tick", "type": "int24"}
        ],
        "name": "getTickLiquidity",
        "outputs": [
            {"internalType": "uint128", "name": "liquidityGross", "type": "uint128"},
            {"internalType": "int128", "name": "liquidityNet", "type": "int128"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "PoolId", "name": "poolId", "type": "bytes32"},
            {"internalType": "int16", "name": "tick", "type": "int16"}   # Note: this is word position (tick >> 8)
        ],
        "name": "getTickBitmap",
        "outputs": [{"internalType": "uint256", "name": "tickBitmap", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "PoolId", "name": "poolId", "type": "bytes32"},
            {"internalType": "int24", "name": "tick", "type": "int24"}
        ],
        "name": "getTickFeeGrowthOutside",
        "outputs": [
            {"internalType": "uint256", "name": "feeGrowthOutside0X128", "type": "uint256"},
            {"internalType": "uint256", "name": "feeGrowthOutside1X128", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "PoolId", "name": "poolId", "type": "bytes32"}],
        "name": "getFeeGrowthGlobals",
        "outputs": [
            {"internalType": "uint256", "name": "feeGrowthGlobal0", "type": "uint256"},
            {"internalType": "uint256", "name": "feeGrowthGlobal1", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "PoolId", "name": "poolId", "type": "bytes32"},
            {"internalType": "int24", "name": "tickLower", "type": "int24"},
            {"internalType": "int24", "name": "tickUpper", "type": "int24"}
        ],
        "name": "getFeeGrowthInside",
        "outputs": [
            {"internalType": "uint256", "name": "feeGrowthInside0X128", "type": "uint256"},
            {"internalType": "uint256", "name": "feeGrowthInside1X128", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]