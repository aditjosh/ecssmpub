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
DAI_CONTRACT_ADDR = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
USDT_CONTRACT_ADDR = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
USDC_CONTRACT_ADDR = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH_CONTRACT_ADDR = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
WBTC_CONTRACT_ADDR = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

# ABIs
MULTICALL_ABI = '[{"inputs":[{"components":[{"internalType":"address","name":"target","type":"address"},{"internalType":"bool","name":"allowFailure","type":"bool"},{"internalType":"bytes","name":"callData","type":"bytes"}],"name":"calls","type":"tuple[]"}],"name":"aggregate3","outputs":[{"components":[{"internalType":"bool","name":"success","type":"bool"},{"internalType":"bytes","name":"returnData","type":"bytes"}],"name":"returnData","type":"tuple[]"}],"stateMutability":"payable","type":"function"}]'
VAT_ABI = '[{"constant":true,"inputs":[{"name":"","type":"bytes32"},{"name":"","type":"address"}],"name":"urns","outputs":[{"name":"ink","type":"uint256"},{"name":"art","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[{"name":"","type":"bytes32"}],"name":"ilks","outputs":[{"name":"Art","type":"uint256"},{"name":"rate","type":"uint256"},{"name":"spot","type":"uint256"},{"name":"line","type":"uint256"},{"name":"dust","type":"uint256"}],"type":"function"}]'
MANAGER_ABI = '[{"constant":true,"inputs":[],"name":"cdpi","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[{"name":"","type":"uint256"}],"name":"urns","outputs":[{"name":"","type":"address"}],"type":"function"},{"constant":true,"inputs":[{"name":"","type":"uint256"}],"name":"ilks","outputs":[{"name":"","type":"bytes32"}],"type":"function"}]'
CHAINLINK_ABI = '[{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"}]'

