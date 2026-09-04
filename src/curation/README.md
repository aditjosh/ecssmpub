# Data Curation
Curating raw telemetry for a specific Uniswap v4 pool from an Ethereum archive node requires interacting with the v4 Singleton architecture. Unlike v3, where every pool is a separate contract, all Uniswap v4 pools reside inside a single contract: the PoolManager (deployed on Ethereum Mainnet at 0x000000000004444c5dc75cb358380d2e3de08a90).

To filter data specifically for the USDC-WETH pair, you must first canonically sort the token addresses to identify which is currency0 and which is currency1:

USDC: 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48

USDT: 0xdAC17F958D2ee523a2206206994597C13D831ec7

WBTC: 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599

WETH: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2

Since 0xA0b... is numerically less than 0xC02a..., USDC is currency0 and WETH is currency1.

## Curated Dataset

The curated dataset is available on Zenodo: [Public URL](https://zenodo.org/records/22230321?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc4ODUyNTkzNiwiZXhwIjoxODA5MTI5NTk5fQ.eyJpZCI6Ijc5ODkwODM1LWMyZTItNDkwMS04NmY2LTI2OTQ5NjQ3YTcwMCIsImRhdGEiOnt9LCJyYW5kb20iOiI1YmEzZmJhZGE1OWQ4YjNhNjg5NmI4ZTllZjA4YmQzMCJ9.7BvYmSTc-cwQCB_QmgEw5MaznAe4Sme-mqAGSoqBAV3x1LvcIDICaOv9aLD7TjZUIHZPNqeU5SX-yBPUNJzlNA)

## Pool ID
A pool's definitive identity in Uniswap v4 is its PoolId (a bytes32 hash). It is derive by matching the exact configuration parameters inside the PoolKey struct and hashing them:

$$\text{PoolId} = \text{keccak256}(\text{abi.encode}(\text{currency0, currency1, fee, tickSpacing, hooks}))$$

This PoolId string is found inside the topic1 (first indexed parameter) of all events related to the target pool.

## LP addresses

To track who owns a position, you must read logs from both the core PoolManager and the periphery PositionManager contract (0x3fc79444f8eacc1894775493ff3fa41f1e35ce11), because positions are wrapped as ERC-721 NFTs.

### Filter the ModifyLiquidity Event

Query logs from the PoolManager using the topic signature for liquidity modifications:

Topic 0 (Event Signature): ModifyLiquidity(bytes32,address,int24,int24,int256,bytes32)$$\to \text{0x23f7d159... (hash of signature)}$$

Topic 1 (Indexed PoolId): The target currency0-currency1 PoolId.

### Resolve the Actual User Wallet

The sender field in the ModifyLiquidity log will often point to the PositionManager contract address rather than the individual LP's wallet. To find the real position owner:

1. Parse the transaction hash where ModifyLiquidity was fired.
2. Look for the corresponding Transfer or IncreaseLiquidity / DecreaseLiquidity log emitted by the PositionManager contract in that same transaction block.
3. The PositionManager utilizes a unique tokenId mapping back to the owner's address. Tracking the Transfer(address indexed from, address indexed to, uint256 indexed tokenId) logs tells exactly which wallet currently holds that liquidity position.

## Extracting Pool Fees - Static and Dynamic
### Static Frees
Look for the Initialize event emitted by the PoolManager when the pool was first created:
```
Event Signature: Initialize(bytes32,address,address,uint24,int24,address,uint160,int24)
```
The 4th parameter (uint24 fee) displays the base fee.

### Dynamic Fee
If the 24-bit fee integer has its most significant bit set ($24\text{th bit} = 1$, or mathematically fee & 0x800000 != 0), it signals that the pool uses Dynamic Fees controlled by an external Hook contract.To curate historical dynamic fee changes over time:Query all Swap logs for the target PoolId.
```
Event Signature: Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)
```
The final unindexed parameter in the data payload is uint24 fee. Because every swap log explicitly records the exact fee charged at that specific moment, stacking these historical values sequentially provides a perfect timeline of the dynamic fee adjustments made by the hook.

## Curating Token Amounts in Each Position

The raw token balances of a position cannot be read directly from standard logs; token amounts fluctuate based on the current pool price. The balances must compute them using concentrated liquidity math equations.

### Reconstruct the Position State
For a given position identifier (composed of owner, tickLower, tickUpper, and salt), parse all historical ModifyLiquidity logs up to your target block height $B$. Maintain a running sum of the liquidityDelta parameter to determine the active net liquidity ($L$) at that block.

### Fetch the Active Price
Find the most recent Swap or Initialize log up to block $B$. Read the uint160 sqrtPriceX96 parameter. This represents the current global asset price of the pool: $$\sqrt{P} = \frac{\text{sqrtPriceX96}}{2^{96}}$$

### Calculate Token Inventories

Convert the tickLower and tickUpper integers into their geometric price representations, where $$\sqrt{P_l} = \sqrt{1.0001^{\text{tickLower}}}$$ and $$\sqrt{P_u} = \sqrt{1.0001^{\text{tickUpper}}}$$.

Apply the following conditional math checks depending on where the pool's current active price $$\sqrt{P}$$ sits relative to the position's bounds:

- Case 1: Price is below the range ($$\sqrt{P} < \sqrt{P_l}$$)The position is composed entirely of currency0: $$\text{Amount}_{\text{currency0}} = 0, \quad \text{Amount}\_{\text{currency1}} = L \cdot \left( \frac{1}{\sqrt{P_l}} \right), \quad \text{Amount}\_{\text{USDT}} = 0$$
  
- Case 2: Price is above the range ($$\sqrt{P} > \sqrt{P_u}$$). The position is composed entirely of currency1: $$\text{Amount}\_{\text{currency0}} = 0, \quad \text{Amount}_{\text{currency1}} = L \cdot (\sqrt{P_u} - \sqrt{P_l})$$
- Case 3: Price is inside the range ($$\sqrt{P_l} \le \sqrt{P} \le \sqrt{P_u}$$)

The position holds a mix of both assets:

$$\text{Amount}_{\text{currency0}} = L \cdot \left( \frac{1}{\sqrt{P}} - \frac{1}{\sqrt{P_u}} \right)$$
$$\text{Amount}_{\text{currency1}} = L \cdot (\sqrt{P} - \sqrt{P_l})$$
