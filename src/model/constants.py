class MetaConstants(type):
    def __setattr__(cls, key, value):
        raise TypeError(f"Cannot reassign constant '{key}'")
    def __delattr__(cls, key):
        raise TypeError(f"Cannot delete constant '{key}'")

class Constants(metaclass=MetaConstants):
    HIDDEN_DIM = 128
    
    # LAMBDA1: ETH-USDC=1.0, ETH-USDT=10.0, WBTC-USDC=10.0
    # LAMBDA2: ETH-USDC=0.05,0.1, ETH-USDT=1.0, WBTC-USDC=1.0
    # LAMBDA5: ETH-USDC=1.0,0.01, ETH-USDT=0.1, WBTC-USDC=1.0
    # LAMBDA6: ETH-USDC=1.0,1.0, ETH-USDT=1.0, WBTC-USDC=1.0
    LAMBDA1 = 10.0      # 100.0 Ret mag
    LAMBDA2 = 10.0      # 100.0 Ret direction
    LAMBDA3 = 1.0       # Combined ret + dir for weighted ret loss times precision
    LAMBDA4 = 1.0       # 1.0 Volatility
    LAMBDA5 = 0.5       # 0.5 Liquidity

    LAMBDA6 = 10.0       # 1.0 Phy loss return
    LAMBDA7 = 1.0       # 1.0 Phy loss volatility
    LAMBDA_IC = 0.0
    ## LAMBDA_PHYS = 0.5 (usdc/eth, eth/usdt, wbtc/usdc)
    LAMBDA_PHYS = 0.5     # 0.1
    LAMBDA_DATA = 1.0
    MU_D = 1.0
    MU_P = 1.0
    OMEGA = 1.0
    GAMMA_D = 0.96

    WETH_DECIMAL_SCALE = 18
    WBTC_DECIMAL_SCALE = 8
    USDT_DECIMAL_SCALE = 6
    USDC_DECIMAL_SCALE = 6