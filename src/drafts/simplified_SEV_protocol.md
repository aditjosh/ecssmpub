### [draft-subsection] Off-Chain Safety Verification for Hook Computation

Autonomous liquidity hooks that ingest off-chain intelligence (e.g., volatility forecasts, optimal tick ranges) conventionally rely on zero-knowledge coprocessors such as Brevis to attest to computational integrity. In this paradigm, the off-chain prover flattens machine-learning inference or statistical aggregation into an arithmetic circuit, generates a succinct proof, and submits the result on-chain for pairing-based verification. While this guarantees correct execution of the off-chain trace, it imposes substantial overhead: circuit compilation, trusted setup, proof generation latency, and non-trivial verification gas. For hooks whose primary requirement is to ensure that submitted parameters are *safe* rather than *mathematically optimal*, this full pipeline is unnecessarily heavyweight.

**Simplified Symbolic Execution for Hook Safety.** We propose replacing the ZK coprocessor with a protocol-specific symbolic execution verifier (SEV) that checks safety invariants over the pool state before the hook executes an off-chain directive. In our architecture, Action F6 denotes the event in which the off-chain Liquidity Manager submits computed parameters (e.g., `tickLower`, `tickUpper`, `liquidityDelta`, or a dynamic fee adjustment) to the Pool Operator Hook. Rather than proving that the off-chain model correctly computed the standard deviation over the last 20 ticks, the SEV verifies that the resulting state transition is safe for the pool.

An off-chain instruction is abstracted as a tuple $\mathcal{I} = (\tau, \phi, \beta, \sigma)$, where $\tau \in \{\text{swap}, \text{modifyLiquidity}, \text{updateFee}\}$ is the transaction type, $\phi$ is a precondition predicate, $\beta$ describes the intended state transformation, and $\sigma$ is a slippage tolerance. For a `ModifyLiquidity` callback triggered by Action F6, the verifier evaluates the current pool state $\mathcal{S}_{\text{pool}}$ and enforces the following path constraints:

1. **Tick Range Validity.** Let $\alpha$ and $\beta$ denote the proposed `tickLower` and `tickUpper`, respectively. The constraint ensures the range is non-empty and within protocol bounds:
   $$
   \alpha < \beta \;\land\; \text{MIN\_TICK} \leq \alpha \;\land\; \beta \leq \text{MAX\_TICK}.
   $$

2. **Liquidity Conservation.** Let $\gamma$ denote the signed liquidity delta. The pool's total active liquidity $L_{\text{pool}}$ after the transition must remain above a protocol-specific minimum $L_{\min}$:
   $$
   L_{\text{pool}} + \gamma \geq L_{\min}.
   $$

3. **Slippage Bound.** Let $p'$ be the implied spot price after the transition and $p_{\text{twap}}$ the current time-weighted average price. The relative deviation must not exceed threshold $\sigma$:
   $$
   \frac{|p' - p_{\text{twap}}|}{p_{\text{twap}}} \leq \sigma.
   $$

4. **Fee Tier Sanity.** For dynamic-fee hooks, the proposed fee $\delta$ must satisfy the initialization constraint $\delta \in [\delta_{\min}, \delta_{\max}]$.

The verifier encodes the Uniswap V4 `ModifyLiquidity` state transition as a system of constraints over symbolic variables $(\alpha, \beta, \gamma, \delta)$. If the constraint system is unsatisfiable for all violation paths, the verifier emits a **SAFE** decision; otherwise the instruction is rejected before reaching the `PoolManager`.

**System Integration.** The SEV is positioned between the off-chain Listener Service and the on-chain Pool Operator Hook. Upon receiving an Action F6 event from the ML Core, the verifier queries the current pool state via RPC, evaluates the constraints above, and produces a signed safety attestation. The hook's callback accepts this attestation together with the rebalancing parameters and executes the `ModifyLiquidity` call only if the safety bit is asserted. This preserves trustless execution while removing proof verification from the critical path.

**Eliminating the ZK Coprocessor Step.** Under the Brevis approach, the hook trusts the off-chain computation because a SNARK attests to the correctness of the arithmetic trace. Under our approach, the hook trusts the computation because a semantic firewall attests that the resulting state transition is safe for the pool. This eliminates circuit compilation, off-chain proof generation, and on-chain pairing verification. The verifier operates with sub-millisecond latency in a pure off-chain environment, forwarding only a lightweight boolean safety signal and thereby removing the dominant gas costs associated with ZK verification.

**Scope.** This simplification targets deterministic off-chain computations where safety dominates optimality. It does not replace ZK coprocessors for use cases requiring computational integrity attestation over complex historical traces (e.g., TWAP aggregation over thousands of blocks or reproducible ML inference). For such cases, a hybrid architecture is recommended: a ZK coprocessor attests to the ML inference trace, while the simplified SEV performs the final safety check on the resulting transaction parameters.