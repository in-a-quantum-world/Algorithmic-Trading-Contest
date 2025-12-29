# Algorithmic-Trading-Contest
My coded solution to the Algorithmic Trading Contest at Imperial College London. The aim is to formulate a high-frequency trading strategy to trade 4 assets, maximising the Sharpe ratio and making use of market making and arbitrage strategies, such that the PnL can be quickly evaluated (thus success of algorithm) in under a minute.

---

## Strategy Overview

The strategy follows a hybrid architecture that balances the steady returns of liquidity provision with the opportunistic gains of trend-following and mean-reversion.

### 1. Market-Making 
We utilised an **Avellaneda-Stoikov** inspired framework to provide two-sided liquidity.
* **Reservation Price:** Instead of quoting around the mid-price, we calculate a "Reservation Price" $P_{res} = Mid + \mu_i$. 
* **Drift & Skew:** The mid-price is shifted by the predicted short-term drift ($\mu_i$) and skewed by an inventory penalty ($k_{inv}$) to reduce exposure.
* **Adaptive Spreads:** Quoted spreads are dynamically widened or narrowed based on estimated volatility ($\sigma_i$) and current risk-aversion parameters.

### 2. Directional & Stat-Arb Core
We deployed **Short-Horizon Predictors** (XGBoost/LightGBM) to forecast price movement over the next $k$ ticks.
* **Trigger Mechanism:** Aggressive trades (market orders) are executed only when the signal-to-noise ratio ($E[R] / \sigma_{est}$) exceeds a calibrated threshold.
* **Adverse Selection Avoidance:** Quotes are aggressively skewed away from the predicted direction to avoid being filled on the wrong side of a trend.

### 3. Portfolio Optimization
Capital is allocated across four assets by solving a **Mean-Variance Proxy** to maximize the Portfolio Sharpe Ratio:
* **Objective:** Minimize $\mathcal{J}(w) = -w^T\mu + \lambda w^T\Sigma w$.
* **Constraints:** Budgeted for leverage caps, dollar neutrality, and maximum per-asset exposure.

---

## Feature Engineering Pipeline

We transformed raw Limit Order Book (LOB) snapshots into high-frequency features (Top-5 Levels):

| Feature Category | Metrics |
| :--- | :--- |
| **Micro-Structure** | Mid-price, Microprice, Spread, multi-level imbalance. |
| **Depth & Flow** | Level depth sums, queue imbalances, order-flow deltas (incoming trade signs/sizes). |
| **Volatility** | Short-term realized volatility and volume-weighted aggregates. |
| **Temporal** | Horizon features ($t, t-1, ...$), VWAP, and rolling means. |


---

## Algorithmic Implementation

### Event Loop Logic
1.  **Data Ingestion:** Update internal `OrderBook` snapshots and compute micro-features.
2.  **Inference:** Feed features into LightGBM/LSTM to produce $E[R]$ and $\sigma_{est}$.
3.  **Quote Generation:** * Calculate `reservation = mid + drift + inventory_penalty + signal_skew`.
    * Post `bid/ask` via the `Order` class with adaptive spreads.
4.  **Directional Execution:** If signal strength is sufficient, submit market orders for immediate liquidity take.
5.  **Rebalancing:** Periodically recompute optimal weights via `scipy.optimize`.

