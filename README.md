# Algorithmic-Trading-Contest
My coded solution to the Algorithmic Trading Contest at Imperial College London. The aim is to formulate a high-frequency trading strategy to trade 4 assets, maximising the Sharpe ratio and making use of market making and arbitrage strategies, such that the PnL can be quickly evaluated (thus success of algorithm) in under a minute.

We came third place!
---

## Strategy Overview

The strategy is hybrid: it balances the steady returns of liquidity provision with opportunistic gains from a short-horizon directional signal.

### 1. Market-Making
We provide two-sided liquidity at multiple price levels.
* **Quote placement:** Resting orders are posted at the best bid and ask, with additional smaller orders one and two ticks inside the spread whenever the spread is wide enough (≥ 2 and ≥ 4 ticks respectively).
* **Inventory control:** Rather than shifting a reservation price, we use hard inventory caps that tighten by quote level (the inside-spread quotes switch off first as a position builds) and scale order size down as inventory approaches the position limit.
* **Adaptive sizing:** Quote sizes are scaled by the current spread relative to its EMA (bigger in unusually wide spreads), by short-term realised volatility, and by remaining position capacity.

### 2. Directional Core
We use a **regularised linear model** (`Ridge`, α = 0.3, on standardised features) to forecast the next-tick mid-price change for each asset. The model is retrained online every 25 ticks on a rolling 150-tick window so it adapts to changing conditions.

We tried tree-based models early on but with only a few days of tick data they overfit badly out of sample; a small linear model was far more stable and, importantly, fast enough to retrain inside the event loop.
* **Signal:** The predicted move is divided by short-term realised volatility (std of the last 10 mid changes) to give a signal-to-noise ratio.
* **Trigger Mechanism:** Aggressive orders are sent in tiers as the signal-to-noise ratio crosses calibrated thresholds (0.6 / 1.0 / 1.8 / 2.5), with size increasing at each tier.
* **Adverse Selection Avoidance:** Passive quotes on the side opposite the signal are only kept while inventory is comfortably within limits, so we are less likely to be filled against a move we expect.

### 3. Risk Limits
Each asset has a maximum absolute position, with tiered soft limits below it that progressively disable the more aggressive quote levels and directional orders. Positions are settled at mid at the end of the evaluation window.

---

## Feature Engineering Pipeline

Raw Limit Order Book (LOB) snapshots and trade prints are turned into ~16 per-asset features each tick:
* **Microstructure:** mid price, spread, microprice
* **Depth and flow:** bid/ask depth sums, queue imbalance, top-of-book imbalance, a rolling 5-tick average imbalance
* **Volatility:** realised volatility over a 40-tick lookback and a recent-vs-historical volatility ratio
* **Temporal:** VWAP-mid difference, last-tick price change, fast/slow EMA crossover, short (3-tick) and medium (10-tick) momentum, mean recent return

---

## Algorithmic Implementation

### Event Loop
1.  **Data Ingestion:** Update internal `OrderBook` snapshots and compute the features above.
2.  **Model Update:** Every 25 ticks, refit the per-asset `StandardScaler` + `Ridge` model on the rolling buffer.
3.  **Inference:** Predict the next-tick mid change and divide by realised volatility to get the signal.
4.  **Quote Generation:** Post multi-level passive quotes via the `Order` class, sized by spread, volatility and inventory capacity.
5.  **Directional Execution:** If the signal clears a threshold tier, submit aggressive orders sized by `adaptive_size`.

---

## Room for Improvement
* The inventory-penalty and skew parameters (`k_inv`, `k_skew`) are defined but never used — a proper Avellaneda-Stoikov reservation-price shift would be the natural next step.
* The signal is scaled by a fixed multiplier before thresholding; this was a last-hour calibration hack and should be replaced by tuning the thresholds directly on held-out data.
* Cross-asset information is ignored — each asset is modelled independently, so there is no portfolio-level allocation or hedging.
4.  **Directional Execution:** If signal strength is sufficient, submit market orders for immediate liquidity take.
5.  **Rebalancing:** Periodically recompute optimal weights via `scipy.optimize`.

