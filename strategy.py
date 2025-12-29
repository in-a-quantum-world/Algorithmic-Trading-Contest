import websockets
import json
import pandas as pd
import matplotlib.pyplot as plt
from backtest import evaluate

import pandas as pd
from extra_classes import Orderbook, Trade, Order
from pprint import pprint

DATASET = pd.read_csv("train.csv", header=[0, 1], index_col=0) # FIX THIS
SYMBOLS = ["A","B","C","D"]
URL = "ws://34.72.232.39:8765"
train_data = pd.read_csv("train.csv", header=[0, 1], index_col=0)
plt.plot(train_data.iloc[:,0])

POS_LIMIT = 100

def row_to_orderbook(row) -> Orderbook:
    orderbook = Orderbook()

    sells = {row['Ask1']: row['Ask1_size'],
             row['Ask2']: row['Ask2_size'],
             row['Ask3']: row['Ask3_size'],
             row['Ask4']: row['Ask4_size'],
             row['Ask5']: row['Ask5_size']
             }
    buys = {row['Bid1']: row['Bid1_size'],
            row['Bid2']: row['Bid2_size'],
            row['Bid3']: row['Bid3_size'],
            row['Bid4']: row['Bid4_size'],
            row['Bid5']: row['Bid5_size']
            }

    orderbook.sell_orders = sells
    orderbook.buy_orders = buys

    return orderbook

orderbooks = []
for i in range(len(DATASET)):
    row = DATASET.iloc[i]
    books = {'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D']}
    current_orderbook = {} # Turn the row data into Orderbook object
    for k, v in books.items():
        current_orderbook[k] = row_to_orderbook(v)
    orderbooks.append(current_orderbook)

def evaluate(trader_instance):
    try:

        current_positions = {"A":0,"B":0,"C":0,"D":0}
        trades = {"A":[],"B":[],"C":[],"D":[]}
        pnl = 0

        for i in range(len(DATASET)-1):
            current_orderbook = orderbooks[i]
            next_orderbook = orderbooks[i+1]

            # Data input for the trader's response
            state = {
                "orderbook" : current_orderbook,
                "positions": current_positions,
                "trades": trades
            }

            traders_orders = trader_instance.run(state)

            next_row = DATASET.iloc[i+1]
            best_ask = {}
            best_bid = {}
            worst_ask = {}
            worst_bid = {}
            for symbol in SYMBOLS:
                best_ask[symbol] = next_row[symbol]['Ask1']
                best_bid[symbol] = next_row[symbol]['Bid1']
                worst_ask[symbol] = next_row[symbol]['Ask5']
                worst_bid[symbol] = next_row[symbol]['Bid5']

            unfilled_bids = {}
            unfilled_asks = {}

            #processing regular trade orders
            for order in traders_orders:
                filled = False
                symbol = order.symbol
                price = order.price
                quantity = order.quantity
                side = order.side
                if side == "buy":
                    quantity = min(quantity, abs(POS_LIMIT - current_positions[symbol]))
                elif side == "sell":
                    quantity = min(quantity, abs(POS_LIMIT + current_positions[symbol]))
                try:
                    if side == "buy":
                        fill_size = min(next_orderbook[symbol].sell_orders[price], quantity)
                        pnl -= price * fill_size
                        current_positions[symbol] += fill_size
                        trades[symbol].append(Trade(price, fill_size, side))
                        filled = True
                    elif side == "sell":
                        fill_size = min(next_orderbook[symbol].buy_orders[price], quantity)
                        pnl += price * fill_size
                        current_positions[symbol] -= fill_size
                        trades[symbol].append(Trade(price, fill_size, side))
                        filled = True
                except KeyError:
                    if side == "buy" and price > worst_ask[symbol]:
                        pnl -= price * quantity
                        current_positions[symbol] += quantity
                        trades[symbol].append(Trade(price, quantity, side))
                        filled = True
                    elif side == "sell" and price < worst_bid[symbol]:
                        pnl += price * quantity
                        current_positions[symbol] -= quantity
                        trades[symbol].append(Trade(price, quantity, side))
                        filled = True

                if (not filled) and (side == "buy") and (price == best_bid[symbol]):
                    unfilled_bids[symbol] = (price, quantity)
                elif (not filled) and (side == "sell") and (price == best_ask[symbol]):
                    unfilled_asks[symbol] = (price, quantity)

            #calculate pnl from perfect market making orders
            for (k, v) in unfilled_bids.items():
                try:
                    pnl += ((unfilled_asks[k][0]-v[0])*min(v[1], unfilled_asks[k][1]))
                except KeyError:
                    continue

            #settling at mid if last row and positions remain
            if i == len(DATASET)-2:
                for symbol in SYMBOLS:
                    mid = (best_ask[symbol] + best_bid[symbol]) / 2
                    pnl += current_positions[symbol] * mid

        result = f"Final train PnL: {pnl}"
        return result
    except Exception as e:
        raise e

#BEST SO FAR - USE THIS!!!!

from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class Trader:
    def __init__(self):
        self.max_position = 100
        self.min_spread = 0.5
        self.k_inv = 0.15  #Very low inventory penalty
        self.k_skew = 0.6
        self.k_vol = 0.08
        self.threshold_signal = 0.6  #Low threshold for more trades
        self.lambda_risk = 1.0
        self.lookback = 40  #Longer lookback

        #tracks state
        self.history = {sym: [] for sym in SYMBOLS}
        self.features_buffer = {sym: [] for sym in SYMBOLS}
        self.models = {sym: None for sym in SYMBOLS}
        self.scalers = {sym: StandardScaler() for sym in SYMBOLS}
        self.tick_count = 0

        self.portfolio_value = 100000
        self.last_mid = {sym: None for sym in SYMBOLS}

        #Enhanced tracking
        self.avg_spread = {sym: 2.0 for sym in SYMBOLS}
        self.avg_volatility = {sym: 1.0 for sym in SYMBOLS}
        self.price_trends = {sym: [] for sym in SYMBOLS}
        self.imbalance_history = {sym: [] for sym in SYMBOLS}

        #multi-timeframe features
        self.fast_ema = {sym: None for sym in SYMBOLS}
        self.slow_ema = {sym: None for sym in SYMBOLS}

        #Correlation tracking
        self.returns_matrix = []

    def update_ema(self, symbol, price):

        if self.fast_ema[symbol] is None:
            self.fast_ema[symbol] = price
            self.slow_ema[symbol] = price
        else:
            # Fast EMA (alpha = 0.3)
            self.fast_ema[symbol] = 0.3 * price + 0.7 * self.fast_ema[symbol]
            # Slow EMA (alpha = 0.1)
            self.slow_ema[symbol] = 0.1 * price + 0.9 * self.slow_ema[symbol]

    def update_market_stats(self, symbol, spread, volatility, imbalance):
        alpha = 0.05
        self.avg_spread[symbol] = alpha * spread + (1 - alpha) * self.avg_spread[symbol]
        self.avg_volatility[symbol] = alpha * volatility + (1 - alpha) * self.avg_volatility[symbol]

        #Track imbalance history
        self.imbalance_history[symbol].append(imbalance)
        if len(self.imbalance_history[symbol]) > 20:
            self.imbalance_history[symbol].pop(0)

    def compute_features(self, symbol, orderbook, trades):
        if len(orderbook.buy_orders) > 0 and len(orderbook.sell_orders) > 0:
            best_bid = float(max(orderbook.buy_orders.keys()))
            best_ask = float(min(orderbook.sell_orders.keys()))
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

            #Deep book analysis - top 10 levels
            bid_prices = sorted(orderbook.buy_orders.keys(), reverse=True)[:10]
            ask_prices = sorted(orderbook.sell_orders.keys())[:10]

            bid_depth = sum(float(orderbook.buy_orders.get(p, 0)) for p in bid_prices)
            ask_depth = sum(float(orderbook.sell_orders.get(p, 0)) for p in ask_prices)

            #Level-by-level imbalance
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / (total_depth + 1e-8)

            #Top-of-book imbalance
            bid_size_top = float(orderbook.buy_orders.get(best_bid, 0))
            ask_size_top = float(orderbook.sell_orders.get(best_ask, 0))
            total_top = bid_size_top + ask_size_top
            tob_imbalance = (bid_size_top - ask_size_top) / (total_top + 1e-8)

            #Microprice
            microprice = (best_bid * ask_size_top + best_ask * bid_size_top) / (total_top + 1e-8)

            #Weighted mid (VWAP approximation)
            bid_vwap = sum(p * orderbook.buy_orders[p] for p in bid_prices[:5]) / (sum(orderbook.buy_orders[p] for p in bid_prices[:5]) + 1e-8)
            ask_vwap = sum(p * orderbook.sell_orders[p] for p in ask_prices[:5]) / (sum(orderbook.sell_orders[p] for p in ask_prices[:5]) + 1e-8)
            vwap_mid = (bid_vwap + ask_vwap) / 2.0

            features = [mid, spread, imbalance, tob_imbalance, microprice, bid_depth, ask_depth, vwap_mid]

            #Price changes at multiple horizons
            if self.last_mid[symbol] is not None:
                price_change = mid - self.last_mid[symbol]
                features.append(price_change)
            else:
                features.append(0.0)

            self.last_mid[symbol] = mid
            self.update_ema(symbol, mid)

            if self.fast_ema[symbol] is not None:
                ema_diff = self.fast_ema[symbol] - self.slow_ema[symbol]
                features.append(ema_diff)
            else:
                features.append(0.0)

            # Historical features
            self.history[symbol].append(float(mid))
            if len(self.history[symbol]) > self.lookback:
                self.history[symbol].pop(0)

            if len(self.history[symbol]) >= 5:
                history_array = np.array(self.history[symbol], dtype=np.float64)

                # Multiple horizon returns
                returns = np.diff(history_array)
                if len(returns) > 0:
                    vol = float(np.std(returns))
                    mean_return = float(np.mean(returns))

                    #Short-term momentum (last 3 bars)
                    if len(history_array) >= 3:
                        short_momentum = (history_array[-1] - history_array[-3]) / (history_array[-3] + 1e-8)
                    else:
                        short_momentum = 0.0

                    #Medium-term momentum (last 10 bars)
                    if len(history_array) >= 10:
                        medium_momentum = (history_array[-1] - history_array[-10]) / (history_array[-10] + 1e-8)
                    else:
                        medium_momentum = 0.0

                    #Volatility ratio (recent vs historical)
                    if len(returns) >= 10:
                        recent_vol = np.std(returns[-5:])
                        historical_vol = np.std(returns)
                        vol_ratio = recent_vol / (historical_vol + 1e-8)
                    else:
                        vol_ratio = 1.0

                    features.extend([vol, mean_return, short_momentum, medium_momentum, vol_ratio])

                    #Update market stats
                    self.update_market_stats(symbol, spread, vol, imbalance)
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 1.0])

            #Persistent imbalance feature
            if len(self.imbalance_history[symbol]) >= 5:
                avg_imbalance = np.mean(self.imbalance_history[symbol][-5:])
                features.append(avg_imbalance)
            else:
                features.append(0.0)

            return np.array(features, dtype=np.float64), mid, spread, imbalance, best_bid, best_ask, tob_imbalance

        return None, None, None, None, None, None, None

    def train_simple_model(self, symbol):
        if len(self.features_buffer[symbol]) < 20:
            return

        X = []
        y = []

        #Multi-step ahead prediction
        for i in range(len(self.features_buffer[symbol]) - 2):
            X.append(self.features_buffer[symbol][i])
            #Predict 1-step ahead return
            next_price = self.features_buffer[symbol][i+1][0]
            curr_price = self.features_buffer[symbol][i][0]
            y.append(float(next_price - curr_price))

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        X_scaled = self.scalers[symbol].fit_transform(X)

        #Light regularization for more signal
        self.models[symbol] = Ridge(alpha=0.3)
        self.models[symbol].fit(X_scaled, y)

    def predict_return(self, symbol, features):
        if self.models[symbol] is None or features is None:
            return 0.0, 1.0

        features_scaled = self.scalers[symbol].transform(features.reshape(1, -1))
        mu = float(self.models[symbol].predict(features_scaled)[0])


        mu = mu * 1.3

        if len(self.features_buffer[symbol]) > 10:
            recent = np.array([float(f[0]) for f in self.features_buffer[symbol][-10:]], dtype=np.float64)
            sigma = float(np.std(np.diff(recent)))
        else:
            sigma = 1.0

        return mu, max(sigma, 0.01)

    def adaptive_size(self, base_size, symbol, inventory, signal_strength, spread_ratio):

        #Scale by volatility (trade smaller in high vol)
        vol_factor = min(1.8, 1.0 / (self.avg_volatility[symbol] + 0.3))
        #Scale by signal strength
        signal_factor = min(1.8, 0.5 + abs(signal_strength))
        #Scale by spread (trade bigger in wide spreads)
        spread_factor = min(1.5, spread_ratio)
        #Scale by available capacity
        if signal_strength > 0:
            capacity = self.max_position - inventory
        else:
            capacity = inventory + self.max_position
        capacity_factor = min(1.0, capacity / 40.0)

        adjusted_size = int(base_size * vol_factor * signal_factor * spread_factor * capacity_factor)
        return max(8, min(adjusted_size, 35))

    def run(self, state):
        orders: List[Order] = []
        orderbook: Dict[str, Orderbook] = state["orderbook"]
        positions: Dict[str, int] = state["positions"]
        trades: Dict[str, List[Trade]] = state["trades"]

        self.tick_count += 1

        # Collect cross-asset signals
        all_signals = {}
        all_imbalances = {}

        for symbol in SYMBOLS:
            if symbol not in orderbook:
                continue

            features, mid, spread, imbalance, best_bid, best_ask, tob_imbalance = self.compute_features(
                symbol, orderbook[symbol], trades.get(symbol, [])
            )

            if features is None or mid is None:
                continue

            self.features_buffer[symbol].append(features)
            if len(self.features_buffer[symbol]) > 150:
                self.features_buffer[symbol].pop(0)

            #Train frequently for adaptation
            if self.tick_count % 25 == 0:
                self.train_simple_model(symbol)

            mu, sigma = self.predict_return(symbol, features)
            sharpe_signal = mu / (sigma + 1e-8)

            all_signals[symbol] = {
                'mu': mu,
                'sigma': sigma,
                'sharpe': sharpe_signal,
                'mid': mid,
                'spread': spread,
                'imbalance': imbalance,
                'tob_imbalance': tob_imbalance,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'features': features
            }
            all_imbalances[symbol] = imbalance

        #Generate orders for each symbol
        for symbol in SYMBOLS:
            if symbol not in all_signals:
                continue

            sig = all_signals[symbol]
            mid = sig['mid']
            spread = sig['spread']
            imbalance = sig['imbalance']
            tob_imbalance = sig['tob_imbalance']
            best_bid = sig['best_bid']
            best_ask = sig['best_ask']
            mu = sig['mu']
            sigma = sig['sigma']
            sharpe_signal = sig['sharpe']

            inventory = positions.get(symbol, 0)

            #Position limits with buffer
            can_buy = inventory < self.max_position - 2
            can_sell = inventory > -self.max_position + 2

            #Spread analysis
            spread_ratio = spread / (self.avg_spread[symbol] + 1e-8)

            #mULTI-LEVEL MARKET MAKING
            #Adaptive MM sizing based on conditions
            if spread_ratio > 1.3:  #Very wide spread
                mm_size_base = 30
            elif spread_ratio > 1.0:
                mm_size_base = 25
            else:
                mm_size_base = 20

            #Post at best for perfect MM
            if can_buy and inventory < 92:
                orders.append(Order(symbol, int(best_bid), 'buy', mm_size_base))

            if can_sell and inventory > -92:
                orders.append(Order(symbol, int(best_ask), 'sell', mm_size_base))

            #Post inside spread aggressively
            if spread >= 2:
                inside_size = mm_size_base // 2
                if can_buy and inventory < 80:
                    orders.append(Order(symbol, int(best_bid + 1), 'buy', inside_size))
                if can_sell and inventory > -80:
                    orders.append(Order(symbol, int(best_ask - 1), 'sell', inside_size))

            if spread >= 4:
                deeper_size = mm_size_base // 3
                if can_buy and inventory < 70:
                    orders.append(Order(symbol, int(best_bid + 2), 'buy', deeper_size))
                if can_sell and inventory > -70:
                    orders.append(Order(symbol, int(best_ask - 2), 'sell', deeper_size))

            #SIGNAL-BASED DIRECTIONAL
            #Very strong signals
            if sharpe_signal > 1.8 and can_buy:
                size = self.adaptive_size(25, symbol, inventory, sharpe_signal, spread_ratio)
                if inventory < 75 and size > 0:
                    orders.append(Order(symbol, int(best_ask), 'buy', size))
                    if sharpe_signal > 2.5 and inventory < 50:
                        orders.append(Order(symbol, int(best_ask + 1), 'buy', size // 2))

            elif sharpe_signal < -1.8 and can_sell:
                size = self.adaptive_size(25, symbol, inventory, abs(sharpe_signal), spread_ratio)
                if inventory > -75 and size > 0:
                    orders.append(Order(symbol, int(best_bid), 'sell', size))
                    if sharpe_signal < -2.5 and inventory > -50:
                        orders.append(Order(symbol, int(best_bid - 1), 'sell', size // 2))

            #Strong signals
            elif sharpe_signal > 1.0 and sharpe_signal <= 1.8 and can_buy:
                size = self.adaptive_size(18, symbol, inventory, sharpe_signal, spread_ratio)
                if inventory < 65 and size > 0:
                    orders.append(Order(symbol, int(best_ask), 'buy', size))

            elif sharpe_signal < -1.0 and sharpe_signal >= -1.8 and can_sell:
                size = self.adaptive_size(18, symbol, inventory, abs(sharpe_signal), spread_ratio)
                if inventory > -65 and size > 0:
                    orders.append(Order(symbol, int(best_bid), 'sell', size))

            #Moderate signals
            elif sharpe_signal > 0.6 and sharpe_signal <= 1.0 and can_buy:
                size = self.adaptive_size(12, symbol, inventory, sharpe_signal, spread_ratio)
                if inventory < 50 and size > 0:
                    orders.append(Order(symbol, int(best_ask), 'buy', size))

            elif sharpe_signal < -0.6 and sharpe_signal >= -1.0 and can_sell:
                size = self.adaptive_size(12, symbol, inventory, abs(sharpe_signal), spread_ratio)
                if inventory > -50 and size > 0:
                    orders.append(Order(symbol, int(best_bid), 'sell', size))

            # ORDER BOOK IMBALANCE (Combined signals)
            combined_imbalance = (imbalance + tob_imbalance) / 2.0

            if combined_imbalance > 0.4 and can_buy:
                size = int(12 + 15 * min(1.0, abs(combined_imbalance)))
                if inventory < 60:
                    orders.append(Order(symbol, int(best_bid), 'buy', size))
                    if len(self.imbalance_history[symbol]) >= 5:
                        recent_imb = self.imbalance_history[symbol][-5:]
                        if all(x > 0.3 for x in recent_imb) and inventory < 45:
                            orders.append(Order(symbol, int(best_ask), 'buy', 10))

            elif combined_imbalance < -0.4 and can_sell:
                size = int(12 + 15 * min(1.0, abs(combined_imbalance)))
                if inventory > -60:
                    orders.append(Order(symbol, int(best_ask), 'sell', size))
                    if len(self.imbalance_history[symbol]) >= 5:
                        recent_imb = self.imbalance_history[symbol][-5:]
                        if all(x < -0.3 for x in recent_imb) and inventory > -45:
                            orders.append(Order(symbol, int(best_bid), 'sell', 10))

            # ema crossover
            if self.fast_ema[symbol] is not None and self.slow_ema[symbol] is not None:
                ema_diff = self.fast_ema[symbol] - self.slow_ema[symbol]
                ema_signal = ema_diff / (self.slow_ema[symbol] + 1e-8)

                if ema_signal > 0.002 and can_buy and inventory < 55:
                    size = self.adaptive_size(14, symbol, inventory, 1.0, spread_ratio)
                    orders.append(Order(symbol, int(best_ask), 'buy', size))

                elif ema_signal < -0.002 and can_sell and inventory > -55:
                    size = self.adaptive_size(14, symbol, inventory, 1.0, spread_ratio)
                    orders.append(Order(symbol, int(best_bid), 'sell', size))

            # MULTI-TIMEFRAME MOMENTUM
            if len(self.history[symbol]) >= 15:
                recent_prices = self.history[symbol]

                # Short momentum (3 bars)
                short_mom = (recent_prices[-1] - recent_prices[-3]) / (recent_prices[-3] + 1e-8)
                # Medium momentum (8 bars)
                medium_mom = (recent_prices[-1] - recent_prices[-8]) / (recent_prices[-8] + 1e-8)
                # Long momentum (15 bars)
                long_mom = (recent_prices[-1] - recent_prices[-15]) / (recent_prices[-15] + 1e-8)

                # All timeframes agree on direction
                if short_mom > 0.0008 and medium_mom > 0.001 and long_mom > 0.0015:
                    if can_buy and inventory < 60:
                        size = self.adaptive_size(16, symbol, inventory, 1.2, spread_ratio)
                        orders.append(Order(symbol, int(best_ask), 'buy', size))

                elif short_mom < -0.0008 and medium_mom < -0.001 and long_mom < -0.0015:
                    if can_sell and inventory > -60:
                        size = self.adaptive_size(16, symbol, inventory, 1.2, spread_ratio)
                        orders.append(Order(symbol, int(best_bid), 'sell', size))

            # mean reversion strat
            if len(self.history[symbol]) >= 20:
                recent_avg = np.mean(self.history[symbol][-20:])
                deviation = (mid - recent_avg) / (recent_avg + 1e-8)

                # Strong deviation - fade it
                if deviation > 0.004 and can_sell and inventory > -55:
                    orders.append(Order(symbol, int(best_bid), 'sell', 18))

                elif deviation < -0.004 and can_buy and inventory < 55:
                    orders.append(Order(symbol, int(best_ask), 'buy', 18))

            # voltalityl breakoit strat
            if len(self.history[symbol]) >= 10:
                recent_vol = self.avg_volatility[symbol]
                returns = np.diff(self.history[symbol][-10:])
                current_return = returns[-1] if len(returns) > 0 else 0

                # Large move relative to recent vol - ride it
                if abs(current_return) > 2.0 * recent_vol:
                    if current_return > 0 and can_buy and inventory < 50:
                        orders.append(Order(symbol, int(best_ask), 'buy', 15))
                    elif current_return < 0 and can_sell and inventory > -50:
                        orders.append(Order(symbol, int(best_bid), 'sell', 15))

            #control aggressive inventory
            if inventory > 82:
                unwind_size = min(30, inventory - 70)
                orders.append(Order(symbol, int(best_ask), 'sell', unwind_size))
                if inventory > 90:
                    orders.append(Order(symbol, int(best_ask - 1), 'sell', 20))
                    orders.append(Order(symbol, int(best_bid), 'sell', 15))

            elif inventory < -82:
                unwind_size = min(30, -inventory - 70)
                orders.append(Order(symbol, int(best_bid), 'buy', unwind_size))
                if inventory < -90:
                    orders.append(Order(symbol, int(best_bid + 1), 'buy', 20))
                    orders.append(Order(symbol, int(best_ask), 'buy', 15))

        return orders
