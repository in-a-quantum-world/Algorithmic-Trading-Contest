import pandas as pd
from extra_classes import Orderbook, Trade, Order
from pprint import pprint

DATASET = pd.read_csv("train.csv", header=[0, 1], index_col=0) # FIX THIS
SYMBOLS = ["A","B","C","D"]
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
        """
        Put evaluation code here.
        """
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
            
        result = f"Final train PnL: {pnl}" # Change this if you want. It has to be a string/int
        return result
    except Exception as e:
        raise e
        
