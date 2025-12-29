"""File containing extra classes for structuring data"""
from typing import Dict, List

class Orderbook:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {} # price : quantity
        self.sell_orders: Dict[int, int] = {} # price : quantity

class Order:
    def __init__(self, symbol, price, side, quantity):
        self.symbol = symbol
        self.price = price
        self.side = side
        self.quantity = quantity

class Trade:
    def __init__(self, price, side, quantity): 
        #changed here to allow defining variables upon initialisation
        self.price = price
        self.side = side
        self.quantity = quantity