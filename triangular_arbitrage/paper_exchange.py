import asyncio
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from decimal import Decimal
import json
from datetime import datetime

@dataclass
class PaperBalance:
    currency: str
    free: float
    used: float = 0.0
    
    @property
    def total(self) -> float:
        return self.free + self.used

@dataclass
class PaperOrder:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    filled: float = 0.0
    status: str = 'open'  # 'open', 'filled', 'cancelled'
    timestamp: datetime = field(default_factory=datetime.now)
    fee: float = 0.0

@dataclass
class TradeExecution:
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    fee: float
    timestamp: datetime
    slippage: float = 0.0

class PaperExchange:
    def __init__(self, initial_balances: Optional[Dict[str, float]] = None, 
                 trading_fee: float = 0.001, max_slippage: float = 0.002):
        """
        Initialize paper trading exchange
        
        Args:
            initial_balances: Starting balances for each currency
            trading_fee: Trading fee percentage (0.001 = 0.1%)
            max_slippage: Maximum slippage percentage (0.002 = 0.2%)
        """
        self.balances = {}
        self.trading_fee = trading_fee
        self.max_slippage = max_slippage
        self.orders = {}
        self.trade_history = []
        self.order_counter = 0
        self.logger = logging.getLogger(__name__)
        
        # Set default balances
        default_balances = initial_balances or {
            'USDT': 10000.0,  # Start with $10,000 USDT
            'BTC': 0.0,
            'ETH': 0.0,
            'BNB': 0.0
        }
        
        for currency, amount in default_balances.items():
            self.balances[currency] = PaperBalance(currency, amount)
    
    def get_next_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"paper_order_{self.order_counter}_{int(time.time())}"
    
    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """Get account balances in ccxt format"""
        result = {'info': {}}
        
        for currency, balance in self.balances.items():
            result[currency] = {
                'free': balance.free,
                'used': balance.used,
                'total': balance.total
            }
            result['free'] = result.get('free', {})
            result['used'] = result.get('used', {})
            result['total'] = result.get('total', {})
            result['free'][currency] = balance.free
            result['used'][currency] = balance.used
            result['total'][currency] = balance.total
        
        return result
    
    async def create_market_order(self, symbol: str, side: str, amount: float, 
                                current_price: float) -> PaperOrder:
        """Simulate placing a market order"""
        order_id = self.get_next_order_id()
        
        # Simulate slippage (random between 0 and max_slippage)
        import random
        slippage_factor = random.uniform(0, self.max_slippage)
        if side == 'buy':
            execution_price = current_price * (1 + slippage_factor)
        else:
            execution_price = current_price * (1 - slippage_factor)
        
        # Calculate fee
        fee_amount = amount * execution_price * self.trading_fee
        
        # Create order
        order = PaperOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            amount=amount,
            price=execution_price,
            filled=amount,
            status='filled',
            fee=fee_amount
        )
        
        # Execute the trade
        await self._execute_order(order, slippage_factor)
        
        self.orders[order_id] = order
        self.logger.info(f"Paper order executed: {side} {amount} {symbol} at {execution_price:.6f}")
        
        return order
    
    async def _execute_order(self, order: PaperOrder, slippage: float):
        """Execute the order and update balances"""
        base_currency, quote_currency = order.symbol.split('/')
        
        if order.side == 'buy':
            # Buying base currency with quote currency
            total_cost = order.amount * order.price + order.fee
            
            # Check if we have enough quote currency
            if self.balances.get(quote_currency, PaperBalance(quote_currency, 0)).free < total_cost:
                raise Exception(f"Insufficient {quote_currency} balance")
            
            # Update balances
            self.balances[quote_currency].free -= total_cost
            if base_currency not in self.balances:
                self.balances[base_currency] = PaperBalance(base_currency, 0)
            self.balances[base_currency].free += order.amount
            
        else:  # sell
            # Selling base currency for quote currency
            # Check if we have enough base currency
            if self.balances.get(base_currency, PaperBalance(base_currency, 0)).free < order.amount:
                raise Exception(f"Insufficient {base_currency} balance")
            
            # Update balances
            self.balances[base_currency].free -= order.amount
            if quote_currency not in self.balances:
                self.balances[quote_currency] = PaperBalance(quote_currency, 0)
            self.balances[quote_currency].free += (order.amount * order.price) - order.fee
        
        # Record trade
        trade = TradeExecution(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            amount=order.amount,
            price=order.price,
            fee=order.fee,
            timestamp=datetime.now(),
            slippage=slippage
        )
        self.trade_history.append(trade)
    
    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Get order details"""
        if order_id not in self.orders:
            raise Exception(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        return {
            'id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'amount': order.amount,
            'price': order.price,
            'filled': order.filled,
            'status': order.status,
            'fee': {'cost': order.fee, 'currency': order.symbol.split('/')[1]},
            'timestamp': order.timestamp.timestamp() * 1000
        }
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value in USDT"""
        total_value = 0.0
        
        for currency, balance in self.balances.items():
            if currency == 'USDT':
                total_value += balance.total
            else:
                # Convert to USDT using current price
                price_key = f"{currency}/USDT"
                if price_key in current_prices:
                    total_value += balance.total * current_prices[price_key]
        
        return total_value
    
    def get_trade_summary(self) -> Dict:
        """Get trading performance summary"""
        if not self.trade_history:
            return {'total_trades': 0, 'total_fees': 0, 'avg_slippage': 0}
        
        total_fees = sum(trade.fee for trade in self.trade_history)
        avg_slippage = sum(trade.slippage for trade in self.trade_history) / len(self.trade_history)
        
        return {
            'total_trades': len(self.trade_history),
            'total_fees': total_fees,
            'avg_slippage': avg_slippage,
            'trade_history': self.trade_history
        }
    
    def reset_account(self, initial_balances: Optional[Dict[str, float]] = None):
        """Reset account to initial state"""
        self.__init__(initial_balances, self.trading_fee, self.max_slippage)