import asyncio
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import ccxt.async_support as ccxt

from .detector import ShortTicker
from .paper_exchange import PaperExchange

@dataclass
class PaperArbitrageResult:
    success: bool
    initial_balance: float
    final_balance: float
    profit_loss: float
    profit_percentage: float
    trades_executed: int
    total_fees: float
    execution_time: float
    opportunities_used: List[ShortTicker]
    error_message: Optional[str] = None

class PaperArbitrageExecutor:
    def __init__(self, paper_exchange: PaperExchange, real_exchange_name: str = "binanceus"):
        self.paper_exchange = paper_exchange
        self.real_exchange_name = real_exchange_name
        self.real_exchange = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize_real_exchange(self):
        """Initialize real exchange for price data"""
        exchange_class = getattr(ccxt, self.real_exchange_name)
        self.real_exchange = exchange_class({'enableRateLimit': True})
    
    async def close_real_exchange(self):
        """Close real exchange connection"""
        if self.real_exchange:
            await self.real_exchange.close()
    
    async def execute_arbitrage_cycle(self, opportunities: List[ShortTicker], 
                                    base_currency: str = 'USDT') -> PaperArbitrageResult:
        """Execute a complete arbitrage cycle in paper trading"""
        start_time = datetime.now()
        
        # Get initial balance
        initial_balances = await self.paper_exchange.fetch_balance()
        initial_balance = initial_balances.get(base_currency, {}).get('free', 0)
        
        if initial_balance < 100:  # Minimum $100 for arbitrage
            return PaperArbitrageResult(
                success=False,
                initial_balance=initial_balance,
                final_balance=initial_balance,
                profit_loss=0,
                profit_percentage=0,
                trades_executed=0,
                total_fees=0,
                execution_time=0,
                opportunities_used=opportunities,
                error_message="Insufficient balance for arbitrage"
            )
        
        # Use a portion of available balance (e.g., 10%)
        trade_amount = min(initial_balance * 0.1, 1000)  # Max $1000 per trade
        
        trades_executed = 0
        total_fees = 0
        
        try:
            # Execute each step in the arbitrage cycle
            current_amount = trade_amount
            
            for i, opportunity in enumerate(opportunities):
                # Get current real market price
                current_price = await self._get_current_price(opportunity.symbol)
                
                # Determine trade direction and amount
                if opportunity.reversed:
                    # Sell: we have base currency, selling for quote currency
                    side = 'sell'
                    symbol = str(opportunity.symbol)
                    amount = current_amount
                else:
                    # Buy: we have quote currency, buying base currency
                    side = 'buy'
                    symbol = str(opportunity.symbol)
                    amount = current_amount / current_price
                
                # Execute the trade
                order = await self.paper_exchange.create_market_order(
                    symbol, side, amount, current_price
                )
                
                trades_executed += 1
                total_fees += order.fee
                
                # Update current amount for next trade
                if side == 'buy':
                    current_amount = order.filled
                else:
                    current_amount = order.filled * order.price - order.fee
                
                # Small delay to simulate real trading
                await asyncio.sleep(0.1)
                
                self.logger.info(f"Step {i+1}: {side} {amount:.6f} {symbol} at {current_price:.6f} -> {current_amount:.6f}")
            
            # Calculate final results
            final_balances = await self.paper_exchange.fetch_balance()
            final_balance = final_balances.get(base_currency, {}).get('free', 0)
            
            profit_loss = final_balance - initial_balance
            profit_percentage = (profit_loss / initial_balance) * 100 if initial_balance > 0 else 0
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PaperArbitrageResult(
                success=True,
                initial_balance=initial_balance,
                final_balance=final_balance,
                profit_loss=profit_loss,
                profit_percentage=profit_percentage,
                trades_executed=trades_executed,
                total_fees=total_fees,
                execution_time=execution_time,
                opportunities_used=opportunities
            )
            
        except Exception as e:
            self.logger.error(f"Arbitrage execution failed: {e}")
            
            # Get current balance after failure
            current_balances = await self.paper_exchange.fetch_balance()
            current_balance = current_balances.get(base_currency, {}).get('free', 0)
            
            return PaperArbitrageResult(
                success=False,
                initial_balance=initial_balance,
                final_balance=current_balance,
                profit_loss=current_balance - initial_balance,
                profit_percentage=0,
                trades_executed=trades_executed,
                total_fees=total_fees,
                execution_time=(datetime.now() - start_time).total_seconds(),
                opportunities_used=opportunities,
                error_message=str(e)
            )
    
    async def _get_current_price(self, symbol) -> float:
        """Get current market price from real exchange"""
        try:
            if self.real_exchange is None:
                self.logger.warning(f"Real exchange not initialized, using default price for {symbol}")
                return 1.0
            
            ticker = await self.real_exchange.fetch_ticker(str(symbol))
            return ticker['last']
        except Exception as e:
            self.logger.warning(f"Failed to get current price for {symbol}: {e}")
            # Fall back to a reasonable default (this should be handled better in production)
            return 1.0