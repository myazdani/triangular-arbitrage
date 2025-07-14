# Live Trading Implementation Guide

## ⚠️ **CRITICAL WARNING**
Adding live trading capabilities introduces significant financial risk. This implementation could result in:
- **Financial losses** due to market volatility, slippage, or bugs
- **Exchange account suspension** if not properly implemented
- **Regulatory compliance issues** depending on your jurisdiction

**NEVER deploy this without extensive testing on testnet/sandbox environments first.**

## Architecture Overview

### Current State
- **Read-only market data fetching**
- **Theoretical opportunity detection**
- **No trading capabilities**

### Required Additions
1. **Authentication & Credentials Management**
2. **Account & Balance Management**
3. **Order Execution Engine**
4. **Risk Management System**
5. **Error Handling & Recovery**
6. **Performance Monitoring**
7. **Safety Mechanisms**

## Implementation Plan

### Phase 1: Core Trading Infrastructure

#### 1. Create Exchange Manager
```python
# triangular_arbitrage/exchange_manager.py
import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

@dataclass
class ExchangeConfig:
    exchange_name: str
    api_key: str
    secret: str
    sandbox: bool = True
    passphrase: Optional[str] = None  # For some exchanges like Coinbase Pro

@dataclass
class Balance:
    currency: str
    free: float
    used: float
    total: float

class ExchangeManager:
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize exchange connection with credentials"""
        exchange_class = getattr(ccxt, self.config.exchange_name)
        self.exchange = exchange_class({
            'apiKey': self.config.api_key,
            'secret': self.config.secret,
            'sandbox': self.config.sandbox,
            'enableRateLimit': True,
        })
        if self.config.passphrase:
            self.exchange.passphrase = self.config.passphrase
        
        # Test connection
        try:
            await self.exchange.fetch_balance()
            self.logger.info(f"Successfully connected to {self.config.exchange_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.config.exchange_name}: {e}")
            raise
    
    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances"""
        try:
            balance_data = await self.exchange.fetch_balance()
            balances = {}
            for currency, data in balance_data.items():
                if currency not in ['info', 'free', 'used', 'total']:
                    balances[currency] = Balance(
                        currency=currency,
                        free=data.get('free', 0),
                        used=data.get('used', 0),
                        total=data.get('total', 0)
                    )
            return balances
        except Exception as e:
            self.logger.error(f"Failed to fetch balances: {e}")
            raise
    
    async def place_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """Place a market order"""
        try:
            self.logger.info(f"Placing {side} order: {amount} {symbol}")
            order = await self.exchange.create_market_order(symbol, side, amount)
            return order
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
```

#### 2. Create Order Execution Engine
```python
# triangular_arbitrage/order_executor.py
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from decimal import Decimal
import logging

@dataclass
class TradeOrder:
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    expected_price: float
    max_slippage: float = 0.02  # 2% max slippage

@dataclass
class ExecutionResult:
    success: bool
    executed_orders: List[dict]
    total_cost: float
    actual_profit: float
    error_message: Optional[str] = None

class OrderExecutor:
    def __init__(self, exchange_manager, risk_manager):
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
    
    async def execute_arbitrage_cycle(self, opportunities: List[ShortTicker], 
                                    initial_amount: float) -> ExecutionResult:
        """Execute a complete arbitrage cycle"""
        executed_orders = []
        current_amount = initial_amount
        
        try:
            # Check if we have sufficient balance
            if not await self.risk_manager.check_balance_sufficient(
                opportunities[0].symbol.quote, initial_amount):
                return ExecutionResult(
                    success=False,
                    executed_orders=[],
                    total_cost=0,
                    actual_profit=0,
                    error_message="Insufficient balance"
                )
            
            # Execute each trade in the cycle
            for i, opportunity in enumerate(opportunities):
                order = await self._execute_single_trade(opportunity, current_amount)
                executed_orders.append(order)
                
                # Update amount for next trade
                current_amount = float(order['filled'])
                
                # Add small delay to avoid rate limits
                await asyncio.sleep(0.1)
            
            # Calculate actual profit
            actual_profit = current_amount - initial_amount
            
            return ExecutionResult(
                success=True,
                executed_orders=executed_orders,
                total_cost=initial_amount,
                actual_profit=actual_profit
            )
            
        except Exception as e:
            self.logger.error(f"Arbitrage execution failed: {e}")
            return ExecutionResult(
                success=False,
                executed_orders=executed_orders,
                total_cost=initial_amount,
                actual_profit=0,
                error_message=str(e)
            )
    
    async def _execute_single_trade(self, opportunity: ShortTicker, amount: float) -> dict:
        """Execute a single trade in the arbitrage cycle"""
        symbol = str(opportunity.symbol)
        side = 'sell' if opportunity.reversed else 'buy'
        
        # Get current market price to check for slippage
        current_ticker = await self.exchange_manager.exchange.fetch_ticker(symbol)
        current_price = current_ticker['last']
        
        # Check for excessive slippage
        price_diff = abs(current_price - opportunity.last_price) / opportunity.last_price
        if price_diff > 0.02:  # 2% slippage threshold
            raise Exception(f"Excessive slippage detected: {price_diff:.2%}")
        
        # Calculate trade amount
        if side == 'buy':
            trade_amount = amount / current_price
        else:
            trade_amount = amount
        
        # Execute the trade
        order = await self.exchange_manager.place_market_order(symbol, side, trade_amount)
        
        # Wait for order to be filled
        await self._wait_for_order_fill(order['id'], symbol)
        
        # Get final order details
        final_order = await self.exchange_manager.exchange.fetch_order(order['id'], symbol)
        return final_order
    
    async def _wait_for_order_fill(self, order_id: str, symbol: str, timeout: int = 30):
        """Wait for order to be filled"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            order = await self.exchange_manager.exchange.fetch_order(order_id, symbol)
            
            if order['status'] == 'closed':
                return order
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise Exception(f"Order {order_id} timeout")
            
            await asyncio.sleep(0.5)
```

#### 3. Create Risk Management System
```python
# triangular_arbitrage/risk_manager.py
from typing import Dict, List
from dataclasses import dataclass
import logging

@dataclass
class RiskLimits:
    max_position_size: float = 1000.0  # USD
    max_daily_loss: float = 500.0  # USD
    min_profit_threshold: float = 0.005  # 0.5%
    max_slippage: float = 0.02  # 2%
    max_cycle_length: int = 5

class RiskManager:
    def __init__(self, exchange_manager, limits: RiskLimits):
        self.exchange_manager = exchange_manager
        self.limits = limits
        self.daily_pnl = 0.0
        self.logger = logging.getLogger(__name__)
    
    async def check_opportunity_valid(self, opportunities: List[ShortTicker], 
                                    profit_rate: float) -> bool:
        """Check if opportunity meets risk criteria"""
        
        # Check minimum profit threshold
        if profit_rate - 1 < self.limits.min_profit_threshold:
            self.logger.info(f"Profit {profit_rate-1:.2%} below threshold {self.limits.min_profit_threshold:.2%}")
            return False
        
        # Check cycle length
        if len(opportunities) > self.limits.max_cycle_length:
            self.logger.info(f"Cycle length {len(opportunities)} exceeds limit {self.limits.max_cycle_length}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.limits.max_daily_loss:
            self.logger.warning("Daily loss limit reached")
            return False
        
        return True
    
    async def check_balance_sufficient(self, currency: str, amount: float) -> bool:
        """Check if we have sufficient balance"""
        balances = await self.exchange_manager.get_balances()
        
        if currency not in balances:
            return False
        
        return balances[currency].free >= amount
    
    def calculate_position_size(self, opportunities: List[ShortTicker]) -> float:
        """Calculate safe position size"""
        # Start with maximum position size
        position_size = self.limits.max_position_size
        
        # Adjust based on cycle length (more steps = more risk)
        risk_multiplier = 1.0 / len(opportunities)
        position_size *= risk_multiplier
        
        # Ensure we don't exceed remaining daily limit
        remaining_daily_limit = self.limits.max_daily_loss + self.daily_pnl
        position_size = min(position_size, remaining_daily_limit * 0.1)  # Risk 10% of remaining limit
        
        return max(position_size, 10.0)  # Minimum $10 position
    
    def update_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl
        self.logger.info(f"Daily P&L updated: ${self.daily_pnl:.2f}")
```

### Phase 2: Enhanced Detection with Trading Integration

#### 4. Modify Main Detection Loop
```python
# Modified main.py for live trading
import asyncio
import logging
from triangular_arbitrage.detector import run_detection
from triangular_arbitrage.exchange_manager import ExchangeManager, ExchangeConfig
from triangular_arbitrage.order_executor import OrderExecutor
from triangular_arbitrage.risk_manager import RiskManager, RiskLimits

async def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize exchange manager
    config = ExchangeConfig(
        exchange_name="binanceus",
        api_key="YOUR_API_KEY",
        secret="YOUR_SECRET_KEY",
        sandbox=True  # ALWAYS start with sandbox/testnet
    )
    
    exchange_manager = ExchangeManager(config)
    await exchange_manager.initialize()
    
    # Initialize risk manager
    risk_limits = RiskLimits(
        max_position_size=100.0,  # Start small
        max_daily_loss=50.0,
        min_profit_threshold=0.01  # 1% minimum profit
    )
    risk_manager = RiskManager(exchange_manager, risk_limits)
    
    # Initialize order executor
    order_executor = OrderExecutor(exchange_manager, risk_manager)
    
    try:
        while True:
            logger.info("Scanning for opportunities...")
            
            # Detect opportunities
            opportunities, profit_rate = await run_detection(
                config.exchange_name,
                max_cycle=5  # Limit cycle length for faster execution
            )
            
            if opportunities and await risk_manager.check_opportunity_valid(opportunities, profit_rate):
                logger.info(f"Found opportunity with {profit_rate-1:.2%} profit")
                
                # Calculate position size
                position_size = risk_manager.calculate_position_size(opportunities)
                
                # Execute arbitrage
                result = await order_executor.execute_arbitrage_cycle(opportunities, position_size)
                
                if result.success:
                    logger.info(f"Arbitrage executed successfully! Profit: ${result.actual_profit:.2f}")
                    risk_manager.update_pnl(result.actual_profit)
                else:
                    logger.error(f"Arbitrage execution failed: {result.error_message}")
                    risk_manager.update_pnl(-result.total_cost * 0.01)  # Assume 1% loss on failure
            
            # Wait before next scan
            await asyncio.sleep(5)  # 5 second intervals
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await exchange_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 3: Advanced Features

#### 5. Add Configuration Management
```python
# triangular_arbitrage/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingConfig:
    # Exchange settings
    exchange_name: str
    api_key: str
    secret: str
    sandbox: bool = True
    
    # Risk settings
    max_position_size: float = 1000.0
    max_daily_loss: float = 500.0
    min_profit_threshold: float = 0.005
    
    # Execution settings
    scan_interval: int = 5  # seconds
    max_cycle_length: int = 5
    max_slippage: float = 0.02
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        return cls(
            exchange_name=os.getenv('EXCHANGE_NAME', 'binanceus'),
            api_key=os.getenv('API_KEY'),
            secret=os.getenv('SECRET_KEY'),
            sandbox=os.getenv('SANDBOX', 'true').lower() == 'true',
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '1000')),
            max_daily_loss=float(os.getenv('MAX_DAILY_LOSS', '500')),
            min_profit_threshold=float(os.getenv('MIN_PROFIT_THRESHOLD', '0.005')),
        )
```

#### 6. Add Performance Monitoring
```python
# triangular_arbitrage/monitor.py
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class TradeMetrics:
    timestamp: datetime
    opportunity_count: int
    executed_trades: int
    total_profit: float
    avg_execution_time: float
    success_rate: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.start_time = time.time()
    
    def record_trade(self, execution_time: float, profit: float, success: bool):
        """Record trade execution metrics"""
        # Implementation here
        pass
    
    def get_daily_summary(self) -> dict:
        """Get daily performance summary"""
        # Implementation here
        pass
    
    def save_metrics(self, filename: str):
        """Save metrics to file"""
        with open(filename, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, default=str)
```

## Safety Mechanisms

### 1. Environment Variables for Secrets
```bash
# .env file
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
SANDBOX=true
MAX_POSITION_SIZE=100
MAX_DAILY_LOSS=50
```

### 2. Circuit Breakers
```python
# Add to risk_manager.py
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_open = False
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logging.warning("Circuit breaker opened - stopping trading")
    
    def can_execute(self) -> bool:
        if not self.is_open:
            return True
        
        if time.time() - self.last_failure_time > self.recovery_time:
            self.is_open = False
            self.failure_count = 0
            logging.info("Circuit breaker reset")
            return True
        
        return False
```

## Testing Strategy

### 1. Unit Tests
```python
# tests/test_live_trading.py
import pytest
from unittest.mock import Mock, AsyncMock
from triangular_arbitrage.order_executor import OrderExecutor

@pytest.mark.asyncio
async def test_order_execution():
    # Mock exchange manager
    exchange_manager = Mock()
    exchange_manager.place_market_order = AsyncMock(return_value={
        'id': '12345',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'filled': 100
    })
    
    # Test order executor
    executor = OrderExecutor(exchange_manager, Mock())
    # Add test implementation
```

### 2. Integration Tests
```python
# tests/test_integration.py
@pytest.mark.asyncio
async def test_full_arbitrage_cycle():
    # Test complete arbitrage cycle on testnet
    pass
```

### 3. Backtesting
```python
# triangular_arbitrage/backtester.py
class Backtester:
    def __init__(self, historical_data):
        self.historical_data = historical_data
    
    def run_backtest(self, start_date, end_date):
        # Simulate trading on historical data
        pass
```

## Deployment Considerations

### 1. Infrastructure Requirements
- **Low-latency VPS** near exchange servers
- **Monitoring and alerting** system
- **Database** for trade history and metrics
- **Backup and recovery** procedures

### 2. Security Best Practices
- **API key restrictions** (IP whitelisting, trading-only permissions)
- **Secret management** (AWS Secrets Manager, HashiCorp Vault)
- **Audit logging** for all trades
- **Regular security updates**

### 3. Regulatory Compliance
- **Know Your Customer (KYC)** requirements
- **Anti-Money Laundering (AML)** compliance
- **Tax reporting** obligations
- **Licensing requirements** in your jurisdiction

## Implementation Timeline

### Week 1-2: Foundation
- [ ] Set up development environment
- [ ] Implement ExchangeManager and basic authentication
- [ ] Create comprehensive test suite

### Week 3-4: Core Trading
- [ ] Implement OrderExecutor
- [ ] Add RiskManager with basic limits
- [ ] Test on sandbox/testnet extensively

### Week 5-6: Safety & Monitoring
- [ ] Add circuit breakers and emergency stops
- [ ] Implement performance monitoring
- [ ] Create alerting system

### Week 7-8: Testing & Optimization
- [ ] Extensive backtesting
- [ ] Performance optimization
- [ ] Security audit

### Week 9-10: Deployment
- [ ] Deploy to production environment
- [ ] Start with minimal position sizes
- [ ] Monitor and adjust parameters

## Final Warnings

1. **Start small**: Begin with tiny position sizes
2. **Use testnet**: Extensively test on sandbox environments
3. **Monitor constantly**: Never leave automated trading unattended
4. **Have kill switches**: Always have manual override capabilities
5. **Legal compliance**: Ensure compliance with local regulations
6. **Insurance**: Consider getting appropriate insurance coverage

Remember: **Arbitrage opportunities are rare and fleeting**. By the time you detect and execute, the opportunity may have disappeared. Professional arbitrage requires millisecond execution times and sophisticated infrastructure.