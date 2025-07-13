import asyncio
import logging
from datetime import datetime
import json

from triangular_arbitrage.detector import run_detection
from triangular_arbitrage.paper_exchange import PaperExchange
from triangular_arbitrage.paper_executor import PaperArbitrageExecutor

class PaperTradingBot:
    def __init__(self, exchange_name: str = "binanceus"):
        self.exchange_name = exchange_name
        self.paper_exchange = PaperExchange(
            initial_balances={'USDT': 10000.0},  # Start with $10,000
            trading_fee=0.001,  # 0.1% trading fee
            max_slippage=0.002  # 0.2% max slippage
        )
        self.executor = PaperArbitrageExecutor(self.paper_exchange, exchange_name)
        self.logger = logging.getLogger(__name__)
        self.session_results = []
    
    async def run_session(self, duration_minutes: int = 60, scan_interval: int = 30):
        """Run a paper trading session"""
        await self.executor.initialize_real_exchange()
        
        start_time = datetime.now()
        end_time = start_time.timestamp() + (duration_minutes * 60)
        
        self.logger.info(f"Starting paper trading session for {duration_minutes} minutes")
        self.logger.info(f"Initial balance: ${self.paper_exchange.balances['USDT'].free:.2f}")
        
        opportunities_found = 0
        trades_executed = 0
        
        try:
            while datetime.now().timestamp() < end_time:
                self.logger.info("Scanning for arbitrage opportunities...")
                
                # Detect opportunities
                opportunities, profit_rate = await run_detection(
                    self.exchange_name,
                    max_cycle=6
                )
                
                if opportunities and profit_rate > 1.005:  # Minimum 0.5% profit
                    opportunities_found += 1
                    self.logger.info(f"Opportunity #{opportunities_found}: {(profit_rate-1)*100:.3f}% profit potential")
                    
                    # Execute arbitrage
                    result = await self.executor.execute_arbitrage_cycle(opportunities)
                    
                    if result.success:
                        trades_executed += 1
                        self.logger.info(f"✅ Arbitrage executed! Profit: ${result.profit_loss:.2f} ({result.profit_percentage:.3f}%)")
                    else:
                        self.logger.error(f"❌ Arbitrage failed: {result.error_message}")
                    
                    self.session_results.append(result)
                    
                    # Print current portfolio status
                    await self._print_portfolio_status()
                
                else:
                    self.logger.info("No profitable opportunities found")
                
                # Wait before next scan
                await asyncio.sleep(scan_interval)
        
        except KeyboardInterrupt:
            self.logger.info("Session interrupted by user")
        
        finally:
            await self.executor.close_real_exchange()
            
            # Print session summary
            await self._print_session_summary(start_time, opportunities_found, trades_executed)
    
    async def _print_portfolio_status(self):
        """Print current portfolio status"""
        balances = await self.paper_exchange.fetch_balance()
        
        print("\n" + "="*50)
        print("PORTFOLIO STATUS")
        print("="*50)
        
        for currency, balance in balances.items():
            if currency not in ['info', 'free', 'used', 'total'] and balance['total'] > 0:
                print(f"{currency}: {balance['free']:.6f} (free), {balance['total']:.6f} (total)")
        
        trade_summary = self.paper_exchange.get_trade_summary()
        print(f"\nTrades executed: {trade_summary['total_trades']}")
        print(f"Total fees paid: ${trade_summary['total_fees']:.2f}")
        print(f"Average slippage: {trade_summary['avg_slippage']*100:.3f}%")
        print("="*50 + "\n")
    
    async def _print_session_summary(self, start_time: datetime, opportunities_found: int, trades_executed: int):
        """Print final session summary"""
        session_duration = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Duration: {session_duration:.1f} minutes")
        print(f"Opportunities found: {opportunities_found}")
        print(f"Trades executed: {trades_executed}")
        
        if self.session_results:
            successful_trades = [r for r in self.session_results if r.success]
            total_profit = sum(r.profit_loss for r in successful_trades)
            
            print(f"Successful trades: {len(successful_trades)}")
            print(f"Total profit/loss: ${total_profit:.2f}")
            print(f"Success rate: {len(successful_trades)/len(self.session_results)*100:.1f}%")
            
            if successful_trades:
                avg_profit = total_profit / len(successful_trades)
                print(f"Average profit per trade: ${avg_profit:.2f}")
        
        # Final balance
        final_balances = await self.paper_exchange.fetch_balance()
        final_usdt = final_balances['USDT']['free']
        initial_usdt = 10000.0
        total_return = ((final_usdt - initial_usdt) / initial_usdt) * 100
        
        print(f"\nFinal balance: ${final_usdt:.2f}")
        print(f"Total return: {total_return:.2f}%")
        print("="*60)
        
        # Save results to file
        self._save_session_results()
    
    def _save_session_results(self):
        """Save session results to JSON file"""
        filename = f"paper_trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = {
            'session_info': {
                'exchange': self.exchange_name,
                'timestamp': datetime.now().isoformat(),
                'initial_balance': 10000.0,
                'final_balance': self.paper_exchange.balances['USDT'].free
            },
            'trades': []
        }
        
        for result in self.session_results:
            results_data['trades'].append({
                'success': result.success,
                'profit_loss': result.profit_loss,
                'profit_percentage': result.profit_percentage,
                'trades_executed': result.trades_executed,
                'total_fees': result.total_fees,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run paper trading bot
    bot = PaperTradingBot(exchange_name="binanceus")
    
    print("🚀 Starting Paper Trading Simulation")
    print("📊 Initial balance: $10,000 USDT")
    print("⚙️  Trading fee: 0.1%")
    print("📈 Max slippage: 0.2%")
    print("Press Ctrl+C to stop at any time")
    print("\n" + "="*50)
    
    # Run for 60 minutes, scanning every 30 seconds
    await bot.run_session(duration_minutes=60, scan_interval=30)

if __name__ == "__main__":
    asyncio.run(main())