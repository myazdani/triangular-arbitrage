import json
from datetime import datetime
from typing import List, Dict

class TradingAnalyzer:
    def __init__(self, results_file: str):
        with open(results_file, 'r') as f:
            self.data = json.load(f)
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        trades = self.data['trades']
        
        if not trades:
            print("No trades to analyze")
            return
        
        # Basic statistics
        total_trades = len(trades)
        successful_trades = [t for t in trades if t['success']]
        failed_trades = [t for t in trades if not t['success']]
        
        success_rate = len(successful_trades) / total_trades * 100
        
        # Profit/Loss analysis
        total_profit = sum(t['profit_loss'] for t in successful_trades)
        total_fees = sum(t['total_fees'] for t in trades)
        
        avg_profit_per_trade = total_profit / len(successful_trades) if successful_trades else 0
        
        # Print report
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS REPORT")
        print("="*60)
        print(f"Total Trades: {total_trades}")
        print(f"Successful Trades: {len(successful_trades)}")
        print(f"Failed Trades: {len(failed_trades)}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Profit/Loss: ${total_profit:.2f}")
        print(f"Total Fees Paid: ${total_fees:.2f}")
        print(f"Net Profit/Loss: ${total_profit - total_fees:.2f}")
        print(f"Average Profit per Successful Trade: ${avg_profit_per_trade:.2f}")
        
        # Performance metrics
        if successful_trades:
            profits = [t['profit_loss'] for t in successful_trades]
            print(f"Best Trade: ${max(profits):.2f}")
            print(f"Worst Trade: ${min(profits):.2f}")
            
            # Calculate simple standard deviation
            mean_profit = sum(profits) / len(profits)
            variance = sum((p - mean_profit) ** 2 for p in profits) / len(profits)
            std_dev = variance ** 0.5
            print(f"Profit Standard Deviation: ${std_dev:.2f}")
        
        # Session info
        session_info = self.data['session_info']
        initial_balance = session_info['initial_balance']
        final_balance = session_info['final_balance']
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        print(f"\nSession Info:")
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Exchange: {session_info['exchange']}")
        print(f"Session Time: {session_info['timestamp']}")
        
        print("="*60)
        
    def get_trade_details(self):
        """Print detailed trade information"""
        trades = self.data['trades']
        
        print("\n" + "="*60)
        print("DETAILED TRADE ANALYSIS")
        print("="*60)
        
        for i, trade in enumerate(trades, 1):
            status = "✅ SUCCESS" if trade['success'] else "❌ FAILED"
            print(f"\nTrade #{i}: {status}")
            print(f"  Profit/Loss: ${trade['profit_loss']:.2f}")
            print(f"  Profit %: {trade['profit_percentage']:.3f}%")
            print(f"  Trades Executed: {trade['trades_executed']}")
            print(f"  Total Fees: ${trade['total_fees']:.2f}")
            print(f"  Execution Time: {trade['execution_time']:.3f}s")
            if trade['error_message']:
                print(f"  Error: {trade['error_message']}")
        
        print("="*60)