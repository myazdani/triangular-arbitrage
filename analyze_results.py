#!/usr/bin/env python3
"""
Paper Trading Results Analyzer

Usage:
    python analyze_results.py <results_file.json>

Example:
    python analyze_results.py paper_trading_results_20241201_143022.json
"""

import sys
import os
from triangular_arbitrage.analysis import TradingAnalyzer

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file.json>")
        print("\nExample: python analyze_results.py paper_trading_results_20241201_143022.json")
        return
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: File '{results_file}' not found")
        return
    
    try:
        # Create analyzer
        analyzer = TradingAnalyzer(results_file)
        
        # Generate performance report
        analyzer.generate_performance_report()
        
        # Show detailed trade information
        analyzer.get_trade_details()
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return

if __name__ == "__main__":
    main()