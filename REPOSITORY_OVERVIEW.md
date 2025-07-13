# Repository Overview: OctoBot Triangular Arbitrage

## Project Summary

This repository contains a Python-based cryptocurrency arbitrage detection system that identifies profitable trading opportunities across multiple cryptocurrency exchanges. The project uses graph algorithms to find cycles in currency trading pairs that could yield profit through a series of trades.

## What is Triangular Arbitrage?

Triangular arbitrage (or multi-asset arbitrage) is a trading strategy where you:
1. Start with one cryptocurrency (e.g., BTC)
2. Trade through a series of different cryptocurrencies 
3. End up back at the original cryptocurrency
4. Potentially profit from price differences between exchanges

For example: BTC → ETH → USDT → BTC (if the exchange rates allow for profit)

## Project Structure

### Core Components

```
triangular_arbitrage/
├── __init__.py          # Project metadata (name, version)
├── detector.py          # Core arbitrage detection logic
main.py                  # Entry point and output formatting
requirements.txt         # Python dependencies
setup.py                 # Package configuration
```

### Key Dependencies

- **ccxt**: Cryptocurrency exchange library for fetching market data
- **networkx**: Graph algorithms library for cycle detection
- **OctoBot-Commons**: Common utilities from the OctoBot project

## How It Works

### 1. Data Collection (`detector.py`)
- Connects to cryptocurrency exchanges using the CCXT library
- Fetches current ticker prices for all available trading pairs
- Filters out delisted or stale symbols
- Supports whitelisting/blacklisting of specific symbols

### 2. Graph Construction
- Creates a directed graph where:
  - **Nodes**: Individual cryptocurrencies (BTC, ETH, USDT, etc.)
  - **Edges**: Trading pairs with their exchange rates
- For each trading pair A/B, creates two edges:
  - A → B (direct rate)
  - B → A (inverse rate: 1/original_rate)

### 3. Cycle Detection
- Uses NetworkX's `simple_cycles()` algorithm to find all possible trading cycles
- Supports cycles of length 3 (triangular) up to 10 assets
- For each cycle, calculates potential profit by multiplying all exchange rates

### 4. Profit Calculation
- Multiplies exchange rates along each cycle path
- A profitable opportunity exists when the product > 1.0
- Selects the cycle with the highest profit percentage

### 5. Output Formatting (`main.py`)
- Displays the best opportunity found
- Shows step-by-step trading instructions
- Calculates and displays profit percentage

## Example Output

```
-------------------------------------------
New 2.33873% binanceus opportunity:
1. buy DOGE with BTC at 552486.18785
2. sell DOGE for USDT at 0.12232
3. buy ETH with USDT at 0.00038
4. buy ADA with ETH at 7570.02271
5. sell ADA for USDC at 0.35000
6. buy SOL with USDC at 0.00662
7. sell SOL for BTC at 0.00226
-------------------------------------------
```

## Key Features

### Configuration Options
- **Exchange Selection**: Configurable via `exchange_name` in `main.py`
- **Symbol Filtering**: Support for ignored and whitelisted symbols
- **Cycle Length**: Configurable maximum cycle length (default: 10)

### Technical Features
- **Asynchronous Processing**: Uses asyncio for efficient API calls
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Dockerized**: Includes Dockerfile for containerized deployment
- **Performance Monitoring**: Optional benchmarking mode

### Data Quality Controls
- **Delisted Symbol Detection**: Filters out symbols older than 1 day
- **Null Price Handling**: Ignores pairs with missing price data
- **Exchange Time Synchronization**: Uses exchange-provided timestamps

## Limitations

⚠️ **Important**: The results do not account for trading fees, which can significantly impact actual profitability.

## Usage

### Basic Usage
```bash
python3 main.py
```

### With Custom Configuration
```python
# In main.py, modify these variables:
exchange_name = "binanceus"  # Change to desired exchange
ignored_symbols = ["SYMBOL1", "SYMBOL2"]  # Optional
whitelisted_symbols = ["BTC", "ETH"]  # Optional
```

### Supported Exchanges
Any exchange supported by the CCXT library (100+ exchanges including Binance, Coinbase, Kraken, etc.)

## Architecture Benefits

1. **Modular Design**: Clear separation between data fetching, graph analysis, and output
2. **Scalable**: Can handle hundreds of trading pairs efficiently
3. **Extensible**: Easy to add new exchanges or modify detection algorithms
4. **Testable**: Includes unit tests for core functionality

## Real-world Considerations

While this tool can identify theoretical arbitrage opportunities, successful implementation requires:
- Ultra-low latency execution
- Sufficient account balances on multiple exchanges
- Consideration of trading fees and slippage
- Risk management for volatile markets

This project serves as an excellent foundation for understanding cryptocurrency arbitrage and market inefficiencies.