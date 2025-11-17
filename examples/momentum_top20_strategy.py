"""
Momentum Top 20 Strategy Example

Strategy: Invest in the 20 stocks with highest 5-day returns, equal weighted.
Universe: 100 stocks from yfinance
Period: 5 years
Rebalance: Daily

This demonstrates:
- Multi-asset backtesting with 100 tickers
- Signal-based strategy (momentum rankings)
- Portfolio rebalancing with helper methods
- Real market data from yfinance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from typing import Dict, List

from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data import PolarsDataFeed
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.commission import PercentageCommission
from ml4t.backtest.core.event import MarketEvent


class MomentumTop20Strategy(Strategy):
    """
    Select top 20 stocks by 5-day momentum, equal weight portfolio.
    Rebalance daily to maintain equal weights.
    """

    def __init__(
        self,
        top_n: int = 20,
        lookback_days: int = 5,
        rebalance_threshold: float = 0.05,  # Rebalance if weight drifts >5%
    ):
        super().__init__()
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.rebalance_threshold = rebalance_threshold

        # Track momentum signals for all assets
        self.momentum_signals: Dict[str, float] = {}
        self.last_rebalance_date = None

    def on_event(self, event):
        """Required by base Strategy class - delegates to on_market_event."""
        if isinstance(event, MarketEvent):
            self.on_market_event(event)

    def on_market_event(self, event: MarketEvent, context: dict = None):
        """
        Calculate momentum, select top N, rebalance portfolio.
        """
        # Extract momentum signal from event (pre-calculated)
        asset_id = event.asset_id

        # Get 5-day momentum from signals dict
        if 'momentum_5d' in event.signals:
            self.momentum_signals[asset_id] = event.signals['momentum_5d']

        # Only rebalance once per day (using first ticker event as trigger)
        current_date = event.timestamp.date()
        if self.last_rebalance_date == current_date:
            return

        # Wait until we have momentum for all assets
        if len(self.momentum_signals) < 50:  # Wait for some data
            return

        # Rank assets by momentum
        ranked_assets = sorted(
            self.momentum_signals.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top N
        top_assets = [asset for asset, _ in ranked_assets[:self.top_n]]

        # Build target weights (equal weight for top 20)
        # Also track rank and momentum value for each asset (for reporting)
        target_weights = {}
        asset_metadata = {}

        for rank, (asset, momentum) in enumerate(ranked_assets[:self.top_n], 1):
            target_weights[asset] = 1.0 / self.top_n
            asset_metadata[asset] = {
                "strategy": "momentum_top20",
                "rank": rank,
                "momentum_value": momentum,
                "rebalance_date": str(current_date),
                "reason": f"top_{rank}_momentum_{momentum:.4f}",
            }

        # Rebalance portfolio using helper method
        # The helper will create orders with metadata for tracking
        self.rebalance_to_weights(
            target_weights=target_weights,
            current_prices={asset: event.close for asset in top_assets},
            tolerance=0.001,  # 0.1% tolerance to avoid floating point precision issues
            metadata_per_asset=asset_metadata,  # Pass metadata for each asset
        )

        self.last_rebalance_date = current_date


def download_sp500_data(
    num_tickers: int = 100,
    start_date: str = "2019-01-01",
    end_date: str = "2024-01-01",
    output_dir: Path = Path("data/sp500_momentum")
) -> List[str]:
    """
    Download historical data for top S&P 500 stocks.

    Returns:
        List of tickers successfully downloaded
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Top 100 S&P 500 stocks by market cap (as of 2024)
    sp500_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM',
        'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
        'KO', 'AVGO', 'COST', 'LLY', 'MCD', 'TMO', 'WMT', 'CSCO', 'ACN', 'ABT',
        'DHR', 'VZ', 'ADBE', 'NKE', 'TXN', 'NEE', 'PM', 'COP', 'RTX', 'UPS',
        'QCOM', 'INTU', 'LOW', 'HON', 'AMGN', 'SPGI', 'BA', 'SBUX', 'AXP', 'CAT',
        'DE', 'GE', 'AMD', 'ELV', 'PLD', 'MDLZ', 'BLK', 'GILD', 'ADI', 'ISRG',
        'TJX', 'CI', 'MMC', 'REGN', 'SYK', 'BKNG', 'MO', 'CB', 'ZTS', 'C',
        'PGR', 'SO', 'DUK', 'BSX', 'BMY', 'ETN', 'EOG', 'CME', 'ITW', 'SLB',
        'APD', 'ICE', 'PNC', 'CL', 'NOC', 'GD', 'MCO', 'EMR', 'WM', 'USB',
        'AON', 'TGT', 'F', 'NSC', 'SHW', 'GM', 'HCA', 'PSA', 'MSI', 'COF'
    ][:num_tickers]

    print(f"Downloading {num_tickers} tickers from {start_date} to {end_date}...")

    successful_tickers = []
    failed_tickers = []

    for ticker in sp500_tickers:
        try:
            print(f"  Downloading {ticker}...", end=' ')

            # Download data
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Use adjusted prices
            )

            if df.empty or len(df) < 100:  # Need minimum data
                print(f"SKIP (insufficient data)")
                failed_tickers.append(ticker)
                continue

            # Flatten multi-index columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Reset index to make timestamp a column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'timestamp'})

            # Calculate 5-day momentum
            df['momentum_5d'] = df['close'].pct_change(5)

            # Drop NaN values (first 5 days have no momentum)
            df = df.dropna()

            # Add asset_id column
            df['asset_id'] = ticker

            # Save to parquet
            output_file = output_dir / f"{ticker}.parquet"
            df.to_parquet(output_file, index=False)

            successful_tickers.append(ticker)
            print(f"OK ({len(df)} bars)")

        except Exception as e:
            print(f"FAILED ({e})")
            failed_tickers.append(ticker)

    print(f"\nDownload complete:")
    print(f"  Success: {len(successful_tickers)}")
    print(f"  Failed: {len(failed_tickers)}")

    return successful_tickers


def run_momentum_backtest(
    tickers: List[str],
    data_dir: Path = Path("data/sp500_momentum"),
    initial_cash: float = 1_000_000.0,
    commission_rate: float = 0.001  # 0.1% per trade
):
    """
    Run the momentum strategy backtest.
    """
    print(f"\n{'='*80}")
    print(f"MOMENTUM TOP 20 BACKTEST")
    print(f"{'='*80}\n")

    print(f"Universe: {len(tickers)} stocks")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Commission: {commission_rate*100:.2f}%")
    print(f"Strategy: Top 20 by 5-day momentum, equal weighted\n")

    # Create data feed from parquet files
    print("Loading data...")
    data_feeds = []
    for ticker in tickers:
        parquet_file = data_dir / f"{ticker}.parquet"
        if parquet_file.exists():
            feed = PolarsDataFeed(
                price_path=parquet_file,
                asset_id=ticker,
                signal_columns=['momentum_5d'],  # Auto-extract to event.signals
                validate_signal_timing=False,
            )
            data_feeds.append(feed)

    print(f"  Loaded {len(data_feeds)} data feeds\n")

    # Create broker with commission
    broker = SimulationBroker(
        initial_cash=initial_cash,
        commission_model=PercentageCommission(rate=commission_rate)
    )

    # Create strategy
    strategy = MomentumTop20Strategy(
        top_n=20,
        lookback_days=5,
        rebalance_threshold=0.05
    )

    # Create and run backtest engine
    print("Running backtest...")
    engine = BacktestEngine(
        broker=broker,
        strategy=strategy,
        data_feed=data_feeds[0]  # Pass first feed
    )

    # Add remaining feeds to clock for multi-asset support
    for feed in data_feeds[1:]:
        engine.clock.add_data_feed(feed)

    results = engine.run()

    # Display results
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*80}\n")

    portfolio = broker.portfolio
    final_equity = portfolio.equity

    print(f"Initial Value: ${initial_cash:,.2f}")
    print(f"Final Value: ${final_equity:,.2f}")
    print(f"Total Return: {(final_equity / initial_cash - 1) * 100:.2f}%")
    print(f"Total P&L: ${final_equity - initial_cash:,.2f}")
    print(f"Total Commission: ${portfolio.total_commission:,.2f}")

    print(f"\nPortfolio Analytics:")
    if portfolio._analyzer:
        sharpe = portfolio._analyzer.calculate_sharpe_ratio()
        print(f"  Sharpe Ratio: {sharpe:.2f}" if sharpe else "  Sharpe Ratio: N/A")
        print(f"  Max Drawdown: {portfolio._analyzer.max_drawdown * 100:.2f}%")
    else:
        print(f"  Analytics: Disabled")

    print(f"\nFinal Positions: {len(portfolio.positions)}")
    if len(portfolio.positions) > 0:
        print("\nTop 5 positions:")
        sorted_positions = sorted(
            portfolio.positions.items(),
            key=lambda x: abs(x[1].market_value),
            reverse=True
        )[:5]
        for asset_id, position in sorted_positions:
            print(f"  {asset_id}: {position.quantity:.2f} shares @ ${position.last_price:.2f} = ${position.market_value:,.2f}")

    print(f"\n{'='*80}\n")

    return engine  # Return engine so caller can access results.get_results()


def main():
    """
    Main execution: download data and run backtest.
    """
    # Configuration
    NUM_TICKERS = 100
    START_DATE = "2019-01-01"
    END_DATE = "2024-01-01"
    DATA_DIR = Path("data/sp500_momentum")

    # Step 1: Download data (skip if already exists)
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*.parquet"))) < NUM_TICKERS:
        print("Downloading market data...")
        tickers = download_sp500_data(
            num_tickers=NUM_TICKERS,
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=DATA_DIR
        )
    else:
        print(f"Using existing data in {DATA_DIR}")
        tickers = [f.stem for f in DATA_DIR.glob("*.parquet")]
        print(f"Found {len(tickers)} tickers\n")

    # Step 2: Run backtest
    engine = run_momentum_backtest(
        tickers=tickers,
        data_dir=DATA_DIR,
        initial_cash=1_000_000.0,
        commission_rate=0.001
    )

    print("Backtest complete!")

    # Step 3: Export results (NEW - demonstrates robust reporting)
    print("\nExporting results...")
    results_dir = Path("results/momentum_top20")

    # Get results exporter
    results = engine.get_results()

    # Export all data
    exported_files = results.export_all(results_dir)
    print(f"\nExported results to:")
    for file_type, path in exported_files.items():
        print(f"  {file_type}: {path}")

    # Show summary
    summary = results.summary()
    print(f"\nResults Summary:")
    print(f"  Total trades: {summary['num_trades']}")
    print(f"  Open positions: {summary['num_open_positions']}")
    if 'final_equity' in summary:
        print(f"  Final equity: ${summary['final_equity']:,.2f}")
        print(f"  Total return: {summary['total_return']*100:.2f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
