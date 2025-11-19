"""Top 25 ML Strategy: Using BacktestEngine with MultiSymbolDataFeed

This example demonstrates the CORRECT way to use ml4t.backtest for multi-asset strategies:
- Uses BacktestEngine (not manual loops)
- Uses MultiSymbolDataFeed (not manual event creation)
- Uses Strategy base class (not ad-hoc logic)

This is the HONEST implementation that can be compared to VectorBT/Backtrader.
"""

import time
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

# ml4t.backtest imports
from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import AssetId, OrderType, OrderSide
from ml4t.backtest.data.multi_symbol_feed import MultiSymbolDataFeed
from ml4t.backtest.engine import BacktestEngine
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.order import Order
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.strategy.base import Strategy
from ml4t.backtest.risk.manager import RiskManager
from ml4t.backtest.risk.rules import (
    VolatilityScaledStopLoss,
    DynamicTrailingStop,
    TimeBasedExit,
)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
INITIAL_CAPITAL = 1_000_000.0
N_POSITIONS = 25
MAX_VIX = 30.0

print("=" * 80)
print("TOP 25 ML STRATEGY - USING BACKTEST ENGINE")
print("=" * 80)
print()

# Load data
print("[1/8] Loading data...")
stock_data = pl.read_parquet(DATA_DIR / "stock_data.parquet")
vix_data = pl.read_parquet(DATA_DIR / "vix_data.parquet")

print(f"  Stock data: {stock_data.shape[0]:,} rows, {stock_data['asset_id'].n_unique()} stocks")
print(f"  VIX data: {vix_data.shape[0]} rows")
print(f"  Date range: {stock_data['timestamp'].min()} to {stock_data['timestamp'].max()}")
print()

# Prepare signals DataFrame (ml_score, atr)
print("[2/8] Preparing signals...")
signals_df = stock_data.select(['timestamp', 'asset_id', 'ml_score', 'atr'])
print(f"  Signal columns: {signals_df.columns[2:]}")
print()

# Prepare context DataFrame (vix)
print("[3/8] Preparing context...")
context_df = vix_data.select(['timestamp', 'vix'])
print(f"  Context columns: {context_df.columns[1:]}")
print()

# Create DataFeed
print("[4/8] Creating MultiSymbolDataFeed...")
feed = MultiSymbolDataFeed(
    price_df=stock_data.select(['timestamp', 'asset_id', 'open', 'high', 'low', 'close', 'volume']),
    signals_df=signals_df,
    context_df=context_df,
)
print(f"  Feed ready: {len(stock_data):,} events")
print()

# Create Strategy
print("[5/8] Creating strategy...")


class Top25MLStrategy(Strategy):
    """Top-N momentum strategy using ML scores with batch processing."""

    # Enable batch mode for simultaneous portfolio rebalancing
    execution_mode = "batch"

    def __init__(self, n_positions: int, max_vix: float):
        super().__init__(name="Top25ML")
        self.n_positions = n_positions
        self.max_vix = max_vix
        self.rebalances = 0
        self.vix_filtered = 0

    def on_event(self, event):
        """Stub implementation for abstract method (not used in batch mode)."""
        pass

    def on_timestamp_batch(self, timestamp: datetime, events: list[MarketEvent], context: dict = None):
        """Process all assets simultaneously for this timestamp.

        This is the batch API that enables correct portfolio rebalancing:
        - Receives ALL assets at once (not one-by-one)
        - Fills have already been processed (correct sequencing)
        - Can use order_target_percent() for proper rebalancing
        """
        # Convert list of events to dict for easier lookup
        market_map = {event.asset_id: event for event in events}
        # VIX filter
        vix = context.get('vix', 0.0) if context else 0.0
        if vix > self.max_vix:
            self.vix_filtered += 1
            return

        # Extract ML scores from all events
        asset_scores = []
        for asset_id, event in market_map.items():
            ml_score = event.signals.get('ml_score', 0.0)
            atr = event.signals.get('atr', None)
            if atr is not None and not np.isnan(atr):
                asset_scores.append((asset_id, ml_score, event.close))

        if not asset_scores:
            return

        # Rank by ML score and select top N
        asset_scores.sort(key=lambda x: x[1], reverse=True)
        top_assets = asset_scores[:self.n_positions]

        # Calculate target portfolio (equal weight across top N)
        target_pct = 1.0 / self.n_positions
        new_targets = {asset_id: target_pct for asset_id, _, _ in top_assets}

        # Get current positions
        current_positions = set(self.broker.get_positions().keys())

        # Exit positions no longer in top N
        for asset_id in current_positions:
            if asset_id not in new_targets:
                # Set target to 0% (closes position)
                self.order_target_percent(asset_id, 0.0)

        # Enter/rebalance top N positions
        for asset_id, target_pct in new_targets.items():
            # Use order_target_percent() - the KEY FIX!
            # This calculates delta from current position automatically
            self.order_target_percent(asset_id, target_pct)

        self.rebalances += 1


strategy = Top25MLStrategy(
    n_positions=N_POSITIONS,
    max_vix=MAX_VIX,
)
print(f"  Strategy: Top {N_POSITIONS} by ML score")
print(f"  VIX filter: Skip rebalancing if VIX > {MAX_VIX}")
print()

# Create RiskManager (exact params from working example)
print("[6/8] Creating risk manager...")
risk_manager = RiskManager()

vol_stop = VolatilityScaledStopLoss(atr_multiplier=2.0, volatility_key='atr', priority=100)
risk_manager.add_rule(vol_stop)

trailing_stop = DynamicTrailingStop(
    initial_trail_pct=0.05,
    minimum_trail_pct=0.005,
    tighten_rate=0.001,
    priority=100,
)
risk_manager.add_rule(trailing_stop)

time_exit = TimeBasedExit(max_bars=60)
risk_manager.add_rule(time_exit)
print(f"  Risk rules: 3 (VolatilityStop, TrailingStop, TimeExit)")
print()

# Create BacktestEngine
print("[7/8] Creating BacktestEngine...")
engine = BacktestEngine(
    data_feed=feed,
    strategy=strategy,
    initial_capital=INITIAL_CAPITAL,
    risk_manager=risk_manager,
)
print(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")
print()

# Run backtest
print("[8/8] Running backtest...")
start_time = time.time()
engine.run()
end_time = time.time()

elapsed = end_time - start_time
events_per_sec = len(stock_data) / elapsed if elapsed > 0 else 0

print(f"  ✓ Backtest complete in {elapsed:.2f}s")
print(f"  Events processed: {len(stock_data):,}")
print(f"  Throughput: {events_per_sec:,.0f} events/second")
print()

# Results
print("=" * 80)
print("RESULTS")
print("=" * 80)
portfolio = engine.broker._internal_portfolio  # Use broker's portfolio (source of truth)
print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final value: ${portfolio.equity:,.2f}")
print(f"Total return: {(portfolio.equity / INITIAL_CAPITAL - 1) * 100:.2f}%")
print(f"P&L: ${portfolio.equity - INITIAL_CAPITAL:,.2f}")
print()
print(f"Rebalances: {strategy.rebalances}")
print(f"VIX-filtered days: {strategy.vix_filtered}")
print(f"Final positions: {len(portfolio.positions)}")
print()
print("=" * 80)
print("COMPLETE!")
print("=" * 80)
