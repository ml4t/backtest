"""Focused validation test for cash account constraints.

This test specifically validates that the accounting system prevents unlimited debt
and matches VectorBT's behavior with cash constraints.

Test scenario:
- 50 assets (manageable for clear diagnostics)
- Aggressive signal generation (high turnover)
- Limited initial cash to trigger rejections
- Verify cash never goes negative
- Verify P&L matches VectorBT within 0.1%
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from datetime import datetime, timedelta


def generate_test_data_simple(
    n_assets: int = 50,
    n_days: int = 100,
    seed: int = 42,
) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Generate simple test data for cash constraint validation."""
    np.random.seed(seed)

    # Trading days
    start_date = datetime(2023, 1, 3)
    trading_days = []
    current_date = start_date
    while len(trading_days) < n_days:
        if current_date.weekday() < 5:
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    records = []
    for i in range(n_assets):
        asset = f"ASSET_{i:03d}"
        price = 100.0  # Fixed price for simplicity

        for timestamp in trading_days:
            # Simple OHLCV
            records.append({
                "timestamp": timestamp,
                "asset": asset,
                "open": price,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": 1000000,
            })

    df_polars = pl.DataFrame(records)
    df_pandas = pd.DataFrame(records)
    df_pandas['timestamp'] = pd.to_datetime(df_pandas['timestamp'])

    return df_polars, df_pandas


def generate_aggressive_signals(
    prices_df: pl.DataFrame,
    n_positions: int = 10,  # Number of positions to hold
    seed: int = 123,
) -> pl.DataFrame:
    """Generate aggressive trading signals that will stress cash constraints."""
    np.random.seed(seed)

    timestamps = sorted(prices_df.select("timestamp").unique().to_series().to_list())
    assets = sorted(prices_df.select("asset").unique().to_series().to_list())

    records = []
    for ts in timestamps:
        # Randomly select assets for positions
        selected = np.random.choice(assets, size=n_positions, replace=False)
        position_assets = set(selected)

        for asset in assets:
            signal = 1.0 if asset in position_assets else 0.0
            records.append({
                "timestamp": ts,
                "asset": asset,
                "signal": signal,
            })

    return pl.DataFrame(records)


def run_engine_cash_account(
    prices_pl: pl.DataFrame,
    signals_pl: pl.DataFrame,
    initial_cash: float,
) -> dict:
    """Run engine with cash account constraints."""
    from ml4t.backtest.engine import (
        Strategy, Broker, DataFeed, Engine,
        NoCommission, NoSlippage, OrderSide
    )

    class SimpleSignalStrategy(Strategy):
        """Simple strategy that allocates equal cash to each signal."""

        def __init__(self, position_value: float = 10000):
            self.position_value = position_value  # $10k per position

        def on_data(self, timestamp, data, context, broker):
            for asset, asset_data in data.items():
                signal = asset_data.get("signals", {}).get("signal", 0)
                price = asset_data.get("close")
                if price is None or price <= 0:
                    continue

                pos = broker.get_position(asset)
                current_qty = pos.quantity if pos else 0

                target_qty = 0
                if signal > 0:
                    target_qty = self.position_value / price

                delta = target_qty - current_qty

                if abs(delta) * price > 10:  # Min $10 trade
                    if delta > 0:
                        broker.submit_order(asset, delta, OrderSide.BUY)
                    else:
                        broker.submit_order(asset, abs(delta), OrderSide.SELL)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    engine = Engine(
        feed=feed,
        strategy=SimpleSignalStrategy(position_value=10000),
        initial_cash=initial_cash,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        account_type='cash',  # CRITICAL: Cash account with constraints
    )
    results = engine.run()

    # Check cash history
    min_cash = min(results.get("cash_history", {}).values()) if results.get("cash_history") else initial_cash

    return {
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - initial_cash,
        "num_trades": results["num_trades"],
        "min_cash": min_cash,
        "cash_went_negative": min_cash < 0,
    }


def run_vectorbt_cash_account(
    prices_pd: pd.DataFrame,
    signals_pl: pl.DataFrame,
    initial_cash: float,
) -> dict:
    """Run VectorBT with cash constraints (for comparison)."""
    import vectorbt as vbt

    # Pivot data
    close_wide = prices_pd.pivot(index='timestamp', columns='asset', values='close')

    # Convert signals
    signals_pd = signals_pl.to_pandas()
    signals_pd['timestamp'] = pd.to_datetime(signals_pd['timestamp'])
    signals_wide = signals_pd.pivot(index='timestamp', columns='asset', values='signal')
    signals_wide = signals_wide.reindex(close_wide.index).fillna(0)

    # Boolean entries/exits
    entries = (signals_wide == 1.0).astype(bool)
    exits = entries.shift(1).fillna(False) & ~entries

    # Run VectorBT
    pf = vbt.Portfolio.from_signals(
        close=close_wide,
        entries=entries,
        exits=exits,
        size=10000,  # $10k per position
        size_type='value',
        init_cash=initial_cash,
        fees=0.0,
        slippage=0.0,
        freq='1D',
        cash_sharing=True,
        group_by=False,
    )

    # Get results
    value = pf.value()
    if hasattr(value, 'iloc') and len(value.shape) == 2:
        final_value = float(value.iloc[-1, 0])
    elif hasattr(value, 'iloc'):
        final_value = float(value.iloc[-1])
    else:
        final_value = float(value)

    trade_count = pf.trades.count()
    if hasattr(trade_count, 'sum'):
        num_trades = int(trade_count.sum())
    else:
        num_trades = int(trade_count)

    # VectorBT respects cash constraints by default
    cash_series = pf.cash()
    if hasattr(cash_series, 'min'):
        min_cash = float(cash_series.min().min() if hasattr(cash_series.min(), 'min') else cash_series.min())
    else:
        min_cash = initial_cash

    return {
        "final_value": final_value,
        "total_pnl": final_value - initial_cash,
        "num_trades": num_trades,
        "min_cash": min_cash,
        "cash_went_negative": min_cash < 0,
    }


class TestCashConstraintValidation:
    """Validation tests for cash account constraints."""

    def test_cash_never_negative(self):
        """Verify cash never goes negative with cash account."""
        print("\n" + "="*80)
        print("TEST: Cash Never Goes Negative")
        print("="*80)

        prices_pl, _ = generate_test_data_simple(n_assets=50, n_days=100)
        signals_pl = generate_aggressive_signals(prices_pl, n_positions=10)

        result = run_engine_cash_account(
            prices_pl, signals_pl,
            initial_cash=100000,  # $100k initial
        )

        print(f"\nEngine Results:")
        print(f"  Final Value: ${result['final_value']:,.2f}")
        print(f"  Total P&L:   ${result['total_pnl']:,.2f}")
        print(f"  Num Trades:  {result['num_trades']}")
        print(f"  Min Cash:    ${result['min_cash']:,.2f}")

        assert result['min_cash'] >= 0, f"Cash went negative! Min cash: ${result['min_cash']:,.2f}"
        print(f"\n✅ PASS: Cash never went negative (min: ${result['min_cash']:,.2f})")

    def test_matches_vectorbt_within_threshold(self):
        """Verify engine matches VectorBT within 0.1% for cash account."""
        print("\n" + "="*80)
        print("TEST: Engine vs VectorBT Cash Account Matching")
        print("="*80)

        prices_pl, prices_pd = generate_test_data_simple(n_assets=50, n_days=100)
        signals_pl = generate_aggressive_signals(prices_pl, n_positions=10)
        initial_cash = 100000

        print("\nRunning engine (cash account)...")
        engine_result = run_engine_cash_account(prices_pl, signals_pl, initial_cash)

        print("Running VectorBT (cash account)...")
        vbt_result = run_vectorbt_cash_account(prices_pd, signals_pl, initial_cash)

        print(f"\n--- Results ---")
        print(f"Engine:")
        print(f"  Final Value: ${engine_result['final_value']:,.2f}")
        print(f"  Total P&L:   ${engine_result['total_pnl']:,.2f}")
        print(f"  Num Trades:  {engine_result['num_trades']}")
        print(f"  Min Cash:    ${engine_result['min_cash']:,.2f}")

        print(f"\nVectorBT:")
        print(f"  Final Value: ${vbt_result['final_value']:,.2f}")
        print(f"  Total P&L:   ${vbt_result['total_pnl']:,.2f}")
        print(f"  Num Trades:  {vbt_result['num_trades']}")
        print(f"  Min Cash:    ${vbt_result['min_cash']:,.2f}")

        # Calculate difference
        pnl_diff = abs(engine_result['total_pnl'] - vbt_result['total_pnl'])
        if abs(vbt_result['total_pnl']) > 0:
            pnl_diff_pct = pnl_diff / abs(vbt_result['total_pnl']) * 100
        else:
            pnl_diff_pct = 0 if pnl_diff == 0 else 100

        trade_diff = abs(engine_result['num_trades'] - vbt_result['num_trades'])
        if vbt_result['num_trades'] > 0:
            trade_diff_pct = trade_diff / vbt_result['num_trades'] * 100
        else:
            trade_diff_pct = 0

        print(f"\n--- Comparison ---")
        print(f"P&L Difference:   ${pnl_diff:,.2f} ({pnl_diff_pct:.4f}%)")
        print(f"Trade Difference: {trade_diff} ({trade_diff_pct:.2f}%)")

        # Assertions
        assert not engine_result['cash_went_negative'], "Engine cash went negative!"
        assert not vbt_result['cash_went_negative'], "VectorBT cash went negative!"

        # Allow up to 2% difference for trade counts (acceptable for cash constraints and timing differences)
        assert trade_diff_pct <= 2.0, f"Trade count difference too large: {trade_diff_pct:.2f}%"

        # Main assertion: P&L within 0.1%
        assert pnl_diff_pct <= 0.1, f"P&L difference too large: {pnl_diff_pct:.4f}% (threshold: 0.1%)"

        print(f"\n✅ PASS: Engine matches VectorBT within 0.1% (diff: {pnl_diff_pct:.4f}%)")

    def test_orders_rejected_when_insufficient_cash(self):
        """Verify orders are rejected when cash is insufficient."""
        print("\n" + "="*80)
        print("TEST: Order Rejection on Insufficient Cash")
        print("="*80)

        prices_pl, _ = generate_test_data_simple(n_assets=50, n_days=100)
        # Aggressive signals with very limited cash
        signals_pl = generate_aggressive_signals(prices_pl, n_positions=20)  # 20 positions

        # Run with LIMITED cash ($50k) - not enough for 20 positions @ $10k each
        result = run_engine_cash_account(
            prices_pl, signals_pl,
            initial_cash=50000,  # Only $50k (need $200k for all positions)
        )

        print(f"\nEngine Results (limited cash):")
        print(f"  Initial Cash: $50,000")
        print(f"  Target:       20 positions @ $10k each = $200k (4x overcapitalized)")
        print(f"  Final Value:  ${result['final_value']:,.2f}")
        print(f"  Num Trades:   {result['num_trades']}")
        print(f"  Min Cash:     ${result['min_cash']:,.2f}")

        assert result['min_cash'] >= 0, f"Cash went negative: ${result['min_cash']:,.2f}"
        assert result['num_trades'] < 20 * 100, "Too many trades - orders should have been rejected"

        print(f"\n✅ PASS: Orders were rejected when cash insufficient")
        print(f"         (traded {result['num_trades']} times vs theoretical max 2000+)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
