"""
Multi-Asset Portfolio Strategy Validation
Trades 20+ stocks using momentum/mean-reversion to generate 500+ trades

This validates frameworks with realistic portfolio strategies using Wiki data.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add paths
qengine_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(qengine_src))


def load_wiki_universe(
    start_date: str = "2013-01-01",
    end_date: str = "2017-12-31",
    min_price: float = 5.0,
    min_volume: float = 1000000,
    top_n: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load Wiki data for multiple stocks, filtered by liquidity.

    Returns:
        DataFrame with MultiIndex (date, ticker) and list of selected tickers
    """
    projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"

    print(f"Loading Wiki data from {wiki_path}")
    df = pd.read_parquet(wiki_path)

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Filter date range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df[mask].copy()

    # Calculate average volume and price for filtering
    avg_stats = (
        df.groupby("ticker")
        .agg(
            {
                "close": "mean",
                "volume": "mean",
            },
        )
        .reset_index()
    )

    # Filter by minimum price and volume
    liquid_tickers = (
        avg_stats[(avg_stats["close"] >= min_price) & (avg_stats["volume"] >= min_volume)]
        .nlargest(top_n * 2, "volume")["ticker"]
        .tolist()
    )  # Get more than needed

    # Further filter to ensure we have complete data for the period
    ticker_counts = df[df["ticker"].isin(liquid_tickers)].groupby("ticker").size()
    min_days = 250 * 4  # At least 4 years of data
    complete_tickers = ticker_counts[ticker_counts >= min_days].index.tolist()

    # Select top N by volume
    if len(complete_tickers) > top_n:
        volume_rank = (
            df[df["ticker"].isin(complete_tickers)]
            .groupby("ticker")["volume"]
            .mean()
            .nlargest(top_n)
        )
        selected_tickers = volume_rank.index.tolist()
    else:
        selected_tickers = complete_tickers[:top_n]

    print(f"Selected {len(selected_tickers)} liquid stocks: {selected_tickers[:10]}...")

    # Filter to selected tickers and create multi-index DataFrame
    result = df[df["ticker"].isin(selected_tickers)].copy()
    result = result.set_index(["date", "ticker"]).sort_index()

    # Also create a wide-format DataFrame for easier processing
    wide_close = result["close"].unstack("ticker").fillna(method="ffill")
    wide_open = result["open"].unstack("ticker").fillna(method="ffill")
    wide_high = result["high"].unstack("ticker").fillna(method="ffill")
    wide_low = result["low"].unstack("ticker").fillna(method="ffill")
    wide_volume = result["volume"].unstack("ticker").fillna(0)

    # Combine into single DataFrame with MultiIndex columns
    wide_data = pd.concat(
        {
            "open": wide_open,
            "high": wide_high,
            "low": wide_low,
            "close": wide_close,
            "volume": wide_volume,
        },
        axis=1,
    )

    return wide_data, selected_tickers


def momentum_ranking_strategy(
    data: pd.DataFrame,
    tickers: list[str],
    lookback: int = 20,
    holding_period: int = 5,
    top_n: int = 5,
    bottom_n: int = 5,
) -> pd.DataFrame:
    """
    Momentum ranking strategy: Long top N, Short bottom N by momentum.
    Rebalances every holding_period days.

    Returns DataFrame with entry/exit signals for each ticker.
    """
    # Extract close prices
    close_prices = data["close"][tickers]

    # Calculate momentum (simple returns over lookback period)
    momentum = close_prices.pct_change(lookback)

    # Initialize signals DataFrame
    signals = pd.DataFrame(index=data.index, columns=tickers)
    signals.fillna(0, inplace=True)

    # Rebalance every holding_period days
    rebalance_dates = data.index[::holding_period][1:]  # Skip first period for warmup

    print(f"Rebalancing on {len(rebalance_dates)} dates")

    for rebal_date in rebalance_dates:
        if rebal_date not in momentum.index:
            continue

        # Get momentum scores on rebalance date
        mom_scores = momentum.loc[rebal_date].dropna()

        if len(mom_scores) < top_n + bottom_n:
            continue

        # Select top and bottom stocks
        top_stocks = mom_scores.nlargest(top_n).index.tolist()
        bottom_stocks = mom_scores.nsmallest(bottom_n).index.tolist()

        # Generate signals (1 for long, -1 for short)
        for ticker in tickers:
            if ticker in top_stocks:
                signals.loc[rebal_date, ticker] = 1  # Long signal
            elif ticker in bottom_stocks:
                signals.loc[rebal_date, ticker] = -1  # Short signal
            else:
                signals.loc[rebal_date, ticker] = 0  # Close/No position

    # Forward fill signals until next rebalance
    signals = signals.fillna(method="ffill").fillna(0)

    # Convert to entry/exit signals for backtesting
    entries = pd.DataFrame(index=data.index, columns=tickers, data=False)
    exits = pd.DataFrame(index=data.index, columns=tickers, data=False)

    for ticker in tickers:
        # Detect position changes
        signals[ticker].diff()

        # Entry when position goes from 0 to non-zero
        entries[ticker] = (signals[ticker] != 0) & (signals[ticker].shift(1) == 0)

        # Exit when position goes to 0 from non-zero
        exits[ticker] = (signals[ticker] == 0) & (signals[ticker].shift(1) != 0)

    return signals, entries, exits


def run_multi_asset_qengine(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = 100000,
) -> dict[str, Any]:
    """Run multi-asset backtest using QEngine approach."""
    start_time = time.time()

    # Initialize portfolio
    cash = initial_capital
    positions = dict.fromkeys(tickers, 0)  # Shares held
    trades = []
    portfolio_values = []

    # Position sizing: Equal weight per position
    position_size = initial_capital / (len(tickers) // 2)  # Assume half positions at a time

    for date in data.index:
        # Get current prices
        current_prices = data.loc[date, "close"]

        # Process exits first (to free up capital)
        for ticker in tickers:
            if exits.loc[date, ticker] and positions[ticker] != 0:
                # Close position
                cash += positions[ticker] * current_prices[ticker]
                trades.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "action": "SELL",
                        "shares": positions[ticker],
                        "price": current_prices[ticker],
                    },
                )
                positions[ticker] = 0

        # Process entries
        for ticker in tickers:
            if entries.loc[date, ticker] and positions[ticker] == 0:
                # Open position
                shares_to_buy = position_size / current_prices[ticker]
                if cash >= position_size:
                    positions[ticker] = shares_to_buy
                    cash -= shares_to_buy * current_prices[ticker]
                    trades.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "action": "BUY",
                            "shares": shares_to_buy,
                            "price": current_prices[ticker],
                        },
                    )

        # Calculate portfolio value
        portfolio_value = cash
        for ticker in tickers:
            if positions[ticker] != 0:
                portfolio_value += positions[ticker] * current_prices[ticker]
        portfolio_values.append(portfolio_value)

    # Calculate metrics
    final_value = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_value / initial_capital - 1) * 100

    # Calculate daily returns
    portfolio_series = pd.Series(portfolio_values, index=data.index)
    returns = portfolio_series.pct_change().dropna()

    # Sharpe ratio
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100 if len(drawdown) > 0 else 0

    return {
        "framework": "QEngine",
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": len(trades),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "trades": trades,
        "execution_time": time.time() - start_time,
    }


def run_multi_asset_vectorbt(
    data: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    tickers: list[str],
    initial_capital: float = 100000,
) -> dict[str, Any]:
    """Run multi-asset backtest using VectorBT."""
    try:
        import vectorbt as vbt

        start_time = time.time()

        # Get close prices
        close_prices = data["close"][tickers]

        # Position sizing
        size = initial_capital / (len(tickers) // 2) / close_prices  # Equal weight

        # Run portfolio for each asset
        portfolio = vbt.Portfolio.from_signals(
            close_prices,
            entries=entries,
            exits=exits,
            size=size,
            init_cash=initial_capital,
            fees=0.0,
            slippage=0.0,
            freq="D",
            group_by=True,  # Treat as single portfolio
            cash_sharing=True,  # Share cash across assets
        )

        # Get metrics
        final_value = float(portfolio.final_value())
        total_return = float(portfolio.total_return()) * 100
        num_trades = len(portfolio.orders.records) if hasattr(portfolio.orders, "records") else 0

        sharpe = float(portfolio.sharpe_ratio()) if hasattr(portfolio, "sharpe_ratio") else 0
        max_dd = float(portfolio.max_drawdown()) * 100 if hasattr(portfolio, "max_drawdown") else 0

        return {
            "framework": "VectorBT",
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": num_trades,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "execution_time": time.time() - start_time,
        }

    except ImportError:
        return {"framework": "VectorBT", "error": "Not installed"}
    except Exception as e:
        return {"framework": "VectorBT", "error": str(e)}


def run_multi_asset_zipline(
    tickers: list[str],
    start_date: str = "2013-01-01",
    end_date: str = "2017-12-31",
    lookback: int = 20,
    holding_period: int = 5,
    top_n: int = 5,
    initial_capital: float = 100000,
) -> dict[str, Any]:
    """Run multi-asset strategy on Zipline."""
    from zipline import run_algorithm
    from zipline.api import (
        date_rules,
        get_datetime,
        order_target_percent,
        schedule_function,
        set_commission,
        symbols,
        time_rules,
    )
    from zipline.finance import commission

    trades = []

    def initialize(context):
        """Initialize the algorithm."""
        context.stocks = symbols(*tickers)
        context.lookback = lookback
        context.holding_period = holding_period
        context.top_n = top_n
        context.rebalance_count = 0
        context.trades = []

        # No commission for fair comparison
        set_commission(commission.PerShare(cost=0.0, min_trade_cost=0.0))

        # Schedule rebalancing
        schedule_function(
            rebalance,
            date_rules.every_day(),
            time_rules.market_open(),
        )

    def rebalance(context, data):
        """Rebalance the portfolio."""
        # Only rebalance every holding_period days
        context.rebalance_count += 1
        if context.rebalance_count % context.holding_period != 0:
            return

        # Get price history
        prices = data.history(
            context.stocks,
            "price",
            context.lookback + 1,
            "1d",
        )

        if len(prices) < context.lookback:
            return

        # Calculate momentum
        returns = prices.pct_change(context.lookback).iloc[-1]

        # Rank stocks
        ranked = returns.dropna().sort_values(ascending=False)

        if len(ranked) < context.top_n * 2:
            return

        # Select top and bottom stocks
        longs = ranked.head(context.top_n).index
        shorts = ranked.tail(context.top_n).index

        # Calculate position size
        long_weight = 0.5 / context.top_n  # 50% long
        short_weight = -0.5 / context.top_n  # 50% short

        # Rebalance portfolio
        for stock in context.stocks:
            if stock in longs:
                order_target_percent(stock, long_weight)
                context.trades.append(
                    {
                        "date": get_datetime(),
                        "ticker": stock.symbol,
                        "action": "LONG",
                        "weight": long_weight,
                    },
                )
            elif stock in shorts:
                order_target_percent(stock, short_weight)
                context.trades.append(
                    {
                        "date": get_datetime(),
                        "ticker": stock.symbol,
                        "action": "SHORT",
                        "weight": short_weight,
                    },
                )
            else:
                order_target_percent(stock, 0)
                if context.portfolio.positions[stock].amount != 0:
                    context.trades.append(
                        {
                            "date": get_datetime(),
                            "ticker": stock.symbol,
                            "action": "CLOSE",
                            "weight": 0,
                        },
                    )

    def analyze(context, perf):
        """Analyze results after backtest."""
        nonlocal trades
        trades = context.trades

    # Run the algorithm
    print(f"Running Zipline multi-asset strategy with {len(tickers)} stocks")

    start_time = time.time()

    result = run_algorithm(
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        initialize=initialize,
        analyze=analyze,
        capital_base=initial_capital,
        bundle="quandl",
    )

    # Extract metrics
    final_value = result["portfolio_value"].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    num_trades = len(trades)

    # Calculate Sharpe ratio
    returns = result["returns"]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "total_return": total_return,
        "num_trades": num_trades,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "trades": trades,
        "execution_time": time.time() - start_time,
    }


def main():
    """Run comprehensive multi-asset validation."""
    print("=" * 80)
    print("MULTI-ASSET PORTFOLIO STRATEGY VALIDATION")
    print("=" * 80)
    print()

    # Parameters
    start_date = "2013-01-01"
    end_date = "2017-12-31"
    initial_capital = 100000
    lookback = 20
    holding_period = 5
    top_n = 5
    bottom_n = 5
    num_stocks = 30

    print(f"Period: {start_date} to {end_date} (5 years)")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Strategy: Momentum Ranking (Long top {top_n}, Short bottom {bottom_n})")
    print(f"Rebalance: Every {holding_period} days")
    print(f"Universe: Top {num_stocks} liquid stocks from Wiki data")
    print()

    # Load data
    print("Loading Wiki universe...")
    try:
        data, tickers = load_wiki_universe(
            start_date=start_date,
            end_date=end_date,
            top_n=num_stocks,
        )
        print(f"Loaded data for {len(tickers)} stocks with {len(data)} trading days")
        print(f"Stocks: {', '.join(tickers[:10])}...")
        print()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Generate signals
    print("Generating momentum signals...")
    signals, entries, exits = momentum_ranking_strategy(
        data,
        tickers,
        lookback,
        holding_period,
        top_n,
        bottom_n,
    )

    total_entries = entries.sum().sum()
    total_exits = exits.sum().sum()
    print(f"Generated {total_entries} entry signals and {total_exits} exit signals")
    print(f"Expected total trades: {total_entries + total_exits}")
    print()

    results = []

    # 1. Run Zipline
    print("1. Running Zipline-Reloaded...")
    try:
        # Get ticker symbols that exist in Zipline
        zipline_tickers = []
        from zipline.data import bundles

        bundle = bundles.load("quandl")

        for ticker in tickers[:20]:  # Limit to 20 for Zipline
            try:
                asset = bundle.asset_finder.lookup_symbol(ticker, as_of_date=None)
                if asset:
                    zipline_tickers.append(ticker)
            except:
                pass

        if len(zipline_tickers) >= 10:
            print(f"   Using {len(zipline_tickers)} stocks available in Zipline")
            zipline_result = run_multi_asset_zipline(
                zipline_tickers,
                start_date=start_date,
                end_date=end_date,
                lookback=lookback,
                holding_period=holding_period,
                top_n=min(5, len(zipline_tickers) // 3),
                initial_capital=initial_capital,
            )
            results.append(zipline_result)
            print(
                f"   ✓ Final: ${zipline_result['final_value']:,.2f} | "
                f"Return: {zipline_result['total_return']:.2f}% | "
                f"Trades: {zipline_result['num_trades']}",
            )
        else:
            print("   ✗ Not enough tickers available in Zipline bundle")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # 2. Run QEngine
    print("\n2. Running QEngine...")
    try:
        qengine_result = run_multi_asset_qengine(
            data,
            signals,
            entries,
            exits,
            tickers,
            initial_capital,
        )
        results.append(qengine_result)
        print(
            f"   ✓ Final: ${qengine_result['final_value']:,.2f} | "
            f"Return: {qengine_result['total_return']:.2f}% | "
            f"Trades: {qengine_result['num_trades']}",
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # 3. Run VectorBT
    print("\n3. Running VectorBT...")
    try:
        vectorbt_result = run_multi_asset_vectorbt(
            data,
            entries,
            exits,
            tickers,
            initial_capital,
        )
        if "error" not in vectorbt_result:
            results.append(vectorbt_result)
            print(
                f"   ✓ Final: ${vectorbt_result['final_value']:,.2f} | "
                f"Return: {vectorbt_result['total_return']:.2f}% | "
                f"Trades: {vectorbt_result['num_trades']}",
            )
        else:
            print(f"   ✗ {vectorbt_result['error']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Display comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    if len(results) < 2:
        print("Not enough successful results for comparison")
        return

    print(
        f"\n{'Framework':<15} {'Final Value':<15} {'Return (%)':<12} {'Trades':<10} {'Sharpe':<10} {'Max DD (%)':<12} {'Time (s)':<10}",
    )
    print("-" * 94)

    for r in results:
        if "error" not in r:
            print(
                f"{r['framework']:<15} ${r['final_value']:<14,.2f} {r['total_return']:<11.2f} "
                f"{r.get('num_trades', 0):<10} {r.get('sharpe_ratio', 0):<10.2f} "
                f"{r.get('max_drawdown', 0):<11.2f} {r.get('execution_time', 0):<10.3f}",
            )

    # Performance comparison
    if len(results) > 1:
        print("\nPerformance Ranking (by execution time):")
        sorted_results = sorted(
            [r for r in results if "error" not in r],
            key=lambda x: x.get("execution_time", float("inf")),
        )

        if sorted_results:
            fastest_time = sorted_results[0].get("execution_time", 1)
            for i, r in enumerate(sorted_results, 1):
                speedup = r.get("execution_time", 0) / fastest_time if fastest_time > 0 else 0
                print(
                    f"  {i}. {r['framework']}: {r.get('execution_time', 0):.3f}s"
                    + (" (baseline)" if i == 1 else f" ({speedup:.1f}x slower)"),
                )

    # Check for VectorBT Pro
    print("\n4. Checking for VectorBT Pro...")
    try:
        import vectorbtpro as vbt_pro

        print("   ✓ VectorBT Pro is installed!")
        print("   Version:", vbt_pro.__version__)
    except ImportError:
        print("   ✗ VectorBT Pro not installed")
        print("   Note: VectorBT Pro requires a license. Using standard VectorBT for now.")


if __name__ == "__main__":
    main()
