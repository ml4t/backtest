"""
Analyze trade variance across 3 frameworks to identify EXACT sources of differences.

This script performs detailed trade-by-trade comparison to answer:
1. Why do trade counts differ so much? (1042 vs 80 vs 36)
2. Are there systematic price differences?
3. What's the commission cost breakdown?
4. Are there timing differences?
"""

import pandas as pd
from pathlib import Path


def load_trades():
    """Load trade logs from all frameworks."""
    base_dir = Path("tests/validation/trade_logs")

    trades = {
        "ml4t.backtest": pd.read_csv(base_dir / "trades_ml4t_backtest.csv", parse_dates=["timestamp"]),
        "VectorBT": pd.read_csv(base_dir / "trades_VectorBT.csv", parse_dates=["timestamp"]),
        "Backtrader": pd.read_csv(base_dir / "trades_Backtrader.csv", parse_dates=["timestamp"]),
    }

    return trades


def analyze_trade_counts(trades):
    """Analyze why trade counts differ."""
    print("=" * 80)
    print("TRADE COUNT ANALYSIS")
    print("=" * 80)

    for name, df in trades.items():
        buys = len(df[df["action"] == "BUY"])
        sells = len(df[df["action"] == "SELL"])
        print(f"\n{name}:")
        print(f"  Total Trades: {len(df)}")
        print(f"  BUY:  {buys}")
        print(f"  SELL: {sells}")

        # Analyze trade frequency
        df_sorted = df.sort_values("timestamp")
        print(f"  First Trade: {df_sorted.iloc[0]['timestamp']}")
        print(f"  Last Trade:  {df_sorted.iloc[-1]['timestamp']}")

        # Time between trades
        time_diffs = df_sorted["timestamp"].diff().dropna()
        print(f"  Avg Time Between Trades: {time_diffs.mean()}")
        print(f"  Min Time Between Trades: {time_diffs.min()}")


def analyze_commissions(trades):
    """Analyze commission costs."""
    print("\n" + "=" * 80)
    print("COMMISSION ANALYSIS")
    print("=" * 80)

    for name, df in trades.items():
        total_commission = df["commission"].sum()
        avg_commission = df["commission"].mean()
        print(f"\n{name}:")
        print(f"  Total Commissions: ${total_commission:,.2f}")
        print(f"  Avg Per Trade: ${avg_commission:.2f}")
        print(f"  Commission as % of Initial Capital: {(total_commission / 100000) * 100:.4f}%")


def analyze_timing(trades):
    """Analyze trade timing alignment."""
    print("\n" + "=" * 80)
    print("TIMING ANALYSIS")
    print("=" * 80)

    # Get unique trade dates from each framework
    dates = {}
    for name, df in trades.items():
        dates[name] = set(df["timestamp"].dt.date)

    # Find common dates
    common_dates = dates["ml4t.backtest"] & dates["VectorBT"] & dates["Backtrader"]
    print(f"\nCommon trade dates across all frameworks: {len(common_dates)}")

    # Find unique dates
    for name in trades.keys():
        unique = dates[name] - (dates["VectorBT"] | dates["Backtrader"])
        if name == "ml4t.backtest":
            unique = dates[name] - dates["VectorBT"] - dates["Backtrader"]
        print(f"{name} unique dates: {len(unique)}")


def find_matching_trades(trades):
    """Try to match trades across frameworks by timestamp and action."""
    print("\n" + "=" * 80)
    print("TRADE MATCHING ANALYSIS")
    print("=" * 80)

    # Convert to date (ignore time) for matching
    for name in trades.keys():
        trades[name]["date"] = trades[name]["timestamp"].dt.date
        trades[name]["date_action"] = trades[name]["date"].astype(str) + "_" + trades[name]["action"]

    # Find common date+action pairs
    ml4t_pairs = set(trades["ml4t.backtest"]["date_action"])
    vbt_pairs = set(trades["VectorBT"]["date_action"])
    bt_pairs = set(trades["Backtrader"]["date_action"])

    common_all = ml4t_pairs & vbt_pairs & bt_pairs
    print(f"\nCommon date+action pairs (all 3 frameworks): {len(common_all)}")

    # VectorBT vs ml4t.backtest
    common_ml4t_vbt = ml4t_pairs & vbt_pairs
    print(f"Common date+action pairs (ml4t.backtest vs VectorBT): {len(common_ml4t_vbt)}")

    # ml4t.backtest unique
    ml4t_unique = ml4t_pairs - vbt_pairs - bt_pairs
    print(f"ml4t.backtest unique date+action pairs: {len(ml4t_unique)}")

    # Show some examples of ml4t.backtest unique trades
    print("\nSample of ml4t.backtest unique trades:")
    unique_trades = trades["ml4t.backtest"][trades["ml4t.backtest"]["date_action"].isin(list(ml4t_unique)[:10])]
    print(unique_trades[["timestamp", "action", "quantity", "price", "value"]].to_string(index=False))


def analyze_price_differences(trades):
    """Analyze price differences on matching trades."""
    print("\n" + "=" * 80)
    print("PRICE COMPARISON (Matching Dates)")
    print("=" * 80)

    # Simple approach: compare BUY trades by date
    for action in ["BUY", "SELL"]:
        ml4t_trades = trades["ml4t.backtest"][trades["ml4t.backtest"]["action"] == action].copy()
        vbt_trades = trades["VectorBT"][trades["VectorBT"]["action"] == action].copy()

        if len(ml4t_trades) == 0 or len(vbt_trades) == 0:
            continue

        ml4t_trades["date"] = ml4t_trades["timestamp"].dt.date
        vbt_trades["date"] = vbt_trades["timestamp"].dt.date

        # Group by date and get average price
        ml4t_by_date = ml4t_trades.groupby("date")["price"].mean()
        vbt_by_date = vbt_trades.groupby("date")["price"].mean()

        # Find common dates
        common = ml4t_by_date.index.intersection(vbt_by_date.index)

        if len(common) > 0:
            price_diffs = []
            for date in common:
                diff = ((ml4t_by_date[date] - vbt_by_date[date]) / vbt_by_date[date]) * 100
                price_diffs.append(diff)

            print(f"\n{action} Trades:")
            print(f"  Common dates: {len(common)}")
            print(f"  Avg price difference: {sum(price_diffs) / len(price_diffs):.4f}%")
            print(f"  Max price difference: {max(price_diffs):.4f}%")
            print(f"  Min price difference: {min(price_diffs):.4f}%")


def main():
    """Run all analyses."""
    print("\nLOADING TRADE LOGS...")
    trades = load_trades()

    analyze_trade_counts(trades)
    analyze_commissions(trades)
    analyze_timing(trades)
    find_matching_trades(trades)
    analyze_price_differences(trades)

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    ml4t_count = len(trades["ml4t.backtest"])
    vbt_count = len(trades["VectorBT"])
    bt_count = len(trades["Backtrader"])

    print(f"\n1. TRADE COUNT DISCREPANCY:")
    print(f"   - ml4t.backtest: {ml4t_count} trades ({ml4t_count / vbt_count:.1f}x more than VectorBT)")
    print(f"   - VectorBT: {vbt_count} trades")
    print(f"   - Backtrader: {bt_count} trades")
    print(f"   - Extra trades (ml4t.backtest - VectorBT): {ml4t_count - vbt_count}")

    ml4t_commission = trades["ml4t.backtest"]["commission"].sum()
    vbt_commission = trades["VectorBT"]["commission"].sum()
    bt_commission = trades["Backtrader"]["commission"].sum()

    print(f"\n2. COMMISSION IMPACT:")
    print(f"   - ml4t.backtest total: ${ml4t_commission:,.2f}")
    print(f"   - VectorBT total: ${vbt_commission:,.2f}")
    print(f"   - Backtrader total: ${bt_commission:,.2f}")
    print(f"   - Extra commission cost (ml4t.backtest - VectorBT): ${ml4t_commission - vbt_commission:,.2f}")
    print(f"   - This explains {((ml4t_commission - vbt_commission) / 1143.72) * 100:.1f}% of $1,143.72 variance")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
