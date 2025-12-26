"""Risk Parity Portfolio with ml4t.backtest.

This example demonstrates:
- Risk parity (equal risk contribution) portfolio
- Custom optimizer implementation without external dependencies
- Volatility-based weight calculation
- Integration with TargetWeightExecutor

No external optimizer dependencies required - implements risk parity from scratch.
"""

import polars as pl

from ml4t.backtest import (
    DataFeed,
    Engine,
    ExecutionMode,
    RebalanceConfig,
    Strategy,
    TargetWeightExecutor,
)


def generate_sample_data(assets: list[str], n_bars: int = 504, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic OHLCV data with realistic volatility differences."""
    from datetime import datetime, timedelta

    import numpy as np

    np.random.seed(seed)

    rows = []
    base_date = datetime(2020, 1, 1)

    # Different volatility profiles
    asset_vols = {
        "SPY": 0.012,  # Low vol (S&P 500)
        "QQQ": 0.018,  # Medium vol (Nasdaq)
        "IWM": 0.020,  # Higher vol (Russell 2000)
        "TLT": 0.010,  # Low vol (Bonds)
        "GLD": 0.008,  # Lowest vol (Gold)
        "EEM": 0.025,  # Highest vol (Emerging Markets)
    }

    for asset in assets:
        sigma = asset_vols.get(asset, 0.015)
        mu = 0.0003  # Similar expected return
        returns = np.random.normal(mu, sigma, n_bars)
        prices = 100 * np.exp(np.cumsum(returns))

        for i in range(n_bars):
            timestamp = base_date + timedelta(days=i)
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, sigma / 2)))
            low = close * (1 - abs(np.random.normal(0, sigma / 2)))
            open_price = (high + low) / 2

            rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": np.random.randint(100000, 1000000),
                }
            )

    return pl.DataFrame(rows).sort(["timestamp", "asset"])


class RiskParityStrategy(Strategy):
    """Risk Parity portfolio: equal risk contribution from each asset.

    Weights are inversely proportional to volatility:
    w_i = (1/vol_i) / sum(1/vol_j for all j)

    This is a simplified version that doesn't account for correlations.
    For full risk parity with correlations, use riskfolio-lib or skfolio.
    """

    def __init__(
        self,
        assets: list[str],
        lookback: int = 63,  # ~3 months for volatility estimation
        rebalance_frequency: int = 21,  # Monthly
    ):
        """Initialize strategy.

        Args:
            assets: List of asset symbols.
            lookback: Rolling window for volatility estimation.
            rebalance_frequency: Bars between rebalances.
        """
        self.assets = assets
        self.lookback = lookback
        self.rebalance_frequency = rebalance_frequency
        self.bar_count = 0

        # Price history
        self.prices: dict[str, list[float]] = {asset: [] for asset in assets}

        # Rebalancer
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=200,
                min_weight_change=0.02,
                allow_fractional=True,
            )
        )

    def on_data(self, timestamp, data, context, broker):
        """Called on each bar."""
        # Update prices
        for asset in self.assets:
            if asset in data and "close" in data[asset]:
                self.prices[asset].append(data[asset]["close"])

        self.bar_count += 1

        # Check history
        min_history = min(len(p) for p in self.prices.values())
        if min_history < self.lookback + 1:
            return

        # Rebalance check
        if self.bar_count % self.rebalance_frequency != 0:
            return

        # Compute risk parity weights
        target_weights = self._compute_risk_parity_weights()

        if target_weights:
            orders = self.executor.execute(target_weights, data, broker)
            if orders:
                print(f"\n[{timestamp}] Risk Parity Rebalance:")
                for asset, w in sorted(target_weights.items(), key=lambda x: -x[1]):
                    vol = self._compute_volatility(asset)
                    print(f"  {asset}: {w:.1%} (vol: {vol:.1%})")

    def _compute_volatility(self, asset: str) -> float:
        """Compute annualized volatility for an asset."""
        import numpy as np

        prices = self.prices[asset][-self.lookback :]
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252)

    def _compute_risk_parity_weights(self) -> dict[str, float]:
        """Compute inverse-volatility weights.

        Simple risk parity: w_i = (1/vol_i) / sum(1/vol_j)
        """

        # Compute volatilities
        volatilities = {}
        for asset in self.assets:
            vol = self._compute_volatility(asset)
            if vol > 0:
                volatilities[asset] = vol

        if not volatilities:
            return {}

        # Inverse volatility weights
        inv_vols = {asset: 1.0 / vol for asset, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        # Normalize to sum to 1
        weights = {asset: inv_vol / total_inv_vol for asset, inv_vol in inv_vols.items()}

        return weights


def main():
    """Run risk parity backtest."""
    # ETF universe with different volatilities
    assets = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EEM"]

    print("Risk Parity Portfolio Example")
    print("=" * 50)
    print("Assets: SPY (S&P 500), QQQ (Nasdaq), IWM (Russell 2000)")
    print("        TLT (Bonds), GLD (Gold), EEM (Emerging Markets)")
    print("\nRisk parity allocates inversely to volatility,")
    print("so low-vol assets get higher weights.\n")

    # Generate data
    print("Generating sample data...")
    df = generate_sample_data(assets, n_bars=504)

    # Create components
    feed = DataFeed(prices_df=df)
    strategy = RiskParityStrategy(assets, lookback=63, rebalance_frequency=21)

    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,
    )

    # Run
    print("Running backtest...")
    result = engine.run()

    # Results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Value:     ${result['final_value']:,.2f}")
    print(f"Total Return:    {result['total_return']:.2%}")
    print(f"Sharpe Ratio:    {result['sharpe']:.2f}")

    print("\nFinal Positions:")
    for asset, pos in sorted(engine.broker.positions.items()):
        value = pos.quantity * pos.entry_price
        pct = value / result["final_value"]
        print(f"  {asset}: ${value:,.0f} ({pct:.1%})")


if __name__ == "__main__":
    main()
