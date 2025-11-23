"""Validation tests for ml4t-backtest risk management features.

Tests position-level risk rules, execution model, and portfolio risk management.
"""

from datetime import datetime

import polars as pl

from ml4t.backtest import (
    DataFeed,
    Engine,
    Strategy,
    VolumeParticipationLimit,
    LinearImpact,
)
from ml4t.backtest.risk import (
    # Position rules
    StopLoss,
    TakeProfit,
    TimeExit,
    TrailingStop,
    ScaledExit,
    RuleChain,
    # Portfolio risk
    RiskManager,
    MaxDrawdownLimit,
    MaxPositionsLimit,
    DailyLossLimit,
)


def create_price_data(
    n_bars: int = 10,
    start_price: float = 100.0,
    daily_returns: list[float] | None = None,
    volume: float = 10000.0,
) -> pl.DataFrame:
    """Create test price data."""
    dates = [datetime(2024, 1, i + 1) for i in range(n_bars)]

    if daily_returns:
        prices = [start_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        closes = prices[:n_bars]
    else:
        closes = [start_price] * n_bars

    return pl.DataFrame({
        "timestamp": dates,
        "asset": ["AAPL"] * n_bars,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [volume] * n_bars,
    })


# === TEST 1: Stop-Loss ===

def test_stop_loss():
    """Test stop-loss triggers at correct level."""
    print("\n" + "=" * 60)
    print("TEST 1: Stop-Loss Rule")
    print("=" * 60)

    # Price goes: 100 -> 98 -> 95 -> 92 -> 90 (10% decline)
    returns = [0.0, -0.02, -0.0306, -0.0316, -0.0217]
    prices = create_price_data(n_bars=5, start_price=100.0, daily_returns=returns)

    class StopLossStrategy(Strategy):
        def __init__(self):
            self.entered = False

        def on_start(self, broker):
            broker.set_position_rules(StopLoss(pct=0.05))  # 5% stop

        def on_data(self, timestamp, data, context, broker):
            if not self.entered:
                broker.submit_order("AAPL", 100)
                self.entered = True

    feed = DataFeed(prices_df=prices)
    engine = Engine(feed=feed, strategy=StopLossStrategy(), initial_cash=100000.0)
    results = engine.run()

    print(f"Initial price: $100.00")
    print(f"Stop level: 5%")
    print(f"Trades: {len(results['trades'])}")

    if results['trades']:
        trade = results['trades'][0]
        print(f"Entry: ${trade.entry_price:.2f}")
        print(f"Exit: ${trade.exit_price:.2f}")
        pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        print(f"Loss: {pnl_pct:.1%}")

        # Stop should trigger around 95 (5% below 100)
        assert trade.exit_price <= 95.5, f"Stop should trigger at ~$95, got ${trade.exit_price:.2f}"
        print("✓ Stop-loss triggered correctly")
        return True
    else:
        print("✗ No trades - stop didn't trigger")
        return False


# === TEST 2: Take-Profit ===

def test_take_profit():
    """Test take-profit triggers at correct level."""
    print("\n" + "=" * 60)
    print("TEST 2: Take-Profit Rule")
    print("=" * 60)

    # Price goes: 100 -> 102 -> 105 -> 110 -> 115 (15% gain)
    returns = [0.02, 0.0294, 0.0476, 0.0455]  # Cumulative: 2%, 5%, 10%, 15%
    prices = create_price_data(n_bars=5, start_price=100.0, daily_returns=returns)

    class TakeProfitStrategy(Strategy):
        def __init__(self):
            self.entered = False

        def on_start(self, broker):
            broker.set_position_rules(TakeProfit(pct=0.10))  # 10% take-profit

        def on_data(self, timestamp, data, context, broker):
            if not self.entered:
                broker.submit_order("AAPL", 100)
                self.entered = True

    feed = DataFeed(prices_df=prices)
    engine = Engine(feed=feed, strategy=TakeProfitStrategy(), initial_cash=100000.0)
    results = engine.run()

    print(f"Initial price: $100.00")
    print(f"Take-profit level: 10%")
    print(f"Trades: {len(results['trades'])}")

    if results['trades']:
        trade = results['trades'][0]
        print(f"Entry: ${trade.entry_price:.2f}")
        print(f"Exit: ${trade.exit_price:.2f}")
        pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        print(f"Gain: {pnl_pct:.1%}")

        # Take-profit should trigger around 110 (10% above 100)
        assert trade.exit_price >= 109.5, f"TP should trigger at ~$110, got ${trade.exit_price:.2f}"
        print("✓ Take-profit triggered correctly")
        return True
    else:
        print("✗ No trades - take-profit didn't trigger")
        return False


# === TEST 3: Time Exit ===

def test_time_exit():
    """Test time-based exit after N bars."""
    print("\n" + "=" * 60)
    print("TEST 3: Time Exit Rule")
    print("=" * 60)

    prices = create_price_data(n_bars=10, start_price=100.0)

    class TimeExitStrategy(Strategy):
        def __init__(self):
            self.entered = False

        def on_start(self, broker):
            broker.set_position_rules(TimeExit(max_bars=3))  # Exit after 3 bars

        def on_data(self, timestamp, data, context, broker):
            if not self.entered:
                broker.submit_order("AAPL", 100)
                self.entered = True

    feed = DataFeed(prices_df=prices)
    engine = Engine(feed=feed, strategy=TimeExitStrategy(), initial_cash=100000.0)
    results = engine.run()

    print(f"Max bars: 3")
    print(f"Trades: {len(results['trades'])}")

    if results['trades']:
        trade = results['trades'][0]
        print(f"Bars held: {trade.bars_held}")

        assert trade.bars_held <= 3, f"Should exit after 3 bars, held {trade.bars_held}"
        print("✓ Time exit triggered correctly")
        return True
    else:
        print("✗ No trades - time exit didn't trigger")
        return False


# === TEST 4: Trailing Stop ===

def test_trailing_stop():
    """Test trailing stop follows price up then triggers on reversal."""
    print("\n" + "=" * 60)
    print("TEST 4: Trailing Stop Rule")
    print("=" * 60)

    # Price goes: 100 -> 105 -> 110 -> 108 -> 103 -> 98 (peaks at 110, drops 11%)
    returns = [0.05, 0.0476, -0.0182, -0.0463, -0.0485]  # Cumulative: +5%, +10%, +8%, +3%, -2%
    prices = create_price_data(n_bars=6, start_price=100.0, daily_returns=returns)

    class TrailingStopStrategy(Strategy):
        def __init__(self):
            self.entered = False

        def on_start(self, broker):
            broker.set_position_rules(TrailingStop(pct=0.05))  # 5% trailing stop

        def on_data(self, timestamp, data, context, broker):
            if not self.entered:
                broker.submit_order("AAPL", 100)
                self.entered = True

    feed = DataFeed(prices_df=prices)
    engine = Engine(feed=feed, strategy=TrailingStopStrategy(), initial_cash=100000.0)
    results = engine.run()

    print(f"Trail: 5%")
    print(f"Peak price: ~$110")
    print(f"Trades: {len(results['trades'])}")

    if results['trades']:
        trade = results['trades'][0]
        print(f"Entry: ${trade.entry_price:.2f}")
        print(f"Exit: ${trade.exit_price:.2f}")
        pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        print(f"Return: {pnl_pct:.1%}")

        # Trailing stop from 110 peak = 104.5 level
        # Should lock in some profit (not full peak)
        assert trade.exit_price > 100, "Should have locked in some profit"
        print("✓ Trailing stop triggered correctly")
        return True
    else:
        print("✗ No trades - trailing stop didn't trigger")
        return False


# === TEST 5: Rule Chain (Stop-Loss + Take-Profit) ===

def test_rule_chain():
    """Test combined stop-loss and take-profit rules."""
    print("\n" + "=" * 60)
    print("TEST 5: Rule Chain (SL + TP)")
    print("=" * 60)

    # Price drops to trigger stop
    returns = [0.0, -0.02, -0.04, -0.02, -0.02]
    prices = create_price_data(n_bars=5, start_price=100.0, daily_returns=returns)

    class RuleChainStrategy(Strategy):
        def __init__(self):
            self.entered = False

        def on_start(self, broker):
            rules = RuleChain([
                StopLoss(pct=0.05),
                TakeProfit(pct=0.10),
            ])
            broker.set_position_rules(rules)

        def on_data(self, timestamp, data, context, broker):
            if not self.entered:
                broker.submit_order("AAPL", 100)
                self.entered = True

    feed = DataFeed(prices_df=prices)
    engine = Engine(feed=feed, strategy=RuleChainStrategy(), initial_cash=100000.0)
    results = engine.run()

    print(f"Rules: Stop-Loss 5%, Take-Profit 10%")
    print(f"Trades: {len(results['trades'])}")

    if results['trades']:
        trade = results['trades'][0]
        pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        print(f"Exit at: ${trade.exit_price:.2f} ({pnl_pct:.1%})")

        # Stop should trigger first (5% loss)
        assert pnl_pct < 0, "Should have exited at loss (stop triggered)"
        print("✓ Rule chain worked correctly")
        return True
    else:
        print("✗ No trades")
        return False


# === TEST 6: Volume Participation Limit ===

def test_volume_limit():
    """Test volume participation limits for partial fills."""
    print("\n" + "=" * 60)
    print("TEST 6: Volume Participation Limit")
    print("=" * 60)

    # 10 bars with 1000 volume each
    prices = create_price_data(n_bars=10, start_price=100.0, volume=1000.0)

    class LargeBuyStrategy(Strategy):
        def __init__(self):
            self.bought = False

        def on_data(self, timestamp, data, context, broker):
            if not self.bought:
                broker.submit_order("AAPL", 500)  # 500 shares vs 1000 vol/bar
                self.bought = True

    # With 10% participation limit, max 100 shares/bar
    feed = DataFeed(prices_df=prices)
    vol_limit = VolumeParticipationLimit(max_participation=0.10)
    engine = Engine(
        feed=feed,
        strategy=LargeBuyStrategy(),
        initial_cash=100000.0,
        execution_limits=vol_limit,
    )
    results = engine.run()

    total_filled = sum(f.quantity for f in results['fills'])
    num_fills = len(results['fills'])

    print(f"Order: 500 shares")
    print(f"Volume limit: 10% of 1000 = 100 shares/bar")
    print(f"Fills: {num_fills}")
    print(f"Total filled: {total_filled}")

    assert num_fills == 5, f"Should take 5 bars to fill 500 shares, got {num_fills}"
    assert total_filled == 500, f"Should fill 500 total, got {total_filled}"
    print("✓ Volume participation limit works correctly")
    return True


# === TEST 7: Market Impact ===

def test_market_impact():
    """Test market impact increases execution cost."""
    print("\n" + "=" * 60)
    print("TEST 7: Market Impact Model")
    print("=" * 60)

    prices = create_price_data(n_bars=5, start_price=100.0, volume=1000.0)

    class BuyStrategy(Strategy):
        def __init__(self):
            self.bought = False

        def on_data(self, timestamp, data, context, broker):
            if not self.bought:
                broker.submit_order("AAPL", 500)
                self.bought = True

    # Without impact
    feed1 = DataFeed(prices_df=prices)
    engine1 = Engine(feed=feed1, strategy=BuyStrategy(), initial_cash=100000.0)
    results1 = engine1.run()
    price_no_impact = results1['fills'][0].price

    # With impact
    feed2 = DataFeed(prices_df=prices)
    impact = LinearImpact(coefficient=0.1)
    engine2 = Engine(
        feed=feed2,
        strategy=BuyStrategy(),
        initial_cash=100000.0,
        market_impact_model=impact,
    )
    results2 = engine2.run()
    price_with_impact = results2['fills'][0].price

    print(f"Price without impact: ${price_no_impact:.4f}")
    print(f"Price with impact: ${price_with_impact:.4f}")
    print(f"Impact cost: ${price_with_impact - price_no_impact:.4f}")

    assert price_with_impact > price_no_impact, "Impact should increase buy price"
    print("✓ Market impact model works correctly")
    return True


# === TEST 8: Portfolio Max Drawdown Limit ===

def test_max_drawdown_limit():
    """Test portfolio drawdown limit halts trading."""
    print("\n" + "=" * 60)
    print("TEST 8: Portfolio Max Drawdown Limit")
    print("=" * 60)

    # Price drops 30%
    returns = [0.0, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05]
    prices = create_price_data(n_bars=7, start_price=100.0, daily_returns=returns)

    class DrawdownStrategy(Strategy):
        def __init__(self, risk_manager):
            self.risk_manager = risk_manager
            self.entered = False

        def on_start(self, broker):
            self.risk_manager.initialize(broker.initial_cash)

        def on_data(self, timestamp, data, context, broker):
            # Update risk manager
            positions = {}
            for asset, pos in broker.positions.items():
                price = data.get(asset, {}).get('close', pos.entry_price)
                positions[asset] = pos.quantity * price

            self.risk_manager.update(broker.get_account_value(), positions, timestamp)

            # Only trade if not halted
            if not self.risk_manager.is_halted and not self.entered:
                broker.submit_order("AAPL", 1000)
                self.entered = True

    risk_manager = RiskManager(limits=[
        MaxDrawdownLimit(max_drawdown=0.20, warn_threshold=0.10),
    ])

    feed = DataFeed(prices_df=prices)
    engine = Engine(
        feed=feed,
        strategy=DrawdownStrategy(risk_manager),
        initial_cash=100000.0,
    )
    results = engine.run()

    print(f"Max drawdown limit: 20%")
    print(f"Final equity: ${results['final_value']:,.2f}")
    print(f"Max drawdown: {results['max_drawdown']:.1%}")
    print(f"Risk halted: {risk_manager.is_halted}")
    print(f"Halt reason: {risk_manager.halt_reason}")

    assert risk_manager.is_halted, "Should be halted after 20% drawdown"
    print("✓ Max drawdown limit works correctly")
    return True


# === TEST 9: Max Positions Limit ===

def test_max_positions_limit():
    """Test portfolio position count limit."""
    print("\n" + "=" * 60)
    print("TEST 9: Max Positions Limit")
    print("=" * 60)

    from ml4t.backtest.risk.portfolio import PortfolioState

    state = PortfolioState(
        equity=100000,
        initial_equity=100000,
        high_water_mark=100000,
        current_drawdown=0.0,
        num_positions=5,
        positions={'AAPL': 20000, 'MSFT': 20000, 'GOOG': 20000, 'AMZN': 20000, 'META': 20000},
        daily_pnl=0,
        gross_exposure=100000,
        net_exposure=100000,
    )

    limit = MaxPositionsLimit(max_positions=3)
    result = limit.check(state)

    print(f"Max positions: 3")
    print(f"Current positions: 5")
    print(f"Breached: {result.breached}")
    print(f"Action: {result.action}")

    assert result.breached, "Should breach with 5 positions when limit is 3"
    print("✓ Max positions limit works correctly")
    return True


# === TEST 10: Daily Loss Limit ===

def test_daily_loss_limit():
    """Test daily loss limit halts trading."""
    print("\n" + "=" * 60)
    print("TEST 10: Daily Loss Limit")
    print("=" * 60)

    from ml4t.backtest.risk.portfolio import PortfolioState

    state = PortfolioState(
        equity=97000,  # Down 3%
        initial_equity=100000,
        high_water_mark=100000,
        current_drawdown=0.03,
        num_positions=1,
        positions={'AAPL': 50000},
        daily_pnl=-3000,  # Lost 3% today
        gross_exposure=50000,
        net_exposure=50000,
    )

    limit = DailyLossLimit(max_daily_loss_pct=0.02)  # 2% daily limit
    result = limit.check(state)

    print(f"Max daily loss: 2%")
    print(f"Actual daily loss: 3%")
    print(f"Breached: {result.breached}")
    print(f"Action: {result.action}")

    assert result.breached, "Should breach with 3% loss when limit is 2%"
    print("✓ Daily loss limit works correctly")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("ML4T-BACKTEST RISK MANAGEMENT VALIDATION")
    print("=" * 60)

    tests = [
        ("Stop-Loss", test_stop_loss),
        ("Take-Profit", test_take_profit),
        ("Time Exit", test_time_exit),
        ("Trailing Stop", test_trailing_stop),
        ("Rule Chain", test_rule_chain),
        ("Volume Limit", test_volume_limit),
        ("Market Impact", test_market_impact),
        ("Max Drawdown", test_max_drawdown_limit),
        ("Max Positions", test_max_positions_limit),
        ("Daily Loss", test_daily_loss_limit),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)

    for name, p, err in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")
        if err:
            print(f"         Error: {err}")

    print(f"\nTotal: {passed}/{len(results)} passed")

    if failed == 0:
        print("\n✓ ALL RISK VALIDATION TESTS PASSED!")
    else:
        print(f"\n✗ {failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
