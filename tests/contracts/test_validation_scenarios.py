"""Validation scenarios bridged into pytest CI.

Runs the 16 cross-framework validation scenarios through ml4t only
(no external framework venvs required) and checks:
1. Each scenario runs without error and produces trades
2. Each scenario passes all accounting invariants
3. Smoke: final value is positive

This does NOT compare against external frameworks — that's the separate
validation suite. This ensures ml4t scenarios don't regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add validation directory to path for imports
_VALIDATION_DIR = Path(__file__).parent.parent.parent / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

from tests.helpers.invariants import assert_result_invariants  # noqa: E402

# Try to import validation infrastructure
try:
    from common.data_generators import (  # noqa: E402
        generate_bracket_data,
        generate_random_walk,
        generate_rule_combo_data,
        generate_short_signals,
        generate_short_trending_data,
        generate_stop_loss_data,
        generate_stress_data,
        generate_take_profit_data,
        generate_trending_data,
    )
    from scenarios.definitions import SCENARIOS  # noqa: E402

    _HAS_VALIDATION = True
except ImportError:
    _HAS_VALIDATION = False
    SCENARIOS = {}


pytestmark = pytest.mark.skipif(
    not _HAS_VALIDATION, reason="Validation infrastructure not available"
)


# Map generator names to functions
_GENERATORS = {}
if _HAS_VALIDATION:
    _GENERATORS = {
        "generate_random_walk": generate_random_walk,
        "generate_stop_loss_data": generate_stop_loss_data,
        "generate_take_profit_data": generate_take_profit_data,
        "generate_trending_data": generate_trending_data,
        "generate_bracket_data": generate_bracket_data,
        "generate_short_signals": generate_short_signals,
        "generate_short_trending_data": generate_short_trending_data,
        "generate_rule_combo_data": generate_rule_combo_data,
        "generate_stress_data": generate_stress_data,
    }


def _run_ml4t_scenario(scenario_config):
    """Run an ml4t scenario and return the BacktestResult."""
    import pandas as pd
    import polars as pl

    from ml4t.backtest import (
        BacktestConfig,
        DataFeed,
        Engine,
        StopLoss,
        Strategy,
        TakeProfit,
        TrailingStop,
    )
    from ml4t.backtest.types import OrderSide

    # Generate data
    gen_fn = _GENERATORS.get(scenario_config.data_generator)
    if gen_fn is None:
        pytest.skip(f"Generator {scenario_config.data_generator} not found")

    gen_result = gen_fn(**scenario_config.data_kwargs)
    prices_df = gen_result[0]  # pandas DataFrame
    entries = gen_result[1]  # numpy boolean array
    exits = gen_result[2] if len(gen_result) > 2 else None

    asset = "ASSET"

    # Convert to Polars
    timestamps = []
    for ts in prices_df.index:
        if isinstance(ts, pd.Timestamp):
            timestamps.append(ts.to_pydatetime().replace(tzinfo=None))
        else:
            timestamps.append(ts)

    prices_pl = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": [asset] * len(prices_df),
            "open": prices_df["open"].tolist(),
            "high": prices_df["high"].tolist(),
            "low": prices_df["low"].tolist(),
            "close": prices_df["close"].tolist(),
            "volume": prices_df["volume"].astype(float).tolist(),
        }
    )

    # Build signals
    signals_dict = {
        "timestamp": timestamps,
        "asset": [asset] * len(prices_df),
    }
    for col in scenario_config.signal_columns:
        if col == "entry":
            signals_dict["entry"] = entries.tolist()
        elif col == "exit" and exits is not None:
            signals_dict["exit"] = exits.tolist()
        elif "entry" in col:
            signals_dict[col] = entries.tolist()
        elif "exit" in col and exits is not None:
            signals_dict[col] = exits.tolist()
        else:
            signals_dict[col] = [False] * len(entries)

    signals_pl = pl.DataFrame(signals_dict)

    # Build strategy
    strategy_type = scenario_config.strategy_type
    is_short = "short" in strategy_type

    class ScenarioStrategy(Strategy):
        def __init__(self):
            self._entered = False

        def on_data(self, timestamp, data, context, broker):
            if asset not in data:
                return
            bar = data[asset]
            sigs = bar.get("signals", {})

            if strategy_type == "long_signal":
                if sigs.get("entry") and not broker.get_position(asset):
                    broker.submit_order(asset, scenario_config.shares, OrderSide.BUY)
                elif sigs.get("exit") and broker.get_position(asset):
                    broker.close_position(asset)
            elif strategy_type == "long_short":
                pos = broker.get_position(asset)
                if sigs.get("long_entry") and pos is None:
                    broker.submit_order(asset, scenario_config.shares, OrderSide.BUY)
                elif sigs.get("long_exit") and pos and pos.quantity > 0:
                    broker.close_position(asset)
                elif sigs.get("short_entry") and pos is None:
                    broker.submit_order(asset, scenario_config.shares, OrderSide.SELL)
                elif sigs.get("short_exit") and pos and pos.quantity < 0:
                    broker.close_position(asset)
            elif strategy_type == "short_only":
                pos = broker.get_position(asset)
                if sigs.get("short_entry") and pos is None:
                    broker.submit_order(asset, scenario_config.shares, OrderSide.SELL)
                elif sigs.get("short_exit") and pos:
                    broker.close_position(asset)
            elif strategy_type in ("single_entry", "risk_entry_only"):
                if sigs.get("entry") and not self._entered and not broker.get_position(asset):
                    side = OrderSide.SELL if is_short else OrderSide.BUY
                    broker.submit_order(asset, scenario_config.shares, side)
                    if strategy_type == "single_entry":
                        self._entered = True
                if (
                    strategy_type == "single_entry"
                    and sigs.get("exit")
                    and broker.get_position(asset)
                ):
                    broker.close_position(asset)

    # Build config
    cfg_kwargs = {
        "initial_cash": scenario_config.initial_cash,
        "commission_rate": scenario_config.constants.get("commission_rate", 0.0),
        "slippage_rate": scenario_config.constants.get("slippage_rate", 0.0),
    }
    cfg_kwargs.update(scenario_config.ml4t_config)
    config = BacktestConfig(**cfg_kwargs)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    engine = Engine(feed, ScenarioStrategy(), config)

    # Add risk rules
    rules = []
    for rule_def in scenario_config.risk_rules:
        rule_type = rule_def["type"]
        pct = rule_def["pct"]
        if rule_type == "StopLoss":
            rules.append(StopLoss(pct=pct))
        elif rule_type == "TakeProfit":
            rules.append(TakeProfit(pct=pct))
        elif rule_type == "TrailingStop":
            rules.append(TrailingStop(pct=pct))
    if rules:
        from ml4t.backtest.risk.position.composite import RuleChain

        engine.broker.set_position_rules(RuleChain(rules))

    return engine.run(), config


# Parameterize over all 16 scenarios
_SCENARIO_PARAMS = list(SCENARIOS.items()) if _HAS_VALIDATION else []


@pytest.mark.parametrize(
    "scenario_id,scenario_config",
    _SCENARIO_PARAMS,
    ids=[f"scenario_{sid}" for sid, _ in _SCENARIO_PARAMS],
)
def test_scenario_runs_without_error(scenario_id, scenario_config):
    """Smoke: each scenario runs and produces trades with positive final value."""
    result, config = _run_ml4t_scenario(scenario_config)

    assert result.equity_curve, f"Scenario {scenario_id} produced no equity curve"
    final_value = result.equity_curve[-1][1]
    assert final_value > 0, f"Scenario {scenario_id} final value={final_value} <= 0"

    # Should produce at least one trade for any meaningful scenario
    closed = [t for t in result.trades if t.status == "closed"]
    assert len(closed) >= 1, f"Scenario {scenario_id} produced no closed trades"


@pytest.mark.parametrize(
    "scenario_id,scenario_config",
    _SCENARIO_PARAMS,
    ids=[f"scenario_{sid}" for sid, _ in _SCENARIO_PARAMS],
)
@pytest.mark.no_invariant_check  # We check invariants explicitly here
def test_scenario_invariants(scenario_id, scenario_config):
    """Each scenario passes all accounting invariants."""
    result, config = _run_ml4t_scenario(scenario_config)

    initial_cash = config.initial_cash
    assert_result_invariants(result, initial_cash)
