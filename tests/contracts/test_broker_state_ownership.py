from __future__ import annotations

import ast
from pathlib import Path

from ml4t.backtest import Broker

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "ml4t" / "backtest"
FORBIDDEN_BROKER_STATE = {
    "_asset_bars_seen",
    "_bar_index",
    "_current_ask_sizes",
    "_current_asks",
    "_current_bid_sizes",
    "_current_bids",
    "_current_closes",
    "_current_highs",
    "_current_lows",
    "_current_mids",
    "_current_opens",
    "_current_prices",
    "_current_signals",
    "_current_time",
    "_current_volumes",
    "_fill_engine",
    "_fill_executor",
    "_filled_this_bar",
    "_last_prices",
    "_order_counter",
    "_orders_this_bar",
    "_orders_this_bar_ids",
    "_partial_orders",
    "_pending_exits",
    "_position_rules",
    "_position_rules_by_asset",
    "_positions_created_this_bar",
    "_stop_exits_this_bar",
    "_submitting_before_risk",
}
TRANSITIONAL_ENGINE_LIFECYCLE_STATE = {"_submitting_before_risk"}
STRATEGY_CALLBACKS = {"on_before_risk", "on_data", "on_end", "on_prepare", "on_start"}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _references_broker(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "broker"
        or (isinstance(node, ast.Attribute) and node.attr == "broker")
    )


def _references_strategy(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name)
        and child.id == "strategy"
        or isinstance(child, ast.Attribute)
        and child.attr == "strategy"
        for child in ast.walk(node)
    )


def _strategy_callback_access(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and node.attr in STRATEGY_CALLBACKS:
        return node.attr if _references_strategy(node.value) else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and _references_strategy(node.args[0])
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in STRATEGY_CALLBACKS
    ):
        return str(node.args[1].value)
    return None


def test_account_state_is_the_only_position_ledger() -> None:
    broker = Broker()

    assert broker.positions is broker.account.positions


def test_mutable_domain_state_delegates_to_its_owner() -> None:
    broker = Broker()
    assert broker.orders is broker._order_state.orders
    assert broker.pending_orders is broker._order_state.pending
    assert broker.fills is broker._execution_journal.fills
    assert broker.trades is broker._execution_journal.trades
    assert broker._current_prices is broker._market_state.prices
    assert broker._pending_exits is broker._risk_state.pending_exits

    replacement: list = []
    broker.fills = replacement
    assert broker.fills is broker._execution_journal.fills is replacement

    replacement_positions = {}
    broker.positions = replacement_positions
    assert broker.positions is broker.account.positions is replacement_positions

    replacement_pending = []
    broker._order_state.pending = replacement_pending
    assert broker.pending_orders is replacement_pending

    replacement_trades = []
    broker._execution_journal.trades = replacement_trades
    assert broker.trades is replacement_trades


def test_collaborators_do_not_reach_through_broker_private_state() -> None:
    violations: list[str] = []
    broker_path = SOURCE_ROOT / "broker.py"
    engine_path = SOURCE_ROOT / "engine.py"

    for path in SOURCE_ROOT.rglob("*.py"):
        if path == broker_path:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in FORBIDDEN_BROKER_STATE
                and _references_broker(node.value)
                and not (path == engine_path and node.attr in TRANSITIONAL_ENGINE_LIFECYCLE_STATE)
            ):
                violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: {node.attr}")

    assert not violations, "Broker state bypasses:\n" + "\n".join(violations)


def test_engine_is_the_only_strategy_callback_sequencer() -> None:
    engine_path = SOURCE_ROOT / "engine.py"
    engine_calls: set[str] = set()
    violations: list[str] = []

    for path in SOURCE_ROOT.rglob("*.py"):
        tree = _parse(path)
        for node in ast.walk(tree):
            callback = _strategy_callback_access(node)
            if callback is None:
                continue
            if path == engine_path:
                engine_calls.add(callback)
            else:
                violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: {callback}")

    assert engine_calls == STRATEGY_CALLBACKS, (
        f"Engine callbacks differ: missing={STRATEGY_CALLBACKS - engine_calls}, "
        f"extra={engine_calls - STRATEGY_CALLBACKS}"
    )
    assert not violations, "Callback sequencing bypasses:\n" + "\n".join(violations)
