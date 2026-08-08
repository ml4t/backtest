from __future__ import annotations

import ast
from pathlib import Path

from ml4t.backtest import Broker
from ml4t.backtest.core.state import STATE_OWNERS

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "ml4t" / "backtest"
COLLABORATORS = (
    SOURCE_ROOT / "core" / "execution_engine.py",
    SOURCE_ROOT / "core" / "fill_engine.py",
    SOURCE_ROOT / "core" / "order_book.py",
    SOURCE_ROOT / "core" / "portfolio_ledger.py",
    SOURCE_ROOT / "core" / "risk_engine.py",
    SOURCE_ROOT / "execution" / "fill_executor.py",
)
LEGACY_BROKER_STATE = {
    "_bar_index",
    "_current_asks",
    "_current_bid_sizes",
    "_current_bids",
    "_current_closes",
    "_current_highs",
    "_current_lows",
    "_current_opens",
    "_current_prices",
    "_current_time",
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
BROKER_MUTABLE_COLLECTIONS = {"fills", "orders", "pending_orders", "positions", "trades"}


def test_account_state_is_the_only_position_ledger() -> None:
    broker = Broker()

    assert broker.positions is broker.account.positions


def test_mutable_domain_state_has_one_declared_owner() -> None:
    assert STATE_OWNERS == {
        "cash": "AccountState",
        "positions": "AccountState",
        "orders": "OrderState",
        "fills": "ExecutionJournal",
        "trades": "ExecutionJournal",
        "market": "MarketState",
        "risk": "RiskState",
        "callback_sequence": "Engine",
    }

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


def test_collaborators_do_not_reach_through_broker_private_state() -> None:
    violations: list[str] = []

    for path in COLLABORATORS:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in LEGACY_BROKER_STATE:
                violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: {node.attr}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr in BROKER_MUTABLE_COLLECTIONS
                and (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "broker"
                    or isinstance(node.value, ast.Attribute)
                    and node.value.attr == "broker"
                )
            ):
                violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: {node.attr}")

    assert not violations, "Broker state bypasses:\n" + "\n".join(violations)
