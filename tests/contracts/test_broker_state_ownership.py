from __future__ import annotations

import ast
from pathlib import Path

from ml4t.backtest import Broker

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "ml4t" / "backtest"
BROKER_MUTABLE_COLLECTIONS = {"fills", "orders", "pending_orders", "positions", "trades"}
FORBIDDEN_BROKER_STATE = {
    "_asset_stats",
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
    "_contract_specs",
    "_execution_engine",
    "_execution_journal",
    "_fill_engine",
    "_fill_executor",
    "_filled_this_bar",
    "_last_prices",
    "_last_session_id",
    "_market_state",
    "_order_book",
    "_order_counter",
    "_order_state",
    "_orders_this_bar",
    "_orders_this_bar_ids",
    "_partial_orders",
    "_pending_exits",
    "_portfolio_ledger",
    "_position_rules",
    "_position_rules_by_asset",
    "_positions_created_this_bar",
    "_rebalance_counter",
    "_risk_engine",
    "_risk_state",
    "_session_config",
    "_stats_config",
    "_stop_exits_this_bar",
    "_submitting_before_risk",
}
TRANSITIONAL_ENGINE_LIFECYCLE_STATE = {"_submitting_before_risk"}
STRATEGY_CALLBACKS = {"on_before_risk", "on_data", "on_end", "on_prepare", "on_start"}
COLLECTION_MUTATORS = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "popitem",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _references_broker(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "broker"
        or isinstance(node, ast.Attribute)
        and (node.attr == "broker" or _references_broker(node.value))
    )


def _references_strategy(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name)
        and child.id == "strategy"
        or isinstance(child, ast.Attribute)
        and child.attr == "strategy"
        for child in ast.walk(node)
    )


def _private_state_names(broker_class: ast.ClassDef) -> set[str]:
    assigned = {
        node.attr
        for node in ast.walk(broker_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr.startswith("_")
    }
    properties = {
        node.name
        for node in broker_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_")
        and any(
            isinstance(decorator, ast.Name)
            and decorator.id.endswith("property")
            or isinstance(decorator, ast.Attribute)
            and (
                decorator.attr.endswith("property")
                or decorator.attr in {"getter", "setter", "deleter"}
            )
            for decorator in node.decorator_list
        )
    }
    annotated = {
        node.target.id
        for node in broker_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("_")
    }
    dynamic_assignments = {
        node.args[1].value
        for node in ast.walk(broker_class)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "self"
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
        and node.args[1].value.startswith("_")
    }
    return assigned | properties | annotated | dynamic_assignments


def _broker_private_assignments() -> set[str]:
    broker_tree = _parse(SOURCE_ROOT / "broker.py")
    broker_class = next(
        node
        for node in broker_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Broker"
    )
    return _private_state_names(broker_class)


def _broker_collection_access(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and node.attr in BROKER_MUTABLE_COLLECTIONS
        and _references_broker(node.value)
    ):
        return node.attr
    return None


def _mutable_collection_access(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and _broker_collection_access(node) is not None
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ):
        return node.attr
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and (collection := _broker_collection_access(node.value)) is not None
    ):
        return collection
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in COLLECTION_MUTATORS
        and (collection := _broker_collection_access(node.func.value)) is not None
    ):
        return collection
    if (
        isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        and node.value is not None
        and (collection := _broker_collection_access(node.value)) is not None
    ):
        return collection
    return None


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


def test_new_broker_private_state_requires_an_explicit_boundary_decision() -> None:
    discovered = _broker_private_assignments()

    assert discovered == FORBIDDEN_BROKER_STATE, (
        f"Broker private state differs: missing decisions="
        f"{sorted(discovered - FORBIDDEN_BROKER_STATE)}, "
        f"stale decisions={sorted(FORBIDDEN_BROKER_STATE - discovered)}"
    )


def test_private_state_discovery_covers_property_variants_and_setattr() -> None:
    tree = ast.parse(
        "class Broker:\n"
        "    @cached_property\n"
        "    def _cached(self): ...\n"
        "    @_cached.getter\n"
        "    def _getter(self): ...\n"
        "    def configure(self):\n"
        "        setattr(self, '_dynamic', {})\n"
    )
    broker_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    assert _private_state_names(broker_class) == {"_cached", "_dynamic", "_getter"}


def test_mutable_collection_contract_distinguishes_reads_from_mutations() -> None:
    tree = ast.parse(
        "broker.account.positions['AAPL'] = position\n"
        "broker.fills.append(fill)\n"
        "orders = broker.orders\n"
        "order_count = len(broker.orders)\n"
    )

    assert {
        access
        for node in ast.walk(tree)
        if (access := _mutable_collection_access(node)) is not None
    } == {"positions", "fills", "orders"}


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
            mutable_collection = _mutable_collection_access(node)
            if mutable_collection is not None:
                violations.append(
                    f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}: {mutable_collection}"
                )

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
