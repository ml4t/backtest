"""All 16 validation scenarios as declarative configs.

Each scenario is a ScenarioConfig that fully describes:
- What data to generate
- What signals to use
- What risk rules to apply
- How to configure ml4t for each framework
- What tolerances to apply for comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.types import ScenarioConfig, Tolerance

# ============================================================================
# Framework-specific default tolerances
# ============================================================================

VBT_TOLERANCE = Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0)
BT_TOLERANCE = Tolerance(trade_count=0, value_pct=0.1, pnl_abs=10.0)
ZIPLINE_TOLERANCE = Tolerance(trade_count=0, value_pct=0.01, pnl_abs=5.0)

ALL_FRAMEWORKS = ["vectorbt_pro", "vectorbt_oss", "backtrader", "zipline"]
NO_ZIPLINE = ["vectorbt_pro", "vectorbt_oss", "backtrader"]


def _fw_tolerances() -> dict[str, Tolerance]:
    return {
        "vectorbt_pro": VBT_TOLERANCE,
        "vectorbt_oss": VBT_TOLERANCE,
        "backtrader": BT_TOLERANCE,
        "zipline": ZIPLINE_TOLERANCE,
    }


# ============================================================================
# Scenario 01: Long Only
# ============================================================================

SCENARIO_01 = ScenarioConfig(
    id="01",
    name="Long Only",
    description="Simple long-only strategy with predefined entry/exit signals.",
    data_generator="generate_random_walk",
    data_kwargs={"n_bars": 100, "seed": 42},
    signal_columns=["entry", "exit"],
    strategy_type="long_signal",
    tolerances=_fw_tolerances(),
)

# ============================================================================
# Scenario 02: Long/Short
# ============================================================================

SCENARIO_02 = ScenarioConfig(
    id="02",
    name="Long/Short",
    description="Long and short positions with direction switching.",
    data_generator="generate_random_walk",
    data_kwargs={"n_bars": 100, "seed": 42},
    signal_columns=["long_entry", "long_exit", "short_entry", "short_exit"],
    strategy_type="long_short",
    ml4t_config={
        "allow_short_selling": True,
        "allow_leverage": True,
    },
    tolerances=_fw_tolerances(),
)

# ============================================================================
# Scenario 03: Stop Loss (5%)
# ============================================================================

SCENARIO_03 = ScenarioConfig(
    id="03",
    name="Stop Loss",
    description="5% stop-loss on a declining price path.",
    data_generator="generate_stop_loss_data",
    data_kwargs={"seed": 42},
    signal_columns=["entry"],
    strategy_type="single_entry",
    risk_rules=[{"type": "StopLoss", "pct": 0.05}],
    constants={"sl_pct": 0.05},
    ml4t_overrides={
        "backtrader": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "SIGNAL_PRICE",
        },
        "vectorbt_pro": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "FILL_PRICE",
        },
        "vectorbt_oss": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "FILL_PRICE",
        },
        "zipline": {
            "stop_fill_mode": "NEXT_BAR_OPEN",
            "stop_level_basis": "FILL_PRICE",
        },
    },
    extra_checks=["exit_price"],
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=0.01),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=0.01),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "zipline": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=5.0),
    },
)

# ============================================================================
# Scenario 04: Take Profit (10%)
# ============================================================================

SCENARIO_04 = ScenarioConfig(
    id="04",
    name="Take Profit",
    description="10% take-profit on a rising price path.",
    data_generator="generate_take_profit_data",
    data_kwargs={"seed": 42},
    signal_columns=["entry"],
    strategy_type="single_entry",
    risk_rules=[{"type": "TakeProfit", "pct": 0.10}],
    constants={"tp_pct": 0.10},
    ml4t_overrides={
        "backtrader": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "SIGNAL_PRICE",
        },
        "vectorbt_pro": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "FILL_PRICE",
        },
        "vectorbt_oss": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "FILL_PRICE",
        },
        "zipline": {
            "stop_fill_mode": "NEXT_BAR_OPEN",
            "stop_level_basis": "FILL_PRICE",
        },
    },
    extra_checks=["exit_price"],
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=0.01),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=0.01),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "zipline": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=5.0),
    },
)

# ============================================================================
# Scenario 05: Percentage Commission (0.1%)
# ============================================================================

SCENARIO_05 = ScenarioConfig(
    id="05",
    name="Commission (Pct)",
    description="0.1% percentage-based commission.",
    data_generator="generate_random_walk",
    data_kwargs={"n_bars": 100, "seed": 42},
    signal_columns=["entry", "exit"],
    strategy_type="long_signal",
    constants={"commission_rate": 0.001},
    extra_checks=["commission"],
    tolerances=_fw_tolerances(),
)

# ============================================================================
# Scenario 06: Per-Share Commission ($0.005)
# ============================================================================

SCENARIO_06 = ScenarioConfig(
    id="06",
    name="Commission (Per-Share)",
    description="$0.005 per-share commission.",
    data_generator="generate_random_walk",
    data_kwargs={"n_bars": 100, "seed": 42},
    signal_columns=["entry", "exit"],
    strategy_type="long_signal",
    constants={"per_share_rate": 0.005},
    extra_checks=["commission"],
    # VBT OSS only supports percentage fees, not per-share
    supported_frameworks=["vectorbt_pro", "backtrader", "zipline"],
    tolerances=_fw_tolerances(),
)

# ============================================================================
# Scenario 07: Fixed Slippage ($0.01)
# ============================================================================

SCENARIO_07 = ScenarioConfig(
    id="07",
    name="Slippage (Fixed)",
    description="$0.01 fixed slippage per share.",
    data_generator="generate_random_walk",
    data_kwargs={"n_bars": 100, "seed": 42},
    signal_columns=["entry", "exit"],
    strategy_type="long_signal",
    constants={"slippage_fixed": 0.01},
    tolerances=_fw_tolerances(),
)

# ============================================================================
# Scenario 08: Percentage Slippage (0.1%)
# ============================================================================

SCENARIO_08 = ScenarioConfig(
    id="08",
    name="Slippage (Pct)",
    description="0.1% percentage-based slippage.",
    data_generator="generate_random_walk",
    data_kwargs={"n_bars": 100, "seed": 42},
    signal_columns=["entry", "exit"],
    strategy_type="long_signal",
    constants={"slippage_rate": 0.001},
    tolerances=_fw_tolerances(),
)

# ============================================================================
# Scenario 09: Trailing Stop (5%) - Long
# ============================================================================

SCENARIO_09 = ScenarioConfig(
    id="09",
    name="Trailing Stop",
    description="5% trailing stop on long positions with trending data.",
    data_generator="generate_trending_data",
    data_kwargs={"n_bars": 100, "seed": 42, "entry_bars": [0, 40]},
    signal_columns=["entry"],
    strategy_type="risk_entry_only",
    risk_rules=[{"type": "TrailingStop", "pct": 0.05}],
    constants={"trail_pct": 0.05},
    ml4t_overrides={
        "vectorbt_oss": {
            "stop_fill_mode": "STOP_PRICE",
            "trail_hwm_source": "BAR_EXTREME",
        },
        "vectorbt_pro": {
            "stop_fill_mode": "STOP_PRICE",
        },
        "backtrader": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "SIGNAL_PRICE",
        },
        "zipline": {
            "stop_fill_mode": "NEXT_BAR_OPEN",
            "trail_hwm_source": "BAR_EXTREME",
            "trail_stop_timing": "INTRABAR",
        },
    },
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.5, pnl_abs=50.0),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "zipline": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=5.0),
    },
)

# ============================================================================
# Scenario 10: Bracket Order (SL 5% + TP 10%)
# ============================================================================

SCENARIO_10 = ScenarioConfig(
    id="10",
    name="Bracket Order",
    description="OCO bracket order with 5% stop-loss and 10% take-profit.",
    data_generator="generate_bracket_data",
    data_kwargs={"n_bars": 100, "seed": 42, "entry_bars": [0, 20, 55]},
    signal_columns=["entry"],
    strategy_type="risk_entry_only",
    risk_rules=[
        {"type": "StopLoss", "pct": 0.05},
        {"type": "TakeProfit", "pct": 0.10},
    ],
    constants={"sl_pct": 0.05, "tp_pct": 0.10},
    supported_frameworks=NO_ZIPLINE,
    ml4t_overrides={
        "vectorbt_oss": {
            "stop_fill_mode": "STOP_PRICE",
        },
        "vectorbt_pro": {
            "stop_fill_mode": "STOP_PRICE",
        },
        "backtrader": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "SIGNAL_PRICE",
        },
    },
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.5, pnl_abs=50.0),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
    },
)

# ============================================================================
# Scenario 11: Short Only
# ============================================================================

SCENARIO_11 = ScenarioConfig(
    id="11",
    name="Short Only",
    description="Short-only strategy with predefined entry/exit signals.",
    data_generator="generate_short_signals",
    data_kwargs={"n_bars": 200, "seed": 42},
    signal_columns=["short_entry", "short_exit"],
    strategy_type="short_only",
    initial_cash=1_000_000.0,
    ml4t_config={
        "allow_short_selling": True,
        "allow_leverage": True,
    },
    tolerances=_fw_tolerances(),
)

# ============================================================================
# Scenario 12: Short + Trailing Stop (5%)
# ============================================================================

SCENARIO_12 = ScenarioConfig(
    id="12",
    name="Short Trailing Stop",
    description="5% trailing stop on short positions.",
    data_generator="generate_short_trending_data",
    data_kwargs={"n_bars": 100, "seed": 42, "entry_bars": [0, 40]},
    signal_columns=["entry"],
    strategy_type="risk_entry_only",
    risk_rules=[{"type": "TrailingStop", "pct": 0.05}],
    constants={"trail_pct": 0.05},
    initial_cash=1_000_000.0,
    ml4t_config={
        "allow_short_selling": True,
        "allow_leverage": True,
    },
    ml4t_overrides={
        "vectorbt_oss": {
            "stop_fill_mode": "STOP_PRICE",
            "trail_hwm_source": "BAR_EXTREME",
        },
        "vectorbt_pro": {
            "stop_fill_mode": "STOP_PRICE",
        },
        "backtrader": {
            "stop_fill_mode": "STOP_PRICE",
            "stop_level_basis": "SIGNAL_PRICE",
        },
        "zipline": {
            "stop_fill_mode": "NEXT_BAR_OPEN",
            "trail_hwm_source": "BAR_EXTREME",
            "trail_stop_timing": "INTRABAR",
        },
    },
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.5, pnl_abs=50.0),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "zipline": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=5.0),
    },
)

# ============================================================================
# Scenario 13: TSL + TP Combo
# ============================================================================

SCENARIO_13 = ScenarioConfig(
    id="13",
    name="TSL + TP Combo",
    description="Trailing stop (5%) + take profit (8%) combination.",
    data_generator="generate_rule_combo_data",
    data_kwargs={"scenario": "tp_first", "seed": 42},
    signal_columns=["entry"],
    strategy_type="single_entry",
    risk_rules=[
        {"type": "TrailingStop", "pct": 0.05},
        {"type": "TakeProfit", "pct": 0.08},
    ],
    constants={"trail_pct": 0.05, "tp_pct": 0.08},
    ml4t_overrides={
        "vectorbt_oss": {"stop_fill_mode": "STOP_PRICE", "trail_hwm_source": "BAR_EXTREME"},
        "vectorbt_pro": {"stop_fill_mode": "STOP_PRICE"},
        "backtrader": {"stop_fill_mode": "STOP_PRICE", "stop_level_basis": "SIGNAL_PRICE"},
        "zipline": {"stop_fill_mode": "NEXT_BAR_OPEN", "trail_hwm_source": "BAR_EXTREME", "trail_stop_timing": "INTRABAR"},
    },
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "zipline": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=5.0),
    },
)

# ============================================================================
# Scenario 14: TSL + SL Combo
# ============================================================================

SCENARIO_14 = ScenarioConfig(
    id="14",
    name="TSL + SL Combo",
    description="Trailing stop (5%) + stop loss (8%) combination.",
    data_generator="generate_rule_combo_data",
    data_kwargs={"scenario": "tsl_first", "seed": 42},
    signal_columns=["entry"],
    strategy_type="single_entry",
    risk_rules=[
        {"type": "TrailingStop", "pct": 0.05},
        {"type": "StopLoss", "pct": 0.08},
    ],
    constants={"trail_pct": 0.05, "sl_pct": 0.08},
    ml4t_overrides={
        "vectorbt_oss": {"stop_fill_mode": "STOP_PRICE", "trail_hwm_source": "BAR_EXTREME"},
        "vectorbt_pro": {"stop_fill_mode": "STOP_PRICE"},
        "backtrader": {"stop_fill_mode": "STOP_PRICE", "stop_level_basis": "SIGNAL_PRICE"},
        "zipline": {"stop_fill_mode": "NEXT_BAR_OPEN", "trail_hwm_source": "BAR_EXTREME", "trail_stop_timing": "INTRABAR"},
    },
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "zipline": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=5.0),
    },
)

# ============================================================================
# Scenario 15: Triple Rule (TSL + TP + SL)
# ============================================================================

SCENARIO_15 = ScenarioConfig(
    id="15",
    name="Triple Rule",
    description="Trailing stop (3%) + take profit (10%) + stop loss (5%).",
    data_generator="generate_rule_combo_data",
    data_kwargs={"scenario": "sl_first", "seed": 42},
    signal_columns=["entry"],
    strategy_type="single_entry",
    risk_rules=[
        {"type": "TrailingStop", "pct": 0.03},
        {"type": "TakeProfit", "pct": 0.10},
        {"type": "StopLoss", "pct": 0.05},
    ],
    constants={"trail_pct": 0.03, "tp_pct": 0.10, "sl_pct": 0.05},
    ml4t_overrides={
        "vectorbt_oss": {"stop_fill_mode": "STOP_PRICE", "trail_hwm_source": "BAR_EXTREME"},
        "vectorbt_pro": {"stop_fill_mode": "STOP_PRICE"},
        "backtrader": {"stop_fill_mode": "STOP_PRICE", "stop_level_basis": "SIGNAL_PRICE"},
        "zipline": {"stop_fill_mode": "NEXT_BAR_OPEN", "trail_hwm_source": "BAR_EXTREME", "trail_stop_timing": "INTRABAR"},
    },
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        # Zipline SL reference from signal close vs ml4t from fill price = ~$9.46 diff
        "zipline": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=10.0),
    },
)

# ============================================================================
# Scenario 16: Stress Test (1500 bars)
# ============================================================================

SCENARIO_16 = ScenarioConfig(
    id="16",
    name="Stress Test",
    description="1500-bar stress test with 9 market regimes.",
    data_generator="generate_stress_data",
    data_kwargs={"n_bars": 1500, "seed": 42},
    signal_columns=["entry"],
    strategy_type="risk_entry_only",
    risk_rules=[{"type": "TrailingStop", "pct": 0.05}],
    constants={"trail_pct": 0.05, "allow_reentry": True},
    ml4t_overrides={
        "vectorbt_oss": {"stop_fill_mode": "STOP_PRICE", "trail_hwm_source": "BAR_EXTREME"},
        "vectorbt_pro": {"stop_fill_mode": "STOP_PRICE"},
        "backtrader": {"stop_fill_mode": "STOP_PRICE", "stop_level_basis": "SIGNAL_PRICE"},
        "zipline": {"stop_fill_mode": "NEXT_BAR_OPEN", "trail_hwm_source": "BAR_EXTREME", "trail_stop_timing": "INTRABAR"},
    },
    tolerances={
        "vectorbt_pro": Tolerance(trade_count=0, value_pct=1.0, pnl_abs=200.0),
        "vectorbt_oss": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "backtrader": Tolerance(trade_count=0, value_pct=0.01, pnl_abs=1.0),
        "zipline": Tolerance(trade_count=0, value_pct=0.1, pnl_abs=50.0),
    },
)

# ============================================================================
# Registry
# ============================================================================

SCENARIOS: dict[str, ScenarioConfig] = {
    "01": SCENARIO_01,
    "02": SCENARIO_02,
    "03": SCENARIO_03,
    "04": SCENARIO_04,
    "05": SCENARIO_05,
    "06": SCENARIO_06,
    "07": SCENARIO_07,
    "08": SCENARIO_08,
    "09": SCENARIO_09,
    "10": SCENARIO_10,
    "11": SCENARIO_11,
    "12": SCENARIO_12,
    "13": SCENARIO_13,
    "14": SCENARIO_14,
    "15": SCENARIO_15,
    "16": SCENARIO_16,
}

SCENARIO_NAMES: dict[str, str] = {s.id: s.name for s in SCENARIOS.values()}
