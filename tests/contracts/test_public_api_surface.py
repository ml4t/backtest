from __future__ import annotations

import ml4t.backtest as bt


def test_root_api_contains_only_intended_core_surface() -> None:
    required = {
        "DataFeed",
        "Broker",
        "Strategy",
        "Engine",
        "run_backtest",
        "BacktestConfig",
        "Mode",
        "BacktestResult",
        "OrderType",
        "OrderSide",
        "OrderStatus",
        "ExecutionMode",
        "ExitReason",
        "StopFillMode",
        "StopLevelBasis",
        "Order",
        "Position",
        "Fill",
        "Trade",
        "StopLoss",
        "TrailingStop",
        "RuleChain",
    }
    assert required.issubset(set(bt.__all__))

    removed_legacy_exports = {
        "NoCommission",
        "NoSlippage",
        "PercentageCommission",
        "PercentageSlippage",
        "PerShareCommission",
        "RebalanceConfig",
        "TargetWeightExecutor",
        "LinearImpact",
        "VolumeParticipationLimit",
        "WaterMarkSource",
        "InitialHwmSource",
        "TrailHwmSource",
    }
    assert removed_legacy_exports.isdisjoint(set(bt.__all__))
