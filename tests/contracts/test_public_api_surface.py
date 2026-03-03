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
        "BacktestResult",
        "CommissionType",
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
        "AssetClass",
        "ContractSpec",
        "RebalanceConfig",
        "TargetWeightExecutor",
        "StopLoss",
        "TakeProfit",
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
        "LinearImpact",
        "VolumeParticipationLimit",
        "WaterMarkSource",
        "InitialHwmSource",
        "TrailHwmSource",
    }
    assert removed_legacy_exports.isdisjoint(set(bt.__all__))
