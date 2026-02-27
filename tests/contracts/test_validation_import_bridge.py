from ml4t.backtest import _validation_imports as bridge


def test_validation_bridge_exports_stable_aliases() -> None:
    assert "TrailHwmSource" in bridge.__all__
    assert bridge.TrailHwmSource is bridge.WaterMarkSource
    assert "TargetWeightExecutor" in bridge.__all__
    assert "VolumeParticipationLimit" in bridge.__all__
