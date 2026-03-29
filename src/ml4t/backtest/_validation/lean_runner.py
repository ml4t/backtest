"""Shared LEAN orchestration helpers for internal validation workflows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def parse_lean_int(value: object) -> int:
    """Parse an integer-like value from LEAN summary text."""
    if value is None:
        return 0
    text = str(value).replace(",", "").strip()
    if not text:
        return 0
    token = text.split()[0]
    try:
        return int(float(token))
    except ValueError:
        return 0


def parse_lean_float(value: object) -> float:
    """Parse a float-like value from LEAN summary text."""
    if value is None:
        return 0.0
    text = str(value).replace(",", "").replace("$", "").replace("%", "").strip()
    if not text:
        return 0.0
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        return 0.0


def encode_sequential_ticker(idx: int) -> str:
    """Create a deterministic short ticker namespace from an integer index."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value = idx
    chars: list[str] = []
    for _ in range(4):
        chars.append(letters[value % 26])
        value //= 26
    return "".join(reversed(chars))


def encode_hashed_ticker(project_slug: str, asset_name: str, attempt: int = 0) -> str:
    """Create a project-scoped hashed ticker namespace."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digest = hashlib.sha1(f"{project_slug}|{asset_name}|{attempt}".encode()).digest()
    value = int.from_bytes(digest[:8], "big")
    chars: list[str] = []
    for _ in range(6):
        chars.append(letters[value % 26])
        value //= 26
    return "".join(reversed(chars))


def build_sequential_ticker_map(asset_names: list[str]) -> dict[str, str]:
    """Map asset names to stable short tickers in sorted asset order."""
    return {asset_name: encode_sequential_ticker(i) for i, asset_name in enumerate(asset_names)}


def build_hashed_ticker_map(project_slug: str, asset_names: list[str]) -> dict[str, str]:
    """Map asset names to unique project-scoped hashed tickers."""
    asset_to_ticker: dict[str, str] = {}
    used: set[str] = set()
    for asset_name in asset_names:
        attempt = 0
        while True:
            ticker = encode_hashed_ticker(project_slug, asset_name, attempt)
            if ticker not in used:
                asset_to_ticker[asset_name] = ticker
                used.add(ticker)
                break
            attempt += 1
    return asset_to_ticker


def read_lean_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    """Read a LEAN CSV artifact if it exists and is non-empty."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path, parse_dates=parse_dates or [], low_memory=False)
    return df if not df.empty else None


def load_lean_symbol_map(output_dir: Path) -> dict[str, str]:
    """Load a decoded LEAN symbol map from an output directory."""
    symbol_map_path = output_dir / "ml4t_symbol_map.json"
    if not symbol_map_path.exists():
        return {}
    try:
        data = json.loads(symbol_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(symbol): str(asset) for symbol, asset in data.items()}


def load_lean_artifacts(
    output_dir: Path,
) -> tuple[int, float, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load standard LEAN summary, equity, and order-event artifacts."""
    summary_files = sorted(output_dir.glob("*-summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"LEAN summary file not found in {output_dir}")

    summary = json.loads(summary_files[-1].read_text(encoding="utf-8"))
    trade_stats = summary.get("totalPerformance", {}).get("tradeStatistics", {})
    stats = summary.get("statistics", {})
    portfolio_stats = summary.get("totalPerformance", {}).get("portfolioStatistics", {})
    state = summary.get("state", {})

    num_trades = parse_lean_int(trade_stats.get("totalNumberOfTrades"))
    if num_trades == 0:
        num_trades = parse_lean_int(state.get("OrderCount"))
    if num_trades == 0:
        num_trades = parse_lean_int(stats.get("Total Orders"))

    final_value = parse_lean_float(portfolio_stats.get("endEquity"))
    if final_value == 0.0:
        final_value = parse_lean_float(stats.get("End Equity"))

    symbol_map = load_lean_symbol_map(output_dir)
    order_events_df = read_lean_csv(output_dir / "ml4t_order_events.csv", parse_dates=["timestamp"])
    equity_df = read_lean_csv(output_dir / "ml4t_daily_equity.csv", parse_dates=["timestamp"])

    if final_value == 0.0 and equity_df is not None and "equity" in equity_df.columns:
        final_value = float(equity_df.iloc[-1]["equity"])

    trades_df = None
    if order_events_df is not None:
        order_events_df = order_events_df.sort_values("timestamp").reset_index(drop=True)
        if "symbol" in order_events_df.columns:
            symbols = order_events_df["symbol"].astype(str)
            order_events_df["asset"] = symbols.map(symbol_map).fillna(symbols)
        if "status" in order_events_df.columns:
            status_series = order_events_df["status"].astype(str).str.lower()
            fill_mask = status_series.isin({"filled", "partiallyfilled", "partially_filled"})
        else:
            fill_mask = pd.Series(True, index=order_events_df.index)

        fills_df = order_events_df.loc[fill_mask].copy()
        if not fills_df.empty:
            if "fill_quantity" in fills_df.columns:
                fills_df["quantity"] = fills_df["fill_quantity"].abs()
            if "direction" in fills_df.columns:
                fills_df["side"] = fills_df["direction"].astype(str).str.lower()
            if "asset" in fills_df.columns:
                fills_df["asset"] = fills_df["asset"].astype(str)
            elif "symbol" in fills_df.columns:
                fills_df["asset"] = fills_df["symbol"].astype(str)

            keep_cols = [
                "timestamp",
                "asset",
                "side",
                "quantity",
                "fill_price",
                "fee",
                "status",
                "order_id",
                "message",
            ]
            available_cols = [col for col in keep_cols if col in fills_df.columns]
            trades_df = fills_df[available_cols].reset_index(drop=True)
            if num_trades == 0:
                num_trades = int(len(trades_df))

    return num_trades, final_value, trades_df, equity_df, order_events_df


def resolve_lean_command() -> list[str]:
    """Resolve the local LEAN CLI command."""
    lean_binary = shutil.which("lean")
    if lean_binary is not None:
        return [lean_binary]

    uvx_binary = shutil.which("uvx")
    if uvx_binary is None:
        raise FileNotFoundError("Neither 'lean' nor 'uvx' executable found.")
    return [uvx_binary, "--python", "3.12", "--with", "setuptools<81", "lean"]


def make_lean_env() -> dict[str, str]:
    """Build the subprocess environment for local LEAN runs."""
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.setdefault("UV_TOOL_DIR", "/tmp/uv-tools")
    return env


def check_lean_cli(lean_cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    """Return the local LEAN version string or raise on failure."""
    result = subprocess.run(
        lean_cmd + ["--version"],
        cwd=str(cwd),
        env=env or make_lean_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"LEAN CLI unavailable: {error_text}")
    return (result.stdout or result.stderr).strip()


def export_lean_daily_data(
    *,
    data_root: Path,
    prices_by_asset: Mapping[str, pd.DataFrame],
    asset_to_ticker: Mapping[str, str],
    manifest_path: Path,
    signature_payload: dict[str, object],
) -> bool:
    """Export daily OHLCV data to LEAN zip format with manifest caching."""
    (data_root / "map_files").mkdir(parents=True, exist_ok=True)
    (data_root / "factor_files").mkdir(parents=True, exist_ok=True)
    (data_root / "daily").mkdir(parents=True, exist_ok=True)

    signature = hashlib.md5(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    cache_hit = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        cache_hit = manifest.get("signature") == signature
        if cache_hit:
            expected_files = [
                data_root / "daily" / f"{ticker.lower()}.zip" for ticker in asset_to_ticker.values()
            ]
            cache_hit = all(path.exists() for path in expected_files)

    if cache_hit:
        return True

    for asset_name, asset_df in prices_by_asset.items():
        ticker = asset_to_ticker[asset_name]
        ticker_lower = ticker.lower()
        asset_df = asset_df.sort_index()
        if asset_df.empty:
            continue

        lines: list[str] = []
        for ts, row in asset_df.iterrows():
            dt = pd.Timestamp(ts)
            dt = dt.tz_convert(None) if dt.tz is not None else dt
            open_px = int(round(float(row["open"]) * 10000.0))
            high_px = int(round(float(row["high"]) * 10000.0))
            low_px = int(round(float(row["low"]) * 10000.0))
            close_px = int(round(float(row["close"]) * 10000.0))
            volume = max(1, int(round(float(row["volume"]))))
            lines.append(
                f"{dt.strftime('%Y%m%d')} 00:00,"
                f"{open_px},{max(open_px, high_px, low_px, close_px)},"
                f"{min(open_px, high_px, low_px, close_px)},{close_px},{volume}"
            )

        with zipfile.ZipFile(
            data_root / "daily" / f"{ticker_lower}.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zf:
            zf.writestr(f"{ticker_lower}.csv", "\n".join(lines))

        first_dt = pd.Timestamp(asset_df.index[0])
        first_dt = first_dt.tz_convert(None) if first_dt.tz is not None else first_dt
        first_key = first_dt.strftime("%Y%m%d")
        (data_root / "map_files" / f"{ticker_lower}.csv").write_text(
            f"{first_key},{ticker_lower}\n20501231,{ticker_lower}\n",
            encoding="utf-8",
        )
        (data_root / "factor_files" / f"{ticker_lower}.csv").write_text(
            f"{first_key},1,1,1\n20501231,1,1,0\n",
            encoding="utf-8",
        )

    manifest_path.write_text(
        json.dumps({"signature": signature, "payload": signature_payload}, indent=2),
        encoding="utf-8",
    )
    return False


def run_lean_backtest(
    *,
    lean_cmd: list[str],
    cwd: Path,
    project_dir: Path,
    lean_config: Path,
    output_dir: Path,
    timeout: int = 1800,
    env: dict[str, str] | None = None,
) -> float:
    """Run a LEAN backtest and return the runtime in seconds."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_cmd = lean_cmd + [
        "backtest",
        str(project_dir),
        "--lean-config",
        str(lean_config),
        "--no-update",
        "--output",
        str(output_dir),
    ]
    start_time = time.perf_counter()
    result = subprocess.run(
        run_cmd,
        cwd=str(cwd),
        env=env or make_lean_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    runtime_sec = time.perf_counter() - start_time
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"LEAN backtest failed: {error_text}")
    return runtime_sec


def copy_lean_artifacts(project_dir: Path, output_dir: Path, artifact_names: list[str]) -> None:
    """Copy locally-emitted LEAN artifacts from project to output directory."""
    for artifact_name in artifact_names:
        src = project_dir / artifact_name
        dst = output_dir / artifact_name
        if src.exists():
            shutil.copy2(src, dst)
