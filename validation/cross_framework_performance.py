#!/usr/bin/env python3
"""Generate controlled and idiomatic cross-framework performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
import urllib.parse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from statistics import median
from time import perf_counter, sleep
from typing import Any, cast

VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_DIR))

import benchmark_suite as suite  # noqa: E402
import large_scale_evidence as scale  # noqa: E402
from common.framework_registry import load_framework_manifest  # noqa: E402

SCHEMA_VERSION = 1
MIN_SAMPLES = 10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
WORKER_TIMEOUT_SECONDS = 900
ACCEPTED_PATH = VALIDATION_DIR / "PERFORMANCE_RESULTS.json"
CANDIDATE_PATH = VALIDATION_DIR / "candidates" / "PERFORMANCE_RESULTS.candidate.json"
THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
MEASUREMENT_PROTOCOL = {
    "process_isolation": "one fresh child process per warm-up and measured repetition",
    "warmup": "one unmeasured isolated process per runner before measured repetitions",
    "runtime_primary": (
        "parent perf_counter around the complete child process, including interpreter startup, "
        "imports, deterministic input generation, adapter setup, engine execution, output "
        "extraction, canonical validation, and process teardown"
    ),
    "runtime_secondary": "adapter-reported engine interval and worker stage intervals",
    "memory": "child resource.getrusage(RUSAGE_SELF).ru_maxrss over the complete process",
    "cache_policy": (
        "installed packages and compiled framework caches persist; inputs and framework state are "
        "rebuilt in every child; Zipline uses a new bundle root for every child"
    ),
    "thread_environment": THREAD_ENVIRONMENT,
    "uncertainty": {
        "method": "deterministic percentile bootstrap of the sample median",
        "confidence": BOOTSTRAP_CONFIDENCE,
        "resamples": BOOTSTRAP_RESAMPLES,
    },
}


@dataclass(frozen=True)
class RunnerSpec:
    """One framework side measured in its locked environment."""

    runner_id: str
    framework: str
    side: str
    profile: str


WORKLOAD = scale.ScaleWorkload(
    name="performance_50_assets_252_sessions",
    seed=42,
    bars=252,
    assets=50,
    top_n=5,
    bottom_n=5,
    rebalance_frequency=1,
    end_session="2025-12-31",
)

RUNNERS = tuple(
    runner
    for framework, profile in (
        ("vectorbt_pro", "vectorbt_strict"),
        ("vectorbt_oss", "vectorbt_oss_strict"),
        ("backtrader", "backtrader_strict"),
        ("zipline", "zipline_strict"),
        ("lean", "lean"),
    )
    for runner in (
        RunnerSpec(f"{framework}:external", framework, "external", profile),
        RunnerSpec(f"{framework}:ml4t", framework, "ml4t", profile),
    )
)
RUNNERS_BY_ID = {runner.runner_id: runner for runner in RUNNERS}

IDIOMATIC_RUNNERS = {
    "ml4t_backtest": "vectorbt_oss:ml4t",
    "vectorbt_pro": "vectorbt_pro:external",
    "vectorbt_oss": "vectorbt_oss:external",
    "backtrader": "backtrader:external",
    "zipline": "zipline:external",
    "lean": "lean:external",
}
IDIOMATIC_DISCLOSURES = {
    "ml4t_backtest": (
        "Event-driven Strategy and Engine API with the vectorbt_oss_strict profile; this is one "
        "documented ML4T configuration, not a framework-wide default-performance claim."
    ),
    "vectorbt_pro": (
        "Vectorized Portfolio.from_orders target-amount API with cash sharing, same-bar close "
        "execution, zero costs, and locked short collateral."
    ),
    "vectorbt_oss": (
        "Vectorized Portfolio.from_orders target-amount API with cash sharing, same-bar close "
        "execution, zero costs, and lock_cash enabled."
    ),
    "backtrader": (
        "Native Cerebro and Strategy callbacks with target orders, next-session open fills, "
        "integer shares, zero commission, and enabled submission cash checks."
    ),
    "zipline": (
        "Native run_algorithm callback and order_target API with daily bundle ingestion, a custom "
        "open-price slippage model, and zero commission. Bundle creation is included."
    ),
    "lean": (
        "Native QCAlgorithm daily callback and market-order API in the frozen LEAN image with the "
        "declared margin account, zero fee, and zero slippage models."
    ),
}


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _source_digests() -> dict[str, str]:
    paths = {
        "benchmark_suite": VALIDATION_DIR / "benchmark_suite.py",
        "framework_manifest": VALIDATION_DIR / "framework_targets.toml",
        "performance_runner": Path(__file__),
        "profiles": PROJECT_ROOT / "src" / "ml4t" / "backtest" / "profiles.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def _peak_rss_mb() -> float:
    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _process_tree_rss_bytes(root_pid: int) -> int:
    """Return current RSS for a Linux process and all current descendants."""
    pending = [root_pid]
    observed: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
            rss_line = next(
                (line for line in status.splitlines() if line.startswith("VmRSS:")),
                "VmRSS: 0 kB",
            )
            total += int(rss_line.split()[1]) * 1024
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
            pending.extend(int(child) for child in children.split())
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total


class _DockerConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/docker.sock")


def _docker_json(path: str) -> Any:
    connection = _DockerConnection("localhost", timeout=1.0)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f"Docker API {path} returned {response.status}")
        return json.loads(payload)
    finally:
        connection.close()


def _docker_container_ids(ancestor: str) -> set[str]:
    filters = urllib.parse.quote(json.dumps({"ancestor": [ancestor]}, separators=(",", ":")))
    payload = _docker_json(f"/containers/json?filters={filters}")
    if not isinstance(payload, list):
        raise RuntimeError("Docker container list was not an array")
    return {str(container["Id"]) for container in payload if isinstance(container, dict)}


def _docker_container_pid(container_id: str) -> int | None:
    payload = _docker_json(f"/containers/{container_id}/json")
    if not isinstance(payload, dict) or not isinstance(payload.get("State"), dict):
        return None
    pid = payload["State"].get("Pid")
    return int(pid) if isinstance(pid, int) and pid > 0 else None


def _container_cgroup_memory_path(pid: int) -> Path | None:
    try:
        entries = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    for entry in entries:
        parts = entry.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            path = Path("/sys/fs/cgroup") / parts[2].lstrip("/") / "memory.current"
            return path if path.is_file() else None
    return None


def _execute_with_memory_monitor(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    docker_ancestor: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], float, float, bool]:
    """Execute a worker while sampling its process tree and optional container cgroup."""
    existing_containers = (
        _docker_container_ids(docker_ancestor) if docker_ancestor is not None else set()
    )
    started = perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_bytes = 0
    container_path: Path | None = None
    container_observed = docker_ancestor is None
    next_docker_poll = started
    deadline = started + timeout
    while process.poll() is None:
        if perf_counter() > deadline:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"Performance worker exceeded {timeout:.0f}s: {(stderr or stdout).strip()[-2_000:]}"
            )
        tree_bytes = _process_tree_rss_bytes(process.pid)
        container_bytes = 0
        if (
            docker_ancestor is not None
            and container_path is None
            and perf_counter() >= next_docker_poll
        ):
            next_docker_poll = perf_counter() + 0.1
            current = _docker_container_ids(docker_ancestor) - existing_containers
            if current:
                container_id = sorted(current)[0]
                container_pid = _docker_container_pid(container_id)
                if container_pid is not None:
                    container_path = _container_cgroup_memory_path(container_pid)
                    container_observed = container_path is not None
        if container_path is not None:
            try:
                container_bytes = int(container_path.read_text(encoding="utf-8").strip())
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                container_path = None
        peak_bytes = max(peak_bytes, tree_bytes + container_bytes)
        sleep(0.02)
    peak_bytes = max(peak_bytes, _process_tree_rss_bytes(process.pid))
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return completed, perf_counter() - started, peak_bytes / (1024 * 1024), container_observed


def _run_external(
    framework: str,
    config: suite.BenchmarkConfig,
    price_data: dict[str, Any],
    signals: Any,
    dates: Any,
) -> suite.BenchmarkResult:
    runners = {
        "vectorbt_pro": suite.benchmark_vectorbt_pro,
        "vectorbt_oss": suite.benchmark_vectorbt_oss,
        "backtrader": suite.benchmark_backtrader,
        "zipline": suite.benchmark_zipline,
        "lean": suite.benchmark_lean,
    }
    return runners[framework](config, price_data, signals, dates)


def _run_ml4t(
    runner: RunnerSpec,
    config: suite.BenchmarkConfig,
    price_data: dict[str, Any],
    signals: Any,
    dates: Any,
) -> suite.BenchmarkResult:
    execution_mode = "same_bar" if runner.framework.startswith("vectorbt") else "next_bar"
    return suite.benchmark_ml4t(
        config,
        price_data,
        signals,
        dates,
        execution_mode=execution_mode,
        profile_override=runner.profile,
    )


def _surface_signature(result: suite.BenchmarkResult, effective_prices: dict[str, Any]) -> dict:
    result.trades_df = suite.closed_trades_from_fills(result.fills_df)
    trade_records = suite.canonical_trade_records(result.trades_df, timestamp_domain="session_date")
    result.num_trades = len(trade_records) if trade_records is not None else 0
    intents = suite.canonical_target_records(result.target_trace_df)
    fills = suite.canonical_fill_records(result.fills_df, timestamp_domain="session_date")
    trades = trade_records
    if intents is None or fills is None or trades is None:
        raise RuntimeError("Performance runner did not expose every controlled output surface")
    terminal = scale._terminal_state(result, effective_prices, WORKLOAD.initial_cash)
    return {
        "order_intents": {"count": len(intents), "sha256": suite._records_hash(intents)},
        "fills": {"count": len(fills), "sha256": suite._records_hash(fills)},
        "trades": {"count": len(trades), "sha256": suite._records_hash(trades)},
        "trade_count": len(trades),
        "total_pnl": scale._canonical_money(result.final_value - WORKLOAD.initial_cash),
        "final_value": scale._canonical_money(result.final_value),
        "terminal_state_sha256": terminal["sha256"],
    }


def run_worker(runner: RunnerSpec) -> dict[str, Any]:
    """Run one measured framework side and return its validated output signature."""
    started = perf_counter()
    manifest = load_framework_manifest()
    target = manifest.targets[runner.framework]
    actual_target = scale._actual_target(target)
    config = WORKLOAD.benchmark_config()

    input_started = perf_counter()
    price_data, signals, dates = suite.generate_benchmark_data(config, seed=WORKLOAD.seed)
    raw_input_sha256 = scale.input_digest(price_data, signals, dates)
    effective_prices = scale._effective_prices(runner.framework, price_data)
    effective_input_sha256 = scale.input_digest(effective_prices, signals, dates)
    input_seconds = perf_counter() - input_started

    framework_started = perf_counter()
    output_log = StringIO()
    with redirect_stdout(output_log):
        if runner.framework == "zipline":
            with tempfile.TemporaryDirectory(prefix="ml4t-performance-zipline-") as root:
                previous_root = os.environ.get("ZIPLINE_ROOT")
                os.environ["ZIPLINE_ROOT"] = root
                try:
                    result = (
                        _run_external(runner.framework, config, price_data, signals, dates)
                        if runner.side == "external"
                        else _run_ml4t(runner, config, price_data, signals, dates)
                    )
                finally:
                    if previous_root is None:
                        os.environ.pop("ZIPLINE_ROOT", None)
                    else:
                        os.environ["ZIPLINE_ROOT"] = previous_root
        else:
            result = (
                _run_external(runner.framework, config, price_data, signals, dates)
                if runner.side == "external"
                else _run_ml4t(runner, config, price_data, signals, dates)
            )
    framework_call_seconds = perf_counter() - framework_started
    if result.error is not None:
        raise RuntimeError(f"{runner.runner_id} failed: {result.error}")

    validation_started = perf_counter()
    signature = _surface_signature(result, effective_prices)
    validation_seconds = perf_counter() - validation_started
    return {
        "runner_id": runner.runner_id,
        "framework": runner.framework,
        "side": runner.side,
        "profile": runner.profile,
        "target": actual_target,
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
        },
        "ml4t": _git_identity(),
        "input": {
            "raw_sha256": raw_input_sha256,
            "effective_sha256": effective_input_sha256,
        },
        "stages_seconds": {
            "input_generation": input_seconds,
            "framework_call": framework_call_seconds,
            "adapter_reported_engine": result.runtime_sec,
            "output_validation": validation_seconds,
            "worker_total": perf_counter() - started,
        },
        "process_peak_rss_mb": _peak_rss_mb(),
        "thread_environment": {key: os.environ.get(key) for key in sorted(THREAD_ENVIRONMENT)},
        "output": signature,
        "output_sha256": _json_digest(signature),
        "captured_log_tail": output_log.getvalue()[-2_000:],
    }


def run_calibration_worker() -> dict[str, Any]:
    """Exercise wall-time and peak-RSS instrumentation with known work."""
    started = perf_counter()
    allocation = bytearray(32 * 1024 * 1024)
    allocation[0] = 1
    sleep(0.025)
    return {
        "worker_seconds": perf_counter() - started,
        "process_peak_rss_mb": _peak_rss_mb(),
        "allocated_bytes": len(allocation),
    }


def _resolve_python(runner: RunnerSpec) -> Path:
    target = load_framework_manifest().targets[runner.framework]
    override = os.getenv(target.python_env_var or "")
    if override:
        return Path(override).expanduser().resolve()
    if target.environment is None:
        raise RuntimeError(f"No environment configured for {runner.framework}")
    return PROJECT_ROOT / target.environment / "bin" / "python"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_process(runner: RunnerSpec) -> dict[str, Any]:
    interpreter = _resolve_python(runner)
    if not interpreter.is_file():
        raise RuntimeError(f"Missing interpreter for {runner.runner_id}: {interpreter}")
    with tempfile.TemporaryDirectory(prefix="ml4t-performance-worker-") as directory:
        output = Path(directory) / "sample.json"
        command = [
            str(interpreter),
            str(Path(__file__).resolve()),
            "--worker",
            runner.runner_id,
            "--output",
            str(output),
        ]
        environment = os.environ.copy()
        environment.update(THREAD_ENVIRONMENT)
        paths = [str(PROJECT_ROOT / "src"), str(VALIDATION_DIR)]
        sibling_specs = PROJECT_ROOT.parent / "ml4t-specs" / "src"
        if sibling_specs.is_dir():
            paths.append(str(sibling_specs))
        environment["PYTHONPATH"] = os.pathsep.join(
            paths + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        docker_ancestor = None
        if runner.runner_id == "lean:external":
            docker_ancestor = load_framework_manifest().targets["lean"].artifact
            if docker_ancestor is None:
                raise RuntimeError("LEAN target lacks an immutable engine image")
        completed, process_wall_seconds, monitored_peak_rss_mb, container_observed = (
            _execute_with_memory_monitor(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                timeout=WORKER_TIMEOUT_SECONDS,
                docker_ancestor=docker_ancestor,
            )
        )
        if completed.returncode != 0 or not output.is_file():
            details = "\n".join(
                value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
            )
            raise RuntimeError(
                f"Performance worker {runner.runner_id} exited {completed.returncode}: "
                f"{details[-10_000:]}"
            )
        sample = cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))
        sample["process_wall_seconds"] = process_wall_seconds
        sample["worker_self_peak_rss_mb"] = sample.pop("process_peak_rss_mb")
        sample["process_peak_rss_mb"] = monitored_peak_rss_mb
        sample["lean_container_observed"] = container_observed
        return sample


def run_calibration_process() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ml4t-performance-calibration-") as directory:
        output = Path(directory) / "calibration.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--calibration-worker",
            "--output",
            str(output),
        ]
        started = perf_counter()
        completed = subprocess.run(command, check=False, timeout=60)
        wall_seconds = perf_counter() - started
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError("Performance instrumentation calibration failed")
        result = cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))
        result["process_wall_seconds"] = wall_seconds
        return result


def _percentile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_median_interval(values: list[float], *, seed: int) -> list[float]:
    """Return a deterministic percentile-bootstrap 95 percent median interval."""
    if len(values) < MIN_SAMPLES:
        raise ValueError(f"At least {MIN_SAMPLES} samples are required")
    generator = random.Random(seed)
    estimates = sorted(
        median(generator.choices(values, k=len(values))) for _ in range(BOOTSTRAP_RESAMPLES)
    )
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return [_percentile(estimates, tail), _percentile(estimates, 1.0 - tail)]


def _summary(samples: list[dict[str, Any]], key: str, *, seed: int) -> dict[str, Any]:
    values = [float(sample[key]) for sample in samples]
    return {
        "median": median(values),
        "ci95": bootstrap_median_interval(values, seed=seed),
        "minimum": min(values),
        "maximum": max(values),
    }


def _runner_evidence(
    runner: RunnerSpec,
    warmup: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    seed = int(hashlib.sha256(runner.runner_id.encode()).hexdigest()[:8], 16)
    output_sha256 = str(warmup["output_sha256"])
    passed = all(sample.get("output_sha256") == output_sha256 for sample in samples)
    return {
        "runner": asdict(runner),
        "target": warmup["target"],
        "python": warmup["python"],
        "warmup": {
            "output_sha256": output_sha256,
            "process_wall_seconds": warmup["process_wall_seconds"],
        },
        "output": warmup["output"],
        "output_sha256": output_sha256,
        "process_wall_seconds": _summary(samples, "process_wall_seconds", seed=seed),
        "process_peak_rss_mb": _summary(samples, "process_peak_rss_mb", seed=seed + 1),
        "adapter_engine_seconds": _summary(
            [{"value": sample["stages_seconds"]["adapter_reported_engine"]} for sample in samples],
            "value",
            seed=seed + 2,
        ),
        "samples": samples,
        "passed": passed,
    }


def _host_metadata() -> dict[str, Any]:
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def _capability_matrix() -> list[dict[str, Any]]:
    return [
        {
            "workload": WORKLOAD.name,
            "dimensions": f"{WORKLOAD.assets} assets x {WORKLOAD.bars} daily sessions",
            "coverage": list(scale.FRAMEWORKS),
            "use": "controlled and idiomatic repeated performance evidence",
        },
        {
            "workload": scale.WORKLOAD.name,
            "dimensions": f"{scale.WORKLOAD.assets} assets x {scale.WORKLOAD.bars} daily sessions",
            "coverage": list(scale.FRAMEWORKS),
            "use": "large-scale equivalence only; recorded runtimes are not performance claims",
        },
        {
            "workload": "historical_real_250_assets_5040_sessions",
            "dimensions": "250 assets x 5,040 daily sessions",
            "coverage": ["vectorbt_oss", "backtrader", "zipline", "lean"],
            "use": (
                "superseded audit input; single samples, inconsistent setup boundaries, missing "
                "peak RSS for LEAN, and a non-reconstructable temporary data path"
            ),
        },
        {
            "workload": "performance_baselines.json workloads",
            "dimensions": "single-asset, 250-asset, quote, rebalance, and partial-fill workloads",
            "coverage": ["ml4t_backtest"],
            "use": "internal regression evidence; not a cross-framework comparison",
        },
    ]


def collect_evidence(samples: int) -> dict[str, Any]:
    """Warm every runner, interleave measured repetitions, and build both tracks."""
    if samples < MIN_SAMPLES:
        raise ValueError(f"At least {MIN_SAMPLES} measured samples are required")
    calibration = run_calibration_process()
    if (
        calibration["allocated_bytes"] != 32 * 1024 * 1024
        or calibration["worker_seconds"] < 0.02
        or calibration["process_wall_seconds"] < calibration["worker_seconds"]
        or calibration["process_peak_rss_mb"] < 20
    ):
        raise RuntimeError(f"Performance instrumentation calibration failed: {calibration}")

    warmups: dict[str, dict[str, Any]] = {}
    measured: dict[str, list[dict[str, Any]]] = {runner.runner_id: [] for runner in RUNNERS}
    for runner in RUNNERS:
        print(f"Warming {runner.runner_id}...", flush=True)
        warmups[runner.runner_id] = _run_process(runner)
    for repetition in range(samples):
        offset = repetition % len(RUNNERS)
        ordered = RUNNERS[offset:] + RUNNERS[:offset]
        for runner in ordered:
            print(
                f"Measuring {runner.runner_id} sample {repetition + 1}/{samples}...",
                flush=True,
            )
            measured[runner.runner_id].append(_run_process(runner))

    runners = {
        runner.runner_id: _runner_evidence(
            runner,
            warmups[runner.runner_id],
            measured[runner.runner_id],
        )
        for runner in RUNNERS
    }
    pairs: dict[str, dict[str, Any]] = {}
    for framework in scale.FRAMEWORKS:
        external = runners[f"{framework}:external"]
        ml4t = runners[f"{framework}:ml4t"]
        exact = external["output"] == ml4t["output"]
        pairs[framework] = {
            "external_runner": f"{framework}:external",
            "ml4t_runner": f"{framework}:ml4t",
            "exact_output": exact,
            "passed": bool(exact and external["passed"] and ml4t["passed"]),
        }

    idiomatic = {
        name: {
            "sample_source": runner_id,
            "semantic_disclosure": IDIOMATIC_DISCLOSURES[name],
            "invariants_passed": runners[runner_id]["passed"],
        }
        for name, runner_id in IDIOMATIC_RUNNERS.items()
    }
    controlled_passed = all(pair["passed"] for pair in pairs.values())
    idiomatic_passed = all(entry["invariants_passed"] for entry in idiomatic.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "workload": {
            "recipe": asdict(WORKLOAD),
            "recipe_sha256": _json_digest(asdict(WORKLOAD)),
            "data_points": WORKLOAD.data_points,
        },
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "protocol_sha256": _json_digest(MEASUREMENT_PROTOCOL),
        "samples_per_runner": samples,
        "host": _host_metadata(),
        "calibration": calibration,
        "source_digests": _source_digests(),
        "runners": runners,
        "controlled": {
            "equivalence_required": True,
            "pairs": pairs,
            "passed": controlled_passed,
        },
        "idiomatic": {
            "equivalence_claim": False,
            "runners": idiomatic,
            "passed": idiomatic_passed,
        },
        "capability_matrix": _capability_matrix(),
        "historical_assessment": {
            "comparable_to_current_measurements": False,
            "reason": (
                "The 2026-02-28 artifacts used one sample, inconsistent setup boundaries, old "
                "framework revisions, different hardware provenance, and incomplete memory data."
            ),
        },
        "release_gate_passed": controlled_passed and idiomatic_passed,
    }


def report_failures(report: dict[str, Any]) -> list[str]:
    """Return every reason performance evidence cannot be accepted."""
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        return [f"Unsupported performance schema: {report.get('schema_version')!r}"]
    workload = report.get("workload")
    if not isinstance(workload, dict) or workload.get("recipe") != asdict(WORKLOAD):
        failures.append("Performance workload recipe differs")
    elif workload.get("recipe_sha256") != _json_digest(asdict(WORKLOAD)):
        failures.append("Performance workload recipe digest differs")
    if report.get("measurement_protocol") != MEASUREMENT_PROTOCOL:
        failures.append("Performance measurement protocol differs")
    if report.get("protocol_sha256") != _json_digest(MEASUREMENT_PROTOCOL):
        failures.append("Performance protocol digest differs")
    samples_per_runner = report.get("samples_per_runner")
    if not isinstance(samples_per_runner, int) or samples_per_runner < MIN_SAMPLES:
        failures.append("Performance evidence has too few measured samples")
        samples_per_runner = 0
    if report.get("source_digests") != _source_digests():
        failures.append("Performance source digests differ")

    raw_runners = report.get("runners")
    if not isinstance(raw_runners, dict) or set(raw_runners) != set(RUNNERS_BY_ID):
        return failures + ["Performance runner coverage differs"]
    runners = cast(dict[str, dict[str, Any]], raw_runners)
    manifest = load_framework_manifest()
    reconstructed_input = scale._expected_input_digest(WORKLOAD)
    for runner_id, runner in runners.items():
        samples = runner.get("samples")
        if not isinstance(samples, list) or len(samples) != samples_per_runner:
            failures.append(f"{runner_id} sample count differs")
            continue
        if any(not isinstance(sample, dict) for sample in samples):
            failures.append(f"{runner_id} contains a malformed sample")
            continue
        runner_spec = RUNNERS_BY_ID[runner_id]
        expected_target = manifest.targets[runner_spec.framework].evidence_metadata()
        target = runner.get("target")
        if (
            not isinstance(target, dict)
            or scale._declared_target_metadata(target) != expected_target
            or target.get("actual_version") != expected_target["version"]
            or any(sample.get("target") != target for sample in samples)
        ):
            failures.append(f"{runner_id} target identity differs")
        elif runner_spec.framework == "vectorbt_pro" and (
            target.get("actual_commit") != expected_target["commit"]
            or target.get("actual_immutable_id") != expected_target["immutable_id"]
        ):
            failures.append(f"{runner_id} installed source identity differs")
        output_sha256 = runner.get("output_sha256")
        if not isinstance(output_sha256, str) or any(
            sample.get("output_sha256") != output_sha256
            or sample.get("ml4t", {}).get("dirty") is not False
            or not isinstance(sample.get("ml4t", {}).get("commit"), str)
            or len(sample["ml4t"]["commit"]) != 40
            for sample in samples
        ):
            failures.append(f"{runner_id} output checksum or clean-tree identity differs")
        if runner.get("output") is None or _json_digest(runner["output"]) != output_sha256:
            failures.append(f"{runner_id} retained output digest differs")
        if any(
            sample.get("input", {}).get("raw_sha256") != reconstructed_input
            or sample.get("input") != samples[0].get("input")
            or sample.get("thread_environment") != THREAD_ENVIRONMENT
            for sample in samples
        ):
            failures.append(f"{runner_id} input or thread environment differs")
        if any(
            float(sample.get("process_wall_seconds", 0)) <= 0
            or float(sample.get("process_peak_rss_mb", 0)) <= 0
            or (runner_id == "lean:external" and sample.get("lean_container_observed") is not True)
            for sample in samples
        ):
            failures.append(f"{runner_id} wall-time or full-process memory evidence is missing")
        for key, source_key, seed_offset in (
            ("process_wall_seconds", "process_wall_seconds", 0),
            ("process_peak_rss_mb", "process_peak_rss_mb", 1),
        ):
            seed = int(hashlib.sha256(runner_id.encode()).hexdigest()[:8], 16) + seed_offset
            if runner.get(key) != _summary(samples, source_key, seed=seed):
                failures.append(f"{runner_id} {key} summary differs")
        engine_samples = [
            {"value": sample["stages_seconds"]["adapter_reported_engine"]} for sample in samples
        ]
        seed = int(hashlib.sha256(runner_id.encode()).hexdigest()[:8], 16) + 2
        if runner.get("adapter_engine_seconds") != _summary(engine_samples, "value", seed=seed):
            failures.append(f"{runner_id} engine summary differs")
        if runner.get("passed") is not True:
            failures.append(f"{runner_id} did not pass")

    controlled = report.get("controlled")
    pairs = controlled.get("pairs") if isinstance(controlled, dict) else None
    if not isinstance(pairs, dict) or set(pairs) != set(scale.FRAMEWORKS):
        failures.append("Controlled performance pair coverage differs")
    else:
        for framework, pair in pairs.items():
            external = runners[f"{framework}:external"]
            ml4t = runners[f"{framework}:ml4t"]
            if (
                pair.get("passed") is not True
                or pair.get("exact_output") is not True
                or external.get("output") != ml4t.get("output")
            ):
                failures.append(f"{framework} controlled performance output differs")
    if not isinstance(controlled, dict) or controlled.get("passed") is not True:
        failures.append("Controlled performance track did not pass")

    idiomatic = report.get("idiomatic")
    idiomatic_runners = idiomatic.get("runners") if isinstance(idiomatic, dict) else None
    if (
        not isinstance(idiomatic, dict)
        or idiomatic.get("equivalence_claim") is not False
        or idiomatic.get("passed") is not True
        or not isinstance(idiomatic_runners, dict)
        or set(idiomatic_runners) != set(IDIOMATIC_RUNNERS)
    ):
        failures.append("Idiomatic performance track is incomplete")
    else:
        for name, runner_id in IDIOMATIC_RUNNERS.items():
            entry = idiomatic_runners[name]
            if (
                entry.get("sample_source") != runner_id
                or entry.get("semantic_disclosure") != IDIOMATIC_DISCLOSURES[name]
                or entry.get("invariants_passed") is not True
            ):
                failures.append(f"{name} idiomatic disclosure differs")
    if report.get("capability_matrix") != _capability_matrix():
        failures.append("Performance capability matrix differs")
    if report.get("release_gate_passed") is not True:
        failures.append("Performance release gate did not pass")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=tuple(RUNNERS_BY_ID))
    parser.add_argument("--calibration-worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        _write_json(args.output, run_worker(RUNNERS_BY_ID[args.worker]))
        return 0
    if args.calibration_worker:
        if args.output is None:
            parser.error("--calibration-worker requires --output")
        _write_json(args.output, run_calibration_worker())
        return 0
    if args.check:
        report = cast(dict[str, Any], json.loads(ACCEPTED_PATH.read_text(encoding="utf-8")))
        failures = report_failures(report)
        for failure in failures:
            print(f"- {failure}")
        return 1 if failures else 0
    candidate = collect_evidence(args.samples)
    _write_json(CANDIDATE_PATH, candidate)
    failures = report_failures(candidate)
    if failures:
        print("Accepted performance evidence unchanged:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    _write_json_atomic(ACCEPTED_PATH, candidate)
    print(f"Accepted performance evidence: {ACCEPTED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
