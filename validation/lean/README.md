# Lean CLI Validation

**Status**: Setup In Progress
**Date**: January 2026

## Overview

QuantConnect LEAN is an open-source C# algorithmic trading engine. The `lean-cli` provides
a command-line interface for running LEAN locally via Docker.

## Installation

```bash
# Install lean-cli
pip install lean

# Verify installation
lean --version  # 1.0.221
```

## Setup Challenges

### 1. QuantConnect Account Required

The `lean init` command requires QuantConnect API credentials:

```
$ lean init
Your user id and API token are needed to make authenticated requests to the QuantConnect API
You can request these credentials on https://www.quantconnect.com/account
```

**Impact**: Cannot run `lean backtest` without completing `lean init`.

### 2. CLI vs Direct Engine

**Two approaches to run LEAN locally:**

| Approach | Pros | Cons |
|----------|------|------|
| `lean-cli` | Easy setup, Docker-based | Requires QuantConnect account |
| Direct LEAN | No account needed | Requires .NET, manual setup |

### 3. Data Requirements

LEAN expects data in a specific format:
- Location: `./data/equity/usa/daily/` (for daily equity data)
- Format: CSV with timestamp, OHLCV columns
- Or: Custom data class for arbitrary formats

## Workarounds

### Option A: Create QuantConnect Account (Free)

1. Register at https://www.quantconnect.com/account
2. Get User ID and API Token
3. Run `lean init` with credentials
4. Use `lean backtest` normally

### Option B: Direct LEAN Engine (No Account)

```bash
# Install .NET 6.0+
# Clone LEAN repository
git clone https://github.com/QuantConnect/Lean.git

# Build
cd Lean
dotnet build QuantConnect.Lean.sln

# Configure
# Edit Launcher/config.json for your algorithm

# Run
cd Launcher/bin/Debug
dotnet QuantConnect.Lean.Launcher.dll
```

### Option C: Docker Direct (Partial)

```bash
# Pull LEAN image
docker pull quantconnect/lean:latest

# Run with mounted data and algorithm
docker run -v $(pwd)/data:/Lean/Data \
           -v $(pwd)/algorithm:/Lean/Algorithm.Python \
           quantconnect/lean:latest
```

## Comparison with ml4t-backtest

| Aspect | LEAN | ml4t-backtest |
|--------|------|---------------|
| Language | C# (Python bindings) | Python (Rust planned) |
| Setup | Complex (Docker/.NET) | Simple (pip install) |
| Data Format | Proprietary | Polars DataFrame |
| Account Req | For CLI | None |
| Speed | Fast | Fast (faster with Rust) |
| Event-Driven | Yes | Yes |
| Vectorized | No | Hybrid |

## Project Structure

```
validation/lean/
├── README.md                  # This file
├── scenario_01_long_only/     # Long-only momentum strategy
│   ├── main.py               # LEAN algorithm
│   └── config.json           # Project config
└── data/                      # Local data (if using direct approach)
```

## Validation Scenarios

Planned scenarios to port from ml4t-backtest:

1. **scenario_01_long_only**: Simple long-only momentum
2. **scenario_02_long_short**: Long/short with stops
3. **scenario_05_take_profit**: Take-profit targets

## References

- [Lean CLI Documentation](https://www.quantconnect.com/docs/v2/lean-cli)
- [LEAN GitHub Repository](https://github.com/QuantConnect/Lean)
- [Using LEAN Offline (Forum)](https://www.quantconnect.com/forum/discussion/18614/using-lean-completely-offline-without-quantconnect-in-2025-still-a-thing/)

## Conclusion

**Recommendation**: For ml4t-backtest validation purposes:

1. **Skip LEAN CLI validation** for now - setup overhead too high
2. **Focus on VectorBT Pro comparison** - already validated, exact match achieved
3. **Pursue Rust backend** - better ROI than LEAN integration

LEAN is a valid reference but the setup friction makes it less practical for quick
cross-validation. The Rust backend approach documented in `docs/rust-backend-feasibility.md`
offers a clearer path to high-performance backtesting with Python-native workflow.
