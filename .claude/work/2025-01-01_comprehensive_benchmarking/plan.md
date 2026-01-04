# Comprehensive Performance & Correctness Benchmarking

**Work Unit**: `2025-01-01_comprehensive_benchmarking`
**Location**: `/home/stefan/ml4t/software/backtest/.claude/work/2025-01-01_comprehensive_benchmarking/`
**Goal**: Benchmark performance and re-validate correctness across ALL frameworks including LEAN CLI and Zipline

---

## Current State (from exploration)

### Correctness Validation
| Framework | Scenarios 1-4 | Scenarios 5-10 | Large-Scale (500×10yr) |
|-----------|---------------|----------------|------------------------|
| VectorBT Pro | ✅ PASS | ✅ PASS (Dec 26) | ✅ 119,591 trades |
| VectorBT OSS | ✅ PASS | ✅ PASS | ✅ 114,607 trades |
| Backtrader | ✅ PASS | ✅ PASS | ✅ 119,577 trades |
| Zipline | ✅ PASS | ✅ PASS | ✅ 119,577 trades |
| **LEAN CLI** | ❌ NOT DONE | ❌ NOT DONE | ❌ NOT DONE |

### Performance Benchmarking
| Framework | Dedicated benchmark_performance.py | benchmark_suite.py |
|-----------|-----------------------------------|-------------------|
| VectorBT Pro | ✅ Yes | ✅ Yes |
| Backtrader | ✅ Yes | ✅ Yes |
| **Zipline** | ❌ **NO** | ✅ Partial |
| **LEAN CLI** | ❌ NOT DONE | ❌ NOT DONE |

---

## Phase 1: LEAN CLI Setup

**Prerequisite**: User has QuantConnect account (stefan@applied-ai.com)
**Needs**: API Token from https://www.quantconnect.com/account

### Task 1.1: Initialize LEAN CLI
```bash
cd /home/stefan/ml4t/software/backtest/validation/lean
source ../.venv/bin/activate
lean init  # Enter User ID and API Token when prompted
```

### Task 1.2: Create LEAN Validation Scenarios
Port scenarios 1-4 to LEAN Python algorithm format:

**Files to create:**
- `validation/lean/scenario_01_long_only/main.py`
- `validation/lean/scenario_02_long_short/main.py`
- `validation/lean/scenario_03_stop_loss/main.py`
- `validation/lean/scenario_04_take_profit/main.py`

### Task 1.3: Create Data Converter
LEAN requires specific CSV format. Create utility to convert test data.

**File**: `validation/lean/data_converter.py`

---

## Phase 2: Zipline Performance Benchmarking

### Task 2.1: Create Dedicated Zipline Benchmark Script
Mirror structure of `vectorbt_pro/benchmark_performance.py`

**File to create**: `validation/zipline/benchmark_performance.py`

**Configurations to test:**
- (100, 1), (1000, 1), (10000, 1)
- (1000, 10), (1000, 50)
- (2520, 500) - Daily baseline

### Task 2.2: Add Warm-up Runs
Modify `benchmark_suite.py` to add warm-up runs for Zipline (currently missing)

---

## Phase 3: Re-validate All Correctness

### Task 3.1: Create Unified Validation Runner
**File to create**: `validation/run_all_correctness.py`

Runs all 10 scenarios across all 5 frameworks:
- VectorBT Pro (requires `.venv-vectorbt-pro`)
- VectorBT OSS (uses `.venv` or `.venv-validation`)
- Backtrader (uses `.venv-backtrader` or `.venv-validation`)
- Zipline (uses `.venv-zipline` or `.venv-validation`)
- LEAN CLI (uses Docker)

### Task 3.2: Generate Correctness Report
Output: `validation/CORRECTNESS_RESULTS.md`

---

## Phase 4: Unified Performance Benchmarking

### Task 4.1: Create Unified Benchmark Runner
**File to create**: `validation/run_all_benchmarks.py`

### Task 4.2: Generate Performance Report
**File to create**: `validation/BENCHMARK_RESULTS.md`

Template:
```markdown
# Performance Benchmark Results

**Generated**: {timestamp}
**Machine**: {specs}

## Summary Table

| Framework | 100×1 | 1000×1 | 10000×1 | 1000×10 | 1000×50 | 2520×500 |
|-----------|-------|--------|---------|---------|---------|----------|
| ml4t.backtest | X.XXs | ... | ... | ... | ... | ... |
| VectorBT Pro | X.XXs | ... | ... | ... | ... | ... |
| VectorBT OSS | X.XXs | ... | ... | ... | ... | ... |
| Backtrader | X.XXs | ... | ... | ... | ... | ... |
| Zipline | X.XXs | ... | ... | ... | ... | ... |
| LEAN CLI | X.XXs | ... | ... | ... | ... | ... |
```

---

## Phase 5: Documentation Updates

### Task 5.1: Update validation/README.md
- Add LEAN CLI section with setup instructions
- Update Test Coverage Matrix with all results
- Add Zipline performance benchmark section

### Task 5.2: Update docs/competitive-positioning.md
- Replace assumed numbers with measured data
- Add LEAN comparison (if validation succeeds)

---

## Execution Order

1. **Phase 1.1**: Get QuantConnect API token, run `lean init`
2. **Phase 1.2-1.3**: Create LEAN scenarios and data converter
3. **Phase 2.1-2.2**: Create Zipline benchmark script
4. **Phase 3**: Run all correctness validation (re-verify)
5. **Phase 4**: Run all performance benchmarks
6. **Phase 5**: Update documentation with real results

---

## Files to Create/Modify

### New Files (8)
1. `validation/lean/data_converter.py`
2. `validation/lean/scenario_02_long_short/main.py`
3. `validation/lean/scenario_03_stop_loss/main.py`
4. `validation/lean/scenario_04_take_profit/main.py`
5. `validation/zipline/benchmark_performance.py`
6. `validation/run_all_correctness.py`
7. `validation/run_all_benchmarks.py`
8. `validation/BENCHMARK_RESULTS.md`

### Files to Modify (3)
1. `validation/benchmark_suite.py` - Add Zipline warm-up, LEAN support
2. `validation/README.md` - Update with actual results
3. `docs/competitive-positioning.md` - Replace assumptions with data

---

## Virtual Environments

| Framework | Environment | Notes |
|-----------|-------------|-------|
| ml4t.backtest | `.venv` | Main dev environment |
| VectorBT Pro | `.venv-vectorbt-pro` | Cannot coexist with OSS |
| VectorBT OSS | `.venv` or `.venv-validation` | |
| Backtrader | `.venv-backtrader` or `.venv-validation` | |
| Zipline | `.venv-zipline` or `.venv-validation` | |
| LEAN CLI | Docker | No venv needed |

---

## Success Criteria

1. ✅ LEAN CLI initialized and running backtests
2. ✅ Zipline performance benchmarked (not just correctness)
3. ✅ All 10 scenarios × 5 frameworks validated
4. ✅ Unified benchmark report with measured data
5. ✅ Documentation updated with real numbers (no assumptions)
