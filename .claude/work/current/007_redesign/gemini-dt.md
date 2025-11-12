# **Product Requirements Document: Event-Driven Backtesting Engine**

| Status      | Draft                       |
|:------------|:----------------------------|
| **Author**  | Principal Product Architect |
| **Date**    | 2025-11-10                  |
| **Version** | 0.9                         |
| **Target**  | v0 (MVP), v1, v2            |

---

## **0\) Executive Summary**

This document specifies the requirements for "Helios," a next-generation, high-performance, event-driven backtesting
engine for quantitative and machine-learning-based trading strategies. Its purpose is to replace legacy Zipline
workflows, providing a modern, transparent, and extensible platform that leverages **Polars** for memory-efficient data
representation and **Numba** for just-in-time (JIT) compilation of critical performance paths. The engine is built on
principles of **determinism**, **reproducibility**, **modularity**, **performance**, and **developer experience (UX)
clarity**.

* **v0 (MVP)** will ship a core event loop, single-asset (equity) support, bar-based data, basic order types (market,
  limit), simple cost/slippage models, and a fully reproducible run manifest.
* **v1** will add multi-asset/multi-currency support, futures and crypto, advanced order types, pluggable
  slippage/impact models, and integration hooks for external analysis libraries (replacing Alphalens/Pyfolio).
* **v2** will introduce advanced microstructure simulation, portfolio-level optimizers, dynamic universe management, and
  adapters for paper/live trading.

---

##   

## **1\) Context & Goals**

### **Why a new engine is needed**

The existing Zipline-based ecosystem suffers from significant technical debt. Its reliance on unmaintained
dependencies (e.g., bcolz), deep and complex Cython integrations, and a monolithic architecture make it difficult to
maintain, modernize, and extend. Key pain points include:

* **Maintenance Burden:** Cython code is difficult to debug and profile, and it creates a high barrier for new
  contributors.
* **Data I/O:** Legacy storage formats are slow and incompatible with modern data tooling (e.g., Arrow, Parquet,
  Polars).
* **UX Friction:** The API, particularly for data bundling and configuration, is cumbersome and opaque.
* **ML Integration:** Implementing strict, leak-free, point-in-time ML workflows is non-trivial and error-prone.

This engine will address these gaps by building on a modern, high-performance Python stack (Polars, Numba, Pydantic)
with a clean, modular design.

### **Primary Use Cases**

* **Event-Driven (Vectorized-Hybrid):** Simulation of stateful, path-dependent strategies that cannot be fully
  vectorized.
* **ML-Driven Strategies:** Backtesting workflows that consume pre-computed features and predictions, with strict
  temporal safeguards against look-ahead bias.
* **Flexible Simulation:** Modeling realistic market friction, including various order types, slippage, impact, and
  commission structures.
* **Portfolio-Level Research:** Simulating multi-asset, multi-strategy portfolios, managing dynamic universes, and
  handling data across different frequencies.

### **Non-Goals (for v0/v1)**

* **Microsecond-Level HFT Simulation:** We will not model order book queue priority or full tick-by-tick matching logic.
  The smallest time quantum will be the individual event (e.g., a trade or a bar).
* **Real-time Paper/Live Trading:** v0/v1 is purely for simulation. Adapters for live execution are a v2+ goal.
* **Complex Derivatives Engine:** Full options greeks, volatility surface modeling, or complex fixed-income analytics
  are out of scope.

### **Success Metrics**

| Metric            | Target (v1)            | Description                                                                            |
|:------------------|:-----------------------|:---------------------------------------------------------------------------------------|
| **Performance**   | \> 1M bar-events/sec   | Throughput on a single-asset, 10-year, 1-minute bar dataset (on reference hardware).   |
| **Determinism**   | 100%                   | Bit-for-bit identical BacktestResult artifacts from identical Run Manifests.           |
| **Adoption**      | 80% of quant workflows | Migration of active quant research from legacy Zipline within 12 months of v1 launch.  |
| **Test Coverage** | \> 95%                 | Line and branch coverage for core event loop, accounting, and fill logic.              |
| **Documentation** | 100%                   | All public APIs, config schemas, and plugin interfaces fully documented with examples. |

---

## **2\) Users & Personas**

* **Quantitative Researchers:** Need a fast, flexible, and accurate tool to iterate on alpha ideas. They value speed,
  statistical rigor, and realistic friction modeling.
* **ML Engineers (Quant):** Need a *provably correct* platform to test predictive models. They require strict
  anti-leakage guards, easy integration of feature/prediction streams, and robust cross-validation tools.
* **Portfolio Managers:** Need auditable, reproducible, and transparent results to validate strategies. They care about
  realistic P\&L, risk/exposure reporting, and the ability to compare experiments.

### **Desired UX**

* **Config-First:** Most runs are defined by a single, validated YAML/JSON file.
* **Concise Python API:** The Python API for strategy definition is minimal, expressive, and type-hinted.
* **Excellent Diagnostics:** Error messages are precise (e.g., config validation errors point to the exact line/field)
  and actionable.
* **Strong Introspection:** The BacktestResult object is easy to query, and debug-level event traces are available.
* **Experiment Comparison:** Standardized, hash-based run manifests make comparing "what-if" scenarios trivial.

---

## **3\) Core Product Principles**

* **Deterministic & Auditable:** All operations must be deterministic. All randomness (e.g., slippage, order matching
  tie-breaks) must be controlled by a single, seedable PRNG. Run outputs must be attributable to a complete hash of all
  inputs.
* **Config-First, Code-Second:** A validated config file (via Pydantic) is the primary entry point for a run. The Python
  API is used to define complex *logic*, not *orchestration*.
* **Event-Driven Core:** A time-ordered priority queue of typed events is the engine's "heartbeat." This allows for
  flexible handling of multi-frequency data and asynchronous events.
* **Modular Plugin Architecture:** The Broker, Slippage, Cost, Execution, and Analytics layers are pluggable interfaces.
  This allows users to swap out a simple slippage model for a complex, microstructure-aware one without changing
  strategy code.
* **Performance via Modern Python:**
    * **Data:** Use **Polars** and **Arrow** as the native in-memory representation for all tabular data (market data,
      ledgers, results). This avoids pandas copies and numpy type-coercion issues.
    * **Logic:** Use **Numba** to JIT-compile the core event loop, fill-matching logic, and portfolio state updates.
      Strategy code itself *can* be JIT-ted if users opt-in.
    * **Future-Proof:** The architecture will allow performance-critical plugins (e.g., a fill engine) to be rewritten
      in **Rust** (via pyo3) without changing the core Python API.
* **Standardized Interoperability:** The engine's sole output is a standardized BacktestResult artifact (containing
  Polars DataFrames serialized to Parquet/Arrow). Separate libraries—which will replace Alphalens and Pyfolio—will
  consume this artifact for analysis and plotting.

---

## **4\) Architecture Overview**

The engine operates as a discrete-event simulation built around a central, time-ordered event bus.

1. **Data Layer:**
    * **Connectors:** Read data (Parquet, CSV, Arrow IPC) into Polars LazyFrames.
    * **Schema Registry:** Ensures input data (bars, trades, signals) conforms to expected schemas.
    * **Context:** Manages calendars, timezones, corporate actions, FX rates, and instrument metadata (tick sizes,
      multipliers).
2. **Event Fabric (Core Loop):**
    * A single heapq priority queue containing (timestamp, priority, event\_id, event\_object) tuples. This ensures
      total, deterministic ordering.
    * **Events:** ClockEvent, MarketDataEvent, SignalEvent, OrderRequestEvent, FillEvent, PortfolioUpdateEvent,
      RiskCheckEvent.
    * The loop pops the next event and dispatches it to subscribed services (Strategy, Broker, Portfolio).
3. **Strategy Layer:**
    * User-defined Strategy objects subscribe to events (e.g., on\_clock, on\_bar).
    * Maintains internal state.
    * Accesses data context for rolling windows (e.g., self.data.get\_window(asset, 30)).
    * Emits OrderRequestEvents to the event bus.
4. **Broker/Execution Layer:**
    * Subscribes to OrderRequestEvents.
    * Manages an internal order book/queue.
    * Subscribes to MarketDataEvents to check for fill conditions.
    * Applies **Slippage** and **Cost** plugins.
    * Emits FillEvents back to the bus.
5. **Portfolio/Accounting Layer:**
    * Subscribes to FillEvents to update positions and cash.
    * Subscribes to MarketDataEvents (e.g., end-of-day clock) to mark-to-market (MTM) positions and update equity
      curves.
    * Handles corporate actions, dividends, and margin calls.
    * Emits PortfolioUpdateEvents.
6. **Risk & Controls:**
    * **Pre-Trade:** Subscribes to OrderRequestEvents; can veto them (emits OrderRejectEvent).
    * **Post-Trade:** Subscribes to PortfolioUpdateEvents; checks drawdown, exposure, etc. Can emit a HaltTradingEvent.
7. **Outputs & Analytics:**
    * **Loggers:** Simple subscribers that listen to the bus (FillEvent, PortfolioUpdateEvent) and build the final
      BacktestResult artifact in memory (as Polars DataFrames).
    * **Artifacts:** At the end of the run, these loggers serialize their DataFrames to the output directory.
8. **Experimentation & Repro:**
    * A wrapper layer that:
        1. Validates the input Config object.
        2. Hashes the config, data references, and code.
        3. Initializes the PRNG seed.
        4. Sets up all layers and starts the event loop.
        5. Saves the BacktestResult and the RunManifest.json.

---

## **5\) Functional Requirements**

### **5.1 Data Ingestion & Time Semantics**

* **FR-5.1.1: Supported Inputs (Must)**
    * Time-based OHLCV bars (e.g., 1-minute, 1-hour, 1-day).
    * Irregular event streams (e.g., trades, quotes, signals).
    * *Rationale:* Covers the majority of quant and ML use cases.
    * *Acceptance:* Engine can ingest and process a Parquet file of 1-minute bars and a separate Parquet file of
      timestamped signal events.
* **FR-5.1.2: Clocks & Alignment (v0 Must, v1 Should)**
    * v0: A single "master" clock based on the primary data frequency (e.g., every minute bar).
    * v1: Support for multiple, independent clocks (e.g., a 1-minute bar clock and a 1-hour signal clock) with
      deterministic alignment rules.
    * *Rationale:* v0 simplifies the event loop; v1 is required for multi-frequency strategies.
    * *Acceptance (v1):* A strategy can subscribe to both a 15min clock and a 1H clock, with events firing
      deterministically.
* **FR-5.1.3: Data Quality Policies (Must)**
    * Engine must have configurable policies for missing timestamps (fill forward, drop), duplicates (drop), and
      out-of-order records (drop, error).
    * *Rationale:* Real data is dirty; behavior must be explicit.
    * *Acceptance:* A config setting on\_missing\_data: "ffill" correctly uses the last known price.
* **FR-5.1.4: Calendars & Sessions (Must)**
    * Support for exchange calendars (e.g., NYSE, CME) including sessions, early closes, holidays, and DST.
    * *Rationale:* Critical for correct P\&L, bar alignment, and TIF logic.
    * *Acceptance:* A DAY order placed on Friday expires at the market close and is not active on Monday.
* **FR-5.1.5: Corporate Actions (v1 Must)**
    * Engine must ingest and apply splits, dividends (cash/stock), and symbol changes.
    * Policies for adjustment (raw, back-adjusted) must be configurable.
    * *Rationale:* Ignoring these invalidates all long-term backtests.
    * *Acceptance:* A 2:1 split correctly doubles the strategy's position and halves the price, with no P\&L impact.
* **FR-5.1.6: Instrument Metadata (Must)**
    * System must load and use tick size, lot size, multipliers (futures), and margin requirements.
    * *Rationale:* Required for realistic order sizing and P\&L calculation.
    * *Acceptance:* An order for 1.5 lots on an asset with lot\_size=1 is rejected or rounded based on config.

### **5.2 Strategy Definition API**

* **FR-5.2.1: Python Class API (Must)**
    * Provide a StrategyBase class with lifecycle methods (e.g., on\_init, on\_bar, on\_event).
    * Must provide state storage (self.state) and methods to emit orders (self.order(...)).
    * Must provide an accessor for data windows (self.data.history(assets, lookback)).
    * *Rationale:* Provides maximum flexibility for complex, stateful logic.
    * *Acceptance:* A user can implement a stateful moving average crossover strategy.
* **FR-5.2.2: Config-Driven DSL (v1 Should)**
    * Provide a YAML-based "graph" definition for common strategies (e.g., "Signal \-\> Sizer \-\> Order").
    * *Rationale:* Lowers the barrier to entry for simple "signal-to-weight" strategies, common in ML.
    * *Acceptance:* A YAML file can define a strategy that loads a 'PREDICTION' column and orders a target percentage.
* **FR-5.2.3: Anti-Leakage (ML) (Must)**
    * All data access (self.data.history) must be strictly point-in-time.
    * The event loop must process signals *before* the market data they are intended to be executed against. (e.g., a
      signal computed on 09:30:00 bar close data is processed *before* the 09:31:00 bar open).
    * *Rationale:* This is the single most critical feature for ML users.
    * *Acceptance:* A test strategy that tries to access bar.close *before* the bar close event fails with an explicit
      LookAheadError.
* **FR-5.2.4: Warm-up & State (Must)**
    * The engine must support a "warm-up" period where strategy logic is run (e.g., to populate indicators) but no
      orders are placed.
    * *Rationale:* Prevents strategies from starting with "cold" indicators.
    * *Acceptance:* A 50-period SMA strategy places no orders for the first 50 bars, then begins trading.

### **5.3 Orders, Fills, and Execution Model**

* **FR-5.3.1: Order Types (v0 Must, v1 Should)**
    * v0: Market, Limit, Stop, Stop-Limit.
    * v1: Trailing Stop, Bracket (OCO/OTO), Reduce-Only.
    * *Rationale:* Covers basic needs first, then adds complex types used by active traders.
    * *Acceptance (v0):* A Stop order is correctly triggered and filled as a Market order when the price penetrates the
      stop level.
* **FR-5.3.2: Time-in-Force (TIF) (v0 Must, v1 Should)**
    * v0: DAY, GTC (Good-til-Canceled).
    * v1: IOC (Immediate-or-Cancel), FOK (Fill-or-Kill), GTD (Good-til-Date).
    * *Rationale:* Core to order lifecycle management.
    * *Acceptance:* A DAY order is automatically canceled at the session close if unfilled.
* **FR-5.3.3: Deterministic Bar Matching (Must)**
    * The fill logic for bar data must be deterministic and configurable.
    * Example policies: "Market orders fill at open", "Limit buys fill if low \<= limit\_price", "Fills occur at
      limit\_price (optimistic) or open (pessimistic)".
    * *Rationale:* This is the "heart" of the simulation. Its behavior must be explicit and auditable.
    * *Acceptance:* Given identical data and a "fill-at-open" policy, a market order always fills at the open price.
* **FR-5.3.4: Slippage Models (v0 Must, v1 Must)**
    * v0: Simple models: fixed bps, fixed dollar amount, pct of volume.
    * v1: Pluggable, advanced models: volume-participation, square-root impact.
    * *Rationale:* Friction is a primary driver of strategy decay.
    * *Acceptance:* Enabling a 10bps slippage model correctly adjusts all fill prices by 0.10%.
* **FR-5.3.5: Broker Behaviors (v1 Must)**
    * Support for shorting constraints (e.g., no shorting, hard-to-borrow fees).
    * Margin/leverage modeling (e.g., 2:1 leverage, margin calls).
    * Settlement (T+N) and futures mark-to-market.
    * *Rationale:* Required for realistic portfolio P\&L.
    * *Acceptance:* A strategy attempting to short a non-shortable asset has its order rejected.

### **5.4 Portfolio, Cash & Accounting**

* **FR-5.4.1: Double-Entry Ledgers (Must)**
    * All transactions (fills, fees, dividends) must be recorded in immutable ledgers (e.g., Polars DataFrames) that
      track cash and positions.
    * *Rationale:* Ensures auditable, provably correct P\&L.
    * *Acceptance:* The sum of cash\_ledger and market\_value always equals the equity\_curve (barring floating point
      error).
* **FR-5.4.2: P\&L Calculation (Must)**
    * Must compute and log realized (on trade close) and unrealized (MTM) P\&L.
    * *Rationale:* Core output of any backtest.
    * *Acceptance:* A buy-and-hold backtest shows 0 realized P\&L and an unrealized P\&L matching the asset's price
      change.
* **FR-5.4.3: Multi-Currency (v1 Must)**
    * Support for strategies trading in multiple currencies, with a consistent base valuation currency.
    * Requires a configurable FX rate source for MTM.
    * *Rationale:* Essential for global macro, FX, and crypto strategies.
    * *Acceptance:* Buying a EUR-denominated asset with a USD account correctly logs the FX conversion and MTM P\&L in
      USD.
* **FR-5.4.4: Sizing & Allocation (v0 Must, v1 Should)**
    * v0: Simple order sizing (share count, target notional, target percent).
    * v1: Pluggable PortfolioAllocator interface for complex rebalancing (e.g., risk parity, MVO) at the portfolio
      level.
    * *Rationale:* Simple sizing is needed for v0, but advanced PMs need portfolio-level optimization.
    * *Acceptance (v0):* order\_target\_percent(asset, 0.10) correctly calculates share count to match 10% of current
      portfolio equity.

### **5.5 Risk Management & Compliance**

* **FR-5.5.1: Pre-Trade Checks (Must)**
    * Configurable rules to reject orders *before* they are sent to the broker.
    * Examples: max position size, max notional, banned list.
    * *Rationale:* Basic risk controls are fundamental.
    * *Acceptance:* An order for 1,000,000 shares is rejected if max\_position\_notional is $1,000,000.
* **FR-5.5.2: Post-Trade Controls (v1 Should)**
    * Running controls that monitor the portfolio state.
    * Examples: max drawdown (halts trading), exposure limits.
    * *Rationale:* Simulates realistic PM-level risk management.
    * *Acceptance:* A strategy that hits the max\_drawdown: 0.20 config limit is "killed" (all positions liquidated, no
      new orders).

### **5.6 Experimentation, Reproducibility & Configuration**

* **FR-5.6.1: Config-First Execution (Must)**
    * The primary entry point is run\_backtest(config\_path\_or\_dict).
    * All config files are validated via Pydantic models.
    * *Rationale:* Enforces explicit, declarative, and reproducible runs.
    * *Acceptance:* Running the engine with a typo in the YAML config produces a clear Pydantic validation error.
* **FR-5.6.2: Run Manifest (Must)**
    * Every run *must* produce a RunManifest.json artifact.
    * This file must contain:
        * The complete, resolved config used.
        * Content hashes (e.g., SHA256) of all input data files.
        * Git commit hash of the strategy code.
        * The global PRNG seed.
        * Environment info (Python version, library versions).
    * *Rationale:* This is the core of 100% reproducibility.
    * *Acceptance:* Running run\_backtest(manifest\_path) re-runs the *exact* same simulation and produces bit-identical
      results.
* **FR-5.6.3: Batch/Sweep Execution (v1 Should)**
    * A helper utility to run batch backtests over a parameter grid (e.g., grid search, random search).
    * Must support parallel execution (e.g., via multiprocessing or ray).
    * *Rationale:* Core workflow for parameter optimization.
    * *Acceptance:* A sweep config produces N BacktestResult artifacts, one for each parameter combination.

### **5.7 Outputs & Analytics (Standardized Artifacts)**

* **FR-5.7.1: Unified BacktestResult Object (Must)**
    * All runs return a single BacktestResult object (serializable).
    * This object contains Polars DataFrames for: equity\_curve, returns, positions, trades, orders.
    * It also contains summary metrics (Sharpe, CAGR, DD) as a dict.
    * *Rationale:* A standard, predictable output is required for all downstream tooling.
    * *Acceptance:* The returned object can be saved (result.save("path")) and re-loaded (load\_result("path")).
* **FR-5.7.2: Standardized Schemas (Must)**
    * The output DataFrames (trades, positions) must have stable, versioned schemas. See Appendix C.
    * The trades log *must* include MFE/MAE (Max Favorable/Adverse Excursion).
    * *Rationale:* Enables building a separate, stable ecosystem of analysis tools.
    * *Acceptance:* The trades DataFrame contains the mfe and mae columns.
* **FR-5.7.3: Analysis Hooks (v1 Must)**
    * The BacktestResult object will have methods/properties formatted for consumption by Alphalens/Pyfolio replacement
      libraries.
    * *Rationale:* The engine *does not* plot, but it *must* enable plotting.
    * *Acceptance:* A separate helios-analysis library can take BacktestResult and generate a tearsheet.

---

## **6\) Non-Functional Requirements (NFRs)**

* **NFR-1: Performance (Must)**
    * v0: Target \> 1M 1-asset bar-events/sec on a reference machine (e.g., M2 Pro or equivalent).
    * v1: Target \> 500k multi-asset (100 assets) bar-events/sec.
    * Core event loop and accounting logic must be Numba-JIT-compiled.
* **NFR-2: Determinism (Must)**
    * Given an identical RunManifest (same data hashes, config, code hash, seed), runs must be bit-for-bit reproducible.
    * This implies stable sorting for all set/dict iterations and deterministic tie-breaking in the event loop.
* **NFR-3: Extensibility (Must)**
    * Broker, Slippage, Cost, and Risk models must be pluggable via a registry or entry-point system.
    * Plugin interfaces must be versioned.
* **NFR-4: Reliability (Must)**
    * Property-based tests (e.g., via hypothesis) must be used to validate accounting invariants (e.g., cash \+
      market\_value \== equity).
    * Golden tests must be established, comparing simple strategy outputs against Zipline and Backtrader.
* **NFR-5: Operability (Should)**
    * Use structured logging (e.g., structlog).
    * Provide a debug\_trace: true config flag that outputs a complete, timestamped event log for debugging order
      lifecycles.
* **NFR-6: Portability (Must)**
    * Must run on Python 3.12+ on Linux, macOS, and Windows.
    * Core dependencies are Polars, Numba, Pydantic.

---

## **7\) Detailed Interface Sketches (Illustrative)**

### **1\. Event Types**

Python

import polars as pl  
from dataclasses import dataclass  
from datetime import datetime

@dataclass(frozen=True)  
class ClockEvent:  
"""Event indicating a new simulation time step."""  
timestamp: datetime  
session\_open: bool  
session\_close: bool

@dataclass(frozen=True)  
class MarketDataEvent:  
"""Event carrying new market data for a single asset."""  
timestamp: datetime  
asset\_id: int  
bar: pl.Struct \# e.g., {'open': 100.0, 'high': 101.0, ...}

@dataclass(frozen=True)  
class OrderRequestEvent:  
"""Event emitted by a Strategy to request a new order."""  
strategy\_id: str  
order: "Order"  \# See sketch below

@dataclass(frozen=True)  
class FillEvent:  
"""Event emitted by the Broker indicating a trade execution."""  
timestamp: datetime  
order\_id: str  
fill\_id: str  
asset\_id: int  
qty: float  
price: float  
commission: float  
exchange\_fee: float

### **2\. Strategy API**

Python

from helios.engine import StrategyBase, DataContext  
from helios.events import ClockEvent, MarketDataEvent  
from helios.orders import Order

class MyStrategy(StrategyBase):  
"""User-defined strategy class."""

    def on\_init(self, data\_context: DataContext, state: dict):  
        """Called once at the start, before the warm-up period."""  
        self.data \= data\_context  
        self.state \= state  \# Persistent state dict  
        self.state.setdefault("sma\_50", {})

    def on\_clock(self, event: ClockEvent):  
        """Called on every master clock tick."""  
        if event.session\_close:  
            \# End of day logic  
            return

        \# Get 50-day rolling data for all assets in our universe  
        \# This access is point-in-time correct.  
        history \= self.data.history(  
            assets=self.data.universe,  
            fields=\["close"\],  
            lookback=50  
        )

        \# \`history\` is a Polars DataFrame: \[asset, time, close\]  
        for asset, data in history.group\_by("asset\_id"):  
            sma \= data\["close"\].mean()  
            last\_close \= data\["close"\].last()  
              
            if last\_close \> sma and self.portfolio.get\_position(asset) \== 0:  
                self.order\_target\_percent(asset, 0.05, comment="Buy signal")  
            elif last\_close \< sma and self.portfolio.get\_position(asset) \> 0:  
                self.close\_position(asset, comment="Sell signal")

### **3\. Order/Execution Objects**

Python

from enum import Enum

class OrderStatus(Enum):  
PENDING \= 0  
OPEN \= 1  
FILLED \= 2  
PARTIALLY\_FILLED \= 3  
CANCELED \= 4  
REJECTED \= 5

@dataclass  
class Order:  
"""Internal representation of an order."""  
asset\_id: int  
qty: float \# Positive for buy, negative for sell  
limit\_price: float | None \= None  
stop\_price: float | None \= None  
tif: str \= "DAY"  
reduce\_only: bool \= False

    \# Internal state  
    order\_id: str \= "ord\_..."  
    status: OrderStatus \= OrderStatus.PENDING  
    filled\_qty: float \= 0.0

### **4\. Config Schemas (Pydantic \+ YAML)**

Python

from pydantic import BaseModel, FilePath, field\_validator  
from typing import Literal

class DataConfig(BaseModel):  
provider: Literal\["parquet"\]  
path: FilePath  
calendar: str \# e.g., "NYSE"  
asset\_class: Literal\["equity", "future", "crypto"\]  
adjustments: Literal\["raw", "split\_adjusted", "full\_adjusted"\] \= "full\_adjusted"

class BrokerConfig(BaseModel):  
slippage\_model: dict \= {"type": "BpsSlippage", "bps": 5}  
cost\_model: dict \= {"type": "PerTradeCost", "cost": 0.50}  
shorting: bool \= True  
borrow\_fee\_bps: float \= 0.0

class StrategyConfig(BaseModel):  
module: str \# e.g., "my\_strategies.sma"  
class\_name: str \# e.g., "MyStrategy"  
warmup\_periods: int \= 50  
params: dict \= {}

class Config(BaseModel):  
"""Top-level configuration schema."""  
run\_id: str  
start\_date: datetime  
end\_date: datetime  
base\_currency: str \= "USD"  
seed: int \= 42

    data: DataConfig  
    strategy: StrategyConfig  
    broker: BrokerConfig  
      
    @field\_validator("end\_date")  
    def dates\_must\_be\_ordered(cls, v, info):  
        if v \<= info.data\["start\_date"\]:  
            raise ValueError("end\_date must be after start\_date")  
        return v

\# \--- Corresponding YAML example (see Appendix A) \---

### **5\. Results Object**

Python

import polars as pl

class BacktestResult:  
"""Standardized output object."""  
def \_\_init\_\_(self, manifest: dict, metrics: dict, artifacts: dict\[str, pl.DataFrame\]):  
self.manifest: dict \= manifest \# The RunManifest.json  
self.metrics: dict \= metrics \# Summary: {'sharpe': 1.5, ...}

        \# Core data artifacts  
        self.trades: pl.DataFrame \= artifacts\["trades"\]  
        self.positions: pl.DataFrame \= artifacts\["positions"\]  
        self.equity\_curve: pl.DataFrame \= artifacts\["equity\_curve"\]  
        self.returns: pl.DataFrame \= artifacts\["returns"\]  
        self.orders: pl.DataFrame \= artifacts\["orders"\]

    def save(self, path: str):  
        """Saves the result as a directory of Parquet files \+ JSON."""  
        ...  
      
    @staticmethod  
    def load(path: str) \-\> "BacktestResult":  
        """Loads a previously saved result."""  
        ...

---

## **8\) Time & Event Semantics (Formalization)**

* **Global Clock:** The simulation is driven by a master clock, which advances to the timestamp of the next event in the
  global priority queue.
* **Total Ordering:** The event queue is a min-heap of tuples: (timestamp, priority, event\_id).
    * timestamp: The simulation time at which the event occurs.
    * priority: A deterministic integer defining processing order *at the same timestamp*.
        * 0: ClockEvent (Start of new bar)
        * 10: MarketDataEvent (Market data for new bar is published)
        * 20: SignalEvent (External signals are processed)
        * 30: Strategy.on\_event() (Strategy logic runs, OrderRequestEvents are created)
        * 40: RiskCheckEvent (Pre-trade risk checks run)
        * 50: Broker.match() (Broker matches orders, FillEvents are created)
        * 60: Portfolio.update() (Portfolio updates on fills)
    * event\_id: A monotonically increasing counter (tie-breaker) to ensure stable sorting for events with the same
      timestamp and priority.
* **Bar Semantics:** Time bars represent an interval \[start, end). A bar with timestamp \= 09:31:00 represents all
  market activity from 09:30:00.000 to 09:30:59.999...
* **Execution Model:** By default, all strategy logic on\_bar(bar) runs *after* the MarketDataEvent for that bar has
  been published. Orders generated from that logic are placed *at* the bar's timestamp and are eligible for matching
  against *that same bar's* data (e.g., fill at open or close) or the *next* bar's data, based on configuration (
  execute\_on\_same\_bar: bool).
* **Latency:** Latency (data, decision, execution) will be modeled via configuration, which can delay the timestamp of
  events placed into the queue.
* **Signal TTL (Time-to-Live):** A SignalEvent can have a ttl (e.g., '60s'). The strategy layer is responsible for
  tracking its validity.

---

## **9\) Pricing & Fill Rules (Bar-Aware)**

The Broker's matching engine must be deterministic and configurable.

* **Rule 1: Market Orders:**
    * policy: "at\_open": Fills at the open price of the *next* bar (if execute\_on\_same\_bar=False) or the *current*
      bar (if True).
    * policy: "at\_close": Fills at the close price of the current bar.
* **Rule 2: Limit Orders:**
    * Limit buys are eligible if bar.low \<= limit\_price.
    * Limit sells are eligible if bar.high \>= limit\_price.
    * Fill price is determined by config:
        * policy: "at\_limit": Fills at limit\_price (optimistic).
        * policy: "at\_open\_or\_limit": Fills at max(limit\_price, bar.open) for buys, min(limit\_price, bar.open) for
          sells, assuming price gapped through the limit.
* **Rule 3: Stop Orders:**
    * Stop buys are triggered if bar.high \>= stop\_price.
    * Stop sells are triggered if bar.low \<= stop\_price.
    * Once triggered, they become Market orders and follow Rule 1\.
* **Rule 4: Gaps:** If a market gaps (e.g., overnight), orders are evaluated against the open price. A DAY limit order
  placed yesterday with a limit\_price inside today's gap will *not* fill.
* **Rule 5: Priority:** On the same bar, the configurable priority is: 1\) Stops, 2\) Limits, 3\) Market.

---

## **10\) Slippage, Costs, and Market Impact**

These are pluggable Broker components.

* **CostModel Interface:**
    * calculate(trade: FillEvent) \-\> float
    * *Implementations:* PerTradeCost, PerShareCost, BpsCost (Tiered), MakerTakerCost (for crypto).
* **SlippageModel Interface:**
    * apply(order: Order, bar: MarketDataEvent) \-\> float (returns fill price)
    * *Implementations:*
        * FixedBpsSlippage: Simple price \* (1 \+ bps).
        * VolumeParticipationSlippage: price \= f(order\_size / bar\_volume).
        * SquareRootImpact: price\_impact \= f(volatility \* sqrt(order\_size / daily\_volume)).
* **Capacity Analysis:** The BacktestResult's trades DataFrame, which includes bar\_volume and trade\_pct\_of\_volume,
  will be the primary input for downstream capacity and crowding analysis.

---

## **11\) Portfolio Construction & Optimization Interfaces**

* **Sizing Hooks:** The StrategyBase will provide simple sizing methods:
    * self.order(asset, qty)
    * self.order\_target\_notional(asset, notional)
    * self.order\_target\_percent(asset, pct\_equity)
* **Centralized Allocator (v1):**
    * A strategy can optionally emit SignalEvents (e.g., {asset: 'AAPL', score: 0.8}) instead of OrderRequestEvents.
    * A pluggable PortfolioAllocator (e.g., MarkowitzMVO, RiskParityAllocator, HRPAllocator) subscribes to these signals
      and a ClockEvent (e.g., 'EOD') and runs an optimization to generate the target portfolio, emitting
      OrderRequestEvents.
* **Multi-Strategy (v2):** A top-level PortfolioManager will allocate capital between different Strategy objects and
  resolve order conflicts (e.g., Strategy A wants to buy AAPL, Strategy B wants to sell).

---

## **12\) Asset Class Specifics**

The engine will be asset-class-aware via the DataConfig and InstrumentRegistry.

* **Equities (v0):**
    * *Behavior:* T+N settlement, shorting (requires borrow), dividends, splits.
    * *Metadata:* lot\_size, currency.
* **Futures (v1):**
    * *Behavior:* Daily MTM P\&L, margin calls, expiry, automatic rolling (configurable).
    * *Metadata:* multiplier, expiry\_date, tick\_size.
* **FX (v1):**
    * *Behavior:* Traded in pairs, P\&L in base currency, overnight roll/interest.
    * *Metadata:* base\_currency, quote\_currency.
* **Crypto (v1):**
    * *Behavior:* 24/7 calendar, maker/taker fees, funding rates (for perps).
    * *Metadata:* lot\_size, price\_precision.
* **Future Scope (Options, v2+):** To support options, the MarketDataEvent would need to carry chain/surface data, and
  the Portfolio layer would need a greeks/valuation model. This is a significant extension.

---

## **13\) Universe & Corporate Actions Management**

* **Dynamic Universe (v1):** The DataContext will support dynamic universes (e.g., S\&P 500 membership). The strategy's
  self.data.universe property will return the correct list of assets for that simulation day.
* **Corporate Actions (v1):**
    * Corp action events (split, dividend) will be ingested as a separate data stream.
    * They will be placed on the event bus with high priority.
    * The Portfolio layer will subscribe to them and adjust positions/cash *before* the next MarketDataEvent.
    * The DataContext will use them to apply adjustment ratios to historical price data requests, ensuring a consistent,
      point-in-time view.

---

## **14\) Testing & Validation Strategy**

* **Unit Tests:** All components (e.g., CostModel, FillEngine) will have 100% unit test coverage.
* **Property-Based Tests:** Use hypothesis to test the Portfolio layer's accounting invariants. (e.g., for any
  list\_of\_fills, sum(fills.pnl) \== final\_equity \- initial\_cash).
* **Golden Tests:**
    * Create simple, self-contained backtests (e.g., "buy-and-hold", "SMA crossover") with *all* friction disabled.
    * Run the same logic in Zipline and Backtrader.
    * Assert that Helios produces the identical equity curve and trade log.
* **Benchmark Suite:** A pytest-benchmark suite will run on every PR to monitor the performance NFRs (e.g.,
  test\_sma\_10y\_1m\_benchmark) and detect regressions.

---

## **15\) Observability, Diagnostics & Developer Experience**

* **Clear Exceptions (Must):** All internal exceptions will be caught and re-raised as user-friendly HeliosErrors.
    * ConfigValidationError (from Pydantic)
    * LookAheadError (from DataContext)
    * InsufficientCashError (from Broker)
* **Event Trace (Should):** A debug: true config will serialize the *entire* event bus (every event) to a trace.parquet
  file. This allows developers to see the exact sequence of events that led to a bug.
* **Cookbook & Examples (Must):** The documentation must include a "Cookbook" of common patterns (e.g., "How to use ML
  predictions", "How to build a custom slippage model").

---

## **16\) Reproducibility & Governance**

* **Run Manifest (Must):** This is the core governance artifact. See FR-5.6.2. It is the "source of truth" for any
  backtest result.
* **Content-Addressing (Should):** Input data references in the manifest should ideally be content-addressed (e.g.,
  sha256:abc...) rather than path-based (/data/my\_data.parquet) to guarantee immutability.
* **Semantic Versioning (Must):** The engine, its plugin interfaces, and its output schemas (Appendix C) will follow
  SemVer 2.0.

---

## **17\) Roadmap & Milestones**

* **v0 (MVP) \- (Target: Q1 2026\)**
    * *Theme:* Core engine & reproducibility.
    * *Features:* Single-asset equities, time bars only, Market/Limit/Stop orders, simple cost/slippage, basic portfolio
      accounting, RunManifest for 100% reproducibility, standardized BacktestResult object (Polars/Parquet).
* **v1 (Target: Q3 2026\)**
    * *Theme:* Feature parity & extensibility.
    * *Features:* Multi-asset & multi-currency, Futures & Crypto support, corporate actions, advanced order types (TIF,
      brackets), pluggable slippage/impact models, analysis hooks for tearsheets, batch sweep runner.
* **v2 (Target: Q1 2027\)**
    * *Theme:* Advanced modeling & live-readiness.
    * *Features:* Dynamic universes, portfolio-level optimizers (MVO, HRP), advanced microstructure (intrabar fill
      logic), risk attribution, capacity modeling, optional Rust-based core loop, adapters for paper trading.

---

## **18\) Risks & Open Questions**

* **Risk: Numba JIT Compilation Overhead**
    * *Description:* Numba's first-run JIT compilation can be slow, harming the "fast iteration" UX.
    * *Mitigation:* Use cache=True for JIT functions. Provide an AOT (Ahead-of-Time) compilation script for users.
      Profile aggressively and plan for Rust-based (pyo3) replacement of the event loop/fill engine in v2 if Numba is
      insufficient.
* **Risk: Polars API Velocity**
    * *Description:* The Polars API is still evolving. An update could break internal logic.
    * *Mitigation:* Pin the Polars version strictly in pyproject.toml. Build an internal "data context" adapter around
      Polars, so strategy code does not call Polars directly, insulating users from API changes.
* **Risk: Correctness of Bar-Fill Logic**
    * *Description:* Simulating fills on OHLC bars is heuristic-based and notoriously difficult to get "right."
    * *Mitigation:* Default to the simplest, most pessimistic logic (e.g., "fill at next open"). Make all fill logic
      explicit, configurable, and heavily documented. Rely on golden tests (Sec 14\) for validation.
* **Open Question: Default Bar Matching Logic**
    * *Decision:* What is the default execute\_on\_same\_bar setting?
    * *Recommendation:* False. This is the most conservative, anti-lookahead-bias default. It assumes orders placed
      during a bar execute at the *next* bar's open. True will be an advanced, opt-in config.
* **Open Question: Schema Versioning**
    * *Decision:* How do we manage versions for the BacktestResult Parquet schemas?
    * *Recommendation:* Embed a schema\_version: "1.0" key in the RunManifest.json. The BacktestResult.load() function
      will use this to apply migration logic.

---

## **Appendix A) Example Config (YAML)**

YAML

version: "1.0"  
run\_id: "sma\_crossover\_v0\_run\_001"  
seed: 42  
base\_currency: "USD"

run\_window:  
start\_date: "2020-01-01T00:00:00Z"  
end\_date: "2024-12-31T23:59:00Z"  
warmup\_periods: 50 \# Based on strategy's 'slow\_ma'

data:  
provider: "parquet"  
calendar: "NYSE"  
asset\_class: "equity"  
path\_template: "/data/equity/minute\_bars/{asset\_id}.parquet"  
assets: \["AAPL", "MSFT"\]  
adjustments: "full\_adjusted" \# Handle splits/dividends

strategy:  
module: "strategies.sma\_cross"  
class\_name: "SmaCrossoverStrategy"  
\# These params are passed to the strategy's \_\_init\_\_  
params:  
fast\_ma: 20  
slow\_ma: 50  
sizing\_pct: 0.10 \# 10% of equity per position

broker:  
execution:  
\# Orders placed on bar 't' are matched against bar 't+1'  
execute\_on\_same\_bar: false  
fill\_policy: "at\_open" \# Market orders fill at 'open'

cost\_model:  
\# IBKR-style tiered model  
type: "TieredCommission"  
per\_share\_tiers:  
\- \[0, 200000, 0.0035\]  
\- \[200001, 1000000, 0.0020\]  
min\_per\_order: 0.35  
max\_per\_order\_pct: 0.005 \# 0.5% of trade value

slippage\_model:  
\# 2.5 bps slippage \+ 1% of bar volume impact  
type: "VolumeParticipationSlippage"  
base\_bps: 2.5  
pct\_volume\_impact: 0.01

risk:  
\# Pre-trade checks  
max\_order\_notional: 1\_000\_000  
banned\_assets: \["META"\]

\# Post-trade checks  
max\_drawdown\_pct: 0.25 \# Halts trading if hit  
max\_leverage: 2.0

outputs:  
path: "./results/{run\_id}"  
artifacts:  
\- "trades"  
\- "positions"  
\- "equity\_curve"  
\- "summary\_metrics"  
debug\_trace: false \# Set to true to log every event

\# Example of a parameter sweep  
sweep:  
\# Run 4 backtests in parallel  
enabled: true  
parallel\_jobs: 4  
search\_type: "grid"  
parameters:  
strategy.params.fast\_ma: \[10, 20\]  
strategy.params.slow\_ma: \[40, 50\]

---

## **Appendix B) Example Python Interface Sketches**

Python

from abc import ABC, abstractmethod  
from datetime import datetime  
import polars as pl  
from helios.events import OrderRequestEvent, FillEvent, MarketDataEvent  
from helios.orders import Order

class DataContext(ABC):  
"""Interface provided to the strategy for data access."""

    @property  
    @abstractmethod  
    def universe(self) \-\> list\[int\]:  
        """Gets the current list of asset IDs for this timestamp."""  
        ...

    @abstractmethod  
    def history(self, assets: list\[int\], fields: list\[str\], lookback: int) \-\> pl.DataFrame:  
        """  
        Gets point-in-time historical data.  
        Returns DataFrame \[asset\_id, timestamp, field1, field2, ...\]  
        """  
        ...  
      
    @abstractmethod  
    def get\_last\_bar(self, asset: int) \-\> pl.Struct:  
        """Gets the most recent bar data for an asset."""  
        ...

class PortfolioContext(ABC):  
"""Interface provided to the strategy for portfolio access."""

    @property  
    @abstractmethod  
    def current\_cash(self) \-\> float:  
        ...  
          
    @property  
    @abstractmethod  
    def current\_equity(self) \-\> float:  
        ...  
          
    @abstractmethod  
    def get\_position(self, asset: int) \-\> float:  
        """Returns current holding in shares."""  
        ...

class StrategyBase(ABC):  
"""Base class all user strategies must inherit from."""

    def \_\_init\_\_(self):  
        \# These are injected by the engine  
        self.data: DataContext \= None  
        self.portfolio: PortfolioContext \= None  
        self.\_event\_bus \= None \# Internal bus interface  
        self.state: dict \= {} \# User-managed state

    @abstractkey  
    def on\_init(self, data\_context: DataContext, portfolio\_context: PortfolioContext, state: dict):  
        """Called once at the start of the simulation."""  
        self.data \= data\_context  
        self.portfolio \= portfolio\_context  
        self.state \= state

    def on\_clock(self, event: ClockEvent):  
        """(Optional) Called on every master clock tick."""  
        pass  
          
    def on\_bar(self, event: MarketDataEvent):  
        """(Optional) Called on every market data event for subscribed assets."""  
        pass  
          
    \# \--- Helper methods \---  
      
    def order(self, asset: int, qty: float, \*\*kwargs) \-\> str:  
        """Places a new order. Returns order\_id."""  
        order\_obj \= Order(asset\_id=asset, qty=qty, \*\*kwargs)  
        self.\_event\_bus.put(OrderRequestEvent(strategy\_id=self.id, order=order\_obj))  
        return order\_obj.order\_id  
          
    def order\_target\_percent(self, asset: int, pct: float, \*\*kwargs):  
        """Calculates size for a target portfolio percentage."""  
        target\_notional \= self.portfolio.current\_equity \* pct  
        last\_price \= self.data.get\_last\_bar(asset)\["close"\]  
        target\_shares \= target\_notional / last\_price  
        current\_shares \= self.portfolio.get\_position(asset)  
          
        self.order(asset, qty=target\_shares \- current\_shares, \*\*kwargs)

    def close\_position(self, asset: int, \*\*kwargs):  
        """Closes any open position for an asset."""  
        qty \= self.portfolio.get\_position(asset)  
        if qty \!= 0:  
            self.order(asset, qty=-qty, \*\*kwargs)

class BrokerBase(ABC):  
"""Interface for a pluggable Broker."""

    @abstractmethod  
    def submit\_order(self, event: OrderRequestEvent):  
        """Adds an order to the internal book."""  
        ...  
          
    @abstractmethod  
    def match(self, event: MarketDataEvent):  
        """  
        Called on market data to check for fills.  
        Must emit FillEvent(s) to the event bus.  
        """  
        ...

\# \--- Order, Fill, and Result objects are sketched in Section 7 \---

---

## **Appendix C) Standard Output Schemas**

All DataFrames will be stored in Parquet format with Arrow types.

### **1\. orders.parquet**

| Column       | Type                    | Description                                          |
|:-------------|:------------------------|:-----------------------------------------------------|
| order\_id    | String                  | Unique order identifier.                             |
| ts\_created  | Timestamp(ns, tz='UTC') | Simulation time when the strategy created the order. |
| ts\_filled   | Timestamp(ns, tz='UTC') | Simulation time of final fill (or Null).             |
| ts\_canceled | Timestamp(ns, tz='UTC') | Simulation time of cancellation (or Null).           |
| asset\_id    | Categorical             | Asset identifier.                                    |
| qty\_ordered | Float64                 | Requested quantity (+ for buy, \- for sell).         |
| qty\_filled  | Float64                 | Total filled quantity.                               |
| limit\_price | Float64                 | Limit price (Null for market orders).                |
| stop\_price  | Float64                 | Stop price (Null for non-stop orders).               |
| tif          | String                  | Time in Force (e.g., 'DAY', 'GTC').                  |
| status       | String                  | Final status ('FILLED', 'CANCELED', 'REJECTED').     |

### **2\. trades.parquet**

| Column               | Type                    | Description                                           |
|:---------------------|:------------------------|:------------------------------------------------------|
| fill\_id             | String                  | Unique fill/trade identifier.                         |
| order\_id            | String                  | Parent order ID.                                      |
| ts\_fill             | Timestamp(ns, tz='UTC') | Simulation time of the fill.                          |
| asset\_id            | Categorical             | Asset identifier.                                     |
| qty                  | Float64                 | Executed quantity (+ for buy, \- for sell).           |
| price                | Float64                 | Execution price (after slippage, before commission).  |
| commission           | Float64                 | Commission paid for this fill (in base currency).     |
| slippage\_bps        | Float64                 | Slippage vs. reference price (e.g., bar.open) in bps. |
| pct\_of\_bar\_volume | Float64                 | abs(qty) / bar.volume (if available).                 |
| mfe                  | Float64                 | Max Favorable Excursion (in base currency).           |
| mae                  | Float64                 | Max Adverse Excursion (in base currency).             |

### **3\. positions.parquet**

| Column          | Type                    | Description                                     |
|:----------------|:------------------------|:------------------------------------------------|
| timestamp       | Timestamp(ns, tz='UTC') | Timestamp of the position snapshot (e.g., EOD). |
| asset\_id       | Categorical             | Asset identifier.                               |
| qty             | Float64                 | Quantity held at this timestamp.                |
| cost\_basis     | Float64                 | Average cost price per share.                   |
| market\_price   | Float64                 | Mark-to-market price at this timestamp.         |
| market\_value   | Float64                 | qty \* market\_price \* multiplier.             |
| unrealized\_pnl | Float64                 | Unrealized P\&L on the position.                |

### **4\. equity\_curve.parquet**

| Column        | Type                    | Description                          |
|:--------------|:------------------------|:-------------------------------------|
| timestamp     | Timestamp(ns, tz='UTC') | Timestamp of the snapshot.           |
| cash          | Float64                 | Cash balance in base currency.       |
| market\_value | Float64                 | Total market value of all positions. |
| equity        | Float64                 | cash \+ market\_value.               |
| realized\_pnl | Float64                 | Cumulative realized P\&L.            |

### **5\. RunManifest.json (Metadata)**

JSON

{  
"run\_id": "sma\_crossover\_v0\_run\_001",  
"engine\_version": "0.1.0",  
"schema\_version": "1.0",  
"run\_timestamp\_utc": "2025-11-10T11:43:00Z",  
"seed": 42,  
"data\_hashes": {  
"/data/equity/minute\_bars/AAPL.parquet": "sha256:abc...",  
"/data/equity/minute\_bars/MSFT.parquet": "sha256:def..."  
},  
"code\_version": {  
"git\_commit\_hash": "a1b2c3d4e5f6...",  
"strategy\_file\_hash": "sha256:789..."  
},  
"environment": {  
"python\_version": "3.12.1",  
"helios\_version": "0.1.0",  
"polars\_version": "0.20.10",  
"numba\_version": "0.59.1"  
},  
"full\_config": {  
"...": "The complete, resolved YAML config from Appendix A"  
}  
}

---

## **Appendix D) Comparison Matrix**

| Feature             | Zipline (Reloaded)       | Backtrader                      | vectorbt pro                                  | Helios (This PRD)                           |
|:--------------------|:-------------------------|:--------------------------------|:----------------------------------------------|:--------------------------------------------|
| **Core Model**      | Event-Driven             | Event-Driven                    | Vectorized                                    | **Event-Driven (Numba-JIT)**                |
| **Performance**     | Slow (Python loop)       | Slow (Python loop)              | Very Fast (NumPy/Numba)                       | **Very Fast (Numba/Polars)**                |
| **Stateful Logic**  | Yes (Native)             | Yes (Native)                    | Difficult (Requires complex state management) | **Yes (Native, Numba-JIT)**                 |
| **Data Model**      | pandas (via bcolz)       | list / array                    | numpy / pandas                                | **Polars / Arrow (Native)**                 |
| **UX / API**        | Cumbersome (Bundles)     | Verbose, "batteries-included"   | pandas-like, concise                          | **Config-first, clean Python API**          |
| **Microstructure**  | Basic (bar fills)        | Basic (bar fills)               | None (vectorized)                             | **Pluggable (v1+ advanced)**                |
| **Reproducibility** | Poor (Data bundles, env) | Fair (Requires user discipline) | Good (Vectorized is deterministic)            | **Excellent (Core design goal)**            |
| **Extensibility**   | Difficult (Monolithic)   | Good (Strategy, Analyzer)       | Fair (Custom indicators)                      | **Excellent (Plugin-first)**                |
| **ML Anti-Leakage** | Manual, error-prone      | Manual, error-prone             | Excellent (Vectorized by nature)              | **Excellent (Core design goal)**            |
| **Accounting**      | Good (Equities, Futures) | Good (Equities, Futures)        | Basic (P\&L only)                             | **Rigorous (Ledger-based, v1 Multi-asset)** |

---

## **Appendix E) Acceptance Criteria Checklist**

### **v0 (MVP) Checklist**

* **Core Engine**
    * \[ \] run\_backtest(config) executes a run from a YAML config.
    * \[ \] Event loop correctly processes events in (timestamp, priority, id) order.
    * \[ \] Numba JIT is applied to the event loop and portfolio updates.
* **Reproducibility**
    * \[ \] RunManifest.json is created for every run.
    * \[ \] Running a backtest from a RunManifest.json produces bit-identical BacktestResult artifacts.
    * \[ \] All randomness is controlled by the global seed.
* **Data & Assets**
    * \[ \] Engine ingests 1-minute Parquet bar data for a *single equity*.
    * \[ \] NYSE calendar is correctly applied (session times, holidays).
* **Strategy & Orders**
    * \[ \] StrategyBase class can be subclassed.
    * \[ \] on\_bar and on\_clock methods are called correctly.
    * \[ \] self.data.history() provides point-in-time correct data.
    * \[ \] self.order() successfully creates Market and Limit orders.
    * \[ \] TIF='DAY' orders are correctly canceled at session close.
* **Broker & Portfolio**
    * \[ \] Bar-fill logic (e.g., at\_open) is applied deterministically.
    * \[ \] Simple BpsSlippage and PerTradeCost models are applied.
    * \[ \] Portfolio layer correctly tracks cash and position for one asset.
    * \[ \] Realized/Unrealized P\&L is calculated.
* **Outputs**
    * \[ \] BacktestResult object is returned.
    * \[ \] trades, orders, positions, equity\_curve DataFrames are generated per Appendix C schema.
    * \[ \] result.save() and result.load() function correctly.
* **Testing & Docs**
    * \[ \] Core accounting logic passes property-based tests.
    * \[ \] Golden tests for "buy-and-hold" and "SMA-cross" match reference output.
    * \[ \] All public APIs (Config, StrategyBase, Result) are documented.
