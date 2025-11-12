# **Product Requirements Document: Event-Driven Backtesting Engine**

## **0\) Executive Summary**

The Event-Driven Backtesting Engine (EDBE) is a next-generation market simulation platform designed to replace legacy
Zipline-based workflows while delivering deterministic, reproducible backtests with superior performance, modularity,
and user experience. Built on modern Python foundations (Polars/Arrow for data, Numba for compute-intensive paths), EDBE
provides a plugin-based architecture supporting multi-asset strategies, sophisticated execution modeling, and ML-driven
workflows with strict anti-leakage guarantees.

**Key Design Principles:**

* **Determinism**: Bit-for-bit reproducible results under fixed configuration
* **Reproducibility**: Complete run manifests with content-addressed inputs
* **Modularity**: Plugin architecture for brokers, cost models, and analytics
* **Performance**: Vectorized operations via Polars/Numba, targeting 1M+ events/second
* **UX Clarity**: Config-first workflows, comprehensive error messages, introspection tools

**Release Scope:**

* **v0 (MVP)**: Single-asset equities, OHLCV bars, market/limit orders, basic accounting
* **v1**: Multi-asset/currency, full order types, slippage models, ML integration
* **v2**: Options support, microstructure models, capacity analysis, live adapters

## **1\) Context & Goals**

### **Why a New Engine**

Zipline's architecture, while pioneering, suffers from:

* **Maintenance burden**: Cython codebase requiring specialized expertise
* **Dead dependencies**: bcolz, empyrical no longer maintained
* **Poor modularity**: Monolithic design resists extension
* **UX friction**: Obscure errors, limited introspection, complex setup
* **Performance ceiling**: Single-threaded event loop, inefficient data structures

### **Primary Use Cases**

1. **Event-driven backtests** with stateful strategies maintaining complex internal state
2. **ML-driven strategies** ingesting features/predictions with temporal alignment guarantees
3. **Flexible execution modeling** supporting realistic broker behaviors and market microstructure
4. **Multi-asset portfolios** with cross-asset dependencies and dynamic universe management
5. **Cross-frequency strategies** mixing signals from multiple timeframes

### **Non-Goals (v0/v1)**

* Full options pricing/greeks engine (v2+)
* Real-time paper trading connectivity (v2+)
* Microsecond-resolution order book simulation
* Built-in alpha generation/feature engineering
* Live broker connectivity

### **Success Metrics**

| Metric            | Target                                          | Priority |
|-------------------|-------------------------------------------------|----------|
| Throughput        | 1M+ events/second on 8-core machine             | Must     |
| Determinism       | 100% reproducible with fixed seeds              | Must     |
| Test Coverage     | \>95% line coverage, 100% critical paths        | Must     |
| Documentation     | 100% public API documented                      | Must     |
| Memory efficiency | \<10GB for 10-year daily backtest, 1000 symbols | Should   |
| Startup time      | \<1s for standard config                        | Should   |

**Assumptions:**

* Python 3.12+ adoption acceptable
* Users willing to migrate from Zipline given sufficient tooling
* Polars/Arrow ecosystem mature enough for production

## **2\) Users & Personas**

### **Primary Personas**

**Quant Researcher (Primary)**

* Needs: Rapid iteration, parameter sweeps, factor analysis
* Pain points: Slow backtests, debugging black boxes, result comparison
* Success: 10x faster experiment velocity

**ML Engineer**

* Needs: Feature/label alignment, anti-leakage guarantees, model integration
* Pain points: Data leakage bugs, pipeline complexity
* Success: Seamless sklearn/pytorch integration

**Portfolio Manager**

* Needs: Risk controls, realistic execution, multi-strategy allocation
* Pain points: Unrealistic fills, missing corporate actions
* Success: Production-ready signal evaluation

### **Desired UX**

\# Complete backtest in 5 lines of YAML \+ strategy code

data:

source: parquet

path: /data/equities/daily/\*.parquet

strategy:

module: strategies.momentum

params:

    lookback: 20

broker:

initial\_capital: 100000

commission: 0.001

**Design Principles:**

* Config-first, code-second
* Fail fast with actionable errors
* Progressive disclosure of complexity
* Excellent defaults, infinite customization

## **3\) Core Product Principles**

### **Deterministic & Auditable**

* Seeded PRNG for all randomness (NumPy Generator with PCG64)
* Stable sort ordering for simultaneous events
* Content-addressed input hashing
* **Rationale**: Reproducibility essential for research credibility
* **Priority**: Must

### **Config-First, Code-Second**

* YAML/JSON configuration with Pydantic validation
* Strategy DSL for common patterns
* Python API for complex logic
* **Rationale**: Lower barrier to entry, easier experiment management
* **Priority**: Must

### **Event-Driven with First-Class Time**

* Unified event bus with typed events
* Multi-clock synchronization
* Configurable time semantics (interval endpoints, bar close timing)
* **Rationale**: Flexibility for diverse data sources and strategies
* **Priority**: Must

### **Modular Plugin Architecture**

* Registry-based discovery
* Versioned interfaces
* Hot-swappable components
* **Rationale**: Extensibility without core changes
* **Priority**: Should

### **Performance via Modern Python**

* Polars for columnar data operations
* Numba JIT for hot paths
* Arrow for zero-copy interchange
* Future Rust extensions via PyO3
* **Rationale**: 100x performance vs pure Python
* **Priority**: Must for v1

### **Interoperability**

* Standardized output formats (Parquet/Arrow)
* Stable schemas with semantic versioning
* Direct integration with analysis libraries
* **Rationale**: Ecosystem compatibility
* **Priority**: Must

## **4\) Architecture Overview**

### **System Architecture**

┌─────────────────────────────────────────────────────────────┐

│ Configuration Layer │

│                   (YAML/JSON \+ Pydantic)                     │

└─────────────────────────────────────────────────────────────┘

                               │

┌─────────────────────────────────────────────────────────────┐

│ Data Layer │

│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │

│ │ Parquet │ │ Arrow │ │ CSV │ │ Polars │ │

│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │

│ ┌──────────────────────────────────────────────────┐ │

│ │ Schema Registry & Validation │ │

│ └──────────────────────────────────────────────────┘ │

└─────────────────────────────────────────────────────────────┘

                               │

┌─────────────────────────────────────────────────────────────┐

│ Event Fabric │

│ ┌──────────────────────────────────────────────────┐ │

│ │ Deterministic Event Queue │ │

│ │    (Total ordering, stable tie-breaks)           │ │

│ └──────────────────────────────────────────────────┘ │

│ MarketData | Signal | Order | Fill | Portfolio | Risk │

└─────────────────────────────────────────────────────────────┘

                               │

         ┌────────────┬────────┴────────┬──────────┐

         ▼            ▼                 ▼          ▼

┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐

│ Strategy │ │ Broker │ │ Portfolio │ │ Risk │

│ Layer │ │ Execution │ │ Accounting │ │Controls │

└─────────────┘ └─────────────┘ └─────────────┘ └─────────┘

                               │

┌─────────────────────────────────────────────────────────────┐

│ Outputs & Analytics │

│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │

│ │ Results │ │Tearsheet │ │ Factor │ │ Artifacts│ │

│ │ Object │ │ Reports │ │ Analysis │ │ Registry │ │

│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │

└─────────────────────────────────────────────────────────────┘

### **Data Flow Narrative**

1. **Ingestion**: Data sources stream time-indexed events through validators into the schema registry
2. **Normalization**: Corporate actions processor adjusts prices; FX converter handles currency
3. **Event Generation**: Clock ticks trigger market data events, which flow to strategies
4. **Signal Generation**: Strategies consume events, maintain state, emit signals
5. **Order Routing**: Signals convert to orders via sizing logic, enter broker queue
6. **Execution**: Broker matches orders against market data, generates fills
7. **Accounting**: Portfolio updates positions/cash, calculates P\&L
8. **Risk Checks**: Pre/post-trade controls validate limits
9. **Output**: Results accumulate into standardized artifacts

## **5\) Functional Requirements**

### **5.1 Data Ingestion & Time Semantics**

#### **Supported Inputs**

* **Time bars**: 1min, 5min, hourly, daily (OHLCV \+ volume)
* **Volume/dollar bars**: Fixed notional intervals
* **Tick data**: Trade-by-trade with microsecond timestamps
* **Irregular streams**: Corporate actions, signals, predictions

**Priority**: Must (time bars), Should (volume bars), Could (tick data)

#### **Time Alignment**

@dataclass

class ClockPolicy:

    primary\_clock: str \= "market"  \# market, signal, wallclock

    alignment: Literal\["floor", "ceil", "nearest"\] \= "floor"

    missing\_data: Literal\["forward\_fill", "drop", "halt"\] \= "forward\_fill"

    duplicate\_handling: Literal\["keep\_last", "keep\_first", "average"\] \= "keep\_last"

    timezone: str \= "America/New\_York"

**Acceptance Test**: Verify deterministic behavior with mixed-frequency data

#### **Signal TTL**

@dataclass

class SignalTiming:

    computed\_at: pd.Timestamp

    valid\_until: pd.Timestamp  

    execution\_delay: pd.Timedelta \= pd.Timedelta(seconds=0)

**Priority**: Must

#### **Exchange Calendars**

* NYSE, NASDAQ, CME, major crypto exchanges
* Holiday handling, early closes, trading halts
* **Implementation**: Use `exchange_calendars` library
* **Priority**: Must

#### **Corporate Actions**

@dataclass

class CorporateAction:

    symbol: str

    ex\_date: pd.Timestamp

    action\_type: Literal\["split", "dividend", "spin\_off", "symbol\_change"\]

    ratio: float  \# For splits

    amount: float  \# For dividends

    adjustment\_method: Literal\["backwards", "forwards", "none"\] \= "backwards"

**Priority**: Must (splits/dividends), Should (symbol changes)

#### **Data Quality**

* Validation rules via Pandera schemas
* Spike detection (\>10 sigma moves)
* Gap filling policies
* Audit trail of modifications

**Priority**: Should

**Open Questions:**

* How to handle tick data aggregation efficiently?
* Support for alt data with irregular schemas?

### **5.2 Strategy Definition API**

#### **Python API**

class Strategy(ABC):

    def \_\_init\_\_(self, context: StrategyContext):

        self.context \= context

        self.state \= {}  \# User-defined state

        

    @abstractmethod

    def on\_start(self) \-\> None:

        """Initialize strategy state"""

        

    @abstractmethod

    def on\_bar(self, bar: MarketBar) \-\> None:

        """Process market data"""

        

    def on\_event(self, event: Event) \-\> None:

        """Handle generic events"""

        

    def get\_historical(

        self, 

        symbols: list\[str\], 

        fields: list\[str\], 

        bars: int

    ) \-\> pl.DataFrame:

        """Access rolling window of data"""

        return self.context.get\_historical(symbols, fields, bars)

        

    def order(

        self,

        symbol: str,

        quantity: float,

        order\_type: OrderType \= OrderType.MARKET

    ) \-\> OrderId:

        """Submit order"""

        return self.context.submit\_order(...)

**Priority**: Must

#### **Config/DSL Strategy**

strategy:

type: momentum

universe:

    static: \["AAPL", "GOOGL", "MSFT"\]

signals:

    \- name: momentum\_score

      formula: (close \- close.shift(20)) / close.shift(20)

      window: 20

rules:

    \- condition: momentum\_score \> 0.05

      action: 

        type: order\_target\_percent

        weight: 0.1

    \- condition: momentum\_score \< \-0.05  

      action:

        type: order\_target\_percent

        weight: 0

**Priority**: Should

#### **ML Integration**

class MLStrategy(Strategy):

    def \_\_init\_\_(self, context: StrategyContext, model\_path: str):

        super().\_\_init\_\_(context)

        self.model \= joblib.load(model\_path)

        self.feature\_window \= 60

        

    def on\_bar(self, bar: MarketBar) \-\> None:

        \# Strict point-in-time feature computation

        features \= self.compute\_features(

            self.get\_historical(

                symbols=bar.symbol,

                fields=\["close", "volume"\],

                bars=self.feature\_window

            )

        )

        

        \# Model prediction with anti-leakage guarantee

        with self.context.no\_future\_data():

            prediction \= self.model.predict(features)

            

        if prediction \> self.threshold:

            self.order(bar.symbol, 100\)

**Priority**: Must

**Acceptance Tests:**

* Strategy can maintain state across bars
* Historical data access respects point-in-time
* ML predictions cannot access future data

### **5.3 Orders, Fills, and Execution Model**

#### **Order Types**

@dataclass

class Order:

    order\_id: str

    symbol: str

    quantity: float  \# Negative for sell/short

    order\_type: OrderType

    limit\_price: Optional\[float\] \= None

    stop\_price: Optional\[float\] \= None

    time\_in\_force: TimeInForce \= TimeInForce.DAY

    submitted\_at: pd.Timestamp

    metadata: dict \= field(default\_factory=dict)

class OrderType(Enum):

    MARKET \= "market"

    LIMIT \= "limit"  

    STOP \= "stop"

    STOP\_LIMIT \= "stop\_limit"

    TRAILING\_STOP \= "trailing\_stop"

class TimeInForce(Enum):

    DAY \= "day"

    GTC \= "good\_till\_cancel"

    IOC \= "immediate\_or\_cancel"

    FOK \= "fill\_or\_kill"

**Priority**: Must (market/limit), Should (stop/stop\_limit), Could (trailing/bracket)

#### **Matching Engine**

@dataclass

class FillPolicy:

    """Determines execution price within OHLC bar"""

    market\_order\_fill: Literal\["open", "close", "midpoint", "vwap\_estimate"\] \= "close"

    limit\_touch\_rule: Literal\["pessimistic", "optimistic", "midpoint"\] \= "pessimistic"

    stop\_trigger: Literal\["close", "high\_low", "any\_print"\] \= "close"

    same\_bar\_execution: bool \= True  \# Can orders fill on submission bar?

    partial\_fills: bool \= True

    priority: list\[OrderType\] \= field(default\_factory=lambda: \[

        OrderType.STOP,  \# Stops first (risk management)

        OrderType.LIMIT,

        OrderType.MARKET

    \])

**Acceptance Test**: Deterministic fill prices across identical runs

#### **Slippage Models**

class SlippageModel(ABC):

    @abstractmethod

    def apply(self, order: Order, fill\_price: float, volume: float) \-\> float:

        """Return adjusted fill price"""

class FixedSlippage(SlippageModel):

    def \_\_init\_\_(self, spread\_bps: float \= 10):

        self.spread\_bps \= spread\_bps

        

    def apply(self, order: Order, fill\_price: float, volume: float) \-\> float:

        slippage \= fill\_price \* self.spread\_bps / 10000

        return fill\_price \+ slippage if order.quantity \> 0 else fill\_price \- slippage

class VolumeSlippage(SlippageModel):

    def \_\_init\_\_(self, impact: float \= 0.1):

        self.impact \= impact

        

    def apply(self, order: Order, fill\_price: float, volume: float) \-\> float:

        participation \= abs(order.quantity) / volume

        impact \= self.impact \* participation \*\* 0.5

        return fill\_price \* (1 \+ impact) if order.quantity \> 0 else fill\_price \* (1 \- impact)

**Priority**: Must

### **5.4 Portfolio, Cash & Accounting**

#### **Ledgers**

@dataclass

class PortfolioState:

    timestamp: pd.Timestamp

    positions: dict\[str, Position\]

    cash: float

    total\_value: float

    leverage: float

@dataclass

class Position:

    symbol: str

    quantity: float

    cost\_basis: float

    current\_price: float

    unrealized\_pnl: float

    realized\_pnl: float

@dataclass

class Transaction:

    timestamp: pd.Timestamp

    order\_id: str

    symbol: str

    quantity: float

    price: float

    commission: float

    slippage: float

    side: Literal\["buy", "sell", "short", "cover"\]

**Priority**: Must

#### **P\&L Calculation**

* FIFO/LIFO/Average cost basis methods
* Mark-to-market at configurable frequency
* Separate realized/unrealized tracking

**Priority**: Must

#### **Multi-Currency**

class CurrencyConverter:

    def \_\_init\_\_(self, base\_currency: str \= "USD"):

        self.base \= base\_currency

        self.rates \= {}  \# Time series of FX rates

        

    def convert(self, amount: float, from\_ccy: str, to\_ccy: str, timestamp: pd.Timestamp) \-\> float:

        if from\_ccy \== to\_ccy:

            return amount

        rate \= self.get\_rate(from\_ccy, to\_ccy, timestamp)

        return amount \* rate

**Priority**: Should (v1)

### **5.5 Risk Management & Compliance**

#### **Pre-Trade Checks**

@dataclass

class RiskLimits:

    max\_position\_size: float \= 0.1  \# As % of portfolio

    max\_leverage: float \= 2.0

    max\_sector\_exposure: float \= 0.3

    banned\_symbols: set\[str\] \= field(default\_factory=set)

class PreTradeRiskCheck:

    def validate(self, order: Order, portfolio: PortfolioState) \-\> tuple\[bool, str\]:

        \# Check position limits

        if self.would\_exceed\_position\_limit(order, portfolio):

            return False, f"Would exceed position limit of {self.limits.max\_position\_size}"

        \# Check leverage

        if self.would\_exceed\_leverage(order, portfolio):

            return False, f"Would exceed leverage limit of {self.limits.max\_leverage}"

        return True, "OK"

**Priority**: Must

#### **Post-Trade Controls**

class DrawdownMonitor:

    def \_\_init\_\_(self, max\_drawdown: float \= 0.2):

        self.max\_drawdown \= max\_drawdown

        self.high\_water\_mark \= 0

        

    def check(self, portfolio\_value: float) \-\> tuple\[bool, float\]:

        self.high\_water\_mark \= max(self.high\_water\_mark, portfolio\_value)

        drawdown \= (self.high\_water\_mark \- portfolio\_value) / self.high\_water\_mark

        return drawdown \<= self.max\_drawdown, drawdown

**Priority**: Should

### **5.6 Experimentation, Reproducibility & Configuration**

#### **Configuration Schema**

class BacktestConfig(BaseModel):

    """Complete backtest configuration"""

    

    class DataConfig(BaseModel):

        source: Literal\["parquet", "csv", "arrow"\]

        path: str

        schema: Optional\[dict\] \= None

        calendar: str \= "NYSE"

        

    class StrategyConfig(BaseModel):

        module: str  \# Python module path

        class\_name: str \= "Strategy"

        params: dict \= {}

        universe: Optional\[list\[str\]\] \= None

        

    class BrokerConfig(BaseModel):

        initial\_capital: float \= 100000

        commission: float \= 0.001

        min\_commission: float \= 1.0

        slippage\_model: str \= "fixed"

        slippage\_params: dict \= {"spread\_bps": 10}

        

    class RiskConfig(BaseModel):

        pre\_trade\_checks: bool \= True

        max\_position\_size: float \= 0.1

        max\_leverage: float \= 2.0

        max\_drawdown: Optional\[float\] \= 0.3

        

    data: DataConfig

    strategy: StrategyConfig

    broker: BrokerConfig \= BrokerConfig()

    risk: RiskConfig \= RiskConfig()

    

    \# Reproducibility

    random\_seed: int \= 42

    start\_date: Optional\[str\] \= None

    end\_date: Optional\[str\] \= None

    

    \# Output

    output\_dir: str \= "./results"

    save\_trades: bool \= True

    save\_positions: bool \= True

**Priority**: Must

#### **Run Manifest**

@dataclass

class RunManifest:

    run\_id: str  \# UUID

    timestamp: pd.Timestamp

    config\_hash: str  \# SHA256 of config

    data\_hash: str  \# SHA256 of input data

    code\_version: str  \# Git commit or version

    environment: dict  \# Python version, packages

    random\_seeds: dict  \# All PRNG seeds used

    

    def save(self, path: Path) \-\> None:

        with open(path / "manifest.json", "w") as f:

            json.dump(asdict(self), f, indent=2, default=str)

**Priority**: Must

### **5.7 Outputs & Analytics**

#### **BacktestResult Object**

@dataclass

class BacktestResult:

    \# Time series

    equity\_curve: pl.DataFrame  \# timestamp, value

    returns: pl.DataFrame  \# timestamp, daily/period returns

    positions: pl.DataFrame  \# timestamp, symbol, quantity, value

    

    \# Transactions

    trades: pl.DataFrame  \# All fills with prices, costs

    orders: pl.DataFrame  \# All orders with status

    

    \# Summary metrics  

    metrics: dict\[str, float\]  \# CAGR, Sharpe, max\_dd, etc.

    

    \# Metadata

    manifest: RunManifest

    

    def to\_parquet(self, path: Path) \-\> None:

        """Save all dataframes to parquet files"""

        

    def to\_json(self) \-\> dict:

        """Export summary for API consumption"""

        

    @property

    def sharpe\_ratio(self) \-\> float:

        return self.metrics\["sharpe\_ratio"\]

        

    @property  

    def max\_drawdown(self) \-\> float:

        return self.metrics\["max\_drawdown"\]

**Priority**: Must

## **6\) Non-Functional Requirements**

### **Performance**

| Metric                | Target                     | Test Method                              |
|-----------------------|----------------------------|------------------------------------------|
| Event throughput      | \>1M events/sec            | Benchmark with 1000 symbols, minute bars |
| Memory usage          | \<10GB for 10yr daily      | Memory profiler on standard dataset      |
| Startup time          | \<1s                       | Time from config load to first event     |
| Sweep parallelization | Linear scaling to 32 cores | Parameter sweep benchmark                |

**Implementation**:

* Hot path identification via profiling
* Numba JIT compilation for critical loops
* Polars for vectorized operations

**Priority**: Must

### **Determinism**

* Bit-identical results with fixed configuration
* Stable sorting for simultaneous events
* Seeded random generators (PCG64)
* No floating-point accumulation errors

**Test**: Run identical backtest 100 times, verify SHA256 of results

**Priority**: Must

### **Extensibility**

Plugin interfaces for:

* Data sources
* Execution models
* Cost models
* Risk checks
* Output formats

**Implementation**: Abstract base classes with version tags

**Priority**: Should

### **Reliability**

* Property-based testing with Hypothesis
* Accounting invariants (positions \* price \+ cash \= portfolio value)
* Golden test suite against reference outputs
* Continuous fuzzing for edge cases

**Priority**: Must

## **7\) Detailed Interface Sketches**

### **Event Types**

from dataclasses import dataclass

from typing import Optional, Any

import pandas as pd

@dataclass(frozen=True)

class Event:

    """Base event class"""

    timestamp: pd.Timestamp

    event\_type: str

    source: str

@dataclass(frozen=True)

class MarketDataEvent(Event):

    symbol: str

    open: float

    high: float

    low: float

    close: float

    volume: float

@dataclass(frozen=True)

class SignalEvent(Event):

    symbol: str

    signal\_type: str

    value: float

    metadata: dict

@dataclass(frozen=True)

class OrderEvent(Event):

    order\_id: str

    symbol: str

    quantity: float

    order\_type: str

    limit\_price: Optional\[float\] \= None

    stop\_price: Optional\[float\] \= None

@dataclass(frozen=True)

class FillEvent(Event):

    order\_id: str

    symbol: str

    quantity: float

    price: float

    commission: float

    slippage: float

### **Strategy API**

from abc import ABC, abstractmethod

from typing import Optional

import polars as pl

class Strategy(ABC):

    """Base strategy interface"""

    

    def \_\_init\_\_(self, context: 'StrategyContext'):

        self.context \= context

        self.state \= {}

        

    @abstractmethod

    def on\_start(self) \-\> None:

        """Called once before simulation starts"""

        pass

        

    @abstractmethod

    def on\_bar(self, bar: MarketDataEvent) \-\> None:

        """Called for each market data update"""

        pass

        

    def on\_signal(self, signal: SignalEvent) \-\> None:

        """Called for each signal event"""

        pass

        

    def on\_fill(self, fill: FillEvent) \-\> None:

        """Called when order is filled"""

        pass

        

    def get\_history(

        self,

        symbols: list\[str\],

        fields: list\[str\], 

        bars: int

    ) \-\> pl.DataFrame:

        """Get rolling window of historical data"""

        return self.context.get\_history(symbols, fields, bars)

        

    def order(

        self,

        symbol: str,

        quantity: float,

        order\_type: str \= "market",

        \*\*kwargs

    ) \-\> str:

        """Submit an order"""

        return self.context.submit\_order(

            symbol=symbol,

            quantity=quantity, 

            order\_type=order\_type,

            \*\*kwargs

        )

        

    def order\_target\_percent(

        self,

        symbol: str,

        target: float

    ) \-\> Optional\[str\]:

        """Order to target percentage of portfolio"""

        return self.context.order\_target\_percent(symbol, target)

### **Configuration Schema**

\# Complete backtest configuration

data:

source: parquet

path: /data/equities/daily/\*.parquet

calendar: NYSE

adjustments: backwards \# Corporate action adjustments

strategy:

module: strategies.momentum

class\_name: MomentumStrategy

params:

    lookback\_period: 20

    rebalance\_frequency: daily

    universe\_size: 100

broker:

initial\_capital: 1000000

commission: 0.001 \# Per share

min\_commission: 1.0

slippage\_model: volume\_impact

slippage\_params:

    impact: 0.1

    gamma: 0.5

risk:

pre\_trade\_checks: true

max\_position\_size: 0.05 \# 5% of portfolio

max\_leverage: 1.5

max\_sector\_concentration: 0.3

max\_drawdown: 0.2

execution:

fill\_policy:

    market\_fill: close  \# open, close, midpoint, vwap\_estimate

    limit\_touch: pessimistic  \# optimistic, midpoint

    same\_bar\_execution: true

    partial\_fills: true

outputs:

directory: ./results/{run\_id}

save\_trades: true

save\_positions: true

save\_returns: true

tearsheet: true

\# Reproducibility

random\_seed: 42

start\_date: "2015-01-01"

end\_date: "2024-12-31"

\# Sweep parameters (optional)

sweep:

param\_grid:

    strategy.params.lookback\_period: \[10, 20, 30\]

    broker.slippage\_params.impact: \[0.05, 0.1, 0.15\]

n\_jobs: 4

## **8\) Time & Event Semantics**

### **Global Simulation Clock**

class SimulationClock:

    """Master clock maintaining simulation time"""

    

    def \_\_init\_\_(self, start: pd.Timestamp, end: pd.Timestamp, frequency: str):

        self.current \= start

        self.end \= end

        self.frequency \= frequency

        self.tick\_count \= 0

        

    def tick(self) \-\> pd.Timestamp:

        """Advance clock by one period"""

        self.tick\_count \+= 1

        self.current \+= pd.Timedelta(self.frequency)

        return self.current

        

    def align\_timestamp(self, ts: pd.Timestamp, policy: str \= "floor") \-\> pd.Timestamp:

        """Align timestamp to clock frequency"""

        if policy \== "floor":

            return ts.floor(self.frequency)

        elif policy \== "ceil":

            return ts.ceil(self.frequency)

        else:  \# nearest

            return ts.round(self.frequency)

### **Event Ordering**

Total ordering with deterministic tie-breaking:

1. Primary: Simulation timestamp
2. Secondary: Event priority (market data → signals → orders → fills)
3. Tertiary: Source hash (stable across runs)
4. Quaternary: Event content hash

### **Bar Semantics**

@dataclass

class BarSemantics:

    """Define how bars are interpreted"""

    

    \# Interval endpoints

    left\_inclusive: bool \= True  \# \[start, end)

    right\_inclusive: bool \= False

    

    \# When bar "occurs" for execution

    execution\_time: Literal\["open", "close"\] \= "close"

    

    \# Can orders placed during bar fill in same bar?

    same\_bar\_execution: bool \= True

    

    \# How to handle gaps

    gap\_handling: Literal\["halt", "limit", "market"\] \= "limit"

## **9\) Pricing & Fill Rules**

### **Market Orders**

def fill\_market\_order(order: Order, bar: MarketDataEvent, policy: FillPolicy) \-\> float:

    """Determine fill price for market order"""

    

    if policy.market\_order\_fill \== "open":

        return bar.open

    elif policy.market\_order\_fill \== "close":

        return bar.close

    elif policy.market\_order\_fill \== "midpoint":

        return (bar.high \+ bar.low) / 2

    else:  \# vwap\_estimate

        \# Approximate VWAP as weighted average of OHLC

        return (bar.open \+ bar.high \+ bar.low \+ bar.close) / 4

### **Limit Orders**

def fill\_limit\_order(order: Order, bar: MarketDataEvent, policy: FillPolicy) \-\> Optional\[float\]:

    """Determine if and at what price limit order fills"""

    

    if order.quantity \> 0:  \# Buy limit

        if bar.low \<= order.limit\_price:

            if policy.limit\_touch\_rule \== "pessimistic":

                return order.limit\_price

            elif policy.limit\_touch\_rule \== "optimistic":

                return bar.low

            else:  \# midpoint

                return (bar.low \+ order.limit\_price) / 2

    else:  \# Sell limit

        if bar.high \>= order.limit\_price:

            if policy.limit\_touch\_rule \== "pessimistic":

                return order.limit\_price

            elif policy.limit\_touch\_rule \== "optimistic":

                return bar.high

            else:

                return (bar.high \+ order.limit\_price) / 2

                

    return None  \# No fill

### **Stop Orders**

def check\_stop\_trigger(order: Order, bar: MarketDataEvent, policy: FillPolicy) \-\> bool:

    """Check if stop order should trigger"""

    

    if policy.stop\_trigger \== "close":

        trigger\_price \= bar.close

    elif policy.stop\_trigger \== "high\_low":

        trigger\_price \= bar.high if order.quantity \< 0 else bar.low

    else:  \# any\_print

        \# Assumes any price in bar range could trigger

        if order.quantity \> 0:  \# Buy stop

            return bar.high \>= order.stop\_price

        else:  \# Sell stop

            return bar.low \<= order.stop\_price

            

    if order.quantity \> 0:

        return trigger\_price \>= order.stop\_price

    else:

        return trigger\_price \<= order.stop\_price

## **10\) Slippage, Costs, and Market Impact**

### **Slippage Models**

from abc import ABC, abstractmethod

import numpy as np

class SlippageModel(ABC):

    """Base class for slippage models"""

    

    @abstractmethod

    def calculate(

        self,

        order: Order,

        fill\_price: float,

        volume: float,

        volatility: float

    ) \-\> float:

        """Calculate slippage-adjusted price"""

        pass

class FixedBasisPointsSlippage(SlippageModel):

    """Fixed slippage in basis points"""

    

    def \_\_init\_\_(self, basis\_points: float \= 10):

        self.bps \= basis\_points / 10000

        

    def calculate(self, order: Order, fill\_price: float, volume: float, volatility: float) \-\> float:

        slippage \= fill\_price \* self.bps

        return fill\_price \+ slippage if order.quantity \> 0 else fill\_price \- slippage

class VolumeImpactSlippage(SlippageModel):

    """Square-root market impact model"""

    

    def \_\_init\_\_(self, impact: float \= 0.1, gamma: float \= 0.5):

        self.impact \= impact

        self.gamma \= gamma

        

    def calculate(self, order: Order, fill\_price: float, volume: float, volatility: float) \-\> float:

        participation \= abs(order.quantity) / max(volume, 1\)

        temporary\_impact \= self.impact \* (participation \*\* self.gamma) \* volatility

        return fill\_price \* (1 \+ temporary\_impact) if order.quantity \> 0 else fill\_price \* (1 \- temporary\_impact)

class AlmgrenChrissModel(SlippageModel):

    """Almgren-Chriss optimal execution model"""

    

    def \_\_init\_\_(self, eta: float \= 2.5e-7, gamma: float \= 2.5e-6, alpha: float \= 0.95):

        self.eta \= eta  \# Temporary impact

        self.gamma \= gamma  \# Permanent impact  

        self.alpha \= alpha  \# Exponent

        

    def calculate(self, order: Order, fill\_price: float, volume: float, volatility: float) \-\> float:

        adv\_fraction \= abs(order.quantity) / max(volume, 1\)

        permanent \= self.gamma \* adv\_fraction

        temporary \= self.eta \* (adv\_fraction \*\* self.alpha)

        total\_impact \= permanent \+ temporary

        return fill\_price \* (1 \+ total\_impact) if order.quantity \> 0 else fill\_price \* (1 \- total\_impact)

### **Commission Models**

class CommissionModel(ABC):

    """Base commission model"""

    

    @abstractmethod

    def calculate(self, order: Order, fill\_price: float) \-\> float:

        pass

class PerShareCommission(CommissionModel):

    """Fixed commission per share"""

    

    def \_\_init\_\_(self, cost\_per\_share: float \= 0.001, minimum: float \= 1.0):

        self.cost\_per\_share \= cost\_per\_share

        self.minimum \= minimum

        

    def calculate(self, order: Order, fill\_price: float) \-\> float:

        commission \= abs(order.quantity) \* self.cost\_per\_share

        return max(commission, self.minimum)

class TieredCommission(CommissionModel):

    """Tiered commission based on volume"""

    

    def \_\_init\_\_(self, tiers: list\[tuple\[float, float\]\]):

        self.tiers \= tiers  \# \[(threshold, rate), ...\]

        

    def calculate(self, order: Order, fill\_price: float) \-\> float:

        value \= abs(order.quantity \* fill\_price)

        for threshold, rate in reversed(self.tiers):

            if value \>= threshold:

                return value \* rate

        return 0

## **11\) Portfolio Construction & Optimization**

### **Sizing Interface**

from abc import ABC, abstractmethod

import numpy as np

class PositionSizer(ABC):

    """Base class for position sizing"""

    

    @abstractmethod

    def size(

        self,

        signal: float,

        symbol: str,

        portfolio: PortfolioState,

        market\_data: dict

    ) \-\> float:

        """Calculate position size from signal"""

        pass

class FixedSizer(PositionSizer):

    """Fixed dollar amount per position"""

    

    def \_\_init\_\_(self, position\_size: float \= 10000):

        self.position\_size \= position\_size

        

    def size(self, signal: float, symbol: str, portfolio: PortfolioState, market\_data: dict) \-\> float:

        price \= market\_data\[symbol\].close

        return self.position\_size / price if signal \> 0 else 0

class KellySizer(PositionSizer):

    """Kelly criterion sizing"""

    

    def \_\_init\_\_(self, lookback: int \= 252, cap: float \= 0.25):

        self.lookback \= lookback

        self.cap \= cap  \# Cap at 25% of portfolio

        

    def size(self, signal: float, symbol: str, portfolio: PortfolioState, market\_data: dict) \-\> float:

        returns \= self.get\_historical\_returns(symbol)

        

        \# Estimate win probability and profit/loss ratio

        wins \= returns\[returns \> 0\]

        losses \= returns\[returns \< 0\]

        

        if len(wins) \== 0 or len(losses) \== 0:

            return 0

            

        p \= len(wins) / len(returns)  \# Win probability

        b \= wins.mean() / abs(losses.mean())  \# Profit/loss ratio

        

        \# Kelly formula: f \= (p\*b \- q) / b where q \= 1-p

        kelly\_fraction \= (p \* b \- (1 \- p)) / b

        kelly\_fraction \= np.clip(kelly\_fraction, 0, self.cap)

        

        return kelly\_fraction \* portfolio.total\_value / market\_data\[symbol\].close

### **Portfolio Optimizer Interface**

class PortfolioOptimizer(ABC):

    """Base portfolio optimizer"""

    

    @abstractmethod

    def optimize(

        self,

        expected\_returns: np.ndarray,

        covariance: np.ndarray,

        current\_weights: np.ndarray,

        constraints: dict

    ) \-\> np.ndarray:

        """Optimize portfolio weights"""

        pass

class MeanVarianceOptimizer(PortfolioOptimizer):

    """Markowitz mean-variance optimization"""

    

    def \_\_init\_\_(self, risk\_aversion: float \= 1.0):

        self.risk\_aversion \= risk\_aversion

        

    def optimize(

        self,

        expected\_returns: np.ndarray,

        covariance: np.ndarray,

        current\_weights: np.ndarray,

        constraints: dict

    ) \-\> np.ndarray:

        \# Implementation using cvxpy or scipy.optimize

        pass

class RiskParityOptimizer(PortfolioOptimizer):

    """Equal risk contribution"""

    

    def optimize(

        self,

        expected\_returns: np.ndarray,

        covariance: np.ndarray,

        current\_weights: np.ndarray,

        constraints: dict

    ) \-\> np.ndarray:

        \# Iterative algorithm to equalize risk contributions

        pass

## **12\) Asset Class Specifics**

### **Equities**

@dataclass

class EquitySpecifics:

    """Equity-specific parameters"""

    

    \# Shorting

    shortable: bool \= True

    short\_fee\_rate: float \= 0.02  \# Annual borrow rate

    

    \# Settlement

    settlement\_days: int \= 2  \# T+2

    

    \# Corporate actions

    dividend\_reinvestment: bool \= False

    

    \# Lot sizes

    round\_lots: bool \= False

    lot\_size: int \= 100

### **Futures**

@dataclass

class FutureSpecifics:

    """Futures-specific parameters"""

    

    \# Contract specs

    multiplier: float

    tick\_size: float

    expiry: pd.Timestamp

    

    \# Margin

    initial\_margin: float

    maintenance\_margin: float

    

    \# Rolling

    roll\_method: Literal\["calendar", "volume", "oi"\] \= "calendar"

    roll\_days\_before: int \= 5

    

    \# P\&L

    daily\_settlement: bool \= True  \# Mark to market daily

### **Crypto**

@dataclass

class CryptoSpecifics:

    """Crypto-specific parameters"""

    

    \# Trading

    trading\_hours: str \= "24/7"

    maker\_fee: float \= 0.001

    taker\_fee: float \= 0.0015

    

    \# Funding

    funding\_rate: float \= 0.0001  \# Per 8 hours

    funding\_interval: pd.Timedelta \= pd.Timedelta(hours=8)

    

    \# Decimals

    base\_decimals: int \= 8

    quote\_decimals: int \= 2

**Future Support Required for Options:**

* Greeks calculation engine
* Volatility surface modeling
* Exercise/assignment logic
* Pin risk management

## **13\) Universe & Corporate Actions**

### **Universe Management**

class UniverseManager:

    """Manages dynamic universe of tradeable assets"""

    

    def \_\_init\_\_(self, static\_universe: Optional\[list\[str\]\] \= None):

        self.static\_universe \= set(static\_universe or \[\])

        self.dynamic\_filters \= \[\]

        self.universe\_history \= {}

        

    def add\_filter(self, filter\_func: Callable) \-\> None:

        """Add dynamic universe filter"""

        self.dynamic\_filters.append(filter\_func)

        

    def get\_universe(self, date: pd.Timestamp, market\_data: pl.DataFrame) \-\> set\[str\]:

        """Get universe for given date"""

        

        \# Start with static universe

        universe \= self.static\_universe.copy()

        

        \# Apply dynamic filters

        for filter\_func in self.dynamic\_filters:

            universe \= filter\_func(universe, date, market\_data)

            

        \# Handle delistings

        universe \= self.remove\_delisted(universe, date)

        

        \# Record for history

        self.universe\_history\[date\] \= universe

        

        return universe

        

    def remove\_delisted(self, universe: set\[str\], date: pd.Timestamp) \-\> set\[str\]:

        """Remove delisted securities"""

        \# Check delisting database

        pass

### **Corporate Actions Processor**

class CorporateActionsProcessor:

    """Handles corporate actions"""

    

    def \_\_init\_\_(self, adjustment\_method: str \= "backwards"):

        self.adjustment\_method \= adjustment\_method

        self.actions\_log \= \[\]

        

    def process\_split(

        self,

        symbol: str,

        date: pd.Timestamp,

        ratio: float,

        prices: pl.DataFrame,

        positions: dict

    ) \-\> tuple\[pl.DataFrame, dict\]:

        """Process stock split"""

        

        if self.adjustment\_method \== "backwards":

            \# Adjust historical prices

            mask \= (prices\["symbol"\] \== symbol) & (prices\["date"\] \< date)

            prices\[mask, "close"\] \= prices\[mask, "close"\] / ratio

            prices\[mask, "volume"\] \= prices\[mask, "volume"\] \* ratio

            

        \# Adjust positions

        if symbol in positions:

            positions\[symbol\].quantity \*= ratio

            positions\[symbol\].cost\_basis /= ratio

            

        self.actions\_log.append({

            "type": "split",

            "symbol": symbol,

            "date": date,

            "ratio": ratio

        })

        

        return prices, positions

        

    def process\_dividend(

        self,

        symbol: str,

        date: pd.Timestamp,

        amount: float,

        positions: dict,

        cash: float

    ) \-\> float:

        """Process cash dividend"""

        

        if symbol in positions:

            dividend \= positions\[symbol\].quantity \* amount

            cash \+= dividend

            

            self.actions\_log.append({

                "type": "dividend",

                "symbol": symbol,

                "date": date,

                "amount": amount,

                "total": dividend

            })

            

        return cash

## **14\) Testing & Validation Strategy**

### **Unit Tests**

\# test\_event\_ordering.py

def test\_deterministic\_ordering():

    """Test that events are processed in deterministic order"""

    events \= \[

        MarketDataEvent(timestamp=t1, symbol="AAPL", ...),

        SignalEvent(timestamp=t1, symbol="GOOGL", ...),

        OrderEvent(timestamp=t1, symbol="MSFT", ...)

    \]

    

    queue1 \= EventQueue(seed=42)

    queue2 \= EventQueue(seed=42)

    

    for event in events:

        queue1.put(event)

        queue2.put(event)

        

    assert \[queue1.get() for \_ in range(3)\] \== \[queue2.get() for \_ in range(3)\]

### **Property-Based Tests**

\# test\_accounting\_invariants.py

from hypothesis import given, strategies as st

@given(

    trades=st.lists(

        st.tuples(

            st.floats(min\_value=1, max\_value=1000),  \# quantity

            st.floats(min\_value=1, max\_value=1000)   \# price

        )

    )

)

def test\_portfolio\_value\_conservation(trades):

    """Test that portfolio value is conserved across trades"""

    portfolio \= Portfolio(initial\_cash=100000)

    

    for quantity, price in trades:

        cost \= quantity \* price

        if cost \<= portfolio.cash:

            portfolio.execute\_trade("TEST", quantity, price)

            

    \# Value should equal cash \+ position values

    total\_value \= portfolio.cash \+ sum(

        pos.quantity \* pos.current\_price for pos in portfolio.positions.values()

    )

    assert abs(total\_value \- 100000\) \< 0.01  \# Floating point tolerance

### **Golden Tests**

\# test\_golden.py

def test\_simple\_buy\_and\_hold():

    """Test against known good output"""

    config \= BacktestConfig(

        data=DataConfig(source="csv", path="test\_data/simple.csv"),

        strategy=StrategyConfig(module="test\_strategies.buy\_and\_hold"),

        broker=BrokerConfig(initial\_capital=10000, commission=0)

    )

    

    result \= run\_backtest(config)

    

    \# Load golden results

    golden \= pd.read\_csv("golden/buy\_and\_hold\_results.csv")

    

    assert result.final\_value \== pytest.approx(golden\["final\_value"\].iloc\[0\], rel=1e-6)

    assert result.sharpe\_ratio \== pytest.approx(golden\["sharpe"\].iloc\[0\], rel=1e-6)

### **Benchmark Suite**

\# benchmarks/performance.py

def benchmark\_event\_throughput():

    """Measure event processing throughput"""

    

    events \= generate\_events(n\_symbols=1000, n\_days=252 \* 10, frequency="1min")

    

    start \= time.perf\_counter()

    engine \= BacktestEngine()

    engine.process\_events(events)

    elapsed \= time.perf\_counter() \- start

    

    throughput \= len(events) / elapsed

    assert throughput \> 1\_000\_000  \# 1M events/second target

## **15\) Observability, Diagnostics & Developer Experience**

### **Event Timeline Explorer**

class EventTimeline:

    """Interactive event timeline for debugging"""

    

    def \_\_init\_\_(self, events: list\[Event\]):

        self.events \= sorted(events, key=lambda e: (e.timestamp, e.event\_type))

        

    def print\_range(self, start: pd.Timestamp, end: pd.Timestamp):

        """Print events in time range"""

        for event in self.events:

            if start \<= event.timestamp \<= end:

                print(f"{event.timestamp}: {event.event\_type} \- {event}")

                

    def trace\_order(self, order\_id: str):

        """Trace lifecycle of specific order"""

        related \= \[e for e in self.events if hasattr(e, 'order\_id') and e.order\_id \== order\_id\]

        for event in related:

            print(f"{event.timestamp}: {type(event).\_\_name\_\_} \- {event}")

### **Error Messages**

class BacktestError(Exception):

    """Base exception with helpful context"""

    

    def \_\_init\_\_(self, message: str, context: dict, suggestion: str \= None):

        self.message \= message

        self.context \= context

        self.suggestion \= suggestion

        

        full\_message \= f"{message}\\n"

        full\_message \+= f"Context: {json.dumps(context, indent=2)}\\n"

        if suggestion:

            full\_message \+= f"Suggestion: {suggestion}"

            

        super().\_\_init\_\_(full\_message)

\# Example usage

if not symbol in self.universe:

    raise BacktestError(

        f"Symbol {symbol} not in universe",

        context={"symbol": symbol, "universe": list(self.universe)\[:5\]},

        suggestion=f"Add {symbol} to universe in config or check for typos"

    )

### **Cookbook Examples**

\# examples/simple\_momentum.py

"""

Simple momentum strategy example

This strategy:

1\. Ranks stocks by 20-day momentum

2\. Buys top 10 stocks

3\. Rebalances monthly

"""

class SimpleMomentum(Strategy):

    def \_\_init\_\_(self, context: StrategyContext):

        super().\_\_init\_\_(context)

        self.lookback \= 20

        self.n\_positions \= 10

        self.last\_rebalance \= None

        

    def on\_bar(self, bar: MarketDataEvent) \-\> None:

        \# Check if month changed

        if self.last\_rebalance and bar.timestamp.month \== self.last\_rebalance.month:

            return

            

        \# Calculate momentum for all symbols

        momentum \= {}

        for symbol in self.context.universe:

            hist \= self.get\_history(\[symbol\], \["close"\], self.lookback)

            if len(hist) \== self.lookback:

                momentum\[symbol\] \= (hist\["close"\].iloc\[-1\] / hist\["close"\].iloc\[0\]) \- 1

                

        \# Select top N

        ranked \= sorted(momentum.items(), key=lambda x: x\[1\], reverse=True)

        selected \= \[symbol for symbol, \_ in ranked\[:self.n\_positions\]\]

        

        \# Rebalance portfolio

        for symbol in self.context.portfolio.positions:

            if symbol not in selected:

                self.order\_target\_percent(symbol, 0\)

                

        for symbol in selected:

            self.order\_target\_percent(symbol, 1.0 / self.n\_positions)

            

        self.last\_rebalance \= bar.timestamp

## **16\) Reproducibility & Governance**

### **Run Manifest**

@dataclass

class CompleteManifest:

    """Complete record for reproducibility"""

    

    \# Identifiers

    run\_id: str  \# UUID4

    timestamp: pd.Timestamp

    

    \# Configuration

    config: dict  \# Full config as dict

    config\_hash: str  \# SHA256 of config

    

    \# Data

    data\_files: list\[str\]  \# Paths to input files

    data\_hashes: dict\[str, str\]  \# SHA256 per file

    

    \# Code

    code\_version: str  \# Git commit hash

    strategy\_hash: str  \# Hash of strategy code

    

    \# Environment

    python\_version: str

    packages: dict\[str, str\]  \# Package versions

    system: dict  \# OS, CPU, memory

    

    \# Seeds

    random\_seeds: dict\[str, int\]  \# All PRNG seeds

    

    def verify\_reproducibility(self, other: 'CompleteManifest') \-\> bool:

        """Verify this run can be reproduced"""

        return all(\[

            self.config\_hash \== other.config\_hash,

            self.data\_hashes \== other.data\_hashes,

            self.code\_version \== other.code\_version,

            self.random\_seeds \== other.random\_seeds

        \])

### **Schema Versioning**

class SchemaVersion:

    """Semantic versioning for output schemas"""

    

    CURRENT\_VERSION \= "1.0.0"

    

    \# Column schemas with versions

    TRADES\_SCHEMA \= {

        "1.0.0": {

            "timestamp": "datetime64\[ns\]",

            "order\_id": "str",

            "symbol": "str",

            "quantity": "float64",

            "price": "float64",

            "commission": "float64",

            "slippage": "float64"

        }

    }

    

    @classmethod

    def migrate(cls, data: pl.DataFrame, from\_version: str, to\_version: str) \-\> pl.DataFrame:

        """Migrate data between schema versions"""

        \# Migration logic

        pass

## **17\) Roadmap & Milestones**

### **v0 (MVP) \- 3 months**

* ✅ Core event loop with deterministic ordering
* ✅ Single-asset equity backtests
* ✅ Daily and minute OHLCV bars
* ✅ Market and limit orders
* ✅ Basic portfolio accounting
* ✅ Fixed commission model
* ✅ Simple momentum example strategy
* ✅ Parquet output format
* ✅ Basic reproducibility (config \+ seed)

**Acceptance**: Successfully replicate Zipline tutorial strategies

### **v1 (Production) \- 6 months**

* Multi-asset support (equities \+ futures \+ crypto)
* Multi-currency portfolios
* Full order types (stop, stop-limit, OCO)
* Slippage models (fixed, volume impact, Almgren-Chriss)
* ML strategy support with anti-leakage
* Corporate actions (splits, dividends)
* Dynamic universe management
* Risk controls (pre/post trade)
* Parameter sweep runner
* Integration with factor analysis library
* Complete test suite (\>95% coverage)

**Acceptance**: Handle production strategies from quant team

### **v2 (Advanced) \- 12 months**

* Options support (Black-Scholes pricing, Greeks)
* Advanced microstructure models
* Intraday portfolio optimization
* Capacity analysis
* Live paper trading adapter
* Real-time data feeds
* Distributed backtesting
* Web UI for configuration
* Cloud deployment templates

**Acceptance**: Feature parity with professional platforms

## **18\) Risks & Open Questions**

### **Technical Risks**

| Risk                        | Impact | Probability | Mitigation                                |
|-----------------------------|--------|-------------|-------------------------------------------|
| Polars API instability      | High   | Low         | Pin versions, abstract data layer         |
| Numba compilation overhead  | Medium | Medium      | Lazy compilation, caching                 |
| Memory usage with tick data | High   | Medium      | Streaming architecture, chunking          |
| Floating point determinism  | High   | Low         | Use decimal or fixed point for accounting |

### **Product Risks**

| Risk                | Impact | Probability | Mitigation                              |
|---------------------|--------|-------------|-----------------------------------------|
| Adoption resistance | High   | Medium      | Migration tools, compatibility layer    |
| Feature creep       | Medium | High        | Strict MVP scope, plugin architecture   |
| Documentation debt  | Medium | Medium      | Docstring requirements, auto-generation |

### **Open Questions**

1. **Python vs Rust core**: Start pure Python, profile bottlenecks, optimize with Rust?

    * **Decision deadline**: After v0 benchmarks
    * **Recommendation**: Python with Numba for v0, evaluate Rust for v1
2. **Default bar fill rules**: Close, midpoint, or VWAP estimate?

    * **Decision deadline**: Before v0
    * **Recommendation**: Close for daily, midpoint for intraday
3. **Polars vs Pandas**: Commit to Polars or support both?

    * **Decision deadline**: Before v0
    * **Recommendation**: Polars native, Pandas compatibility layer
4. **State management**: In-memory, Redis, or SQLite for large backtests?

    * **Decision deadline**: v1 planning
    * **Recommendation**: In-memory default, pluggable backends

## **Appendices**

### **A) Example Config (YAML)**

\# complete\_backtest.yaml

version: "1.0.0"

data:

source: parquet

path: /data/equities/minute\_bars/

columns:

    timestamp: "date"

    open: "open"

    high: "high"

    low: "low"  

    close: "close"

    volume: "volume"

calendar: NYSE

timezone: "America/New\_York"

adjustments: backwards

universe:

type: static

symbols: \["AAPL", "GOOGL", "MSFT", "AMZN", "META"\]

strategy:

module: strategies.mean\_reversion

class\_name: MeanReversionStrategy

params:

    lookback\_period: 20

    entry\_threshold: \-2.0  \# Z-score

    exit\_threshold: 0.0

    position\_size: 0.1  \# 10% per position

    max\_positions: 5

broker:

initial\_capital: 1000000.0

commission:

    model: per\_share

    cost: 0.001

    minimum: 1.0

slippage:

    model: volume\_impact

    impact: 0.1

    gamma: 0.5

restrictions:

    allow\_shorts: true

    short\_fee\_rate: 0.02  \# 2% annual

execution:

fill\_policy:

    market\_fill: close

    limit\_touch: pessimistic

    stop\_trigger: close

    same\_bar\_execution: false

    partial\_fills: true

order\_delay: 0 \# seconds

risk:

pre\_trade\_checks:

    enabled: true

    max\_position\_size: 0.15  \# 15% of portfolio

    max\_sector\_exposure: 0.4

    max\_leverage: 1.5

post\_trade\_monitoring:

    max\_drawdown: 0.20

    trailing\_stop: 0.15

outputs:

directory: ./results/{run\_id}

formats: \["parquet", "json"\]

include:

    trades: true

    positions: true

    returns: true

    metrics: true

compression: snappy

simulation:

start\_date: "2020-01-01"

end\_date: "2023-12-31"

random\_seed: 42

log\_level: INFO

sweep:

enabled: false

parameters:

    strategy.params.lookback\_period:

      type: range

      start: 10

      stop: 30

      step: 5

    strategy.params.entry\_threshold:

      type: list

      values: \[-1.5, \-2.0, \-2.5\]

method: grid \# grid, random, bayesian

n\_jobs: 4

monitoring:

metrics\_port: 8080

enable\_profiling: false

trace\_orders: false

### **B) Example Python Interface Sketches**

\# strategy\_interface.py

from typing import Optional, Dict, Any

from dataclasses import dataclass

import polars as pl

import pandas as pd

class Strategy:

    """Base strategy class for event-driven backtesting

    

    All strategies should inherit from this class and implement

    the required methods for handling events and generating signals.

    """

    

    def \_\_init\_\_(self, context: 'StrategyContext'):

        """Initialize strategy with backtesting context

        

        Args:

            context: StrategyContext providing access to data and ordering

        """

        self.context \= context

        self.state: Dict\[str, Any\] \= {}

        

    def on\_start(self) \-\> None:

        """Called once before backtesting begins

        

        Use this to initialize strategy state, precompute features,

        or set up any required data structures.

        """

        pass

        

    def on\_bar(self, bar: 'MarketBar') \-\> None:

        """Process new market data bar

        

        Args:

            bar: MarketBar containing OHLCV data

        """

        pass

        

    def on\_fill(self, fill: 'Fill') \-\> None:

        """Notification of order fill

        

        Args:

            fill: Fill object with execution details

        """

        pass

\# order\_types.py

@dataclass

class Order:

    """Order representation

    

    Attributes:

        order\_id: Unique order identifier

        symbol: Security symbol

        quantity: Number of shares (negative for sell)

        order\_type: Type of order (market, limit, etc.)

        limit\_price: Limit price for limit orders

        stop\_price: Stop price for stop orders

        time\_in\_force: Order duration

    """

    order\_id: str

    symbol: str

    quantity: float

    order\_type: str \= "market"

    limit\_price: Optional\[float\] \= None

    stop\_price: Optional\[float\] \= None

    time\_in\_force: str \= "day"

@dataclass

class Fill:

    """Execution fill details

    

    Attributes:

        order\_id: Original order ID

        fill\_id: Unique fill identifier

        symbol: Security symbol

        quantity: Filled quantity

        price: Execution price

        commission: Commission charged

        slippage: Slippage cost

        timestamp: Execution timestamp

    """

    order\_id: str

    fill\_id: str

    symbol: str

    quantity: float

    price: float

    commission: float

    slippage: float

    timestamp: pd.Timestamp

\# broker\_interface.py

class Broker:

    """Broker simulation interface

    

    Handles order matching, fill generation, and commission calculation.

    """

    

    def submit\_order(self, order: Order) \-\> str:

        """Submit order for execution

        

        Args:

            order: Order to submit

            

        Returns:

            order\_id: Unique identifier for tracking

        """

        pass

        

    def cancel\_order(self, order\_id: str) \-\> bool:

        """Cancel pending order

        

        Args:

            order\_id: Order to cancel

            

        Returns:

            success: Whether cancellation succeeded

        """

        pass

\# results\_interface.py

@dataclass

class BacktestResult:

    """Container for backtest results

    

    Provides access to performance metrics, trade history,

    and analytics.

    """

    

    \# Time series data

    equity\_curve: pl.DataFrame

    returns: pl.DataFrame

    positions: pl.DataFrame

    

    \# Transaction records

    trades: pl.DataFrame

    orders: pl.DataFrame

    

    \# Summary statistics

    total\_return: float

    annual\_return: float

    sharpe\_ratio: float

    max\_drawdown: float

    win\_rate: float

    

    def to\_parquet(self, path: str) \-\> None:

        """Save results to parquet files"""

        pass

        

    def plot\_equity\_curve(self) \-\> None:

        """Generate equity curve plot"""

        pass

### **C) Standard Output Schemas**

\# Orders Schema

ORDERS\_SCHEMA \= {

    "order\_id": pl.Utf8,

    "timestamp": pl.Datetime("ns"),

    "symbol": pl.Utf8,

    "quantity": pl.Float64,

    "order\_type": pl.Utf8,

    "limit\_price": pl.Float64,

    "stop\_price": pl.Float64,

    "status": pl.Utf8,  \# submitted, partial, filled, cancelled

    "filled\_quantity": pl.Float64,

    "average\_price": pl.Float64,

    "submitted\_at": pl.Datetime("ns"),

    "updated\_at": pl.Datetime("ns")

}

\# Trades Schema (with MFE/MAE)

TRADES\_SCHEMA \= {

    "trade\_id": pl.Utf8,

    "order\_id": pl.Utf8,

    "timestamp": pl.Datetime("ns"),

    "symbol": pl.Utf8,

    "quantity": pl.Float64,

    "entry\_price": pl.Float64,

    "exit\_price": pl.Float64,

    "commission": pl.Float64,

    "slippage": pl.Float64,

    "realized\_pnl": pl.Float64,

    "mfe": pl.Float64,  \# Maximum Favorable Excursion

    "mae": pl.Float64,  \# Maximum Adverse Excursion

    "duration\_bars": pl.Int32,

    "duration\_time": pl.Duration("ns")

}

\# Positions Schema

POSITIONS\_SCHEMA \= {

    "timestamp": pl.Datetime("ns"),

    "symbol": pl.Utf8,

    "quantity": pl.Float64,

    "cost\_basis": pl.Float64,

    "current\_price": pl.Float64,

    "market\_value": pl.Float64,

    "unrealized\_pnl": pl.Float64,

    "realized\_pnl": pl.Float64,

    "exposure": pl.Float64,  \# As % of portfolio

    "days\_held": pl.Int32

}

\# Returns Schema

RETURNS\_SCHEMA \= {

    "timestamp": pl.Datetime("ns"),

    "period\_return": pl.Float64,

    "cumulative\_return": pl.Float64,

    "portfolio\_value": pl.Float64,

    "cash": pl.Float64,

    "leverage": pl.Float64,

    "long\_exposure": pl.Float64,

    "short\_exposure": pl.Float64,

    "net\_exposure": pl.Float64

}

\# Run Manifest JSON Schema

RUN\_MANIFEST\_SCHEMA \= {

    "run\_id": str,

    "timestamp": str,  \# ISO format

    "engine\_version": str,

    "config": dict,

    "config\_hash": str,

    "data\_sources": list,

    "data\_hashes": dict,

    "strategy": {

        "module": str,

        "class": str,

        "hash": str

    },

    "environment": {

        "python\_version": str,

        "packages": dict,

        "system": dict

    },

    "random\_seeds": dict,

    "performance": {

        "total\_return": float,

        "sharpe\_ratio": float,

        "max\_drawdown": float,

        "total\_trades": int,

        "win\_rate": float

    }

}

### **D) Comparison Matrix**

| Feature               | Proposed Engine     | Zipline Reloaded | Backtrader     | vectorbt pro      |
|-----------------------|---------------------|------------------|----------------|-------------------|
| **Event Model**       | Deterministic queue | Pipeline-based   | Event-driven   | Vectorized        |
| **Languages**         | Python \+ Numba     | Python \+ Cython | Pure Python    | Python \+ Numba   |
| **Data Backend**      | Polars/Arrow        | Pandas           | Custom         | NumPy/Pandas      |
| **Order Types**       | All standard        | Basic            | Extensive      | Basic             |
| **Microstructure**    | Pluggable models    | Limited          | Basic          | None              |
| **Performance**       | 1M+ events/sec      | 100K events/sec  | 50K events/sec | 10M+ (vectorized) |
| **Memory Usage**      | Streaming capable   | High             | Medium         | Very high         |
| **Accounting**        | Double-entry        | Basic            | Good           | Basic             |
| **Multi-Asset**       | Native              | Limited          | Yes            | Yes               |
| **ML Integration**    | Native              | Via pipeline     | Manual         | Good              |
| **Reproducibility**   | Full manifest       | Seed only        | Seed only      | Seed only         |
| **Configuration**     | YAML/JSON first     | Python only      | Python only    | Python/JSON       |
| **Error Messages**    | Contextual          | Poor             | Basic          | Good              |
| **Documentation**     | Comprehensive       | Good             | Good           | Excellent         |
| **Extensibility**     | Plugin architecture | Limited          | Good           | Limited           |
| **Testing**           | Property-based      | Unit only        | Unit only      | Unit only         |
| **Corporate Actions** | Full support        | Basic            | Manual         | None              |
| **Risk Management**   | Pre/post checks     | None             | Basic          | None              |
| **Live Trading**      | Future              | Alpha            | Yes            | No                |

### **E) Acceptance Criteria Checklist**

#### **Functional Requirements**

* \[ \] Processes OHLCV data from Parquet/CSV sources
* \[ \] Supports market and limit orders
* \[ \] Calculates commissions and slippage
* \[ \] Tracks positions and cash accurately
* \[ \] Handles corporate actions (splits, dividends)
* \[ \] Enforces risk limits (position size, leverage)
* \[ \] Generates standard output files (trades, positions, returns)
* \[ \] Produces reproducible results with fixed seed

#### **Non-Functional Requirements**

* \[ \] Processes \>1M events per second on reference hardware
* \[ \] Memory usage \<10GB for standard backtest
* \[ \] 100% deterministic with fixed configuration
* \[ \] \>95% test coverage
* \[ \] All public APIs documented
* \[ \] Plugin interfaces versioned and stable
* \[ \] Error messages include context and suggestions

#### **Strategy Support**

* \[ \] Python strategy API with state management
* \[ \] Config-driven strategy for simple cases
* \[ \] ML model integration with anti-leakage
* \[ \] Access to rolling data windows
* \[ \] Multiple time frame support

#### **Asset Classes**

* \[ \] Equities with proper settlement
* \[ \] Futures with roll logic (v1)
* \[ \] Crypto with 24/7 trading (v1)
* \[ \] Multi-currency portfolios (v1)

#### **Development Experience**

* \[ \] Setup takes \<5 minutes
* \[ \] First backtest runs in \<10 lines
* \[ \] Cookbook with 10+ examples
* \[ \] Interactive debugging tools
* \[ \] Comprehensive error messages

#### **Documentation**

* \[ \] API reference for all public interfaces
* \[ \] Architecture guide
* \[ \] Migration guide from Zipline
* \[ \] Performance tuning guide
* \[ \] Plugin development guide

#### **Deliverables**

* \[ \] Core engine package
* \[ \] Standard strategies library
* \[ \] Test data generator
* \[ \] Benchmark suite
* \[ \] Migration tools from Zipline

---

**Final Notes:**

This PRD provides a comprehensive specification for a modern backtesting engine that addresses the limitations of
Zipline while incorporating best practices from contemporary tools. The architecture prioritizes determinism,
performance, and extensibility while maintaining a clear upgrade path from MVP to production readiness.

Key decisions still requiring validation:

1. Polars performance at scale with tick data
2. Numba compilation overhead vs. runtime benefits
3. Plugin API stability across versions
4. Default microstructure assumptions

Recommended next steps:

1. Prototype event loop with deterministic ordering
2. Benchmark Polars vs. Pandas for typical operations
3. Design plugin interface with version negotiation
4. Build MVP with single-asset equities
5. Gather feedback from quant team on critical features
