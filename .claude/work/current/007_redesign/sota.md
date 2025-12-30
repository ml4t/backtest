# **A State-of-the-Art Event-Driven Backtesting Engine**

### **1\. Executive Summary**

This document presents a comprehensive market review, technical architecture, and implementation plan for a new,
high-performance, event-driven backtesting engine, hereafter referred to as **ml4t.backtest**. The analysis concludes with a
definitive **GO** recommendation for the development of this engine.

The strategic rationale is compelling. The current Python backtesting landscape exhibits a significant gap for a tool
architected from the ground up for modern machine learning (ML) workflows. Incumbent solutions are either
technologically stagnant and difficult to maintain (Zipline Reloaded, Backtrader), architecturally optimized for rapid
parameter sweeping at the expense of simulation realism (vectorbt), or built on a non-native technology stack that
creates friction for the Python data science community (QuantConnect/Lean). ml4t.backtest is designed to fill this void,
providing a foundational asset for the 2025 edition of the *Machine Learning for Trading* book and its associated
ecosystem.

The **target users** are quantitative researchers, data scientists, and sophisticated systematic traders who develop
predictive models in Python and require a robust, high-fidelity simulation environment. These users demand rigorous
historical accuracy, particularly concerning the prevention of data leakage, which is a primary focus of ml4t.backtest's
design.

ml4t.backtest's competitive advantage will be built on four key **differentiators**:

1. **Polars-First Architecture:** The engine will be built on a modern, memory-efficient columnar data core using Polars
   and Apache Arrow. This design provides a fundamental performance advantage over legacy Pandas-based systems and
   enables zero-copy data flow between components, including potential high-performance modules written in Rust.
2. **Leakage-Safe ML Signal Ingestion:** ml4t.backtest will treat the point-in-time (PIT) correct alignment of asynchronous
   data—market data, features, and external ML signals—as a first-class architectural concern. This directly addresses
   the most critical and common failure mode in the backtesting of ML-driven strategies.
3. **Extensible Realism:** A modular framework will allow for pluggable, high-fidelity models of market impact, borrow
   costs, perpetual futures funding rates, and complex order fill logic. This commitment to realism will meet and exceed
   the capabilities of current open-source leaders, enabling users to generate results that more accurately reflect
   potential live performance.
4. **Python-Native Performance:** Through the targeted application of Numba for just-in-time (JIT) compilation of user
   strategy code and optional Rust components for the performance-critical core loop, ml4t.backtest will achieve execution
   speeds comparable to engines written in C\# or C++ without forcing users to leave the familiar and productive Python
   ecosystem.

By delivering on these differentiators, ml4t.backtest will not only serve as a superior replacement for Zipline for the book's
audience but will also establish itself as the premier backtesting framework for the next generation of quantitative and
machine learning-driven traders.

### **2\. Market Landscape (2025)**

The backtesting library market in 2025 is characterized by a set of mature but aging incumbents and a new wave of
performance-focused challengers. A critical distinction in this landscape is the architectural trade-off between
vectorized and event-driven engines. Vectorized engines excel at speed for simple, non-path-dependent strategies, making
them ideal for large-scale parameter optimization. Event-driven engines provide higher fidelity and realism, capable of
simulating complex, path-dependent logic and intricate order types, which is essential for ML-driven strategies
operating on high-frequency data. ml4t.backtest is positioned to capture the best of both worlds: the performance of modern
data-parallel tools within a high-fidelity, event-driven framework.

#### **Comparative Table of Backtesting Engines**

| Feature                  | vectorbt (Pro)                                            | Backtrader                                  | Zipline Reloaded                                   | QuantConnect/Lean                                   | Nautilus Trader            | Jesse                           |
|:-------------------------|:----------------------------------------------------------|:--------------------------------------------|:---------------------------------------------------|:----------------------------------------------------|:---------------------------|:--------------------------------|
| **License**              | Open Core (Pro is paid)                                   | GPL-3.0                                     | Apache 2.0                                         | Apache 2.0                                          | LGPL-3.0                   | Custom (Free \+ Paid Live)      |
| **Core Architecture**    | Vectorized (NumPy/Numba)                                  | Event-Driven (Python)                       | Event-Driven (Py/Cython)                           | Event-Driven (C\#/.NET)                             | Event-Driven (Rust/Python) | Event-Driven (Python)           |
| **Data Freq. Support**   | Tick to Daily                                             | Tick to Daily                               | Minute/Daily                                       | Tick to Daily                                       | Tick to Daily              | Minute/Daily                    |
| **Calendars/PIT Rigor**  | Basic                                                     | Basic                                       | Strong (Bundles)                                   | Very Strong (PIT Data)                              | Strong                     | Basic                           |
| **Order Types**          | Basic (Signals)                                           | Extensive                                   | Good                                               | Very Extensive                                      | Very Extensive             | Good                            |
| **Slippage/Impact**      | Basic (Fees only)                                         | Pluggable Slippage                          | Pluggable Slippage                                 | Pluggable Slippage/Impact                           | Pluggable                  | Basic                           |
| **Portfolio Accounting** | Single Currency                                           | Basic                                       | Good                                               | Multi-currency, Margin                              | Multi-currency, Margin     | Crypto-centric                  |
| **ML-Signal Ingestion**  | Manual (via arrays)                                       | Manual (via DataFeeds)                      | Manual (via custom data)                           | First-class (Alpha models)                          | Manual                     | Manual                          |
| **Streaming/Live Path**  | No (Signals only)                                         | Yes (Adapters)                              | Yes (Forks like zipline-broker)                    | Yes (First-class)                                   | Yes (First-class)          | Yes (Paid Plugin)               |
| **Reporting**            | Interactive (Plotly)                                      | Basic (matplotlib)                          | pyfolio-reloaded                                   | Web UI, custom reports                              | empyrical metrics          | Web UI                          |
| **Extensibility**        | Good (Pythonic)                                           | Good (Class-based)                          | Good (Class-based)                                 | Very Strong (Interfaces)                            | Very Strong (Adapters)     | Good (Class-based)              |
| **Performance Notes**    | Extremely fast for vectorized tasks; slow for iteration 1 | Slow for large datasets; pure Python loop 1 | Moderate; Cython helps but architecture is dated 2 | Very fast (C\# core); Python interop has overhead 2 | Very fast (Rust core) 3    | Good for crypto workloads       |
| **Community/Momentum**   | Very Active 4                                             | Mature but inactive dev 5                   | Maintained, slow dev 6                             | Very Active, Corporate-backed 7                     | Growing, Active Dev 8      | Active 9                        |
| **Commercial Model**     | Open Core (Pro features)                                  | Open Source                                 | Open Source                                        | Hosted Platform \+ On-Prem                          | Open Source \+ Consulting  | Open Source \+ Paid Live Plugin |

#### **Narrative Analysis of Incumbents**

* **vectorbt (incl. Pro):** vectorbt is a dominant force in the Python quantitative analysis space, primarily due to its
  exceptional performance on vectorized operations.5 Its architecture is built around NumPy arrays and accelerated with
  Numba, allowing it to test tens of thousands of parameter combinations in seconds.10 This makes it an outstanding tool
  for initial research, feature engineering, and optimizing simple, non-path-dependent strategies. However, this same
  vectorized design is its primary weakness for high-fidelity simulation. It cannot easily model complex, path-dependent
  logic such as trailing stops, intricate portfolio constraints, or market impact that depends on the sequence of
  trades. Its model of trading is based on signals, not on a realistic simulation of an order book or execution queue,
  making it unsuitable for testing high-frequency strategies or those requiring precise execution modeling.1 The "Pro"
  version follows a successful open-core model, validating a commercial path for high-value features.12
* **Backtrader:** Backtrader represents a mature, feature-rich, first-generation event-driven backtester.13 For years,
  it was the de facto standard for Python developers needing more realism than a simple vectorized script. It has
  extensive documentation, supports a wide range of order types, and provides a clear path to live trading via broker
  adapters.14 However, its development has largely stalled since the original author stepped back, with the last major
  feature release several years ago.5 The core engine is pure Python, which leads to significant performance bottlenecks
  when backtesting over large datasets or with high-frequency data.1 While its community remains active on platforms
  like Stack Overflow 16, the lack of core development makes it a legacy system and creates a clear opportunity for a
  modern, high-performance successor.
* **Zipline Reloaded:** As the direct predecessor in the *Machine Learning for Trading* book, Zipline holds a
  significant legacy. Zipline Reloaded is a community-led effort to keep the original Quantopian engine alive and
  compatible with modern Python versions.6 Its strengths lie in its event-driven nature and its "batteries-included"
  approach, particularly its robust data
  bundle system for managing point-in-time correct historical data, including corporate actions.19 However, the codebase
  is large, heavily reliant on Cython, and notoriously difficult to install and maintain.6 The effort to modernize its
  core architecture to leverage new technologies like Polars or Arrow would be tantamount to a complete rewrite, making
  a clean-slate build a more efficient path forward.
* **QuantConnect/Lean:** QuantConnect's open-source engine, Lean, is the most comprehensive and institutionally-aligned
  backtester in the open-source domain. Written primarily in C\#, it is extremely fast and features a highly modular,
  pluggable architecture.21 It provides first-class support for a vast array of asset classes (equities, futures,
  options, crypto, forex), realistic margin modeling, multi-currency accounting, and rigorous handling of corporate
  actions and point-in-time data.7
  Lean serves as the feature-parity benchmark for ml4t.backtest. Its primary strategic weakness, from the perspective of our
  target audience, is its non-native integration with Python. While it offers a Python wrapper, strategies are
  ultimately executed within the C\#/.NET runtime, creating a disconnect from the native Python data science ecosystem
  where models are typically researched and developed.2 This "two-language problem" introduces complexity and potential
  friction that a pure Python-native engine can eliminate.
* **Nautilus Trader:** Nautilus Trader is a modern, high-performance entrant that validates the architectural direction
  proposed for ml4t.backtest.8 Its core is written in Rust for maximum speed and safety, with a Python API for strategy
  definition.8 It is event-driven, supports tick-level data with nanosecond precision, and is designed for a seamless
  transition from backtesting to live trading.25 Its modular design, with concepts like a central
  MessageBus and pluggable adapters, is a strong architectural pattern. As a relatively new project, its community and
  third-party ecosystem are still developing, but its technical foundation is exceptionally strong and serves as a key
  competitor and reference design.
* **Jesse:** Jesse is a popular framework focused exclusively on the cryptocurrency market.9 Its strengths are its
  user-friendly API, excellent documentation, and features tailored to crypto, such as handling for futures funding
  rates.27 It offers a complete workflow from backtesting to live trading (as a paid plugin) and has built a dedicated
  community.28 While its scope is narrower than ml4t.backtest's, its emphasis on a smooth user experience and crypto-specific
  realism provides valuable lessons.

The market analysis reveals a clear bifurcation. Tools like vectorbt are optimized for "wide and shallow"
analysis—testing many simple parameter sets quickly. In contrast, tools like QuantConnect/Lean and Nautilus Trader are
built for "narrow and deep" simulation—testing a single complex strategy with high fidelity. This creates a significant
workflow gap for ML practitioners, who often start with wide, exploratory research and then need to transition to deep,
realistic simulation without rewriting their entire strategy. ml4t.backtest is positioned to bridge this gap by offering a
unified, Python-native platform that supports both modes of analysis.

### **3\. Feature Parity Matrix**

This matrix provides a granular comparison of features across the competitive landscape. It serves as a checklist to
define the minimum viable product (MVP) and subsequent development phases for ml4t.backtest, ensuring it is competitive upon
launch and offers clear advantages.

| Feature Area          | ml4t.backtest (Target)               | vectorbt Pro         | Backtrader                  | Zipline Reloaded           | QuantConnect/Lean           | Nautilus Trader             |
|:----------------------|:-------------------------------|:---------------------|:----------------------------|:---------------------------|:----------------------------|:----------------------------|
| **Data Ingestion**    | Polars/Arrow, Parquet, CSV     | Pandas, various APIs | Pandas, CSV, custom feeds   | Custom format, CSV, APIs   | Parquet, CSV, custom        | CSV, custom                 |
| **Data Frequency**    | Tick, Bar (any), Quotes        | Tick to Daily        | Tick to Daily               | Tick to Daily              | Tick (ns), Bar, Quotes      | Minute/Daily                |
| **Calendar Support**  | Full (pandas-market-calendars) | None                 | Partial                     | Full                       | Full                        | Basic                       |
| **Corporate Actions** | Full (Splits, Dividends)       | None                 | Manual                      | Full (PIT)                 | Partial                     | None                        |
| **Futures Handling**  | Full (Rolls, Continuous)       | None                 | Partial                     | Full (PIT)                 | Full                        | Basic                       |
| **Order Types**       | All standard \+ Bracket/OCO    | Signals only         | All standard \+ Bracket/OCO | All standard \+ custom     | All standard \+ Contingency | All standard                |
| **Fill Models**       | Pluggable, Partial Fills       | N/A                  | Basic                       | Pluggable, Partial Fills   | Pluggable, Partial Fills    | Pluggable, Partial Fills    |
| **Slippage Models**   | Pluggable (Fixed, Volume)      | Basic fees           | Pluggable (Fixed, Perc)     | Pluggable (Volume, Impact) | Pluggable                   | Basic                       |
| **Market Impact**     | Pluggable (Almgren-Chriss)     | None                 | None                        | Yes (VWAP, Custom)         | None                        | None                        |
| **Portfolio Acct.**   | Multi-Asset, Multi-Currency    | Single Currency      | Multi-Asset                 | Multi-Asset                | Multi-Asset, Multi-Currency | Multi-Asset, Multi-Currency |
| **Margin/Leverage**   | Full simulation, Margin Calls  | Yes (Pro)            | Manual                      | Full simulation            | Full simulation             | Yes                         |
| **Borrow/Funding**    | Pluggable (Fees, Rates)        | None                 | Manual (Credit Interest)    | Manual                     | Full simulation             | Manual                      |
| **Reporting Suite**   | pyfolio-style, Plotly, Parquet | Interactive Plotly   | Matplotlib                  | Web UI, JSON               | empyrical, custom           | Web UI, quantstats          |
| **Extensibility**     | Full (pluggable models)        | Good                 | Good                        | Excellent                  | Excellent                   | Good                        |
| **Live Trading**      | Via Adapters                   | No                   | Yes                         | Yes                        | Yes                         | Yes (Paid)                  |

**Analysis of Parity vs. Differentiation:**

* **Must-Have Parity Set:** To be a credible replacement for Backtrader and Zipline, and to compete with
  QuantConnect/Lean, ml4t.backtest's MVP must include:
    * Support for daily and minute bars from standard formats (CSV, Parquet).
    * A comprehensive set of standard order types (Market, Limit, Stop).
    * Basic pluggable models for commissions and percentage-based slippage.
    * Robust single-currency, multi-asset portfolio accounting.
    * A pyfolio-style reporting suite with standard metrics and plots.
    * A clear, class-based Python API for strategy definition.
* **Key Differentiators (Post-MVP):** ml4t.backtest will distinguish itself by focusing on features where incumbents are weak
  or where the implementation can be made significantly more robust and user-friendly for ML workflows:
    * **Tick-level data handling** as a first-class feature.
    * **Built-in, realistic Market Impact models** (e.g., Almgren-Chriss).
    * **Full multi-currency accounting** with automated FX conversion.
    * **First-class modeling of borrow fees and crypto funding rates.**
    * **Rigorous, built-in PIT alignment for ML signals**, which is a manual and error-prone process in all other
      Python-native frameworks.

### **4\. Product Requirements (PRD-style)**

This section defines the users, use cases, and nonfunctional requirements that will guide the design and implementation
of ml4t.backtest.

#### **Users & Use Cases**

* **Persona 1: The ML Quant Researcher**
    * **Description:** A data scientist or researcher who specializes in creating predictive models (e.g., using
      scikit-learn, LightGBM, PyTorch) to generate trading signals. Their primary concern is research integrity and
      avoiding subtle forms of data leakage.
    * **Use Cases:**
        * Backtesting alpha signals generated from external ML models against high-frequency (tick or minute) data.
        * Evaluating the impact of transaction costs and market impact on the profitability of a high-turnover strategy.
        * Performing sensitivity analysis on model parameters and execution assumptions.
        * Ensuring that asynchronous data sources (e.g., daily fundamentals, hourly sentiment scores) are joined in a
          point-in-time correct manner.
* **Persona 2: The Systematic Trader**
    * **Description:** An experienced trader migrating from a legacy platform like Backtrader or Zipline. They are
      comfortable with event-driven logic and Python but require higher performance, greater reliability, and more
      realistic simulation capabilities.
    * **Use Cases:**
        * Porting existing strategies to a faster, more modern engine with minimal code changes.
        * Testing multi-asset strategies that involve complex order types like bracket or trailing stops.
        * Simulating the impact of margin requirements and borrow costs on a leveraged strategy.
        * Developing a single strategy codebase that can be used for both backtesting and eventual live deployment.
* **Persona 3: The Advanced Retail Trader / "Prosumer"**
    * **Description:** A sophisticated individual trader who uses a combination of technical analysis and quantitative
      methods. They value ease of use, high-quality visualizations, and a clear path from research to paper and live
      trading.
    * **Use Cases:**
        * Rapidly prototyping and testing indicator-based strategies (e.g., moving average crossovers, RSI
          mean-reversion).
        * Using the declarative configuration system to test simple strategies without writing extensive Python code.
        * Generating comprehensive, visually appealing performance reports to analyze strategy performance and
          drawdowns.
        * Connecting the engine to a retail-friendly broker like Alpaca for paper and live trading.

#### **Nonfunctional Requirements**

* **Performance:**
    * **Throughput:** The engine must process over 1,000,000 events per second on a single modern CPU core for a simple
      strategy (e.g., market data event in, no action out).
    * **Latency:** In a live trading context, the time from an event entering the engine to an order being dispatched to
      the broker adapter should be less than 1 millisecond.
    * **Memory Footprint:** A 10-year backtest of 500 equities on daily data should consume less than 1 GB of RAM. The
      engine must support out-of-core computation for datasets larger than available memory, leveraging Polars'
      streaming capabilities.
* **Determinism & Reproducibility:**
    * A backtest must be 100% reproducible. Given the same configuration, code, and versioned data inputs, the engine
      must produce bit-for-bit identical output artifacts.
    * Each run must generate a "run manifest" that captures all configuration parameters, versions of key libraries, and
      cryptographic hashes of all input data files to guarantee reproducibility.
* **Scalability:**
    * The engine must be architected to handle backtests comprising tens of millions of events without running out of
      memory (e.g., tick data for a universe of 50 assets over 5 years).
    * The design should not preclude future extension to distributed execution for large-scale parameter optimization.
* **Extensibility & Modularity:**
    * All key simulation components—including fee models, slippage models, market impact models, risk management rules,
      and portfolio optimizers—must be implemented as pluggable modules with clearly defined Python interfaces (Abstract
      Base Classes).
    * Users must be able to easily implement and substitute their own custom components without modifying the engine's
      core code.
* **Testability:**
    * The codebase must have a high degree of unit test coverage (\>90%).
    * The system must support a suite of deterministic replay tests, where known inputs produce known, verifiable
      outputs, covering complex scenarios like corporate actions and margin calls.

### **5\. System Architecture**

The architecture of ml4t.backtest is an event-driven, modular system designed for performance, realism, and extensibility. It
draws inspiration from the robust, decoupled designs of QuantConnect/Lean and Nautilus Trader 24 but is implemented with
a modern, Python-native data stack centered on

Polars and Apache Arrow. The core principle is separation of concerns, where each component has a single, well-defined
responsibility.

#### **System Diagram**

The following diagram illustrates the flow of data and events through the system's major components.

Code snippet

graph TD
subgraph Input Layer
A \--\> B{EventBus};
C \--\> B;
end

    subgraph Core Engine
        B \-- Events \--\> D\[Clock & Calendar\];
        D \-- Tick \--\> E;
        E \-- SignalEvent \--\> B;
        B \-- SignalEvent \--\> F\[PortfolioConstruction\];
        F \-- TargetPortfolio \--\> G;
        G \-- AdjustedTargets \--\> H;
    end

    subgraph Simulation & State
        H \-- OrderEvent \--\> I;
        I \-- FillEvent \--\> J\[PortfolioAccounting\];
        J \-- StateUpdate \--\> E;
        H \-- FillEvent \--\> B;
    end

    subgraph Output Layer
        B \-- All Events \--\> K;
        K \--\> L;
    end

    subgraph Pluggable Modules
        M \--\> H;
        N\[FeeModels\] \--\> H;
        O\[PortfolioOptimizers\] \--\> F;
    end

    style E fill:\#f9f,stroke:\#333,stroke-width:2px
    style J fill:\#ccf,stroke:\#333,stroke-width:2px

#### **Modules & Interfaces**

Each component is defined by a Python Abstract Base Class (ABC), ensuring a clean, pluggable interface.

* **Event & EventBus:**
    * Event: A simple data class serving as the base for all system messages. Key subclasses include MarketEvent,
      SignalEvent, OrderEvent, FillEvent, and CorporateActionEvent.
    * EventBus: A central, high-performance queue (e.g., collections.deque) that orchestrates the flow of events.
      Components publish events to the bus, and the main loop dispatches them to subscribed listeners.
* **Clock & Data Handling:**
    * Clock: The master timekeeper of the simulation. It polls all registered DataFeed and SignalSource instances to
      find the chronologically next event, advances the simulation time to that event's timestamp, and places the event
      on the EventBus. This mechanism is fundamental to handling asynchronous data streams and guaranteeing
      point-in-time correctness.30
    * DataFeed(ABC): The interface for all market data sources.
      Python
      from abc import ABC, abstractmethod
      from typing import Optional, Generator

      class DataFeed(ABC):
      @abstractmethod
      def get\_next\_event(self) \-\> Optional\[Event\]:
      """Returns the next event from this feed, or None if exhausted."""
      pass

    * SignalSource(ABC): A similar interface for ingesting external ML predictions.
* **Strategy & Logic:**
    * Strategy(ABC): The user-defined class containing the trading logic. It subscribes to events and reacts by
      generating SignalEvents.
      Python
      class Strategy(ABC):
      def \_\_init\_\_(self, broker: 'Broker', data: 'PITData'):
      self.broker \= broker
      self.data \= data

          def on\_start(self): pass
          def on\_event(self, event: Event): pass
          def on\_stop(self): pass

* **Execution & Simulation:**
    * PortfolioConstruction(ABC): Consumes SignalEvents and translates them into a target portfolio state (e.g., "hold
      50% AAPL, \-20% GOOG"). This layer can incorporate portfolio optimization logic.
    * RiskManager(ABC): A pipeline of checks that can veto or modify the target portfolio based on rules (e.g., max
      position size, sector exposure limits).
    * Broker(ABC): Simulates the broker. It receives adjusted portfolio targets and is responsible for managing the
      lifecycle of orders required to reach that target. It uses pluggable models for fills, slippage, and costs.
    * Order: A data class representing a single order with properties like asset, quantity, type, status, limit\_price,
      etc.
    * ExecutionModel(ABC): A pluggable model that defines how to translate a portfolio target change into concrete
      orders (e.g., a single market order vs. a TWAP execution).
    * FillModel(ABC), SlippageModel(ABC), ImpactModel(ABC): Pluggable modules used by the Broker to simulate realistic
      fills.
* **State & Reporting:**
    * PortfolioAccounting: Tracks all state: cash balances per currency, positions, realized/unrealized P\&L. It updates
      its state in response to FillEvents.
    * Metrics(ABC): Listens to the EventBus to calculate performance and risk statistics throughout the backtest.
    * Reporter(ABC): Consumes the final state from PortfolioAccounting and Metrics to generate the final output
      artifacts (HTML reports, Parquet files).

#### **Data Model & Schemas**

To achieve high performance and enable zero-copy data transfer between Python and potential Rust components, all event
and data schemas will be strictly defined using Apache Arrow. Polars will be used as the primary DataFrame API for its
native Arrow support.

* **Core Event Schema:** All events will share a common header.
    * timestamp: Timestamp(nanosecond, timezone='UTC')
    * event\_type: Dictionary(UInt8, String)
* **MarketEvent (Trade):**
    * asset\_id: UInt64
    * price: Float64
    * size: UInt64
* **MarketEvent (Quote):**
    * asset\_id: UInt64
    * bid\_price, ask\_price: Float64
    * bid\_size, ask\_size: UInt64
* **SignalEvent:**
    * asset\_id: UInt64
    * score: Float64
    * model\_id: String
    * arrival\_timestamp: Timestamp(nanosecond, timezone='UTC') (to model latency)
    * metadata: Map(String, String)

This columnar, Arrow-native approach ensures that data can be passed between different system components, including
across language boundaries, without expensive serialization or copying, which is a critical foundation for the engine's
performance targets.

### **6\. Data Handling & Calendars**

Robust and accurate data handling is the bedrock of a reliable backtesting engine. ml4t.backtest will implement a
sophisticated data management layer that correctly handles time, exchange rules, corporate actions, and derivatives.

* **Data Sources and Formats:** The primary data input format will be Apache Parquet due to its high performance,
  compression, and rich type support. The engine's DataFeed implementations will leverage Polars' lazy scanning
  capabilities to efficiently read only the required data from large Parquet datasets, including predicate pushdown to
  filter by date or asset ID at the storage level. Support for legacy CSV files will also be included.
* **Bar Types and Construction:** The engine will natively support standard time-based bars (e.g., tick, 1-minute,
  5-minute, 1-hour, 1-day). A flexible bar construction utility will be provided to allow users to create custom bars
  from tick data, including tick bars, volume bars, and dollar bars. This is crucial for strategies sensitive to market
  activity rather than wall-clock time.
* **Calendars and Time Management:**
    * **Timezone Policy:** All internal timestamps will be strictly handled as timezone-aware UTC Timestamp\[ns\]. This
      avoids ambiguity and ensures correct alignment across global assets. Data inputs will be converted to UTC upon
      ingestion.
    * **Exchange Calendars:** The Clock module will be tightly integrated with the pandas-market-calendars library. For
      each asset, the engine will load the corresponding exchange calendar to accurately determine valid trading
      sessions, market open/close times, early closes, and holidays. The Clock will use this information to skip
      non-trading periods and correctly align bar timestamps to session boundaries. This prevents unrealistic trading
      during market closures, a common error in simpler backtesters.
* **Corporate Actions:**
    * A dedicated CorporateActionManager will handle adjustments for splits, dividends, and mergers to ensure
      point-in-time correctness.
    * **Data Representation:** Price data will be provided in both adjusted and unadjusted forms. The Broker simulation
      will always use **unadjusted** prices for matching orders, as this reflects the actual prices at which trades
      occur.
    * **Event Handling:** Corporate actions will be treated as first-class events on the EventBus.
        * On a SplitEvent, the CorporateActionManager will instruct the PortfolioAccounting module to adjust the
          quantity of any holdings for the affected asset. For example, a 2-for-1 split will double the number of shares
          held.
        * On a DividendEvent, the cash balance in the portfolio will be increased by the dividend amount multiplied by
          the number of shares held on the ex-date.
    * This event-driven approach, modeled on QuantConnect's robust implementation 7, ensures that adjustments are
      applied at the correct point in time and avoids look-ahead bias inherent in using pre-adjusted data for all
      purposes.
* **Futures and Continuous Contracts:**
    * A FuturesManager module will be responsible for handling futures contract rolls. Users will define a
      ContinuousFuture asset, specifying the underlying root symbol (e.g., 'ES') and a rolling rule (e.g., "roll 5
      business days before expiry to the next active contract").
    * During the backtest, the FuturesManager will monitor the active contract. When the rolling rule is triggered, it
      will automatically generate orders to close the position in the expiring contract and open an equivalent position
      in the new contract, accounting for any price differences. This provides a far more realistic simulation than
      simply using a back-adjusted continuous price series, which can mask the P\&L impact of rolling. This is a known
      area of weakness in older engines like Zipline that ml4t.backtest will explicitly address.32

### **7\. Strategy Definition & Config**

ml4t.backtest will offer two complementary methods for defining strategies and configuring backtests: a flexible Python API
for maximum power and a simple declarative configuration for ease of use and reproducibility.

#### **Python API**

The primary interface for strategy development will be a class-based Python API. This design provides familiarity for
users coming from Backtrader or Zipline and offers the full power of the Python language for implementing complex,
stateful logic.

The Strategy base class will expose a simple set of lifecycle methods and provide access to the core engine components
via instance attributes.

**Example Strategy Skeletons:**

1. **Simple Indicator-Based Strategy:**
   Python
   from ml4t.backtest import Strategy, Event, MarketEvent
   from ml4t.backtest.indicators import SMA

   class MovingAverageCross(Strategy):
   def on\_start(self):
   \# Subscribe to daily bars for SPY
   self.subscribe(asset="SPY", event\_type="BAR", frequency="1D")
   \# Initialize indicators
   self.slow\_ma \= SMA(period=200)
   self.fast\_ma \= SMA(period=50)

       def on\_event(self, event: Event):
           if isinstance(event, MarketEvent) and event.asset \== "SPY":
               \# Update indicators with the latest close price
               slow\_val \= self.slow\_ma.update(event.bar\['close'\])
               fast\_val \= self.fast\_ma.update(event.bar\['close'\])

               if fast\_val \> slow\_val:
                   \# Target 100% allocation to SPY
                   self.broker.order\_target\_percent(asset="SPY", percent=1.0)
               else:
                   \# Exit position
                   self.broker.order\_target\_percent(asset="SPY", percent=0.0)

2. **ML Signal-Based Strategy:**
   Python
   from ml4t.backtest import Strategy, Event, SignalEvent

   class MLSignalStrategy(Strategy):
   def on\_start(self):
   \# Subscribe to signals from a specific model
   self.subscribe(event\_type="SIGNAL", model\_id="my\_xgboost\_v2")

       def on\_event(self, event: Event):
           if isinstance(event, SignalEvent):
               \# Access point-in-time data for the asset
               last\_price \= self.data.get\_latest\_price(event.asset)

               if event.score \> 0.75 and last\_price:
                   \# Buy if signal is strong and we have a recent price
                   self.broker.order\_target\_value(asset=event.asset, value=10000)
               elif event.score \< 0.25:
                   \# Liquidate position on a weak signal
                   self.broker.order\_target\_value(asset=event.asset, value=0)

#### **Declarative Configuration (TOML/YAML)**

For reproducibility and separation of concerns, the entire backtest environment—including data sources, broker settings,
and even simple strategies—will be definable in a TOML or YAML file. This allows users to change parameters without
altering Python code and ensures that every backtest run is fully specified by its configuration.

**Example config.toml:**

Ini, TOML

\[backtest\]
start\_date \= "2020-01-01"
end\_date \= "2024-12-31"
base\_currency \= "USD"
run\_manifest \= "run\_manifest.json"

\[broker\]
initial\_cash \= 100000.0
commission\_model \= { type \= "FixedPerShare", fee \= 0.005 }
slippage\_model \= { type \= "VolumeShare", volume\_limit \= 0.025, price\_impact \= 0.1 }

\[\[data\_feeds\]\]
name \= "us\_equity\_daily"
type \= "ParquetFeed"
path \= "/data/us\_equity\_daily.parquet"

\[\[signal\_sources\]\]
name \= "ml\_signals"
type \= "ParquetSignalSource"
path \= "/data/signals\_v2.parquet"

\# A simple strategy can be defined directly in the config
\[\[strategies\]\]
name \= "SimpleSignalFollower"
type \= "DeclarativeStrategy"
rules \= \[
{ signal\_source \= "ml\_signals", condition \= "score \> 0.7", action \= "target\_percent", asset \= "signal.asset",
percent \= 1.0 },
{ signal\_source \= "ml\_signals", condition \= "score \< 0.3", action \= "target\_percent", asset \= "signal.asset",
percent \= 0.0 }
\]

**Guidelines for Use:**

* **Declarative Config is Sufficient For:**
    * Defining the entire simulation environment (dates, cash, fees, slippage).
    * Specifying all data inputs.
    * Implementing simple, stateless strategies that react directly to a signal (e.g., "if signal \> X, buy").
* **Python API is Required For:**
    * Strategies with internal state (e.g., tracking a custom portfolio metric).
    * Strategies that use path-dependent logic (e.g., trailing stops).
    * Strategies that compute custom indicators or features on the fly.
    * Any logic that cannot be expressed as a simple set of conditional rules.

### **8\. Execution, Orders & Fills**

A high-fidelity backtester must realistically model the entire lifecycle of an order, from its creation to its final
execution, accounting for market microstructure effects, costs, and constraints. ml4t.backtest's execution layer is designed
to be both comprehensive and extensible.

#### **Supported Order Types**

To achieve parity with market leaders like QuantConnect and Backtrader, ml4t.backtest will support a full suite of order
types:

* **Basic:** Market, Limit, Stop (Stop-Market), StopLimit.
* **Time-in-Force:** GTC (Good 'Til Canceled), DAY, IOC (Immediate or Cancel), FOK (Fill or Kill).
* **Advanced:**
    * TrailingStop: A stop order where the stop price trails the market price by a fixed amount or percentage.
    * Bracket: An OCO (One-Cancels-Other) order that brackets an entry order with a TakeProfit (Limit) and a StopLoss (
      Stop) order.

#### **Execution and Fill Modeling**

The Broker module simulates the matching of orders against market data. Its behavior is governed by a set of pluggable
models.

* **Order State Machine:** Every order progresses through a well-defined state machine: CREATED \-\> SUBMITTED \-\>
  ACCEPTED \-\> (PARTIALLY\_FILLED\*) \-\> FILLED / CANCELED / REJECTED. This ensures that the status of every order is
  tracked precisely.
* **Fill Logic:**
    * **Tick-Level Data:** For backtests running on tick data, limit orders are filled when the market price crosses the
      order's limit price (ask \<= limit\_price for a buy; bid \>= limit\_price for a sell).
    * **Bar-Level Data:** For backtests on bar data (e.g., minute or daily), a conservative fill logic is applied. A buy
      limit order is assumed to fill at the limit price if bar.low \<= limit\_price. A market order is filled at
      bar.open of the next bar.
    * **Partial Fills:** The engine will model partial fills. If an order's size exceeds a configurable liquidity cap (
      e.g., 2.5% of the bar's volume), only a portion of the order will be filled in that bar, and the remainder will be
      carried over.
* **Liquidity and Discretization:**
    * **Liquidity Caps:** Orders can be constrained by historical volume. A LiquidityModel can be configured to reject
      or partially fill orders that exceed a certain percentage of the average daily volume or the current bar's volume.
    * **Price/Size Discretization:** The engine will respect asset-specific contract specifications, such as tick size
      and lot size, rounding order prices and sizes to valid increments.

#### **Realism Modules: Costs and Constraints**

* **Shorting and Borrows:** The PortfolioAccounting module will track short positions. A pluggable BorrowFeeModel will
  calculate and deduct borrow costs from the cash balance on a daily basis, using either a fixed rate or a dynamic feed
  of borrow rates.
* **Margin and Leverage:** A MarginModel, inspired by QuantConnect's implementation 34, will continuously monitor the
  portfolio's maintenance margin. If the portfolio value drops below the required margin, it will trigger a
  MarginCallEvent, and a ForcedLiquidationModel will automatically liquidate positions to bring the account back into
  compliance.
* **Cryptocurrency Perpetuals Funding:** For crypto assets, a FundingRateModel will be supported. This model will apply
  periodic funding payments to or from the portfolio's cash balance based on the size of the perpetual futures position,
  accurately simulating a key cost in crypto trading.
* **FX Conversion:** For multi-currency strategies, all transactions will be recorded in the asset's native currency. A
  ForexRateFeed will provide real-time conversion rates, and the Broker will automatically handle currency conversions
  for trades where the asset currency differs from the account's cash currency, applying a configurable spread.
* **Pluggable Slippage and Market Impact Models:**
    * **Slippage:** The engine will ship with standard SlippageModel implementations, including FixedPercentageSlippage
      and a VolumeShareSlippage model where slippage increases as the order's size relative to the bar's volume grows.
      This is modeled after the QuantConnect VolumeShareSlippageModel.36
    * **Market Impact:** A key differentiator will be the inclusion of a built-in, academically grounded
      MarketImpactModel. The reference implementation will be based on the Almgren-Chriss "square-root" model, where the
      price impact is proportional to the square root of the order size relative to the average daily volume.36 This
      provides a much more realistic cost estimate for large orders than simple slippage models. The API will be
      pluggable, allowing users to implement their own, more sophisticated impact models.

### **9\. Accounting & Portfolio**

The PortfolioAccounting module is the authoritative source of truth for all portfolio state, including cash, positions,
and performance. It is designed for accuracy, supporting multi-currency operations and providing hooks for advanced
portfolio management techniques.

* **State Management:** The module maintains a real-time record of:
    * **Cash Book:** A dictionary mapping each currency (e.g., USD, EUR, JPY) to its current cash balance.
    * **Positions:** A collection of Position objects, each tracking the quantity, average entry price, and current
      market value of an asset holding.
    * **Net Asset Value (NAV):** The total value of the portfolio (sum of all cash balances and position market values,
      converted to a single base currency), calculated at the end of each trading day to produce the equity curve.
* **P\&L Calculation:**
    * **Unrealized P\&L:** Calculated continuously for open positions based on the current market price versus the
      average entry price.
    * **Realized P\&L:** Calculated whenever a position is closed or reduced. The calculation is done on a First-In,
      First-Out (FIFO) basis.
    * **Cost Basis:** All P\&L calculations will fully account for all transaction costs, including commissions, fees,
      slippage, market impact, and borrow/funding costs, providing a true net performance figure.
* **Rebalance Hooks and Portfolio Optimization:**
    * The Strategy API will provide specific hooks, such as on\_daily\_close() or on\_rebalance\_trigger(), that are
      designed for portfolio rebalancing logic.
    * To integrate with external portfolio optimization libraries like Riskfolio-lib or PyPortfolioOpt, a dedicated
      PortfolioOptimizer adapter interface will be defined.
    * **Workflow:**
        1. Inside a rebalance hook, the strategy provides the optimizer adapter with a list of target assets and the
           current portfolio state.
        2. The adapter formats this data and passes it to the external library (e.g., Riskfolio-lib) to solve for the
           optimal weights based on a specified objective (e.g., maximize Sharpe ratio, minimize variance).
        3. The optimizer returns the target weights.
        4. The strategy then uses a broker.order\_target\_weights() method, which automatically generates the necessary
           trades to align the current portfolio with the new target weights.
    * This design cleanly separates the concerns of signal generation, portfolio optimization, and order execution,
      promoting modular and reusable code.

### **10\. ML-Signal Ingestion & Leakage Safety**

The primary differentiator of ml4t.backtest is its first-class, architecturally-enforced support for backtesting machine
learning models without data leakage. The core challenge, known as point-in-time (PIT) correctness, arises when
combining multiple data sources with different timestamps and frequencies. A naive join of this data will invariably
leak future information into the past, leading to unrealistic backtest results.38 ml4t.backtest solves this problem at the
architectural level.

#### **Signal Schema**

To accurately model the information flow from an ML model, signals will be represented with a detailed schema. This
allows the backtester to account for factors like data processing and model inference latency.

* **Schema Definition:**
    * ts\_event: The timestamp of the market data that the ML model used to make its prediction. This is the "knowledge
      timestamp."
    * ts\_arrival: The timestamp when the signal becomes available to the trading strategy. The delta between
      ts\_arrival and ts\_event represents the total data-to-signal latency.
    * asset\_id: The asset the signal pertains to.
    * score: The raw output of the model (e.g., a probability, a predicted return).
    * confidence: An optional measure of the model's confidence in the prediction.
    * version: A string identifying the version of the ML model that generated the signal.
    * model\_id: A unique identifier for the model.

#### **Alignment Utilities and Leakage Safety**

ml4t.backtest guarantees no look-ahead bias through its core event-driven architecture and specialized data access utilities.

* **Architectural Guarantee:** The Clock module is the ultimate arbiter of time. It processes all events (market data,
  signals, corporate actions) in strict chronological order based on their timestamps. A Strategy at simulation time T
  is therefore architecturally incapable of seeing an event with a timestamp greater than T.
* **The PITData Object:** To simplify data access for the user while maintaining rigor, the Strategy will be provided
  with a PITData object. This object acts as an abstraction layer over all historical data streams.
    * **Functionality:** When a strategy requests data (e.g., self.data.get\_latest\_bar("AAPL") or
      self.data.get\_feature("sentiment", "AAPL")), the PITData object queries the underlying data store for the most
      recent entry whose timestamp is less than or equal to the current simulation time.
    * **Example PIT Join:**
        * Simulation Time: 2025-07-15 10:30:05.123
        * Strategy calls: self.data.get\_feature("daily\_fundamentals", "AAPL")
        * The PITData object looks up the "daily\_fundamentals" data stream and finds the latest entry with a
          timestamp \<= 2025-07-15 10:30:05.123. This would typically be the data released at the close of the previous
          day, 2025-07-14 16:00:00.000. This prevents the strategy from incorrectly using fundamental data for July 15th
          before it would have been released.
* **Embargo Rules:** The system will support embargo periods. For example, a signal generated from the close of day D
  can be configured to only become available for trading at the open of day D+1, preventing unrealistic trading on
  closing prices. This is handled by setting the ts\_arrival in the SignalEvent appropriately.
* **Streaming vs. Batch Ingestion:**
    * **Historical Batch:** For backtesting, signals are typically pre-computed and stored in a Parquet file. The
      ParquetSignalSource reads this file and feeds SignalEvents into the Clock's priority queue.
    * **Streaming Ingestion:** For paper and live trading, the architecture supports streaming sources. A
      KafkaSignalSource or WebsocketSignalSource could listen to a message queue or API endpoint, converting incoming
      messages into SignalEvents and adding them to the event stream in real-time.

### **11\. Performance & Engineering Plan**

ml4t.backtest is designed to deliver performance competitive with engines written in compiled languages, while retaining the
flexibility and productivity of the Python ecosystem. This is achieved through a modern data stack and targeted
application of acceleration technologies.

#### **Hot Paths and Acceleration Strategy**

The performance-critical components of the backtester will be aggressively optimized.

* **Core Event Loop and Matching Engine:** The main event processing loop, which dispatches events from the EventBus,
  and the order matching logic within the Broker are the primary performance bottlenecks. These components involve tight
  loops and simple data structure manipulations. They are ideal candidates for implementation in **Rust**, exposed to
  Python via PyO3. This follows the successful pattern of Nautilus Trader and provides C-level speed and memory safety
  for the engine's core.3
* **Data Handling:** All data loading, filtering, and aggregation will be delegated to **Polars**. Its lazy execution
  engine, query optimizer, and highly efficient Rust backend will ensure that data I/O is not a bottleneck. Using
  predicate pushdown on Parquet files will allow the engine to read only the necessary row groups and columns from disk.
* **Strategy Logic and Indicators:** User-defined strategy logic, particularly functions that perform calculations over
  rolling windows of data (e.g., custom indicators), will be JIT-compiled using **Numba**. By decorating these functions
  with @numba.jit, slow Python loops can be converted into highly efficient machine code at runtime.
* **Micro-batching/Event Chunking:** For vectorized-like operations within the event-driven framework, events can be
  processed in small chunks (e.g., all events within a 1-second window) instead of one by one. This allows for more
  efficient use of vectorized Polars or NumPy operations within the strategy logic, blending the performance of
  vectorized systems with the realism of an event loop.

#### **Memory Layout and Zero-Copy Data Flow**

* **Apache Arrow:** Apache Arrow will be the standard in-memory format for all tabular data. Polars uses Arrow as its
  native format. By ensuring that the Rust core also uses Arrow, data can be passed between the Python and Rust layers
  via memory pointers without any serialization or data copying. This **zero-copy** data flow is critical for
  eliminating overhead and achieving maximum throughput.

#### **Determinism and Reproducibility**

Ensuring that a backtest can be perfectly reproduced is a non-negotiable requirement for serious research.

* **Seeding:** All sources of randomness in the simulation (e.g., in slippage models) will be derived from a single
  pseudo-random number generator (PRNG) that is initialized with a master seed at the start of the backtest.
* **Stable Sorting:** Any operations that involve sorting (e.g., prioritizing orders) must use a stable sorting
  algorithm to ensure consistent order when values are equal.
* **Run Manifest and Artifacts:** At the conclusion of each backtest, a **run manifest** (a YAML or JSON file) will be
  saved as a primary artifact. This file will contain:
    * The full configuration used for the run.
    * The master seed for the PRNG.
    * Cryptographic hashes (e.g., SHA-256) of all input data files (prices, signals, features).
    * Version information for ml4t.backtest and key dependencies (polars, numba, etc.).
      This manifest provides a complete "fingerprint" of the backtest, allowing anyone to replicate the results exactly.

#### **Benchmarks and Targets**

A suite of performance benchmarks will be developed to track and validate the engine's performance.

* **Micro-benchmarks:** Measure the speed of individual components (e.g., event bus dispatch time, order matching
  latency).
* **Macro-benchmarks:**
    * **Workload 1 (High-Frequency):** Backtest a simple strategy on 1 year of tick data for a single asset (\~20-30
      million events). **Target: \< 30 seconds.**
    * **Workload 2 (Broad Universe):** Backtest a daily rebalancing strategy on 10 years of daily bar data for 1000
      assets (\~2.5 million events). **Target: \< 10 seconds.**

### **12\. Testing & Validation**

A rigorous and multi-layered testing strategy is essential to ensure the correctness, reliability, and determinism of
ml4t.backtest.

* **Unit Tests:** Each module and class will have a comprehensive suite of unit tests written using pytest. These tests
  will verify the logic of individual components in isolation. All public methods of the interfaces defined in the
  architecture (e.g., DataFeed, Broker, SlippageModel) will be thoroughly tested. A code coverage target of \>90% will
  be enforced via CI.
* **Integration Tests:** These tests will verify the interactions between components. For example, an integration test
  would confirm that a SignalEvent generated by a Strategy is correctly received by the PortfolioConstruction module,
  which then generates a TargetPortfolio that is passed to the Broker.
* **Scenario Tests (Deterministic Replay):** This is the most critical layer of testing for ensuring correctness. A
  collection of "gold standard" test scenarios will be created. Each scenario will consist of:
    * A set of input data files (prices, corporate actions, signals).
    * A strategy file and configuration.
    * A pre-computed set of "golden" output artifacts (trade logs, portfolio NAV series) that represent the known
      correct result.
      The CI pipeline will run these backtests and perform a bit-for-bit comparison of the generated artifacts against
      the golden files. Any discrepancy indicates a regression. These scenarios will cover complex edge cases,
      including:
    * Correct handling of stock splits and dividends.
    * Accurate futures contract rolling.
    * Triggering of margin calls and forced liquidations.
    * Interactions between complex order types (e.g., a trailing stop being triggered before a limit order).
    * Handling of trading halts and irregular market sessions.
* **Property-Based Tests:** Libraries like Hypothesis will be used to test components against a vast range of
  automatically generated, often unexpected, inputs. This is particularly useful for stress-testing data parsers and
  ensuring that numerical calculations within the portfolio and risk modules are robust to edge cases like zero or
  negative prices/volumes.

### **13\. Reporting & Visualization**

The output of a backtest should be comprehensive, easy to interpret, and suitable for further analysis. ml4t.backtest will
produce a standardized set of artifacts for each run.

#### **Output Artifacts and Formats**

* **Data Formats:** To facilitate programmatic analysis, all primary outputs will be saved in the high-performance
  Apache Parquet format. This includes:
    * trades.parquet: A detailed log of every executed trade, including entry/exit timestamps, prices, size, P\&L, fees,
      and slippage.
    * orders.parquet: A log of every order submitted, including its status transitions.
    * portfolio\_history.parquet: A daily snapshot of portfolio metrics (NAV, cash, exposure, leverage).
    * event\_log.parquet: An optional, verbose log of all events that passed through the EventBus for deep debugging.
* **Human-Readable Report:** A self-contained HTML file will be generated as the primary visual report. This report will
  use Plotly for interactive charts and will be suitable for sharing and archiving.
* **Run Manifest:** A run\_manifest.json file containing all configuration and data hashes to ensure reproducibility, as
  detailed in Section 11\.

#### **Report Content**

The HTML report will include a comprehensive suite of tables and plots, similar to those provided by pyfolio-reloaded
and other institutional reporting tools.

* **Key Tables:**
    * **Performance Summary:** A table of standard risk/return metrics (Total Return, Sharpe Ratio, Sortino Ratio,
      Calmar Ratio, Max Drawdown, Annualized Volatility, etc.).
    * **Trade Statistics:** Win Rate, Average Win/Loss, Profit Factor, etc.
    * **Configuration Summary:** A human-readable display of the parameters from the run manifest.
* **Required Plots:**
    * **Equity Curve:** An interactive plot of the portfolio's NAV over time, with periods of significant drawdown
      highlighted.
    * **Returns Distribution:** A histogram of daily/monthly returns.
    * **Drawdown Underwater Plot:** A chart showing the depth and duration of all drawdowns.
    * **Rolling Volatility and Sharpe Ratio:** Plots showing how these metrics evolved over the backtest period.
    * **Asset Allocation:** A stacked area chart showing the portfolio's exposure to different assets over time.
* **Advanced Features:**
    * **Run Comparison:** A utility will be provided to generate a side-by-side comparison report for two different
      backtest runs, highlighting differences in performance and trades.
    * **Factor Attribution:** If the user provides factor data (e.g., P/E ratio, momentum score), the report can include
      basic attribution analysis, showing the P\&L contribution of different factors.

### **14\. Live/Paper Trading Path**

A key architectural goal for ml4t.backtest is to provide a seamless transition from historical backtesting to paper and live
trading. The event-driven architecture is ideally suited for this, as the core strategy and portfolio logic can remain
identical across all environments.30 The transition is achieved by swapping out a few key modules.

#### **Minimal Delta Architecture**

The difference between a backtesting environment and a live trading environment is primarily the source of time and the
destination of orders.

| Component    | Backtesting Mode                                                | Live/Paper Trading Mode                                                   |
|:-------------|:----------------------------------------------------------------|:--------------------------------------------------------------------------|
| **Clock**    | Master of time; advances based on historical event timestamps.  | Slave to wall time; synchronizes with real-world time.                    |
| **DataFeed** | Reads from historical data files (e.g., ParquetDataFeed).       | Connects to a broker's real-time market data stream (e.g., IBKRDataFeed). |
| **Broker**   | A simulation (SimulationBroker) that matches orders internally. | An adapter (LiveBroker) that sends orders to a real brokerage API.        |

The core components—EventBus, Strategy, PortfolioConstruction, RiskManager, and PortfolioAccounting—require **no changes
**. This "backtest/live parity" is a crucial feature that reduces the risk of bugs and behavioral differences when
deploying a strategy.

#### **Live Trading Adapters**

To enable live trading, a set of broker-specific adapters must be implemented. Each adapter will consist of two parts:

1. **A DataFeed implementation:** This component will connect to the broker's market data WebSocket/API, receive
   real-time ticks or bars, and transform them into ml4t.backtest's standard MarketEvent format before placing them on the
   EventBus.
2. **A Broker implementation:** This component will implement the Broker interface. When it receives an order from the
   engine, it will translate it into the broker's specific API format and submit it. It will also listen for execution
   reports from the broker, converting them into ml4t.backtest FillEvents to update the portfolio's state.

#### **Live Trading Considerations**

The live trading adapters must handle complexities not present in backtesting:

* **State Synchronization:** Upon startup, the live adapter must query the broker for current positions and cash
  balances to initialize the PortfolioAccounting module correctly.
* **Idempotency:** Order submission logic must be idempotent to prevent duplicate orders in case of network disconnects
  and reconnections. Each order should have a unique ID that can be used for reconciliation.
* **Failover and Reconciliation:** The adapter must have robust error handling for API failures and network issues. A
  reconciliation loop should periodically compare the engine's internal portfolio state with the broker's state to
  detect any discrepancies.

**Target Brokerages:**

* **Initial Adapters:** Alpaca (for its simple, modern API, ideal for retail users) and Interactive Brokers (for its
  broad market access and professional features).
* **Future Adapters:** Major cryptocurrency exchanges like Binance and Coinbase.

### **15\. Migration Guides**

To accelerate adoption, ml4t.backtest will provide clear migration paths and tools for users of the most popular legacy
backtesting libraries, Zipline and Backtrader. The goal is to lower the switching costs and demonstrate the clear
advantages of the new engine.

#### **API Compatibility Shims**

Optional "shim" or "adapter" layers will be created to allow users to run their existing Zipline and Backtrader
strategies on ml4t.backtest with minimal modifications.

* **Zipline Adapter:**
    * A base ZiplineCompatStrategy class will be provided that inherits from ml4t.backtest.Strategy.
    * This class will implement the familiar initialize(context) and handle\_data(context, data) methods.
    * Inside, it will map the Zipline API calls (e.g., order\_target\_percent, data.history) to their corresponding
      ml4t.backtest Broker and PITData methods.
    * This allows users to get their old strategies running quickly, though they will be encouraged to migrate to the
      native ml4t.backtest API to access advanced features.
* **Backtrader Adapter:**
    * Similarly, a BacktraderCompatStrategy class will replicate the \_\_init\_\_(self) and next(self) structure.
    * It will map Backtrader's data line access (self.data.close) and order methods (self.buy()) to the ml4t.backtest
      equivalents.

#### **Documentation and Examples**

The official documentation will include a dedicated "Migration" section with:

* **API Mapping Tables:** Side-by-side tables comparing common operations in Zipline/Backtrader with their ml4t.backtest
  equivalents.
* **Porting Examples:** Step-by-step tutorials showing how to port a canonical strategy (e.g., a dual moving average
  crossover) from each legacy platform to ml4t.backtest. These examples will explicitly highlight improvements, such as the
  performance gains from Numba JIT compilation or the increased realism from using a market impact model.
* **Conceptual Differences:** An explanation of the key architectural differences, such as ml4t.backtest's rigorous PIT data
  handling and its Polars-based data model, to educate users on the benefits of the new paradigm.

### **16\. Security, Packaging, and Ops**

The operational aspects of the library are critical for establishing trust, ensuring ease of use, and fostering a
healthy open-source community.

* **Packaging and Distribution:**
    * The primary distribution channel will be the Python Package Index (PyPI).
    * The package will be named ml4t.backtest (or a suitable alternative).
    * Pre-compiled binary wheels will be built for all major platforms (Windows, macOS, Linux) and Python versions. This
      is especially important if Rust extensions are used, as it eliminates the need for users to have a Rust compiler
      installed.
    * A conda-forge package will also be maintained for users in the Anaconda ecosystem.
* **Dependency Management:**
    * A minimal set of core dependencies (polars, pyarrow, numba, pandas-market-calendars) will be strictly
      version-pinned to ensure stable builds.
    * Optional features (e.g., reporting, specific broker adapters, portfolio optimizers) will be managed as "extras" in
      pyproject.toml. Users can install them as needed (e.g., pip install ml4t.backtest\[reporting,ibkr\]), which keeps the
      core installation lightweight.
* **Continuous Integration (CI) and Reproducible Builds:**
    * A CI pipeline (e.g., using GitHub Actions) will be set up to automatically run the full test suite (unit,
      integration, and scenario tests) on every commit and pull request.
    * The build process itself will be containerized using Docker to ensure it is fully reproducible.
* **Documentation Plan:**
    * Documentation will be built using Sphinx or a similar tool and hosted on a dedicated website.
    * It will include a quick-start tutorial, detailed user guides for each component, the migration guides, and a
      comprehensive, auto-generated API reference.
* **License Recommendation:**
    * **Apache License 2.0.** This is a permissive, business-friendly open-source license that is standard for major
      data science and infrastructure projects (e.g., Zipline, Spark, Polars). It allows for commercial use and
      modification without the copyleft restrictions of licenses like the GPL, which is crucial for encouraging adoption
      by both individuals and institutions.
* **Governance Model:**
    * The project will adopt a clear governance model outlined in a GOVERNANCE.md file.
    * A CONTRIBUTING.md file will provide clear guidelines for community contributions, including coding standards, pull
      request procedures, and a developer certificate of origin (DCO).
    * A public roadmap will be maintained to provide transparency into the project's future direction.

### **17\. Roadmap & Resourcing**

This phased roadmap outlines a realistic plan for developing ml4t.backtest, balancing speed to market with feature
completeness. It assumes a core team of experienced Python and systems engineers.

#### **Phase 1: MVP \- Core Engine (Duration: 3 Months; Team: 2 Engineers)**

* **Goal:** Build the fundamental, working skeleton of the backtester.
* **Milestones:**
    * Core event loop, Clock, and EventBus implemented in Python with Numba acceleration.
    * ParquetDataFeed for ingesting daily and minute bar data using Polars.
    * Basic Strategy API with on\_start, on\_event, and on\_stop hooks.
    * SimulationBroker supporting Market and Limit orders with a simple, immediate fill model.
    * PortfolioAccounting module for single-currency, multi-asset P\&L tracking.
    * Basic output of trade logs and daily portfolio history to Parquet files.
    * Establish CI/CD pipeline and unit testing framework.

#### **Phase 2: Parity \- Realistic Simulation (Duration: 4 Months; Team: 3 Engineers)**

* **Goal:** Achieve feature parity with legacy event-driven backtesters and add core ML support.
* **Milestones:**
    * Implement the full suite of required order types (Stop, Trailing, Bracket/OCO).
    * Develop the pluggable model architecture for Commission, Slippage, Margin, and BorrowFee models, with reference
      implementations for each.
    * Implement the CorporateActionManager and FuturesManager for handling splits, dividends, and contract rolls.
    * Integrate pandas-market-calendars into the Clock for accurate session handling.
    * Develop the ML SignalIngestor and the leakage-safe PITData access object.
    * Build the initial HTML reporting suite using Plotly.

#### **Phase 3: Differentiators \- Performance & UX (Duration: 3 Months; Team: 3 Engineers)**

* **Goal:** Implement the key performance and usability features that set ml4t.backtest apart.
* **Milestones:**
    * **Performance Optimization:** Profile the engine's hot paths and port the core event loop and matching engine to
      Rust/PyO3 for a significant performance boost.
    * **Market Impact Model:** Implement the reference Almgren-Chriss market impact model.
    * **Portfolio Optimization:** Create the adapter interface and reference implementation for Riskfolio-lib.
    * **Declarative Config:** Build the TOML/YAML configuration layer for defining environments and simple strategies.
    * **Migration Tools:** Develop the Zipline and Backtrader compatibility shims and write the migration guides.

#### **Phase 4: Live Trading & Ecosystem (Ongoing; Team: 2 Engineers)**

* **Goal:** Extend the engine to support paper and live trading and grow the ecosystem.
* **Milestones:**
    * Develop, test, and release the live trading adapter for **Alpaca**.
    * Develop, test, and release the live trading adapter for **Interactive Brokers**.
    * Develop adapters for major cryptocurrency exchanges (e.g., **Binance**).
    * Expand the library of reference strategies and community-contributed modules.

#### **Risk Register & Mitigations**

| Risk                                | Likelihood | Impact | Mitigation Strategy                                                                                                                                                                               |
|:------------------------------------|:-----------|:-------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Technical Complexity**            | Medium     | High   | Phased development approach (MVP first). Leverage mature, high-quality libraries (Polars, Arrow, Numba). Draw from proven architectural patterns of competitors.                                  |
| **Performance Targets Not Met**     | Low        | High   | Architect for performance from day one (Rust core, zero-copy). Implement continuous benchmarking to detect performance regressions early.                                                         |
| **Subtle Bugs in Simulation Logic** | High       | High   | Implement a comprehensive, multi-layered testing strategy with a focus on deterministic replay tests for complex, known-correct scenarios.                                                        |
| **Scope Creep**                     | Medium     | Medium | Adhere strictly to the phased roadmap. Defer non-essential features to later releases. Use the feature parity matrix as a clear definition of scope for each phase.                               |
| **Community Adoption is Slow**      | Medium     | Medium | Provide high-quality documentation, tutorials, and migration guides. Engage with the community early. The tie-in with the popular *ML for Trading* book provides a significant initial user base. |

### **18\. Go/No-Go & Business Rationale**

#### **Recommendation: GO**

The analysis presented in this document leads to a clear and confident **GO** recommendation to proceed with the
development of the ml4t.backtest backtesting engine.

#### **Business Rationale and Strategic Alignment**

The decision to build ml4t.backtest is fundamentally a strategic one. The *Machine Learning for Trading* book is a successful
asset with a large and engaged audience. However, its reliance on the aging and unmaintained Zipline library represents
a significant and growing liability. By developing a modern, high-performance successor, we not only mitigate this risk
but also create a powerful, self-owned platform that enhances the value of the entire ecosystem.

1. **Fills a Clear Market Gap:** As detailed in Section 2, the Python ecosystem lacks a tool that combines the
   high-fidelity simulation of an event-driven engine with the performance and modern data stack required for serious
   ML-based quantitative research. ml4t.backtest is precisely designed to fill this gap.
2. **Creates a Durable Competitive Advantage:** Owning the core backtesting engine provides a durable competitive
   advantage. It ensures that the book's curriculum can showcase cutting-edge, best-practice techniques without being
   constrained by the limitations of third-party tools. This reinforces the brand's position as a market leader.
3. **Enables Future Commercialization:** A robust, open-source engine can serve as the foundation for future commercial
   offerings. Potential avenues include enterprise support contracts, a hosted cloud backtesting platform (similar to
   QuantConnect's model), or a marketplace for strategies and data.
4. **Technical Feasibility:** The technical risks are well-understood and manageable. The emergence of powerful
   libraries like Polars, Arrow, and Numba, along with the proven success of Rust-in-Python architectures like Nautilus
   Trader, provides a clear and feasible path to achieving the project's ambitious performance and feature goals.

The estimated effort is significant, spanning approximately one year for the core development phases. However, the
resulting asset will be a cornerstone of the *ML for Trading* ecosystem for years to come, driving book sales, enabling
advanced courseware, and creating new consulting opportunities.

#### **Alternative Paths Considered and Rejected**

1. **Extend Zipline Reloaded:** This option was rejected due to the high technical debt of the Zipline codebase. Its
   architecture is not conducive to the required performance improvements, and the effort to refactor its complex,
   Cython-heavy core would likely exceed the effort of a clean-slate build, with a technologically inferior result.
2. **Contribute to/Fork an Existing Modern Engine (e.g., Nautilus Trader):** While Nautilus Trader is architecturally
   impressive, relying on it would mean ceding control over the project's roadmap and design priorities. Building
   ml4t.backtest ensures that it is perfectly tailored to the pedagogical needs of the book and the specific focus on
   leakage-safe ML workflows.
3. **Build Adapters Only:** This path, which involves building adapters for QFeatures and QEval to plug into existing
   backtesters, was rejected because it fails to solve the fundamental problems. Users would still be constrained by the
   performance limitations, lack of realism, and data leakage risks of the underlying legacy engines, thereby
   undermining the credibility and value proposition of the entire ecosystem.

In conclusion, building ml4t.backtest is the only path that fully aligns with the strategic objectives, addresses the needs of
the target audience, and creates a lasting, valuable asset.
