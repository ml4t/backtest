Here is a careful evaluation of the four proposals, a ranking, and a synthesis of the key items for a comprehensive
Product Requirements Document (PRD) targeting your specified audience.

## Evaluation of Proposals

To evaluate these four documents, I've developed a set of metrics that assess their strategic vision, technical depth,
and fitness as an actionable product specification.

* **Vision & Strategy:** Clarity of the problem, target audience, and strategic differentiation.
* **Architectural Soundness:** The quality, modernity, and robustness of the proposed technical design.
* **Specification & Actionability:** The level of detail and clarity in the requirements; whether a team could begin
  work from the document.
* **Realism & Risk:** Acknowledgment of hard problems (e.g., data leakage, multi-asset complexity) and a plan to address
  them.

---

### 1. SOTA (State-of-the-Art Event-Driven Backtesting Engine)

* **Summary:** This is a high-level strategic proposal and technical design document. Its core thesis is built on
  filling a market gap for a **Polars-first** engine specifically designed for **leakage-safe Machine Learning (ML)
  workflows**.
* **Vision & Strategy (Excellent):** Its greatest strength. The market analysis and competitor matrix are superb,
  clearly identifying a gap. The four differentiators (Polars-first, ML-leakage-safe, extensible realism, Python-native
  performance) are compelling and form a powerful "why."
* **Architectural Soundness (Excellent):S** The architecture is modern, sound, and forward-thinking. It correctly
  identifies the "hard parts": point-in-time (PIT) data alignment, corporate actions, and futures rolling, and it builds
  the architecture around solving them. The focus on Apache Arrow and zero-copy data flow is a sign of deep technical
  understanding.
* **Specification & Actionability (Good):** It contains a solid "PRD-style" section (Section 4) and good feature
  definitions. It's less a line-by-line spec than OPUS/GEM, but it provides a clear *direction* for that spec.
* **Realism & Risk (Excellent):** The entire document is centered on solving the *real*, subtle problems that invalidate
  backtests, especially data leakage (Section 10) and unrealistic execution (Section 8).

### 2. Multi-Asset Backtesting Engine

* **Summary:** This document is not a product proposal but a deep, academic-level *architectural treatise* on the
  complexities of building a true multi-asset system.
* **Vision & Strategy (Fair):** Its "problem statement" is simply that multi-asset is hard. It doesn't analyze a market
  or a user, but rather a *technical domain*.
* **Architectural Soundness (Excellent):** This is its sole purpose. Its analysis of the differences between asset
  classes (SPAN margin for futures, option greeks, bond P&L) is comprehensive and more detailed than any of the other
  proposals.
* **Specification & Actionability (Poor):** It is not actionable as a PRD. It is a reference guide, a "what you must be
  aware of" document. It provides no roadmap, no prioritization, and no specific product.
* **Realism & Risk (Excellent):** It is the *definition* of realistic, outlining all the complex edge cases (
  survivorship bias, time zones, MTM P&L, etc.) that other engines often ignore.

### 3. OPUS

* **Summary:** This is a classic, comprehensive, and highly actionable Product Requirements Document (PRD). It is a
  direct *blueprint* for an engineering team.
* **Vision & Strategy (Good):** It has clear principles ("Deterministic," "Config-First") and good user personas. It's
  less focused on a novel market *gap* (like SOTA) and more on *executing* a known good product (a Zipline replacement)
  with modern technology.
* **Architectural Soundness (Good):** It proposes the correct modern stack (Polars/Numba) and a standard event-driven
  architecture. It's sound and practical.
* **Specification & Actionability (Excellent):** This is its key strength. The detailed functional requirements (
  5.1-5.7), clear API sketches (Section 7), and concrete fill rules (Section 9) are outstanding. The v0/v1/v2 roadmap is
  clear. A team could start work on this immediately.
* **Realism & Risk (Excellent):** Its testing and validation strategy (Section 14) is superb, explicitly calling out
  unit, property-based, and "golden" tests. It also has a good "Risks & Open Questions" section.

### 4. GEM

* **Summary:** This is also a classic, comprehensive PRD, remarkably similar in quality and structure to OPUS.
* **Vision & Strategy (Good):** Like OPUS, it has clear goals, principles, and a strong v0/v1/v2 roadmap. Its
  formalization of the event-loop priority (Section 8) is a nice touch, showing a deep focus on determinism.
* **Architectural Soundness (Good):** Also proposes the Polars/Numba stack and a sound event-driven core.
* **Specification & Actionability (Excellent):** Virtually tied with OPUS. The API sketches, config schema examples, and
  detailed requirements are a complete specification.
* **Realism & Risk (Excellent):** Also has a very strong testing section (Section 14) and a good analysis of risks,
  including Numba's JIT overhead and Polars' API velocity.

---

### 📊 Comparative Ranking

1. **SOTA (State-of-the-Art):** This is the **best *proposal***. It provides the strongest strategic vision, market
   justification, and novel architectural insight. It answers "Why should we build this?" with a compelling, modern
   answer (ML-first).
2. **OPUS / GEM (Tie):** These are the **best *specifications***. They are the blueprints. After agreeing on the
   vision (from SOTA), you would use one of these documents to *manage the project*. They answer "What, exactly, are we
   building, and how?"
3. **Multi-Asset Engine:** This is the **best *reference document***. It is not a proposal or a spec. It's the "
   textbook" the engineering team should read before they start work on the multi-asset features in the v1/v2 roadmap.

**Recommendation:** Adopt the **vision and core architectural principles of SOTA**, but use the **PRD structure and
actionability of OPUS/GEM** as the template to execute it. Use the **Multi-Asset document** as the domain-expert guide
for the roadmap.

## 💡 Key Takeaways from All Proposals

Reading all four documents together provides a powerful, unified vision for a modern backtesting engine.

1. **The Modern Stack is Decided:** The consensus is clear. The era of pure Python loops and legacy Pandas/Cython is
   over. The new stack is **Polars + Apache Arrow** for memory-efficient, zero-copy data handling and **Numba (or Rust)
   ** for JIT-compiled, high-performance compute on critical paths.
2. **Reproducibility is a Non-Negotiable Feature:** The single most important feature for a research-grade engine is
   determinism. The "Run Manifest" concept (from SOTA, OPUS, and GEM) is the solution: a file that hashes all inputs (
   data, config, code) and seeds, guaranteeing bit-for-bit reproducibility.
3. **"Config-First" is the Superior UX:** A core idea from OPUS and GEM. The engine's environment (data, dates,
   commissions, slippage) should be defined in a declarative YAML/TOML file, cleanly separating it from the Python
   *strategy logic*. This dramatically improves experimentation and reproducibility.
4. **ML Leakage is the Primary Antagonist:** SOTA names this brilliantly. A modern engine's architecture *must* be built
   to prevent point-in-time (PIT) data leakage by default. This is not a feature; it is the foundation.
5. **Extensible Realism is the Path to Trust:** All proposals agree on a modular, **plugin-based architecture** for
   friction models (slippage, commissions, fill logic). This allows a user (like your pro-am target) to start with
   simple models and increase realism as their strategy matures.
6. **Multi-Asset is a Separate, Hard Problem:** The Multi-Asset doc proves that this is not a "v0" feature. It requires
   dedicated architectural components for margin (Reg-T vs. SPAN), P&L (MTM, FIFO), and asset lifecycles (expirations,
   rolls). This justifies a staggered roadmap.
7. **Testing Must Go Beyond Unit Tests:** The OPUS/GEM PRDs provide a new standard. A robust engine requires *
   *Property-Based Tests** (e.g., "does portfolio value always equal cash + positions?") and **"Golden" Deterministic
   Replay Tests** (e.g., "does this exact strategy file + data file always produce this exact P&L?").
8. **The Goal is a "Live Parity" Architecture:** The event-driven design (present in all) is crucial because it provides
   a seamless path from backtesting to paper/live trading by simply swapping out the *data source* (historical file ->
   live websocket) and the *broker* (simulation -> real API).

---

## 📋 Synthesized PRD Items for a State-of-the-Art Engine

Here is a bullet-point synthesis of the key items a comprehensive PRD should include, combining the best elements from
all proposals and targeting your audience of **ambitious retail traders and professional traders with limited resources
**.

This audience needs professional-grade features (ML, reproducibility, futures, crypto) but with a focus on usability and
a low-friction "config-first" experience.

### **1. Executive Summary & Principles**

* **Core Principles:**
    * **Config-First:** Separate environment (YAML) from logic (Python).
    * **Determinism & Reproducibility:** 100% bit-for-bit reproducible runs via a "Run Manifest."
    * **ML-First:** Architecturally guarantee no point-in-time (PIT) data leakage.
    * **Performance:** Polars-native data layer (Arrow) with Numba/Rust for hot loops.
    * **Extensible Realism:** Pluggable interfaces for all friction and asset models.

### **2. Target Personas**

* **The Pro-Am Retail Coder:** A sophisticated trader (often using Backtrader/vectorbt) who is hitting performance and
  realism walls. They are comfortable with Python and want to test complex strategies on daily and minute data,
  including crypto and futures.
* **The ML Researcher:** A data scientist who lives in Polars/scikit-learn. Their primary concern is a "leakage-safe"
  environment to test predictive models as signals.

### **3. System Architecture**

* **Core:** Event-Driven (priority-queue-based) to ensure total chronological ordering and live-trading parity.
* **Data Layer:** Polars-native (Polars LazyFrames). Use Apache Arrow for all internal data transfer to be zero-copy.
* **Compute:** Python API, with critical loops (event bus, matching engine, accounting) JIT-compiled with Numba.
* **Interfaces (ABCs):** Define clear abstract base classes for:
    * `DataFeed` (e.g., `ParquetFeed`, `LiveWebsocketFeed`)
    * `Broker` (e.g., `SimulationBroker`, `AlpacaBroker`)
    * `SlippageModel` (e.g., `FixedBpsSlippage`, `VolumeShareSlippage`)
    * `CommissionModel` (e.g., `PerShareCommission`, `BinanceTakerFee`)

### **4. Functional Requirements (Key Items)**

* **Data Handling:**
    * Natively read Parquet and CSV files.
    * Support for multiple timeframes (e.g., daily bars for signals, 1-minute bars for execution).
    * Robust exchange calendar and timezone management (normalize all to UTC internally).
* **Strategy & ML Integration:**
    * Python class-based API (`Strategy` base class).
    * Provide a `PITData` object to the strategy for all data access, guaranteeing no look-ahead.
    * First-class support for asynchronous "Signal" events (e.g., from an external ML model).
    * Support for "warm-up" periods to populate indicators.
* **Execution & Fill Logic:**
    * **Order Types:** (MVP) Market, Limit. (v1) Stop, Stop-Limit, TrailingStop, Bracket (OCO).
    * **Time-in-Force:** (MVP) DAY, GTC. (v1) IOC, FOK.
    * **Bar-Fill Logic:** Must be explicit and configurable (e.g., `market_fill_at="next_open"`).
* **Portfolio & Accounting:**
    * (MVP) Single-currency, multi-asset (equities).
    * (v1) Multi-currency (for FX, Crypto) with automated FX conversion handling.
    * (v1) Asset-specific P&L (Mark-to-Market for Futures/Crypto Perps).
* **Asset Class Support (Roadmap):**
    * **(v0 - MVP):** US Equities.
    * **(v1 - Pro-Am):** Add **Cryptocurrency Spot & Perpetuals** (funding rates, 24/7 calendar) and **Futures** (
      continuous contracts, roll logic, MTM P&L).
    * **(v2 - Pro):** Options (basic valuation, exercise/assignment).
* **Friction Models (Pluggable):**
    * **Slippage:** `FixedPercentage`, `VolumeShare`.
    * **Commissions:** `FixedPerShare`, `PercentOfValue`, `Tiered`.
    * **Other:** `BorrowFeeModel` (for shorts), `FundingRateModel` (for crypto perps).

### **5. Reproducibility & Testing**

* **The Run Manifest:** A JSON/YAML file saved with every run, containing:
    * The complete, resolved configuration.
    * SHA-256 hashes of all input data files.
    * Git commit hash of the strategy code.
    * The global PRNG seed.
* **Testing Strategy:**
    * **Unit Tests:** For all individual components.
    * **Property-Based Tests:** (Hypothesis) To validate accounting invariants (e.g.,
      `cash + market_value == total_equity`).
    * **"Golden" Replay Tests:** A suite of tests that run a known strategy on known data and assert that the final P&L
      is bit-for-bit identical to a "golden" (pre-committed) result.

### **6. Output & Analytics**

* **Artifacts:** The engine's *only* output is a directory containing:
    1. `run_manifest.json` (see above).
    2. `trades.parquet` (A log of all fills).
    3. `orders.parquet` (A log of all order status changes).
    4. `portfolio_history.parquet` (A daily/hourly snapshot of equity, cash, positions, etc.).
* **Analysis:** A separate package (`engine-analysis`) will consume this artifact directory to generate reports,
  tearsheets, and plots (replaces pyfolio/quantstats).

### **7. Roadmap (Pro-Am Focus)**

* **v0 (MVP):** "The Reliable Coder" - Focus on Equities, reproducibility, and the Polars/Numba core. Get the
  `run_manifest` and `config-first` UX right.
* **v1 (The Leap):** "The Pro-Am" - Add **Crypto** and **Futures**. This is the key expansion for the target audience.
  Add multi-currency support, MTM P&L, and advanced order types.
* **v2 (The Pro):** "The Quant" - Add **Options**, portfolio-level optimizers, and live-trading adapters (Alpaca,
  Interactive Brokers, Binance).

This combined approach leverages the strategic vision of SOTA, the actionable rigor of OPUS/GEM, and the domain
expertise of the Multi-Asset document to create a powerful, modern, and highly relevant product.
