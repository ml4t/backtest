You are a senior quant developer working on an event-driven Python backtesting engine (NumPy + Polars). Your task is to propose a design to incorporate richer, context-dependent risk management and execution behavior into the existing backtest engine.

Do not just implement something ad hoc. First, explore the design space and then propose a coherent, extensible design that fits the current architecture of the library.

Focus on the following areas:
	1.	Access to strategy and market state
Design an interface so that user-defined risk rules can see, at a minimum:
	•	Strategy state:
	•	Whether there is an open position or not (per instrument).
	•	Position direction, size, and leverage.
	•	Entry price and entry time; number of bars/sessions since entry.
	•	Unrealized PnL; realized PnL; current drawdown; per-trade max favorable excursion (MFE) and max adverse excursion (MAE), if available.
	•	Market state:
	•	Current bar OHLCV and best bid/ask or mid quote (depending on what is available).
	•	Any externally computed features (e.g., volatility metrics, average true range, realized vol, volume indicators, spread, liquidity measures, regime labels).
The risk rule interface should treat these inputs as read-only and return decisions (e.g., updated TP/SL, new exit orders) in a deterministic way.
	2.	Take-profit / stop-loss rules and their adaptation to context
Risk management should allow users to define TP/SL behavior in a context-dependent way rather than as fixed numbers. Explore and propose a design that supports:
	•	Volatility-scaled TP/SL:
	•	Stops and targets as multiples of ATR, realized volatility, or recent range.
	•	Ability to change scaling parameters by regime (e.g., low/high volatility regimes).
	•	Regime-dependent rules:
	•	Different TP/SL parameters conditioned on regime features: volatility regime, trend regime, liquidity regime, time-of-day (cash session vs overnight), etc.
	•	Clear way for users to plug in externally computed regime labels/features.
	•	Dynamic and trailing logic:
	•	Trailing stops based on price extremes since entry.
	•	Break-even rules: after a certain favorable move, move stop to entry or better.
	•	Stepwise or partial profit-taking: scale out at predefined levels; optional re-tightening of stops on the remaining position.
	•	Portfolio-level constraints (at least at the design level):
	•	Per-trade and per-day max loss; per-instrument and portfolio risk budgets.
	•	How these constraints interact with per-trade TP/SL rules.
Your design should make it straightforward for a user to express these rules as functions of “state + features → actions/parameters”.
	3.	Time-based exits (third barrier)
Many strategies use a time-based barrier in addition to TP and SL. Propose how to support:
	•	Max holding period:
	•	Exit after N bars since entry.
	•	Exit after a calendar time interval (e.g., close all intraday positions by session end, or after X minutes/hours).
	•	Session-aware exits:
	•	Rules such as: “Always flatten at end of session” or “Never hold across weekend/holiday gaps if the calendar provides this information”.
	•	Interaction with TP/SL:
	•	Model trades as having three barriers (TP, SL, time) and define how the engine checks and prioritizes them within the event loop.
Make sure the design lets the user configure these as simple, composable rules, again with access to position age, session information, and external features if needed.
	4.	Slippage and transaction-cost models beyond fixed percentages
The engine should support more realistic, pluggable slippage/TC models. Explore and propose an interface and set of baseline models:
	•	Baseline models:
	•	Fixed basis points / fixed fraction of spread (for backward compatibility).
	•	Spread-aware models: execution at mid ± k × spread; modeling crossing the spread for market orders.
	•	Volatility- and volume-aware impact:
	•	Slippage as a function of order size vs typical volume (participation rate).
	•	Simple impact models (e.g., square-root impact or other parametric functions of volume and volatility).
	•	Order-type-dependent behavior:
	•	Market orders vs limit/stop limit orders.
	•	Partial fills and probability of fill for limit orders, at least at a simple modeling level.
	•	Design considerations:
	•	A clear slippage model interface (e.g., object or callable) that takes the intended order (side, size, limit/market, timestamp), the available market data around that time, and returns the effective execution price(s) and filled size.
	•	How to configure different slippage models per instrument, per venue, or globally.
	•	How to calibrate slippage model parameters from historical data (design hooks, even if not fully implemented now).
	5.	Configuration and API design for user-defined rules
Propose how users specify these risk rules and models in a way that is:
	•	Explicitly tied to the strategy’s event loop / lifecycle:
	•	Where in the event loop are risk rules evaluated (before signal generation, after signal but before order creation, before order submission, etc.)?
	•	How the risk manager interacts with the “signal/alpha” component and the order generation component.
	•	Composable and testable:
	•	Ability to stack multiple rules (e.g., volatility-scaled stop + time-based exit + portfolio max-loss rule).
	•	Simple configuration (Python API and potentially a declarative/config-driven layer later).
	•	Easy unit testing of each rule in isolation via small synthetic scenarios.
	•	Backwards compatible:
	•	Keep a simple path for strategies that only want fixed TP/SL and simple slippage, while allowing advanced users to opt into the richer rule set.
	6.	Deliverables
Based on the above, produce:
	•	A design proposal document that:
	•	Describes the risk management / slippage abstraction(s) and their responsibilities.
	•	Shows how these components plug into the current backtest engine architecture (event flow, main objects).
	•	Lists key data structures and interfaces (e.g., Python class/method signatures or type aliases) but avoids committing to low-level implementation details prematurely.
	•	A short set of example scenarios:
	•	For each major feature (vol-scaled stops, time-based exits, spread-aware slippage, etc.), show how a strategy author would configure or implement the behavior using the proposed API.
Emphasize clarity, extensibility, and alignment with the existing event-driven engine. Do not assume a complete rewrite of the system; instead, design components that can be introduced incrementally and that respect the existing data model (NumPy/Polars, event types, order/position objects).
