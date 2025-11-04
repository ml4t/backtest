This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where line numbers have been added.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: README.md, CLAUDE.md, src/qengine/__init__.py, src/qengine/engine.py, src/qengine/core/*.py, src/qengine/data/*.py, src/qengine/execution/*.py, src/qengine/portfolio/*.py, src/qengine/strategy/*.py, src/qengine/reporting/*.py
- Files matching these patterns are excluded: .venv/**, docs/**, examples/**, tests/**, resources/**, htmlcov/**, __pycache__/**, *.pyc, benchmarks/**, config/**
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
src/
  qengine/
    core/
      __init__.py
      assets.py
      clock.py
      event.py
      types.py
    data/
      __init__.py
      feed.py
      schemas.py
    execution/
      __init__.py
      broker.py
      commission.py
      corporate_actions.py
      market_impact.py
      order.py
      slippage.py
    portfolio/
      __init__.py
      accounting.py
      margin.py
      portfolio.py
      simple.py
    reporting/
      __init__.py
      base.py
      html.py
      parquet.py
      reporter.py
    strategy/
      __init__.py
      adapters.py
      base.py
      crypto_basis_adapter.py
      spy_order_flow_adapter.py
    __init__.py
    engine.py
CLAUDE.md
README.md
```

# Files

## File: src/qengine/core/__init__.py
````python
 1: """Core event system and time management for QEngine."""
 2: 
 3: from qengine.core.assets import AssetClass, AssetRegistry, AssetSpec
 4: from qengine.core.clock import Clock
 5: from qengine.core.event import (
 6:     Event,
 7:     EventBus,
 8:     FillEvent,
 9:     MarketEvent,
10:     OrderEvent,
11:     SignalEvent,
12: )
13: from qengine.core.types import AssetId, Price, Quantity, Timestamp
14: 
15: __all__ = [
16:     "AssetClass",
17:     "AssetId",
18:     "AssetRegistry",
19:     "AssetSpec",
20:     "Clock",
21:     "Event",
22:     "EventBus",
23:     "FillEvent",
24:     "MarketEvent",
25:     "OrderEvent",
26:     "Price",
27:     "Quantity",
28:     "SignalEvent",
29:     "Timestamp",
30: ]
````

## File: src/qengine/core/assets.py
````python
  1: """Asset class definitions and specifications for QEngine."""
  2: 
  3: from dataclasses import dataclass
  4: from datetime import datetime
  5: from enum import Enum
  6: 
  7: from qengine.core.types import AssetId, Price
  8: 
  9: 
 10: class AssetClass(Enum):
 11:     """Supported asset classes."""
 12: 
 13:     EQUITY = "equity"
 14:     FUTURE = "future"
 15:     OPTION = "option"
 16:     FX = "fx"
 17:     CRYPTO = "crypto"
 18:     BOND = "bond"
 19:     COMMODITY = "commodity"
 20: 
 21: 
 22: class ContractType(Enum):
 23:     """Contract types for derivatives."""
 24: 
 25:     SPOT = "spot"
 26:     FUTURE = "future"
 27:     PERPETUAL = "perpetual"
 28:     CALL = "call"
 29:     PUT = "put"
 30: 
 31: 
 32: @dataclass
 33: class AssetSpec:
 34:     """
 35:     Complete specification for an asset.
 36: 
 37:     This class handles the different requirements for various asset classes:
 38:     - Equities: Simple spot trading with T+2 settlement
 39:     - Futures: Margin requirements, expiry, rolling
 40:     - Options: Greeks, expiry, exercise
 41:     - FX: Currency pairs, pip values
 42:     - Crypto: 24/7 trading, fractional shares
 43:     """
 44: 
 45:     asset_id: AssetId
 46:     asset_class: AssetClass
 47:     contract_type: ContractType = ContractType.SPOT
 48: 
 49:     # Common fields
 50:     currency: str = "USD"
 51:     tick_size: float = 0.01
 52:     lot_size: float = 1.0
 53:     min_quantity: float = 1.0
 54: 
 55:     # Equity-specific
 56:     exchange: str | None = None
 57: 
 58:     # Futures-specific
 59:     contract_size: float = 1.0  # Multiplier for futures/options
 60:     initial_margin: float = 0.0  # Initial margin requirement
 61:     maintenance_margin: float = 0.0  # Maintenance margin requirement
 62:     expiry: datetime | None = None
 63:     underlying: AssetId | None = None  # For derivatives
 64:     roll_date: datetime | None = None  # When to roll to next contract
 65: 
 66:     # Options-specific
 67:     strike: Price | None = None
 68:     option_type: str | None = None  # "call" or "put"
 69:     exercise_style: str | None = None  # "american", "european"
 70: 
 71:     # FX-specific
 72:     base_currency: str | None = None
 73:     quote_currency: str | None = None
 74:     pip_value: float = 0.0001  # Standard pip value
 75: 
 76:     # Crypto-specific
 77:     is_24_7: bool = False  # Trades 24/7
 78:     network_fees: bool = False  # Has blockchain network fees
 79: 
 80:     # Trading specifications
 81:     maker_fee: float = 0.001  # 0.1% default
 82:     taker_fee: float = 0.001  # 0.1% default
 83:     short_enabled: bool = True
 84:     leverage_available: float = 1.0  # Max leverage
 85: 
 86:     @property
 87:     def is_derivative(self) -> bool:
 88:         """Check if asset is a derivative."""
 89:         return self.asset_class in [AssetClass.FUTURE, AssetClass.OPTION]
 90: 
 91:     @property
 92:     def requires_margin(self) -> bool:
 93:         """Check if asset requires margin."""
 94:         return self.asset_class in [AssetClass.FUTURE, AssetClass.FX] or self.leverage_available > 1
 95: 
 96:     @property
 97:     def has_expiry(self) -> bool:
 98:         """Check if asset has expiry."""
 99:         return self.expiry is not None
100: 
101:     def get_margin_requirement(self, quantity: float, price: Price) -> float:
102:         """
103:         Calculate margin requirement for position.
104: 
105:         Args:
106:             quantity: Position size
107:             price: Current price
108: 
109:         Returns:
110:             Required margin
111:         """
112:         if self.asset_class == AssetClass.FUTURE:
113:             # Futures use fixed margin per contract
114:             return abs(quantity) * self.initial_margin
115:         if self.asset_class == AssetClass.FX:
116:             # FX uses percentage of notional
117:             notional = abs(quantity) * price
118:             return notional / self.leverage_available if self.leverage_available > 0 else notional
119:         if self.asset_class == AssetClass.CRYPTO and self.leverage_available > 1:
120:             # Leveraged crypto trading
121:             notional = abs(quantity) * price
122:             return notional / self.leverage_available
123:         if self.asset_class == AssetClass.OPTION:
124:             # Options: buyers pay premium, sellers need margin
125:             if quantity > 0:  # Buying options
126:                 return abs(quantity) * price * self.contract_size
127:             # Selling options - simplified margin
128:             return abs(quantity) * self.strike * self.contract_size * 0.2  # 20% of notional
129:         # Spot trading - full cash required
130:         return abs(quantity) * price
131: 
132:     def get_notional_value(self, quantity: float, price: Price) -> float:
133:         """
134:         Calculate notional value of position.
135: 
136:         Args:
137:             quantity: Position size
138:             price: Current price
139: 
140:         Returns:
141:             Notional value
142:         """
143:         if self.asset_class in [AssetClass.FUTURE, AssetClass.OPTION]:
144:             return abs(quantity) * price * self.contract_size
145:         if self.asset_class == AssetClass.FX:
146:             # FX notional in base currency
147:             return abs(quantity) * price
148:         return abs(quantity) * price
149: 
150:     def calculate_pnl(
151:         self,
152:         entry_price: Price,
153:         exit_price: Price,
154:         quantity: float,
155:         include_costs: bool = True,
156:     ) -> float:
157:         """
158:         Calculate P&L for a trade.
159: 
160:         Args:
161:             entry_price: Entry price
162:             exit_price: Exit price
163:             quantity: Position size (positive for long, negative for short)
164:             include_costs: Whether to include trading costs
165: 
166:         Returns:
167:             Profit/loss
168:         """
169:         if self.asset_class == AssetClass.FUTURE:
170:             # Futures P&L includes contract multiplier
171:             pnl = quantity * (exit_price - entry_price) * self.contract_size
172:         elif self.asset_class == AssetClass.OPTION:
173:             # Options P&L depends on type
174:             if self.option_type == "call":
175:                 if quantity > 0:  # Long call
176:                     pnl = max(exit_price - self.strike, 0) * quantity * self.contract_size
177:                     pnl -= entry_price * quantity * self.contract_size  # Premium paid
178:                 else:  # Short call
179:                     pnl = entry_price * abs(quantity) * self.contract_size  # Premium received
180:                     pnl -= max(exit_price - self.strike, 0) * abs(quantity) * self.contract_size
181:             else:  # Put
182:                 if quantity > 0:  # Long put
183:                     pnl = max(self.strike - exit_price, 0) * quantity * self.contract_size
184:                     pnl -= entry_price * quantity * self.contract_size  # Premium paid
185:                 else:  # Short put
186:                     pnl = entry_price * abs(quantity) * self.contract_size  # Premium received
187:                     pnl -= max(self.strike - exit_price, 0) * abs(quantity) * self.contract_size
188:         elif self.asset_class == AssetClass.FX:
189:             # FX P&L in quote currency
190:             pnl = quantity * (exit_price - entry_price)
191:             # Convert pips to currency if needed
192:             if self.pip_value > 0:
193:                 pnl = pnl / self.pip_value
194:         else:
195:             # Standard P&L calculation
196:             pnl = quantity * (exit_price - entry_price)
197: 
198:         # Subtract trading costs if requested
199:         if include_costs:
200:             entry_cost = abs(quantity * entry_price) * self.taker_fee
201:             exit_cost = abs(quantity * exit_price) * self.taker_fee
202:             pnl -= entry_cost + exit_cost
203: 
204:         return pnl
205: 
206:     def calculate_pnl_premium_based(
207:         self,
208:         entry_premium: Price,
209:         exit_premium: Price,
210:         quantity: float,
211:         include_costs: bool = True,
212:     ) -> float:
213:         """
214:         Calculate P&L for options using premium change methodology.
215: 
216:         This is the CORRECT way to calculate options P&L for positions
217:         closed before expiry. It uses the change in option premium,
218:         not intrinsic value.
219: 
220:         Args:
221:             entry_premium: Option premium at entry
222:             exit_premium: Option premium at exit
223:             quantity: Position size (positive for long, negative for short)
224:             include_costs: Whether to include trading costs
225: 
226:         Returns:
227:             Profit/loss based on premium change
228: 
229:         Raises:
230:             ValueError: If called on non-option assets
231: 
232:         Example:
233:             # Long 1 call option: bought at $2.00, sold at $1.50
234:             # P&L = (1.50 - 2.00) * 1 * 100 = -$50
235:             pnl = call_spec.calculate_pnl_premium_based(2.00, 1.50, 1.0)
236:             assert pnl == -50.0
237:         """
238:         if self.asset_class != AssetClass.OPTION:
239:             raise ValueError("Premium-based P&L calculation is only for options")
240: 
241:         # CORRECT: P&L = (exit_premium - entry_premium) * quantity * contract_size
242:         pnl = (exit_premium - entry_premium) * quantity * self.contract_size
243: 
244:         # Subtract trading costs if requested
245:         if include_costs:
246:             entry_cost = abs(quantity * entry_premium) * getattr(self, "taker_fee", 0.0)
247:             exit_cost = abs(quantity * exit_premium) * getattr(self, "taker_fee", 0.0)
248:             pnl -= entry_cost + exit_cost
249: 
250:         return pnl
251: 
252:     def calculate_pnl_enhanced(
253:         self,
254:         entry_price: Price,
255:         exit_price: Price,
256:         quantity: float,
257:         entry_premium: Price = None,
258:         exit_premium: Price = None,
259:         include_costs: bool = True,
260:     ) -> float:
261:         """
262:         Enhanced P&L calculation with options premium support.
263: 
264:         For options, this method will use premium-based calculation when
265:         premium data is provided, falling back to intrinsic value calculation
266:         when no premium data is available.
267: 
268:         Args:
269:             entry_price: Entry price (underlying price for options)
270:             exit_price: Exit price (underlying price for options)
271:             quantity: Position size
272:             entry_premium: Option premium at entry (for options only)
273:             exit_premium: Option premium at exit (for options only)
274:             include_costs: Whether to include trading costs
275: 
276:         Returns:
277:             Profit/loss
278:         """
279:         if (
280:             self.asset_class == AssetClass.OPTION
281:             and entry_premium is not None
282:             and exit_premium is not None
283:         ):
284:             # Use premium-based calculation for options when premium data available
285:             return self.calculate_pnl_premium_based(
286:                 entry_premium,
287:                 exit_premium,
288:                 quantity,
289:                 include_costs,
290:             )
291:         # Use original calculation method
292:         return self.calculate_pnl(entry_price, exit_price, quantity, include_costs)
293: 
294: 
295: class AssetRegistry:
296:     """Registry for managing asset specifications."""
297: 
298:     def __init__(self):
299:         """Initialize asset registry."""
300:         self._assets: dict[AssetId, AssetSpec] = {}
301: 
302:     def register(self, asset_spec: AssetSpec) -> None:
303:         """Register an asset specification."""
304:         self._assets[asset_spec.asset_id] = asset_spec
305: 
306:     def get(self, asset_id: AssetId) -> AssetSpec | None:
307:         """Get asset specification by ID."""
308:         return self._assets.get(asset_id)
309: 
310:     def get_or_create_equity(self, asset_id: AssetId) -> AssetSpec:
311:         """Get or create a default equity specification."""
312:         if asset_id not in self._assets:
313:             self._assets[asset_id] = AssetSpec(
314:                 asset_id=asset_id,
315:                 asset_class=AssetClass.EQUITY,
316:                 contract_type=ContractType.SPOT,
317:             )
318:         return self._assets[asset_id]
319: 
320:     def create_future(
321:         self,
322:         asset_id: AssetId,
323:         underlying: AssetId,
324:         expiry: datetime,
325:         contract_size: float = 1.0,
326:         initial_margin: float = 0.0,
327:         maintenance_margin: float = 0.0,
328:     ) -> AssetSpec:
329:         """Create a futures contract specification."""
330:         spec = AssetSpec(
331:             asset_id=asset_id,
332:             asset_class=AssetClass.FUTURE,
333:             contract_type=ContractType.FUTURE,
334:             underlying=underlying,
335:             expiry=expiry,
336:             contract_size=contract_size,
337:             initial_margin=initial_margin,
338:             maintenance_margin=maintenance_margin,
339:         )
340:         self._assets[asset_id] = spec
341:         return spec
342: 
343:     def create_option(
344:         self,
345:         asset_id: AssetId,
346:         underlying: AssetId,
347:         strike: Price,
348:         expiry: datetime,
349:         option_type: str,
350:         contract_size: float = 100.0,
351:         exercise_style: str = "american",
352:     ) -> AssetSpec:
353:         """Create an option contract specification."""
354:         spec = AssetSpec(
355:             asset_id=asset_id,
356:             asset_class=AssetClass.OPTION,
357:             contract_type=ContractType.CALL if option_type == "call" else ContractType.PUT,
358:             underlying=underlying,
359:             strike=strike,
360:             expiry=expiry,
361:             option_type=option_type,
362:             contract_size=contract_size,
363:             exercise_style=exercise_style,
364:         )
365:         self._assets[asset_id] = spec
366:         return spec
367: 
368:     def create_fx_pair(
369:         self,
370:         asset_id: AssetId,
371:         base_currency: str,
372:         quote_currency: str,
373:         pip_value: float = 0.0001,
374:         leverage_available: float = 100.0,
375:     ) -> AssetSpec:
376:         """Create an FX pair specification."""
377:         spec = AssetSpec(
378:             asset_id=asset_id,
379:             asset_class=AssetClass.FX,
380:             contract_type=ContractType.SPOT,
381:             base_currency=base_currency,
382:             quote_currency=quote_currency,
383:             currency=quote_currency,
384:             pip_value=pip_value,
385:             leverage_available=leverage_available,
386:             tick_size=pip_value,
387:             lot_size=1000.0,  # Mini lot
388:         )
389:         self._assets[asset_id] = spec
390:         return spec
391: 
392:     def create_crypto(
393:         self,
394:         asset_id: AssetId,
395:         base_currency: str,
396:         quote_currency: str = "USD",
397:         min_quantity: float = 0.00001,
398:         maker_fee: float = 0.001,
399:         taker_fee: float = 0.001,
400:         leverage_available: float = 1.0,
401:     ) -> AssetSpec:
402:         """Create a cryptocurrency specification."""
403:         spec = AssetSpec(
404:             asset_id=asset_id,
405:             asset_class=AssetClass.CRYPTO,
406:             contract_type=ContractType.SPOT,
407:             base_currency=base_currency,
408:             quote_currency=quote_currency,
409:             currency=quote_currency,
410:             min_quantity=min_quantity,
411:             tick_size=0.01,
412:             lot_size=1.0,
413:             is_24_7=True,
414:             network_fees=True,
415:             maker_fee=maker_fee,
416:             taker_fee=taker_fee,
417:             leverage_available=leverage_available,
418:         )
419:         self._assets[asset_id] = spec
420:         return spec
````

## File: src/qengine/core/clock.py
````python
  1: """Time management and synchronization for QEngine."""
  2: 
  3: import heapq
  4: from datetime import datetime
  5: from enum import Enum
  6: 
  7: import pandas_market_calendars as mcal
  8: 
  9: from qengine.core.event import Event
 10: from qengine.data.feed import DataFeed, SignalSource
 11: 
 12: 
 13: class ClockMode(Enum):
 14:     """Clock operation modes."""
 15: 
 16:     BACKTEST = "backtest"  # Historical simulation
 17:     PAPER = "paper"  # Paper trading with real-time data
 18:     LIVE = "live"  # Live trading
 19: 
 20: 
 21: class Clock:
 22:     """
 23:     Master time keeper for the simulation.
 24: 
 25:     The Clock is responsible for:
 26:     - Advancing simulation time
 27:     - Coordinating multiple data sources
 28:     - Ensuring point-in-time correctness
 29:     - Managing trading calendar
 30:     """
 31: 
 32:     def __init__(
 33:         self,
 34:         mode: ClockMode = ClockMode.BACKTEST,
 35:         calendar: str | None = "NYSE",
 36:         start_time: datetime | None = None,
 37:         end_time: datetime | None = None,
 38:     ):
 39:         """
 40:         Initialize the Clock.
 41: 
 42:         Args:
 43:             mode: Operating mode (backtest, paper, live)
 44:             calendar: Market calendar name (e.g., 'NYSE', 'NASDAQ')
 45:             start_time: Simulation start time
 46:             end_time: Simulation end time
 47:         """
 48:         self.mode = mode
 49:         self.calendar_name = calendar
 50:         self.start_time = start_time
 51:         self.end_time = end_time
 52: 
 53:         # Current simulation time
 54:         self._current_time = start_time
 55: 
 56:         # Data sources
 57:         self._data_feeds: list[DataFeed] = []
 58:         self._signal_sources: list[SignalSource] = []
 59: 
 60:         # Event queue (min heap by timestamp)
 61:         self._event_queue: list[tuple[datetime, Event]] = []
 62: 
 63:         # Market calendar
 64:         if calendar:
 65:             self.calendar = mcal.get_calendar(calendar)
 66:             if start_time and end_time:
 67:                 self.trading_sessions = self.calendar.schedule(
 68:                     start_date=start_time.date(),
 69:                     end_date=end_time.date(),
 70:                 )
 71:         else:
 72:             self.calendar = None
 73:             self.trading_sessions = None
 74: 
 75:         # Statistics
 76:         self._events_processed = 0
 77:         self._ticks_processed = 0
 78: 
 79:     def add_data_feed(self, feed: DataFeed) -> None:
 80:         """
 81:         Add a data feed to the clock.
 82: 
 83:         Args:
 84:             feed: Data feed to add
 85:         """
 86:         self._data_feeds.append(feed)
 87:         self._prime_feed(feed)
 88: 
 89:     def add_signal_source(self, source: SignalSource) -> None:
 90:         """
 91:         Add a signal source to the clock.
 92: 
 93:         Args:
 94:             source: Signal source to add
 95:         """
 96:         self._signal_sources.append(source)
 97:         self._prime_signal_source(source)
 98: 
 99:     def _prime_feed(self, feed: DataFeed) -> None:
100:         """
101:         Prime a data feed by adding its first event to the queue.
102: 
103:         Args:
104:             feed: Data feed to prime
105:         """
106:         next_event = feed.get_next_event()
107:         if next_event:
108:             heapq.heappush(self._event_queue, (next_event.timestamp, next_event))
109: 
110:     def _prime_signal_source(self, source: SignalSource) -> None:
111:         """
112:         Prime a signal source.
113: 
114:         Args:
115:             source: Signal source to prime
116:         """
117:         next_signal = source.get_next_signal()
118:         if next_signal:
119:             heapq.heappush(self._event_queue, (next_signal.timestamp, next_signal))
120: 
121:     def get_next_event(self) -> Event | None:
122:         """
123:         Get the next event across all data sources.
124: 
125:         Returns:
126:             Next event in chronological order or None
127:         """
128:         if not self._event_queue:
129:             return None
130: 
131:         # Get event with earliest timestamp
132:         timestamp, event = heapq.heappop(self._event_queue)
133: 
134:         # Update current time (ensures PIT correctness)
135:         self._current_time = timestamp
136: 
137:         # Check if we're past end time
138:         if self.end_time and timestamp > self.end_time:
139:             return None
140: 
141:         # Replenish queue from the source that provided this event
142:         self._replenish_queue(event)
143: 
144:         self._events_processed += 1
145: 
146:         return event
147: 
148:     def _replenish_queue(self, last_event: Event) -> None:
149:         """
150:         Add the next event from the source that provided the last event.
151: 
152:         Args:
153:             last_event: The event that was just processed
154:         """
155:         # This is simplified - in production we'd track which source
156:         # provided which event and replenish accordingly
157:         for feed in self._data_feeds:
158:             if not feed.is_exhausted:
159:                 next_timestamp = feed.peek_next_timestamp()
160:                 if next_timestamp and (not self.end_time or next_timestamp <= self.end_time):
161:                     next_event = feed.get_next_event()
162:                     if next_event:
163:                         heapq.heappush(self._event_queue, (next_event.timestamp, next_event))
164:                         break
165: 
166:         for source in self._signal_sources:
167:             next_timestamp = source.peek_next_timestamp()
168:             if next_timestamp and (not self.end_time or next_timestamp <= self.end_time):
169:                 next_signal = source.get_next_signal()
170:                 if next_signal:
171:                     heapq.heappush(self._event_queue, (next_signal.timestamp, next_signal))
172:                     break
173: 
174:     @property
175:     def current_time(self) -> datetime | None:
176:         """Get the current simulation time."""
177:         return self._current_time
178: 
179:     @property
180:     def is_market_open(self) -> bool:
181:         """
182:         Check if the market is currently open.
183: 
184:         Returns:
185:             True if market is open at current time
186:         """
187:         if not self.calendar or not self._current_time:
188:             return True  # Assume always open if no calendar
189: 
190:         # Check if current time is within a trading session
191:         current_date = self._current_time.date()
192:         if current_date in self.trading_sessions.index:
193:             session = self.trading_sessions.loc[current_date]
194:             market_open = session["market_open"]
195:             market_close = session["market_close"]
196: 
197:             # Convert to timezone-aware if needed
198:             if self._current_time.tzinfo:
199:                 return market_open <= self._current_time <= market_close
200:             return (
201:                 market_open.replace(tzinfo=None)
202:                 <= self._current_time
203:                 <= market_close.replace(tzinfo=None)
204:             )
205: 
206:         return False
207: 
208:     @property
209:     def next_market_open(self) -> datetime | None:
210:         """
211:         Get the next market open time.
212: 
213:         Returns:
214:             Next market open datetime or None
215:         """
216:         if not self.calendar or not self._current_time:
217:             return None
218: 
219:         current_date = self._current_time.date()
220:         future_sessions = self.trading_sessions[self.trading_sessions.index >= current_date]
221: 
222:         for _date, session in future_sessions.iterrows():
223:             market_open = session["market_open"]
224:             if market_open > self._current_time:
225:                 return market_open
226: 
227:         return None
228: 
229:     @property
230:     def next_market_close(self) -> datetime | None:
231:         """
232:         Get the next market close time.
233: 
234:         Returns:
235:             Next market close datetime or None
236:         """
237:         if not self.calendar or not self._current_time:
238:             return None
239: 
240:         current_date = self._current_time.date()
241:         if current_date in self.trading_sessions.index:
242:             market_close = self.trading_sessions.loc[current_date]["market_close"]
243:             if market_close > self._current_time:
244:                 return market_close
245: 
246:         # Look for next session
247:         future_sessions = self.trading_sessions[self.trading_sessions.index > current_date]
248:         if not future_sessions.empty:
249:             return future_sessions.iloc[0]["market_close"]
250: 
251:         return None
252: 
253:     def is_trading_day(self, date: datetime) -> bool:
254:         """
255:         Check if a given date is a trading day.
256: 
257:         Args:
258:             date: Date to check
259: 
260:         Returns:
261:             True if date is a trading day
262:         """
263:         if not self.calendar:
264:             return True
265: 
266:         return date.date() in self.trading_sessions.index
267: 
268:     def advance_to(self, timestamp: datetime) -> None:
269:         """
270:         Advance the clock to a specific time.
271: 
272:         Used for jumping forward in time during backtesting.
273: 
274:         Args:
275:             timestamp: Target timestamp
276:         """
277:         if self.mode != ClockMode.BACKTEST:
278:             raise RuntimeError("Can only advance time in backtest mode")
279: 
280:         if self._current_time is not None and timestamp < self._current_time:
281:             raise ValueError("Cannot go back in time")
282: 
283:         self._current_time = timestamp
284: 
285:         # Drop events before the new time
286:         while self._event_queue and self._event_queue[0][0] < timestamp:
287:             heapq.heappop(self._event_queue)
288: 
289:     def reset(self) -> None:
290:         """Reset the clock to initial state."""
291:         self._current_time = self.start_time
292:         self._event_queue.clear()
293:         self._events_processed = 0
294:         self._ticks_processed = 0
295: 
296:         # Reset all data feeds
297:         for feed in self._data_feeds:
298:             feed.reset()
299:             self._prime_feed(feed)
300: 
301:         # Reset all signal sources
302:         for source in self._signal_sources:
303:             source.reset()
304:             self._prime_signal_source(source)
305: 
306:     @property
307:     def stats(self) -> dict:
308:         """Get clock statistics."""
309:         return {
310:             "current_time": self._current_time,
311:             "events_processed": self._events_processed,
312:             "queue_size": len(self._event_queue),
313:             "data_feeds": len(self._data_feeds),
314:             "signal_sources": len(self._signal_sources),
315:             "mode": self.mode.value,
316:         }
317: 
318:     def __repr__(self) -> str:
319:         return (
320:             f"Clock(mode={self.mode.value}, "
321:             f"time={self._current_time}, "
322:             f"events={self._events_processed})"
323:         )
````

## File: src/qengine/core/event.py
````python
  1: """Event system for QEngine."""
  2: 
  3: import heapq
  4: import logging
  5: import threading
  6: from abc import ABC
  7: from collections import deque
  8: from collections.abc import Callable
  9: from datetime import datetime
 10: from typing import Any
 11: 
 12: from qengine.core.types import (
 13:     AssetId,
 14:     EventType,
 15:     MarketDataType,
 16:     OrderId,
 17:     OrderSide,
 18:     OrderType,
 19:     Price,
 20:     Quantity,
 21:     TimeInForce,
 22:     Volume,
 23: )
 24: 
 25: logger = logging.getLogger(__name__)
 26: 
 27: 
 28: class Event(ABC):
 29:     """Base class for all events in the system."""
 30: 
 31:     def __init__(
 32:         self,
 33:         timestamp: datetime,
 34:         event_type: EventType,
 35:         metadata: dict[str, Any] | None = None,
 36:     ):
 37:         self.timestamp = timestamp
 38:         self.event_type = event_type
 39:         self.metadata = metadata or {}
 40: 
 41:     def __lt__(self, other: "Event") -> bool:
 42:         """Compare events by timestamp for priority queue."""
 43:         return self.timestamp < other.timestamp
 44: 
 45:     def __repr__(self) -> str:
 46:         return f"{self.__class__.__name__}(timestamp={self.timestamp})"
 47: 
 48: 
 49: class MarketEvent(Event):
 50:     """Market data event (trade, quote, or bar)."""
 51: 
 52:     def __init__(
 53:         self,
 54:         timestamp: datetime,
 55:         asset_id: AssetId,
 56:         data_type: MarketDataType,
 57:         price: Price | None = None,
 58:         size: Quantity | None = None,
 59:         bid_price: Price | None = None,
 60:         ask_price: Price | None = None,
 61:         bid_size: Quantity | None = None,
 62:         ask_size: Quantity | None = None,
 63:         open: Price | None = None,
 64:         high: Price | None = None,
 65:         low: Price | None = None,
 66:         close: Price | None = None,
 67:         volume: Volume | None = None,
 68:         metadata: dict[str, Any] | None = None,
 69:     ):
 70:         super().__init__(timestamp, EventType.MARKET, metadata)
 71:         self.asset_id = asset_id
 72:         self.data_type = data_type
 73:         self.price = price
 74:         self.size = size
 75:         self.bid_price = bid_price
 76:         self.ask_price = ask_price
 77:         self.bid_size = bid_size
 78:         self.ask_size = ask_size
 79:         self.open = open
 80:         self.high = high
 81:         self.low = low
 82:         self.close = close
 83:         self.volume = volume
 84: 
 85: 
 86: class SignalEvent(Event):
 87:     """ML model signal event."""
 88: 
 89:     def __init__(
 90:         self,
 91:         timestamp: datetime,
 92:         asset_id: AssetId,
 93:         signal_value: float,
 94:         model_id: str,
 95:         confidence: float | None = None,
 96:         features: dict[str, Any] | None = None,
 97:         ts_event: datetime | None = None,
 98:         ts_arrival: datetime | None = None,
 99:         metadata: dict[str, Any] | None = None,
100:     ):
101:         super().__init__(timestamp, EventType.SIGNAL, metadata)
102:         self.asset_id = asset_id
103:         self.signal_value = signal_value
104:         self.model_id = model_id
105:         self.confidence = confidence
106:         self.features = features or {}
107:         self.ts_event = ts_event
108:         self.ts_arrival = ts_arrival or timestamp
109: 
110: 
111: class OrderEvent(Event):
112:     """Order submission event."""
113: 
114:     def __init__(
115:         self,
116:         timestamp: datetime,
117:         order_id: OrderId,
118:         asset_id: AssetId,
119:         order_type: OrderType,
120:         side: OrderSide,
121:         quantity: Quantity,
122:         limit_price: Price | None = None,
123:         stop_price: Price | None = None,
124:         time_in_force: TimeInForce = TimeInForce.DAY,
125:         parent_order_id: OrderId | None = None,
126:         metadata: dict[str, Any] | None = None,
127:     ):
128:         super().__init__(timestamp, EventType.ORDER, metadata)
129:         self.order_id = order_id
130:         self.asset_id = asset_id
131:         self.order_type = order_type
132:         self.side = side
133:         self.quantity = quantity
134:         self.limit_price = limit_price
135:         self.stop_price = stop_price
136:         self.time_in_force = time_in_force
137:         self.parent_order_id = parent_order_id
138: 
139: 
140: class FillEvent(Event):
141:     """Order fill/execution event."""
142: 
143:     def __init__(
144:         self,
145:         timestamp: datetime,
146:         order_id: OrderId,
147:         trade_id: str,
148:         asset_id: AssetId,
149:         side: OrderSide,
150:         fill_quantity: Quantity,
151:         fill_price: Price,
152:         commission: float = 0.0,
153:         slippage: float = 0.0,
154:         market_impact: float = 0.0,
155:         metadata: dict[str, Any] | None = None,
156:     ):
157:         super().__init__(timestamp, EventType.FILL, metadata)
158:         self.order_id = order_id
159:         self.trade_id = trade_id
160:         self.asset_id = asset_id
161:         self.side = side
162:         self.fill_quantity = fill_quantity
163:         self.fill_price = fill_price
164:         self.commission = commission
165:         self.slippage = slippage
166:         self.market_impact = market_impact
167: 
168:     @property
169:     def total_cost(self) -> float:
170:         """Total transaction cost including all fees."""
171:         return self.commission + self.slippage + self.market_impact
172: 
173: 
174: class CorporateActionEvent(Event):
175:     """Corporate action event (split, dividend, etc)."""
176: 
177:     def __init__(
178:         self,
179:         timestamp: datetime,
180:         asset_id: AssetId,
181:         action_type: str,
182:         ex_date: datetime,
183:         record_date: datetime | None = None,
184:         payment_date: datetime | None = None,
185:         adjustment_factor: float | None = None,
186:         dividend_amount: float | None = None,
187:         metadata: dict[str, Any] | None = None,
188:     ):
189:         super().__init__(timestamp, EventType.CORPORATE_ACTION, metadata)
190:         self.asset_id = asset_id
191:         self.action_type = action_type
192:         self.ex_date = ex_date
193:         self.record_date = record_date
194:         self.payment_date = payment_date
195:         self.adjustment_factor = adjustment_factor
196:         self.dividend_amount = dividend_amount
197: 
198: 
199: class EventBus:
200:     """Central event distribution system."""
201: 
202:     def __init__(self, use_priority_queue: bool = True):
203:         """
204:         Initialize the event bus.
205: 
206:         Args:
207:             use_priority_queue: If True, events are processed by timestamp priority
208:         """
209:         self.use_priority_queue = use_priority_queue
210: 
211:         if use_priority_queue:
212:             self._queue: list[Event] = []  # Will use heapq
213:         else:
214:             self._queue: deque = deque()
215: 
216:         self._subscribers: dict[EventType, list[Callable]] = {}
217:         self._running = False
218:         self._lock = threading.Lock()
219: 
220:     def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
221:         """
222:         Subscribe to events of a specific type.
223: 
224:         Args:
225:             event_type: Type of events to subscribe to
226:             handler: Callback function to handle events
227:         """
228:         with self._lock:
229:             if event_type not in self._subscribers:
230:                 self._subscribers[event_type] = []
231:             if handler not in self._subscribers[event_type]:
232:                 self._subscribers[event_type].append(handler)
233: 
234:     def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
235:         """
236:         Unsubscribe from events.
237: 
238:         Args:
239:             event_type: Type of events to unsubscribe from
240:             handler: Handler to remove
241:         """
242:         with self._lock:
243:             if event_type in self._subscribers:
244:                 if handler in self._subscribers[event_type]:
245:                     self._subscribers[event_type].remove(handler)
246: 
247:     def publish(self, event: Event) -> None:
248:         """
249:         Publish an event to the queue.
250: 
251:         Args:
252:             event: Event to publish
253:         """
254:         with self._lock:
255:             if self.use_priority_queue:
256:                 heapq.heappush(self._queue, event)
257:             else:
258:                 self._queue.append(event)
259: 
260:     def process_next(self) -> Event | None:
261:         """
262:         Process the next event in the queue.
263: 
264:         Returns:
265:             The processed event or None if queue is empty
266:         """
267:         with self._lock:
268:             if not self._queue:
269:                 return None
270: 
271:             event = heapq.heappop(self._queue) if self.use_priority_queue else self._queue.popleft()
272: 
273:         # Notify subscribers (outside of lock to prevent deadlock)
274:         self._notify_subscribers(event)
275:         return event
276: 
277:     def process_all(self, max_events: int | None = None) -> int:
278:         """
279:         Process all pending events.
280: 
281:         Args:
282:             max_events: Maximum number of events to process
283: 
284:         Returns:
285:             Number of events processed
286:         """
287:         count = 0
288:         while self._queue and (max_events is None or count < max_events):
289:             if self.process_next() is not None:
290:                 count += 1
291:             else:
292:                 break
293:         return count
294: 
295:     def peek(self) -> Event | None:
296:         """
297:         Peek at the next event without removing it.
298: 
299:         Returns:
300:             Next event or None if queue is empty
301:         """
302:         with self._lock:
303:             if not self._queue:
304:                 return None
305: 
306:             if self.use_priority_queue:
307:                 return self._queue[0] if self._queue else None
308:             return self._queue[0] if self._queue else None
309: 
310:     def clear(self) -> None:
311:         """Clear all pending events."""
312:         with self._lock:
313:             if self.use_priority_queue:
314:                 self._queue.clear()
315:             else:
316:                 self._queue.clear()
317: 
318:     @property
319:     def pending_count(self) -> int:
320:         """Number of pending events in the queue."""
321:         with self._lock:
322:             return len(self._queue)
323: 
324:     def _notify_subscribers(self, event: Event) -> None:
325:         """
326:         Notify all subscribers of an event.
327: 
328:         Args:
329:             event: Event to send to subscribers
330:         """
331:         handlers = []
332:         with self._lock:
333:             if event.event_type in self._subscribers:
334:                 handlers = self._subscribers[event.event_type].copy()
335: 
336:         for handler in handlers:
337:             try:
338:                 handler(event)
339:             except Exception as e:
340:                 logger.error(f"Error in event handler: {e}", exc_info=True)
````

## File: src/qengine/core/types.py
````python
  1: """Core type definitions for QEngine."""
  2: 
  3: from datetime import datetime
  4: from decimal import Decimal
  5: from enum import Enum
  6: from typing import NewType, Union
  7: 
  8: # Time types
  9: Timestamp = NewType("Timestamp", datetime)
 10: Nanoseconds = NewType("Nanoseconds", int)
 11: 
 12: # Market data types
 13: AssetId = NewType("AssetId", str)
 14: Price = Union[float, Decimal]
 15: Quantity = Union[float, int]
 16: Volume = Union[float, int]
 17: 
 18: # Order types
 19: OrderId = NewType("OrderId", str)
 20: TradeId = NewType("TradeId", str)
 21: PositionId = NewType("PositionId", str)
 22: 
 23: # Portfolio types
 24: Cash = Union[float, Decimal]
 25: Currency = NewType("Currency", str)
 26: 
 27: 
 28: class EventType(Enum):
 29:     """Types of events in the system."""
 30: 
 31:     MARKET = "market"
 32:     SIGNAL = "signal"
 33:     ORDER = "order"
 34:     FILL = "fill"
 35:     CORPORATE_ACTION = "corporate_action"
 36:     TIMER = "timer"
 37:     CUSTOM = "custom"
 38: 
 39: 
 40: class OrderType(Enum):
 41:     """Types of orders."""
 42: 
 43:     MARKET = "market"
 44:     LIMIT = "limit"
 45:     STOP = "stop"
 46:     STOP_LIMIT = "stop_limit"
 47:     TRAILING_STOP = "trailing_stop"
 48:     BRACKET = "bracket"
 49:     OCO = "oco"
 50: 
 51: 
 52: class OrderStatus(Enum):
 53:     """Order status lifecycle."""
 54: 
 55:     CREATED = "created"
 56:     SUBMITTED = "submitted"
 57:     ACCEPTED = "accepted"
 58:     PARTIALLY_FILLED = "partially_filled"
 59:     FILLED = "filled"
 60:     CANCELED = "canceled"
 61:     REJECTED = "rejected"
 62:     EXPIRED = "expired"
 63: 
 64: 
 65: class OrderSide(Enum):
 66:     """Order side (buy/sell)."""
 67: 
 68:     BUY = "buy"
 69:     SELL = "sell"
 70:     SHORT = "short"
 71:     COVER = "cover"
 72: 
 73: 
 74: class TimeInForce(Enum):
 75:     """Time-in-force constraints for orders."""
 76: 
 77:     DAY = "day"  # Valid for the day
 78:     GTC = "gtc"  # Good till canceled
 79:     IOC = "ioc"  # Immediate or cancel
 80:     FOK = "fok"  # Fill or kill
 81:     GTD = "gtd"  # Good till date
 82:     MOC = "moc"  # Market on close
 83:     MOO = "moo"  # Market on open
 84: 
 85: 
 86: class AssetType(Enum):
 87:     """Types of tradeable assets."""
 88: 
 89:     EQUITY = "equity"
 90:     FUTURE = "future"
 91:     OPTION = "option"
 92:     FOREX = "forex"
 93:     CRYPTO = "crypto"
 94:     BOND = "bond"
 95:     COMMODITY = "commodity"
 96:     INDEX = "index"
 97: 
 98: 
 99: class BarType(Enum):
100:     """Types of price bars."""
101: 
102:     TICK = "tick"
103:     TIME = "time"
104:     VOLUME = "volume"
105:     DOLLAR = "dollar"
106:     TICK_IMBALANCE = "tick_imbalance"
107:     VOLUME_IMBALANCE = "volume_imbalance"
108: 
109: 
110: class MarketDataType(Enum):
111:     """Types of market data."""
112: 
113:     TRADE = "trade"
114:     QUOTE = "quote"
115:     BAR = "bar"
116:     ORDERBOOK = "orderbook"
````

## File: src/qengine/data/__init__.py
````python
 1: """Data management for QEngine."""
 2: 
 3: from qengine.data.feed import DataFeed, SignalSource
 4: from qengine.data.schemas import MarketDataSchema, SignalSchema
 5: 
 6: __all__ = [
 7:     "DataFeed",
 8:     "MarketDataSchema",
 9:     "SignalSchema",
10:     "SignalSource",
11: ]
````

## File: src/qengine/data/feed.py
````python
  1: """Data feed interfaces and implementations for QEngine."""
  2: 
  3: from abc import ABC, abstractmethod
  4: from datetime import datetime
  5: from pathlib import Path
  6: from typing import Any
  7: 
  8: import polars as pl
  9: 
 10: from qengine.core.event import Event, MarketEvent, SignalEvent
 11: from qengine.core.types import AssetId, MarketDataType
 12: 
 13: 
 14: class DataFeed(ABC):
 15:     """Abstract base class for all data feeds."""
 16: 
 17:     @abstractmethod
 18:     def get_next_event(self) -> Event | None:
 19:         """
 20:         Get the next event from this data feed.
 21: 
 22:         Returns:
 23:             Next event or None if no more data
 24:         """
 25: 
 26:     @abstractmethod
 27:     def peek_next_timestamp(self) -> datetime | None:
 28:         """
 29:         Peek at the timestamp of the next event without consuming it.
 30: 
 31:         Returns:
 32:             Timestamp of next event or None if no more data
 33:         """
 34: 
 35:     @abstractmethod
 36:     def reset(self) -> None:
 37:         """Reset the data feed to the beginning."""
 38: 
 39:     @abstractmethod
 40:     def seek(self, timestamp: datetime) -> None:
 41:         """
 42:         Seek to a specific timestamp.
 43: 
 44:         Args:
 45:             timestamp: Target timestamp to seek to
 46:         """
 47: 
 48:     @property
 49:     @abstractmethod
 50:     def is_exhausted(self) -> bool:
 51:         """Check if the data feed has no more events."""
 52: 
 53: 
 54: class SignalSource(ABC):
 55:     """Abstract base class for ML signal sources."""
 56: 
 57:     @abstractmethod
 58:     def get_next_signal(self) -> SignalEvent | None:
 59:         """
 60:         Get the next signal from this source.
 61: 
 62:         Returns:
 63:             Next signal event or None if no more signals
 64:         """
 65: 
 66:     @abstractmethod
 67:     def peek_next_timestamp(self) -> datetime | None:
 68:         """
 69:         Peek at the timestamp of the next signal.
 70: 
 71:         Returns:
 72:             Timestamp of next signal or None
 73:         """
 74: 
 75:     @abstractmethod
 76:     def reset(self) -> None:
 77:         """Reset the signal source."""
 78: 
 79: 
 80: class ParquetDataFeed(DataFeed):
 81:     """Data feed that reads from Parquet files using Polars."""
 82: 
 83:     def __init__(
 84:         self,
 85:         path: Path,
 86:         asset_id: AssetId,
 87:         data_type: MarketDataType = MarketDataType.BAR,
 88:         timestamp_column: str = "timestamp",
 89:         filters: list[tuple] | None = None,
 90:     ):
 91:         """
 92:         Initialize Parquet data feed.
 93: 
 94:         Args:
 95:             path: Path to Parquet file
 96:             asset_id: Asset identifier
 97:             data_type: Type of market data
 98:             timestamp_column: Name of timestamp column
 99:             filters: Optional Polars filters to apply
100:         """
101:         self.path = Path(path)
102:         self.asset_id = asset_id
103:         self.data_type = data_type
104:         self.timestamp_column = timestamp_column
105: 
106:         # Load data lazily with Polars
107:         self.lazy_df = pl.scan_parquet(str(self.path))
108: 
109:         # Apply filters if provided
110:         if filters:
111:             for filter_expr in filters:
112:                 self.lazy_df = self.lazy_df.filter(filter_expr)
113: 
114:         # Sort by timestamp and collect
115:         self.df = self.lazy_df.sort(timestamp_column).collect()
116: 
117:         self.current_index = 0
118:         self.max_index = len(self.df) - 1
119: 
120:     def get_next_event(self) -> MarketEvent | None:
121:         """Get the next market event."""
122:         if self.is_exhausted:
123:             return None
124: 
125:         row = self.df.row(self.current_index, named=True)
126:         self.current_index += 1
127: 
128:         # Create MarketEvent based on data type
129:         event = self._create_market_event(row)
130:         return event
131: 
132:     def _create_market_event(self, row: dict[str, Any]) -> MarketEvent:
133:         """Create a MarketEvent from a data row."""
134:         timestamp = row[self.timestamp_column]
135: 
136:         # Convert timestamp if needed
137:         if not isinstance(timestamp, datetime):
138:             timestamp = datetime.fromisoformat(str(timestamp))
139: 
140:         # Map column names to MarketEvent fields
141:         return MarketEvent(
142:             timestamp=timestamp,
143:             asset_id=self.asset_id,
144:             data_type=self.data_type,
145:             open=row.get("open"),
146:             high=row.get("high"),
147:             low=row.get("low"),
148:             close=row.get("close"),
149:             volume=row.get("volume"),
150:             price=row.get("price", row.get("close")),
151:             size=row.get("size"),
152:             bid_price=row.get("bid"),
153:             ask_price=row.get("ask"),
154:             bid_size=row.get("bid_size"),
155:             ask_size=row.get("ask_size"),
156:         )
157: 
158:     def peek_next_timestamp(self) -> datetime | None:
159:         """Peek at the next timestamp."""
160:         if self.is_exhausted:
161:             return None
162: 
163:         timestamp = self.df[self.timestamp_column][self.current_index]
164:         if not isinstance(timestamp, datetime):
165:             timestamp = datetime.fromisoformat(str(timestamp))
166: 
167:         return timestamp
168: 
169:     def reset(self) -> None:
170:         """Reset to the beginning."""
171:         self.current_index = 0
172: 
173:     def seek(self, timestamp: datetime) -> None:
174:         """Seek to a specific timestamp."""
175:         # Find the index of the first row >= timestamp
176:         mask = self.df[self.timestamp_column] >= timestamp
177:         indices = mask.arg_true()
178: 
179:         if len(indices) > 0:
180:             self.current_index = indices[0]
181:         else:
182:             self.current_index = self.max_index + 1
183: 
184:     @property
185:     def is_exhausted(self) -> bool:
186:         """Check if all data has been consumed."""
187:         return self.current_index > self.max_index
188: 
189: 
190: class CSVDataFeed(DataFeed):
191:     """Data feed that reads from CSV files."""
192: 
193:     def __init__(
194:         self,
195:         path: Path,
196:         asset_id: AssetId,
197:         data_type: MarketDataType = MarketDataType.BAR,
198:         timestamp_column: str = "timestamp",
199:         parse_dates: bool = True,
200:         **csv_kwargs,
201:     ):
202:         """
203:         Initialize CSV data feed.
204: 
205:         Args:
206:             path: Path to CSV file
207:             asset_id: Asset identifier
208:             data_type: Type of market data
209:             timestamp_column: Name of timestamp column
210:             parse_dates: Whether to parse dates automatically
211:             **csv_kwargs: Additional arguments for Polars read_csv
212:         """
213:         self.path = Path(path)
214:         self.asset_id = asset_id
215:         self.data_type = data_type
216:         self.timestamp_column = timestamp_column
217: 
218:         # Read CSV with Polars
219:         if parse_dates:
220:             csv_kwargs["try_parse_dates"] = True
221: 
222:         self.df = pl.read_csv(str(self.path), **csv_kwargs).sort(timestamp_column)
223: 
224:         self.current_index = 0
225:         self.max_index = len(self.df) - 1
226: 
227:     def get_next_event(self) -> MarketEvent | None:
228:         """Get the next market event."""
229:         if self.is_exhausted:
230:             return None
231: 
232:         row = self.df.row(self.current_index, named=True)
233:         self.current_index += 1
234: 
235:         return self._create_market_event(row)
236: 
237:     def _create_market_event(self, row: dict[str, Any]) -> MarketEvent:
238:         """Create a MarketEvent from a data row."""
239:         timestamp = row[self.timestamp_column]
240: 
241:         if not isinstance(timestamp, datetime):
242:             timestamp = datetime.fromisoformat(str(timestamp))
243: 
244:         return MarketEvent(
245:             timestamp=timestamp,
246:             asset_id=self.asset_id,
247:             data_type=self.data_type,
248:             open=row.get("open"),
249:             high=row.get("high"),
250:             low=row.get("low"),
251:             close=row.get("close"),
252:             volume=row.get("volume"),
253:             price=row.get("price", row.get("close")),
254:         )
255: 
256:     def peek_next_timestamp(self) -> datetime | None:
257:         """Peek at the next timestamp."""
258:         if self.is_exhausted:
259:             return None
260: 
261:         timestamp = self.df[self.timestamp_column][self.current_index]
262:         if not isinstance(timestamp, datetime):
263:             timestamp = datetime.fromisoformat(str(timestamp))
264: 
265:         return timestamp
266: 
267:     def reset(self) -> None:
268:         """Reset to the beginning."""
269:         self.current_index = 0
270: 
271:     def seek(self, timestamp: datetime) -> None:
272:         """Seek to a specific timestamp."""
273:         mask = self.df[self.timestamp_column] >= timestamp
274:         indices = mask.arg_true()
275: 
276:         if len(indices) > 0:
277:             self.current_index = indices[0]
278:         else:
279:             self.current_index = self.max_index + 1
280: 
281:     @property
282:     def is_exhausted(self) -> bool:
283:         """Check if all data has been consumed."""
284:         return self.current_index > self.max_index
285: 
286: 
287: class ParquetSignalSource(SignalSource):
288:     """Signal source that reads ML signals from Parquet files."""
289: 
290:     def __init__(
291:         self,
292:         path: Path,
293:         model_id: str,
294:         timestamp_column: str = "timestamp",
295:         asset_column: str = "asset_id",
296:         signal_column: str = "signal",
297:         confidence_column: str | None = "confidence",
298:         ts_event_column: str | None = "ts_event",
299:         ts_arrival_column: str | None = "ts_arrival",
300:     ):
301:         """
302:         Initialize Parquet signal source.
303: 
304:         Args:
305:             path: Path to Parquet file with signals
306:             model_id: Identifier for the ML model
307:             timestamp_column: Column with signal timestamp
308:             asset_column: Column with asset identifiers
309:             signal_column: Column with signal values
310:             confidence_column: Optional column with confidence scores
311:             ts_event_column: Optional column with event generation time
312:             ts_arrival_column: Optional column with signal arrival time
313:         """
314:         self.path = Path(path)
315:         self.model_id = model_id
316:         self.timestamp_column = timestamp_column
317:         self.asset_column = asset_column
318:         self.signal_column = signal_column
319:         self.confidence_column = confidence_column
320:         self.ts_event_column = ts_event_column
321:         self.ts_arrival_column = ts_arrival_column
322: 
323:         # Load signals with Polars
324:         self.df = pl.scan_parquet(str(self.path)).sort(timestamp_column).collect()
325: 
326:         self.current_index = 0
327:         self.max_index = len(self.df) - 1
328: 
329:     def get_next_signal(self) -> SignalEvent | None:
330:         """Get the next signal event."""
331:         if self.current_index > self.max_index:
332:             return None
333: 
334:         row = self.df.row(self.current_index, named=True)
335:         self.current_index += 1
336: 
337:         timestamp = row[self.timestamp_column]
338:         if not isinstance(timestamp, datetime):
339:             timestamp = datetime.fromisoformat(str(timestamp))
340: 
341:         return SignalEvent(
342:             timestamp=timestamp,
343:             asset_id=AssetId(row[self.asset_column]),
344:             signal_value=float(row[self.signal_column]),
345:             model_id=self.model_id,
346:             confidence=float(row[self.confidence_column])
347:             if self.confidence_column and self.confidence_column in row
348:             else None,
349:             ts_event=row.get(self.ts_event_column) if self.ts_event_column else None,
350:             ts_arrival=row.get(self.ts_arrival_column) if self.ts_arrival_column else timestamp,
351:         )
352: 
353:     def peek_next_timestamp(self) -> datetime | None:
354:         """Peek at the next signal timestamp."""
355:         if self.current_index > self.max_index:
356:             return None
357: 
358:         timestamp = self.df[self.timestamp_column][self.current_index]
359:         if not isinstance(timestamp, datetime):
360:             timestamp = datetime.fromisoformat(str(timestamp))
361: 
362:         return timestamp
363: 
364:     def reset(self) -> None:
365:         """Reset to the beginning."""
366:         self.current_index = 0
````

## File: src/qengine/data/schemas.py
````python
  1: """Data schemas for QEngine."""
  2: 
  3: from dataclasses import dataclass
  4: 
  5: import polars as pl
  6: 
  7: 
  8: @dataclass
  9: class MarketDataSchema:
 10:     """Schema definition for market data."""
 11: 
 12:     timestamp_col: str = "timestamp"
 13:     open_col: str = "open"
 14:     high_col: str = "high"
 15:     low_col: str = "low"
 16:     close_col: str = "close"
 17:     volume_col: str = "volume"
 18: 
 19:     def get_dtypes(self) -> dict[str, pl.DataType]:
 20:         """Get Polars data types for the schema."""
 21:         return {
 22:             self.timestamp_col: pl.Datetime("ns"),
 23:             self.open_col: pl.Float64,
 24:             self.high_col: pl.Float64,
 25:             self.low_col: pl.Float64,
 26:             self.close_col: pl.Float64,
 27:             self.volume_col: pl.Int64,
 28:         }
 29: 
 30:     def validate(self, df: pl.DataFrame) -> None:
 31:         """Validate a DataFrame against this schema."""
 32:         required_cols = [
 33:             self.timestamp_col,
 34:             self.open_col,
 35:             self.high_col,
 36:             self.low_col,
 37:             self.close_col,
 38:             self.volume_col,
 39:         ]
 40: 
 41:         missing_cols = set(required_cols) - set(df.columns)
 42:         if missing_cols:
 43:             raise ValueError(f"Missing required columns: {missing_cols}")
 44: 
 45:         # Validate data types
 46:         for col, expected_dtype in self.get_dtypes().items():
 47:             if col in df.columns:
 48:                 actual_dtype = df[col].dtype
 49:                 if not self._compatible_dtypes(actual_dtype, expected_dtype):
 50:                     raise TypeError(
 51:                         f"Column {col} has type {actual_dtype}, expected {expected_dtype}",
 52:                     )
 53: 
 54:     def _compatible_dtypes(self, actual: pl.DataType, expected: pl.DataType) -> bool:
 55:         """Check if data types are compatible."""
 56:         # Allow int to float conversion
 57:         if expected == pl.Float64 and actual in [pl.Int32, pl.Int64]:
 58:             return True
 59:         # Allow different datetime precisions
 60:         if isinstance(expected, pl.Datetime) and isinstance(actual, pl.Datetime):
 61:             return True
 62:         return actual == expected
 63: 
 64: 
 65: @dataclass
 66: class SignalSchema:
 67:     """Schema definition for ML signals."""
 68: 
 69:     timestamp_col: str = "timestamp"
 70:     asset_id_col: str = "asset_id"
 71:     signal_col: str = "signal"
 72:     confidence_col: str | None = "confidence"
 73:     model_id_col: str | None = "model_id"
 74: 
 75:     def get_dtypes(self) -> dict[str, pl.DataType]:
 76:         """Get Polars data types for the schema."""
 77:         dtypes = {
 78:             self.timestamp_col: pl.Datetime("ns"),
 79:             self.asset_id_col: pl.Utf8,
 80:             self.signal_col: pl.Float64,
 81:         }
 82: 
 83:         if self.confidence_col:
 84:             dtypes[self.confidence_col] = pl.Float64
 85:         if self.model_id_col:
 86:             dtypes[self.model_id_col] = pl.Utf8
 87: 
 88:         return dtypes
 89: 
 90:     def validate(self, df: pl.DataFrame) -> None:
 91:         """Validate a DataFrame against this schema."""
 92:         required_cols = [
 93:             self.timestamp_col,
 94:             self.asset_id_col,
 95:             self.signal_col,
 96:         ]
 97: 
 98:         missing_cols = set(required_cols) - set(df.columns)
 99:         if missing_cols:
100:             raise ValueError(f"Missing required columns: {missing_cols}")
101: 
102:         # Validate data types
103:         for col, expected_dtype in self.get_dtypes().items():
104:             if col in df.columns:
105:                 actual_dtype = df[col].dtype
106:                 if not self._compatible_dtypes(actual_dtype, expected_dtype):
107:                     raise TypeError(
108:                         f"Column {col} has type {actual_dtype}, expected {expected_dtype}",
109:                     )
110: 
111:     def _compatible_dtypes(self, actual: pl.DataType, expected: pl.DataType) -> bool:
112:         """Check if data types are compatible."""
113:         # Allow int to float conversion
114:         if expected == pl.Float64 and actual in [pl.Int32, pl.Int64]:
115:             return True
116:         # Allow different datetime precisions
117:         if isinstance(expected, pl.Datetime) and isinstance(actual, pl.Datetime):
118:             return True
119:         return actual == expected
````

## File: src/qengine/execution/__init__.py
````python
 1: """Execution module for QEngine."""
 2: 
 3: from qengine.execution.broker import Broker, SimulationBroker
 4: from qengine.execution.order import Order, OrderState
 5: 
 6: __all__ = [
 7:     "Broker",
 8:     "Order",
 9:     "OrderState",
10:     "SimulationBroker",
11: ]
````

## File: src/qengine/execution/broker.py
````python
  1: """Broker implementations for QEngine."""
  2: 
  3: import logging
  4: from abc import ABC, abstractmethod
  5: from collections import defaultdict
  6: from datetime import datetime
  7: from typing import TYPE_CHECKING, Any
  8: 
  9: from qengine.core.assets import AssetRegistry
 10: from qengine.core.event import FillEvent, MarketEvent
 11: from qengine.core.types import (
 12:     AssetId,
 13:     OrderId,
 14:     OrderSide,
 15:     OrderStatus,
 16:     OrderType,
 17:     Price,
 18:     Quantity,
 19: )
 20: from qengine.execution.order import Order, OrderState
 21: from qengine.portfolio.margin import MarginAccount
 22: 
 23: if TYPE_CHECKING:
 24:     from qengine.execution.commission import CommissionModel
 25:     from qengine.execution.market_impact import MarketImpactModel
 26:     from qengine.execution.slippage import SlippageModel
 27: 
 28: logger = logging.getLogger(__name__)
 29: 
 30: 
 31: class Broker(ABC):
 32:     """Abstract base class for broker implementations."""
 33: 
 34:     @abstractmethod
 35:     def submit_order(self, order: Order) -> OrderId:
 36:         """Submit an order for execution."""
 37: 
 38:     @abstractmethod
 39:     def cancel_order(self, order_id: OrderId) -> bool:
 40:         """Cancel an existing order."""
 41: 
 42:     @abstractmethod
 43:     def get_order(self, order_id: OrderId) -> Order | None:
 44:         """Get order by ID."""
 45: 
 46:     @abstractmethod
 47:     def get_open_orders(self, asset_id: AssetId | None = None) -> list[Order]:
 48:         """Get all open orders, optionally filtered by asset."""
 49: 
 50:     @abstractmethod
 51:     def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
 52:         """Process market event and generate fills."""
 53: 
 54: 
 55: class SimulationBroker(Broker):
 56:     """
 57:     Simulated broker for backtesting.
 58: 
 59:     Handles order execution with configurable realism models.
 60:     Supports multiple asset classes including equities, futures, options, FX, and crypto.
 61:     """
 62: 
 63:     def __init__(
 64:         self,
 65:         initial_cash: float = 100000.0,
 66:         asset_registry: AssetRegistry | None = None,
 67:         commission_model: "CommissionModel | None" = None,
 68:         slippage_model: "SlippageModel | None" = None,
 69:         market_impact_model: "MarketImpactModel | None" = None,
 70:         fill_model: Any | None = None,
 71:         enable_margin: bool = True,
 72:     ):
 73:         """
 74:         Initialize simulation broker.
 75: 
 76:         Args:
 77:             initial_cash: Starting cash balance
 78:             asset_registry: Registry of asset specifications
 79:             commission_model: Model for calculating commissions
 80:             slippage_model: Model for calculating slippage
 81:             market_impact_model: Model for calculating market impact
 82:             fill_model: Model for determining fills
 83:             enable_margin: Whether to enable margin trading for derivatives
 84:         """
 85:         self.cash = initial_cash
 86:         self.asset_registry = asset_registry or AssetRegistry()
 87:         self.commission_model = commission_model
 88:         self.slippage_model = slippage_model
 89:         self.market_impact_model = market_impact_model
 90:         self.fill_model = fill_model
 91:         self.enable_margin = enable_margin
 92: 
 93:         # Order tracking
 94:         self._orders: dict[OrderId, Order] = {}
 95:         self._open_orders: dict[AssetId, list[Order]] = defaultdict(list)
 96:         self._stop_orders: dict[AssetId, list[Order]] = defaultdict(list)
 97:         self._trailing_stops: dict[AssetId, list[Order]] = defaultdict(list)
 98:         self._bracket_orders: dict[AssetId, list[Order]] = defaultdict(list)
 99: 
100:         # Position tracking
101:         self._positions: dict[AssetId, Quantity] = defaultdict(float)
102: 
103:         # Margin account for derivatives
104:         if enable_margin:
105:             self.margin_account = MarginAccount(initial_cash, self.asset_registry)
106: 
107:         # Latest market prices
108:         self._last_prices: dict[AssetId, Price] = {}
109: 
110:         # Statistics
111:         self._total_commission = 0.0
112:         self._total_slippage = 0.0
113:         self._fill_count = 0
114: 
115:     def submit_order(self, order: Order) -> OrderId:
116:         """
117:         Submit an order for execution.
118: 
119:         Args:
120:             order: Order to submit
121: 
122:         Returns:
123:             Order ID
124:         """
125:         # Store order
126:         self._orders[order.order_id] = order
127: 
128:         # Update order state
129:         order.state = OrderState.SUBMITTED
130:         order.status = OrderStatus.SUBMITTED
131:         order.submitted_time = datetime.now()
132: 
133:         # Route based on order type
134:         if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
135:             self._stop_orders[order.asset_id].append(order)
136:         elif order.order_type == OrderType.TRAILING_STOP:
137:             self._trailing_stops[order.asset_id].append(order)
138:             # Initialize trailing stop price
139:             if order.asset_id in self._last_prices:
140:                 order.update_trailing_stop(self._last_prices[order.asset_id])
141:         elif order.order_type == OrderType.BRACKET:
142:             # Bracket orders start as regular orders and create legs after fill
143:             self._open_orders[order.asset_id].append(order)
144:         else:
145:             self._open_orders[order.asset_id].append(order)
146: 
147:             # Try immediate execution for market orders
148:             if order.order_type == OrderType.MARKET:
149:                 if order.asset_id in self._last_prices:
150:                     self._try_fill_order(order, self._last_prices[order.asset_id])
151: 
152:         logger.debug(f"Submitted order: {order}")
153:         return order.order_id
154: 
155:     def cancel_order(self, order_id: OrderId) -> bool:
156:         """
157:         Cancel an existing order.
158: 
159:         Args:
160:             order_id: ID of order to cancel
161: 
162:         Returns:
163:             True if cancelled successfully
164:         """
165:         if order_id not in self._orders:
166:             return False
167: 
168:         order = self._orders[order_id]
169: 
170:         if not order.is_active:
171:             return False
172: 
173:         # Remove from order queues
174:         if order in self._open_orders[order.asset_id]:
175:             self._open_orders[order.asset_id].remove(order)
176:         if order in self._stop_orders[order.asset_id]:
177:             self._stop_orders[order.asset_id].remove(order)
178:         if order in self._trailing_stops[order.asset_id]:
179:             self._trailing_stops[order.asset_id].remove(order)
180:         # Bracket orders are tracked in open orders, not separately during submission
181: 
182:         # Update order state
183:         order.cancel()
184: 
185:         logger.debug(f"Cancelled order: {order}")
186:         return True
187: 
188:     def get_order(self, order_id: OrderId) -> Order | None:
189:         """Get order by ID."""
190:         return self._orders.get(order_id)
191: 
192:     def get_open_orders(self, asset_id: AssetId | None = None) -> list[Order]:
193:         """Get all open orders, optionally filtered by asset."""
194:         if asset_id:
195:             return [o for o in self._open_orders[asset_id] if o.is_active]
196:         orders = []
197:         for asset_orders in self._open_orders.values():
198:             orders.extend([o for o in asset_orders if o.is_active])
199:         return orders
200: 
201:     def on_market_event(self, event: MarketEvent) -> list[FillEvent]:
202:         """
203:         Process market event and generate fills.
204: 
205:         Args:
206:             event: Market data event
207: 
208:         Returns:
209:             List of fill events generated
210:         """
211:         fills = []
212:         asset_id = event.asset_id
213: 
214:         # Determine execution price
215:         if event.close is not None:
216:             price = event.close
217:         elif event.price is not None:
218:             price = event.price
219:         else:
220:             return fills  # No price available
221: 
222:         # Update last known price
223:         self._last_prices[asset_id] = price
224: 
225:         # Check stop orders for triggering
226:         triggered_stops = []
227:         for order in list(self._stop_orders[asset_id]):
228:             if self._should_trigger_stop(order, price):
229:                 triggered_stops.append(order)
230:                 self._stop_orders[asset_id].remove(order)
231: 
232:         # Process triggered stops immediately
233:         for order in triggered_stops:
234:             if order.order_type == OrderType.STOP:
235:                 # Convert to market order and try to fill immediately
236:                 original_type = order.order_type
237:                 order.metadata["original_type"] = "STOP"
238:                 order.order_type = OrderType.MARKET
239:                 fill_event = self._try_fill_order(order, price, event.timestamp)
240:                 if fill_event:
241:                     fills.append(fill_event)
242: 
243:                     # FIX: Handle OCO logic for bracket legs that are filled as stop orders
244:                     if (
245:                         order.is_filled
246:                         and order.child_order_ids
247:                         and order.metadata.get("bracket_type")
248:                     ):
249:                         # This is a bracket leg that was filled - cancel sibling orders
250:                         for child_id in order.child_order_ids:
251:                             if child_id in self._orders:
252:                                 self.cancel_order(child_id)
253: 
254:                 else:
255:                     # If couldn't fill as market order, restore type and add to open orders
256:                     order.order_type = original_type
257:                     del order.metadata["original_type"]
258:                     self._open_orders[asset_id].append(order)
259:             elif order.order_type == OrderType.STOP_LIMIT:
260:                 # Keep as limit order after triggering
261:                 order.order_type = OrderType.LIMIT
262:                 self._open_orders[asset_id].append(order)
263: 
264:         # Update and check trailing stops
265:         triggered_trailing = []
266:         for order in list(self._trailing_stops[asset_id]):
267:             # Update trailing stop level
268:             order.update_trailing_stop(price)
269:             # Check if triggered
270:             if order.can_fill(price):
271:                 triggered_trailing.append(order)
272:                 self._trailing_stops[asset_id].remove(order)
273: 
274:         # Process triggered trailing stops immediately (as market orders)
275:         for order in triggered_trailing:
276:             original_type = order.order_type
277:             order.metadata["original_type"] = "TRAILING_STOP"
278:             order.order_type = OrderType.MARKET
279:             fill_event = self._try_fill_order(order, price, event.timestamp)
280:             if fill_event:
281:                 fills.append(fill_event)
282: 
283:                 # FIX: Handle OCO logic for bracket legs that are filled as trailing stops
284:                 if order.is_filled and order.child_order_ids and order.metadata.get("bracket_type"):
285:                     # This is a bracket leg that was filled - cancel sibling orders
286:                     for child_id in order.child_order_ids:
287:                         if child_id in self._orders:
288:                             self.cancel_order(child_id)
289: 
290:             else:
291:                 # If couldn't fill as market order, restore type and add to open orders
292:                 order.order_type = original_type
293:                 del order.metadata["original_type"]
294:                 self._open_orders[asset_id].append(order)
295: 
296:         # Process open orders
297:         for order in list(self._open_orders[asset_id]):
298:             if not order.is_active:
299:                 continue
300: 
301:             fill_event = self._try_fill_order(order, price, event.timestamp)
302:             if fill_event:
303:                 fills.append(fill_event)
304: 
305:                 # Remove filled orders
306:                 if order.is_filled:
307:                     self._open_orders[asset_id].remove(order)
308: 
309:                     # Handle bracket order completion
310:                     if order.order_type == OrderType.BRACKET:
311:                         self._handle_bracket_fill(order, fill_event)
312:                     # Handle OCO (One-Cancels-Other) logic for bracket legs
313:                     # Only cancel sibling orders if this is a bracket leg (not the parent)
314:                     elif order.child_order_ids and order.metadata.get("bracket_type"):
315:                         for child_id in order.child_order_ids:
316:                             if child_id in self._orders:
317:                                 self.cancel_order(child_id)
318: 
319:         return fills
320: 
321:     def _should_trigger_stop(self, order: Order, price: Price) -> bool:
322:         """Check if stop order should be triggered."""
323:         if order.stop_price is None:
324:             return False
325: 
326:         if order.is_buy:
327:             return price >= order.stop_price
328:         return price <= order.stop_price
329: 
330:     def _handle_bracket_fill(self, parent_order: Order, fill_event) -> None:
331:         """
332:         Handle completion of bracket order by creating stop-loss and take-profit orders.
333: 
334:         Args:
335:             parent_order: The filled bracket order
336:             fill_event: The fill event that completed the order
337:         """
338:         if parent_order.order_type != OrderType.BRACKET:
339:             return
340: 
341:         if parent_order.stop_loss is None or parent_order.profit_target is None:
342:             return
343: 
344:         # Bracket order was filled from open orders, no need to remove from special tracking
345: 
346:         # Create stop-loss order (opposite side)
347:         stop_side = OrderSide.SELL if parent_order.is_buy else OrderSide.BUY
348:         stop_order = Order(
349:             asset_id=parent_order.asset_id,
350:             order_type=OrderType.STOP,
351:             side=stop_side,
352:             quantity=parent_order.filled_quantity,
353:             stop_price=parent_order.stop_loss,
354:             parent_order_id=parent_order.order_id,
355:             metadata={"bracket_type": "stop_loss"},
356:         )
357: 
358:         # Create take-profit order (opposite side)
359:         profit_order = Order(
360:             asset_id=parent_order.asset_id,
361:             order_type=OrderType.LIMIT,
362:             side=stop_side,
363:             quantity=parent_order.filled_quantity,
364:             limit_price=parent_order.profit_target,
365:             parent_order_id=parent_order.order_id,
366:             metadata={"bracket_type": "take_profit"},
367:         )
368: 
369:         # Link the orders as OCO (One-Cancels-Other)
370:         stop_order.child_order_ids.append(profit_order.order_id)
371:         profit_order.child_order_ids.append(stop_order.order_id)
372: 
373:         # Submit the bracket legs
374:         self.submit_order(stop_order)
375:         self.submit_order(profit_order)
376: 
377:         # Track parent-child relationship
378:         parent_order.child_order_ids.extend([stop_order.order_id, profit_order.order_id])
379: 
380:     def _try_fill_order(
381:         self,
382:         order: Order,
383:         market_price: Price,
384:         timestamp: datetime | None = None,
385:     ) -> FillEvent | None:
386:         """
387:         Try to fill an order at the given price.
388: 
389:         Args:
390:             order: Order to fill
391:             market_price: Current market price
392:             timestamp: Event timestamp
393: 
394:         Returns:
395:             FillEvent if order was filled
396:         """
397:         # Check if order can be filled
398:         if not order.can_fill(market_price):
399:             return None
400: 
401:         # Apply market impact to the market price
402:         impacted_market_price = self._get_market_price_with_impact(
403:             order,
404:             market_price,
405:             timestamp or datetime.now(),
406:         )
407: 
408:         # Determine fill price (with slippage on top of impact)
409:         fill_price = self._calculate_fill_price(order, impacted_market_price)
410: 
411:         # Determine fill quantity (for now, always fill completely)
412:         fill_quantity = order.remaining_quantity
413: 
414:         # Get asset specification
415:         asset_spec = self.asset_registry.get(order.asset_id)
416: 
417:         # Check margin requirements for derivatives
418:         if self.enable_margin and asset_spec and asset_spec.requires_margin:
419:             # Check margin for opening/increasing position
420:             if order.is_buy:
421:                 has_margin, required_margin = self.margin_account.check_margin_requirement(
422:                     order.asset_id,
423:                     fill_quantity,
424:                     fill_price,
425:                 )
426:                 if not has_margin:
427:                     # Try partial fill within margin
428:                     max_quantity = (
429:                         self.margin_account.available_margin / required_margin * fill_quantity
430:                     )
431:                     if max_quantity < asset_spec.min_quantity:
432:                         return None
433:                     fill_quantity = max_quantity
434:             elif order.is_sell and order.asset_id not in self._positions:
435:                 # Short selling - check margin
436:                 has_margin, required_margin = self.margin_account.check_margin_requirement(
437:                     order.asset_id,
438:                     -fill_quantity,
439:                     fill_price,
440:                 )
441:                 if not has_margin:
442:                     return None
443:         else:
444:             # Standard cash trading for equities
445:             if order.is_buy:
446:                 # Estimate commission for calculation
447:                 estimated_commission = self._calculate_commission(order, fill_quantity, fill_price)
448:                 required_cash = fill_quantity * fill_price + estimated_commission
449:                 if required_cash > self.cash:
450:                     # Partial fill to available cash
451:                     fill_quantity = (self.cash - estimated_commission) / fill_price
452:                     if fill_quantity < 0.01:  # Minimum fill size
453:                         return None
454: 
455:             # Check if we have enough shares for sell orders (non-short)
456:             # Only allow short selling if explicitly enabled in asset spec
457:             # Note: This check is only for regular orders, not for triggered stops
458:             if order.is_sell and (not asset_spec or not asset_spec.short_enabled):
459:                 # Allow sell orders that were originally stop/trailing stops
460:                 # (they would have been converted to MARKET by now)
461:                 if order.metadata.get("original_type") not in ["STOP", "TRAILING_STOP"]:
462:                     available_shares = self._positions[order.asset_id]
463:                     if available_shares < fill_quantity:
464:                         fill_quantity = available_shares
465:                         if fill_quantity <= 0:
466:                             return None
467: 
468:         # Calculate costs (asset-specific for different classes)
469:         commission = self._calculate_commission(order, fill_quantity, fill_price, asset_spec)
470:         slippage = self._calculate_slippage(
471:             order,
472:             fill_quantity,
473:             market_price,
474:             fill_price,
475:             asset_spec,
476:         )
477: 
478:         # Update order
479:         order.update_fill(fill_quantity, fill_price, commission, timestamp)
480: 
481:         # Update positions and cash
482:         if order.is_buy:
483:             self._positions[order.asset_id] += fill_quantity
484:             self.cash -= fill_quantity * fill_price + commission
485:         else:
486:             self._positions[order.asset_id] -= fill_quantity
487:             self.cash += fill_quantity * fill_price - commission
488: 
489:         # Update statistics
490:         self._total_commission += commission
491:         self._total_slippage += slippage
492:         self._fill_count += 1
493: 
494:         # Update market impact after fill
495:         self._update_market_impact(
496:             order,
497:             fill_quantity,
498:             market_price,
499:             timestamp or datetime.now(),
500:         )
501: 
502:         # Create fill event
503:         fill_event = FillEvent(
504:             timestamp=timestamp or datetime.now(),
505:             order_id=order.order_id,
506:             trade_id=f"T{self._fill_count:06d}",
507:             asset_id=order.asset_id,
508:             side=order.side,
509:             fill_quantity=fill_quantity,
510:             fill_price=fill_price,
511:             commission=commission,
512:             slippage=slippage,
513:         )
514: 
515:         logger.debug(f"Filled order: {order} with {fill_event}")
516:         return fill_event
517: 
518:     def _calculate_fill_price(self, order: Order, market_price: Price) -> Price:
519:         """Calculate the actual fill price including slippage."""
520:         if self.slippage_model:
521:             return self.slippage_model.calculate_fill_price(order, market_price)
522: 
523:         # Default simple slippage: 0.01% for market orders
524:         if order.order_type == OrderType.MARKET:
525:             if order.is_buy:
526:                 return market_price * 1.0001
527:             return market_price * 0.9999
528: 
529:         # Limit orders fill at limit price or better
530:         if order.order_type == OrderType.LIMIT and order.limit_price is not None:
531:             if order.is_buy:
532:                 return min(order.limit_price, market_price)
533:             return max(order.limit_price, market_price)
534: 
535:         return market_price
536: 
537:     def _get_market_price_with_impact(
538:         self,
539:         order: Order,
540:         market_price: Price,
541:         timestamp: datetime,
542:     ) -> Price:
543:         """Get market price adjusted for market impact."""
544:         if not self.market_impact_model:
545:             return market_price
546: 
547:         # Get current cumulative impact for this asset
548:         current_impact = self.market_impact_model.get_current_impact(
549:             order.asset_id,
550:             timestamp,
551:         )
552: 
553:         # Apply existing impact to market price
554:         return market_price + current_impact
555: 
556:     def _update_market_impact(
557:         self,
558:         order: Order,
559:         fill_quantity: Quantity,
560:         market_price: Price,
561:         timestamp: datetime,
562:     ) -> None:
563:         """Update market impact state after a fill."""
564:         if not self.market_impact_model:
565:             return
566: 
567:         # Calculate new impact from this trade
568:         permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
569:             order,
570:             fill_quantity,
571:             market_price,
572:             timestamp,
573:         )
574: 
575:         # Update market state
576:         self.market_impact_model.update_market_state(
577:             order.asset_id,
578:             permanent_impact,
579:             temporary_impact,
580:             timestamp,
581:         )
582: 
583:     def _calculate_commission(
584:         self,
585:         order: Order,
586:         fill_quantity: Quantity,
587:         fill_price: Price,
588:         asset_spec=None,
589:     ) -> float:
590:         """Calculate commission for the fill."""
591:         if self.commission_model:
592:             return self.commission_model.calculate(order, fill_quantity, fill_price)
593: 
594:         if asset_spec:
595:             # Use asset-specific fee structure
596:             notional = fill_quantity * fill_price * getattr(asset_spec, "contract_size", 1.0)
597:             if order.order_type == OrderType.MARKET:
598:                 return notional * asset_spec.taker_fee
599:             return notional * asset_spec.maker_fee
600: 
601:         # Simple flat commission: $1 per trade for equities
602:         return 1.0
603: 
604:     def _calculate_slippage(
605:         self,
606:         order: Order,
607:         fill_quantity: Quantity,
608:         market_price: Price,
609:         fill_price: Price,
610:         asset_spec=None,
611:     ) -> float:
612:         """Calculate slippage cost."""
613:         if self.slippage_model:
614:             return self.slippage_model.calculate_slippage_cost(
615:                 order,
616:                 fill_quantity,
617:                 market_price,
618:                 fill_price,
619:             )
620: 
621:         # Default asset-specific slippage
622:         if asset_spec:
623:             slippage_rate = 0.0001  # 1 bp default
624:             if asset_spec.asset_class.value == "crypto":
625:                 slippage_rate = 0.001  # 10 bp for crypto
626:             elif asset_spec.asset_class.value == "fx":
627:                 slippage_rate = 0.00005  # 0.5 bp for FX
628:             elif asset_spec.asset_class.value == "future":
629:                 slippage_rate = 0.0002  # 2 bp for futures
630: 
631:             notional = fill_quantity * market_price * getattr(asset_spec, "contract_size", 1.0)
632:             return notional * slippage_rate
633: 
634:         # Simple calculation: difference between market and fill price
635:         return abs(fill_price - market_price) * fill_quantity
636: 
637:     def get_position(self, asset_id: AssetId) -> Quantity:
638:         """Get current position for an asset."""
639:         return self._positions.get(asset_id, 0.0)
640: 
641:     def get_positions(self) -> dict[AssetId, Quantity]:
642:         """Get all current positions."""
643:         return dict(self._positions)
644: 
645:     def get_cash(self) -> float:
646:         """Get current cash balance."""
647:         return self.cash
648: 
649:     def get_statistics(self) -> dict[str, Any]:
650:         """Get broker statistics."""
651:         return {
652:             "total_commission": self._total_commission,
653:             "total_slippage": self._total_slippage,
654:             "fill_count": self._fill_count,
655:             "open_orders": sum(len(orders) for orders in self._open_orders.values()),
656:             "stop_orders": sum(len(orders) for orders in self._stop_orders.values()),
657:         }
658: 
659:     def initialize(self, portfolio, event_bus) -> None:
660:         """Initialize broker with portfolio and event bus.
661: 
662:         Args:
663:             portfolio: Portfolio instance for position tracking
664:             event_bus: Event bus for publishing fill events
665:         """
666:         self.portfolio = portfolio
667:         self.event_bus = event_bus
668:         logger.debug("SimulationBroker initialized")
669: 
670:     def on_order_event(self, event) -> None:
671:         """Handle order event from strategy.
672: 
673:         Args:
674:             event: OrderEvent to process
675:         """
676:         from qengine.execution.order import Order
677: 
678:         # Create Order object from OrderEvent
679:         order = Order(
680:             order_id=event.order_id,
681:             asset_id=event.asset_id,
682:             order_type=event.order_type,
683:             side=event.side,
684:             quantity=event.quantity,
685:             limit_price=getattr(event, "limit_price", None),
686:             stop_price=getattr(event, "stop_price", None),
687:             time_in_force=getattr(event, "time_in_force", None),
688:         )
689: 
690:         # Submit the order
691:         self.submit_order(order)
692: 
693:     def finalize(self) -> None:
694:         """Finalize broker at end of backtest."""
695:         # Cancel all remaining open orders
696:         for asset_orders in self._open_orders.values():
697:             for order in list(asset_orders):
698:                 if order.is_active:
699:                     order.cancel()
700: 
701:         for asset_orders in self._stop_orders.values():
702:             for order in list(asset_orders):
703:                 if order.is_active:
704:                     order.cancel()
705: 
706:         logger.info(f"SimulationBroker finalized. Total fills: {self._fill_count}")
707: 
708:     def get_trades(self) -> Any:
709:         """Get all executed trades.
710: 
711:         Returns:
712:             DataFrame or list of trades
713:         """
714:         import polars as pl
715: 
716:         trades = []
717:         for order_id, order in self._orders.items():
718:             if order.filled_quantity > 0:
719:                 trades.append(
720:                     {
721:                         "order_id": order_id,
722:                         "asset_id": order.asset_id,
723:                         "side": order.side.value,
724:                         "quantity": order.filled_quantity,
725:                         "price": order.average_fill_price,
726:                         "commission": order.commission,
727:                         "status": order.status.value,
728:                         "submitted_time": order.submitted_time,
729:                         "filled_time": order.filled_time,
730:                     },
731:                 )
732: 
733:         if trades:
734:             return pl.DataFrame(trades)
735:         return pl.DataFrame()
736: 
737:     def reset(self) -> None:
738:         """Reset broker to initial state."""
739:         self._orders.clear()
740:         self._open_orders.clear()
741:         self._stop_orders.clear()
742:         self._trailing_stops.clear()
743:         self._bracket_orders.clear()
744:         self._positions.clear()
745:         self._last_prices.clear()
746:         self.cash = self.initial_cash if hasattr(self, "initial_cash") else 100000.0
747:         self._total_commission = 0.0
748:         self._total_slippage = 0.0
749:         self._fill_count = 0
750:         logger.debug("SimulationBroker reset")
````

## File: src/qengine/execution/commission.py
````python
  1: """Commission models for realistic cost simulation."""
  2: 
  3: from abc import ABC, abstractmethod
  4: from typing import TYPE_CHECKING, Optional
  5: 
  6: if TYPE_CHECKING:
  7:     from qengine.core.types import Price, Quantity
  8:     from qengine.execution.order import Order
  9: 
 10: 
 11: class CommissionModel(ABC):
 12:     """Abstract base class for commission models."""
 13: 
 14:     @abstractmethod
 15:     def calculate(
 16:         self,
 17:         order: "Order",
 18:         fill_quantity: "Quantity",
 19:         fill_price: "Price",
 20:     ) -> float:
 21:         """Calculate commission for a filled order.
 22: 
 23:         Args:
 24:             order: The order being filled
 25:             fill_quantity: Quantity of the fill
 26:             fill_price: Price at which the order was filled
 27: 
 28:         Returns:
 29:             Commission amount in currency terms
 30:         """
 31: 
 32:     def __repr__(self) -> str:
 33:         """String representation."""
 34:         return f"{self.__class__.__name__}()"
 35: 
 36: 
 37: class NoCommission(CommissionModel):
 38:     """No commission model for testing."""
 39: 
 40:     def calculate(
 41:         self,
 42:         order: "Order",
 43:         fill_quantity: "Quantity",
 44:         fill_price: "Price",
 45:     ) -> float:
 46:         """Calculate zero commission."""
 47:         return 0.0
 48: 
 49: 
 50: class FlatCommission(CommissionModel):
 51:     """Flat commission per trade."""
 52: 
 53:     def __init__(self, commission: float = 1.0):
 54:         """Initialize flat commission model.
 55: 
 56:         Args:
 57:             commission: Flat fee per trade (default $1)
 58:         """
 59:         if commission < 0:
 60:             raise ValueError("Commission cannot be negative")
 61:         self.commission = commission
 62: 
 63:     def calculate(
 64:         self,
 65:         order: "Order",
 66:         fill_quantity: "Quantity",
 67:         fill_price: "Price",
 68:     ) -> float:
 69:         """Calculate flat commission."""
 70:         return self.commission
 71: 
 72:     def __repr__(self) -> str:
 73:         """String representation."""
 74:         return f"FlatCommission(commission={self.commission})"
 75: 
 76: 
 77: class PercentageCommission(CommissionModel):
 78:     """Percentage-based commission on trade value."""
 79: 
 80:     def __init__(self, rate: float = 0.001):
 81:         """Initialize percentage commission model.
 82: 
 83:         Args:
 84:             rate: Commission rate as decimal (0.001 = 0.1% = 10bps)
 85:         """
 86:         if rate < 0:
 87:             raise ValueError("Commission rate cannot be negative")
 88:         if rate > 0.1:  # 10% cap as sanity check
 89:             raise ValueError("Commission rate too high (>10%)")
 90:         self.rate = rate
 91: 
 92:     def calculate(
 93:         self,
 94:         order: "Order",
 95:         fill_quantity: "Quantity",
 96:         fill_price: "Price",
 97:     ) -> float:
 98:         """Calculate percentage-based commission."""
 99:         notional = fill_quantity * fill_price
100:         return notional * self.rate
101: 
102:     def __repr__(self) -> str:
103:         """String representation."""
104:         return f"PercentageCommission(rate={self.rate})"
105: 
106: 
107: class PerShareCommission(CommissionModel):
108:     """Per-share commission model."""
109: 
110:     def __init__(self, commission_per_share: float = 0.005):
111:         """Initialize per-share commission model.
112: 
113:         Args:
114:             commission_per_share: Commission per share (default $0.005)
115:         """
116:         if commission_per_share < 0:
117:             raise ValueError("Per-share commission cannot be negative")
118:         self.commission_per_share = commission_per_share
119: 
120:     def calculate(
121:         self,
122:         order: "Order",
123:         fill_quantity: "Quantity",
124:         fill_price: "Price",
125:     ) -> float:
126:         """Calculate per-share commission."""
127:         return fill_quantity * self.commission_per_share
128: 
129:     def __repr__(self) -> str:
130:         """String representation."""
131:         return f"PerShareCommission(commission_per_share={self.commission_per_share})"
132: 
133: 
134: class TieredCommission(CommissionModel):
135:     """Tiered commission based on trade size."""
136: 
137:     def __init__(
138:         self,
139:         tiers: list[tuple[float, float]] | None = None,
140:         minimum: float = 1.0,
141:     ):
142:         """Initialize tiered commission model.
143: 
144:         Args:
145:             tiers: List of (threshold, rate) tuples in ascending order
146:                    Default: [(10000, 0.0010), (50000, 0.0008), (100000, 0.0005)]
147:             minimum: Minimum commission per trade
148:         """
149:         if tiers is None:
150:             # Default tiers: better rates for larger trades
151:             tiers = [
152:                 (10_000, 0.0010),  # 10 bps for trades < $10k
153:                 (50_000, 0.0008),  # 8 bps for trades $10k-$50k
154:                 (100_000, 0.0005),  # 5 bps for trades $50k-$100k
155:                 (float("inf"), 0.0003),  # 3 bps for trades > $100k
156:             ]
157: 
158:         # Validate tiers
159:         prev_threshold = 0
160:         for threshold, rate in tiers:
161:             if threshold <= prev_threshold:
162:                 raise ValueError("Tiers must be in ascending order")
163:             if rate < 0:
164:                 raise ValueError("Commission rates cannot be negative")
165:             prev_threshold = threshold
166: 
167:         if minimum < 0:
168:             raise ValueError("Minimum commission cannot be negative")
169: 
170:         self.tiers = tiers
171:         self.minimum = minimum
172: 
173:     def calculate(
174:         self,
175:         order: "Order",
176:         fill_quantity: "Quantity",
177:         fill_price: "Price",
178:     ) -> float:
179:         """Calculate tiered commission based on notional value."""
180:         notional = fill_quantity * fill_price
181: 
182:         # Find applicable tier
183:         rate = self.tiers[-1][1]  # Default to highest tier
184:         for threshold, tier_rate in self.tiers:
185:             if notional < threshold:
186:                 rate = tier_rate
187:                 break
188: 
189:         commission = notional * rate
190:         return max(commission, self.minimum)
191: 
192:     def __repr__(self) -> str:
193:         """String representation."""
194:         return f"TieredCommission(tiers={self.tiers}, minimum={self.minimum})"
195: 
196: 
197: class MakerTakerCommission(CommissionModel):
198:     """Maker-taker commission model (exchanges)."""
199: 
200:     def __init__(
201:         self,
202:         maker_rate: float = -0.0002,  # Maker rebate
203:         taker_rate: float = 0.0003,  # Taker fee
204:     ):
205:         """Initialize maker-taker commission model.
206: 
207:         Args:
208:             maker_rate: Maker fee rate (negative for rebate)
209:             taker_rate: Taker fee rate
210:         """
211:         if taker_rate < 0:
212:             raise ValueError("Taker rate should be positive")
213:         if maker_rate > taker_rate:
214:             raise ValueError("Maker rate should not exceed taker rate")
215: 
216:         self.maker_rate = maker_rate
217:         self.taker_rate = taker_rate
218: 
219:     def calculate(
220:         self,
221:         order: "Order",
222:         fill_quantity: "Quantity",
223:         fill_price: "Price",
224:     ) -> float:
225:         """Calculate maker-taker commission based on order type."""
226:         from qengine.execution.order import OrderType
227: 
228:         notional = fill_quantity * fill_price
229: 
230:         # Market orders always take liquidity
231:         # Limit orders that execute immediately also take liquidity
232:         # For simplicity, we assume limit orders make liquidity
233:         rate = self.taker_rate if order.order_type == OrderType.MARKET else self.maker_rate
234: 
235:         commission = notional * rate
236:         # Negative commission means rebate, but ensure we don't pay too much rebate
237:         return max(commission, -notional * 0.001)  # Cap rebate at 10bps
238: 
239:     def __repr__(self) -> str:
240:         """String representation."""
241:         return f"MakerTakerCommission(maker_rate={self.maker_rate}, taker_rate={self.taker_rate})"
242: 
243: 
244: class AssetClassCommission(CommissionModel):
245:     """Asset class specific commission model."""
246: 
247:     def __init__(
248:         self,
249:         equity_rate: float = 0.001,  # 10 bps
250:         futures_per_contract: float = 2.50,  # $2.50 per contract
251:         options_per_contract: float = 0.65,  # $0.65 per contract
252:         forex_rate: float = 0.0002,  # 2 bps
253:         crypto_rate: float = 0.002,  # 20 bps
254:         default_rate: float = 0.001,  # 10 bps fallback
255:     ):
256:         """Initialize asset class commission model.
257: 
258:         Args:
259:             equity_rate: Commission rate for equities
260:             futures_per_contract: Commission per futures contract
261:             options_per_contract: Commission per options contract
262:             forex_rate: Commission rate for forex
263:             crypto_rate: Commission rate for crypto
264:             default_rate: Default commission rate
265:         """
266:         self.equity_rate = equity_rate
267:         self.futures_per_contract = futures_per_contract
268:         self.options_per_contract = options_per_contract
269:         self.forex_rate = forex_rate
270:         self.crypto_rate = crypto_rate
271:         self.default_rate = default_rate
272: 
273:     def calculate(
274:         self,
275:         order: "Order",
276:         fill_quantity: "Quantity",
277:         fill_price: "Price",
278:     ) -> float:
279:         """Calculate commission based on asset class."""
280:         # Determine asset class from symbol or metadata
281:         asset_class = order.metadata.get("asset_class", "equity")
282: 
283:         if asset_class == "futures":
284:             # Futures charge per contract
285:             return fill_quantity * self.futures_per_contract
286:         if asset_class == "options":
287:             # Options charge per contract (1 contract = 100 shares usually)
288:             contracts = fill_quantity / 100
289:             return contracts * self.options_per_contract
290:         if asset_class == "forex":
291:             notional = fill_quantity * fill_price
292:             return notional * self.forex_rate
293:         if asset_class == "crypto":
294:             notional = fill_quantity * fill_price
295:             return notional * self.crypto_rate
296:         if asset_class == "equity":
297:             notional = fill_quantity * fill_price
298:             return notional * self.equity_rate
299:         # Default rate for unknown asset classes
300:         notional = fill_quantity * fill_price
301:         return notional * self.default_rate
302: 
303:     def __repr__(self) -> str:
304:         """String representation."""
305:         return (
306:             f"AssetClassCommission("
307:             f"equity_rate={self.equity_rate}, "
308:             f"futures_per_contract={self.futures_per_contract}, "
309:             f"options_per_contract={self.options_per_contract}, "
310:             f"forex_rate={self.forex_rate}, "
311:             f"crypto_rate={self.crypto_rate})"
312:         )
313: 
314: 
315: class InteractiveBrokersCommission(CommissionModel):
316:     """Interactive Brokers tiered commission structure."""
317: 
318:     def __init__(self, tier: str = "fixed"):
319:         """Initialize IB commission model.
320: 
321:         Args:
322:             tier: Commission tier ('fixed' or 'tiered')
323:         """
324:         if tier not in ["fixed", "tiered"]:
325:             raise ValueError("Tier must be 'fixed' or 'tiered'")
326:         self.tier = tier
327: 
328:     def calculate(
329:         self,
330:         order: "Order",
331:         fill_quantity: "Quantity",
332:         fill_price: "Price",
333:     ) -> float:
334:         """Calculate IB commission."""
335:         if self.tier == "fixed":
336:             # Fixed pricing: $0.005 per share, $1 minimum, $1% max
337:             per_share = fill_quantity * 0.005
338:             min_commission = 1.0
339:             max_commission = fill_quantity * fill_price * 0.01
340:             return min(max(per_share, min_commission), max_commission)
341:         # Tiered pricing (simplified)
342:         fill_quantity * fill_price
343:         if fill_quantity <= 300:
344:             rate = 0.0035  # $0.0035 per share for first 300
345:         elif fill_quantity <= 3000:
346:             rate = 0.0025  # $0.0025 per share for next 2700
347:         else:
348:             rate = 0.0015  # $0.0015 per share above 3000
349: 
350:         commission = fill_quantity * rate
351:         return max(commission, 0.35)  # $0.35 minimum
352: 
353:     def __repr__(self) -> str:
354:         """String representation."""
355:         return f"InteractiveBrokersCommission(tier='{self.tier}')"
````

## File: src/qengine/execution/corporate_actions.py
````python
  1: """Corporate actions handling for QEngine.
  2: 
  3: Corporate actions are events that affect the equity structure of a company,
  4: requiring adjustments to positions, prices, and orders. This module provides
  5: a comprehensive framework for handling:
  6: 
  7: 1. Dividends (cash dividends, special dividends)
  8: 2. Stock splits and stock dividends
  9: 3. Mergers and acquisitions (cash, stock, mixed)
 10: 4. Spin-offs
 11: 5. Symbol changes/reorganizations
 12: 6. Rights offerings
 13: 
 14: All actions maintain point-in-time correctness and properly adjust positions,
 15: orders, and price histories.
 16: """
 17: 
 18: import logging
 19: from dataclasses import dataclass, field
 20: from datetime import date
 21: from typing import TYPE_CHECKING, Optional
 22: 
 23: if TYPE_CHECKING:
 24:     from qengine.core.types import AssetId, Price, Quantity
 25:     from qengine.execution.order import Order
 26: 
 27: logger = logging.getLogger(__name__)
 28: 
 29: 
 30: @dataclass
 31: class CorporateAction:
 32:     """Base class for corporate actions."""
 33: 
 34:     action_id: str
 35:     asset_id: "AssetId"
 36:     ex_date: date  # Ex-dividend date (when action takes effect)
 37:     record_date: date | None = None  # Record date for eligibility
 38:     payment_date: date | None = None  # When payment/distribution occurs
 39:     announcement_date: date | None = None  # When action was announced
 40:     metadata: dict[str, str] = field(default_factory=dict)
 41: 
 42:     def __post_init__(self):
 43:         """Validate dates."""
 44:         if self.record_date and self.ex_date and self.record_date > self.ex_date:
 45:             raise ValueError("Record date must be before ex-date")
 46: 
 47: 
 48: class CashDividend(CorporateAction):
 49:     """Cash dividend corporate action."""
 50: 
 51:     def __init__(
 52:         self,
 53:         action_id: str,
 54:         asset_id: "AssetId",
 55:         ex_date: date,
 56:         dividend_per_share: float,
 57:         currency: str = "USD",
 58:         record_date: date | None = None,
 59:         payment_date: date | None = None,
 60:         announcement_date: date | None = None,
 61:         metadata: dict[str, str] | None = None,
 62:     ):
 63:         super().__init__(
 64:             action_id=action_id,
 65:             asset_id=asset_id,
 66:             ex_date=ex_date,
 67:             record_date=record_date,
 68:             payment_date=payment_date,
 69:             announcement_date=announcement_date,
 70:             metadata=metadata or {},
 71:         )
 72:         self.dividend_per_share = dividend_per_share
 73:         self.currency = currency
 74: 
 75:     @property
 76:     def action_type(self) -> str:
 77:         return "DIVIDEND"
 78: 
 79: 
 80: class StockSplit(CorporateAction):
 81:     """Stock split corporate action."""
 82: 
 83:     def __init__(
 84:         self,
 85:         action_id: str,
 86:         asset_id: "AssetId",
 87:         ex_date: date,
 88:         split_ratio: float,
 89:         record_date: date | None = None,
 90:         payment_date: date | None = None,
 91:         announcement_date: date | None = None,
 92:         metadata: dict[str, str] | None = None,
 93:     ):
 94:         super().__init__(
 95:             action_id=action_id,
 96:             asset_id=asset_id,
 97:             ex_date=ex_date,
 98:             record_date=record_date,
 99:             payment_date=payment_date,
100:             announcement_date=announcement_date,
101:             metadata=metadata or {},
102:         )
103:         if split_ratio <= 0:
104:             raise ValueError("Split ratio must be positive")
105:         self.split_ratio = split_ratio
106: 
107:     @property
108:     def action_type(self) -> str:
109:         return "SPLIT"
110: 
111: 
112: class StockDividend(CorporateAction):
113:     """Stock dividend corporate action."""
114: 
115:     def __init__(
116:         self,
117:         action_id: str,
118:         asset_id: "AssetId",
119:         ex_date: date,
120:         dividend_ratio: float,
121:         record_date: date | None = None,
122:         payment_date: date | None = None,
123:         announcement_date: date | None = None,
124:         metadata: dict[str, str] | None = None,
125:     ):
126:         super().__init__(
127:             action_id=action_id,
128:             asset_id=asset_id,
129:             ex_date=ex_date,
130:             record_date=record_date,
131:             payment_date=payment_date,
132:             announcement_date=announcement_date,
133:             metadata=metadata or {},
134:         )
135:         self.dividend_ratio = dividend_ratio
136: 
137:     @property
138:     def action_type(self) -> str:
139:         return "STOCK_DIVIDEND"
140: 
141: 
142: class Merger(CorporateAction):
143:     """Merger/acquisition corporate action."""
144: 
145:     def __init__(
146:         self,
147:         action_id: str,
148:         asset_id: "AssetId",
149:         ex_date: date,
150:         target_asset_id: "AssetId",
151:         cash_consideration: float = 0.0,
152:         stock_consideration: float = 0.0,
153:         record_date: date | None = None,
154:         payment_date: date | None = None,
155:         announcement_date: date | None = None,
156:         metadata: dict[str, str] | None = None,
157:     ):
158:         super().__init__(
159:             action_id=action_id,
160:             asset_id=asset_id,
161:             ex_date=ex_date,
162:             record_date=record_date,
163:             payment_date=payment_date,
164:             announcement_date=announcement_date,
165:             metadata=metadata or {},
166:         )
167:         if cash_consideration == 0.0 and stock_consideration == 0.0:
168:             raise ValueError("Must have either cash or stock consideration")
169:         self.target_asset_id = target_asset_id
170:         self.cash_consideration = cash_consideration
171:         self.stock_consideration = stock_consideration
172: 
173:     @property
174:     def action_type(self) -> str:
175:         return "MERGER"
176: 
177: 
178: class SpinOff(CorporateAction):
179:     """Spin-off corporate action."""
180: 
181:     def __init__(
182:         self,
183:         action_id: str,
184:         asset_id: "AssetId",
185:         ex_date: date,
186:         new_asset_id: "AssetId",
187:         distribution_ratio: float,
188:         record_date: date | None = None,
189:         payment_date: date | None = None,
190:         announcement_date: date | None = None,
191:         metadata: dict[str, str] | None = None,
192:     ):
193:         super().__init__(
194:             action_id=action_id,
195:             asset_id=asset_id,
196:             ex_date=ex_date,
197:             record_date=record_date,
198:             payment_date=payment_date,
199:             announcement_date=announcement_date,
200:             metadata=metadata or {},
201:         )
202:         self.new_asset_id = new_asset_id
203:         self.distribution_ratio = distribution_ratio
204: 
205:     @property
206:     def action_type(self) -> str:
207:         return "SPINOFF"
208: 
209: 
210: class SymbolChange(CorporateAction):
211:     """Symbol change/reorganization."""
212: 
213:     def __init__(
214:         self,
215:         action_id: str,
216:         asset_id: "AssetId",
217:         ex_date: date,
218:         new_asset_id: "AssetId",
219:         conversion_ratio: float = 1.0,
220:         record_date: date | None = None,
221:         payment_date: date | None = None,
222:         announcement_date: date | None = None,
223:         metadata: dict[str, str] | None = None,
224:     ):
225:         super().__init__(
226:             action_id=action_id,
227:             asset_id=asset_id,
228:             ex_date=ex_date,
229:             record_date=record_date,
230:             payment_date=payment_date,
231:             announcement_date=announcement_date,
232:             metadata=metadata or {},
233:         )
234:         self.new_asset_id = new_asset_id
235:         self.conversion_ratio = conversion_ratio
236: 
237:     @property
238:     def action_type(self) -> str:
239:         return "SYMBOL_CHANGE"
240: 
241: 
242: class RightsOffering(CorporateAction):
243:     """Rights offering corporate action."""
244: 
245:     def __init__(
246:         self,
247:         action_id: str,
248:         asset_id: "AssetId",
249:         ex_date: date,
250:         subscription_price: float,
251:         rights_ratio: float,
252:         shares_per_right: float,
253:         expiration_date: date,
254:         record_date: date | None = None,
255:         payment_date: date | None = None,
256:         announcement_date: date | None = None,
257:         metadata: dict[str, str] | None = None,
258:     ):
259:         super().__init__(
260:             action_id=action_id,
261:             asset_id=asset_id,
262:             ex_date=ex_date,
263:             record_date=record_date,
264:             payment_date=payment_date,
265:             announcement_date=announcement_date,
266:             metadata=metadata or {},
267:         )
268:         self.subscription_price = subscription_price
269:         self.rights_ratio = rights_ratio
270:         self.shares_per_right = shares_per_right
271:         self.expiration_date = expiration_date
272: 
273:     @property
274:     def action_type(self) -> str:
275:         return "RIGHTS_OFFERING"
276: 
277: 
278: class CorporateActionProcessor:
279:     """Processes corporate actions and adjusts positions/orders."""
280: 
281:     def __init__(self):
282:         """Initialize corporate action processor."""
283:         self.pending_actions: list[CorporateAction] = []
284:         self.processed_actions: list[CorporateAction] = []
285: 
286:     def add_action(self, action: CorporateAction) -> None:
287:         """Add a corporate action for processing.
288: 
289:         Args:
290:             action: Corporate action to add
291:         """
292:         self.pending_actions.append(action)
293:         # Sort by ex-date to ensure proper processing order
294:         self.pending_actions.sort(key=lambda a: a.ex_date)
295:         logger.info(
296:             f"Added corporate action: {action.action_id} ({action.action_type}) for {action.asset_id}",
297:         )
298: 
299:     def get_pending_actions(self, as_of_date: date) -> list[CorporateAction]:
300:         """Get actions that should be processed on the given date.
301: 
302:         Args:
303:             as_of_date: Date to check for pending actions
304: 
305:         Returns:
306:             List of actions to process
307:         """
308:         return [action for action in self.pending_actions if action.ex_date <= as_of_date]
309: 
310:     def process_actions(
311:         self,
312:         as_of_date: date,
313:         positions: dict["AssetId", "Quantity"],
314:         orders: list["Order"],
315:         cash: float,
316:     ) -> tuple[dict["AssetId", "Quantity"], list["Order"], float, list[str]]:
317:         """Process all pending corporate actions as of the given date.
318: 
319:         Args:
320:             as_of_date: Date to process actions through
321:             positions: Current position quantities by asset
322:             orders: List of open orders
323:             cash: Current cash balance
324: 
325:         Returns:
326:             Tuple of (updated_positions, updated_orders, updated_cash, notifications)
327:         """
328:         notifications = []
329:         updated_positions = positions.copy()
330:         updated_orders = orders.copy()
331:         updated_cash = cash
332: 
333:         pending = self.get_pending_actions(as_of_date)
334: 
335:         for action in pending:
336:             logger.info(f"Processing {action.action_type} for {action.asset_id} on {as_of_date}")
337: 
338:             if isinstance(action, CashDividend):
339:                 updated_cash, notification = self._process_cash_dividend(
340:                     action,
341:                     updated_positions,
342:                     updated_cash,
343:                 )
344:                 notifications.append(notification)
345: 
346:             elif isinstance(action, StockSplit):
347:                 updated_positions, updated_orders, notification = self._process_stock_split(
348:                     action,
349:                     updated_positions,
350:                     updated_orders,
351:                 )
352:                 notifications.append(notification)
353: 
354:             elif isinstance(action, StockDividend):
355:                 updated_positions, notification = self._process_stock_dividend(
356:                     action,
357:                     updated_positions,
358:                 )
359:                 notifications.append(notification)
360: 
361:             elif isinstance(action, Merger):
362:                 updated_positions, updated_cash, notification = self._process_merger(
363:                     action,
364:                     updated_positions,
365:                     updated_cash,
366:                 )
367:                 notifications.append(notification)
368: 
369:             elif isinstance(action, SpinOff):
370:                 updated_positions, notification = self._process_spinoff(
371:                     action,
372:                     updated_positions,
373:                 )
374:                 notifications.append(notification)
375: 
376:             elif isinstance(action, SymbolChange):
377:                 updated_positions, updated_orders, notification = self._process_symbol_change(
378:                     action,
379:                     updated_positions,
380:                     updated_orders,
381:                 )
382:                 notifications.append(notification)
383: 
384:             elif isinstance(action, RightsOffering):
385:                 # Rights offerings are complex and typically require user decision
386:                 # For now, just notify
387:                 notifications.append(
388:                     f"Rights offering for {action.asset_id}: "
389:                     f"{action.rights_ratio} rights per share, "
390:                     f"subscription price ${action.subscription_price:.2f}",
391:                 )
392: 
393:             # Move to processed
394:             self.processed_actions.append(action)
395:             self.pending_actions.remove(action)
396: 
397:         return updated_positions, updated_orders, updated_cash, notifications
398: 
399:     def _process_cash_dividend(
400:         self,
401:         dividend: CashDividend,
402:         positions: dict["AssetId", "Quantity"],
403:         cash: float,
404:     ) -> tuple[float, str]:
405:         """Process cash dividend.
406: 
407:         Args:
408:             dividend: Dividend action
409:             positions: Current positions
410:             cash: Current cash balance
411: 
412:         Returns:
413:             Tuple of (updated_cash, notification)
414:         """
415:         position = positions.get(dividend.asset_id, 0.0)
416:         if position > 0:
417:             dividend_payment = position * dividend.dividend_per_share
418:             cash += dividend_payment
419:             notification = (
420:                 f"Dividend received: {position:.0f} shares of {dividend.asset_id} "
421:                 f"× ${dividend.dividend_per_share:.4f} = ${dividend_payment:.2f}"
422:             )
423:             logger.info(notification)
424:             return cash, notification
425: 
426:         return cash, f"No position in {dividend.asset_id} for dividend"
427: 
428:     def _process_stock_split(
429:         self,
430:         split: StockSplit,
431:         positions: dict["AssetId", "Quantity"],
432:         orders: list["Order"],
433:     ) -> tuple[dict["AssetId", "Quantity"], list["Order"], str]:
434:         """Process stock split.
435: 
436:         Args:
437:             split: Stock split action
438:             positions: Current positions
439:             orders: Open orders
440: 
441:         Returns:
442:             Tuple of (updated_positions, updated_orders, notification)
443:         """
444:         # Adjust position
445:         if split.asset_id in positions:
446:             old_position = positions[split.asset_id]
447:             positions[split.asset_id] = old_position * split.split_ratio
448:             notification = (
449:                 f"Stock split: {split.asset_id} {split.split_ratio}:1 split - "
450:                 f"Position adjusted from {old_position:.0f} to {positions[split.asset_id]:.0f} shares"
451:             )
452:         else:
453:             notification = f"No position in {split.asset_id} for stock split"
454: 
455:         # Adjust open orders
456:         updated_orders = []
457:         for order in orders:
458:             if order.asset_id == split.asset_id:
459:                 # Adjust quantity and prices
460:                 order.quantity *= split.split_ratio
461:                 # Note: remaining_quantity is computed from quantity - filled_quantity
462:                 if order.limit_price is not None:
463:                     order.limit_price /= split.split_ratio
464:                 if order.stop_price is not None:
465:                     order.stop_price /= split.split_ratio
466: 
467:                 order.metadata["corporate_action"] = (
468:                     f"Split {split.split_ratio}:1 on {split.ex_date}"
469:                 )
470:             updated_orders.append(order)
471: 
472:         logger.info(notification)
473:         return positions, updated_orders, notification
474: 
475:     def _process_stock_dividend(
476:         self,
477:         stock_div: StockDividend,
478:         positions: dict["AssetId", "Quantity"],
479:     ) -> tuple[dict["AssetId", "Quantity"], str]:
480:         """Process stock dividend.
481: 
482:         Args:
483:             stock_div: Stock dividend action
484:             positions: Current positions
485: 
486:         Returns:
487:             Tuple of (updated_positions, notification)
488:         """
489:         if stock_div.asset_id in positions:
490:             old_position = positions[stock_div.asset_id]
491:             additional_shares = old_position * stock_div.dividend_ratio
492:             positions[stock_div.asset_id] += additional_shares
493: 
494:             notification = (
495:                 f"Stock dividend: {stock_div.asset_id} "
496:                 f"{stock_div.dividend_ratio * 100:.1f}% stock dividend - "
497:                 f"Received {additional_shares:.0f} additional shares"
498:             )
499:         else:
500:             notification = f"No position in {stock_div.asset_id} for stock dividend"
501: 
502:         logger.info(notification)
503:         return positions, notification
504: 
505:     def _process_merger(
506:         self,
507:         merger: Merger,
508:         positions: dict["AssetId", "Quantity"],
509:         cash: float,
510:     ) -> tuple[dict["AssetId", "Quantity"], float, str]:
511:         """Process merger/acquisition.
512: 
513:         Args:
514:             merger: Merger action
515:             positions: Current positions
516:             cash: Current cash balance
517: 
518:         Returns:
519:             Tuple of (updated_positions, updated_cash, notification)
520:         """
521:         if merger.asset_id not in positions or positions[merger.asset_id] <= 0:
522:             return positions, cash, f"No position in {merger.asset_id} for merger"
523: 
524:         old_shares = positions[merger.asset_id]
525: 
526:         # Remove old position
527:         del positions[merger.asset_id]
528: 
529:         # Add cash consideration
530:         cash_received = old_shares * merger.cash_consideration
531:         cash += cash_received
532: 
533:         # Add stock consideration
534:         if merger.stock_consideration > 0:
535:             new_shares = old_shares * merger.stock_consideration
536:             if merger.target_asset_id in positions:
537:                 positions[merger.target_asset_id] += new_shares
538:             else:
539:                 positions[merger.target_asset_id] = new_shares
540: 
541:         notification = (
542:             f"Merger: {merger.asset_id} → {merger.target_asset_id} - "
543:             f"{old_shares:.0f} shares converted to "
544:         )
545: 
546:         if cash_received > 0 and merger.stock_consideration > 0:
547:             notification += f"${cash_received:.2f} cash + {old_shares * merger.stock_consideration:.0f} {merger.target_asset_id} shares"
548:         elif cash_received > 0:
549:             notification += f"${cash_received:.2f} cash"
550:         else:
551:             notification += (
552:                 f"{old_shares * merger.stock_consideration:.0f} {merger.target_asset_id} shares"
553:             )
554: 
555:         logger.info(notification)
556:         return positions, cash, notification
557: 
558:     def _process_spinoff(
559:         self,
560:         spinoff: SpinOff,
561:         positions: dict["AssetId", "Quantity"],
562:     ) -> tuple[dict["AssetId", "Quantity"], str]:
563:         """Process spin-off.
564: 
565:         Args:
566:             spinoff: Spin-off action
567:             positions: Current positions
568: 
569:         Returns:
570:             Tuple of (updated_positions, notification)
571:         """
572:         if spinoff.asset_id not in positions or positions[spinoff.asset_id] <= 0:
573:             return positions, f"No position in {spinoff.asset_id} for spin-off"
574: 
575:         parent_shares = positions[spinoff.asset_id]
576:         spinoff_shares = parent_shares * spinoff.distribution_ratio
577: 
578:         # Add spin-off shares
579:         if spinoff.new_asset_id in positions:
580:             positions[spinoff.new_asset_id] += spinoff_shares
581:         else:
582:             positions[spinoff.new_asset_id] = spinoff_shares
583: 
584:         notification = (
585:             f"Spin-off: {spinoff.asset_id} distributed {spinoff_shares:.0f} shares of "
586:             f"{spinoff.new_asset_id} ({spinoff.distribution_ratio} per share)"
587:         )
588: 
589:         logger.info(notification)
590:         return positions, notification
591: 
592:     def _process_symbol_change(
593:         self,
594:         symbol_change: SymbolChange,
595:         positions: dict["AssetId", "Quantity"],
596:         orders: list["Order"],
597:     ) -> tuple[dict["AssetId", "Quantity"], list["Order"], str]:
598:         """Process symbol change.
599: 
600:         Args:
601:             symbol_change: Symbol change action
602:             positions: Current positions
603:             orders: Open orders
604: 
605:         Returns:
606:             Tuple of (updated_positions, updated_orders, notification)
607:         """
608:         # Update position
609:         if symbol_change.asset_id in positions:
610:             old_shares = positions[symbol_change.asset_id]
611:             new_shares = old_shares * symbol_change.conversion_ratio
612: 
613:             del positions[symbol_change.asset_id]
614:             positions[symbol_change.new_asset_id] = new_shares
615: 
616:             notification = (
617:                 f"Symbol change: {symbol_change.asset_id} → {symbol_change.new_asset_id} "
618:                 f"({old_shares:.0f} → {new_shares:.0f} shares)"
619:             )
620:         else:
621:             notification = f"Symbol change: {symbol_change.asset_id} → {symbol_change.new_asset_id} (no position)"
622: 
623:         # Update orders
624:         for order in orders:
625:             if order.asset_id == symbol_change.asset_id:
626:                 order.asset_id = symbol_change.new_asset_id
627:                 order.quantity *= symbol_change.conversion_ratio
628:                 # Note: remaining_quantity is computed from quantity - filled_quantity
629: 
630:                 if symbol_change.conversion_ratio != 1.0:
631:                     if order.limit_price is not None:
632:                         order.limit_price /= symbol_change.conversion_ratio
633:                     if order.stop_price is not None:
634:                         order.stop_price /= symbol_change.conversion_ratio
635: 
636:                 order.metadata["corporate_action"] = f"Symbol change on {symbol_change.ex_date}"
637: 
638:         logger.info(notification)
639:         return positions, orders, notification
640: 
641:     def adjust_price_for_actions(
642:         self,
643:         asset_id: "AssetId",
644:         price: "Price",
645:         as_of_date: date,
646:     ) -> "Price":
647:         """Adjust historical price for corporate actions.
648: 
649:         This is used to maintain price continuity in backtesting by adjusting
650:         historical prices for splits, dividends, etc.
651: 
652:         Args:
653:             asset_id: Asset to adjust price for
654:             price: Original price
655:             as_of_date: Date the price is from
656: 
657:         Returns:
658:             Adjusted price
659:         """
660:         adjusted_price = price
661: 
662:         # Apply adjustments for all actions after this date
663:         for action in self.processed_actions:
664:             if action.asset_id != asset_id or action.ex_date <= as_of_date:
665:                 continue
666: 
667:             if isinstance(action, StockSplit):
668:                 # Adjust price downward for future splits
669:                 adjusted_price /= action.split_ratio
670: 
671:             elif isinstance(action, CashDividend):
672:                 # Adjust price downward for future dividends
673:                 adjusted_price -= action.dividend_per_share
674: 
675:             elif isinstance(action, StockDividend):
676:                 # Adjust price for stock dividend
677:                 adjusted_price /= 1 + action.dividend_ratio
678: 
679:         return max(adjusted_price, 0.01)  # Minimum price floor
680: 
681:     def get_processed_actions(
682:         self,
683:         asset_id: Optional["AssetId"] = None,
684:         start_date: date | None = None,
685:         end_date: date | None = None,
686:     ) -> list[CorporateAction]:
687:         """Get processed corporate actions with optional filtering.
688: 
689:         Args:
690:             asset_id: Filter by asset ID
691:             start_date: Filter by start date (inclusive)
692:             end_date: Filter by end date (inclusive)
693: 
694:         Returns:
695:             List of matching corporate actions
696:         """
697:         filtered_actions = self.processed_actions
698: 
699:         if asset_id:
700:             filtered_actions = [a for a in filtered_actions if a.asset_id == asset_id]
701: 
702:         if start_date:
703:             filtered_actions = [a for a in filtered_actions if a.ex_date >= start_date]
704: 
705:         if end_date:
706:             filtered_actions = [a for a in filtered_actions if a.ex_date <= end_date]
707: 
708:         return filtered_actions
709: 
710:     def reset(self) -> None:
711:         """Reset processor state."""
712:         self.pending_actions.clear()
713:         self.processed_actions.clear()
714:         logger.info("Corporate action processor reset")
715: 
716: 
717: class CorporateActionDataProvider:
718:     """Provides corporate action data from various sources."""
719: 
720:     def __init__(self):
721:         """Initialize data provider."""
722:         self.actions: dict[str, CorporateAction] = {}
723: 
724:     def load_from_csv(self, file_path: str) -> None:
725:         """Load corporate actions from CSV file.
726: 
727:         Expected CSV format:
728:         action_id,asset_id,action_type,ex_date,dividend_per_share,split_ratio,...
729: 
730:         Args:
731:             file_path: Path to CSV file
732:         """
733:         import pandas as pd
734: 
735:         df = pd.read_csv(file_path)
736: 
737:         for _, row in df.iterrows():
738:             action = self._create_action_from_row(row)
739:             if action:
740:                 self.actions[action.action_id] = action
741:                 logger.info(f"Loaded corporate action: {action.action_id}")
742: 
743:     def _create_action_from_row(self, row) -> CorporateAction | None:
744:         """Create corporate action from CSV row."""
745:         try:
746:             action_type = row["action_type"].upper()
747:             import pandas as pd
748: 
749:             ex_date = pd.to_datetime(row["ex_date"]).date()
750: 
751:             base_args = {
752:                 "action_id": row["action_id"],
753:                 "asset_id": row["asset_id"],
754:                 "ex_date": ex_date,
755:                 "record_date": pd.to_datetime(row.get("record_date")).date()
756:                 if pd.notna(row.get("record_date"))
757:                 else None,
758:                 "payment_date": pd.to_datetime(row.get("payment_date")).date()
759:                 if pd.notna(row.get("payment_date"))
760:                 else None,
761:             }
762: 
763:             if action_type == "DIVIDEND":
764:                 return CashDividend(
765:                     dividend_per_share=float(row["dividend_per_share"]),
766:                     **base_args,
767:                 )
768:             if action_type == "SPLIT":
769:                 return StockSplit(
770:                     split_ratio=float(row["split_ratio"]),
771:                     **base_args,
772:                 )
773:             if action_type == "MERGER":
774:                 return Merger(
775:                     target_asset_id=row["target_asset_id"],
776:                     cash_consideration=float(row.get("cash_consideration", 0)),
777:                     stock_consideration=float(row.get("stock_consideration", 0)),
778:                     **base_args,
779:                 )
780:             if action_type == "SPINOFF":
781:                 return SpinOff(
782:                     new_asset_id=row["new_asset_id"],
783:                     distribution_ratio=float(row["distribution_ratio"]),
784:                     **base_args,
785:                 )
786:             if action_type == "SYMBOL_CHANGE":
787:                 return SymbolChange(
788:                     new_asset_id=row["new_asset_id"],
789:                     conversion_ratio=float(row.get("conversion_ratio", 1.0)),
790:                     **base_args,
791:                 )
792:             logger.warning(f"Unknown action type: {action_type}")
793:             return None
794: 
795:         except Exception as e:
796:             logger.error(f"Error creating action from row: {e}")
797:             return None
798: 
799:     def get_actions_for_asset(
800:         self,
801:         asset_id: "AssetId",
802:         start_date: date | None = None,
803:         end_date: date | None = None,
804:     ) -> list[CorporateAction]:
805:         """Get actions for a specific asset.
806: 
807:         Args:
808:             asset_id: Asset to get actions for
809:             start_date: Optional start date filter
810:             end_date: Optional end date filter
811: 
812:         Returns:
813:             List of corporate actions
814:         """
815:         actions = [action for action in self.actions.values() if action.asset_id == asset_id]
816: 
817:         if start_date:
818:             actions = [a for a in actions if a.ex_date >= start_date]
819: 
820:         if end_date:
821:             actions = [a for a in actions if a.ex_date <= end_date]
822: 
823:         return sorted(actions, key=lambda a: a.ex_date)
````

## File: src/qengine/execution/market_impact.py
````python
  1: """Market impact models for realistic price simulation.
  2: 
  3: Market impact differs from slippage in that it represents the actual change
  4: in market prices due to trading activity, affecting all subsequent orders.
  5: """
  6: 
  7: import math
  8: from abc import ABC, abstractmethod
  9: from dataclasses import dataclass
 10: from datetime import datetime, timedelta
 11: from typing import TYPE_CHECKING, Optional
 12: 
 13: if TYPE_CHECKING:
 14:     from qengine.core.types import AssetId, Price, Quantity
 15:     from qengine.execution.order import Order
 16: 
 17: 
 18: @dataclass
 19: class ImpactState:
 20:     """Tracks market impact state for an asset."""
 21: 
 22:     permanent_impact: float = 0.0  # Permanent price shift
 23:     temporary_impact: float = 0.0  # Temporary price displacement
 24:     last_update: datetime | None = None
 25:     volume_traded: float = 0.0  # Recent volume for impact calculation
 26: 
 27:     def get_total_impact(self) -> float:
 28:         """Get total current impact."""
 29:         return self.permanent_impact + self.temporary_impact
 30: 
 31:     def decay_temporary_impact(self, decay_rate: float, time_elapsed: float) -> None:
 32:         """Decay temporary impact over time."""
 33:         if time_elapsed > 0:
 34:             # Exponential decay
 35:             self.temporary_impact *= math.exp(-decay_rate * time_elapsed)
 36:             # Clean up near-zero values
 37:             if abs(self.temporary_impact) < 1e-10:
 38:                 self.temporary_impact = 0.0
 39: 
 40: 
 41: class MarketImpactModel(ABC):
 42:     """Abstract base class for market impact models."""
 43: 
 44:     def __init__(self):
 45:         """Initialize impact model."""
 46:         # Track impact state per asset
 47:         self.impact_states: dict[AssetId, ImpactState] = {}
 48: 
 49:     @abstractmethod
 50:     def calculate_impact(
 51:         self,
 52:         order: "Order",
 53:         fill_quantity: "Quantity",
 54:         market_price: "Price",
 55:         timestamp: datetime,
 56:     ) -> tuple[float, float]:
 57:         """Calculate permanent and temporary market impact.
 58: 
 59:         Args:
 60:             order: The order being filled
 61:             fill_quantity: Quantity being filled
 62:             market_price: Current market price
 63:             timestamp: Time of the fill
 64: 
 65:         Returns:
 66:             Tuple of (permanent_impact, temporary_impact) as price changes
 67:         """
 68: 
 69:     def update_market_state(
 70:         self,
 71:         asset_id: "AssetId",
 72:         permanent_impact: float,
 73:         temporary_impact: float,
 74:         timestamp: datetime,
 75:     ) -> None:
 76:         """Update the market state with new impact.
 77: 
 78:         Args:
 79:             asset_id: Asset identifier
 80:             permanent_impact: Permanent price change
 81:             temporary_impact: Temporary price displacement
 82:             timestamp: Time of the update
 83:         """
 84:         if asset_id not in self.impact_states:
 85:             self.impact_states[asset_id] = ImpactState()
 86: 
 87:         state = self.impact_states[asset_id]
 88: 
 89:         # Apply time decay to existing temporary impact
 90:         if state.last_update is not None:
 91:             time_elapsed = (timestamp - state.last_update).total_seconds()
 92:             self.apply_decay(asset_id, time_elapsed)
 93: 
 94:         # Add new impacts
 95:         state.permanent_impact += permanent_impact
 96:         state.temporary_impact += temporary_impact
 97:         state.last_update = timestamp
 98: 
 99:     def apply_decay(self, asset_id: "AssetId", time_elapsed: float) -> None:
100:         """Apply time decay to temporary impact.
101: 
102:         Args:
103:             asset_id: Asset identifier
104:             time_elapsed: Time elapsed in seconds
105:         """
106:         if asset_id in self.impact_states:
107:             # Default decay rate (can be overridden)
108:             decay_rate = getattr(self, "decay_rate", 0.1)
109:             self.impact_states[asset_id].decay_temporary_impact(decay_rate, time_elapsed)
110: 
111:     def get_current_impact(
112:         self,
113:         asset_id: "AssetId",
114:         timestamp: datetime | None = None,
115:     ) -> float:
116:         """Get current total market impact for an asset.
117: 
118:         Args:
119:             asset_id: Asset identifier
120:             timestamp: Current time for decay calculation
121: 
122:         Returns:
123:             Total price impact (permanent + temporary)
124:         """
125:         if asset_id not in self.impact_states:
126:             return 0.0
127: 
128:         state = self.impact_states[asset_id]
129: 
130:         # Apply decay if timestamp provided
131:         if timestamp and state.last_update:
132:             time_elapsed = (timestamp - state.last_update).total_seconds()
133:             self.apply_decay(asset_id, time_elapsed)
134: 
135:         return state.get_total_impact()
136: 
137:     def reset(self) -> None:
138:         """Reset all impact states."""
139:         self.impact_states.clear()
140: 
141: 
142: class NoMarketImpact(MarketImpactModel):
143:     """No market impact model for testing."""
144: 
145:     def calculate_impact(
146:         self,
147:         order: "Order",
148:         fill_quantity: "Quantity",
149:         market_price: "Price",
150:         timestamp: datetime,
151:     ) -> tuple[float, float]:
152:         """Calculate zero market impact."""
153:         return 0.0, 0.0
154: 
155: 
156: class LinearMarketImpact(MarketImpactModel):
157:     """Linear market impact model.
158: 
159:     Impact is proportional to order size relative to average daily volume.
160:     """
161: 
162:     def __init__(
163:         self,
164:         permanent_impact_factor: float = 0.1,
165:         temporary_impact_factor: float = 0.5,
166:         avg_daily_volume: float = 1_000_000,
167:         decay_rate: float = 0.1,
168:     ):
169:         """Initialize linear impact model.
170: 
171:         Args:
172:             permanent_impact_factor: Permanent impact per unit of volume fraction
173:             temporary_impact_factor: Temporary impact per unit of volume fraction
174:             avg_daily_volume: Average daily volume for normalization
175:             decay_rate: Decay rate for temporary impact (per second)
176:         """
177:         super().__init__()
178:         self.permanent_impact_factor = permanent_impact_factor
179:         self.temporary_impact_factor = temporary_impact_factor
180:         self.avg_daily_volume = avg_daily_volume
181:         self.decay_rate = decay_rate
182: 
183:     def calculate_impact(
184:         self,
185:         order: "Order",
186:         fill_quantity: "Quantity",
187:         market_price: "Price",
188:         timestamp: datetime,
189:     ) -> tuple[float, float]:
190:         """Calculate linear market impact."""
191:         # Volume fraction (what percentage of ADV is this trade?)
192:         volume_fraction = fill_quantity / self.avg_daily_volume
193: 
194:         # Linear impact proportional to volume fraction
195:         permanent_impact = market_price * self.permanent_impact_factor * volume_fraction
196:         temporary_impact = market_price * self.temporary_impact_factor * volume_fraction
197: 
198:         # Buy orders push price up, sell orders push price down
199:         from qengine.execution.order import OrderSide
200: 
201:         if order.side == OrderSide.SELL:
202:             permanent_impact = -permanent_impact
203:             temporary_impact = -temporary_impact
204: 
205:         return permanent_impact, temporary_impact
206: 
207: 
208: class AlmgrenChrissImpact(MarketImpactModel):
209:     """Almgren-Chriss market impact model.
210: 
211:     Sophisticated model with square-root permanent impact and linear temporary impact.
212:     Based on "Optimal Execution of Portfolio Transactions" (2001).
213:     """
214: 
215:     def __init__(
216:         self,
217:         permanent_impact_const: float = 0.01,
218:         temporary_impact_const: float = 0.1,
219:         daily_volatility: float = 0.02,
220:         avg_daily_volume: float = 1_000_000,
221:         decay_rate: float = 0.05,
222:     ):
223:         """Initialize Almgren-Chriss model.
224: 
225:         Args:
226:             permanent_impact_const: Permanent impact constant (gamma)
227:             temporary_impact_const: Temporary impact constant (eta)
228:             daily_volatility: Daily return volatility
229:             avg_daily_volume: Average daily volume
230:             decay_rate: Decay rate for temporary impact
231:         """
232:         super().__init__()
233:         self.permanent_impact_const = permanent_impact_const
234:         self.temporary_impact_const = temporary_impact_const
235:         self.daily_volatility = daily_volatility
236:         self.avg_daily_volume = avg_daily_volume
237:         self.decay_rate = decay_rate
238: 
239:     def calculate_impact(
240:         self,
241:         order: "Order",
242:         fill_quantity: "Quantity",
243:         market_price: "Price",
244:         timestamp: datetime,
245:     ) -> tuple[float, float]:
246:         """Calculate Almgren-Chriss market impact."""
247:         # Normalized volume (fraction of ADV)
248:         volume_fraction = fill_quantity / self.avg_daily_volume
249: 
250:         # Permanent impact: square-root of volume fraction
251:         # g(v) = gamma * sign(v) * |v|^0.5
252:         permanent_impact = (
253:             self.permanent_impact_const
254:             * self.daily_volatility
255:             * market_price
256:             * math.sqrt(volume_fraction)
257:         )
258: 
259:         # Temporary impact: linear in trading rate
260:         # h(v) = eta * v
261:         temporary_impact = (
262:             self.temporary_impact_const * self.daily_volatility * market_price * volume_fraction
263:         )
264: 
265:         # Adjust sign based on order side
266:         from qengine.execution.order import OrderSide
267: 
268:         if order.side == OrderSide.SELL:
269:             permanent_impact = -permanent_impact
270:             temporary_impact = -temporary_impact
271: 
272:         return permanent_impact, temporary_impact
273: 
274: 
275: class PropagatorImpact(MarketImpactModel):
276:     """Propagator model for market impact.
277: 
278:     Based on Bouchaud et al. model where impact propagates and decays
279:     according to a power law kernel.
280:     """
281: 
282:     def __init__(
283:         self,
284:         impact_coefficient: float = 0.1,
285:         propagator_exponent: float = 0.5,
286:         decay_exponent: float = 0.7,
287:         avg_daily_volume: float = 1_000_000,
288:     ):
289:         """Initialize propagator model.
290: 
291:         Args:
292:             impact_coefficient: Base impact coefficient
293:             propagator_exponent: Exponent for volume impact (typically 0.5)
294:             decay_exponent: Exponent for time decay (typically 0.5-0.7)
295:             avg_daily_volume: Average daily volume
296:         """
297:         super().__init__()
298:         self.impact_coefficient = impact_coefficient
299:         self.propagator_exponent = propagator_exponent
300:         self.decay_exponent = decay_exponent
301:         self.avg_daily_volume = avg_daily_volume
302: 
303:         # Track order history for propagation
304:         self.order_history: list[tuple[datetime, float, float]] = []
305: 
306:     def calculate_impact(
307:         self,
308:         order: "Order",
309:         fill_quantity: "Quantity",
310:         market_price: "Price",
311:         timestamp: datetime,
312:     ) -> tuple[float, float]:
313:         """Calculate propagator market impact."""
314:         # Normalized volume
315:         volume_fraction = fill_quantity / self.avg_daily_volume
316: 
317:         # Instantaneous impact: power law in volume
318:         instant_impact = (
319:             self.impact_coefficient * market_price * (volume_fraction**self.propagator_exponent)
320:         )
321: 
322:         # Calculate propagated impact from historical orders
323:         propagated_impact = 0.0
324:         cutoff_time = timestamp - timedelta(hours=1)  # Only consider recent history
325: 
326:         for hist_time, hist_volume, hist_price in self.order_history[-100:]:  # Limit history
327:             if hist_time < cutoff_time:
328:                 continue
329: 
330:             time_diff = (timestamp - hist_time).total_seconds()
331:             if time_diff > 0:
332:                 # Power law decay
333:                 decay_factor = (1 + time_diff) ** (-self.decay_exponent)
334:                 propagated_impact += (
335:                     self.impact_coefficient
336:                     * hist_price
337:                     * (abs(hist_volume) / self.avg_daily_volume) ** self.propagator_exponent
338:                     * decay_factor
339:                     * (1 if hist_volume > 0 else -1)
340:                 )
341: 
342:         # Store this order for future propagation
343:         from qengine.execution.order import OrderSide
344: 
345:         signed_volume = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
346:         self.order_history.append((timestamp, signed_volume, market_price))
347: 
348:         # Clean old history
349:         if len(self.order_history) > 1000:
350:             self.order_history = self.order_history[-500:]
351: 
352:         # Adjust sign
353:         if order.side == OrderSide.SELL:
354:             instant_impact = -instant_impact
355: 
356:         # Split into permanent and temporary
357:         # Propagator model typically has mostly temporary impact
358:         permanent_impact = instant_impact * 0.2
359:         temporary_impact = instant_impact * 0.8 + propagated_impact
360: 
361:         return permanent_impact, temporary_impact
362: 
363:     def reset(self) -> None:
364:         """Reset impact states and history."""
365:         super().reset()
366:         self.order_history.clear()
367: 
368: 
369: class IntraDayMomentum(MarketImpactModel):
370:     """Intraday momentum impact model.
371: 
372:     Models the tendency for large trades to create momentum that
373:     attracts further trading in the same direction.
374:     """
375: 
376:     def __init__(
377:         self,
378:         base_impact: float = 0.05,
379:         momentum_factor: float = 0.3,
380:         momentum_decay: float = 0.2,
381:         avg_daily_volume: float = 1_000_000,
382:     ):
383:         """Initialize momentum impact model.
384: 
385:         Args:
386:             base_impact: Base impact coefficient
387:             momentum_factor: How much momentum affects impact
388:             momentum_decay: Decay rate for momentum
389:             avg_daily_volume: Average daily volume
390:         """
391:         super().__init__()
392:         self.base_impact = base_impact
393:         self.momentum_factor = momentum_factor
394:         self.momentum_decay = momentum_decay
395:         self.avg_daily_volume = avg_daily_volume
396: 
397:         # Track momentum state per asset
398:         self.momentum_states: dict[AssetId, float] = {}
399: 
400:     def calculate_impact(
401:         self,
402:         order: "Order",
403:         fill_quantity: "Quantity",
404:         market_price: "Price",
405:         timestamp: datetime,
406:     ) -> tuple[float, float]:
407:         """Calculate momentum-based impact."""
408:         asset_id = order.asset_id
409:         volume_fraction = fill_quantity / self.avg_daily_volume
410: 
411:         # Get current momentum
412:         momentum = self.momentum_states.get(asset_id, 0.0)
413: 
414:         # Base impact
415:         base_impact_value = self.base_impact * market_price * volume_fraction
416: 
417:         # Momentum enhancement (same-direction trades have larger impact)
418:         from qengine.execution.order import OrderSide
419: 
420:         trade_direction = 1.0 if order.side == OrderSide.BUY else -1.0
421: 
422:         momentum_enhancement = 1.0 + self.momentum_factor * abs(momentum)
423:         if momentum * trade_direction > 0:  # Same direction as momentum
424:             impact = base_impact_value * momentum_enhancement
425:         else:  # Against momentum
426:             impact = base_impact_value / momentum_enhancement
427: 
428:         # Update momentum (exponential moving average)
429:         new_momentum = (
430:             momentum * (1 - self.momentum_decay)
431:             + trade_direction * volume_fraction * self.momentum_decay
432:         )
433:         self.momentum_states[asset_id] = new_momentum
434: 
435:         # Apply direction
436:         if order.side == OrderSide.SELL:
437:             impact = -impact
438: 
439:         # Split impact (momentum creates more temporary impact)
440:         permanent_impact = impact * 0.3
441:         temporary_impact = impact * 0.7
442: 
443:         return permanent_impact, temporary_impact
444: 
445:     def reset(self) -> None:
446:         """Reset all states."""
447:         super().reset()
448:         self.momentum_states.clear()
449: 
450: 
451: class ObizhaevWangImpact(MarketImpactModel):
452:     """Obizhaev-Wang market impact model.
453: 
454:     Models impact based on order book dynamics and trade informativeness.
455:     """
456: 
457:     def __init__(
458:         self,
459:         price_impact_const: float = 0.1,
460:         information_share: float = 0.3,
461:         book_depth: float = 100_000,
462:         resilience_rate: float = 0.5,
463:     ):
464:         """Initialize Obizhaev-Wang model.
465: 
466:         Args:
467:             price_impact_const: Price impact constant (lambda)
468:             information_share: Share of informed trading (alpha)
469:             book_depth: Typical order book depth
470:             resilience_rate: Rate of order book resilience
471:         """
472:         super().__init__()
473:         self.price_impact_const = price_impact_const
474:         self.information_share = information_share
475:         self.book_depth = book_depth
476:         self.resilience_rate = resilience_rate
477:         self.decay_rate = resilience_rate  # For base class decay
478: 
479:     def calculate_impact(
480:         self,
481:         order: "Order",
482:         fill_quantity: "Quantity",
483:         market_price: "Price",
484:         timestamp: datetime,
485:     ) -> tuple[float, float]:
486:         """Calculate Obizhaev-Wang impact."""
487:         # Normalized order size relative to book depth
488:         size_ratio = fill_quantity / self.book_depth
489: 
490:         # Information-based permanent impact
491:         permanent_impact = (
492:             self.information_share * self.price_impact_const * market_price * size_ratio
493:         )
494: 
495:         # Mechanical temporary impact from eating through book
496:         temporary_impact = (
497:             (1 - self.information_share) * self.price_impact_const * market_price * size_ratio
498:         )
499: 
500:         # Adjust for order side
501:         from qengine.execution.order import OrderSide
502: 
503:         if order.side == OrderSide.SELL:
504:             permanent_impact = -permanent_impact
505:             temporary_impact = -temporary_impact
506: 
507:         return permanent_impact, temporary_impact
````

## File: src/qengine/execution/order.py
````python
  1: """Order management for QEngine."""
  2: 
  3: import uuid
  4: from dataclasses import dataclass, field
  5: from datetime import datetime
  6: from enum import Enum
  7: from typing import Any
  8: 
  9: from qengine.core.types import (
 10:     AssetId,
 11:     OrderId,
 12:     OrderSide,
 13:     OrderStatus,
 14:     OrderType,
 15:     Price,
 16:     Quantity,
 17:     TimeInForce,
 18: )
 19: 
 20: 
 21: class OrderState(Enum):
 22:     """Order lifecycle states."""
 23: 
 24:     PENDING = "pending"  # Created but not yet submitted
 25:     SUBMITTED = "submitted"  # Sent to broker
 26:     ACKNOWLEDGED = "acknowledged"  # Broker confirmed receipt
 27:     PARTIALLY_FILLED = "partially_filled"  # Some quantity filled
 28:     FILLED = "filled"  # Completely filled
 29:     CANCELLED = "cancelled"  # Cancelled by user
 30:     REJECTED = "rejected"  # Rejected by broker
 31:     EXPIRED = "expired"  # Expired due to time constraint
 32: 
 33: 
 34: @dataclass
 35: class Order:
 36:     """Represents a trading order."""
 37: 
 38:     # Core identifiers
 39:     order_id: OrderId = field(default_factory=lambda: str(uuid.uuid4()))
 40:     asset_id: AssetId = ""
 41: 
 42:     # Order specifications
 43:     order_type: OrderType = OrderType.MARKET
 44:     side: OrderSide = OrderSide.BUY
 45:     quantity: Quantity = 0.0
 46: 
 47:     # Price constraints
 48:     limit_price: Price | None = None
 49:     stop_price: Price | None = None
 50: 
 51:     # Advanced order type parameters
 52:     trail_amount: Price | None = None  # For trailing stops (absolute)
 53:     trail_percent: float | None = None  # For trailing stops (percentage)
 54:     trailing_stop_price: Price | None = None  # Current trailing stop level
 55: 
 56:     # Bracket order parameters
 57:     profit_target: Price | None = None  # Take profit level
 58:     stop_loss: Price | None = None  # Stop loss level
 59: 
 60:     # Time constraints
 61:     time_in_force: TimeInForce = TimeInForce.DAY
 62:     expire_time: datetime | None = None
 63: 
 64:     # State tracking
 65:     state: OrderState = OrderState.PENDING
 66:     status: OrderStatus = OrderStatus.CREATED
 67: 
 68:     # Timestamps
 69:     created_time: datetime = field(default_factory=datetime.now)
 70:     submitted_time: datetime | None = None
 71:     acknowledged_time: datetime | None = None
 72:     filled_time: datetime | None = None
 73:     cancelled_time: datetime | None = None
 74: 
 75:     # Fill information
 76:     filled_quantity: Quantity = 0.0
 77:     average_fill_price: Price | None = None
 78:     fill_count: int = 0
 79: 
 80:     # Costs
 81:     commission: float = 0.0
 82:     slippage: float = 0.0
 83: 
 84:     # Relationships
 85:     parent_order_id: OrderId | None = None
 86:     child_order_ids: list[OrderId] = field(default_factory=list)
 87: 
 88:     # Metadata
 89:     metadata: dict[str, Any] = field(default_factory=dict)
 90: 
 91:     def __post_init__(self):
 92:         """Validate order on creation."""
 93:         if self.order_type == OrderType.LIMIT and self.limit_price is None:
 94:             raise ValueError("Limit orders must have a limit price")
 95:         if self.order_type == OrderType.STOP and self.stop_price is None:
 96:             raise ValueError("Stop orders must have a stop price")
 97:         if self.order_type == OrderType.STOP_LIMIT:
 98:             if self.stop_price is None or self.limit_price is None:
 99:                 raise ValueError("Stop-limit orders must have both stop and limit prices")
100:         if self.order_type == OrderType.TRAILING_STOP:
101:             if self.trail_amount is None and self.trail_percent is None:
102:                 raise ValueError("Trailing stop orders must have trail_amount or trail_percent")
103:         if self.order_type == OrderType.BRACKET:
104:             if self.profit_target is None or self.stop_loss is None:
105:                 raise ValueError("Bracket orders must have both profit_target and stop_loss")
106:         if self.quantity <= 0:
107:             raise ValueError("Order quantity must be positive")
108: 
109:     @property
110:     def is_buy(self) -> bool:
111:         """Check if this is a buy order."""
112:         return self.side == OrderSide.BUY
113: 
114:     @property
115:     def is_sell(self) -> bool:
116:         """Check if this is a sell order."""
117:         return self.side == OrderSide.SELL
118: 
119:     @property
120:     def is_filled(self) -> bool:
121:         """Check if order is completely filled."""
122:         return self.state == OrderState.FILLED
123: 
124:     @property
125:     def is_partially_filled(self) -> bool:
126:         """Check if order is partially filled."""
127:         return self.state == OrderState.PARTIALLY_FILLED
128: 
129:     @property
130:     def is_active(self) -> bool:
131:         """Check if order is still active."""
132:         return self.state in [
133:             OrderState.PENDING,
134:             OrderState.SUBMITTED,
135:             OrderState.ACKNOWLEDGED,
136:             OrderState.PARTIALLY_FILLED,
137:         ]
138: 
139:     @property
140:     def is_terminal(self) -> bool:
141:         """Check if order is in a terminal state."""
142:         return self.state in [
143:             OrderState.FILLED,
144:             OrderState.CANCELLED,
145:             OrderState.REJECTED,
146:             OrderState.EXPIRED,
147:         ]
148: 
149:     @property
150:     def remaining_quantity(self) -> Quantity:
151:         """Get remaining quantity to fill."""
152:         return self.quantity - self.filled_quantity
153: 
154:     @property
155:     def fill_ratio(self) -> float:
156:         """Get the ratio of filled quantity to total."""
157:         if self.quantity == 0:
158:             return 0.0
159:         return self.filled_quantity / self.quantity
160: 
161:     def can_fill(self, price: Price) -> bool:
162:         """
163:         Check if order can be filled at given price.
164: 
165:         Args:
166:             price: Current market price
167: 
168:         Returns:
169:             True if order can be filled
170:         """
171:         if not self.is_active:
172:             return False
173: 
174:         if self.order_type == OrderType.MARKET:
175:             return True
176: 
177:         if self.order_type == OrderType.LIMIT:
178:             if self.limit_price is None:
179:                 return False
180:             if self.is_buy:
181:                 return price <= self.limit_price
182:             return price >= self.limit_price
183: 
184:         if self.order_type == OrderType.STOP:
185:             if self.stop_price is None:
186:                 return False
187:             # Stop orders trigger when price crosses the stop level
188:             if self.is_buy:
189:                 return price >= self.stop_price
190:             return price <= self.stop_price
191: 
192:         if self.order_type == OrderType.STOP_LIMIT:
193:             # For simplicity, assume stop has been triggered if we get here
194:             # The broker will handle the trigger logic
195:             if self.limit_price is None:
196:                 return False
197:             if self.is_buy:
198:                 return price <= self.limit_price
199:             return price >= self.limit_price
200: 
201:         if self.order_type == OrderType.TRAILING_STOP:
202:             if self.trailing_stop_price is None:
203:                 return False
204:             # Trailing stop triggers when price crosses the trailing level
205:             if self.is_buy:
206:                 return price >= self.trailing_stop_price
207:             return price <= self.trailing_stop_price
208: 
209:         if self.order_type == OrderType.BRACKET:
210:             # Bracket orders fill based on their entry criteria (limit_price if set)
211:             if self.limit_price is not None:
212:                 # Act like a limit order for entry
213:                 if self.is_buy:
214:                     return price <= self.limit_price
215:                 return price >= self.limit_price
216:             # Act like a market order for entry
217:             return True
218: 
219:         # OCO and other special orders
220:         return False
221: 
222:     def update_fill(
223:         self,
224:         fill_quantity: Quantity,
225:         fill_price: Price,
226:         commission: float = 0.0,
227:         timestamp: datetime | None = None,
228:     ) -> None:
229:         """
230:         Update order with fill information.
231: 
232:         Args:
233:             fill_quantity: Quantity filled
234:             fill_price: Price of fill
235:             commission: Commission charged
236:             timestamp: Time of fill
237:         """
238:         if fill_quantity <= 0:
239:             raise ValueError("Fill quantity must be positive")
240: 
241:         if fill_quantity > self.remaining_quantity:
242:             raise ValueError(
243:                 f"Fill quantity {fill_quantity} exceeds remaining {self.remaining_quantity}",
244:             )
245: 
246:         # Update fill tracking
247:         if self.average_fill_price is None:
248:             self.average_fill_price = fill_price
249:         else:
250:             # Calculate weighted average
251:             total_value = (
252:                 self.filled_quantity * self.average_fill_price + fill_quantity * fill_price
253:             )
254:             self.average_fill_price = total_value / (self.filled_quantity + fill_quantity)
255: 
256:         self.filled_quantity += fill_quantity
257:         self.fill_count += 1
258:         self.commission += commission
259: 
260:         # Update state
261:         if self.filled_quantity >= self.quantity:
262:             self.state = OrderState.FILLED
263:             self.status = OrderStatus.FILLED
264:             self.filled_time = timestamp or datetime.now()
265:         else:
266:             self.state = OrderState.PARTIALLY_FILLED
267:             self.status = OrderStatus.PARTIALLY_FILLED
268: 
269:     def cancel(self, timestamp: datetime | None = None) -> None:
270:         """Cancel the order."""
271:         if self.is_terminal:
272:             raise ValueError(f"Cannot cancel order in state {self.state}")
273: 
274:         self.state = OrderState.CANCELLED
275:         self.status = OrderStatus.CANCELED
276:         self.cancelled_time = timestamp or datetime.now()
277: 
278:     def reject(self, reason: str = "", timestamp: datetime | None = None) -> None:
279:         """Reject the order."""
280:         self.state = OrderState.REJECTED
281:         self.status = OrderStatus.REJECTED
282:         self.metadata["rejection_reason"] = reason
283:         self.cancelled_time = timestamp or datetime.now()
284: 
285:     def update_trailing_stop(self, current_price: Price) -> bool:
286:         """
287:         Update trailing stop price based on current market price.
288: 
289:         Args:
290:             current_price: Current market price
291: 
292:         Returns:
293:             True if trailing stop was updated, False otherwise
294:         """
295:         if self.order_type != OrderType.TRAILING_STOP:
296:             return False
297: 
298:         # Initialize trailing stop price if not set
299:         if self.trailing_stop_price is None:
300:             if self.trail_amount is not None:
301:                 if self.is_buy:
302:                     self.trailing_stop_price = current_price + self.trail_amount
303:                 else:
304:                     self.trailing_stop_price = current_price - self.trail_amount
305:             elif self.trail_percent is not None:
306:                 trail_amount = current_price * (self.trail_percent / 100.0)
307:                 if self.is_buy:
308:                     self.trailing_stop_price = current_price + trail_amount
309:                 else:
310:                     self.trailing_stop_price = current_price - trail_amount
311:             return True
312: 
313:         # Update trailing stop if price moves favorably
314:         updated = False
315: 
316:         if self.trail_amount is not None:
317:             # Absolute trailing amount
318:             if self.is_buy:
319:                 # For buy stops, trail up when price falls
320:                 new_stop = current_price + self.trail_amount
321:                 if new_stop < self.trailing_stop_price:
322:                     self.trailing_stop_price = new_stop
323:                     updated = True
324:             else:
325:                 # For sell stops, trail down when price rises
326:                 new_stop = current_price - self.trail_amount
327:                 if new_stop > self.trailing_stop_price:
328:                     self.trailing_stop_price = new_stop
329:                     updated = True
330: 
331:         elif self.trail_percent is not None:
332:             # Percentage trailing amount
333:             trail_amount = current_price * (self.trail_percent / 100.0)
334:             if self.is_buy:
335:                 new_stop = current_price + trail_amount
336:                 if new_stop < self.trailing_stop_price:
337:                     self.trailing_stop_price = new_stop
338:                     updated = True
339:             else:
340:                 new_stop = current_price - trail_amount
341:                 if new_stop > self.trailing_stop_price:
342:                     self.trailing_stop_price = new_stop
343:                     updated = True
344: 
345:         return updated
346: 
347:     def __repr__(self) -> str:
348:         return (
349:             f"Order(id={self.order_id[:8]}, {self.side.value} {self.quantity} "
350:             f"{self.asset_id} @ {self.order_type.value}, state={self.state.value})"
351:         )
````

## File: src/qengine/execution/slippage.py
````python
  1: """Slippage models for QEngine."""
  2: 
  3: from abc import ABC, abstractmethod
  4: from typing import Optional
  5: 
  6: from qengine.core.types import Price, Quantity
  7: from qengine.execution.order import Order
  8: 
  9: 
 10: class SlippageModel(ABC):
 11:     """Abstract base class for slippage models.
 12: 
 13:     Slippage models determine the actual fill price based on order characteristics
 14:     and market conditions.
 15:     """
 16: 
 17:     @abstractmethod
 18:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
 19:         """Calculate the fill price with slippage.
 20: 
 21:         Args:
 22:             order: The order being filled
 23:             market_price: Current market price
 24: 
 25:         Returns:
 26:             The fill price including slippage
 27:         """
 28: 
 29:     @abstractmethod
 30:     def calculate_slippage_cost(
 31:         self,
 32:         order: Order,
 33:         fill_quantity: Quantity,
 34:         market_price: Price,
 35:         fill_price: Price,
 36:     ) -> float:
 37:         """Calculate the slippage cost in currency terms.
 38: 
 39:         Args:
 40:             order: The order being filled
 41:             fill_quantity: Quantity being filled
 42:             market_price: Current market price
 43:             fill_price: Actual fill price
 44: 
 45:         Returns:
 46:             Slippage cost in currency terms
 47:         """
 48: 
 49: 
 50: class NoSlippage(SlippageModel):
 51:     """No slippage - all orders fill at market price.
 52: 
 53:     Primarily used for testing or ideal conditions.
 54:     """
 55: 
 56:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
 57:         """Fill at market price."""
 58:         return market_price
 59: 
 60:     def calculate_slippage_cost(
 61:         self,
 62:         order: Order,
 63:         fill_quantity: Quantity,
 64:         market_price: Price,
 65:         fill_price: Price,
 66:     ) -> float:
 67:         """No slippage cost."""
 68:         return 0.0
 69: 
 70: 
 71: class FixedSlippage(SlippageModel):
 72:     """Fixed spread slippage model.
 73: 
 74:     Assumes a fixed spread for all orders.
 75:     Buy orders fill at ask (market + spread/2)
 76:     Sell orders fill at bid (market - spread/2)
 77: 
 78:     Args:
 79:         spread: Fixed spread amount (default 0.01)
 80:     """
 81: 
 82:     def __init__(self, spread: float = 0.01):
 83:         """Initialize with fixed spread."""
 84:         if spread < 0:
 85:             raise ValueError("Spread must be non-negative")
 86:         self.spread = spread
 87: 
 88:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
 89:         """Calculate fill price with fixed spread."""
 90:         half_spread = self.spread / 2
 91: 
 92:         if order.is_buy:
 93:             # Buy at ask (worse price)
 94:             return market_price + half_spread
 95:         # Sell at bid (worse price)
 96:         return market_price - half_spread
 97: 
 98:     def calculate_slippage_cost(
 99:         self,
100:         order: Order,
101:         fill_quantity: Quantity,
102:         market_price: Price,
103:         fill_price: Price,
104:     ) -> float:
105:         """Calculate slippage cost from spread."""
106:         # Cost is the absolute difference times quantity
107:         return abs(fill_price - market_price) * fill_quantity
108: 
109: 
110: class PercentageSlippage(SlippageModel):
111:     """Percentage-based slippage model.
112: 
113:     Slippage is a percentage of the market price.
114: 
115:     Args:
116:         slippage_pct: Slippage percentage (default 0.1%)
117:         min_slippage: Minimum slippage amount (default 0.001)
118:     """
119: 
120:     def __init__(self, slippage_pct: float = 0.001, min_slippage: float = 0.001):
121:         """Initialize with percentage parameters."""
122:         if slippage_pct < 0:
123:             raise ValueError("Slippage percentage must be non-negative")
124:         if min_slippage < 0:
125:             raise ValueError("Minimum slippage must be non-negative")
126: 
127:         self.slippage_pct = slippage_pct
128:         self.min_slippage = min_slippage
129: 
130:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
131:         """Calculate fill price with percentage slippage."""
132:         # Calculate slippage amount
133:         slippage_amount = max(market_price * self.slippage_pct, self.min_slippage)
134: 
135:         if order.is_buy:
136:             # Buy at higher price
137:             return market_price + slippage_amount
138:         # Sell at lower price
139:         return market_price - slippage_amount
140: 
141:     def calculate_slippage_cost(
142:         self,
143:         order: Order,
144:         fill_quantity: Quantity,
145:         market_price: Price,
146:         fill_price: Price,
147:     ) -> float:
148:         """Calculate slippage cost."""
149:         return abs(fill_price - market_price) * fill_quantity
150: 
151: 
152: class LinearImpactSlippage(SlippageModel):
153:     """Linear market impact slippage model.
154: 
155:     Slippage increases linearly with order size.
156: 
157:     Args:
158:         base_slippage: Base slippage for minimal orders (default 0.0001)
159:         impact_coefficient: Impact per unit of order size (default 0.00001)
160:     """
161: 
162:     def __init__(
163:         self,
164:         base_slippage: float = 0.0001,
165:         impact_coefficient: float = 0.00001,
166:     ):
167:         """Initialize with impact parameters."""
168:         if base_slippage < 0:
169:             raise ValueError("Base slippage must be non-negative")
170:         if impact_coefficient < 0:
171:             raise ValueError("Impact coefficient must be non-negative")
172: 
173:         self.base_slippage = base_slippage
174:         self.impact_coefficient = impact_coefficient
175: 
176:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
177:         """Calculate fill price with linear impact."""
178:         # Linear impact based on order size
179:         impact = self.base_slippage + self.impact_coefficient * order.quantity
180:         slippage_amount = market_price * impact
181: 
182:         if order.is_buy:
183:             return market_price + slippage_amount
184:         return market_price - slippage_amount
185: 
186:     def calculate_slippage_cost(
187:         self,
188:         order: Order,
189:         fill_quantity: Quantity,
190:         market_price: Price,
191:         fill_price: Price,
192:     ) -> float:
193:         """Calculate slippage cost."""
194:         return abs(fill_price - market_price) * fill_quantity
195: 
196: 
197: class SquareRootImpactSlippage(SlippageModel):
198:     """Square root market impact model (Almgren-Chriss style).
199: 
200:     Slippage increases with the square root of order size, modeling
201:     non-linear market impact for large orders.
202: 
203:     Args:
204:         temporary_impact: Temporary impact coefficient (default 0.1)
205:         permanent_impact: Permanent impact coefficient (default 0.05)
206:     """
207: 
208:     def __init__(
209:         self,
210:         temporary_impact: float = 0.1,
211:         permanent_impact: float = 0.05,
212:     ):
213:         """Initialize with impact parameters."""
214:         if temporary_impact < 0:
215:             raise ValueError("Temporary impact must be non-negative")
216:         if permanent_impact < 0:
217:             raise ValueError("Permanent impact must be non-negative")
218: 
219:         self.temporary_impact = temporary_impact
220:         self.permanent_impact = permanent_impact
221: 
222:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
223:         """Calculate fill price with square root impact."""
224:         import math
225: 
226:         # Square root impact model
227:         order_size_impact = math.sqrt(order.quantity / 1000.0)  # Normalize by 1000 shares
228: 
229:         # Combine temporary and permanent impact
230:         total_impact = (
231:             self.temporary_impact * order_size_impact
232:             + self.permanent_impact * order_size_impact / 2
233:         )
234: 
235:         # Convert to price impact
236:         slippage_amount = market_price * total_impact * 0.01  # Convert to percentage
237: 
238:         if order.is_buy:
239:             return market_price + slippage_amount
240:         return market_price - slippage_amount
241: 
242:     def calculate_slippage_cost(
243:         self,
244:         order: Order,
245:         fill_quantity: Quantity,
246:         market_price: Price,
247:         fill_price: Price,
248:     ) -> float:
249:         """Calculate slippage cost."""
250:         return abs(fill_price - market_price) * fill_quantity
251: 
252: 
253: class VolumeShareSlippage(SlippageModel):
254:     """Volume-based slippage model.
255: 
256:     Slippage is based on the percentage of daily volume being traded.
257:     Larger orders relative to volume have more impact.
258: 
259:     Args:
260:         volume_limit: Maximum percentage of volume per bar (default 0.025 = 2.5%)
261:         price_impact: Price impact coefficient (default 0.1)
262:     """
263: 
264:     def __init__(
265:         self,
266:         volume_limit: float = 0.025,
267:         price_impact: float = 0.1,
268:     ):
269:         """Initialize with volume parameters."""
270:         if not 0 < volume_limit <= 1:
271:             raise ValueError("Volume limit must be between 0 and 1")
272:         if price_impact < 0:
273:             raise ValueError("Price impact must be non-negative")
274: 
275:         self.volume_limit = volume_limit
276:         self.price_impact = price_impact
277:         self._daily_volume: float | None = None
278: 
279:     def set_daily_volume(self, volume: float) -> None:
280:         """Set the daily volume for impact calculation.
281: 
282:         Args:
283:             volume: Daily volume
284:         """
285:         self._daily_volume = volume
286: 
287:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
288:         """Calculate fill price based on volume impact."""
289:         if self._daily_volume is None or self._daily_volume == 0:
290:             # No volume data, use minimal slippage
291:             slippage_amount = market_price * 0.0001
292:         else:
293:             # Calculate volume share
294:             volume_share = min(order.quantity / self._daily_volume, self.volume_limit)
295: 
296:             # Quadratic impact model (like Zipline)
297:             impact = volume_share**2 * self.price_impact
298:             slippage_amount = market_price * impact
299: 
300:         if order.is_buy:
301:             return market_price + slippage_amount
302:         return market_price - slippage_amount
303: 
304:     def calculate_slippage_cost(
305:         self,
306:         order: Order,
307:         fill_quantity: Quantity,
308:         market_price: Price,
309:         fill_price: Price,
310:     ) -> float:
311:         """Calculate slippage cost."""
312:         return abs(fill_price - market_price) * fill_quantity
313: 
314: 
315: class AssetClassSlippage(SlippageModel):
316:     """Asset class specific slippage model.
317: 
318:     Different slippage rates for different asset classes.
319: 
320:     Args:
321:         equity_slippage: Slippage for equities (default 0.01%)
322:         future_slippage: Slippage for futures (default 0.02%)
323:         option_slippage: Slippage for options (default 0.05%)
324:         fx_slippage: Slippage for forex (default 0.005%)
325:         crypto_slippage: Slippage for crypto (default 0.1%)
326:     """
327: 
328:     def __init__(
329:         self,
330:         equity_slippage: float = 0.0001,
331:         future_slippage: float = 0.0002,
332:         option_slippage: float = 0.0005,
333:         fx_slippage: float = 0.00005,
334:         crypto_slippage: float = 0.001,
335:     ):
336:         """Initialize with asset class specific rates."""
337:         self.slippage_rates = {
338:             "equity": equity_slippage,
339:             "future": future_slippage,
340:             "option": option_slippage,
341:             "forex": fx_slippage,
342:             "fx": fx_slippage,  # Alias
343:             "crypto": crypto_slippage,
344:         }
345:         self.default_slippage = equity_slippage
346: 
347:     def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
348:         """Calculate fill price based on asset class."""
349:         # Get asset class from order metadata or default
350:         asset_class = order.metadata.get("asset_class", "equity")
351:         slippage_rate = self.slippage_rates.get(asset_class, self.default_slippage)
352: 
353:         slippage_amount = market_price * slippage_rate
354: 
355:         if order.is_buy:
356:             return market_price + slippage_amount
357:         return market_price - slippage_amount
358: 
359:     def calculate_slippage_cost(
360:         self,
361:         order: Order,
362:         fill_quantity: Quantity,
363:         market_price: Price,
364:         fill_price: Price,
365:     ) -> float:
366:         """Calculate slippage cost."""
367:         return abs(fill_price - market_price) * fill_quantity
````

## File: src/qengine/portfolio/__init__.py
````python
 1: """Portfolio management for QEngine."""
 2: 
 3: from qengine.portfolio.accounting import PortfolioAccounting
 4: from qengine.portfolio.margin import MarginAccount, MarginRequirement
 5: from qengine.portfolio.portfolio import Portfolio, PortfolioState, Position
 6: from qengine.portfolio.simple import SimplePortfolio
 7: 
 8: __all__ = [
 9:     "MarginAccount",
10:     "MarginRequirement",
11:     "Portfolio",
12:     "PortfolioAccounting",
13:     "PortfolioState",
14:     "Position",
15:     "SimplePortfolio",
16: ]
````

## File: src/qengine/portfolio/accounting.py
````python
  1: """Portfolio accounting and P&L tracking for QEngine."""
  2: 
  3: from datetime import datetime
  4: from typing import Any
  5: 
  6: import polars as pl
  7: 
  8: from qengine.core.event import FillEvent
  9: from qengine.core.types import AssetId, Cash
 10: from qengine.portfolio.portfolio import Portfolio
 11: 
 12: 
 13: class PortfolioAccounting:
 14:     """
 15:     Handles portfolio accounting, P&L calculation, and performance tracking.
 16: 
 17:     This class integrates with the broker to track trades and calculate
 18:     real-time portfolio metrics including realized/unrealized P&L,
 19:     returns, and risk metrics.
 20:     """
 21: 
 22:     def __init__(self, initial_cash: Cash = 100000.0, track_history: bool = True):
 23:         """
 24:         Initialize portfolio accounting.
 25: 
 26:         Args:
 27:             initial_cash: Starting cash balance
 28:             track_history: Whether to keep detailed history
 29:         """
 30:         self.portfolio = Portfolio(initial_cash)
 31:         self.track_history = track_history
 32: 
 33:         # Track all fills
 34:         self.fills: list[FillEvent] = []
 35: 
 36:         # Performance tracking
 37:         self.high_water_mark = initial_cash
 38:         self.max_drawdown = 0.0
 39:         self.daily_returns: list[float] = []
 40:         self.timestamps: list[datetime] = []
 41:         self.equity_curve: list[float] = []
 42: 
 43:         # Risk tracking
 44:         self.max_leverage = 0.0
 45:         self.max_concentration = 0.0
 46: 
 47:         if track_history:
 48:             self.timestamps.append(datetime.now())
 49:             self.equity_curve.append(initial_cash)
 50: 
 51:     def process_fill(self, fill_event: FillEvent) -> None:
 52:         """
 53:         Process a fill event and update portfolio.
 54: 
 55:         Args:
 56:             fill_event: Fill event from broker
 57:         """
 58:         # Record fill
 59:         self.fills.append(fill_event)
 60: 
 61:         # Determine quantity change (positive for buy, negative for sell)
 62:         quantity_change = fill_event.fill_quantity
 63:         if fill_event.side.value == "sell":
 64:             quantity_change = -quantity_change
 65: 
 66:         # Update portfolio position
 67:         self.portfolio.update_position(
 68:             asset_id=fill_event.asset_id,
 69:             quantity_change=quantity_change,
 70:             price=fill_event.fill_price,
 71:             commission=fill_event.commission,
 72:             slippage=fill_event.slippage,
 73:         )
 74: 
 75:         # Update performance metrics
 76:         self._update_metrics(fill_event.timestamp)
 77: 
 78:     def update_prices(self, prices: dict[AssetId, float], timestamp: datetime) -> None:
 79:         """
 80:         Update portfolio with new market prices.
 81: 
 82:         Args:
 83:             prices: Dictionary of asset prices
 84:             timestamp: Current timestamp
 85:         """
 86:         self.portfolio.update_prices(prices)
 87:         self._update_metrics(timestamp)
 88: 
 89:     def _update_metrics(self, timestamp: datetime) -> None:
 90:         """Update performance and risk metrics."""
 91:         current_equity = self.portfolio.equity
 92: 
 93:         # Update high water mark and drawdown
 94:         if current_equity > self.high_water_mark:
 95:             self.high_water_mark = current_equity
 96: 
 97:         if self.high_water_mark > 0:
 98:             current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
 99:             self.max_drawdown = max(self.max_drawdown, current_drawdown)
100: 
101:         # Track equity curve
102:         if self.track_history:
103:             self.timestamps.append(timestamp)
104:             self.equity_curve.append(current_equity)
105: 
106:             # Calculate daily return if we have previous data
107:             if len(self.equity_curve) > 1:
108:                 prev_equity = self.equity_curve[-2]
109:                 if prev_equity > 0:
110:                     daily_return = (current_equity - prev_equity) / prev_equity
111:                     self.daily_returns.append(daily_return)
112: 
113:         # Update risk metrics from current state
114:         state = self.portfolio.get_current_state(timestamp)
115:         self.max_leverage = max(self.max_leverage, state.leverage)
116:         self.max_concentration = max(self.max_concentration, state.concentration)
117: 
118:         # Save state if tracking history
119:         if self.track_history:
120:             self.portfolio.save_state(timestamp)
121: 
122:     def get_performance_metrics(self) -> dict[str, Any]:
123:         """Get comprehensive performance metrics."""
124:         metrics = {
125:             "total_return": self.portfolio.returns,
126:             "total_pnl": self.portfolio.total_realized_pnl + self.portfolio.unrealized_pnl,
127:             "realized_pnl": self.portfolio.total_realized_pnl,
128:             "unrealized_pnl": self.portfolio.unrealized_pnl,
129:             "max_drawdown": self.max_drawdown,
130:             "current_equity": self.portfolio.equity,
131:             "current_cash": self.portfolio.cash,
132:             "total_commission": self.portfolio.total_commission,
133:             "total_slippage": self.portfolio.total_slippage,
134:             "num_trades": len(self.fills),
135:             "max_leverage": self.max_leverage,
136:             "max_concentration": self.max_concentration,
137:         }
138: 
139:         # Add Sharpe ratio if we have enough data
140:         if len(self.daily_returns) > 1:
141:             import numpy as np
142: 
143:             returns = np.array(self.daily_returns)
144:             if returns.std() > 0:
145:                 # Annualized Sharpe (assuming 252 trading days)
146:                 metrics["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(252)
147:             else:
148:                 metrics["sharpe_ratio"] = 0.0
149: 
150:         return metrics
151: 
152:     def get_trades_df(self) -> pl.DataFrame | None:
153:         """Get all trades as a Polars DataFrame."""
154:         if not self.fills:
155:             return None
156: 
157:         trades_data = []
158:         for fill in self.fills:
159:             trades_data.append(
160:                 {
161:                     "timestamp": fill.timestamp,
162:                     "order_id": fill.order_id,
163:                     "trade_id": fill.trade_id,
164:                     "asset_id": fill.asset_id,
165:                     "side": fill.side.value,
166:                     "quantity": fill.fill_quantity,
167:                     "price": fill.fill_price,
168:                     "commission": fill.commission,
169:                     "slippage": fill.slippage,
170:                     "total_cost": fill.total_cost,
171:                 },
172:             )
173: 
174:         return pl.DataFrame(trades_data)
175: 
176:     def get_equity_curve_df(self) -> pl.DataFrame | None:
177:         """Get equity curve as a Polars DataFrame."""
178:         if not self.timestamps:
179:             return None
180: 
181:         return pl.DataFrame(
182:             {
183:                 "timestamp": self.timestamps,
184:                 "equity": self.equity_curve,
185:                 "returns": [0.0, *self.daily_returns],  # Pad with 0 for first day
186:             },
187:         )
188: 
189:     def get_positions_df(self) -> pl.DataFrame | None:
190:         """Get current positions as a Polars DataFrame."""
191:         if not self.portfolio.positions:
192:             return None
193: 
194:         positions_data = []
195:         for position in self.portfolio.positions.values():
196:             positions_data.append(
197:                 {
198:                     "asset_id": position.asset_id,
199:                     "quantity": position.quantity,
200:                     "cost_basis": position.cost_basis,
201:                     "last_price": position.last_price,
202:                     "market_value": position.market_value,
203:                     "unrealized_pnl": position.unrealized_pnl,
204:                     "realized_pnl": position.realized_pnl,
205:                     "total_pnl": position.total_pnl,
206:                 },
207:             )
208: 
209:         return pl.DataFrame(positions_data)
210: 
211:     def get_summary(self) -> dict[str, Any]:
212:         """Get portfolio summary."""
213:         summary = self.portfolio.get_position_summary()
214:         summary.update(self.get_performance_metrics())
215:         return summary
216: 
217:     def reset(self) -> None:
218:         """Reset portfolio to initial state."""
219:         initial_cash = self.portfolio.initial_cash
220:         self.portfolio = Portfolio(initial_cash)
221:         self.fills.clear()
222:         self.high_water_mark = initial_cash
223:         self.max_drawdown = 0.0
224:         self.daily_returns.clear()
225:         self.timestamps.clear()
226:         self.equity_curve.clear()
227:         self.max_leverage = 0.0
228:         self.max_concentration = 0.0
229: 
230:         if self.track_history:
231:             self.timestamps.append(datetime.now())
232:             self.equity_curve.append(initial_cash)
````

## File: src/qengine/portfolio/margin.py
````python
  1: """Margin management for derivatives and leveraged trading."""
  2: 
  3: from dataclasses import dataclass, field
  4: from datetime import datetime
  5: 
  6: from qengine.core.assets import AssetRegistry, AssetSpec
  7: from qengine.core.types import AssetId, Cash, Price, Quantity
  8: 
  9: 
 10: @dataclass
 11: class MarginRequirement:
 12:     """Margin requirements for a position."""
 13: 
 14:     asset_id: AssetId
 15:     initial_margin: Cash
 16:     maintenance_margin: Cash
 17:     current_margin: Cash
 18:     excess_margin: Cash
 19:     margin_call: bool = False
 20:     liquidation_price: Price | None = None
 21: 
 22: 
 23: @dataclass
 24: class MarginAccount:
 25:     """
 26:     Manages margin requirements for derivatives and leveraged trading.
 27: 
 28:     Handles:
 29:     - Futures margin requirements
 30:     - Options margin for sellers
 31:     - FX leverage
 32:     - Crypto perpetuals and leveraged trading
 33:     - Portfolio margining
 34:     """
 35: 
 36:     cash_balance: Cash
 37:     initial_margin_requirement: Cash = 0.0
 38:     maintenance_margin_requirement: Cash = 0.0
 39:     margin_used: Cash = 0.0
 40:     available_margin: Cash = 0.0
 41:     positions: dict[AssetId, dict] = field(default_factory=dict)
 42: 
 43:     # Risk parameters
 44:     margin_call_level: float = 1.0  # 100% of maintenance margin
 45:     liquidation_level: float = 0.8  # 80% of maintenance margin
 46: 
 47:     def __init__(self, initial_cash: Cash, asset_registry: AssetRegistry):
 48:         """Initialize margin account."""
 49:         self.cash_balance = initial_cash
 50:         self.available_margin = initial_cash
 51:         self.asset_registry = asset_registry
 52:         self.positions = {}
 53:         self.margin_calls: list[MarginRequirement] = []
 54: 
 55:     def check_margin_requirement(
 56:         self,
 57:         asset_id: AssetId,
 58:         quantity: Quantity,
 59:         price: Price,
 60:     ) -> tuple[bool, Cash]:
 61:         """
 62:         Check if there's sufficient margin for a new position.
 63: 
 64:         Args:
 65:             asset_id: Asset to trade
 66:             quantity: Quantity to trade
 67:             price: Current price
 68: 
 69:         Returns:
 70:             Tuple of (has_sufficient_margin, required_margin)
 71:         """
 72:         asset_spec = self.asset_registry.get(asset_id)
 73:         if not asset_spec:
 74:             # Default to equity-like behavior
 75:             required = abs(quantity) * price
 76:             return self.available_margin >= required, required
 77: 
 78:         required_margin = asset_spec.get_margin_requirement(quantity, price)
 79:         has_margin = self.available_margin >= required_margin
 80: 
 81:         return has_margin, required_margin
 82: 
 83:     def open_position(
 84:         self,
 85:         asset_id: AssetId,
 86:         quantity: Quantity,
 87:         price: Price,
 88:         timestamp: datetime,
 89:     ) -> bool:
 90:         """
 91:         Open or modify a position with margin.
 92: 
 93:         Args:
 94:             asset_id: Asset to trade
 95:             quantity: Quantity to trade (positive for long, negative for short)
 96:             price: Entry price
 97:             timestamp: Transaction time
 98: 
 99:         Returns:
100:             Success status
101:         """
102:         has_margin, required_margin = self.check_margin_requirement(asset_id, quantity, price)
103: 
104:         if not has_margin:
105:             return False
106: 
107:         asset_spec = self.asset_registry.get(asset_id)
108: 
109:         if asset_id in self.positions:
110:             # Modify existing position
111:             pos = self.positions[asset_id]
112:             old_margin = pos["margin_used"]
113: 
114:             # Update position
115:             pos["quantity"] += quantity
116:             pos["avg_price"] = (
117:                 (pos["avg_price"] * abs(pos["quantity"] - quantity) + price * abs(quantity))
118:                 / abs(pos["quantity"])
119:                 if pos["quantity"] != 0
120:                 else 0
121:             )
122:             pos["last_price"] = price
123:             pos["timestamp"] = timestamp
124: 
125:             # Recalculate margin
126:             if asset_spec:
127:                 new_margin = asset_spec.get_margin_requirement(pos["quantity"], price)
128:                 pos["margin_used"] = new_margin
129:                 margin_change = new_margin - old_margin
130:             else:
131:                 pos["margin_used"] = abs(pos["quantity"]) * price
132:                 margin_change = pos["margin_used"] - old_margin
133: 
134:             # Update account margins
135:             self.margin_used += margin_change
136:             self.available_margin -= margin_change
137: 
138:             # Remove position if closed
139:             if pos["quantity"] == 0:
140:                 self.margin_used -= pos["margin_used"]
141:                 self.available_margin += pos["margin_used"]
142:                 del self.positions[asset_id]
143:         else:
144:             # Open new position
145:             self.positions[asset_id] = {
146:                 "quantity": quantity,
147:                 "avg_price": price,
148:                 "last_price": price,
149:                 "margin_used": required_margin,
150:                 "timestamp": timestamp,
151:                 "asset_spec": asset_spec,
152:             }
153: 
154:             self.margin_used += required_margin
155:             self.available_margin -= required_margin
156: 
157:         # Update margin requirements
158:         self._update_margin_requirements()
159: 
160:         return True
161: 
162:     def update_prices(self, prices: dict[AssetId, Price]) -> list[MarginRequirement]:
163:         """
164:         Update positions with new prices and check margin requirements.
165: 
166:         Args:
167:             prices: Current market prices
168: 
169:         Returns:
170:             List of margin requirements/calls
171:         """
172:         margin_status = []
173: 
174:         for asset_id, price in prices.items():
175:             if asset_id not in self.positions:
176:                 continue
177: 
178:             pos = self.positions[asset_id]
179:             pos["last_price"] = price
180: 
181:             # Calculate unrealized P&L
182:             pos["quantity"] * (price - pos["avg_price"])
183: 
184:             # Update margin for this position
185:             asset_spec = pos.get("asset_spec")
186:             if asset_spec and asset_spec.requires_margin:
187:                 current_margin = asset_spec.get_margin_requirement(pos["quantity"], price)
188: 
189:                 # Calculate liquidation price
190:                 liquidation_price = self._calculate_liquidation_price(asset_id, pos, asset_spec)
191: 
192:                 # Check margin status
193:                 margin_req = MarginRequirement(
194:                     asset_id=asset_id,
195:                     initial_margin=asset_spec.initial_margin * abs(pos["quantity"]),
196:                     maintenance_margin=asset_spec.maintenance_margin * abs(pos["quantity"]),
197:                     current_margin=current_margin,
198:                     excess_margin=self.available_margin,
199:                     margin_call=current_margin > self.available_margin * self.margin_call_level,
200:                     liquidation_price=liquidation_price,
201:                 )
202: 
203:                 margin_status.append(margin_req)
204: 
205:                 # Force liquidation if below threshold
206:                 if current_margin > self.available_margin * self.liquidation_level:
207:                     self._force_liquidation(asset_id, price)
208: 
209:         # Update total equity
210:         self._update_equity()
211: 
212:         return margin_status
213: 
214:     def _calculate_liquidation_price(
215:         self,
216:         asset_id: AssetId,
217:         position: dict,
218:         asset_spec: AssetSpec,
219:     ) -> Price | None:
220:         """Calculate liquidation price for a position."""
221:         if not asset_spec.requires_margin:
222:             return None
223: 
224:         quantity = position["quantity"]
225:         avg_price = position["avg_price"]
226: 
227:         if asset_spec.asset_class.value == "future":
228:             # Futures liquidation when margin depleted
229:             maintenance_margin = asset_spec.maintenance_margin * abs(quantity)
230:             if quantity > 0:  # Long position
231:                 return avg_price - (self.available_margin - maintenance_margin) / (
232:                     quantity * asset_spec.contract_size
233:                 )
234:             # Short position
235:             return avg_price + (self.available_margin - maintenance_margin) / (
236:                 abs(quantity) * asset_spec.contract_size
237:             )
238:         if asset_spec.asset_class.value == "fx":
239:             # FX liquidation based on leverage
240:             margin_used = position["margin_used"]
241:             if quantity > 0:
242:                 return avg_price * (
243:                     1 - self.liquidation_level * margin_used / (abs(quantity) * avg_price)
244:                 )
245:             return avg_price * (
246:                 1 + self.liquidation_level * margin_used / (abs(quantity) * avg_price)
247:             )
248: 
249:         return None
250: 
251:     def _force_liquidation(self, asset_id: AssetId, price: Price) -> None:
252:         """Force liquidate a position due to margin call."""
253:         if asset_id in self.positions:
254:             pos = self.positions[asset_id]
255: 
256:             # Return margin to available
257:             self.margin_used -= pos["margin_used"]
258:             self.available_margin += pos["margin_used"]
259: 
260:             # Calculate and apply loss
261:             loss = pos["quantity"] * (price - pos["avg_price"])
262:             self.cash_balance += loss  # Loss reduces cash
263: 
264:             # Remove position
265:             del self.positions[asset_id]
266: 
267:     def _update_margin_requirements(self) -> None:
268:         """Update total margin requirements."""
269:         self.initial_margin_requirement = 0.0
270:         self.maintenance_margin_requirement = 0.0
271: 
272:         for _asset_id, pos in self.positions.items():
273:             asset_spec = pos.get("asset_spec")
274:             if asset_spec and asset_spec.requires_margin:
275:                 self.initial_margin_requirement += asset_spec.initial_margin * abs(pos["quantity"])
276:                 self.maintenance_margin_requirement += asset_spec.maintenance_margin * abs(
277:                     pos["quantity"],
278:                 )
279: 
280:     def _update_equity(self) -> None:
281:         """Update total account equity."""
282:         total_unrealized = 0.0
283: 
284:         for pos in self.positions.values():
285:             unrealized = pos["quantity"] * (pos["last_price"] - pos["avg_price"])
286: 
287:             # Apply contract multiplier for futures
288:             asset_spec = pos.get("asset_spec")
289:             if asset_spec and asset_spec.asset_class.value == "future":
290:                 unrealized *= asset_spec.contract_size
291: 
292:             total_unrealized += unrealized
293: 
294:         # Update available margin with unrealized P&L
295:         self.available_margin = self.cash_balance + total_unrealized - self.margin_used
296: 
297:     def get_margin_status(self) -> dict:
298:         """Get current margin account status."""
299:         total_unrealized = sum(
300:             pos["quantity"] * (pos["last_price"] - pos["avg_price"])
301:             for pos in self.positions.values()
302:         )
303: 
304:         return {
305:             "cash_balance": self.cash_balance,
306:             "margin_used": self.margin_used,
307:             "available_margin": self.available_margin,
308:             "initial_requirement": self.initial_margin_requirement,
309:             "maintenance_requirement": self.maintenance_margin_requirement,
310:             "unrealized_pnl": total_unrealized,
311:             "total_equity": self.cash_balance + total_unrealized,
312:             "margin_utilization": self.margin_used / self.available_margin
313:             if self.available_margin > 0
314:             else 0,
315:             "num_positions": len(self.positions),
316:             "has_margin_call": any(mc.margin_call for mc in self.margin_calls),
317:         }
````

## File: src/qengine/portfolio/portfolio.py
````python
  1: """Portfolio state management for QEngine."""
  2: 
  3: from dataclasses import dataclass, field
  4: from datetime import datetime
  5: from typing import Any
  6: 
  7: from qengine.core.types import AssetId, Cash, Quantity
  8: 
  9: 
 10: @dataclass
 11: class Position:
 12:     """Represents a position in a single asset."""
 13: 
 14:     asset_id: AssetId
 15:     quantity: Quantity = 0.0
 16:     cost_basis: float = 0.0
 17:     last_price: float = 0.0
 18:     realized_pnl: float = 0.0
 19:     unrealized_pnl: float = 0.0
 20: 
 21:     @property
 22:     def market_value(self) -> float:
 23:         """Current market value of the position."""
 24:         return self.quantity * self.last_price
 25: 
 26:     @property
 27:     def total_pnl(self) -> float:
 28:         """Total P&L (realized + unrealized)."""
 29:         return self.realized_pnl + self.unrealized_pnl
 30: 
 31:     def update_price(self, price: float) -> None:
 32:         """Update position with new market price."""
 33:         self.last_price = price
 34:         if self.quantity != 0:
 35:             self.unrealized_pnl = self.quantity * (price - self.cost_basis / self.quantity)
 36: 
 37:     def add_shares(self, quantity: Quantity, price: float) -> None:
 38:         """Add shares to position."""
 39:         if self.quantity + quantity == 0:
 40:             # Closing position
 41:             self.realized_pnl += self.unrealized_pnl
 42:             self.unrealized_pnl = 0.0
 43:             self.cost_basis = 0.0
 44:         else:
 45:             # Update cost basis
 46:             new_cost = quantity * price
 47:             self.cost_basis += new_cost
 48: 
 49:         self.quantity += quantity
 50:         self.update_price(price)
 51: 
 52:     def remove_shares(self, quantity: Quantity, price: float) -> float:
 53:         """Remove shares from position, returns realized P&L."""
 54:         if abs(quantity) > abs(self.quantity):
 55:             raise ValueError(f"Cannot remove {quantity} shares, only have {self.quantity}")
 56: 
 57:         # Calculate realized P&L for the shares being removed
 58:         avg_cost = self.cost_basis / self.quantity if self.quantity != 0 else 0
 59:         realized = quantity * (price - avg_cost)
 60:         self.realized_pnl += realized
 61: 
 62:         # Update cost basis
 63:         self.cost_basis -= quantity * avg_cost
 64:         self.quantity -= quantity
 65: 
 66:         self.update_price(price)
 67:         return realized
 68: 
 69: 
 70: @dataclass
 71: class PortfolioState:
 72:     """Complete portfolio state at a point in time."""
 73: 
 74:     timestamp: datetime
 75:     cash: Cash
 76:     positions: dict[AssetId, Position] = field(default_factory=dict)
 77:     pending_orders: list[Any] = field(default_factory=list)
 78:     filled_orders: list[Any] = field(default_factory=list)
 79: 
 80:     # Performance metrics
 81:     total_value: float = 0.0
 82:     total_realized_pnl: float = 0.0
 83:     total_unrealized_pnl: float = 0.0
 84:     total_commission: float = 0.0
 85:     total_slippage: float = 0.0
 86: 
 87:     # Risk metrics
 88:     leverage: float = 1.0
 89:     max_position_value: float = 0.0
 90:     concentration: float = 0.0
 91: 
 92:     @property
 93:     def equity(self) -> float:
 94:         """Total equity (cash + positions)."""
 95:         position_value = sum(p.market_value for p in self.positions.values())
 96:         return self.cash + position_value
 97: 
 98:     @property
 99:     def total_pnl(self) -> float:
100:         """Total P&L across all positions."""
101:         return self.total_realized_pnl + self.total_unrealized_pnl
102: 
103:     def update_metrics(self) -> None:
104:         """Update portfolio metrics."""
105:         # Update position values
106:         position_values = [p.market_value for p in self.positions.values()]
107: 
108:         if position_values:
109:             self.max_position_value = max(abs(v) for v in position_values)
110:             total_position_value = sum(abs(v) for v in position_values)
111: 
112:             # Update concentration (largest position as % of portfolio)
113:             if self.equity > 0:
114:                 self.concentration = self.max_position_value / self.equity
115:                 self.leverage = total_position_value / self.equity
116:             else:
117:                 self.concentration = 0.0
118:                 self.leverage = 0.0
119:         else:
120:             self.max_position_value = 0.0
121:             self.concentration = 0.0
122:             self.leverage = 0.0
123: 
124:         # Update P&L
125:         self.total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
126:         self.total_realized_pnl = sum(p.realized_pnl for p in self.positions.values())
127:         self.total_value = self.equity
128: 
129: 
130: class Portfolio:
131:     """Portfolio management with state tracking."""
132: 
133:     def __init__(self, initial_cash: Cash = 100000.0):
134:         """Initialize portfolio with starting cash."""
135:         self.initial_cash = initial_cash
136:         self.cash = initial_cash
137:         self.positions: dict[AssetId, Position] = {}
138: 
139:         # Track cumulative costs
140:         self.total_commission = 0.0
141:         self.total_slippage = 0.0
142:         self.total_realized_pnl = 0.0
143: 
144:         # History tracking
145:         self.state_history: list[PortfolioState] = []
146: 
147:     def get_position(self, asset_id: AssetId) -> Position | None:
148:         """Get position for an asset."""
149:         return self.positions.get(asset_id)
150: 
151:     def update_position(
152:         self,
153:         asset_id: AssetId,
154:         quantity_change: Quantity,
155:         price: float,
156:         commission: float = 0.0,
157:         slippage: float = 0.0,
158:     ) -> None:
159:         """
160:         Update a position with a trade.
161: 
162:         Args:
163:             asset_id: Asset identifier
164:             quantity_change: Change in quantity (positive for buy, negative for sell)
165:             price: Execution price
166:             commission: Commission paid
167:             slippage: Slippage cost
168:         """
169:         # Get or create position
170:         if asset_id not in self.positions:
171:             self.positions[asset_id] = Position(asset_id=asset_id)
172: 
173:         position = self.positions[asset_id]
174: 
175:         # Update position
176:         if quantity_change > 0:
177:             position.add_shares(quantity_change, price)
178:             self.cash -= quantity_change * price + commission
179:         else:
180:             realized_pnl = position.remove_shares(-quantity_change, price)
181:             self.cash += (-quantity_change) * price - commission
182:             self.total_realized_pnl += realized_pnl
183: 
184:         # Track costs
185:         self.total_commission += commission
186:         self.total_slippage += slippage
187: 
188:         # Remove empty positions
189:         if position.quantity == 0 and position.realized_pnl == 0:
190:             del self.positions[asset_id]
191: 
192:     def update_prices(self, prices: dict[AssetId, float]) -> None:
193:         """Update all positions with new market prices."""
194:         for asset_id, price in prices.items():
195:             if asset_id in self.positions:
196:                 self.positions[asset_id].update_price(price)
197: 
198:     def get_current_state(self, timestamp: datetime) -> PortfolioState:
199:         """Get current portfolio state."""
200:         state = PortfolioState(
201:             timestamp=timestamp,
202:             cash=self.cash,
203:             positions=self.positions.copy(),
204:             total_commission=self.total_commission,
205:             total_slippage=self.total_slippage,
206:             total_realized_pnl=self.total_realized_pnl,
207:         )
208:         state.update_metrics()
209:         return state
210: 
211:     def save_state(self, timestamp: datetime) -> None:
212:         """Save current state to history."""
213:         self.state_history.append(self.get_current_state(timestamp))
214: 
215:     @property
216:     def equity(self) -> float:
217:         """Total equity (cash + positions)."""
218:         position_value = sum(p.market_value for p in self.positions.values())
219:         return self.cash + position_value
220: 
221:     @property
222:     def returns(self) -> float:
223:         """Simple returns from initial capital."""
224:         if self.initial_cash == 0:
225:             return 0.0
226:         return (self.equity - self.initial_cash) / self.initial_cash
227: 
228:     @property
229:     def unrealized_pnl(self) -> float:
230:         """Total unrealized P&L."""
231:         return sum(p.unrealized_pnl for p in self.positions.values())
232: 
233:     def get_position_summary(self) -> dict[str, Any]:
234:         """Get summary of all positions."""
235:         return {
236:             "cash": self.cash,
237:             "equity": self.equity,
238:             "positions": len(self.positions),
239:             "realized_pnl": self.total_realized_pnl,
240:             "unrealized_pnl": self.unrealized_pnl,
241:             "total_pnl": self.total_realized_pnl + self.unrealized_pnl,
242:             "returns": self.returns,
243:             "commission": self.total_commission,
244:             "slippage": self.total_slippage,
245:         }
````

## File: src/qengine/portfolio/simple.py
````python
  1: """Simple portfolio implementation for basic backtesting."""
  2: 
  3: import logging
  4: from datetime import datetime
  5: from typing import Any
  6: 
  7: import polars as pl
  8: 
  9: from qengine.core.event import FillEvent, MarketEvent
 10: from qengine.core.types import Cash
 11: from qengine.portfolio.portfolio import Portfolio
 12: 
 13: logger = logging.getLogger(__name__)
 14: 
 15: 
 16: class SimplePortfolio(Portfolio):
 17:     """Simple portfolio implementation with basic tracking.
 18: 
 19:     This portfolio provides:
 20:     - Position tracking
 21:     - P&L calculation
 22:     - Basic performance metrics
 23:     - Event handling integration
 24:     """
 25: 
 26:     def __init__(self, initial_capital: Cash = 100000.0, currency: str = "USD"):
 27:         """Initialize simple portfolio.
 28: 
 29:         Args:
 30:             initial_capital: Starting cash balance
 31:             currency: Base currency
 32:         """
 33:         super().__init__(initial_cash=initial_capital)
 34:         self.currency = currency
 35:         self.trades = []  # Track all trades
 36:         self.current_prices = {}  # Latest market prices
 37: 
 38:     def initialize(self) -> None:
 39:         """Initialize portfolio for new backtest."""
 40:         logger.debug(f"Initializing portfolio with ${self.initial_cash:,.2f} {self.currency}")
 41: 
 42:     def on_fill_event(self, event: FillEvent) -> None:
 43:         """Handle fill event from broker.
 44: 
 45:         Args:
 46:             event: Fill event with execution details
 47:         """
 48:         # Update position
 49:         self.update_position(
 50:             asset_id=event.asset_id,
 51:             quantity_change=event.fill_quantity
 52:             if event.side.value in ["buy", "BUY"]
 53:             else -event.fill_quantity,
 54:             price=float(event.fill_price),
 55:             commission=event.commission,
 56:             slippage=event.slippage,
 57:         )
 58: 
 59:         # Record trade
 60:         self.trades.append(
 61:             {
 62:                 "timestamp": event.timestamp,
 63:                 "asset_id": event.asset_id,
 64:                 "side": event.side.value,
 65:                 "quantity": event.fill_quantity,
 66:                 "price": float(event.fill_price),
 67:                 "commission": event.commission,
 68:                 "slippage": event.slippage,
 69:                 "pnl": 0.0,  # Will be calculated later
 70:             },
 71:         )
 72: 
 73:         logger.info(
 74:             f"Fill: {event.side.value.upper()} {event.fill_quantity} {event.asset_id} "
 75:             f"@ ${float(event.fill_price):.2f} (commission: ${event.commission:.2f})",
 76:         )
 77: 
 78:     def update_market_value(self, event: MarketEvent) -> None:
 79:         """Update portfolio with latest market prices.
 80: 
 81:         Args:
 82:             event: Market event with price data
 83:         """
 84:         # Update current price for the asset
 85:         if hasattr(event, "close") and event.close is not None:
 86:             self.current_prices[event.asset_id] = float(event.close)
 87:         elif hasattr(event, "price") and event.price is not None:
 88:             self.current_prices[event.asset_id] = float(event.price)
 89: 
 90:         # Update all positions with latest prices
 91:         self.update_prices(self.current_prices)
 92: 
 93:     def get_total_value(self) -> float:
 94:         """Get total portfolio value (cash + positions).
 95: 
 96:         Returns:
 97:             Total portfolio value
 98:         """
 99:         return self.equity
100: 
101:     def get_positions(self) -> pl.DataFrame:
102:         """Get DataFrame of current positions.
103: 
104:         Returns:
105:             DataFrame with position details
106:         """
107:         if not self.positions:
108:             return pl.DataFrame()
109: 
110:         positions_data = []
111:         for asset_id, position in self.positions.items():
112:             positions_data.append(
113:                 {
114:                     "asset_id": asset_id,
115:                     "quantity": position.quantity,
116:                     "cost_basis": position.cost_basis,
117:                     "market_value": position.market_value,
118:                     "unrealized_pnl": position.unrealized_pnl,
119:                     "realized_pnl": position.realized_pnl,
120:                     "last_price": position.last_price,
121:                 },
122:             )
123: 
124:         return pl.DataFrame(positions_data)
125: 
126:     def get_trades(self) -> pl.DataFrame:
127:         """Get DataFrame of all trades.
128: 
129:         Returns:
130:             DataFrame with trade history
131:         """
132:         if not self.trades:
133:             return pl.DataFrame()
134:         return pl.DataFrame(self.trades)
135: 
136:     def get_returns(self) -> pl.Series:
137:         """Get returns series.
138: 
139:         Returns:
140:             Series of portfolio returns
141:         """
142:         if not self.state_history:
143:             return pl.Series([])
144: 
145:         returns = []
146:         prev_value = self.initial_cash
147: 
148:         for state in self.state_history:
149:             current_value = state.equity
150:             ret = (current_value - prev_value) / prev_value if prev_value != 0 else 0
151:             returns.append(ret)
152:             prev_value = current_value
153: 
154:         return pl.Series(returns)
155: 
156:     def calculate_metrics(self) -> dict[str, Any]:
157:         """Calculate performance metrics.
158: 
159:         Returns:
160:             Dictionary of performance metrics
161:         """
162:         returns = self.get_returns()
163: 
164:         metrics = {
165:             "total_return": self.returns * 100,  # Percentage
166:             "total_trades": len(self.trades),
167:             "winning_trades": sum(1 for t in self.trades if t.get("pnl", 0) > 0),
168:             "losing_trades": sum(1 for t in self.trades if t.get("pnl", 0) < 0),
169:             "total_commission": self.total_commission,
170:             "total_slippage": self.total_slippage,
171:             "final_equity": self.equity,
172:             "cash_remaining": self.cash,
173:         }
174: 
175:         # Calculate returns-based metrics if we have data
176:         if len(returns) > 0:
177:             import numpy as np
178: 
179:             returns_array = returns.to_numpy()
180: 
181:             # Remove any NaN values
182:             returns_array = returns_array[~np.isnan(returns_array)]
183: 
184:             if len(returns_array) > 0:
185:                 metrics["avg_return"] = np.mean(returns_array)
186:                 metrics["std_return"] = np.std(returns_array)
187: 
188:                 # Sharpe ratio (assuming 0 risk-free rate)
189:                 if metrics["std_return"] > 0:
190:                     metrics["sharpe_ratio"] = (
191:                         np.sqrt(252) * metrics["avg_return"] / metrics["std_return"]
192:                     )
193:                 else:
194:                     metrics["sharpe_ratio"] = 0.0
195: 
196:                 # Maximum drawdown
197:                 cumulative = np.cumprod(1 + returns_array)
198:                 running_max = np.maximum.accumulate(cumulative)
199:                 drawdown = (cumulative - running_max) / running_max
200:                 metrics["max_drawdown"] = np.min(drawdown) * 100  # Percentage
201: 
202:                 # Win rate
203:                 if metrics["total_trades"] > 0:
204:                     metrics["win_rate"] = (
205:                         metrics["winning_trades"] / metrics["total_trades"]
206:                     ) * 100
207:                 else:
208:                     metrics["win_rate"] = 0.0
209: 
210:         return metrics
211: 
212:     def finalize(self) -> None:
213:         """Finalize portfolio at end of backtest."""
214:         # Save final state
215:         self.save_state(datetime.now())
216: 
217:         # Calculate P&L for all trades
218:         for i, trade in enumerate(self.trades):
219:             if i > 0 and trade["side"] in ["sell", "SELL"]:
220:                 # Find corresponding buy trade and calculate P&L
221:                 # This is simplified - real implementation would match specific lots
222:                 prev_trades = [t for t in self.trades[:i] if t["asset_id"] == trade["asset_id"]]
223:                 if prev_trades:
224:                     avg_buy_price = sum(
225:                         t["price"] * t["quantity"]
226:                         for t in prev_trades
227:                         if t["side"] in ["buy", "BUY"]
228:                     ) / sum(t["quantity"] for t in prev_trades if t["side"] in ["buy", "BUY"])
229:                     trade["pnl"] = (trade["price"] - avg_buy_price) * trade["quantity"] - trade[
230:                         "commission"
231:                     ]
232: 
233:         logger.info(f"Portfolio finalized. Final equity: ${self.equity:,.2f}")
234: 
235:     def reset(self) -> None:
236:         """Reset portfolio to initial state."""
237:         self.cash = self.initial_cash
238:         self.positions.clear()
239:         self.trades.clear()
240:         self.current_prices.clear()
241:         self.state_history.clear()
242:         self.total_commission = 0.0
243:         self.total_slippage = 0.0
244:         self.total_realized_pnl = 0.0
245: 
246: 
247: __all__ = ["SimplePortfolio"]
````

## File: src/qengine/reporting/__init__.py
````python
 1: """Reporting module for QEngine."""
 2: 
 3: from qengine.reporting.base import ReportGenerator
 4: from qengine.reporting.html import HTMLReportGenerator
 5: from qengine.reporting.parquet import ParquetReportGenerator
 6: from qengine.reporting.reporter import ConsoleReporter, InMemoryReporter, Reporter
 7: 
 8: __all__ = [
 9:     "ConsoleReporter",
10:     "HTMLReportGenerator",
11:     "InMemoryReporter",
12:     "ParquetReportGenerator",
13:     "ReportGenerator",
14:     "Reporter",
15: ]
````

## File: src/qengine/reporting/base.py
````python
  1: """Base reporting functionality for QEngine."""
  2: 
  3: from abc import ABC, abstractmethod
  4: from datetime import datetime
  5: from pathlib import Path
  6: from typing import Any
  7: 
  8: from qengine.portfolio.accounting import PortfolioAccounting
  9: 
 10: 
 11: class ReportGenerator(ABC):
 12:     """
 13:     Abstract base class for report generation.
 14: 
 15:     Different report formats (HTML, Parquet, JSON) should implement this interface.
 16:     """
 17: 
 18:     def __init__(self, output_dir: Path | None = None, report_name: str | None = None):
 19:         """
 20:         Initialize report generator.
 21: 
 22:         Args:
 23:             output_dir: Directory to save reports
 24:             report_name: Base name for report files
 25:         """
 26:         self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "reports"
 27:         self.report_name = report_name or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
 28: 
 29:         # Ensure output directory exists
 30:         self.output_dir.mkdir(parents=True, exist_ok=True)
 31: 
 32:     @abstractmethod
 33:     def generate(
 34:         self,
 35:         accounting: PortfolioAccounting,
 36:         strategy_params: dict[str, Any] | None = None,
 37:         backtest_params: dict[str, Any] | None = None,
 38:     ) -> Path:
 39:         """
 40:         Generate report from portfolio accounting data.
 41: 
 42:         Args:
 43:             accounting: Portfolio accounting with results
 44:             strategy_params: Strategy configuration parameters
 45:             backtest_params: Backtest configuration parameters
 46: 
 47:         Returns:
 48:             Path to generated report
 49:         """
 50: 
 51:     def _prepare_report_data(
 52:         self,
 53:         accounting: PortfolioAccounting,
 54:         strategy_params: dict[str, Any] | None = None,
 55:         backtest_params: dict[str, Any] | None = None,
 56:     ) -> dict[str, Any]:
 57:         """
 58:         Prepare standardized report data from accounting.
 59: 
 60:         Args:
 61:             accounting: Portfolio accounting instance
 62:             strategy_params: Strategy parameters
 63:             backtest_params: Backtest parameters
 64: 
 65:         Returns:
 66:             Dictionary with report data
 67:         """
 68:         # Get performance metrics
 69:         metrics = accounting.get_performance_metrics()
 70: 
 71:         # Get summary data
 72:         summary = accounting.get_summary()
 73: 
 74:         # Prepare report data structure
 75:         report_data = {
 76:             "metadata": {
 77:                 "report_name": self.report_name,
 78:                 "generated_at": datetime.now().isoformat(),
 79:                 "strategy_params": strategy_params or {},
 80:                 "backtest_params": backtest_params or {},
 81:             },
 82:             "performance": {
 83:                 "total_return": metrics.get("total_return", 0.0),
 84:                 "total_pnl": metrics.get("total_pnl", 0.0),
 85:                 "realized_pnl": metrics.get("realized_pnl", 0.0),
 86:                 "unrealized_pnl": metrics.get("unrealized_pnl", 0.0),
 87:                 "max_drawdown": metrics.get("max_drawdown", 0.0),
 88:                 "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
 89:                 "num_trades": metrics.get("num_trades", 0),
 90:                 "win_rate": self._calculate_win_rate(accounting),
 91:                 "profit_factor": self._calculate_profit_factor(accounting),
 92:             },
 93:             "costs": {
 94:                 "total_commission": metrics.get("total_commission", 0.0),
 95:                 "total_slippage": metrics.get("total_slippage", 0.0),
 96:                 "avg_commission_per_trade": self._calculate_avg_commission(accounting),
 97:                 "avg_slippage_per_trade": self._calculate_avg_slippage(accounting),
 98:             },
 99:             "portfolio": {
100:                 "initial_cash": accounting.portfolio.initial_cash,
101:                 "final_equity": summary.get("equity", 0.0),
102:                 "final_cash": summary.get("cash", 0.0),
103:                 "num_positions": summary.get("positions", 0),
104:             },
105:             "risk": {
106:                 "max_leverage": metrics.get("max_leverage", 1.0),
107:                 "max_concentration": metrics.get("max_concentration", 0.0),
108:             },
109:             "trades": accounting.get_trades_df(),
110:             "equity_curve": accounting.get_equity_curve_df(),
111:             "positions": accounting.get_positions_df(),
112:         }
113: 
114:         return report_data
115: 
116:     def _calculate_win_rate(self, accounting: PortfolioAccounting) -> float:
117:         """Calculate win rate from trades."""
118:         if not accounting.fills:
119:             return 0.0
120: 
121:         # Group trades by asset to calculate P&L
122:         winning_trades = 0
123:         total_trades = 0
124: 
125:         # Track positions to determine win/loss
126:         position_pnl = {}
127: 
128:         for fill in accounting.fills:
129:             asset_id = fill.asset_id
130: 
131:             if fill.side.value == "buy":
132:                 if asset_id not in position_pnl:
133:                     position_pnl[asset_id] = []
134:                 position_pnl[asset_id].append(
135:                     {"quantity": fill.fill_quantity, "price": fill.fill_price},
136:                 )
137:             elif fill.side.value == "sell":
138:                 if position_pnl.get(asset_id):
139:                     # Calculate P&L for this trade
140:                     buy_info = position_pnl[asset_id].pop(0)
141:                     pnl = (fill.fill_price - buy_info["price"]) * min(
142:                         fill.fill_quantity,
143:                         buy_info["quantity"],
144:                     )
145: 
146:                     total_trades += 1
147:                     if pnl > 0:
148:                         winning_trades += 1
149: 
150:         return winning_trades / total_trades if total_trades > 0 else 0.0
151: 
152:     def _calculate_profit_factor(self, accounting: PortfolioAccounting) -> float:
153:         """Calculate profit factor (gross profit / gross loss)."""
154:         gross_profit = 0.0
155:         gross_loss = 0.0
156: 
157:         # Track positions to determine profits and losses
158:         position_costs = {}
159: 
160:         for fill in accounting.fills:
161:             asset_id = fill.asset_id
162: 
163:             if fill.side.value == "buy":
164:                 if asset_id not in position_costs:
165:                     position_costs[asset_id] = []
166:                 position_costs[asset_id].append(
167:                     {"quantity": fill.fill_quantity, "price": fill.fill_price},
168:                 )
169:             elif fill.side.value == "sell":
170:                 if position_costs.get(asset_id):
171:                     # Calculate P&L for this trade
172:                     buy_info = position_costs[asset_id].pop(0)
173:                     pnl = (fill.fill_price - buy_info["price"]) * min(
174:                         fill.fill_quantity,
175:                         buy_info["quantity"],
176:                     )
177: 
178:                     if pnl > 0:
179:                         gross_profit += pnl
180:                     else:
181:                         gross_loss += abs(pnl)
182: 
183:         return (
184:             gross_profit / gross_loss
185:             if gross_loss > 0
186:             else float("inf")
187:             if gross_profit > 0
188:             else 0.0
189:         )
190: 
191:     def _calculate_avg_commission(self, accounting: PortfolioAccounting) -> float:
192:         """Calculate average commission per trade."""
193:         if not accounting.fills:
194:             return 0.0
195:         return accounting.portfolio.total_commission / len(accounting.fills)
196: 
197:     def _calculate_avg_slippage(self, accounting: PortfolioAccounting) -> float:
198:         """Calculate average slippage per trade."""
199:         if not accounting.fills:
200:             return 0.0
201:         return accounting.portfolio.total_slippage / len(accounting.fills)
````

## File: src/qengine/reporting/html.py
````python
  1: """HTML report generation for QEngine backtests."""
  2: 
  3: import json
  4: from pathlib import Path
  5: from typing import Any
  6: 
  7: from qengine.portfolio.accounting import PortfolioAccounting
  8: from qengine.reporting.base import ReportGenerator
  9: 
 10: 
 11: class HTMLReportGenerator(ReportGenerator):
 12:     """
 13:     Generates comprehensive HTML reports for backtest results.
 14: 
 15:     Creates interactive reports with:
 16:     - Performance summary
 17:     - Equity curve charts
 18:     - Trade analysis
 19:     - Risk metrics
 20:     - Asset class breakdown
 21:     """
 22: 
 23:     def generate(
 24:         self,
 25:         accounting: PortfolioAccounting,
 26:         strategy_params: dict[str, Any] | None = None,
 27:         backtest_params: dict[str, Any] | None = None,
 28:     ) -> Path:
 29:         """
 30:         Generate HTML report from portfolio accounting data.
 31: 
 32:         Args:
 33:             accounting: Portfolio accounting with results
 34:             strategy_params: Strategy configuration parameters
 35:             backtest_params: Backtest configuration parameters
 36: 
 37:         Returns:
 38:             Path to generated HTML report
 39:         """
 40:         # Prepare report data
 41:         report_data = self._prepare_report_data(accounting, strategy_params, backtest_params)
 42: 
 43:         # Generate HTML content
 44:         html_content = self._generate_html_content(report_data)
 45: 
 46:         # Save report
 47:         report_path = self.output_dir / f"{self.report_name}.html"
 48:         with open(report_path, "w", encoding="utf-8") as f:
 49:             f.write(html_content)
 50: 
 51:         return report_path
 52: 
 53:     def _generate_html_content(self, report_data: dict[str, Any]) -> str:
 54:         """Generate the complete HTML report content."""
 55:         html = f"""
 56: <!DOCTYPE html>
 57: <html lang="en">
 58: <head>
 59:     <meta charset="UTF-8">
 60:     <meta name="viewport" content="width=device-width, initial-scale=1.0">
 61:     <title>QEngine Backtest Report - {report_data["metadata"]["report_name"]}</title>
 62:     <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
 63:     <style>
 64:         {self._get_css_styles()}
 65:     </style>
 66: </head>
 67: <body>
 68:     <div class="container">
 69:         {self._generate_header(report_data)}
 70:         {self._generate_summary_section(report_data)}
 71:         {self._generate_performance_section(report_data)}
 72:         {self._generate_charts_section(report_data)}
 73:         {self._generate_trades_section(report_data)}
 74:         {self._generate_positions_section(report_data)}
 75:         {self._generate_risk_section(report_data)}
 76:         {self._generate_footer(report_data)}
 77:     </div>
 78: 
 79:     <script>
 80:         {self._generate_javascript(report_data)}
 81:     </script>
 82: </body>
 83: </html>
 84: """
 85:         return html
 86: 
 87:     def _get_css_styles(self) -> str:
 88:         """Get CSS styles for the report."""
 89:         return """
 90:         * {
 91:             margin: 0;
 92:             padding: 0;
 93:             box-sizing: border-box;
 94:         }
 95: 
 96:         body {
 97:             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
 98:             background-color: #f5f5f5;
 99:             color: #333;
100:             line-height: 1.6;
101:         }
102: 
103:         .container {
104:             max-width: 1200px;
105:             margin: 0 auto;
106:             padding: 20px;
107:         }
108: 
109:         .header {
110:             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
111:             color: white;
112:             padding: 30px;
113:             border-radius: 10px;
114:             margin-bottom: 30px;
115:             text-align: center;
116:         }
117: 
118:         .header h1 {
119:             font-size: 2.5em;
120:             margin-bottom: 10px;
121:         }
122: 
123:         .header .subtitle {
124:             font-size: 1.2em;
125:             opacity: 0.9;
126:         }
127: 
128:         .section {
129:             background: white;
130:             margin-bottom: 30px;
131:             padding: 25px;
132:             border-radius: 10px;
133:             box-shadow: 0 4px 6px rgba(0,0,0,0.1);
134:         }
135: 
136:         .section h2 {
137:             color: #667eea;
138:             font-size: 1.8em;
139:             margin-bottom: 20px;
140:             border-bottom: 3px solid #667eea;
141:             padding-bottom: 10px;
142:         }
143: 
144:         .metrics-grid {
145:             display: grid;
146:             grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
147:             gap: 20px;
148:             margin-bottom: 20px;
149:         }
150: 
151:         .metric-card {
152:             background: #f8f9ff;
153:             padding: 20px;
154:             border-radius: 8px;
155:             border-left: 4px solid #667eea;
156:         }
157: 
158:         .metric-label {
159:             font-size: 0.9em;
160:             color: #666;
161:             margin-bottom: 5px;
162:             text-transform: uppercase;
163:             letter-spacing: 0.5px;
164:         }
165: 
166:         .metric-value {
167:             font-size: 1.8em;
168:             font-weight: bold;
169:             color: #333;
170:         }
171: 
172:         .metric-value.positive {
173:             color: #28a745;
174:         }
175: 
176:         .metric-value.negative {
177:             color: #dc3545;
178:         }
179: 
180:         .chart-container {
181:             margin: 20px 0;
182:             min-height: 400px;
183:         }
184: 
185:         .table-container {
186:             overflow-x: auto;
187:             margin-top: 20px;
188:         }
189: 
190:         table {
191:             width: 100%;
192:             border-collapse: collapse;
193:             background: white;
194:         }
195: 
196:         th, td {
197:             padding: 12px;
198:             text-align: left;
199:             border-bottom: 1px solid #ddd;
200:         }
201: 
202:         th {
203:             background-color: #f8f9ff;
204:             font-weight: 600;
205:             color: #667eea;
206:         }
207: 
208:         tr:hover {
209:             background-color: #f8f9ff;
210:         }
211: 
212:         .footer {
213:             text-align: center;
214:             color: #666;
215:             font-size: 0.9em;
216:             margin-top: 40px;
217:             padding-top: 20px;
218:             border-top: 1px solid #ddd;
219:         }
220: 
221:         .warning {
222:             background-color: #fff3cd;
223:             border: 1px solid #ffeaa7;
224:             color: #856404;
225:             padding: 15px;
226:             border-radius: 5px;
227:             margin: 15px 0;
228:         }
229: 
230:         .info {
231:             background-color: #d1ecf1;
232:             border: 1px solid #bee5eb;
233:             color: #0c5460;
234:             padding: 15px;
235:             border-radius: 5px;
236:             margin: 15px 0;
237:         }
238: 
239:         @media (max-width: 768px) {
240:             .metrics-grid {
241:                 grid-template-columns: 1fr;
242:             }
243: 
244:             .header h1 {
245:                 font-size: 2em;
246:             }
247:         }
248:         """
249: 
250:     def _generate_header(self, report_data: dict[str, Any]) -> str:
251:         """Generate the report header."""
252:         metadata = report_data["metadata"]
253:         return f"""
254:         <div class="header">
255:             <h1>QEngine Backtest Report</h1>
256:             <div class="subtitle">
257:                 {metadata["report_name"]}<br>
258:                 Generated: {metadata["generated_at"][:19]}
259:             </div>
260:         </div>
261:         """
262: 
263:     def _generate_summary_section(self, report_data: dict[str, Any]) -> str:
264:         """Generate the summary metrics section."""
265:         perf = report_data["performance"]
266:         portfolio = report_data["portfolio"]
267:         costs = report_data["costs"]
268: 
269:         return f"""
270:         <div class="section">
271:             <h2>Performance Summary</h2>
272:             <div class="metrics-grid">
273:                 <div class="metric-card">
274:                     <div class="metric-label">Total Return</div>
275:                     <div class="metric-value {"positive" if perf["total_return"] >= 0 else "negative"}">
276:                         {perf["total_return"]:.2%}
277:                     </div>
278:                 </div>
279: 
280:                 <div class="metric-card">
281:                     <div class="metric-label">Total P&L</div>
282:                     <div class="metric-value {"positive" if perf["total_pnl"] >= 0 else "negative"}">
283:                         ${perf["total_pnl"]:,.2f}
284:                     </div>
285:                 </div>
286: 
287:                 <div class="metric-card">
288:                     <div class="metric-label">Sharpe Ratio</div>
289:                     <div class="metric-value {"positive" if perf.get("sharpe_ratio", 0) >= 0 else "negative"}">
290:                         {perf.get("sharpe_ratio", "N/A")}
291:                     </div>
292:                 </div>
293: 
294:                 <div class="metric-card">
295:                     <div class="metric-label">Max Drawdown</div>
296:                     <div class="metric-value negative">
297:                         -{perf["max_drawdown"]:.2%}
298:                     </div>
299:                 </div>
300: 
301:                 <div class="metric-card">
302:                     <div class="metric-label">Win Rate</div>
303:                     <div class="metric-value">
304:                         {perf["win_rate"]:.2%}
305:                     </div>
306:                 </div>
307: 
308:                 <div class="metric-card">
309:                     <div class="metric-label">Total Trades</div>
310:                     <div class="metric-value">
311:                         {perf["num_trades"]:,}
312:                     </div>
313:                 </div>
314: 
315:                 <div class="metric-card">
316:                     <div class="metric-label">Final Equity</div>
317:                     <div class="metric-value">
318:                         ${portfolio["final_equity"]:,.2f}
319:                     </div>
320:                 </div>
321: 
322:                 <div class="metric-card">
323:                     <div class="metric-label">Total Commission</div>
324:                     <div class="metric-value negative">
325:                         ${costs["total_commission"]:,.2f}
326:                     </div>
327:                 </div>
328:             </div>
329:         </div>
330:         """
331: 
332:     def _generate_performance_section(self, report_data: dict[str, Any]) -> str:
333:         """Generate detailed performance metrics."""
334:         perf = report_data["performance"]
335:         costs = report_data["costs"]
336: 
337:         return f"""
338:         <div class="section">
339:             <h2>Detailed Performance</h2>
340: 
341:             <div class="metrics-grid">
342:                 <div class="metric-card">
343:                     <div class="metric-label">Realized P&L</div>
344:                     <div class="metric-value {"positive" if perf["realized_pnl"] >= 0 else "negative"}">
345:                         ${perf["realized_pnl"]:,.2f}
346:                     </div>
347:                 </div>
348: 
349:                 <div class="metric-card">
350:                     <div class="metric-label">Unrealized P&L</div>
351:                     <div class="metric-value {"positive" if perf["unrealized_pnl"] >= 0 else "negative"}">
352:                         ${perf["unrealized_pnl"]:,.2f}
353:                     </div>
354:                 </div>
355: 
356:                 <div class="metric-card">
357:                     <div class="metric-label">Profit Factor</div>
358:                     <div class="metric-value">
359:                         {perf["profit_factor"]:.2f}
360:                     </div>
361:                 </div>
362: 
363:                 <div class="metric-card">
364:                     <div class="metric-label">Avg Commission/Trade</div>
365:                     <div class="metric-value">
366:                         ${costs["avg_commission_per_trade"]:.2f}
367:                     </div>
368:                 </div>
369:             </div>
370:         </div>
371:         """
372: 
373:     def _generate_charts_section(self, report_data: dict[str, Any]) -> str:
374:         """Generate charts section."""
375:         return """
376:         <div class="section">
377:             <h2>Charts</h2>
378: 
379:             <div class="chart-container">
380:                 <div id="equity-curve-chart"></div>
381:             </div>
382: 
383:             <div class="chart-container">
384:                 <div id="returns-chart"></div>
385:             </div>
386:         </div>
387:         """
388: 
389:     def _generate_trades_section(self, report_data: dict[str, Any]) -> str:
390:         """Generate trades analysis section."""
391:         trades_df = report_data.get("trades")
392: 
393:         if trades_df is None or len(trades_df) == 0:
394:             return """
395:             <div class="section">
396:                 <h2>Trade Analysis</h2>
397:                 <div class="info">No trades found in this backtest.</div>
398:             </div>
399:             """
400: 
401:         # Get first few trades for display
402:         display_trades = trades_df.head(20).to_dicts()
403: 
404:         trades_html = ""
405:         for trade in display_trades:
406:             trades_html += f"""
407:             <tr>
408:                 <td>{trade["timestamp"]}</td>
409:                 <td>{trade["asset_id"]}</td>
410:                 <td>{trade["side"].upper()}</td>
411:                 <td>{trade["quantity"]:.2f}</td>
412:                 <td>${trade["price"]:.2f}</td>
413:                 <td>${trade["commission"]:.2f}</td>
414:                 <td>${trade["total_cost"]:.2f}</td>
415:             </tr>
416:             """
417: 
418:         return f"""
419:         <div class="section">
420:             <h2>Trade Analysis</h2>
421: 
422:             <div class="info">
423:                 Showing first 20 trades out of {len(trades_df)} total trades.
424:             </div>
425: 
426:             <div class="table-container">
427:                 <table>
428:                     <thead>
429:                         <tr>
430:                             <th>Timestamp</th>
431:                             <th>Asset</th>
432:                             <th>Side</th>
433:                             <th>Quantity</th>
434:                             <th>Price</th>
435:                             <th>Commission</th>
436:                             <th>Total Cost</th>
437:                         </tr>
438:                     </thead>
439:                     <tbody>
440:                         {trades_html}
441:                     </tbody>
442:                 </table>
443:             </div>
444:         </div>
445:         """
446: 
447:     def _generate_positions_section(self, report_data: dict[str, Any]) -> str:
448:         """Generate current positions section."""
449:         positions_df = report_data.get("positions")
450: 
451:         if positions_df is None or len(positions_df) == 0:
452:             return """
453:             <div class="section">
454:                 <h2>Current Positions</h2>
455:                 <div class="info">No open positions at end of backtest.</div>
456:             </div>
457:             """
458: 
459:         positions_html = ""
460:         for pos in positions_df.to_dicts():
461:             positions_html += f"""
462:             <tr>
463:                 <td>{pos["asset_id"]}</td>
464:                 <td>{pos["quantity"]:.2f}</td>
465:                 <td>${pos["cost_basis"]:.2f}</td>
466:                 <td>${pos["last_price"]:.2f}</td>
467:                 <td>${pos["market_value"]:.2f}</td>
468:                 <td class="{"positive" if pos["unrealized_pnl"] >= 0 else "negative"}">${pos["unrealized_pnl"]:.2f}</td>
469:                 <td class="{"positive" if pos["total_pnl"] >= 0 else "negative"}">${pos["total_pnl"]:.2f}</td>
470:             </tr>
471:             """
472: 
473:         return f"""
474:         <div class="section">
475:             <h2>Current Positions</h2>
476: 
477:             <div class="table-container">
478:                 <table>
479:                     <thead>
480:                         <tr>
481:                             <th>Asset</th>
482:                             <th>Quantity</th>
483:                             <th>Cost Basis</th>
484:                             <th>Last Price</th>
485:                             <th>Market Value</th>
486:                             <th>Unrealized P&L</th>
487:                             <th>Total P&L</th>
488:                         </tr>
489:                     </thead>
490:                     <tbody>
491:                         {positions_html}
492:                     </tbody>
493:                 </table>
494:             </div>
495:         </div>
496:         """
497: 
498:     def _generate_risk_section(self, report_data: dict[str, Any]) -> str:
499:         """Generate risk metrics section."""
500:         risk = report_data.get("risk", {})
501: 
502:         return f"""
503:         <div class="section">
504:             <h2>Risk Metrics</h2>
505: 
506:             <div class="metrics-grid">
507:                 <div class="metric-card">
508:                     <div class="metric-label">Max Leverage</div>
509:                     <div class="metric-value">
510:                         {risk.get("max_leverage", 1.0):.2f}x
511:                     </div>
512:                 </div>
513: 
514:                 <div class="metric-card">
515:                     <div class="metric-label">Max Concentration</div>
516:                     <div class="metric-value">
517:                         {risk.get("max_concentration", 0.0):.2%}
518:                     </div>
519:                 </div>
520:             </div>
521: 
522:             <div class="warning">
523:                 <strong>Risk Disclaimer:</strong> Past performance does not guarantee future results.
524:                 All trading involves risk of loss.
525:             </div>
526:         </div>
527:         """
528: 
529:     def _generate_footer(self, report_data: dict[str, Any]) -> str:
530:         """Generate report footer."""
531:         return f"""
532:         <div class="footer">
533:             Report generated by QEngine Backtesting Framework<br>
534:             Generated at: {report_data["metadata"]["generated_at"]}
535:         </div>
536:         """
537: 
538:     def _generate_javascript(self, report_data: dict[str, Any]) -> str:
539:         """Generate JavaScript for interactive charts."""
540:         equity_df = report_data.get("equity_curve")
541: 
542:         if equity_df is None or len(equity_df) == 0:
543:             return "// No data available for charts"
544: 
545:         # Convert DataFrame to JSON for JavaScript
546:         equity_data = equity_df.to_dicts()
547: 
548:         # Convert datetime objects to strings for JSON serialization
549:         for item in equity_data:
550:             if "timestamp" in item:
551:                 item["timestamp"] = item["timestamp"].isoformat()
552: 
553:         return f"""
554:         // Equity curve chart
555:         const equityData = {json.dumps(equity_data)};
556: 
557:         const equityTrace = {{
558:             x: equityData.map(d => d.timestamp),
559:             y: equityData.map(d => d.equity),
560:             type: 'scatter',
561:             mode: 'lines',
562:             name: 'Equity',
563:             line: {{
564:                 color: '#667eea',
565:                 width: 2
566:             }}
567:         }};
568: 
569:         const equityLayout = {{
570:             title: 'Equity Curve',
571:             xaxis: {{ title: 'Date' }},
572:             yaxis: {{ title: 'Portfolio Value ($)' }},
573:             margin: {{ t: 50 }}
574:         }};
575: 
576:         Plotly.newPlot('equity-curve-chart', [equityTrace], equityLayout);
577: 
578:         // Returns chart
579:         const returnsTrace = {{
580:             x: equityData.map(d => d.timestamp).slice(1),
581:             y: equityData.map(d => d.returns).slice(1),
582:             type: 'scatter',
583:             mode: 'markers',
584:             name: 'Daily Returns',
585:             marker: {{
586:                 color: equityData.map(d => d.returns > 0 ? '#28a745' : '#dc3545').slice(1),
587:                 size: 4
588:             }}
589:         }};
590: 
591:         const returnsLayout = {{
592:             title: 'Daily Returns Distribution',
593:             xaxis: {{ title: 'Date' }},
594:             yaxis: {{ title: 'Daily Return' }},
595:             margin: {{ t: 50 }}
596:         }};
597: 
598:         Plotly.newPlot('returns-chart', [returnsTrace], returnsLayout);
599:         """
````

## File: src/qengine/reporting/parquet.py
````python
  1: """Parquet report generation for QEngine backtests."""
  2: 
  3: import json
  4: from pathlib import Path
  5: from typing import Any
  6: 
  7: import polars as pl
  8: 
  9: from qengine.portfolio.accounting import PortfolioAccounting
 10: from qengine.reporting.base import ReportGenerator
 11: 
 12: 
 13: class ParquetReportGenerator(ReportGenerator):
 14:     """
 15:     Generates Parquet-based reports for backtest results.
 16: 
 17:     Creates structured data files optimized for:
 18:     - Data science workflows
 19:     - Further analysis with Polars/Pandas
 20:     - Integration with data pipelines
 21:     - Long-term storage and archival
 22:     """
 23: 
 24:     def generate(
 25:         self,
 26:         accounting: PortfolioAccounting,
 27:         strategy_params: dict[str, Any] | None = None,
 28:         backtest_params: dict[str, Any] | None = None,
 29:     ) -> Path:
 30:         """
 31:         Generate Parquet report from portfolio accounting data.
 32: 
 33:         Args:
 34:             accounting: Portfolio accounting with results
 35:             strategy_params: Strategy configuration parameters
 36:             backtest_params: Backtest configuration parameters
 37: 
 38:         Returns:
 39:             Path to generated report directory
 40:         """
 41:         # Create report directory
 42:         report_dir = self.output_dir / f"{self.report_name}_parquet"
 43:         report_dir.mkdir(exist_ok=True)
 44: 
 45:         # Prepare report data
 46:         report_data = self._prepare_report_data(accounting, strategy_params, backtest_params)
 47: 
 48:         # Save metadata as JSON
 49:         self._save_metadata(report_data, report_dir)
 50: 
 51:         # Save performance metrics
 52:         self._save_performance_metrics(report_data, report_dir)
 53: 
 54:         # Save time series data
 55:         self._save_equity_curve(report_data, report_dir)
 56: 
 57:         # Save trades data
 58:         self._save_trades(report_data, report_dir)
 59: 
 60:         # Save positions data
 61:         self._save_positions(report_data, report_dir)
 62: 
 63:         # Create summary file
 64:         self._create_summary_file(report_data, report_dir)
 65: 
 66:         return report_dir
 67: 
 68:     def _save_metadata(self, report_data: dict[str, Any], report_dir: Path) -> None:
 69:         """Save metadata and configuration as JSON."""
 70:         metadata = {
 71:             "report_info": report_data["metadata"],
 72:             "backtest_config": {
 73:                 "strategy_params": report_data["metadata"].get("strategy_params", {}),
 74:                 "backtest_params": report_data["metadata"].get("backtest_params", {}),
 75:             },
 76:             "portfolio_config": report_data["portfolio"],
 77:             "file_manifest": {
 78:                 "metadata": "metadata.json",
 79:                 "performance": "performance_metrics.parquet",
 80:                 "equity_curve": "equity_curve.parquet",
 81:                 "trades": "trades.parquet",
 82:                 "positions": "positions.parquet",
 83:                 "summary": "summary.parquet",
 84:             },
 85:         }
 86: 
 87:         metadata_path = report_dir / "metadata.json"
 88:         with open(metadata_path, "w") as f:
 89:             json.dump(metadata, f, indent=2, default=str)
 90: 
 91:     def _save_performance_metrics(self, report_data: dict[str, Any], report_dir: Path) -> None:
 92:         """Save performance metrics as Parquet."""
 93:         # Combine all performance data
 94:         metrics_data = []
 95: 
 96:         # Performance metrics
 97:         for key, value in report_data["performance"].items():
 98:             metrics_data.append(
 99:                 {
100:                     "category": "performance",
101:                     "metric": key,
102:                     "value": float(value) if isinstance(value, (int, float)) else str(value),
103:                     "format": "percentage"
104:                     if "return" in key or "rate" in key
105:                     else "currency"
106:                     if "pnl" in key
107:                     else "number",
108:                 },
109:             )
110: 
111:         # Cost metrics
112:         for key, value in report_data["costs"].items():
113:             metrics_data.append(
114:                 {
115:                     "category": "costs",
116:                     "metric": key,
117:                     "value": float(value) if isinstance(value, (int, float)) else str(value),
118:                     "format": "currency" if "commission" in key or "slippage" in key else "number",
119:                 },
120:             )
121: 
122:         # Risk metrics
123:         for key, value in report_data["risk"].items():
124:             metrics_data.append(
125:                 {
126:                     "category": "risk",
127:                     "metric": key,
128:                     "value": float(value) if isinstance(value, (int, float)) else str(value),
129:                     "format": "percentage" if "concentration" in key else "number",
130:                 },
131:             )
132: 
133:         # Portfolio metrics
134:         for key, value in report_data["portfolio"].items():
135:             metrics_data.append(
136:                 {
137:                     "category": "portfolio",
138:                     "metric": key,
139:                     "value": float(value) if isinstance(value, (int, float)) else str(value),
140:                     "format": "currency" if "cash" in key or "equity" in key else "number",
141:                 },
142:             )
143: 
144:         metrics_df = pl.DataFrame(metrics_data)
145:         metrics_path = report_dir / "performance_metrics.parquet"
146:         metrics_df.write_parquet(metrics_path)
147: 
148:     def _save_equity_curve(self, report_data: dict[str, Any], report_dir: Path) -> None:
149:         """Save equity curve as Parquet."""
150:         equity_df = report_data.get("equity_curve")
151: 
152:         if equity_df is not None and len(equity_df) > 0:
153:             # Add derived metrics
154:             enhanced_df = equity_df.with_columns(
155:                 [
156:                     # Cumulative returns
157:                     pl.col("returns").cum_sum().alias("cumulative_returns"),
158:                     # Running maximum for drawdown calculation
159:                     pl.col("equity").cum_max().alias("running_max"),
160:                     # Drawdown
161:                     ((pl.col("equity") / pl.col("equity").cum_max()) - 1).alias("drawdown"),
162:                     # Volatility (rolling 30-day)
163:                     pl.col("returns").rolling_std(window_size=30).alias("rolling_volatility_30d"),
164:                     # Rolling Sharpe (annualized)
165:                     (
166:                         pl.col("returns").rolling_mean(window_size=30)
167:                         / pl.col("returns").rolling_std(window_size=30)
168:                         * (252**0.5)
169:                     ).alias("rolling_sharpe_30d"),
170:                 ],
171:             )
172: 
173:             equity_path = report_dir / "equity_curve.parquet"
174:             enhanced_df.write_parquet(equity_path)
175:         else:
176:             # Create empty DataFrame with schema
177:             empty_df = pl.DataFrame(
178:                 {
179:                     "timestamp": [],
180:                     "equity": [],
181:                     "returns": [],
182:                     "cumulative_returns": [],
183:                     "running_max": [],
184:                     "drawdown": [],
185:                     "rolling_volatility_30d": [],
186:                     "rolling_sharpe_30d": [],
187:                 },
188:                 schema={
189:                     "timestamp": pl.Datetime,
190:                     "equity": pl.Float64,
191:                     "returns": pl.Float64,
192:                     "cumulative_returns": pl.Float64,
193:                     "running_max": pl.Float64,
194:                     "drawdown": pl.Float64,
195:                     "rolling_volatility_30d": pl.Float64,
196:                     "rolling_sharpe_30d": pl.Float64,
197:                 },
198:             )
199:             empty_df.write_parquet(report_dir / "equity_curve.parquet")
200: 
201:     def _save_trades(self, report_data: dict[str, Any], report_dir: Path) -> None:
202:         """Save trades data as Parquet."""
203:         trades_df = report_data.get("trades")
204: 
205:         if trades_df is not None and len(trades_df) > 0:
206:             # Add derived columns for analysis
207:             enhanced_df = trades_df.with_columns(
208:                 [
209:                     # Notional value
210:                     (pl.col("quantity") * pl.col("price")).alias("notional_value"),
211:                     # Commission rate
212:                     (pl.col("commission") / (pl.col("quantity") * pl.col("price"))).alias(
213:                         "commission_rate",
214:                     ),
215:                     # Slippage rate
216:                     (pl.col("slippage") / (pl.col("quantity") * pl.col("price"))).alias(
217:                         "slippage_rate",
218:                     ),
219:                     # Trade direction
220:                     pl.when(pl.col("side") == "buy").then(1).otherwise(-1).alias("direction"),
221:                     # Time-based features
222:                     pl.col("timestamp").dt.hour().alias("hour_of_day"),
223:                     pl.col("timestamp").dt.day().alias("day_of_month"),
224:                     pl.col("timestamp").dt.weekday().alias("day_of_week"),
225:                     # Size categories
226:                     pl.when(pl.col("quantity") * pl.col("price") < 1000)
227:                     .then(pl.lit("small"))
228:                     .when(pl.col("quantity") * pl.col("price") < 10000)
229:                     .then(pl.lit("medium"))
230:                     .otherwise(pl.lit("large"))
231:                     .alias("trade_size_category"),
232:                 ],
233:             )
234: 
235:             trades_path = report_dir / "trades.parquet"
236:             enhanced_df.write_parquet(trades_path)
237:         else:
238:             # Create empty DataFrame with schema
239:             empty_df = pl.DataFrame(
240:                 {
241:                     "timestamp": [],
242:                     "order_id": [],
243:                     "trade_id": [],
244:                     "asset_id": [],
245:                     "side": [],
246:                     "quantity": [],
247:                     "price": [],
248:                     "commission": [],
249:                     "slippage": [],
250:                     "total_cost": [],
251:                     "notional_value": [],
252:                     "commission_rate": [],
253:                     "slippage_rate": [],
254:                     "direction": [],
255:                     "hour_of_day": [],
256:                     "day_of_month": [],
257:                     "day_of_week": [],
258:                     "trade_size_category": [],
259:                 },
260:                 schema={
261:                     "timestamp": pl.Datetime,
262:                     "order_id": pl.Utf8,
263:                     "trade_id": pl.Utf8,
264:                     "asset_id": pl.Utf8,
265:                     "side": pl.Utf8,
266:                     "quantity": pl.Float64,
267:                     "price": pl.Float64,
268:                     "commission": pl.Float64,
269:                     "slippage": pl.Float64,
270:                     "total_cost": pl.Float64,
271:                     "notional_value": pl.Float64,
272:                     "commission_rate": pl.Float64,
273:                     "slippage_rate": pl.Float64,
274:                     "direction": pl.Int8,
275:                     "hour_of_day": pl.UInt32,
276:                     "day_of_month": pl.UInt32,
277:                     "day_of_week": pl.UInt32,
278:                     "trade_size_category": pl.Utf8,
279:                 },
280:             )
281:             empty_df.write_parquet(report_dir / "trades.parquet")
282: 
283:     def _save_positions(self, report_data: dict[str, Any], report_dir: Path) -> None:
284:         """Save positions data as Parquet."""
285:         positions_df = report_data.get("positions")
286: 
287:         if positions_df is not None and len(positions_df) > 0:
288:             # Add derived columns
289:             enhanced_df = positions_df.with_columns(
290:                 [
291:                     # Position direction
292:                     pl.when(pl.col("quantity") > 0)
293:                     .then(pl.lit("long"))
294:                     .when(pl.col("quantity") < 0)
295:                     .then(pl.lit("short"))
296:                     .otherwise(pl.lit("flat"))
297:                     .alias("position_type"),
298:                     # Average cost per share
299:                     (pl.col("cost_basis") / pl.col("quantity")).alias("avg_cost_per_share"),
300:                     # Unrealized return percentage
301:                     (pl.col("unrealized_pnl") / pl.col("cost_basis")).alias(
302:                         "unrealized_return_pct",
303:                     ),
304:                     # Total return percentage
305:                     (pl.col("total_pnl") / pl.col("cost_basis")).alias("total_return_pct"),
306:                     # Position weight (would need portfolio value for this)
307:                     pl.col("market_value").alias("position_weight_placeholder"),
308:                 ],
309:             )
310: 
311:             positions_path = report_dir / "positions.parquet"
312:             enhanced_df.write_parquet(positions_path)
313:         else:
314:             # Create empty DataFrame with schema
315:             empty_df = pl.DataFrame(
316:                 {
317:                     "asset_id": [],
318:                     "quantity": [],
319:                     "cost_basis": [],
320:                     "last_price": [],
321:                     "market_value": [],
322:                     "unrealized_pnl": [],
323:                     "realized_pnl": [],
324:                     "total_pnl": [],
325:                     "position_type": [],
326:                     "avg_cost_per_share": [],
327:                     "unrealized_return_pct": [],
328:                     "total_return_pct": [],
329:                     "position_weight_placeholder": [],
330:                 },
331:                 schema={
332:                     "asset_id": pl.Utf8,
333:                     "quantity": pl.Float64,
334:                     "cost_basis": pl.Float64,
335:                     "last_price": pl.Float64,
336:                     "market_value": pl.Float64,
337:                     "unrealized_pnl": pl.Float64,
338:                     "realized_pnl": pl.Float64,
339:                     "total_pnl": pl.Float64,
340:                     "position_type": pl.Utf8,
341:                     "avg_cost_per_share": pl.Float64,
342:                     "unrealized_return_pct": pl.Float64,
343:                     "total_return_pct": pl.Float64,
344:                     "position_weight_placeholder": pl.Float64,
345:                 },
346:             )
347:             empty_df.write_parquet(report_dir / "positions.parquet")
348: 
349:     def _create_summary_file(self, report_data: dict[str, Any], report_dir: Path) -> None:
350:         """Create a summary file with key statistics."""
351:         summary_data = [
352:             {
353:                 "report_name": report_data["metadata"]["report_name"],
354:                 "generated_at": report_data["metadata"]["generated_at"],
355:                 "total_return": report_data["performance"]["total_return"],
356:                 "total_pnl": report_data["performance"]["total_pnl"],
357:                 "sharpe_ratio": report_data["performance"].get("sharpe_ratio", None),
358:                 "max_drawdown": report_data["performance"]["max_drawdown"],
359:                 "win_rate": report_data["performance"]["win_rate"],
360:                 "num_trades": report_data["performance"]["num_trades"],
361:                 "total_commission": report_data["costs"]["total_commission"],
362:                 "total_slippage": report_data["costs"]["total_slippage"],
363:                 "initial_capital": report_data["portfolio"]["initial_cash"],
364:                 "final_equity": report_data["portfolio"]["final_equity"],
365:                 "max_leverage": report_data["risk"]["max_leverage"],
366:                 "max_concentration": report_data["risk"]["max_concentration"],
367:             },
368:         ]
369: 
370:         summary_df = pl.DataFrame(summary_data)
371:         summary_path = report_dir / "summary.parquet"
372:         summary_df.write_parquet(summary_path)
373: 
374:     def load_report(self, report_dir: Path) -> dict[str, Any]:
375:         """
376:         Load a previously generated Parquet report.
377: 
378:         Args:
379:             report_dir: Directory containing the Parquet report
380: 
381:         Returns:
382:             Dictionary with loaded report data
383:         """
384:         if not report_dir.exists():
385:             raise FileNotFoundError(f"Report directory not found: {report_dir}")
386: 
387:         # Load metadata
388:         metadata_path = report_dir / "metadata.json"
389:         with open(metadata_path) as f:
390:             metadata = json.load(f)
391: 
392:         # Load data files
393:         report_data = {
394:             "metadata": metadata,
395:             "performance_metrics": pl.read_parquet(report_dir / "performance_metrics.parquet"),
396:             "equity_curve": pl.read_parquet(report_dir / "equity_curve.parquet"),
397:             "trades": pl.read_parquet(report_dir / "trades.parquet"),
398:             "positions": pl.read_parquet(report_dir / "positions.parquet"),
399:             "summary": pl.read_parquet(report_dir / "summary.parquet"),
400:         }
401: 
402:         return report_data
````

## File: src/qengine/reporting/reporter.py
````python
  1: """Reporter implementations for capturing backtest events and results."""
  2: 
  3: import logging
  4: from datetime import datetime
  5: from typing import Any, Optional
  6: 
  7: from qengine.core.event import Event
  8: 
  9: logger = logging.getLogger(__name__)
 10: 
 11: 
 12: class Reporter:
 13:     """Abstract base class for reporters."""
 14: 
 15:     def on_start(self) -> None:
 16:         """Called at start of backtest."""
 17: 
 18:     def on_event(self, event: Event) -> None:
 19:         """Called for each event processed."""
 20: 
 21:     def on_end(self) -> None:
 22:         """Called at end of backtest."""
 23: 
 24:     def reset(self) -> None:
 25:         """Reset reporter state."""
 26: 
 27:     def get_report(self) -> Any:
 28:         """Get the generated report."""
 29: 
 30: 
 31: class InMemoryReporter(Reporter):
 32:     """Reporter that stores all events and results in memory.
 33: 
 34:     This reporter captures:
 35:     - All events processed during backtest
 36:     - Timestamps and event counts
 37:     - Summary statistics
 38: 
 39:     Useful for debugging and analysis of backtest execution.
 40:     """
 41: 
 42:     def __init__(self, capture_all_events: bool = False):
 43:         """Initialize in-memory reporter.
 44: 
 45:         Args:
 46:             capture_all_events: If True, store all events (can use significant memory)
 47:         """
 48:         self.capture_all_events = capture_all_events
 49:         self.events = []
 50:         self.event_counts = {}
 51:         self.start_time: datetime | None = None
 52:         self.end_time: datetime | None = None
 53:         self.first_event_time: datetime | None = None
 54:         self.last_event_time: datetime | None = None
 55: 
 56:     def on_start(self) -> None:
 57:         """Mark start of backtest."""
 58:         self.start_time = datetime.now()
 59:         logger.debug("InMemoryReporter started")
 60: 
 61:     def on_event(self, event: Event) -> None:
 62:         """Capture event details.
 63: 
 64:         Args:
 65:             event: Event to record
 66:         """
 67:         # Track event counts by type
 68:         event_type = (
 69:             event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
 70:         )
 71:         self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
 72: 
 73:         # Track first and last event timestamps
 74:         if self.first_event_time is None:
 75:             self.first_event_time = event.timestamp
 76:         self.last_event_time = event.timestamp
 77: 
 78:         # Optionally store full event
 79:         if self.capture_all_events:
 80:             self.events.append(
 81:                 {
 82:                     "timestamp": event.timestamp,
 83:                     "type": event_type,
 84:                     "event": event,
 85:                 },
 86:             )
 87: 
 88:     def on_end(self) -> None:
 89:         """Mark end of backtest."""
 90:         self.end_time = datetime.now()
 91:         logger.debug(f"InMemoryReporter finished. Total events: {sum(self.event_counts.values())}")
 92: 
 93:     def get_report(self) -> dict[str, Any]:
 94:         """Get summary report of captured events.
 95: 
 96:         Returns:
 97:             Dictionary with event statistics and timing information
 98:         """
 99:         total_events = sum(self.event_counts.values())
100:         duration = (
101:             (self.end_time - self.start_time).total_seconds()
102:             if self.end_time and self.start_time
103:             else 0
104:         )
105: 
106:         report = {
107:             "summary": {
108:                 "total_events": total_events,
109:                 "event_types": len(self.event_counts),
110:                 "duration_seconds": duration,
111:                 "events_per_second": total_events / duration if duration > 0 else 0,
112:             },
113:             "event_counts": self.event_counts,
114:             "timing": {
115:                 "start_time": self.start_time.isoformat() if self.start_time else None,
116:                 "end_time": self.end_time.isoformat() if self.end_time else None,
117:                 "first_event": self.first_event_time.isoformat() if self.first_event_time else None,
118:                 "last_event": self.last_event_time.isoformat() if self.last_event_time else None,
119:             },
120:         }
121: 
122:         if self.capture_all_events:
123:             report["events"] = self.events
124: 
125:         # Add event type breakdown
126:         if self.event_counts:
127:             report["breakdown"] = {
128:                 event_type: {
129:                     "count": count,
130:                     "percentage": (count / total_events * 100) if total_events > 0 else 0,
131:                 }
132:                 for event_type, count in self.event_counts.items()
133:             }
134: 
135:         return report
136: 
137:     def reset(self) -> None:
138:         """Reset reporter to initial state."""
139:         self.events.clear()
140:         self.event_counts.clear()
141:         self.start_time = None
142:         self.end_time = None
143:         self.first_event_time = None
144:         self.last_event_time = None
145: 
146: 
147: class ConsoleReporter(Reporter):
148:     """Reporter that logs events to console.
149: 
150:     Useful for real-time monitoring of backtest progress.
151:     """
152: 
153:     def __init__(self, log_level: str = "INFO", log_every_n: int = 1000):
154:         """Initialize console reporter.
155: 
156:         Args:
157:             log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
158:             log_every_n: Log summary every N events
159:         """
160:         self.log_level = getattr(logging, log_level.upper(), logging.INFO)
161:         self.log_every_n = log_every_n
162:         self.event_count = 0
163:         self.event_counts = {}
164: 
165:     def on_start(self) -> None:
166:         """Log backtest start."""
167:         logger.log(self.log_level, "=" * 60)
168:         logger.log(self.log_level, "BACKTEST STARTED")
169:         logger.log(self.log_level, "=" * 60)
170: 
171:     def on_event(self, event: Event) -> None:
172:         """Log event if appropriate.
173: 
174:         Args:
175:             event: Event to potentially log
176:         """
177:         self.event_count += 1
178:         event_type = (
179:             event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
180:         )
181:         self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
182: 
183:         # Log periodic summary
184:         if self.event_count % self.log_every_n == 0:
185:             logger.log(
186:                 self.log_level,
187:                 f"Processed {self.event_count:,} events. "
188:                 f"Latest: {event.timestamp} | "
189:                 f"Types: {dict(sorted(self.event_counts.items()))}",
190:             )
191: 
192:     def on_end(self) -> None:
193:         """Log backtest completion."""
194:         logger.log(self.log_level, "=" * 60)
195:         logger.log(self.log_level, "BACKTEST COMPLETED")
196:         logger.log(self.log_level, f"Total Events: {self.event_count:,}")
197:         logger.log(self.log_level, f"Event Breakdown: {dict(sorted(self.event_counts.items()))}")
198:         logger.log(self.log_level, "=" * 60)
199: 
200:     def reset(self) -> None:
201:         """Reset reporter state."""
202:         self.event_count = 0
203:         self.event_counts.clear()
204: 
205:     def get_report(self) -> dict[str, Any]:
206:         """Get simple report for console reporter.
207: 
208:         Returns:
209:             Event count summary
210:         """
211:         return {
212:             "total_events": self.event_count,
213:             "event_counts": self.event_counts,
214:         }
215: 
216: 
217: __all__ = [
218:     "ConsoleReporter",
219:     "InMemoryReporter",
220:     "Reporter",
221: ]
````

## File: src/qengine/strategy/__init__.py
````python
 1: """Strategy framework for QEngine."""
 2: 
 3: from qengine.strategy.adapters import (
 4:     DataFrameAdapter,
 5:     ExternalStrategyInterface,
 6:     PITData,
 7:     StrategyAdapter,
 8:     StrategySignal,
 9: )
10: from qengine.strategy.base import Strategy
11: from qengine.strategy.crypto_basis_adapter import (
12:     CryptoBasisAdapter,
13:     CryptoBasisExternalStrategy,
14:     create_crypto_basis_strategy,
15: )
16: 
17: __all__ = [
18:     "CryptoBasisAdapter",
19:     "CryptoBasisExternalStrategy",
20:     "DataFrameAdapter",
21:     "ExternalStrategyInterface",
22:     "PITData",
23:     "Strategy",
24:     "StrategyAdapter",
25:     "StrategySignal",
26:     "create_crypto_basis_strategy",
27: ]
````

## File: src/qengine/strategy/adapters.py
````python
  1: """
  2: Strategy-QEngine Integration Bridge
  3: ==================================
  4: 
  5: This module provides adapters that bridge external strategy implementations
  6: to QEngine's event-driven architecture. It allows existing strategies to run
  7: within QEngine and benefit from advanced execution models.
  8: """
  9: 
 10: from abc import ABC, abstractmethod
 11: from dataclasses import dataclass
 12: from datetime import datetime
 13: from typing import Any, Callable, Optional
 14: 
 15: import polars as pl
 16: 
 17: from qengine.core.event import Event, FillEvent, MarketEvent
 18: from qengine.core.types import AssetId, OrderSide, OrderType, Price
 19: from qengine.execution.order import Order
 20: from qengine.strategy.base import Strategy
 21: 
 22: 
 23: @dataclass
 24: class PITData:
 25:     """Point-in-time data snapshot for strategy decision making."""
 26: 
 27:     timestamp: datetime
 28:     asset_data: dict[AssetId, dict[str, Any]]
 29:     market_prices: dict[AssetId, Price]
 30: 
 31:     def get_price(self, asset_id: AssetId) -> Price | None:
 32:         """Get current price for asset."""
 33:         return self.market_prices.get(asset_id)
 34: 
 35:     def get_data(self, asset_id: AssetId, field: str) -> Any:
 36:         """Get specific field for asset."""
 37:         return self.asset_data.get(asset_id, {}).get(field)
 38: 
 39: 
 40: @dataclass
 41: class StrategySignal:
 42:     """Signal generated by external strategy."""
 43: 
 44:     timestamp: datetime
 45:     asset_id: AssetId
 46:     position: float  # Target position (-1 to 1, or absolute quantities)
 47:     confidence: float = 0.0
 48:     metadata: dict[str, Any] = None
 49: 
 50:     def __post_init__(self):
 51:         if self.metadata is None:
 52:             self.metadata = {}
 53: 
 54: 
 55: class ExternalStrategyInterface(ABC):
 56:     """
 57:     Interface that external strategies must implement to integrate with QEngine.
 58: 
 59:     This defines the minimal API needed for QEngine integration.
 60:     """
 61: 
 62:     @abstractmethod
 63:     def generate_signal(
 64:         self,
 65:         timestamp: datetime,
 66:         pit_data: PITData,
 67:     ) -> StrategySignal | None:
 68:         """
 69:         Generate trading signal based on point-in-time data.
 70: 
 71:         Args:
 72:             timestamp: Current timestamp
 73:             pit_data: Point-in-time data snapshot
 74: 
 75:         Returns:
 76:             Trading signal or None if no action needed
 77:         """
 78: 
 79:     @abstractmethod
 80:     def initialize(self) -> None:
 81:         """Initialize strategy state."""
 82: 
 83:     @abstractmethod
 84:     def finalize(self) -> None:
 85:         """Cleanup strategy state."""
 86: 
 87: 
 88: class StrategyAdapter(Strategy):
 89:     """
 90:     Base adapter that bridges external strategies to QEngine.
 91: 
 92:     This class handles the translation between QEngine's event-driven architecture
 93:     and external strategy APIs.
 94:     """
 95: 
 96:     def __init__(
 97:         self,
 98:         external_strategy: ExternalStrategyInterface,
 99:         position_sizer: Callable[[StrategySignal, float], float] | None = None,
100:         risk_manager: Callable[[StrategySignal], bool] | None = None,
101:         name: str | None = None,
102:     ):
103:         """
104:         Initialize strategy adapter.
105: 
106:         Args:
107:             external_strategy: External strategy implementation
108:             position_sizer: Optional position sizing function
109:             risk_manager: Optional risk management function
110:             name: Strategy name
111:         """
112:         super().__init__(name=name or f"Adapter_{external_strategy.__class__.__name__}")
113:         self.external_strategy = external_strategy
114:         self.position_sizer = position_sizer or self._default_position_sizer
115:         self.risk_manager = risk_manager or self._default_risk_manager
116: 
117:         # State tracking
118:         self._data_history: dict[AssetId, list[dict]] = {}
119:         self._last_signals: dict[AssetId, StrategySignal] = {}
120:         self._target_positions: dict[AssetId, float] = {}
121:         self._pending_orders: dict[AssetId, list[Order]] = {}
122: 
123:     def on_start(self) -> None:
124:         """Initialize external strategy."""
125:         self.log("Starting strategy adapter")
126:         self.external_strategy.initialize()
127: 
128:     def on_end(self) -> None:
129:         """Cleanup external strategy."""
130:         self.log("Stopping strategy adapter")
131:         self.external_strategy.finalize()
132: 
133:     def on_event(self, event: Event) -> None:
134:         """Route events to appropriate handlers."""
135:         if isinstance(event, MarketEvent):
136:             self.on_market_event(event)
137:         elif isinstance(event, FillEvent):
138:             self.on_fill_event(event)
139: 
140:     def on_market_event(self, event: MarketEvent) -> None:
141:         """Process market event and generate signals."""
142:         try:
143:             # Update data history
144:             self._update_data_history(event)
145: 
146:             # Create point-in-time data snapshot
147:             pit_data = self._create_pit_data(event.timestamp)
148: 
149:             # Generate signal from external strategy
150:             signal = self.external_strategy.generate_signal(event.timestamp, pit_data)
151: 
152:             if signal:
153:                 self._process_signal(signal)
154: 
155:         except Exception as e:
156:             self.log(f"Error processing market event: {e}", level="ERROR")
157: 
158:     def on_fill_event(self, event: FillEvent) -> None:
159:         """Update position tracking when fills occur."""
160:         super().on_fill_event(event)
161: 
162:         # Remove filled orders from pending
163:         if event.asset_id in self._pending_orders:
164:             self._pending_orders[event.asset_id] = [
165:                 order
166:                 for order in self._pending_orders[event.asset_id]
167:                 if order.order_id != event.order_id
168:             ]
169: 
170:         self.log(f"Fill received: {event.asset_id} {event.fill_quantity} @ {event.fill_price}")
171: 
172:     def _update_data_history(self, event: MarketEvent) -> None:
173:         """Update historical data for strategy calculations."""
174:         if event.asset_id not in self._data_history:
175:             self._data_history[event.asset_id] = []
176: 
177:         data_point = {
178:             "timestamp": event.timestamp,
179:             "open": event.open,
180:             "high": event.high,
181:             "low": event.low,
182:             "close": event.close,
183:             "volume": event.volume,
184:         }
185: 
186:         self._data_history[event.asset_id].append(data_point)
187: 
188:         # Keep only last N points for memory efficiency
189:         if len(self._data_history[event.asset_id]) > 1000:
190:             self._data_history[event.asset_id] = self._data_history[event.asset_id][-500:]
191: 
192:     def _create_pit_data(self, timestamp: datetime) -> PITData:
193:         """Create point-in-time data snapshot."""
194:         asset_data = {}
195:         market_prices = {}
196: 
197:         for asset_id, history in self._data_history.items():
198:             if history:
199:                 # Get most recent data up to timestamp
200:                 valid_data = [d for d in history if d["timestamp"] <= timestamp]
201:                 if valid_data:
202:                     latest = valid_data[-1]
203:                     asset_data[asset_id] = latest
204:                     market_prices[asset_id] = latest["close"]
205: 
206:         return PITData(
207:             timestamp=timestamp,
208:             asset_data=asset_data,
209:             market_prices=market_prices,
210:         )
211: 
212:     def _process_signal(self, signal: StrategySignal) -> None:
213:         """Process signal and submit orders if needed."""
214:         # Store last signal
215:         self._last_signals[signal.asset_id] = signal
216: 
217:         # Apply risk management
218:         if not self.risk_manager(signal):
219:             self.log(f"Signal rejected by risk manager: {signal.asset_id}", level="WARNING")
220:             return
221: 
222:         # Calculate position size
223:         current_cash = self.broker.get_cash() if hasattr(self.broker, "get_cash") else 100000
224:         position_size = self.position_sizer(signal, current_cash)
225: 
226:         # Get current position
227:         current_position = self._positions.get(signal.asset_id, 0)
228: 
229:         # Calculate required trade
230:         trade_quantity = position_size - current_position
231: 
232:         if abs(trade_quantity) > 0.001:  # Minimum trade threshold
233:             self._submit_rebalance_order(signal.asset_id, trade_quantity)
234: 
235:     def _submit_rebalance_order(self, asset_id: AssetId, quantity: float) -> None:
236:         """Submit order to rebalance to target position."""
237:         if abs(quantity) < 0.001:
238:             return
239: 
240:         # Determine order side
241:         side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
242:         abs_quantity = abs(quantity)
243: 
244:         # Create market order for immediate execution
245:         order = Order(
246:             asset_id=asset_id,
247:             order_type=OrderType.MARKET,
248:             side=side,
249:             quantity=abs_quantity,
250:         )
251: 
252:         # Submit through broker
253:         if hasattr(self.broker, "submit_order"):
254:             order_id = self.broker.submit_order(order)
255: 
256:             # Track pending order
257:             if asset_id not in self._pending_orders:
258:                 self._pending_orders[asset_id] = []
259:             self._pending_orders[asset_id].append(order)
260: 
261:             self.log(f"Order submitted: {order_id} - {side.value} {abs_quantity} {asset_id}")
262:         else:
263:             self.log("No broker available for order submission", level="ERROR")
264: 
265:     def _default_position_sizer(self, signal: StrategySignal, cash: float) -> float:
266:         """Default position sizing based on signal position and confidence."""
267:         # Use signal position directly, scaled by confidence
268:         base_position = signal.position * signal.confidence
269: 
270:         # Simple position sizing - use portion of cash based on signal
271:         if abs(base_position) > 0:
272:             position_value = cash * 0.1  # 10% of cash
273:             return base_position * position_value  # Signed position value
274: 
275:         return 0.0
276: 
277:     def _default_risk_manager(self, signal: StrategySignal) -> bool:
278:         """Default risk management - always allow signals."""
279:         # Basic checks
280:         if not signal.asset_id:
281:             return False
282:         if abs(signal.position) > 10:  # Sanity check on position size
283:             return False
284:         return True
285: 
286:     def get_strategy_state(self) -> dict[str, Any]:
287:         """Get current strategy state for debugging."""
288:         return {
289:             "name": self.name,
290:             "current_positions": self.current_positions,
291:             "target_positions": self._target_positions.copy(),
292:             "last_signals": {k: v.__dict__ for k, v in self._last_signals.items()},
293:             "pending_orders": {k: len(v) for k, v in self._pending_orders.items()},
294:             "data_history_lengths": {k: len(v) for k, v in self._data_history.items()},
295:         }
296: 
297: 
298: class DataFrameAdapter(StrategyAdapter):
299:     """
300:     Adapter for strategies that work with DataFrame-based data.
301: 
302:     This adapter maintains a rolling DataFrame of market data that can be
303:     accessed by external strategies for calculations.
304:     """
305: 
306:     def __init__(
307:         self,
308:         external_strategy: ExternalStrategyInterface,
309:         window_size: int = 1000,
310:         **kwargs,
311:     ):
312:         """
313:         Initialize DataFrame adapter.
314: 
315:         Args:
316:             external_strategy: External strategy implementation
317:             window_size: Size of rolling data window
318:             **kwargs: Additional arguments for StrategyAdapter
319:         """
320:         super().__init__(external_strategy, **kwargs)
321:         self.window_size = window_size
322:         self._dataframes: dict[AssetId, pl.DataFrame] = {}
323: 
324:     def _update_data_history(self, event: MarketEvent) -> None:
325:         """Update DataFrame with new market data."""
326:         # Call parent to maintain backward compatibility
327:         super()._update_data_history(event)
328: 
329:         # Create new row
330:         new_row = pl.DataFrame(
331:             {
332:                 "timestamp": [event.timestamp],
333:                 "asset_id": [event.asset_id],
334:                 "open": [event.open],
335:                 "high": [event.high],
336:                 "low": [event.low],
337:                 "close": [event.close],
338:                 "volume": [event.volume],
339:             },
340:         )
341: 
342:         # Update or create DataFrame
343:         if event.asset_id not in self._dataframes:
344:             self._dataframes[event.asset_id] = new_row
345:         else:
346:             self._dataframes[event.asset_id] = pl.concat(
347:                 [
348:                     self._dataframes[event.asset_id],
349:                     new_row,
350:                 ],
351:             )
352: 
353:             # Maintain window size
354:             if len(self._dataframes[event.asset_id]) > self.window_size:
355:                 self._dataframes[event.asset_id] = self._dataframes[event.asset_id].tail(
356:                     self.window_size,
357:                 )
358: 
359:     def get_dataframe(self, asset_id: AssetId) -> pl.DataFrame | None:
360:         """Get DataFrame for asset."""
361:         return self._dataframes.get(asset_id)
362: 
363:     def get_all_dataframes(self) -> dict[AssetId, pl.DataFrame]:
364:         """Get all DataFrames."""
365:         return self._dataframes.copy()
````

## File: src/qengine/strategy/base.py
````python
  1: """Base strategy class and interfaces for QEngine."""
  2: 
  3: from abc import ABC, abstractmethod
  4: from dataclasses import dataclass, field
  5: from datetime import datetime
  6: from enum import Enum
  7: from typing import Any
  8: 
  9: from qengine.core.event import Event, FillEvent, MarketEvent, SignalEvent
 10: from qengine.core.types import AssetId, EventType, OrderSide
 11: 
 12: 
 13: class StrategyState(Enum):
 14:     """Strategy lifecycle states."""
 15: 
 16:     INITIALIZED = "initialized"
 17:     STARTING = "starting"
 18:     RUNNING = "running"
 19:     STOPPING = "stopping"
 20:     STOPPED = "stopped"
 21: 
 22: 
 23: @dataclass
 24: class StrategyContext:
 25:     """Context object containing strategy runtime information."""
 26: 
 27:     start_time: datetime
 28:     end_time: datetime
 29:     initial_capital: float
 30:     commission_model: Any | None = None
 31:     slippage_model: Any | None = None
 32:     data_feeds: list[Any] = field(default_factory=list)
 33:     parameters: dict[str, Any] = field(default_factory=dict)
 34: 
 35: 
 36: class Strategy(ABC):
 37:     """
 38:     Abstract base class for all trading strategies.
 39: 
 40:     This class defines the interface that all strategies must implement.
 41:     Strategies receive events through the on_event method and can submit
 42:     orders through the broker interface.
 43:     """
 44: 
 45:     def __init__(self, name: str | None = None):
 46:         """
 47:         Initialize the strategy.
 48: 
 49:         Args:
 50:             name: Optional name for the strategy
 51:         """
 52:         self.name = name or self.__class__.__name__
 53:         self.state = StrategyState.INITIALIZED
 54:         self.broker = None  # Will be injected by the engine
 55:         self.data = None  # PIT data accessor
 56:         self.context = None  # Strategy context
 57:         self._subscriptions: set[tuple] = set()
 58:         self._positions: dict[AssetId, float] = {}
 59:         self._orders: list[Any] = []
 60:         self._trades: list[Any] = []
 61: 
 62:     def on_start(self) -> None:
 63:         """
 64:         Called once when the strategy starts.
 65: 
 66:         Override this method to perform one-time initialization tasks like:
 67:         - Setting up indicators
 68:         - Subscribing to data feeds
 69:         - Initializing internal state
 70:         """
 71: 
 72:     @abstractmethod
 73:     def on_event(self, event: Event) -> None:
 74:         """
 75:         Process an event.
 76: 
 77:         This is the main method where strategy logic is implemented.
 78:         It's called for every event the strategy is subscribed to.
 79: 
 80:         Args:
 81:             event: The event to process
 82:         """
 83: 
 84:     def on_market_event(self, event: MarketEvent) -> None:
 85:         """
 86:         Process a market data event.
 87: 
 88:         Override this for specialized market data handling.
 89: 
 90:         Args:
 91:             event: Market data event
 92:         """
 93: 
 94:     def on_signal_event(self, event: SignalEvent) -> None:
 95:         """
 96:         Process an ML signal event.
 97: 
 98:         Override this for signal-based strategies.
 99: 
100:         Args:
101:             event: Signal event from ML model
102:         """
103: 
104:     def on_fill_event(self, event: FillEvent) -> None:
105:         """
106:         Process an order fill event.
107: 
108:         Override this to track fills and update internal state.
109: 
110:         Args:
111:             event: Fill event
112:         """
113:         # Default implementation updates position tracking
114:         if event.side in [OrderSide.BUY, OrderSide.COVER]:
115:             self._positions[event.asset_id] = (
116:                 self._positions.get(event.asset_id, 0) + event.fill_quantity
117:             )
118:         else:
119:             self._positions[event.asset_id] = (
120:                 self._positions.get(event.asset_id, 0) - event.fill_quantity
121:             )
122: 
123:     def on_end(self) -> None:
124:         """
125:         Called once when the strategy stops.
126: 
127:         Override this to perform cleanup tasks like:
128:         - Closing positions
129:         - Saving state
130:         - Final analysis
131:         """
132: 
133:     def before_trading_start(self) -> None:
134:         """
135:         Called before the start of each trading day.
136: 
137:         Override this for daily preparation tasks like:
138:         - Updating universe
139:         - Recalculating signals
140:         - Adjusting parameters
141:         """
142: 
143:     def after_trading_end(self) -> None:
144:         """
145:         Called after the end of each trading day.
146: 
147:         Override this for end-of-day tasks like:
148:         - Recording metrics
149:         - Rebalancing
150:         - Risk calculations
151:         """
152: 
153:     def subscribe(
154:         self,
155:         asset: AssetId | None = None,
156:         event_type: EventType | None = None,
157:         **kwargs,
158:     ) -> None:
159:         """
160:         Subscribe to specific events.
161: 
162:         Args:
163:             asset: Asset to subscribe to (None for all)
164:             event_type: Type of events to receive
165:             **kwargs: Additional subscription parameters
166:         """
167:         subscription = (asset, event_type, tuple(kwargs.items()))
168:         self._subscriptions.add(subscription)
169: 
170:     def unsubscribe(
171:         self,
172:         asset: AssetId | None = None,
173:         event_type: EventType | None = None,
174:         **kwargs,
175:     ) -> None:
176:         """
177:         Unsubscribe from specific events.
178: 
179:         Args:
180:             asset: Asset to unsubscribe from
181:             event_type: Type of events to stop receiving
182:             **kwargs: Additional parameters
183:         """
184:         subscription = (asset, event_type, tuple(kwargs.items()))
185:         self._subscriptions.discard(subscription)
186: 
187:     @property
188:     def current_positions(self) -> dict[AssetId, float]:
189:         """Get current position quantities by asset."""
190:         return self._positions.copy()
191: 
192:     @property
193:     def is_flat(self) -> bool:
194:         """Check if strategy has no positions."""
195:         return all(qty == 0 for qty in self._positions.values())
196: 
197:     def log(self, message: str, level: str = "INFO") -> None:
198:         """
199:         Log a message.
200: 
201:         Args:
202:             message: Message to log
203:             level: Log level (INFO, WARNING, ERROR, DEBUG)
204:         """
205:         timestamp = datetime.now().isoformat()
206:         print(f"[{timestamp}] [{self.name}] [{level}] {message}")
207: 
208:     def __repr__(self) -> str:
209:         return f"{self.__class__.__name__}(name='{self.name}', state={self.state})"
210: 
211: 
212: class SignalStrategy(Strategy):
213:     """
214:     Base class for signal-based strategies.
215: 
216:     This provides a simpler interface for strategies that primarily
217:     react to ML signals rather than raw market data.
218:     """
219: 
220:     def __init__(self, name: str | None = None, signal_threshold: float = 0.5):
221:         """
222:         Initialize signal strategy.
223: 
224:         Args:
225:             name: Strategy name
226:             signal_threshold: Threshold for acting on signals
227:         """
228:         super().__init__(name)
229:         self.signal_threshold = signal_threshold
230:         self._signal_history: dict[AssetId, list[float]] = {}
231: 
232:     def on_event(self, event: Event) -> None:
233:         """Route events to appropriate handlers."""
234:         if isinstance(event, SignalEvent):
235:             self.on_signal_event(event)
236:         elif isinstance(event, MarketEvent):
237:             self.on_market_event(event)
238:         elif isinstance(event, FillEvent):
239:             self.on_fill_event(event)
240: 
241:     def on_signal_event(self, event: SignalEvent) -> None:
242:         """
243:         Process signal and make trading decision.
244: 
245:         Args:
246:             event: Signal event
247:         """
248:         # Track signal history
249:         if event.asset_id not in self._signal_history:
250:             self._signal_history[event.asset_id] = []
251:         self._signal_history[event.asset_id].append(event.signal_value)
252: 
253:         # Make trading decision based on signal
254:         self.process_signal(event.asset_id, event.signal_value, event.confidence)
255: 
256:     @abstractmethod
257:     def process_signal(
258:         self,
259:         asset_id: AssetId,
260:         signal_value: float,
261:         confidence: float | None,
262:     ) -> None:
263:         """
264:         Process a signal and decide on action.
265: 
266:         Args:
267:             asset_id: Asset the signal is for
268:             signal_value: Signal value (typically -1 to 1)
269:             confidence: Optional confidence score
270:         """
271: 
272: 
273: class IndicatorStrategy(Strategy):
274:     """
275:     Base class for indicator-based strategies.
276: 
277:     Provides utilities for managing technical indicators.
278:     """
279: 
280:     def __init__(self, name: str | None = None):
281:         """Initialize indicator strategy."""
282:         super().__init__(name)
283:         self._indicators: dict[str, Any] = {}
284: 
285:     def add_indicator(self, name: str, indicator: Any) -> None:
286:         """
287:         Add a technical indicator.
288: 
289:         Args:
290:             name: Name for the indicator
291:             indicator: Indicator instance
292:         """
293:         self._indicators[name] = indicator
294: 
295:     def get_indicator(self, name: str) -> Any:
296:         """
297:         Get an indicator by name.
298: 
299:         Args:
300:             name: Indicator name
301: 
302:         Returns:
303:             Indicator instance
304:         """
305:         return self._indicators.get(name)
306: 
307:     def update_indicators(self, price: float) -> None:
308:         """
309:         Update all indicators with new price.
310: 
311:         Args:
312:             price: Latest price
313:         """
314:         for indicator in self._indicators.values():
315:             if hasattr(indicator, "update"):
316:                 indicator.update(price)
````

## File: src/qengine/strategy/crypto_basis_adapter.py
````python
  1: """
  2: Crypto Basis Strategy QEngine Adapter
  3: =====================================
  4: 
  5: Adapter that integrates the CryptoBasisStrategy with QEngine's event-driven architecture.
  6: This allows the basis trading strategy to run within QEngine and benefit from
  7: advanced execution models, slippage simulation, and commission structures.
  8: """
  9: 
 10: from dataclasses import dataclass, field
 11: from datetime import datetime
 12: from typing import Optional
 13: 
 14: import numpy as np
 15: 
 16: from qengine.core.types import AssetId
 17: from qengine.strategy.adapters import (
 18:     DataFrameAdapter,
 19:     ExternalStrategyInterface,
 20:     PITData,
 21:     StrategySignal,
 22: )
 23: 
 24: 
 25: @dataclass
 26: class BasisState:
 27:     """State tracking for basis calculations."""
 28: 
 29:     spot_prices: list[float] = field(default_factory=list)
 30:     futures_prices: list[float] = field(default_factory=list)
 31:     timestamps: list[datetime] = field(default_factory=list)
 32:     basis_history: list[float] = field(default_factory=list)
 33: 
 34:     # Rolling statistics
 35:     basis_mean: float | None = None
 36:     basis_std: float | None = None
 37:     current_z_score: float | None = None
 38: 
 39:     # Position tracking
 40:     current_position: float = 0.0
 41:     entry_basis: float | None = None
 42:     entry_timestamp: datetime | None = None
 43: 
 44: 
 45: class CryptoBasisExternalStrategy(ExternalStrategyInterface):
 46:     """
 47:     External strategy implementation for crypto basis trading.
 48: 
 49:     This wraps the original CryptoBasisStrategy logic in the interface
 50:     required for QEngine integration.
 51:     """
 52: 
 53:     def __init__(
 54:         self,
 55:         spot_asset_id: AssetId = "BTC",
 56:         futures_asset_id: AssetId = "BTC_FUTURE",
 57:         lookback_window: int = 120,
 58:         entry_threshold: float = 2.0,
 59:         exit_threshold: float = 0.5,
 60:         max_position: float = 0.3,
 61:         min_data_points: int = 50,
 62:     ):
 63:         """
 64:         Initialize crypto basis strategy.
 65: 
 66:         Args:
 67:             spot_asset_id: Asset ID for spot prices
 68:             futures_asset_id: Asset ID for futures prices
 69:             lookback_window: Number of periods for rolling statistics
 70:             entry_threshold: Z-score threshold for entry
 71:             exit_threshold: Z-score threshold for exit
 72:             max_position: Maximum position size
 73:             min_data_points: Minimum data points before generating signals
 74:         """
 75:         self.spot_asset_id = spot_asset_id
 76:         self.futures_asset_id = futures_asset_id
 77:         self.lookback_window = lookback_window
 78:         self.entry_threshold = entry_threshold
 79:         self.exit_threshold = exit_threshold
 80:         self.max_position = max_position
 81:         self.min_data_points = min_data_points
 82: 
 83:         # State
 84:         self.state = BasisState()
 85:         self.volatility_lookback = 20
 86: 
 87:     def initialize(self) -> None:
 88:         """Initialize strategy state."""
 89:         self.state = BasisState()
 90: 
 91:     def finalize(self) -> None:
 92:         """Cleanup strategy state."""
 93:         # Log final statistics
 94:         if self.state.basis_history:
 95:             total_signals = len([b for b in self.state.basis_history if b != 0])
 96:             print(f"[CryptoBasisStrategy] Generated {total_signals} signals")
 97:             print(f"[CryptoBasisStrategy] Final position: {self.state.current_position}")
 98: 
 99:     def generate_signal(
100:         self,
101:         timestamp: datetime,
102:         pit_data: PITData,
103:     ) -> StrategySignal | None:
104:         """
105:         Generate trading signal based on basis analysis.
106: 
107:         Args:
108:             timestamp: Current timestamp
109:             pit_data: Point-in-time data snapshot
110: 
111:         Returns:
112:             Trading signal or None
113:         """
114:         # Get current prices
115:         spot_price = pit_data.get_price(self.spot_asset_id)
116:         futures_price = pit_data.get_price(self.futures_asset_id)
117: 
118:         if spot_price is None or futures_price is None:
119:             return None
120: 
121:         # Update state with new data
122:         self._update_state(timestamp, spot_price, futures_price)
123: 
124:         # Calculate current statistics if we have enough data
125:         if len(self.state.basis_history) >= 2:
126:             self._calculate_statistics()
127: 
128:         # Need minimum data points for signal generation
129:         if len(self.state.basis_history) < self.min_data_points:
130:             return None
131: 
132:         # Generate signal
133:         signal = self._generate_basis_signal(timestamp)
134: 
135:         return signal
136: 
137:     def _update_state(
138:         self,
139:         timestamp: datetime,
140:         spot_price: float,
141:         futures_price: float,
142:     ) -> None:
143:         """Update internal state with new price data."""
144:         # Calculate basis
145:         basis = futures_price - spot_price
146: 
147:         # Update history
148:         self.state.timestamps.append(timestamp)
149:         self.state.spot_prices.append(spot_price)
150:         self.state.futures_prices.append(futures_price)
151:         self.state.basis_history.append(basis)
152: 
153:         # Maintain window size
154:         if len(self.state.basis_history) > self.lookback_window:
155:             self.state.timestamps = self.state.timestamps[-self.lookback_window :]
156:             self.state.spot_prices = self.state.spot_prices[-self.lookback_window :]
157:             self.state.futures_prices = self.state.futures_prices[-self.lookback_window :]
158:             self.state.basis_history = self.state.basis_history[-self.lookback_window :]
159: 
160:     def _calculate_statistics(self) -> None:
161:         """Calculate rolling statistics for basis."""
162:         if len(self.state.basis_history) < 2:
163:             return
164: 
165:         basis_array = np.array(self.state.basis_history)
166: 
167:         # Rolling mean and std
168:         self.state.basis_mean = np.mean(basis_array)
169:         self.state.basis_std = np.std(basis_array)
170: 
171:         # Current z-score
172:         if self.state.basis_std > 1e-8:  # Avoid division by zero
173:             current_basis = self.state.basis_history[-1]
174:             self.state.current_z_score = (
175:                 current_basis - self.state.basis_mean
176:             ) / self.state.basis_std
177:         else:
178:             self.state.current_z_score = 0.0
179: 
180:     def _generate_basis_signal(self, timestamp: datetime) -> StrategySignal | None:
181:         """Generate trading signal based on basis z-score."""
182:         if self.state.current_z_score is None:
183:             return None
184: 
185:         z_score = self.state.current_z_score
186:         current_basis = self.state.basis_history[-1]
187: 
188:         # Calculate volatility for position sizing
189:         spot_returns = np.diff(np.log(self.state.spot_prices[-self.volatility_lookback :]))
190:         volatility = np.std(spot_returns) if len(spot_returns) > 1 else 0.01
191: 
192:         position = 0.0
193:         confidence = 0.0
194: 
195:         # Entry logic
196:         if abs(self.state.current_position) < 1e-6:  # Flat position
197:             if z_score > self.entry_threshold:
198:                 # Basis too high: short futures, long spot (negative position)
199:                 position = -1.0
200:                 confidence = min((z_score - self.entry_threshold) / 2, 1.0)
201:                 self.state.entry_basis = current_basis
202:                 self.state.entry_timestamp = timestamp
203: 
204:             elif z_score < -self.entry_threshold:
205:                 # Basis too low: long futures, short spot (positive position)
206:                 position = 1.0
207:                 confidence = min((abs(z_score) - self.entry_threshold) / 2, 1.0)
208:                 self.state.entry_basis = current_basis
209:                 self.state.entry_timestamp = timestamp
210: 
211:         # Exit logic
212:         else:
213:             if self.state.current_position > 0:  # Long futures/short spot
214:                 if z_score > -self.exit_threshold:  # Basis normalized
215:                     position = 0.0
216:                     confidence = 1.0
217:                 elif z_score > self.entry_threshold:  # Reversal
218:                     position = -1.0
219:                     confidence = min((z_score - self.entry_threshold) / 2, 1.0)
220:                 else:
221:                     position = self.state.current_position  # Hold
222: 
223:             elif self.state.current_position < 0:  # Short futures/long spot
224:                 if z_score < self.exit_threshold:  # Basis normalized
225:                     position = 0.0
226:                     confidence = 1.0
227:                 elif z_score < -self.entry_threshold:  # Reversal
228:                     position = 1.0
229:                     confidence = min((abs(z_score) - self.entry_threshold) / 2, 1.0)
230:                 else:
231:                     position = self.state.current_position  # Hold
232: 
233:         # Apply volatility adjustment and position limits
234:         if position != 0:
235:             volatility_scalar = 1 / (1 + volatility * 10)  # Reduce size in high vol
236:             position = position * min(confidence * volatility_scalar, self.max_position)
237: 
238:         # Update position state
239:         old_position = self.state.current_position
240:         self.state.current_position = position
241: 
242:         # Only generate signal if position changed significantly
243:         if abs(position - old_position) > 0.001:
244:             return StrategySignal(
245:                 timestamp=timestamp,
246:                 asset_id=self.spot_asset_id,  # Use spot as primary asset
247:                 position=position,
248:                 confidence=confidence,
249:                 metadata={
250:                     "basis": current_basis,
251:                     "z_score": z_score,
252:                     "volatility": volatility,
253:                     "entry_basis": self.state.entry_basis,
254:                     "strategy_type": "crypto_basis",
255:                     "spot_price": self.state.spot_prices[-1],
256:                     "futures_price": self.state.futures_prices[-1],
257:                 },
258:             )
259: 
260:         return None
261: 
262:     def get_current_statistics(self) -> dict[str, float]:
263:         """Get current basis statistics for monitoring."""
264:         if not self.state.basis_history:
265:             return {}
266: 
267:         return {
268:             "current_basis": self.state.basis_history[-1],
269:             "basis_mean": self.state.basis_mean or 0,
270:             "basis_std": self.state.basis_std or 0,
271:             "z_score": self.state.current_z_score or 0,
272:             "current_position": self.state.current_position,
273:             "data_points": len(self.state.basis_history),
274:         }
275: 
276: 
277: class CryptoBasisAdapter(DataFrameAdapter):
278:     """
279:     Complete adapter for crypto basis trading strategy.
280: 
281:     This combines the external strategy with DataFrame support
282:     and provides a complete QEngine integration.
283:     """
284: 
285:     def __init__(
286:         self,
287:         spot_asset_id: AssetId = "BTC",
288:         futures_asset_id: AssetId = "BTC_FUTURE",
289:         lookback_window: int = 120,
290:         entry_threshold: float = 2.0,
291:         exit_threshold: float = 0.5,
292:         max_position: float = 0.3,
293:         position_scaling: float = 0.1,
294:         window_size: int = 1000,
295:         **kwargs,
296:     ):
297:         """
298:         Initialize crypto basis adapter.
299: 
300:         Args:
301:             spot_asset_id: Asset ID for spot prices
302:             futures_asset_id: Asset ID for futures prices
303:             lookback_window: Rolling window for statistics
304:             entry_threshold: Z-score threshold for entries
305:             exit_threshold: Z-score threshold for exits
306:             max_position: Maximum position size
307:             position_scaling: Scaling factor for position size
308:             **kwargs: Additional arguments for DataFrameAdapter
309:         """
310:         # Create external strategy
311:         external_strategy = CryptoBasisExternalStrategy(
312:             spot_asset_id=spot_asset_id,
313:             futures_asset_id=futures_asset_id,
314:             lookback_window=lookback_window,
315:             entry_threshold=entry_threshold,
316:             exit_threshold=exit_threshold,
317:             max_position=max_position,
318:         )
319: 
320:         # Create custom position sizer
321:         def basis_position_sizer(signal: StrategySignal, cash: float) -> float:
322:             # Scale position based on available cash and signal strength
323:             base_value = cash * position_scaling  # Use X% of cash
324:             position_value = base_value * abs(signal.position) * signal.confidence
325: 
326:             # Return signed position value
327:             return position_value if signal.position > 0 else -position_value
328: 
329:         # Filter kwargs for parent constructor
330:         parent_kwargs = {
331:             k: v for k, v in kwargs.items() if k in ["position_sizer", "risk_manager", "name"]
332:         }
333: 
334:         # Initialize adapter
335:         super().__init__(
336:             external_strategy=external_strategy,
337:             window_size=window_size,
338:             position_sizer=basis_position_sizer,
339:             name=f"CryptoBasisAdapter_{spot_asset_id}_{futures_asset_id}",
340:             **parent_kwargs,
341:         )
342: 
343:         # Store configuration
344:         self.spot_asset_id = spot_asset_id
345:         self.futures_asset_id = futures_asset_id
346:         self._last_statistics = {}
347: 
348:         # Event synchronization for multi-asset basis calculation
349:         self._event_buffer: dict[datetime, dict[AssetId, MarketEvent]] = {}
350:         self._required_assets = {spot_asset_id, futures_asset_id}
351: 
352:     def on_start(self) -> None:
353:         """Start strategy and subscribe to data feeds."""
354:         super().on_start()
355: 
356:         # Subscribe to both spot and futures data
357:         self.subscribe(asset=self.spot_asset_id, event_type="market")
358:         self.subscribe(asset=self.futures_asset_id, event_type="market")
359: 
360:         self.log(f"Subscribed to {self.spot_asset_id} (spot) and {self.futures_asset_id} (futures)")
361: 
362:     def on_market_event(self, event) -> None:
363:         """Process market events with synchronization for basis calculation."""
364:         # Buffer the event by timestamp and asset
365:         if event.timestamp not in self._event_buffer:
366:             self._event_buffer[event.timestamp] = {}
367: 
368:         self._event_buffer[event.timestamp][event.asset_id] = event
369: 
370:         # Check if we have both required assets for this timestamp
371:         buffered_assets = set(self._event_buffer[event.timestamp].keys())
372: 
373:         if self._required_assets.issubset(buffered_assets):
374:             # We have both spot and futures data - process synchronously
375:             self._process_synchronized_events(event.timestamp)
376: 
377:             # Clean up old buffer entries (keep only last 10 timestamps)
378:             timestamps = sorted(self._event_buffer.keys())
379:             if len(timestamps) > 10:
380:                 for old_ts in timestamps[:-10]:
381:                     del self._event_buffer[old_ts]
382:         else:
383:             # Still waiting for the other asset - update both DataFrame and parent's history
384:             super()._update_data_history(event)
385: 
386:     def _process_synchronized_events(self, timestamp: datetime) -> None:
387:         """Process events when both spot and futures data are available."""
388:         buffered_events = self._event_buffer[timestamp]
389: 
390:         # Process both events to update DataFrames AND internal history
391:         for event in buffered_events.values():
392:             # Call parent's update which maintains _data_history dict
393:             super()._update_data_history(event)
394: 
395:         # Now create PITData - parent's _create_pit_data should have both prices now
396:         try:
397:             # Create point-in-time data snapshot with both prices
398:             pit_data = self._create_pit_data(timestamp)
399: 
400:             # Debug: check if PITData has both prices
401:             spot_price = pit_data.get_price(self.spot_asset_id)
402:             futures_price = pit_data.get_price(self.futures_asset_id)
403: 
404:             if spot_price is None or futures_price is None:
405:                 self.log(
406:                     f"Missing prices in PITData: spot={spot_price}, futures={futures_price}",
407:                     level="WARNING",
408:                 )
409:                 return
410: 
411:             # Generate signal from external strategy (now has both prices)
412:             signal = self.external_strategy.generate_signal(timestamp, pit_data)
413: 
414:             if signal:
415:                 self.log(
416:                     f"Signal generated: pos={signal.position:.3f}, conf={signal.confidence:.3f}",
417:                 )
418:                 self._process_signal(signal)
419:             else:
420:                 # Check if we expected a signal but didn't get one
421:                 stats = self.external_strategy.get_current_statistics()
422:                 if abs(stats.get("z_score", 0)) > 1.0:
423:                     self.log(f"Expected signal but got None: z_score={stats.get('z_score', 0):.2f}")
424: 
425:             # Update statistics for monitoring
426:             if hasattr(self.external_strategy, "get_current_statistics"):
427:                 self._last_statistics = self.external_strategy.get_current_statistics()
428: 
429:                 # Log all statistics for debugging
430:                 self.log(
431:                     f"Basis stats: z={self._last_statistics['z_score']:.3f}, "
432:                     f"mean={self._last_statistics.get('basis_mean', 0):.1f}, "
433:                     f"std={self._last_statistics.get('basis_std', 0):.3f}, "
434:                     f"current={self._last_statistics.get('current_basis', 0):.0f}, "
435:                     f"data_pts={self._last_statistics.get('data_points', 0)}",
436:                     level="INFO",
437:                 )
438: 
439:         except Exception as e:
440:             self.log(f"Error processing synchronized events: {e}", level="ERROR")
441: 
442:     def get_strategy_diagnostics(self) -> dict[str, any]:
443:         """Get detailed diagnostics for strategy monitoring."""
444:         base_state = self.get_strategy_state()
445: 
446:         # Add basis-specific statistics
447:         base_state.update(
448:             {
449:                 "basis_statistics": self._last_statistics,
450:                 "spot_asset": self.spot_asset_id,
451:                 "futures_asset": self.futures_asset_id,
452:                 "dataframe_sizes": {
453:                     asset: len(df) for asset, df in self.get_all_dataframes().items()
454:                 },
455:             },
456:         )
457: 
458:         return base_state
459: 
460: 
461: def create_crypto_basis_strategy(
462:     spot_asset_id: AssetId = "BTC",
463:     futures_asset_id: AssetId = "BTC_FUTURE",
464:     **strategy_params,
465: ) -> CryptoBasisAdapter:
466:     """
467:     Factory function to create a crypto basis strategy.
468: 
469:     Args:
470:         spot_asset_id: Asset ID for spot prices
471:         futures_asset_id: Asset ID for futures prices
472:         **strategy_params: Strategy parameters
473: 
474:     Returns:
475:         Configured CryptoBasisAdapter
476:     """
477:     return CryptoBasisAdapter(
478:         spot_asset_id=spot_asset_id,
479:         futures_asset_id=futures_asset_id,
480:         **strategy_params,
481:     )
````

## File: src/qengine/strategy/spy_order_flow_adapter.py
````python
  1: """
  2: SPY Order Flow Strategy Adapter for QEngine Integration
  3: 
  4: Adapts the SPYOrderFlowStrategy to work with QEngine's event-driven architecture.
  5: This strategy uses microstructure features from order flow to predict short-term SPY movements.
  6: """
  7: 
  8: from dataclasses import dataclass, field
  9: from datetime import datetime
 10: from typing import Optional
 11: 
 12: import numpy as np
 13: 
 14: from qengine.core.types import AssetId
 15: from qengine.strategy.adapters import (
 16:     DataFrameAdapter,
 17:     ExternalStrategyInterface,
 18:     PITData,
 19:     StrategySignal,
 20: )
 21: 
 22: 
 23: @dataclass
 24: class OrderFlowState:
 25:     """State tracking for SPY order flow strategy."""
 26: 
 27:     # Price and volume history
 28:     prices: list[float] = field(default_factory=list)
 29:     volumes: list[float] = field(default_factory=list)
 30:     timestamps: list[datetime] = field(default_factory=list)
 31: 
 32:     # Order flow features
 33:     buy_volumes: list[float] = field(default_factory=list)
 34:     sell_volumes: list[float] = field(default_factory=list)
 35:     volume_imbalances: list[float] = field(default_factory=list)
 36: 
 37:     # Derived features
 38:     price_momentum_5: float = 0.0
 39:     price_momentum_20: float = 0.0
 40:     volume_momentum_5: float = 0.0
 41:     imbalance_ratio: float = 0.5
 42: 
 43:     # Trading state
 44:     current_position: float = 0.0
 45:     last_signal_time: datetime | None = None
 46:     signal_count: int = 0
 47: 
 48:     # Rolling statistics
 49:     price_mean_20: float = 0.0
 50:     price_std_10: float = 0.0
 51:     imbalance_mean: float = 0.0
 52:     imbalance_std: float = 0.0
 53: 
 54: 
 55: class SPYOrderFlowExternalStrategy(ExternalStrategyInterface):
 56:     """
 57:     External SPY order flow strategy implementation.
 58: 
 59:     Uses microstructure features from order flow to predict short-term movements.
 60:     Generates directional signals based on order flow imbalances and momentum.
 61:     """
 62: 
 63:     def __init__(
 64:         self,
 65:         asset_id: AssetId = "SPY",
 66:         lookback_window: int = 100,
 67:         momentum_window_short: int = 5,
 68:         momentum_window_long: int = 20,
 69:         imbalance_threshold: float = 0.65,
 70:         momentum_threshold: float = 0.002,
 71:         min_data_points: int = 20,
 72:         signal_cooldown: int = 5,  # Minimum bars between signals
 73:     ):
 74:         """
 75:         Initialize SPY order flow strategy.
 76: 
 77:         Args:
 78:             asset_id: Asset identifier for SPY
 79:             lookback_window: Rolling window for statistics
 80:             momentum_window_short: Short momentum calculation window
 81:             momentum_window_long: Long momentum calculation window
 82:             imbalance_threshold: Threshold for order flow imbalance signals
 83:             momentum_threshold: Threshold for momentum signals
 84:             min_data_points: Minimum data points before generating signals
 85:             signal_cooldown: Minimum bars between signals
 86:         """
 87:         self.asset_id = asset_id
 88:         self.lookback_window = lookback_window
 89:         self.momentum_window_short = momentum_window_short
 90:         self.momentum_window_long = momentum_window_long
 91:         self.imbalance_threshold = imbalance_threshold
 92:         self.momentum_threshold = momentum_threshold
 93:         self.min_data_points = min_data_points
 94:         self.signal_cooldown = signal_cooldown
 95: 
 96:         # Initialize state
 97:         self.state = OrderFlowState()
 98: 
 99:     def initialize(self) -> None:
100:         """Initialize strategy state (required by interface)."""
101:         self.state = OrderFlowState()
102:         print(f"[SPYOrderFlowStrategy] Initialized with asset {self.asset_id}")
103: 
104:     def finalize(self) -> None:
105:         """Clean up strategy state (required by interface)."""
106:         print(f"[SPYOrderFlowStrategy] Generated {self.state.signal_count} signals")
107:         print(f"[SPYOrderFlowStrategy] Final position: {self.state.current_position}")
108: 
109:     def on_start(self) -> None:
110:         """Initialize strategy state (alias for initialize)."""
111:         self.initialize()
112: 
113:     def on_end(self) -> None:
114:         """Clean up strategy state (alias for finalize)."""
115:         self.finalize()
116: 
117:     def generate_signal(self, timestamp: datetime, pit_data: PITData) -> StrategySignal | None:
118:         """
119:         Generate trading signal based on order flow analysis.
120: 
121:         Args:
122:             timestamp: Current timestamp
123:             pit_data: Point-in-time data snapshot
124: 
125:         Returns:
126:             Trading signal or None
127:         """
128:         # Get current market data
129:         price = pit_data.get_price(self.asset_id)
130: 
131:         # Get volume from asset data if available
132:         asset_data = pit_data.asset_data.get(self.asset_id, {})
133:         volume = asset_data.get("volume", 0)
134: 
135:         if price is None or volume == 0:
136:             return None
137: 
138:         # Update state with new data
139:         self._update_state(timestamp, price, volume, pit_data)
140: 
141:         # Need minimum data for analysis
142:         if len(self.state.prices) < self.min_data_points:
143:             return None
144: 
145:         # Check signal cooldown
146:         if self.state.last_signal_time is not None:
147:             # Find the most recent timestamp index
148:             try:
149:                 last_signal_idx = self.state.timestamps.index(self.state.last_signal_time)
150:                 bars_since_signal = len(self.state.timestamps) - last_signal_idx - 1
151:             except ValueError:
152:                 # If timestamp not found, calculate based on current time
153:                 bars_since_signal = self.signal_cooldown + 1  # Allow signal
154: 
155:             if bars_since_signal < self.signal_cooldown:
156:                 return None
157: 
158:         # Calculate current features
159:         self._calculate_features()
160: 
161:         # Generate signal based on order flow and momentum
162:         signal = self._generate_order_flow_signal(timestamp)
163: 
164:         if signal:
165:             self.state.signal_count += 1
166:             self.state.last_signal_time = timestamp
167: 
168:         return signal
169: 
170:     def _update_state(
171:         self,
172:         timestamp: datetime,
173:         price: float,
174:         volume: float,
175:         pit_data: PITData,
176:     ) -> None:
177:         """Update internal state with new market data."""
178:         # Add basic data
179:         self.state.prices.append(price)
180:         self.state.volumes.append(volume)
181:         self.state.timestamps.append(timestamp)
182: 
183:         # Extract order flow features from PITData if available
184:         asset_data = pit_data.asset_data.get(self.asset_id, {})
185: 
186:         # Get buy/sell volumes (use heuristics if not available)
187:         buy_volume = asset_data.get("buy_volume", volume * 0.5)
188:         sell_volume = asset_data.get("sell_volume", volume * 0.5)
189: 
190:         self.state.buy_volumes.append(buy_volume)
191:         self.state.sell_volumes.append(sell_volume)
192: 
193:         # Calculate volume imbalance
194:         total_volume = buy_volume + sell_volume + 1e-10
195:         imbalance = (buy_volume - sell_volume) / total_volume
196:         self.state.volume_imbalances.append(imbalance)
197: 
198:         # Keep only lookback window
199:         if len(self.state.prices) > self.lookback_window:
200:             self.state.prices = self.state.prices[-self.lookback_window :]
201:             self.state.volumes = self.state.volumes[-self.lookback_window :]
202:             self.state.timestamps = self.state.timestamps[-self.lookback_window :]
203:             self.state.buy_volumes = self.state.buy_volumes[-self.lookback_window :]
204:             self.state.sell_volumes = self.state.sell_volumes[-self.lookback_window :]
205:             self.state.volume_imbalances = self.state.volume_imbalances[-self.lookback_window :]
206: 
207:     def _calculate_features(self) -> None:
208:         """Calculate order flow and momentum features."""
209:         prices = np.array(self.state.prices)
210:         volumes = np.array(self.state.volumes)
211:         imbalances = np.array(self.state.volume_imbalances)
212: 
213:         # Price momentum
214:         if len(prices) >= self.momentum_window_short:
215:             self.state.price_momentum_5 = prices[-1] / prices[-self.momentum_window_short] - 1
216: 
217:         if len(prices) >= self.momentum_window_long:
218:             self.state.price_momentum_20 = prices[-1] / prices[-self.momentum_window_long] - 1
219:             self.state.price_mean_20 = np.mean(prices[-self.momentum_window_long :])
220: 
221:         # Volume momentum
222:         if len(volumes) >= self.momentum_window_short:
223:             self.state.volume_momentum_5 = (
224:                 volumes[-1] / np.mean(volumes[-self.momentum_window_short :]) - 1
225:             )
226: 
227:         # Price volatility
228:         if len(prices) >= 10:
229:             self.state.price_std_10 = np.std(prices[-10:])
230: 
231:         # Imbalance statistics
232:         if len(imbalances) >= 20:
233:             self.state.imbalance_mean = np.mean(imbalances[-20:])
234:             self.state.imbalance_std = np.std(imbalances[-20:])
235: 
236:         # Current imbalance ratio
237:         if len(self.state.buy_volumes) > 0:
238:             recent_buy = np.mean(self.state.buy_volumes[-5:])
239:             recent_sell = np.mean(self.state.sell_volumes[-5:])
240:             self.state.imbalance_ratio = recent_buy / (recent_buy + recent_sell + 1e-10)
241: 
242:     def _generate_order_flow_signal(self, timestamp: datetime) -> StrategySignal | None:
243:         """Generate signal based on order flow imbalance and momentum."""
244:         # Signal strength based on multiple factors
245:         signal_strength = 0.0
246:         factors = []
247: 
248:         # 1. Order flow imbalance signal
249:         if self.state.imbalance_ratio > self.imbalance_threshold:
250:             signal_strength += 0.4
251:             factors.append("buy_pressure")
252:         elif self.state.imbalance_ratio < (1 - self.imbalance_threshold):
253:             signal_strength -= 0.4
254:             factors.append("sell_pressure")
255: 
256:         # 2. Price momentum confirmation
257:         if abs(self.state.price_momentum_5) > self.momentum_threshold:
258:             if self.state.price_momentum_5 > 0:
259:                 signal_strength += 0.3
260:                 factors.append("positive_momentum")
261:             else:
262:                 signal_strength -= 0.3
263:                 factors.append("negative_momentum")
264: 
265:         # 3. Volume surge detection
266:         if self.state.volume_momentum_5 > 0.5:  # 50% above average
267:             signal_strength += 0.2 * np.sign(signal_strength)
268:             factors.append("volume_surge")
269: 
270:         # 4. Mean reversion opportunity
271:         if len(self.state.prices) >= 20:
272:             price_deviation = (self.state.prices[-1] - self.state.price_mean_20) / (
273:                 self.state.price_std_10 + 1e-10
274:             )
275:             if abs(price_deviation) > 2:  # 2 standard deviations
276:                 signal_strength -= 0.2 * np.sign(price_deviation)  # Mean reversion
277:                 factors.append("mean_reversion")
278: 
279:         # Generate signal if strong enough
280:         threshold = 0.5
281:         if abs(signal_strength) >= threshold:
282:             # Determine position
283:             if signal_strength > 0:
284:                 position = min(1.0, signal_strength)  # Long
285:                 signal_type = "BUY"
286:             else:
287:                 position = max(-1.0, signal_strength)  # Short
288:                 signal_type = "SELL"
289: 
290:             # Confidence based on signal strength
291:             confidence = min(1.0, abs(signal_strength) / 1.5)
292: 
293:             # Update position tracking
294:             self.state.current_position = position
295: 
296:             return StrategySignal(
297:                 timestamp=timestamp,
298:                 asset_id=self.asset_id,
299:                 position=position,
300:                 confidence=confidence,
301:                 metadata={
302:                     "signal_type": signal_type,
303:                     "factors": factors,
304:                     "imbalance_ratio": round(self.state.imbalance_ratio, 3),
305:                     "price_momentum_5": round(self.state.price_momentum_5, 4),
306:                     "volume_momentum": round(self.state.volume_momentum_5, 3),
307:                     "signal_strength": round(signal_strength, 3),
308:                 },
309:             )
310: 
311:         return None
312: 
313:     def get_current_statistics(self) -> dict[str, any]:
314:         """Get current strategy statistics."""
315:         return {
316:             "data_points": len(self.state.prices),
317:             "current_position": self.state.current_position,
318:             "imbalance_ratio": self.state.imbalance_ratio,
319:             "price_momentum_5": self.state.price_momentum_5,
320:             "price_momentum_20": self.state.price_momentum_20,
321:             "volume_momentum_5": self.state.volume_momentum_5,
322:             "signal_count": self.state.signal_count,
323:         }
324: 
325: 
326: class SPYOrderFlowAdapter(DataFrameAdapter):
327:     """
328:     Complete adapter for SPY order flow trading strategy.
329: 
330:     This combines the external strategy with DataFrame support
331:     and provides complete QEngine integration for order flow analysis.
332:     """
333: 
334:     def __init__(
335:         self,
336:         asset_id: AssetId = "SPY",
337:         lookback_window: int = 100,
338:         momentum_window_short: int = 5,
339:         momentum_window_long: int = 20,
340:         imbalance_threshold: float = 0.65,
341:         momentum_threshold: float = 0.002,
342:         position_scaling: float = 0.2,
343:         window_size: int = 1000,
344:         **kwargs,
345:     ):
346:         """
347:         Initialize SPY order flow adapter.
348: 
349:         Args:
350:             asset_id: Asset identifier for SPY
351:             lookback_window: Rolling window for statistics
352:             momentum_window_short: Short momentum window
353:             momentum_window_long: Long momentum window
354:             imbalance_threshold: Order flow imbalance threshold
355:             momentum_threshold: Price momentum threshold
356:             position_scaling: Scaling factor for position size
357:             window_size: DataFrame history window
358:             **kwargs: Additional arguments for DataFrameAdapter
359:         """
360:         # Create external strategy
361:         external_strategy = SPYOrderFlowExternalStrategy(
362:             asset_id=asset_id,
363:             lookback_window=lookback_window,
364:             momentum_window_short=momentum_window_short,
365:             momentum_window_long=momentum_window_long,
366:             imbalance_threshold=imbalance_threshold,
367:             momentum_threshold=momentum_threshold,
368:         )
369: 
370:         # Create custom position sizer for order flow strategy
371:         def order_flow_position_sizer(signal: StrategySignal, cash: float) -> float:
372:             # Scale position based on signal strength and confidence
373:             base_value = cash * position_scaling
374: 
375:             # Adjust for confidence
376:             position_value = base_value * abs(signal.position) * signal.confidence
377: 
378:             # Apply maximum position limits
379:             max_position = cash * 0.5  # Maximum 50% of capital
380:             position_value = min(position_value, max_position)
381: 
382:             # Return signed position value
383:             return position_value if signal.position > 0 else -position_value
384: 
385:         # Filter kwargs for parent constructor
386:         parent_kwargs = {k: v for k, v in kwargs.items() if k in ["risk_manager", "name"]}
387: 
388:         # Initialize adapter
389:         super().__init__(
390:             external_strategy=external_strategy,
391:             window_size=window_size,
392:             position_sizer=order_flow_position_sizer,
393:             name=f"SPYOrderFlowAdapter_{asset_id}",
394:             **parent_kwargs,
395:         )
396: 
397:         # Store configuration
398:         self.asset_id = asset_id
399:         self._last_statistics = {}
400: 
401:     def on_start(self) -> None:
402:         """Start strategy and subscribe to data feeds."""
403:         super().on_start()
404: 
405:         # Subscribe to SPY market data
406:         self.subscribe(asset=self.asset_id, event_type="market")
407: 
408:         self.log(f"Subscribed to {self.asset_id} order flow data")
409: 
410:     def on_market_event(self, event) -> None:
411:         """Process market events with order flow analysis."""
412:         # Update data history
413:         self._update_data_history(event)
414: 
415:         # Process through parent's market event handler
416:         super().on_market_event(event)
417: 
418:         # Update statistics for monitoring
419:         if hasattr(self.external_strategy, "get_current_statistics"):
420:             self._last_statistics = self.external_strategy.get_current_statistics()
421: 
422:             # Log significant order flow events
423:             if self._last_statistics.get("imbalance_ratio", 0.5):
424:                 imbalance = self._last_statistics["imbalance_ratio"]
425:                 if imbalance > 0.7 or imbalance < 0.3:
426:                     self.log(
427:                         f"Significant order flow: imbalance={imbalance:.3f}, "
428:                         f"momentum={self._last_statistics.get('price_momentum_5', 0):.4f}",
429:                         level="INFO",
430:                     )
431: 
432:     def get_strategy_diagnostics(self) -> dict[str, any]:
433:         """Get detailed diagnostics for strategy monitoring."""
434:         base_state = self.get_strategy_state()
435: 
436:         # Add order flow specific diagnostics
437:         order_flow_stats = self._last_statistics.copy() if self._last_statistics else {}
438: 
439:         return {
440:             **base_state,
441:             "order_flow_statistics": order_flow_stats,
442:             "strategy_type": "SPY Order Flow Momentum",
443:             "asset": self.asset_id,
444:         }
445: 
446: 
447: def create_spy_order_flow_strategy(**kwargs) -> SPYOrderFlowAdapter:
448:     """
449:     Factory function to create SPY order flow strategy adapter.
450: 
451:     Args:
452:         **kwargs: Configuration parameters for the adapter
453: 
454:     Returns:
455:         Configured SPYOrderFlowAdapter instance
456:     """
457:     return SPYOrderFlowAdapter(**kwargs)
````

## File: src/qengine/__init__.py
````python
 1: """QEngine - A state-of-the-art event-driven backtesting engine.
 2: 
 3: QEngine is designed for high-performance backtesting of machine learning-driven
 4: trading strategies with a focus on preventing data leakage and providing
 5: realistic market simulation.
 6: """
 7: 
 8: __version__ = "0.1.0"
 9: 
10: from qengine.core import Clock, Event, EventBus
11: from qengine.data import DataFeed
12: from qengine.engine import BacktestEngine, BacktestResults
13: from qengine.strategy import Strategy
14: 
15: __all__ = [
16:     "BacktestEngine",
17:     "BacktestResults",
18:     "Clock",
19:     "DataFeed",
20:     "Event",
21:     "EventBus",
22:     "Strategy",
23: ]
````

## File: src/qengine/engine.py
````python
  1: """Main backtest engine that orchestrates the simulation."""
  2: 
  3: import logging
  4: from datetime import datetime
  5: from typing import Any, Optional
  6: 
  7: import polars as pl
  8: 
  9: from qengine.core.clock import Clock
 10: from qengine.core.event import EventBus, EventType
 11: from qengine.data.feed import DataFeed
 12: from qengine.execution.broker import Broker
 13: from qengine.portfolio.portfolio import Portfolio
 14: from qengine.reporting.reporter import Reporter
 15: from qengine.strategy.base import Strategy
 16: 
 17: logger = logging.getLogger(__name__)
 18: 
 19: 
 20: class BacktestEngine:
 21:     """Main backtesting engine that coordinates all components.
 22: 
 23:     The engine follows an event-driven architecture where:
 24:     1. Data feeds generate market events
 25:     2. Strategies consume events and generate signals/orders
 26:     3. Broker executes orders and generates fills
 27:     4. Portfolio tracks positions and P&L
 28:     5. Reporter captures results
 29: 
 30:     Example:
 31:         >>> from qengine import BacktestEngine
 32:         >>> from qengine.data import ParquetDataFeed
 33:         >>> from qengine.strategy import BuyAndHoldStrategy
 34:         >>>
 35:         >>> engine = BacktestEngine(
 36:         ...     data_feed=ParquetDataFeed("data.parquet"),
 37:         ...     strategy=BuyAndHoldStrategy(),
 38:         ...     initial_capital=100000
 39:         ... )
 40:         >>> results = engine.run()
 41:     """
 42: 
 43:     def __init__(
 44:         self,
 45:         data_feed: DataFeed,
 46:         strategy: Strategy,
 47:         broker: Broker | None = None,
 48:         portfolio: Portfolio | None = None,
 49:         reporter: Reporter | None = None,
 50:         initial_capital: float = 100000.0,
 51:         currency: str = "USD",
 52:         use_priority_queue: bool = True,
 53:     ):
 54:         """Initialize the backtest engine.
 55: 
 56:         Args:
 57:             data_feed: Source of market data events
 58:             strategy: Trading strategy to execute
 59:             broker: Order execution broker (default: SimulationBroker)
 60:             portfolio: Portfolio tracker (default: SimplePortfolio)
 61:             reporter: Results reporter (default: InMemoryReporter)
 62:             initial_capital: Starting capital
 63:             currency: Base currency for the portfolio
 64:             use_priority_queue: Use priority queue for event ordering
 65:         """
 66:         self.data_feed = data_feed
 67:         self.strategy = strategy
 68:         self.initial_capital = initial_capital
 69:         self.currency = currency
 70: 
 71:         # Create event bus for communication
 72:         self.event_bus = EventBus(use_priority_queue=use_priority_queue)
 73: 
 74:         # Create clock for time management
 75:         self.clock = Clock()
 76: 
 77:         # Initialize broker if not provided
 78:         if broker is None:
 79:             from qengine.execution.broker import SimulationBroker
 80: 
 81:             self.broker = SimulationBroker()
 82:         else:
 83:             self.broker = broker
 84: 
 85:         # Initialize portfolio if not provided
 86:         if portfolio is None:
 87:             from qengine.portfolio.simple import SimplePortfolio
 88: 
 89:             self.portfolio = SimplePortfolio(initial_capital=initial_capital, currency=currency)
 90:         else:
 91:             self.portfolio = portfolio
 92: 
 93:         # Initialize reporter if not provided
 94:         if reporter is None:
 95:             from qengine.reporting.reporter import InMemoryReporter
 96: 
 97:             self.reporter = InMemoryReporter()
 98:         else:
 99:             self.reporter = reporter
100: 
101:         # Wire up event handlers
102:         self._setup_event_handlers()
103: 
104:         # Statistics
105:         self.events_processed = 0
106:         self.start_time: datetime | None = None
107:         self.end_time: datetime | None = None
108: 
109:     def _setup_event_handlers(self) -> None:
110:         """Connect components via event subscriptions."""
111:         # Strategy subscribes to market and fill events
112:         self.event_bus.subscribe(EventType.MARKET, self.strategy.on_market_event)
113:         self.event_bus.subscribe(EventType.FILL, self.strategy.on_fill_event)
114: 
115:         # Broker subscribes to order events
116:         self.event_bus.subscribe(EventType.ORDER, self.broker.on_order_event)
117: 
118:         # Portfolio subscribes to fill events
119:         self.event_bus.subscribe(EventType.FILL, self.portfolio.on_fill_event)
120: 
121:         # Reporter subscribes to all events for logging
122:         for event_type in EventType:
123:             self.event_bus.subscribe(event_type, self.reporter.on_event)
124: 
125:     def run(
126:         self,
127:         start_date: datetime | None = None,
128:         end_date: datetime | None = None,
129:         max_events: int | None = None,
130:     ) -> dict[str, Any]:
131:         """Run the backtest simulation.
132: 
133:         Args:
134:             start_date: Start date for backtest (None = use data start)
135:             end_date: End date for backtest (None = use data end)
136:             max_events: Maximum events to process (for debugging)
137: 
138:         Returns:
139:             Dictionary containing backtest results including:
140:             - trades: DataFrame of executed trades
141:             - positions: DataFrame of position history
142:             - returns: Series of strategy returns
143:             - metrics: Performance metrics dict
144:             - events_processed: Number of events processed
145:         """
146:         logger.info("Starting backtest engine")
147:         self.start_time = datetime.now()
148: 
149:         # Initialize components
150:         self.data_feed.initialize(start_date, end_date)
151:         self.strategy.on_start(self.portfolio, self.event_bus)
152:         self.broker.initialize(self.portfolio, self.event_bus)
153:         self.portfolio.initialize()
154:         self.reporter.on_start()
155: 
156:         # Initialize clock with data feed's time range
157:         if hasattr(self.data_feed, "get_time_range"):
158:             data_start, data_end = self.data_feed.get_time_range()
159:             self.clock.advance_to(start_date or data_start)
160: 
161:         # Main event loop
162:         self.events_processed = 0
163: 
164:         while not self.data_feed.is_exhausted:
165:             # Check max events limit
166:             if max_events and self.events_processed >= max_events:
167:                 logger.info(f"Reached max events limit: {max_events}")
168:                 break
169: 
170:             # Get next market event from data feed
171:             market_event = self.data_feed.get_next_event()
172:             if market_event is None:
173:                 break
174: 
175:             # Update clock
176:             self.clock.advance_to(market_event.timestamp)
177: 
178:             # Publish market event
179:             self.event_bus.publish(market_event)
180: 
181:             # Process all events in queue (market -> signal -> order -> fill)
182:             events_in_cycle = self.event_bus.process_all()
183:             self.events_processed += events_in_cycle + 1  # +1 for market event
184: 
185:             # Update portfolio valuations
186:             self.portfolio.update_market_value(market_event)
187: 
188:             # Log progress periodically
189:             if self.events_processed % 10000 == 0:
190:                 logger.info(f"Processed {self.events_processed:,} events")
191: 
192:         # Finalize
193:         self.strategy.on_end()
194:         self.broker.finalize()
195:         self.portfolio.finalize()
196:         self.reporter.on_end()
197: 
198:         self.end_time = datetime.now()
199:         duration = (self.end_time - self.start_time).total_seconds()
200: 
201:         logger.info(
202:             f"Backtest complete: {self.events_processed:,} events in {duration:.2f}s "
203:             f"({self.events_processed / duration:.0f} events/sec)",
204:         )
205: 
206:         # Compile results
207:         results = self._compile_results()
208:         return results
209: 
210:     def _compile_results(self) -> dict[str, Any]:
211:         """Compile backtest results from all components.
212: 
213:         Returns:
214:             Dictionary with comprehensive backtest results
215:         """
216:         # Get data from components
217:         trades = self.broker.get_trades()
218:         positions = self.portfolio.get_positions()
219:         returns = self.portfolio.get_returns()
220:         metrics = self.portfolio.calculate_metrics()
221: 
222:         # Add engine statistics
223:         duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
224: 
225:         results = {
226:             "trades": trades,
227:             "positions": positions,
228:             "returns": returns,
229:             "metrics": metrics,
230:             "events_processed": self.events_processed,
231:             "duration_seconds": duration,
232:             "events_per_second": self.events_processed / duration if duration > 0 else 0,
233:             "initial_capital": self.initial_capital,
234:             "final_value": self.portfolio.get_total_value(),
235:             "total_return": (self.portfolio.get_total_value() / self.initial_capital - 1) * 100,
236:         }
237: 
238:         # Add reporter data if available
239:         if hasattr(self.reporter, "get_report"):
240:             results["report"] = self.reporter.get_report()
241: 
242:         return results
243: 
244:     def reset(self) -> None:
245:         """Reset the engine for another run."""
246:         logger.info("Resetting backtest engine")
247: 
248:         # Clear event bus
249:         self.event_bus.clear()
250: 
251:         # Reset components
252:         self.data_feed.reset()
253:         self.strategy.reset()
254:         self.broker.reset()
255:         self.portfolio.reset()
256:         self.reporter.reset()
257: 
258:         # Reset statistics
259:         self.events_processed = 0
260:         self.start_time = None
261:         self.end_time = None
262: 
263:         # Re-setup event handlers
264:         self._setup_event_handlers()
265: 
266: 
267: class BacktestResults:
268:     """Container for backtest results with analysis methods."""
269: 
270:     def __init__(self, results: dict[str, Any]):
271:         """Initialize with results dictionary from BacktestEngine.
272: 
273:         Args:
274:             results: Results dictionary from engine.run()
275:         """
276:         self.results = results
277:         self.trades = results.get("trades", pl.DataFrame())
278:         self.positions = results.get("positions", pl.DataFrame())
279:         self.returns = results.get("returns", pl.Series())
280:         self.metrics = results.get("metrics", {})
281: 
282:     @property
283:     def total_return(self) -> float:
284:         """Total return percentage."""
285:         return self.results.get("total_return", 0.0)
286: 
287:     @property
288:     def sharpe_ratio(self) -> float:
289:         """Sharpe ratio of returns."""
290:         return self.metrics.get("sharpe_ratio", 0.0)
291: 
292:     @property
293:     def max_drawdown(self) -> float:
294:         """Maximum drawdown percentage."""
295:         return self.metrics.get("max_drawdown", 0.0)
296: 
297:     @property
298:     def win_rate(self) -> float:
299:         """Percentage of winning trades."""
300:         if self.trades.is_empty():
301:             return 0.0
302:         winning = self.trades.filter(pl.col("pnl") > 0)
303:         return len(winning) / len(self.trades) * 100
304: 
305:     def summary(self) -> str:
306:         """Generate a text summary of results.
307: 
308:         Returns:
309:             Formatted summary string
310:         """
311:         return f"""
312: Backtest Results Summary
313: ========================
314: Total Return: {self.total_return:.2f}%
315: Sharpe Ratio: {self.sharpe_ratio:.2f}
316: Max Drawdown: {self.max_drawdown:.2f}%
317: Win Rate: {self.win_rate:.2f}%
318: Total Trades: {len(self.trades):,}
319: Events Processed: {self.results.get("events_processed", 0):,}
320: Duration: {self.results.get("duration_seconds", 0):.2f}s
321:         """.strip()
322: 
323:     def to_dict(self) -> dict[str, Any]:
324:         """Convert to dictionary for serialization.
325: 
326:         Returns:
327:             Dictionary of results
328:         """
329:         return self.results
330: 
331:     def save(self, path: str) -> None:
332:         """Save results to file.
333: 
334:         Args:
335:             path: Output file path (supports .parquet, .json, .html)
336:         """
337:         if path.endswith(".parquet"):
338:             # Save DataFrames to parquet
339:             self.trades.write_parquet(path.replace(".parquet", "_trades.parquet"))
340:             self.positions.write_parquet(path.replace(".parquet", "_positions.parquet"))
341:         elif path.endswith(".json"):
342:             # Save as JSON
343:             import json
344: 
345:             with open(path, "w") as f:
346:                 # Convert non-serializable objects
347:                 data = {
348:                     k: v if not isinstance(v, (pl.DataFrame, pl.Series)) else None
349:                     for k, v in self.results.items()
350:                 }
351:                 json.dump(data, f, indent=2, default=str)
352:         elif path.endswith(".html"):
353:             # Generate HTML report
354:             from qengine.reporting.html import generate_html_report
355: 
356:             html = generate_html_report(self)
357:             with open(path, "w") as f:
358:                 f.write(html)
359:         else:
360:             raise ValueError(f"Unsupported file format: {path}")
361: 
362: 
363: __all__ = [
364:     "BacktestEngine",
365:     "BacktestResults",
366: ]
````

## File: CLAUDE.md
````markdown
  1: # CLAUDE.md - QEngine Development Guidelines
  2: 
  3: ## Project Overview
  4: 
  5: QEngine is a state-of-the-art event-driven backtesting engine designed for machine learning-driven trading strategies. Built on modern Python tooling (Polars, Arrow, Numba), it provides institutional-grade simulation capabilities while preventing data leakage through architectural guarantees.
  6: 
  7: ## Key Documentation
  8: 
  9: ### Architecture & Design
 10: - **Engine Architecture**: `.claude/reference/ARCHITECTURE.md`
 11: - **Simulation Design**: `.claude/reference/SIMULATION.md`
 12: - **Functionality Inventory**: `FUNCTIONALITY_INVENTORY.md` - Complete list of implemented features (UPDATE EACH SPRINT)
 13: - **Integration with Monorepo**: See `/home/stefan/quantlab/CLAUDE.md`
 14: 
 15: ### Session Management
 16: - **Current Session**: Managed at monorepo level in `/home/stefan/quantlab/.claude/memory/CURRENT_SESSION_CONTEXT.md`
 17: - **Integration Tracking**: `/home/stefan/quantlab/.claude/tracking/INTEGRATION_TRACKER.md`
 18: 
 19: ## Core Principles
 20: 
 21: 1. **Point-in-Time Correctness**: Architectural guarantees against data leakage
 22: 2. **Performance First**: Polars columnar data, Numba JIT, optional Rust extensions
 23: 3. **Extensibility**: Everything pluggable through well-defined ABCs
 24: 4. **Repository Tidiness**: Clear separation of planning, documentation, and code
 25: 5. **Code Quality**: State-of-the-art tooling (Ruff, MyPy, pre-commit)
 26: 6. **Test-Driven Development**: Write tests first, minimum 80% coverage
 27: 
 28: ## Monorepo Context
 29: 
 30: QEngine is part of the **QuantLab monorepo**, a comprehensive quantitative finance ecosystem:
 31: 
 32: - **qfeatures**: Feature engineering and labeling (upstream)
 33: - **qeval**: Statistical validation (upstream)
 34: - **qengine** (this project): Event-driven backtesting engine
 35: 
 36: ### Development Setup
 37: 
 38: Follows monorepo conventions:
 39: 
 40: ```bash
 41: # From monorepo root
 42: make setup       # One-time setup
 43: make test-qng    # Test qengine specifically
 44: make quality     # Format, lint, type-check all projects
 45: ```
 46: 
 47: ### Calendar Integration
 48: 
 49: - **Exchange Calendars**: Follow `/home/stefan/quantlab/.claude/reference/CALENDAR_USAGE_GUIDE.md`
 50: - **Sensible Defaults**: Auto-detect calendar based on asset type
 51: - **Optional Alignment**: Support both sparse and dense time series
 52: 
 53: ### Integration Points
 54: 
 55: QEngine consumes:
 56: 1. **Features from qfeatures**: Standardized DataFrames with features and labels
 57: 2. **Models from qeval**: Validated ML models ready for backtesting
 58: 
 59: ```python
 60: # Expected input from qfeatures/qeval pipeline
 61: {
 62:     "event_time": pl.Datetime("ns"),    # Point-in-time anchor
 63:     "asset_id": pl.Utf8,                # Instrument identifier
 64:     "features...": pl.Float64,          # Feature columns
 65:     "signal": pl.Float64,               # Model predictions
 66: }
 67: ```
 68: 
 69: ## Project Structure
 70: 
 71: ```
 72: qengine/
 73: ├── CLAUDE.md               # This file - project guidelines
 74: ├── README.md               # User-facing project overview
 75: ├── LICENSE                 # Apache 2.0 license
 76: ├── pyproject.toml          # Modern Python packaging
 77: ├── Makefile                # Developer workflow commands
 78: ├── .pre-commit-config.yaml # Code quality hooks
 79: ├── .claude/                # Claude AI workspace (NEVER in root!)
 80: │   ├── planning/           # Implementation plans, roadmaps
 81: │   │   ├── IMPLEMENTATION_PLAN.md
 82: │   │   ├── FRAMEWORK_ANALYSIS.md
 83: │   │   └── ROADMAP.md
 84: │   ├── reference/          # Design reviews, architecture decisions
 85: │   │   ├── DESIGN.md       # Original design specification
 86: │   │   └── ARCHITECTURE_REVIEW.md
 87: │   ├── sprints/            # Sprint-based development
 88: │   └── PROJECT_GUIDELINES.md # Repo organization rules
 89: ├── docs/                   # User documentation
 90: │   ├── architecture/       # System design
 91: │   │   ├── ARCHITECTURE.md
 92: │   │   └── PIT_AND_STATE.md
 93: │   └── guides/            # How-to guides
 94: │       └── MIGRATION_GUIDE.md
 95: ├── src/qengine/           # Source code
 96: │   ├── core/              # Event system, clock, types
 97: │   ├── data/              # Data feeds and schemas
 98: │   ├── strategy/          # Strategy framework
 99: │   ├── execution/         # Order execution (TODO)
100: │   ├── portfolio/         # Portfolio management (TODO)
101: │   └── reporting/         # Output generation (TODO)
102: ├── tests/                 # Test suite
103: │   ├── unit/              # Fast, isolated tests
104: │   ├── integration/       # Component interaction tests
105: │   ├── scenarios/         # Golden scenario tests
106: │   ├── comparison/        # Backtester comparison tests
107: │   └── conftest.py        # Pytest configuration
108: ├── examples/              # Example strategies
109: ├── benchmarks/            # Performance benchmarks
110: └── resources/             # Reference implementations (read-only)
111:     ├── backtrader-master/ # Backtrader source
112:     ├── vectorbt.pro-main/ # VectorBT Pro source
113:     └── zipline-reloaded/  # Zipline source
114: ```
115: 
116: ## Critical Design Decisions (from DESIGN.md)
117: 
118: ### 1. Architecture Choice
119: - **Event-Driven Core**: Chosen over pure vectorized for realism and ML support
120: - **Hybrid Approach**: Event-driven for execution, vectorized for analytics
121: - **Rationale**: Balances performance with correctness for complex strategies
122: 
123: ### 2. Technology Stack
124: - **Polars over Pandas**: 10-100x performance improvement
125: - **Arrow Format**: Zero-copy data transfer, language interoperability
126: - **Numba JIT**: Hot path optimization without leaving Python
127: - **Optional Rust**: Core event loop for C-level performance
128: - **Comparison Backtesters**: VectorBT Pro, Zipline-Reloaded, Backtrader (for validation)
129: 
130: ### 3. Data Leakage Prevention
131: - **Clock as Master**: Controls all time advancement
132: - **Immutable PIT Views**: Strategies receive time-bounded data snapshots
133: - **No Direct Access**: Strategies cannot access raw data feeds
134: 
135: ### 4. Migration Strategy (Simplified per Review)
136: - **No Full Compatibility**: Migration helpers, not compatibility layers
137: - **Documentation Focus**: Clear guides over complex code
138: - **Better UX**: Intuitive API will drive adoption
139: 
140: ## Development Workflow
141: 
142: ### 1. Code Quality Standards
143: 
144: **ALWAYS run before ANY code changes:**
145: ```bash
146: make format      # Ruff formatting
147: make lint        # Ruff linting
148: make type-check  # MyPy validation
149: ```
150: 
151: **Or all at once:**
152: ```bash
153: make check       # Format + lint + type-check
154: ```
155: 
156: **Pre-commit hooks (MANDATORY):**
157: ```bash
158: make dev-install  # Installs pre-commit hooks
159: pre-commit run --all-files  # Manual run
160: ```
161: 
162: ### 2. Repository Organization Rules
163: 
164: **ABSOLUTE RULES:**
165: 1. **Root Directory**: ONLY README.md, LICENSE, pyproject.toml, Makefile, configs
166: 2. **Planning Docs**: ALWAYS in `.claude/planning/`
167: 3. **Reference Docs**: ALWAYS in `.claude/reference/`
168: 4. **User Docs**: ALWAYS in `docs/`
169: 5. **NEVER**: Leave work-in-progress files in root
170: 6. **NEVER**: Create compatibility layers without explicit request
171: 
172: ### 3. Test-Driven Development
173: 
174: **Workflow:**
175: 1. Write test FIRST in `tests/unit/` or `tests/integration/`
176: 2. Run test to see it fail: `pytest tests/unit/test_feature.py::test_specific -v`
177: 3. Implement minimal code to pass
178: 4. Refactor while keeping tests green
179: 5. Run full suite: `make test`
180: 
181: **Testing Standards:**
182: - Minimum 80% coverage (enforced)
183: - Use pytest fixtures from `conftest.py`
184: - Mock external dependencies
185: - Property-based tests for algorithms
186: 
187: ### 4. Performance Considerations
188: 
189: **From DESIGN.md - Performance Targets:**
190: - >1M events/second for simple strategies
191: - <30 seconds for 1 year tick data
192: - <10 seconds for 10 years daily data (1000 assets)
193: - <1GB memory for 10-year backtest
194: 
195: **Optimization Priority:**
196: 1. Algorithmic complexity first
197: 2. Polars query optimization
198: 3. Numba JIT for tight loops
199: 4. Rust only for proven bottlenecks
200: 
201: ### 5. Git Workflow
202: 
203: **Branch Strategy:**
204: ```bash
205: git checkout -b feature/add-broker-simulation
206: git checkout -b fix/event-ordering-bug
207: git checkout -b docs/improve-pit-explanation
208: ```
209: 
210: **Commit Messages:**
211: ```bash
212: git commit -m "feat: Add SimulationBroker with basic order matching
213: 
214: - Implement Market and Limit order types
215: - Add order state machine
216: - Include unit tests with 95% coverage"
217: ```
218: 
219: **NEVER commit unless:**
220: - All tests pass: `make test`
221: - Code quality checks pass: `make qa`
222: - No files in wrong directories
223: 
224: ## Implementation Phases (from IMPLEMENTATION_PLAN.md)
225: 
226: ### Phase 1: MVP Core Engine (Current)
227: - [x] Project structure and infrastructure
228: - [x] Core event system and Clock
229: - [x] Basic data feeds (Parquet, CSV)
230: - [x] Strategy framework
231: - [ ] Simple broker simulation
232: - [ ] Portfolio accounting
233: - [ ] Basic reporting
234: 
235: ### Phase 2: Realistic Simulation
236: - [ ] Advanced order types (Stop, Bracket, OCO)
237: - [ ] Pluggable models (slippage, commission, impact)
238: - [ ] Corporate actions and calendars
239: - [ ] Futures support
240: - [ ] ML signal integration
241: 
242: ### Phase 3: Performance & Differentiation
243: - [ ] Rust core optimization
244: - [ ] Almgren-Chriss market impact
245: - [ ] Portfolio optimization adapters
246: - [ ] TOML/YAML configuration
247: 
248: ### Phase 4: Live Trading
249: - [ ] Broker adapters (Alpaca, IB)
250: - [ ] Real-time data feeds
251: - [ ] State persistence
252: 
253: ## Common Patterns & Solutions
254: 
255: ### Adding a New Component
256: 1. Define interface in appropriate module's `__init__.py`
257: 2. Create ABC in module directory
258: 3. Write comprehensive unit tests
259: 4. Implement concrete class
260: 5. Add integration tests
261: 6. Update documentation
262: 
263: ### Data Feed Implementation
264: ```python
265: # Always follow this pattern from data/feed.py
266: class CustomDataFeed(DataFeed):
267:     def get_next_event(self) -> Optional[Event]:
268:         # Implementation
269: 
270:     def peek_next_timestamp(self) -> Optional[datetime]:
271:         # Implementation
272: 
273:     @property
274:     def is_exhausted(self) -> bool:
275:         # Implementation
276: ```
277: 
278: ### Strategy Implementation
279: ```python
280: # Follow pattern from strategy/base.py
281: class CustomStrategy(Strategy):
282:     def on_start(self):
283:         # One-time setup
284: 
285:     def on_event(self, event: Event, pit_data: PITData):
286:         # Core logic with PIT-safe data access
287: ```
288: 
289: ## Key Technical Decisions
290: 
291: ### From Architecture Review Feedback
292: 
293: 1. **PIT Enforcement**: Immutable data views, not trust-based access
294: 2. **State Management**: Centralized PortfolioState with atomic updates
295: 3. **Migration Approach**: Helpers and docs, not full compatibility
296: 4. **Performance Trade-offs**: Correctness over speed, but optimize hot paths
297: 
298: ### From Framework Analysis
299: 
300: 1. **Avoid Backtrader's**: Metaclass magic, pure Python performance
301: 2. **Avoid Zipline's**: Heavy dependencies, Cython complexity
302: 3. **Avoid VectorBT's**: Sacrificing realism for speed
303: 4. **Adopt Best Practices**: Clean APIs, modern tooling, extensibility
304: 
305: ## Testing Requirements
306: 
307: ### Test Categories
308: - **Unit Tests**: `tests/unit/` - Fast, isolated, mocked
309: - **Integration Tests**: `tests/integration/` - Component interactions
310: - **Scenario Tests**: `tests/scenarios/` - Golden outputs
311: - **Benchmark Tests**: `tests/benchmarks/` - Performance tracking
312: - **Comparison Tests**: `tests/comparison/` - Validation against established backtesters
313: 
314: ### Running Tests
315: ```bash
316: make test           # All tests
317: make test-unit      # Unit tests only
318: make test-cov       # With coverage report
319: pytest tests/unit/test_event_bus.py -v  # Specific test
320: ```
321: 
322: ## Documentation Standards
323: 
324: ### Docstrings (Google Style)
325: ```python
326: def calculate_returns(prices: np.ndarray, method: str = "simple") -> np.ndarray:
327:     """Calculate returns from price series.
328: 
329:     Args:
330:         prices: Array of prices
331:         method: Calculation method ('simple' or 'log')
332: 
333:     Returns:
334:         Array of returns
335: 
336:     Raises:
337:         ValueError: If prices array is empty
338:     """
339: ```
340: 
341: ### Type Hints (REQUIRED)
342: ```python
343: from typing import Optional, List, Dict
344: from datetime import datetime
345: 
346: def process_events(
347:     events: List[Event],
348:     timestamp: datetime
349: ) -> Optional[Dict[str, Any]]:
350:     ...
351: ```
352: 
353: ## Common Pitfalls to Avoid
354: 
355: 1. **Don't Add to Root**: Keep it clean, use .claude/ and docs/
356: 2. **Don't Skip Tests**: TDD is mandatory
357: 3. **Don't Ignore Types**: MyPy must pass
358: 4. **Don't Create Deep Nesting**: Keep structure simple
359: 5. **Don't Use sys.path Hacks**: Use proper packaging
360: 6. **Don't Implement Full Compatibility**: Migration helpers only
361: 
362: ## Quick Commands Reference
363: 
364: ```bash
365: # Development
366: make dev-install    # Set up development environment
367: make format         # Auto-format code
368: make lint           # Check code style
369: make type-check     # Validate types
370: make test           # Run tests
371: make qa             # All quality checks
372: 
373: # Git
374: git status          # Check changes
375: git add -p          # Stage selectively
376: git commit          # With descriptive message
377: git push            # After all checks pass
378: 
379: # Testing
380: pytest -v           # Verbose test output
381: pytest -x           # Stop on first failure
382: pytest --pdb        # Debug on failure
383: pytest -k pattern   # Run matching tests
384: 
385: # Profiling
386: python -m cProfile -o profile.stats examples/benchmark.py
387: py-spy record -o profile.svg -- python examples/benchmark.py
388: ```
389: 
390: ## Session Continuity
391: 
392: ### Starting a New Session
393: 1. Read `.claude/reference/DESIGN.md` for design context
394: 2. Check `docs/architecture/ARCHITECTURE.md` for implementation
395: 3. Review `.claude/planning/ROADMAP.md` for current status
396: 4. Continue from documented next steps
397: 
398: ### Before Ending Session
399: 1. Update relevant tracking documents
400: 2. Commit completed work (if requested)
401: 3. Document next steps clearly
402: 4. Ensure no temporary files in root
403: 
404: ## Notes
405: 
406: - This document supersedes generic templates for QEngine
407: - Always prioritize correctness over performance
408: - Keep migration simple - good UX beats compatibility
409: - Reference DESIGN.md for any architectural questions
410: - Follow PROJECT_GUIDELINES.md for repo organization
````

## File: README.md
````markdown
  1: # QEngine
  2: 
  3: QEngine is a state-of-the-art event-driven backtesting engine designed for high-performance backtesting of machine learning-driven trading strategies. It provides realistic market simulation with architectural guarantees against data leakage.
  4: 
  5: ## Installation
  6: 
  7: ```bash
  8: # From source (recommended)
  9: git clone <repository-url>
 10: cd qengine
 11: pip install -e .
 12: ```
 13: 
 14: ## Quick Start
 15: 
 16: ```python
 17: 
 18: import qengine as qe
 19: 
 20: # Load market data
 21: data_feed = qe.DataFeed.from_parquet("market_data.parquet")
 22: 
 23: 
 24: # Define a simple strategy
 25: class BuyAndHoldStrategy(qe.Strategy):
 26:     def on_market_data(self, event, pit_data):
 27:         if not self.portfolio.has_position("AAPL"):
 28:             order = qe.MarketOrder(
 29:                 asset_id="AAPL",
 30:                 quantity=100,
 31:                 side="BUY"
 32:             )
 33:             self.submit_order(order)
 34: 
 35: 
 36: # Create and run backtest
 37: engine = qe.BacktestEngine(
 38:     data_feed=data_feed,
 39:     strategy=BuyAndHoldStrategy(),
 40:     initial_capital=100000
 41: )
 42: 
 43: results = engine.run()
 44: print(f"Final Value: ${results.final_value:,.2f}")
 45: print(f"Total Return: {results.total_return:.2%}")
 46: ```
 47: 
 48: ## Key Features
 49: 
 50: ### Event-Driven Architecture
 51: - **True Event Processing**: Bar-by-bar simulation with realistic timing
 52: - **Point-in-Time Data**: Architectural guarantees against look-ahead bias
 53: - **Event System**: Extensible event-driven framework for complex strategies
 54: 
 55: ### Realistic Execution Simulation
 56: - **Order Types**: Market, Limit, Stop, Trailing Stop, Bracket orders
 57: - **Slippage Models**: 7 models including Almgren-Chriss market impact
 58: - **Commission Models**: 9 models including tiered and percentage-based
 59: - **Order Matching**: Realistic order book simulation
 60: 
 61: ### Machine Learning Integration
 62: - **Strategy Adapters**: Bridge existing ML models to backtesting framework
 63: - **Model Lifecycle**: Handle model training, prediction, and updates
 64: - **Feature Pipeline**: Integration with QFeatures for online feature generation
 65: 
 66: ### Performance & Accuracy
 67: - **Validated Results**: 100% agreement with VectorBT on identical strategies
 68: - **High Throughput**: 8,552 trades/second processing capability
 69: - **Memory Efficient**: Polars-based data handling for large datasets
 70: 
 71: ## Architecture
 72: 
 73: QEngine follows a clean event-driven architecture:
 74: 
 75: ### Core Components
 76: ```python
 77: # Event system
 78: from qengine.core import Event, EventBus, Clock
 79: 
 80: # Data pipeline
 81: from qengine.data import DataFeed
 82: 
 83: # Strategy framework
 84: from qengine.strategy import Strategy
 85: 
 86: # Portfolio management
 87: from qengine.portfolio import Portfolio
 88: 
 89: # Order execution
 90: from qengine.execution import Broker, Order
 91: ```
 92: 
 93: ### Execution Pipeline
 94: 1. **Clock**: Controls simulation time advancement
 95: 2. **DataFeed**: Provides market data events
 96: 3. **Strategy**: Receives events and generates orders
 97: 4. **Broker**: Executes orders with realistic simulation
 98: 5. **Portfolio**: Tracks positions, cash, and performance
 99: 6. **Reporter**: Generates results and analytics
100: 
101: ## Order Types
102: 
103: QEngine supports sophisticated order types:
104: 
105: ```python
106: # Market orders
107: market_order = qe.MarketOrder(asset_id="AAPL", quantity=100, side="BUY")
108: 
109: # Limit orders
110: limit_order = qe.LimitOrder(
111:     asset_id="AAPL",
112:     quantity=100,
113:     side="BUY",
114:     limit_price=150.0
115: )
116: 
117: # Stop orders
118: stop_order = qe.StopOrder(
119:     asset_id="AAPL",
120:     quantity=100,
121:     side="SELL",
122:     stop_price=140.0
123: )
124: 
125: # Bracket orders (OCO)
126: bracket = qe.BracketOrder(
127:     parent=market_order,
128:     take_profit=155.0,
129:     stop_loss=145.0
130: )
131: ```
132: 
133: ## Execution Models
134: 
135: ### Slippage Models
136: ```python
137: from qengine.execution.slippage import LinearImpactSlippage, VolumeShareSlippage
138: 
139: # Linear market impact
140: slippage = LinearImpactSlippage(impact_coefficient=0.1)
141: 
142: # Volume-based slippage
143: vol_slippage = VolumeShareSlippage(max_participation=0.1)
144: ```
145: 
146: ### Commission Models
147: ```python
148: from qengine.execution.commission import PercentageCommission, TieredCommission
149: 
150: # Simple percentage
151: commission = PercentageCommission(rate=0.001)  # 10 basis points
152: 
153: # Tiered pricing
154: tiered = TieredCommission({
155:     0: 0.005,      # First $100k
156:     100000: 0.003, # Next tier
157:     500000: 0.001  # Highest tier
158: })
159: ```
160: 
161: ## Strategy Development
162: 
163: ### Basic Strategy
164: ```python
165: class MomentumStrategy(qe.Strategy):
166:     def __init__(self, lookback=20):
167:         super().__init__()
168:         self.lookback = lookback
169: 
170:     def on_market_data(self, event, pit_data):
171:         # Calculate momentum signal
172:         returns = pit_data.returns.tail(self.lookback)
173:         momentum = returns.mean()
174: 
175:         if momentum > 0.01:  # Buy signal
176:             self.rebalance_to_target("AAPL", 0.5)  # 50% allocation
177:         elif momentum < -0.01:  # Sell signal
178:             self.close_position("AAPL")
179: ```
180: 
181: ### ML Strategy Integration
182: ```python
183: from qengine.strategy.adapters import MLStrategyAdapter
184: 
185: class MLMomentumStrategy(MLStrategyAdapter):
186:     def predict(self, features):
187:         return self.model.predict(features.to_numpy())
188: 
189:     def generate_orders(self, predictions, pit_data):
190:         for asset_id, signal in predictions.items():
191:             if signal > 0.6:
192:                 yield qe.MarketOrder(asset_id, 100, "BUY")
193:             elif signal < 0.4:
194:                 yield qe.MarketOrder(asset_id, 100, "SELL")
195: ```
196: 
197: ## Performance Analysis
198: 
199: QEngine provides comprehensive performance analytics:
200: 
201: ```python
202: results = engine.run()
203: 
204: # Portfolio metrics
205: print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
206: print(f"Max Drawdown: {results.max_drawdown:.2%}")
207: print(f"Win Rate: {results.win_rate:.1%}")
208: 
209: # Trade analysis
210: print(f"Total Trades: {len(results.trades)}")
211: print(f"Avg Trade Return: {results.avg_trade_return:.2%}")
212: 
213: # Generate reports
214: results.to_html("backtest_report.html")
215: results.to_parquet("detailed_trades.parquet")
216: ```
217: 
218: ## Integration with QuantLab
219: 
220: QEngine integrates seamlessly with other QuantLab libraries:
221: 
222: ```python
223: import qfeatures as qf
224: import qeval as qe
225: import qengine as qng
226: 
227: # Feature engineering with QFeatures
228: features = qf.Pipeline([
229:     qf.features.microstructure.add_returns,
230:     qf.features.volatility.add_garch_features,
231:     qf.labeling.apply_triple_barrier
232: ]).transform(price_data)
233: 
234: # Model validation with QEval
235: cv = qe.PurgedWalkForwardCV(n_splits=5)
236: model_results = qe.Evaluator(cv).evaluate(model, features)
237: 
238: # Strategy backtesting with QEngine
239: strategy = qng.MLStrategyAdapter(model)
240: backtest_results = qng.BacktestEngine(strategy=strategy).run()
241: ```
242: 
243: ## Validation & Testing
244: 
245: QEngine has been extensively validated:
246: 
247: - **154 unit tests** with comprehensive coverage
248: - **Cross-framework validation**: 100% agreement with VectorBT
249: - **Multi-asset testing**: 5,000 trades across 30 stocks
250: - **Performance benchmarking**: 8,552 trades/second processing
251: 
252: ## Contributing
253: 
254: 1. Install development dependencies: `pip install -e ".[dev]"`
255: 2. Run code quality checks: `ruff format . && ruff check . --fix && mypy src/qengine`
256: 3. Run tests: `pytest tests/`
257: 4. Follow event-driven patterns and maintain type safety
258: 
259: ## License
260: 
261: Apache License 2.0
````
