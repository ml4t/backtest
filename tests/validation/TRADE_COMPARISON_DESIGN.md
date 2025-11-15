# Trade-by-Trade Comparison Architecture

## Philosophy

**Never compare aggregates**. Always compare individual trades at the most granular level:
1. Entry/exit timestamps
2. Entry/exit prices
3. Which OHLC component was used
4. Fees and slippage
5. P&L calculations

## Standard Trade Representation

Every platform must convert its trade format to this standard:

```python
@dataclass
class StandardTrade:
    """Platform-independent trade representation."""

    # Identity
    trade_id: int                    # Sequential ID within platform
    platform: str                    # 'ml4t.backtest', 'vectorbt', 'backtrader', 'zipline'

    # Entry
    entry_timestamp: datetime        # When did we enter?
    entry_price: float              # At what price?
    entry_price_component: str      # 'open', 'close', 'high', 'low', 'unknown'
    entry_bar_ohlc: dict           # Complete OHLC of entry bar for analysis

    # Exit
    exit_timestamp: datetime | None # When did we exit? (None if still open)
    exit_price: float | None       # At what price?
    exit_price_component: str | None  # 'open', 'close', 'high', 'low', 'unknown'
    exit_bar_ohlc: dict | None     # Complete OHLC of exit bar
    exit_reason: str | None        # 'signal', 'stop_loss', 'take_profit', 'trailing_stop'

    # Trade details
    symbol: str
    quantity: float
    side: str                      # 'long' or 'short'

    # Economics
    gross_pnl: float | None        # Price difference * quantity
    entry_commission: float        # Commission paid on entry
    exit_commission: float         # Commission paid on exit
    slippage: float                # Total slippage (entry + exit)
    net_pnl: float | None          # After all costs

    # Metadata
    signal_id: str | None          # Link back to originating signal
    notes: str | None              # Platform-specific notes
```

## Platform Extractors

Each platform needs an extractor that produces `List[StandardTrade]`:

### ml4t.backtest Extractor

```python
def extract_ml4t.backtest_trades(results: dict, data: pl.DataFrame) -> List[StandardTrade]:
    """
    Extract trades from ml4t.backtest results.

    ml4t.backtest returns individual orders, so we must:
    1. Match BUY orders with SELL orders to form complete trades
    2. Look up OHLC at entry/exit timestamps from data
    3. Infer which price component was used (likely close for market orders)
    """
    trades_df = results['trades']  # Has: order_id, asset_id, side, quantity, price, filled_time, commission

    # Match BUY/SELL into complete trades
    open_positions = {}
    standard_trades = []

    for row in trades_df.sort('filled_time').iter_rows(named=True):
        if row['side'] == 'buy':
            # Open position
            open_positions[row['asset_id']] = {
                'entry_timestamp': row['filled_time'],
                'entry_price': row['price'],
                'quantity': row['quantity'],
                'entry_commission': row['commission'],
            }
        elif row['side'] == 'sell' and row['asset_id'] in open_positions:
            # Close position
            entry = open_positions.pop(row['asset_id'])

            # Look up OHLC at entry/exit
            entry_bar = get_bar_at_timestamp(data, entry['entry_timestamp'])
            exit_bar = get_bar_at_timestamp(data, row['filled_time'])

            # Infer which component (compare price to OHLC)
            entry_component = infer_price_component(entry['entry_price'], entry_bar)
            exit_component = infer_price_component(row['price'], exit_bar)

            standard_trades.append(StandardTrade(
                trade_id=len(standard_trades),
                platform='ml4t.backtest',
                entry_timestamp=entry['entry_timestamp'],
                entry_price=entry['entry_price'],
                entry_price_component=entry_component,
                entry_bar_ohlc=entry_bar,
                exit_timestamp=row['filled_time'],
                exit_price=row['price'],
                exit_price_component=exit_component,
                exit_bar_ohlc=exit_bar,
                # ... etc
            ))

    return standard_trades
```

### VectorBT Extractor

```python
def extract_vectorbt_trades(portfolio, data: pd.DataFrame) -> List[StandardTrade]:
    """
    Extract trades from VectorBT portfolio.

    VectorBT provides trades.records_readable with:
    - Entry Index (timestamp)
    - Exit Index (timestamp)
    - Avg Entry Price
    - Avg Exit Price
    - Size
    - PnL
    - Entry Fees, Exit Fees
    """
    trades_df = portfolio.trades.records_readable
    standard_trades = []

    for idx, row in trades_df.iterrows():
        entry_ts = row['Entry Index']
        exit_ts = row.get('Exit Index')

        # Look up OHLC
        entry_bar = data.loc[entry_ts] if entry_ts in data.index else None
        exit_bar = data.loc[exit_ts] if exit_ts in data.index else None

        # Infer components
        entry_component = infer_price_component(row['Avg Entry Price'], entry_bar)
        exit_component = infer_price_component(row.get('Avg Exit Price'), exit_bar) if exit_bar else None

        standard_trades.append(StandardTrade(
            trade_id=idx,
            platform='vectorbt',
            entry_timestamp=entry_ts,
            entry_price=row['Avg Entry Price'],
            entry_price_component=entry_component,
            entry_bar_ohlc=dict(entry_bar) if entry_bar is not None else {},
            exit_timestamp=exit_ts if pd.notna(exit_ts) else None,
            exit_price=row.get('Avg Exit Price') if pd.notna(row.get('Avg Exit Price')) else None,
            exit_price_component=exit_component,
            exit_bar_ohlc=dict(exit_bar) if exit_bar is not None else {},
            # ... etc
        ))

    return standard_trades
```

## Helper Functions

```python
def get_bar_at_timestamp(data: pl.DataFrame, timestamp: datetime) -> dict:
    """Get OHLC bar at specific timestamp."""
    bar = data.filter(pl.col('timestamp') == timestamp)
    if len(bar) == 0:
        return {}
    row = bar[0]
    return {
        'timestamp': row['timestamp'][0],
        'open': row['open'][0],
        'high': row['high'][0],
        'low': row['low'][0],
        'close': row['close'][0],
        'volume': row['volume'][0],
    }

def infer_price_component(price: float, bar: dict, tolerance: float = 0.01) -> str:
    """
    Infer which OHLC component was used based on price.

    Args:
        price: The actual execution price
        bar: OHLC bar dict
        tolerance: Price match tolerance (1%)

    Returns:
        'open', 'close', 'high', 'low', or 'unknown'
    """
    if not bar or price is None:
        return 'unknown'

    components = {
        'open': bar.get('open'),
        'high': bar.get('high'),
        'low': bar.get('low'),
        'close': bar.get('close'),
    }

    # Check exact match first
    for name, value in components.items():
        if value and abs(price - value) < 0.001:
            return name

    # Check within tolerance
    for name, value in components.items():
        if value and abs(price - value) / value < tolerance:
            return name

    return 'unknown'
```

## Trade Matcher

```python
@dataclass
class TradeMatch:
    """Result of matching a trade across platforms."""

    # Matched trades (one per platform)
    ml4t.backtest_trade: StandardTrade | None
    vectorbt_trade: StandardTrade | None
    backtrader_trade: StandardTrade | None
    zipline_trade: StandardTrade | None

    # Match quality
    entry_timestamp_match: bool
    exit_timestamp_match: bool
    entry_price_match: float  # % difference
    exit_price_match: float   # % difference

    # Component usage
    entry_components: dict[str, str]  # platform -> component
    exit_components: dict[str, str]

    # Differences
    differences: List[str]
    severity: str  # 'none', 'minor', 'major', 'critical'

def match_trades(
    trades_by_platform: dict[str, List[StandardTrade]],
    timestamp_tolerance_seconds: int = 60
) -> List[TradeMatch]:
    """
    Match trades across platforms.

    Primary key: entry_timestamp (with tolerance)
    Secondary validation: exit_timestamp

    Returns list of TradeMatch objects showing which trades
    correspond across platforms and what differs.
    """
    # Group all trades by entry timestamp (with tolerance)
    # ...
    # For each group, create TradeMatch
    # ...
    pass
```

## Comparison Report

```python
def generate_comparison_report(matches: List[TradeMatch]) -> str:
    """
    Generate human-readable comparison report.

    Format:
    ```
    Trade 1: Entry 2020-02-04 09:30:00
    =====================================

    Entry Timing:
      ml4t.backtest   : 2020-02-04 09:30:00 ✅
      vectorbt  : 2020-02-04 09:30:00 ✅
      backtrader: 2020-02-04 09:31:00 ❌ (+60s)
      zipline   : 2020-02-04 09:30:00 ✅

    Entry Prices:
      ml4t.backtest   : $73.50 (open)  ✅
      vectorbt  : $73.85 (close) ⚠️  +0.48% (+$0.35)
      backtrader: $73.50 (open)  ✅
      zipline   : $73.50 (open)  ✅

    Exit Timing:
      ml4t.backtest   : 2020-04-16 16:00:00 ✅
      vectorbt  : 2020-04-15 16:00:00 ❌ (-1 day)
      backtrader: 2020-04-16 16:00:00 ✅
      zipline   : 2020-04-16 16:00:00 ✅

    Exit Prices:
      ml4t.backtest   : $84.88 (open)  ✅
      vectorbt  : $84.25 (close) ⚠️  -0.74% (-$0.63)
      backtrader: $84.88 (open)  ✅
      zipline   : $84.88 (open)  ✅

    P&L Comparison:
      ml4t.backtest   : $970.31 (net) ✅
      vectorbt  : $896.50 (net) ⚠️  -7.6% (-$73.81)
      backtrader: $970.31 (net) ✅
      zipline   : $968.12 (net) ✅ (rounding)

    VERDICT:
    • ml4t.backtest, backtrader, zipline: MATCH ✅
    • vectorbt: Different execution model (same-bar vs next-bar) ⚠️

    =====================================
    ```
    """
    pass
```

## Key Decisions

1. **Timestamp is primary key** - We match trades by entry timestamp first
2. **Price component matters** - We must identify open/close/high/low
3. **OHLC bar included** - For forensic analysis of price differences
4. **Tolerance for matching** - 60 seconds for timestamp, 1% for price
5. **Clear severity levels** - none/minor/major/critical
6. **Trade-by-trade reports** - Never just aggregate statistics

## Implementation Order

1. ✅ StandardTrade dataclass
2. ✅ Helper functions (get_bar_at_timestamp, infer_price_component)
3. ✅ ml4t.backtest extractor
4. ✅ VectorBT extractor
5. ⏳ Backtrader extractor
6. ⏳ Zipline extractor
7. ✅ Trade matcher
8. ✅ Comparison report generator
9. ✅ HTML report formatter
