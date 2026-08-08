# API Reference

Auto-generated from source docstrings.

## Core

::: ml4t.backtest.engine.Engine
    options:
      show_root_heading: true
      members:
        - run
        - run_dict
        - from_config

::: ml4t.backtest.engine.run_backtest
    options:
      show_root_heading: true

::: ml4t.backtest.strategy.Strategy
    options:
      show_root_heading: true
      members:
        - on_prepare
        - on_start
        - on_before_risk
        - on_data
        - on_end

::: ml4t.backtest.datafeed.DataFeed
    options:
      show_root_heading: true

## Configuration

::: ml4t.backtest.config.BacktestConfig
    options:
      show_root_heading: true
      members:
        - from_preset
        - from_yaml
        - from_dict
        - to_yaml
        - to_dict
        - validate
        - describe

::: ml4t.backtest.profiles
    options:
      show_root_heading: true
      members:
        - get_profile_config
        - list_profiles

## Broker

::: ml4t.backtest.broker.Broker
    options:
      show_root_heading: true
      members:
        - from_config
        - submit_order
        - submit_bracket
        - buy
        - sell
        - close_position
        - reduce_position
        - flatten_all_positions
        - order_target_percent
        - order_target_value
        - rebalance_to_weights
        - update_order
        - cancel_order
        - get_order
        - orders
        - get_pending_orders
        - pending_orders
        - get_rejected_orders
        - get_position
        - get_positions
        - positions
        - get_cash
        - cash
        - equity
        - get_account_value
        - get_buying_power
        - get_trades
        - trades
        - fills
        - get_last_trade
        - last_rejection_reason
        - set_position_rules
        - clear_position_rules
        - update_position_context
        - get_contract_spec
        - get_multiplier
        - get_mark_price
        - get_last_price
        - get_price_for_source
        - get_quote_mid
        - get_quote_context
        - get_available_size
        - configure_stats
        - get_asset_stats
        - set_session_config
        - mark_account_positions
        - evaluate_position_rules

## Domain Types

::: ml4t.backtest.types.Order
    options:
      show_root_heading: true

::: ml4t.backtest.types.Fill
    options:
      show_root_heading: true

::: ml4t.backtest.types.Trade
    options:
      show_root_heading: true

::: ml4t.backtest.types.Position
    options:
      show_root_heading: true

## Enums

::: ml4t.backtest.types.OrderType
    options:
      show_root_heading: true

::: ml4t.backtest.types.OrderSide
    options:
      show_root_heading: true

::: ml4t.backtest.types.ExecutionMode
    options:
      show_root_heading: true

::: ml4t.backtest.types.StopFillMode
    options:
      show_root_heading: true

::: ml4t.backtest.config.CommissionType
    options:
      show_root_heading: true

::: ml4t.backtest.config.SlippageType
    options:
      show_root_heading: true

::: ml4t.backtest.config.FillOrdering
    options:
      show_root_heading: true

## Results

::: ml4t.backtest.result.BacktestResult
    options:
      show_root_heading: true
      members:
        - from_parquet
        - to_trades_dataframe
        - to_fills_dataframe
        - to_rejected_orders_dataframe
        - to_portfolio_state_dataframe
        - to_predictions_dataframe
        - to_equity_dataframe
        - to_daily_pnl
        - to_daily_returns
        - to_returns_series
        - to_trade_records
        - to_dict
        - to_spec_dict
        - to_parquet
        - __getitem__
        - get
        - keys
        - items

## Execution: Market Impact

::: ml4t.backtest.execution.impact.LinearImpact
    options:
      show_root_heading: true

::: ml4t.backtest.execution.impact.SquareRootImpact
    options:
      show_root_heading: true

::: ml4t.backtest.execution.impact.PowerLawImpact
    options:
      show_root_heading: true

## Risk: Position Rules

::: ml4t.backtest.risk.position.static.StopLoss
    options:
      show_root_heading: true

::: ml4t.backtest.risk.position.static.TakeProfit
    options:
      show_root_heading: true

::: ml4t.backtest.risk.position.static.TimeExit
    options:
      show_root_heading: true

::: ml4t.backtest.risk.position.dynamic.TrailingStop
    options:
      show_root_heading: true

::: ml4t.backtest.risk.position.composite.RuleChain
    options:
      show_root_heading: true

::: ml4t.backtest.risk.position.composite.AllOf
    options:
      show_root_heading: true

::: ml4t.backtest.risk.position.composite.AnyOf
    options:
      show_root_heading: true

## Risk: Portfolio Limits

::: ml4t.backtest.risk.portfolio.limits.MaxDrawdownLimit
    options:
      show_root_heading: true

::: ml4t.backtest.risk.portfolio.limits.MaxPositionsLimit
    options:
      show_root_heading: true

::: ml4t.backtest.risk.portfolio.limits.MaxExposureLimit
    options:
      show_root_heading: true

::: ml4t.backtest.risk.portfolio.limits.DailyLossLimit
    options:
      show_root_heading: true

## Strategy Templates

::: ml4t.backtest.strategies.templates.SignalFollowingStrategy
    options:
      show_root_heading: true

::: ml4t.backtest.strategies.templates.MomentumStrategy
    options:
      show_root_heading: true

::: ml4t.backtest.strategies.templates.MeanReversionStrategy
    options:
      show_root_heading: true

::: ml4t.backtest.strategies.templates.LongShortStrategy
    options:
      show_root_heading: true
