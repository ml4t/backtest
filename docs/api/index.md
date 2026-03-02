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
        - submit_order
        - submit_bracket
        - close_position
        - cancel_order
        - get_position
        - get_positions
        - get_cash
        - get_account_value
        - get_rejected_orders
        - set_position_rules
        - rebalance_to_weights

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
        - to_trades_dataframe
        - to_equity_dataframe
        - to_dict
        - to_parquet

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
