"""Framework driver modules for the validation suite.

Each module provides a single `run(scenario, prices_df, entries, exits)` function
that executes a backtest using the respective external framework.

Framework imports are lazy — each module only imports its framework when called.
"""
