ml4t.backtest Documentation
============================

.. image:: https://img.shields.io/badge/version-0.1.0-blue.svg
   :alt: Version 0.1.0

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: MIT License

ml4t.backtest is an event-driven backtesting engine with institutional-grade execution fidelity.

**Core Mission**: Replicate real trading conditions with point-in-time correctness and realistic execution.

**Version:** 0.1.0 (Beta)
**Last Updated:** November 2025

Features
--------

* **Event-Driven Architecture**: Market, signal, order, and fill events
* **Point-in-Time Safety**: No look-ahead bias
* **Vectorized Hybrid**: Event-driven control with vectorized execution
* **Pluggable Components**: Customizable broker, commission, and slippage models
* **High Performance**: 100k+ events/second

Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/engine
   modules/core
   modules/data
   modules/execution
   modules/portfolio
   modules/risk
   modules/strategy
   modules/reporting

Getting Started
---------------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install ml4t.backtest

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

   from ml4t.backtest.engine import BacktestEngine
   from ml4t.backtest.execution.broker import SimulationBroker
   from ml4t.backtest.strategy.base import Strategy

   # Create your strategy
   class MyStrategy(Strategy):
       def on_market_event(self, event):
           # Your trading logic here
           pass

   # Set up the engine
   broker = SimulationBroker(initial_cash=100000.0)
   engine = BacktestEngine(
       data_feed=my_data_feed,
       strategy=MyStrategy(),
       broker=broker,
   )

   # Run the backtest
   results = engine.run()

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
