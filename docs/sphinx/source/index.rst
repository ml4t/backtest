QEngine Documentation
=====================

QEngine is an event-driven backtesting engine with institutional-grade execution fidelity.

**Core Mission**: Replicate real trading conditions with point-in-time correctness and realistic execution.

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

   modules/core
   modules/execution
   modules/portfolio
   modules/strategy
   modules/data
   modules/reporting

Getting Started
---------------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install qengine

Basic Example
^^^^^^^^^^^^^

.. code-block:: python

   from qengine.engine import BacktestEngine
   from qengine.execution.broker import SimulationBroker
   from qengine.strategy.base import Strategy

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
