# region imports
from AlgorithmImports import *
# endregion

class Testvalidation(QCAlgorithm):

    def initialize(self):
        # Locally Lean installs free sample data, to download more data please visit https://www.quantconnect.com/docs/v2/lean-cli/datasets/downloading-data
        self.set_start_date(2013, 10, 7)  # Set Start Date
        self.set_end_date(2013, 10, 11)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        self.add_equity("SPY", Resolution.MINUTE)

    def on_data(self, data: Slice):
        """on_data event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        """
        if not self.portfolio.invested:
            self.set_holdings("SPY", 1)
            self.debug("Purchased Stock")
