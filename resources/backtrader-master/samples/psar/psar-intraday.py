#!/usr/bin/env python
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################


import argparse
import datetime

import backtrader as bt


class St(bt.Strategy):
    params = ()

    def __init__(self):
        self.psar0 = bt.ind.ParabolicSAR(self.data0)
        self.psar1 = bt.ind.ParabolicSAR(self.data1)
        pass

    def next(self):
        txt = []
        txt.append(f"{len(self):04d}")
        txt.append(f"{len(self.data0):04d}")
        txt.append(self.data0.datetime.datetime())
        txt.append(f"{self.data0.close[0]:.2f}")
        txt.append("PSAR")
        txt.append(f"{self.psar0[0]:04.2f}")
        if len(self.data1):
            txt.append(f"{len(self.data1):04d}")
            txt.append(self.data1.datetime.datetime())
            txt.append(f"{self.data1.close[0]:.2f}")
            txt.append("PSAR")
            txt.append(f"{self.psar1[0]:04.2f}")

        print(",".join(str(x) for x in txt))


def runstrat(args=None):
    args = parse_args(args)

    cerebro = bt.Cerebro()

    # Data feed kwargs
    kwargs = dict(
        timeframe=bt.TimeFrame.Minutes,
        compression=5,
    )

    # Parse from/to-date
    dtfmt, tmfmt = "%Y-%m-%d", "T%H:%M:%S"
    for a, d in ((getattr(args, x), x) for x in ["fromdate", "todate"]):
        if a:
            strpfmt = dtfmt + tmfmt * ("T" in a)
            kwargs[d] = datetime.datetime.strptime(a, strpfmt)

    # Data feed
    data0 = bt.feeds.BacktraderCSVData(dataname=args.data0, **kwargs)
    cerebro.adddata(data0)

    cerebro.resampledata(data0, timeframe=bt.TimeFrame.Minutes, compression=15)

    # Broker
    cerebro.broker = bt.brokers.BackBroker(**eval("dict(" + args.broker + ")"))

    # Sizer
    cerebro.addsizer(bt.sizers.FixedSize, **eval("dict(" + args.sizer + ")"))

    # Strategy
    cerebro.addstrategy(St, **eval("dict(" + args.strat + ")"))

    # Execute
    cerebro.run(**eval("dict(" + args.cerebro + ")"))

    if args.plot:  # Plot if requested to
        cerebro.plot(**eval("dict(" + args.plot + ")"))


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=("Sample Skeleton")
    )

    parser.add_argument(
        "--data0", default="../../datas//2006-min-005.txt", required=False, help="Data to read in"
    )

    # Defaults for dates
    parser.add_argument(
        "--fromdate", required=False, default="", help="Date[time] in YYYY-MM-DD[THH:MM:SS] format"
    )

    parser.add_argument(
        "--todate", required=False, default="", help="Date[time] in YYYY-MM-DD[THH:MM:SS] format"
    )

    parser.add_argument(
        "--cerebro", required=False, default="", metavar="kwargs", help="kwargs in key=value format"
    )

    parser.add_argument(
        "--broker", required=False, default="", metavar="kwargs", help="kwargs in key=value format"
    )

    parser.add_argument(
        "--sizer", required=False, default="", metavar="kwargs", help="kwargs in key=value format"
    )

    parser.add_argument(
        "--strat", required=False, default="", metavar="kwargs", help="kwargs in key=value format"
    )

    parser.add_argument(
        "--plot",
        required=False,
        default="",
        nargs="?",
        const="{}",
        metavar="kwargs",
        help="kwargs in key=value format",
    )

    return parser.parse_args(pargs)


if __name__ == "__main__":
    runstrat()
