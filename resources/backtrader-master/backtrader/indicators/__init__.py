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

from backtrader import Indicator
from backtrader.functions import *

from .accdecoscillator import *
from .aroon import *

# depend on basicops, moving averages and deviations
from .atr import *
from .awesomeoscillator import *

# The modules below should/must define __all__ with the Indicator objects
# of prepend an "_" (underscore) to private classes/variables
from .basicops import *
from .bollinger import *
from .cci import *
from .crossover import *
from .dema import *

# depends on moving averages
from .deviation import *
from .directionalmove import *
from .dma import *
from .dpo import *
from .dv2 import *  # depends on percentrank
from .ema import *
from .envelope import *
from .hadelta import *
from .heikinashi import *
from .hma import *
from .hurst import *
from .ichimoku import *
from .kama import *

# Depends on Momentum
from .kst import *
from .lrsi import *

# base for moving averages
from .mabase import *
from .macd import *
from .momentum import *
from .ols import *
from .oscillator import *
from .percentchange import *
from .percentrank import *
from .pivotpoint import *
from .prettygoodoscillator import *
from .priceoscillator import *
from .psar import *
from .rmi import *
from .rsi import *

# moving averages (so envelope and oscillators can be auto-generated)
from .sma import *
from .smma import *
from .stochastic import *
from .trix import *
from .tsi import *
from .ultimateoscillator import *
from .williams import *
from .wma import *
from .zlema import *
from .zlind import *
