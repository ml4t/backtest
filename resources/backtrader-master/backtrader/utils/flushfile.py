#!/usr/bin/env python
###############################################################################
#
#  Copyright (C) 2015, 2016, 2017 Daniel Rodriguez
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

import sys


class flushfile:
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.f.flush()


if sys.platform == "win32":
    sys.stdout = flushfile(sys.stdout)
    sys.stderr = flushfile(sys.stderr)


class StdOutDevNull:
    def __init__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, x):
        pass

    def flush(self):
        pass

    def stop(self):
        sys.stdout = self.stdout
