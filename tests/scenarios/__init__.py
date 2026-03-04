"""Answer-first scenario factory for parametric testing.

Expected results are computed analytically BEFORE the SUT runs,
ensuring the test is truly independent.
"""

from .factory import ExpectedResult, Scenario, make_round_trip

__all__ = ["make_round_trip", "Scenario", "ExpectedResult"]
