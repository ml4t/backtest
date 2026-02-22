"""Tests for futures margin calculations.

Validates margin requirements and margin call logic.
Note: ml4t-backtest uses account-level margin policies rather than
per-contract margin from ContractSpec by default.
"""

from ml4t.backtest.types import ContractSpec


class TestInitialMargin:
    """Test initial margin calculations."""

    def test_initial_margin_single_contract(self, es_contract: ContractSpec):
        """ES initial margin: $15,000 per contract."""
        assert es_contract.margin == 15_000.0

    def test_initial_margin_multiple_contracts(self, es_contract: ContractSpec):
        """Total margin = contracts * margin_per_contract.

        5 ES contracts = 5 * $15,000 = $75,000
        """
        contracts = 5
        assert es_contract.margin is not None
        total_margin = contracts * es_contract.margin
        assert total_margin == 75_000.0

    def test_margin_as_pct_of_notional(self, es_contract: ContractSpec):
        """Margin as percentage of notional.

        ES at 4000: notional = 4000 * 50 = $200,000
        Margin = $15,000
        Margin % = 15,000 / 200,000 = 7.5%
        """
        notional_per_contract = 4000.0 * es_contract.multiplier
        margin_pct = es_contract.margin / notional_per_contract  # type: ignore
        assert margin_pct == 0.075  # 7.5%


class TestPortfolioMargin:
    """Test portfolio-level margin calculations."""

    def test_portfolio_margin_additive(self, es_contract: ContractSpec, cl_contract: ContractSpec):
        """Basic portfolio margin: sum of individual margins.

        5 ES + 2 CL = 5*15,000 + 2*6,000 = $75,000 + $12,000 = $87,000

        Note: This is the simple additive model. Real portfolios may
        get margin offsets for hedged positions.
        """
        assert es_contract.margin is not None
        assert cl_contract.margin is not None

        es_margin = 5 * es_contract.margin
        cl_margin = 2 * cl_contract.margin

        total_margin = es_margin + cl_margin
        assert total_margin == 87_000.0

    def test_margin_available_for_trading(self, es_contract: ContractSpec):
        """Available margin = capital - required margin.

        Capital: $100,000
        2 ES positions = $30,000 margin
        Available: $70,000
        """
        capital = 100_000.0
        contracts = 2
        assert es_contract.margin is not None
        required_margin = contracts * es_contract.margin

        available = capital - required_margin
        assert available == 70_000.0

    def test_max_contracts_from_capital(self, es_contract: ContractSpec):
        """Maximum contracts that can be held with given capital.

        Capital: $100,000
        ES margin: $15,000
        Max contracts: floor(100,000 / 15,000) = 6
        """
        capital = 100_000.0
        assert es_contract.margin is not None
        max_contracts = int(capital / es_contract.margin)
        assert max_contracts == 6


class TestMarginCall:
    """Test margin call trigger conditions."""

    def test_margin_call_trigger(self, es_contract: ContractSpec):
        """Margin call when equity < maintenance margin.

        Initial margin: $15,000 (typically ~7.5% of notional)
        Maintenance margin: typically ~5% of notional

        If equity falls below maintenance, broker issues margin call.

        Example:
        - Position: 1 ES at 4000
        - Notional: $200,000
        - Initial margin: $15,000
        - Maintenance margin: ~$10,000 (assume 5% of notional)
        - If P&L drops by $5,000+, margin call is triggered
        """
        notional = 4000.0 * es_contract.multiplier  # $200,000
        initial_margin = es_contract.margin  # $15,000
        maintenance_margin = notional * 0.05  # $10,000

        # Scenario: lose $6,000 on the position
        initial_equity = 100_000.0
        pnl_loss = -6_000.0
        current_equity = initial_equity + pnl_loss

        # Check if below maintenance
        margin_call_triggered = current_equity < maintenance_margin
        assert not margin_call_triggered  # 94,000 > 10,000

        # But if we only deposited the minimum margin...
        minimal_equity = initial_margin + pnl_loss  # 15,000 - 6,000 = 9,000
        margin_call_triggered = minimal_equity < maintenance_margin
        assert margin_call_triggered  # 9,000 < 10,000

    def test_maintenance_margin_ratio(self, es_contract: ContractSpec):
        """Maintenance margin is typically lower than initial.

        Common ratios:
        - CME ES: Initial ~7.5%, Maintenance ~6%
        - This provides buffer before margin call
        """
        notional = 4000.0 * es_contract.multiplier
        initial_pct = es_contract.margin / notional  # type: ignore

        # Assume maintenance is 80% of initial
        maintenance_pct = initial_pct * 0.8

        assert initial_pct == 0.075  # 7.5%
        assert maintenance_pct == 0.06  # 6%
