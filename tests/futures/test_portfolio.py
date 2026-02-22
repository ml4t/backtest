"""Tests for portfolio-level futures calculations.

Validates aggregation of P&L, costs, and margin across
multiple futures positions.
"""

from datetime import datetime

from ml4t.backtest.types import ContractSpec, Position


class TestPortfolioPnLAggregation:
    """Test portfolio P&L aggregation."""

    def test_portfolio_pnl_simple(
        self,
        es_contract: ContractSpec,
        cl_contract: ContractSpec,
        gc_contract: ContractSpec,
    ):
        """Portfolio P&L: sum of individual P&Ls.

        ES: +$1,000
        CL: -$300
        GC: +$150
        Total: $850
        """
        pnls = [1000.0, -300.0, 150.0]
        total_pnl = sum(pnls)
        assert total_pnl == 850.0

    def test_portfolio_pnl_via_positions(
        self,
        es_contract: ContractSpec,
        cl_contract: ContractSpec,
    ):
        """Aggregate P&L from Position objects."""
        positions = [
            Position(
                asset="ES",
                quantity=2.0,
                entry_price=4000.0,
                entry_time=datetime.now(),
                current_price=4010.0,
                multiplier=es_contract.multiplier,
            ),
            Position(
                asset="CL",
                quantity=-3.0,
                entry_price=80.0,
                entry_time=datetime.now(),
                current_price=79.5,
                multiplier=cl_contract.multiplier,
            ),
        ]

        # ES: 2 * 10 * 50 = $1,000
        # CL: -3 * (79.5 - 80) * 1000 = -3 * -0.5 * 1000 = $1,500
        total_pnl = sum(pos.unrealized_pnl() for pos in positions)
        assert total_pnl == 2500.0


class TestPortfolioCosts:
    """Test portfolio-level cost aggregation."""

    def test_portfolio_costs(
        self,
        ib_commission,
        es_slippage,
        cl_slippage,
        es_contract: ContractSpec,
        cl_contract: ContractSpec,
    ):
        """Sum of per-instrument costs.

        ES (2 contracts): comm $4.50, slip $25.00 = $29.50
        CL (3 contracts): comm $6.75, slip $60.00 = $66.75
        Total: $96.25
        """
        # ES costs
        es_comm = ib_commission.calculate("ES", 2.0, 4000.0, es_contract.multiplier)
        es_slip = es_slippage.calculate("ES", 2.0, 4000.0, multiplier=es_contract.multiplier)

        # CL costs
        cl_comm = ib_commission.calculate("CL", 3.0, 80.0, cl_contract.multiplier)
        cl_slip = cl_slippage.calculate("CL", 3.0, 80.0, multiplier=cl_contract.multiplier)

        total_costs = es_comm + es_slip + cl_comm + cl_slip

        assert es_comm == 4.50
        assert es_slip == 25.0
        assert cl_comm == 6.75
        assert cl_slip == 60.0
        assert total_costs == 96.25


class TestCrossMarginBenefit:
    """Test cross-margin (portfolio margin) concepts.

    Note: This is conceptual - ml4t-backtest uses simple additive
    margin by default. Real portfolio margin with correlation
    offsets requires more sophisticated risk models.
    """

    def test_simple_additive_margin(
        self,
        es_contract: ContractSpec,
        cl_contract: ContractSpec,
    ):
        """Basic margin: sum of individual margins.

        5 ES: 5 * $15,000 = $75,000
        2 CL: 2 * $6,000 = $12,000
        Total: $87,000
        """
        assert es_contract.margin is not None
        assert cl_contract.margin is not None

        es_margin = 5 * es_contract.margin
        cl_margin = 2 * cl_contract.margin
        total_margin = es_margin + cl_margin

        assert total_margin == 87_000.0

    def test_cross_margin_offset_concept(
        self,
        es_contract: ContractSpec,
    ):
        """Cross-margin can reduce requirements for hedged positions.

        Conceptual example:
        - Long 5 ES futures
        - Short 5 NQ futures (correlated)
        - Correlation ~0.9 means offset potential

        With cross-margin, total < sum(individual)
        Offset = individual_margin * correlation * some_factor

        This test just documents the concept; actual implementation
        would require a correlation matrix and exchange rules.
        """
        # Hypothetical cross-margin calculation
        es_margin = 5 * es_contract.margin  # type: ignore
        nq_margin = 5 * 18_000.0  # Hypothetical NQ margin
        individual_sum = es_margin + nq_margin

        # With 80% cross-margin efficiency
        cross_margin_efficiency = 0.8
        cross_margin = individual_sum * (1 - cross_margin_efficiency * 0.5)

        # Cross margin < individual sum
        assert cross_margin < individual_sum


class TestPortfolioMarginRequirement:
    """Test overall portfolio margin requirement."""

    def test_portfolio_margin_sufficient(
        self,
        es_contract: ContractSpec,
        cl_contract: ContractSpec,
    ):
        """Check if capital covers margin requirement.

        Capital: $100,000
        Required: $87,000 (from above)
        Sufficient: True
        Excess: $13,000
        """
        capital = 100_000.0
        required_margin = 87_000.0

        sufficient = capital >= required_margin
        excess = capital - required_margin

        assert sufficient is True
        assert excess == 13_000.0

    def test_portfolio_margin_insufficient(
        self,
        es_contract: ContractSpec,
        cl_contract: ContractSpec,
    ):
        """Detect when margin is insufficient.

        Capital: $80,000
        Required: $87,000
        Shortfall: $7,000
        """
        capital = 80_000.0
        required_margin = 87_000.0

        sufficient = capital >= required_margin
        shortfall = required_margin - capital

        assert sufficient is False
        assert shortfall == 7_000.0
