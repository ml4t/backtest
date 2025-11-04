import os

import pytest
import vectorbtpro as vbt
import vectorbtpro.portfolio.pfopt.base as pfopt
from numba import njit
from vectorbtpro.portfolio.enums import alloc_point_dt, alloc_range_dt

from tests.utils import *

pypfopt_available = True
try:
    import pypfopt
except:
    pypfopt_available = False
riskfolio_available = True
try:
    import riskfolio as rp
except:
    riskfolio_available = False
universal_available = True
try:
    import universal
except:
    universal_available = False


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# PyPortfolioOpt ############# #

prices = pd.DataFrame(
    {
        "XOM": [
            54.068794,
            54.279907,
            54.749043,
            54.577045,
            54.358093,
        ],
        "RRC": [
            51.300568,
            51.993038,
            51.690697,
            51.593170,
            52.597733,
        ],
        "BBY": [
            32.524055,
            33.349487,
            33.090542,
            33.616547,
            32.297466,
        ],
        "MA": [
            22.062426,
            21.997149,
            22.081820,
            21.937523,
            21.945297,
        ],
        "PFE": [
            13.940202,
            13.741367,
            13.697187,
            13.645634,
            13.756095,
        ],
    },
    index=pd.date_range("2020-01-01", "2020-01-05"),
)
market_caps = prices.prod(axis=0)
returns = prices.pct_change().dropna(how="all")
viewdict = {"XOM": 0.20, "RRC": -0.30, "BBY": 0, "MA": -0.2, "PFE": 0.15}


class TestPyPortfolioOpt:
    def test_resolve_pypfopt_expected_returns(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns("mean_historical_return", prices=prices),
                pypfopt.expected_returns.mean_historical_return(prices),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "mean_historical_return", prices=prices, returns_data=False
                ),
                pypfopt.expected_returns.mean_historical_return(prices),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "mean_historical_return", prices=returns, returns_data=True
                ),
                pypfopt.expected_returns.mean_historical_return(returns, returns_data=True),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns("mean_historical_return", returns=returns),
                pypfopt.expected_returns.mean_historical_return(returns, returns_data=True),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "mean_historical_return", prices=prices, returns=returns
                ),
                pypfopt.expected_returns.mean_historical_return(prices, returns_data=False),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "mean_historical_return",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    compounding=False,
                ),
                pypfopt.expected_returns.mean_historical_return(
                    prices, frequency=365, compounding=False
                ),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "ema_historical_return",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    compounding=False,
                    span=350,
                ),
                pypfopt.expected_returns.ema_historical_return(
                    prices, frequency=365, compounding=False, span=350
                ),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "capm_return",
                    prices=prices,
                    market_prices=prices.iloc[:, 0],
                    risk_free_rate=0.03,
                    freq="1d",
                    year_freq="365 days",
                    compounding=False,
                    span=350,
                ),
                pypfopt.expected_returns.capm_return(
                    prices,
                    frequency=365,
                    market_prices=prices.iloc[:, 0].to_frame(),
                    risk_free_rate=0.03,
                    compounding=False,
                ),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    "bl_returns",
                    prices=prices,
                    absolute_views=viewdict,
                    pi=None,
                ),
                pypfopt.black_litterman.BlackLittermanModel(
                    pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                    absolute_views=viewdict,
                ).bl_returns(),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    lambda prices, absolute_views: pypfopt.black_litterman.BlackLittermanModel(
                        pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                        absolute_views=viewdict,
                    ).bl_returns(),
                    prices=prices,
                    absolute_views=viewdict,
                ),
                pypfopt.black_litterman.BlackLittermanModel(
                    pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                    absolute_views=viewdict,
                ).bl_returns(),
            )
            assert_series_equal(
                pfopt.resolve_pypfopt_expected_returns(
                    pypfopt.black_litterman.BlackLittermanModel(
                        pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                        absolute_views=viewdict,
                    ).bl_returns(),
                ),
                pypfopt.black_litterman.BlackLittermanModel(
                    pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
                    absolute_views=viewdict,
                ).bl_returns(),
            )

    def test_resolve_pypfopt_cov_matrix(self):
        if pypfopt_available:
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "sample_cov",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    fix_method="diag",
                ),
                pypfopt.risk_models.sample_cov(
                    prices,
                    frequency=365,
                    fix_method="diag",
                ),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "exp_cov",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                    benchmark=0.1,
                    fix_method="diag",
                ),
                pypfopt.risk_models.exp_cov(
                    prices,
                    frequency=365,
                    benchmark=0.1,
                    fix_method="diag",
                ),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "exp_cov",
                    returns=returns,
                    freq="1d",
                    year_freq="365 days",
                    span=200,
                    fix_method="diag",
                ),
                pypfopt.risk_models.exp_cov(
                    returns,
                    returns_data=True,
                    frequency=365,
                    span=200,
                    fix_method="diag",
                ),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "ledoit_wolf",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(prices, frequency=365).ledoit_wolf(),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "ledoit_wolf_single_factor",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).ledoit_wolf(shrinkage_target="single_factor"),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "ledoit_wolf_constant_correlation",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).ledoit_wolf(shrinkage_target="constant_correlation"),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    "oracle_approximating",
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).oracle_approximating(),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    lambda prices, frequency: pypfopt.risk_models.CovarianceShrinkage(
                        prices,
                        frequency=frequency,
                    ).oracle_approximating(),
                    prices=prices,
                    freq="1d",
                    year_freq="365 days",
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).oracle_approximating(),
            )
            np.testing.assert_array_equal(
                pfopt.resolve_pypfopt_cov_matrix(
                    pypfopt.risk_models.CovarianceShrinkage(
                        prices,
                        frequency=365,
                    ).oracle_approximating(),
                ),
                pypfopt.risk_models.CovarianceShrinkage(
                    prices,
                    frequency=365,
                ).oracle_approximating(),
            )

    def test_resolve_pypfopt_optimizer(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_frontier", prices=prices
            ).min_volatility()
            weights2 = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            ).min_volatility()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_frontier",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                weight_bounds=(-1, 1),
            ).min_volatility()
            weights2 = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_cdar",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cdar()
            weights2 = pypfopt.efficient_frontier.EfficientCDaR(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                returns,
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cdar()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_cvar",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cvar()
            weights2 = pypfopt.efficient_frontier.EfficientCVaR(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                returns,
                beta=0.9,
                weight_bounds=(-1, 1),
            ).min_cvar()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "efficient_semivariance",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                benchmark=0.1,
                weight_bounds=(-1, 1),
            ).min_semivariance()
            weights2 = pypfopt.efficient_frontier.EfficientSemivariance(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                returns,
                benchmark=0.1,
                weight_bounds=(-1, 1),
            ).min_semivariance()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "black_litterman",
                prices=prices,
                pi=None,
                absolute_views=viewdict,
            ).bl_weights()
            weights2 = pypfopt.black_litterman.BlackLittermanModel(
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                absolute_views=viewdict,
            ).bl_weights()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "black_litterman",
                prices=prices,
                market_caps=market_caps,
                market_prices=prices.iloc[:, 0],
                absolute_views=viewdict,
                freq="1d",
                year_freq="365 days",
                risk_free_rate=0.03,
            ).bl_weights()
            weights2 = pypfopt.black_litterman.BlackLittermanModel(
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                pi=pypfopt.black_litterman.market_implied_prior_returns(
                    market_caps,
                    pypfopt.black_litterman.market_implied_risk_aversion(
                        prices.iloc[:, 0],
                        frequency=365,
                        risk_free_rate=0.03,
                    ),
                    pypfopt.risk_models.CovarianceShrinkage(
                        prices=prices, frequency=365
                    ).ledoit_wolf(),
                    risk_free_rate=0.03,
                ),
                absolute_views=viewdict,
            ).bl_weights()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "hrpopt",
                prices=prices,
                freq="1d",
                year_freq="365 days",
            ).optimize()
            weights2 = pypfopt.hierarchical_portfolio.HRPOpt(
                returns,
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
            ).optimize()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                "cla",
                prices=prices,
                freq="1d",
                year_freq="365 days",
                weight_bounds=(-1, 1),
            ).min_volatility()
            weights2 = pypfopt.cla.CLA(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2

            from pypfopt.cla import CLA

            class CustomCLA(CLA):
                def __init__(self, prices, frequency=252, weight_bounds=(0, 1)):
                    expected_returns = pypfopt.expected_returns.mean_historical_return(
                        prices=prices, frequency=frequency
                    )
                    cov_matrix = pypfopt.risk_models.CovarianceShrinkage(
                        prices=prices, frequency=frequency
                    ).ledoit_wolf()
                    CLA.__init__(self, expected_returns, cov_matrix, weight_bounds=weight_bounds)

            weights1 = pfopt.resolve_pypfopt_optimizer(
                CustomCLA,
                prices=prices,
                freq="1d",
                year_freq="365 days",
                weight_bounds=(-1, 1),
            ).min_volatility()
            weights2 = pypfopt.cla.CLA(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2
            weights1 = pfopt.resolve_pypfopt_optimizer(
                pypfopt.cla.CLA(
                    pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                    pypfopt.risk_models.CovarianceShrinkage(
                        prices=prices, frequency=365
                    ).ledoit_wolf(),
                    weight_bounds=(-1, 1),
                )
            ).min_volatility()
            weights2 = pypfopt.cla.CLA(
                pypfopt.expected_returns.mean_historical_return(prices=prices, frequency=365),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices, frequency=365).ledoit_wolf(),
                weight_bounds=(-1, 1),
            ).min_volatility()
            assert weights1 == weights2

    def test_pypfopt_optimize(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.max_sharpe()
            assert pfopt.pypfopt_optimize(prices=prices) == ef.clean_weights()
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.ema_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(
                    shrinkage_target="single_factor"
                ),
            )
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    expected_returns="ema_historical_return",
                    cov_matrix="ledoit_wolf_single_factor",
                )
                == ef.clean_weights()
            )
            ef = pypfopt.efficient_frontier.EfficientCVaR(
                pypfopt.expected_returns.ema_historical_return(prices=prices),
                returns,
                beta=0.9,
                weight_bounds=(-1, 1),
            )
            ef.min_cvar()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    target="min_cvar",
                    expected_returns="ema_historical_return",
                    optimizer="efficient_cvar",
                    beta=0.9,
                    weight_bounds=(-1, 1),
                )
                == ef.clean_weights()
            )
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            from pypfopt import objective_functions

            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives="L2_reg",
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives=["L2_reg"],
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives=objective_functions.L2_reg,
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives=[objective_functions.L2_reg],
                    gamma=0.1,
                )
                == ef.clean_weights()
            )
            from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

            da = DiscreteAllocation(
                ef.clean_weights(), get_latest_prices(prices), total_portfolio_value=20000
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    objectives="L2_reg",
                    gamma=0.1,
                    discrete_allocation=True,
                    total_portfolio_value=20000,
                )
                == da.lp_portfolio()[0]
            )
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.add_constraint(lambda w: w[0] >= 0.2)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    constraints=lambda w: w[0] >= 0.2,
                )
                == ef.clean_weights()
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    constraints=[lambda w: w[0] >= 0.2],
                )
                == ef.clean_weights()
            )
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.add_constraint(lambda w: w[0] >= 0.2)
            ef.add_constraint(lambda w: w >= 0.1)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    constraints=[
                        lambda w: w[0] >= 0.2,
                        lambda w: w >= 0.1,
                    ],
                )
                == ef.clean_weights()
            )

            sector_mapper = {
                "XOM": "tech",
                "RRC": "tech",
                "BBY": "Oil/Gas",
                "MA": "Oil/Gas",
                "PFE": "Financials",
            }
            sector_lower = {"tech": 0.1}
            sector_upper = {"tech": 0.4, "Oil/Gas": 0.1}
            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
            ef.max_sharpe()
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    sector_mapper=sector_mapper,
                    sector_lower=sector_lower,
                    sector_upper=sector_upper,
                )
                == ef.clean_weights()
            )

            import cvxpy as cp

            def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
                log_sum = cp.sum(cp.log(w))
                var = cp.quad_form(w, cov_matrix)
                return var - k * log_sum

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                weight_bounds=(0.01, 0.2),
            )
            ef.convex_objective(
                logarithmic_barrier_objective,
                cov_matrix=pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                k=0.001,
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    target=logarithmic_barrier_objective,
                    k=0.001,
                    weight_bounds=(0.01, 0.2),
                )
                == ef.clean_weights()
            )

            def deviation_risk_parity(w, cov_matrix):
                diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
                return (diff**2).sum().sum()

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                weight_bounds=(0.01, 0.12),
            )
            ef.nonconvex_objective(
                deviation_risk_parity,
                objective_args=(
                    pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
                ),
            )
            assert (
                pfopt.pypfopt_optimize(
                    prices=prices,
                    target=deviation_risk_parity,
                    target_is_convex=False,
                    weight_bounds=(0.01, 0.12),
                )
                == ef.clean_weights()
            )

    def test_riskfolio_optimize(self):
        if riskfolio_available:
            np.random.seed(42)
            longer_returns = pd.DataFrame(
                np.random.uniform(-0.1, 0.1, size=(100, 5)),
                columns=["A", "B", "C", "D", "E"],
            )
            factor_returns = pd.DataFrame(
                np.random.uniform(-0.1, 0.1, size=(100, 3)),
                columns=["F", "G", "H"],
            )
            assert_series_equal(
                pd.Series(pfopt.riskfolio_optimize(longer_returns)),
                pd.Series(
                    {
                        "A": 0.12676800514501807,
                        "B": 0.4760327891840394,
                        "C": 7.2644261969418e-10,
                        "D": 0.3971992047740059,
                        "E": 1.7049415731580372e-10,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(longer_returns, factors=factor_returns, model="FM")
                ),
                pd.Series(
                    {
                        "A": 0.1267680051458212,
                        "B": 0.4760327891836326,
                        "C": 7.264414768832996e-10,
                        "D": 0.3971992047736109,
                        "E": 1.704938888482468e-10,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        factors=factor_returns,
                        model="FM",
                        stats_methods=["assets_stats", "factors_stats"],
                    )
                ),
                pd.Series(
                    {
                        "A": 0.1267680051458212,
                        "B": 0.4760327891836326,
                        "C": 7.264414768832996e-10,
                        "D": 0.3971992047736109,
                        "E": 1.704938888482468e-10,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        factors=factor_returns,
                        func_kwargs=dict(
                            assets_stats=dict(),
                            factors_stats=dict(),
                            optimization=dict(model="FM"),
                        ),
                    )
                ),
                pd.Series(
                    {
                        "A": 0.1267680051458212,
                        "B": 0.4760327891836326,
                        "C": 7.264414768832996e-10,
                        "D": 0.3971992047736109,
                        "E": 1.704938888482468e-10,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        factors=factor_returns,
                        func_kwargs=dict(
                            optimization=dict(model="FM"),
                        ),
                    )
                ),
                pd.Series(
                    {
                        "A": 0.1267680051458212,
                        "B": 0.4760327891836326,
                        "C": 7.264414768832996e-10,
                        "D": 0.3971992047736109,
                        "E": 1.704938888482468e-10,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns, method_mu="hist", method_cov="hist", d=0.94
                    )
                ),
                pd.Series(
                    {
                        "A": 0.12676800514501807,
                        "B": 0.4760327891840394,
                        "C": 7.2644261969418e-10,
                        "D": 0.3971992047740059,
                        "E": 1.7049415731580372e-10,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(pfopt.riskfolio_optimize(longer_returns, port_cls="HCPortfolio")),
                pd.Series(
                    {
                        "A": 0.19369278873359128,
                        "B": 0.20238981440998197,
                        "C": 0.20942969908552786,
                        "D": 0.19933866749954113,
                        "E": 0.19514903027135774,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(pfopt.riskfolio_optimize(longer_returns, port_cls=rp.HCPortfolio)),
                pd.Series(
                    {
                        "A": 0.19369278873359128,
                        "B": 0.20238981440998197,
                        "C": 0.20942969908552786,
                        "D": 0.19933866749954113,
                        "E": 0.19514903027135774,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(pfopt.riskfolio_optimize(longer_returns, opt_method="rp")),
                pd.Series(
                    {
                        "A": 0.22652553527275376,
                        "B": 0.20225103853474868,
                        "C": 0.21538322980670396,
                        "D": 0.1744225679341509,
                        "E": 0.18141762845164283,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(pfopt.riskfolio_optimize(longer_returns, opt_method="rrp")),
                pd.Series(
                    {
                        "A": 0.22652582481582162,
                        "B": 0.2022516322803447,
                        "C": 0.2153823193787835,
                        "D": 0.17442220639910935,
                        "E": 0.18141801712594077,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        constraints=[dict(Type="Assets", Position="A", Sign="<=", Weight=0.1)],
                    )
                ),
                pd.Series(
                    {
                        "A": 0.09999988709015002,
                        "B": 0.4869261339374733,
                        "C": 1.6951444542121246e-08,
                        "D": 0.4130739578334771,
                        "E": 4.187455131058068e-09,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        asset_classes=["C1", "C1", "C2", "C2", "C3"],
                        constraints=[
                            dict(Type="Classes", Set="Class", Position="C1", Sign="<=", Weight=0.2)
                        ],
                    )
                ),
                pd.Series(
                    {
                        "A": 2.7553253657466047e-09,
                        "B": 0.1999999954418845,
                        "C": 1.3301654710792048e-08,
                        "D": 0.7999999875008236,
                        "E": 1.0003118168811534e-09,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        asset_classes=pd.Index(["C1", "C1", "C2", "C2", "C3"], name="my_class"),
                        constraints=[
                            dict(
                                Type="Classes", Set="my_class", Position="C1", Sign="<=", Weight=0.2
                            )
                        ],
                    )
                ),
                pd.Series(
                    {
                        "A": 2.7553253657466047e-09,
                        "B": 0.1999999954418845,
                        "C": 1.3301654710792048e-08,
                        "D": 0.7999999875008236,
                        "E": 1.0003118168811534e-09,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns.vbt.add_levels(
                            pd.Index(["C1", "C1", "C2", "C2", "C3"], name="my_class")
                        ),
                        constraints=[
                            dict(
                                Type="Classes", Set="my_class", Position="C1", Sign="<=", Weight=0.2
                            )
                        ],
                    )
                ),
                pd.Series(
                    {
                        ("C1", "A"): 2.7553253657466047e-09,
                        ("C1", "B"): 0.1999999954418845,
                        ("C2", "C"): 1.3301654710792048e-08,
                        ("C2", "D"): 0.7999999875008236,
                        ("C3", "E"): 1.0003118168811534e-09,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        factors=factor_returns,
                        model="FM",
                        constraints=[dict(Factor="F", Sign="<=", Value=0.5)],
                    )
                ),
                pd.Series(
                    {
                        "A": 0.12677262060399214,
                        "B": 0.47603049596901686,
                        "C": 1.7644613667926323e-10,
                        "D": 0.3971968832083084,
                        "E": 4.223652304226146e-11,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        port_cls="HCPortfolio",
                        constraints=[dict(Type="Assets", Position="A", Sign="<=", Weight=0.1)],
                    )
                ),
                pd.Series(
                    {
                        "A": 0.1,
                        "B": 0.20238981440998197,
                        "C": 0.20942969908552786,
                        "D": 0.29303145623313237,
                        "E": 0.19514903027135774,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        model="BL",
                        asset_classes=["C1", "C1", "C2", "C2", "C3"],
                        views=[
                            dict(Type="Classes", Set="Class", Position="C1", Sign="<=", Return=0.2),
                            dict(Type="Classes", Set="Class", Position="C2", Sign="<=", Return=0.3),
                        ],
                        freq="1d",
                        year_freq="252d",
                        delta=None,
                        eq=True,
                    )
                ),
                pd.Series(
                    {
                        "A": 0.19028479017478658,
                        "B": 0.1964263445744196,
                        "C": 0.3316412298128069,
                        "D": 0.2816476353359129,
                        "E": 1.0207392238984308e-10,
                    }
                ),
                atol=1e-4,
            )
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        pre_opt=True,
                        pre_opt_as_w=True,
                        model="BL",
                        asset_classes=["C1", "C1", "C2", "C2", "C3"],
                        views=[
                            dict(Type="Classes", Set="Class", Position="C1", Sign="<=", Return=0.2),
                            dict(Type="Classes", Set="Class", Position="C2", Sign="<=", Return=0.3),
                        ],
                        freq="1d",
                        year_freq="252d",
                        delta=None,
                        eq=True,
                    )
                ),
                pd.Series(
                    {
                        "A": 0.027579416662377298,
                        "B": 0.485577412107038,
                        "C": 1.6032254651302998e-08,
                        "D": 0.4860571855320755,
                        "E": 0.0007859696662543658,
                    }
                ),
                atol=1e-4,
            )
            longer_returns.iloc[:, 2] = np.nan
            assert_series_equal(
                pd.Series(
                    pfopt.riskfolio_optimize(
                        longer_returns,
                        pre_opt=True,
                        pre_opt_as_w=True,
                        model="BL",
                        asset_classes=["C1", "C1", "C2", "C2", "C3"],
                        views=[
                            dict(Type="Classes", Set="Class", Position="C1", Sign="<=", Return=0.2),
                            dict(Type="Classes", Set="Class", Position="C2", Sign="<=", Return=0.3),
                        ],
                        freq="1d",
                        year_freq="252d",
                        delta=None,
                        eq=True,
                    )
                ),
                pd.Series(
                    {
                        "A": 0.021150803146091784,
                        "B": 0.5705920585309042,
                        "D": 0.40819158602138167,
                        "E": 6.55523016224704e-05,
                    }
                ),
                atol=1e-4,
            )


# ############# PortfolioOptimizer ############# #


class TestPortfolioOptimizer:
    def test_indexing(self):
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            group_configs=[
                dict(on=[1, 3], _name="g1"),
                dict(on=[2, 4], _name="g2"),
            ],
        )
        pfo2 = pfo.loc["2020-01-03":"2020-01-04", "g2"]
        assert_index_equal(pfo2.wrapper.index, pfo.wrapper.index[2:4])
        assert_index_equal(pfo2.wrapper.columns, pfo.wrapper.columns[5:])
        assert_records_close(pfo2.alloc_records.values, np.array([(0, 0, 0)], dtype=alloc_point_dt))
        np.testing.assert_array_equal(pfo2._allocations, pfo._allocations[[2]])
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            group_configs=[
                dict(index_ranges=[0, 1], _name="g1"),
                dict(index_ranges=[2, 3], _name="g2"),
            ],
        )
        pfo2 = pfo.loc["2020-01-03":"2020-01-04", "g2"]
        assert_index_equal(pfo2.wrapper.index, pfo.wrapper.index[2:4])
        assert_index_equal(pfo2.wrapper.columns, pfo.wrapper.columns[5:])
        assert_records_close(
            pfo2.alloc_records.values, np.array([(0, 0, 0, 1, 1, 1)], dtype=alloc_range_dt)
        )
        np.testing.assert_array_equal(pfo2._allocations, pfo._allocations[[1]])

    def test_from_optimize_func(self):
        def get_allocations(pfo):
            start_idx = pfo.alloc_records.values["start_idx"]
            end_idx = pfo.alloc_records.values["end_idx"]
            return np.array(
                [
                    prices.values[start_idx[i] : end_idx[i]].sum(axis=0)
                    for i in range(len(pfo.alloc_records.values))
                ]
            )

        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum().values,
            vbt.Rep("index_slice"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum().to_dict(),
            vbt.Rep("index_slice"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: {
                k: v
                for k, v in prices.iloc[index_slice].sum().to_dict().items()
                if k in prices.columns[:-1]
            },
            vbt.Rep("index_slice"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        allocations = get_allocations(pfo)
        allocations[:, -1] = np.nan
        np.testing.assert_array_equal(
            pfo._allocations,
            allocations,
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            every="2D",
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            every=vbt.RepEval("'2D'"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=[(0, 2), (2, 4)],
            alloc_wait=0,
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 1, 1), (1, 0, 2, 4, 3, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=[(0, 2), (2, 4)],
            index_loc=[2, 4],
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_ranges=vbt.RepEval("[(0, 2), (2, 4)]"),
            index_loc=vbt.RepEval("[2, 4]"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda prices: prices.sum(),
            vbt.Takeable(prices),
            index_ranges=vbt.RepEval("[(0, 2), (2, 4)]"),
            index_loc=vbt.RepEval("[2, 4]"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )

        def ann_optimize_func(prices: vbt.Takeable):
            return prices.sum()

        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            ann_optimize_func,
            prices,
            index_ranges=vbt.RepEval("[(0, 2), (2, 4)]"),
            index_loc=vbt.RepEval("[2, 4]"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda i, index_ranges: prices.iloc[index_ranges[0][i] : index_ranges[1][i]].sum(),
            vbt.Rep("i"),
            vbt.Rep("index_ranges"),
            every="2D",
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            njit(lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx])),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            njit(lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx])),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            chunked=dict(arg_take_spec=dict(args=vbt.ArgsTaker(None))),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            njit(lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx])),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            jitted=dict(parallel=True),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            vbt.Rep("optimize_func"),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            template_context=dict(
                optimize_func=njit(
                    lambda i, from_idx, to_idx, prices: vbt.nb.nansum_nb(prices[from_idx:to_idx])
                )
            ),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 2, 2, 1), (1, 0, 2, 4, 4, 1)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice, mult: prices.iloc[index_slice].sum() * mult,
            vbt.Rep("index_slice"),
            vbt.Param([1, 2]),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocRanges)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0, 5, 5, 0), (0, 1, 0, 5, 5, 0)], dtype=alloc_range_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo) * np.array([[1], [2]]),
        )

        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
        )
        assert_index_equal(
            pfo.wrapper.grouper.group_by,
            pd.Index(["group", "group", "group", "group", "group"], name="group"),
        )
        assert_index_equal(
            pfo.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    ("group", "XOM"),
                    ("group", "RRC"),
                    ("group", "BBY"),
                    ("group", "MA"),
                    ("group", "PFE"),
                ],
                names=["group", None],
            ),
        )
        assert pfo.wrapper.grouped_ndim == 1
        assert_index_equal(
            pfo.alloc_records.wrapper.columns,
            pd.Index(["group"], name="group"),
        )
        assert pfo.alloc_records.wrapper.ndim == 1
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice, mult: prices.iloc[index_slice].sum() * mult,
            vbt.Rep("index_slice"),
            group_configs=[{"args_1": 1}, {"args_1": 2}],
        )
        assert_index_equal(
            pfo.wrapper.grouper.group_by,
            pd.Index([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], name="group_config"),
        )
        assert_index_equal(
            pfo.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (0, "XOM"),
                    (0, "RRC"),
                    (0, "BBY"),
                    (0, "MA"),
                    (0, "PFE"),
                    (1, "XOM"),
                    (1, "RRC"),
                    (1, "BBY"),
                    (1, "MA"),
                    (1, "PFE"),
                ],
                names=["group_config", None],
            ),
        )
        assert pfo.wrapper.grouped_ndim == 2
        assert_index_equal(
            pfo.alloc_records.wrapper.columns,
            pd.Index([0, 1], name="group_config"),
        )
        assert pfo.alloc_records.wrapper.ndim == 2

    def test_from_pypfopt(self):
        if pypfopt_available:
            import pypfopt.expected_returns

            ef = pypfopt.efficient_frontier.EfficientFrontier(
                pypfopt.expected_returns.mean_historical_return(prices=prices),
                pypfopt.risk_models.CovarianceShrinkage(prices=prices).ledoit_wolf(),
            )
            ef.max_sharpe()
            np.testing.assert_array_equal(
                vbt.PortfolioOptimizer.from_pypfopt(prices=prices)._allocations[0],
                pd.Series(ef.clean_weights()).values,
            )

    def test_from_allocate_func(self):
        def get_allocations(pfo):
            idx = pfo.alloc_records.values["alloc_idx"]
            return np.array([prices.values[idx[i]] for i in range(len(pfo.alloc_records.values))])

        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point].values,
            vbt.Rep("index_point"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point].to_dict(),
            vbt.Rep("index_point"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: {
                k: v
                for k, v in prices.iloc[index_point].to_dict().items()
                if k in prices.columns[:-1]
            },
            vbt.Rep("index_point"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt),
        )
        allocations = get_allocations(pfo)
        allocations[:, -1] = np.nan
        np.testing.assert_array_equal(
            pfo._allocations,
            allocations,
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            every="2D",
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            every=vbt.RepEval("'2D'"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            index_points=[2, 4],
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 2), (1, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point: prices.iloc[index_point],
            vbt.Rep("index_point"),
            index_points=vbt.RepEval("[2, 4]"),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 2), (1, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda i, index_points: prices.iloc[index_points[i]],
            vbt.Rep("i"),
            vbt.Rep("index_points"),
            every="2D",
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            njit(lambda i, idx, prices: prices[idx]),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            njit(lambda i, idx, prices: prices[idx]),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            chunked=dict(arg_take_spec=dict(args=vbt.ArgsTaker(None))),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            njit(lambda i, idx, prices: prices[idx]),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            jitted=dict(parallel=True),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            vbt.Rep("allocate_func"),
            vbt.RepEval("prices.values", context=dict(prices=prices)),
            every="2D",
            jitted_loop=True,
            template_context=dict(allocate_func=njit(lambda i, idx, prices: prices[idx])),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo),
        )
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point, mult: prices.iloc[index_point] * mult,
            vbt.Rep("index_point"),
            vbt.Param([1, 2]),
        )
        assert isinstance(pfo.alloc_records, vbt.AllocPoints)
        assert_records_close(
            pfo.alloc_records.values,
            np.array(
                [
                    (0, 0, 0),
                    (1, 0, 1),
                    (2, 0, 2),
                    (3, 0, 3),
                    (4, 0, 4),
                    (0, 1, 0),
                    (1, 1, 1),
                    (2, 1, 2),
                    (3, 1, 3),
                    (4, 1, 4),
                ],
                dtype=alloc_point_dt,
            ),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            get_allocations(pfo) * np.array([[1], [1], [1], [1], [1], [2], [2], [2], [2], [2]]),
        )

        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper, lambda index_point: prices.iloc[index_point], vbt.Rep("index_point")
        )
        assert_index_equal(
            pfo.wrapper.grouper.group_by,
            pd.Index(["group", "group", "group", "group", "group"], name="group"),
        )
        assert_index_equal(
            pfo.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    ("group", "XOM"),
                    ("group", "RRC"),
                    ("group", "BBY"),
                    ("group", "MA"),
                    ("group", "PFE"),
                ],
                names=["group", None],
            ),
        )
        assert pfo.wrapper.grouped_ndim == 1
        assert_index_equal(
            pfo.alloc_records.wrapper.columns,
            pd.Index(["group"], name="group"),
        )
        assert pfo.alloc_records.wrapper.ndim == 1
        pfo = vbt.PortfolioOptimizer.from_allocate_func(
            prices.vbt.wrapper,
            lambda index_point, mult: prices.iloc[index_point] * mult,
            vbt.Rep("index_point"),
            group_configs=[dict(args_1=1), dict(args_1=2)],
        )
        assert_index_equal(
            pfo.wrapper.grouper.group_by,
            pd.Index([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], name="group_config"),
        )
        assert_index_equal(
            pfo.wrapper.columns,
            pd.MultiIndex.from_tuples(
                [
                    (0, "XOM"),
                    (0, "RRC"),
                    (0, "BBY"),
                    (0, "MA"),
                    (0, "PFE"),
                    (1, "XOM"),
                    (1, "RRC"),
                    (1, "BBY"),
                    (1, "MA"),
                    (1, "PFE"),
                ],
                names=["group_config", None],
            ),
        )
        assert pfo.wrapper.grouped_ndim == 2
        assert_index_equal(
            pfo.alloc_records.wrapper.columns,
            pd.Index([0, 1], name="group_config"),
        )
        assert pfo.alloc_records.wrapper.ndim == 2

    def test_from_allocations(self):
        pfo = vbt.PortfolioOptimizer.from_allocations(
            prices.vbt.wrapper,
            prices.iloc[::2],
        )
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            prices.values[::2],
        )
        with pytest.raises(Exception):
            vbt.PortfolioOptimizer.from_allocations(
                prices.vbt.wrapper,
                prices.values[::2],
            )
        pfo = vbt.PortfolioOptimizer.from_allocations(
            prices.vbt.wrapper,
            prices.values[::2],
            every="2D",
        )
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            prices.values[::2],
        )
        pfo = vbt.PortfolioOptimizer.from_allocations(
            prices.vbt.wrapper,
            prices.iloc[::2].to_dict(orient="records"),
            every="2D",
        )
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            prices.values[::2],
        )

    def test_from_filled_allocations(self):
        filled_allocations = prices.copy()
        filled_allocations.iloc[1::2] = 0
        pfo = vbt.PortfolioOptimizer.from_filled_allocations(filled_allocations)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 2), (2, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            prices.values[::2],
        )
        pfo = vbt.PortfolioOptimizer.from_filled_allocations(
            filled_allocations,
            valid_only=False,
            nonzero_only=False,
            unique_only=False,
        )
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            filled_allocations,
        )
        with pytest.raises(Exception):
            vbt.PortfolioOptimizer.from_filled_allocations(
                prices.vbt.wrapper,
                filled_allocations.iloc[::2],
            )

    def test_from_uniform(self):
        pfo = vbt.PortfolioOptimizer.from_uniform(prices.vbt.wrapper, on=3)
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 3)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
        )

    def test_from_random(self):
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=3,
        )
        assert_records_close(
            pfo.alloc_records.values,
            np.array([(0, 0, 3)], dtype=alloc_point_dt),
        )
        np.testing.assert_array_equal(
            pfo._allocations,
            np.array(
                [
                    [
                        0.13319702814025883,
                        0.33810081711389406,
                        0.26031768763785473,
                        0.2128998389048247,
                        0.05548462820316767,
                    ]
                ]
            ),
        )

    def test_from_universal_algo(self):
        if universal_available:
            pfo = vbt.PortfolioOptimizer.from_universal_algo(
                "CRP",
                prices,
            )
            assert_records_close(
                pfo.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pfo._allocations,
                np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
            )
            pfo = vbt.PortfolioOptimizer.from_universal_algo(
                "CRP", prices, valid_only=False, unique_only=False
            )
            assert_records_close(
                pfo.alloc_records.values,
                np.array(
                    [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)], dtype=alloc_point_dt
                ),
            )
            np.testing.assert_array_equal(
                pfo._allocations,
                np.array(
                    [
                        [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                        [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                        [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                        [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                        [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
                    ]
                ),
            )
            pfo = vbt.PortfolioOptimizer.from_universal_algo(
                universal.algos.CRP,
                prices,
            )
            assert_records_close(
                pfo.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pfo._allocations,
                np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
            )
            pfo = vbt.PortfolioOptimizer.from_universal_algo(
                universal.algos.CRP(),
                prices,
            )
            assert_records_close(
                pfo.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pfo._allocations,
                np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
            )
            pfo = vbt.PortfolioOptimizer.from_universal_algo(
                universal.algos.CRP().run(prices),
                wrapper=prices.vbt.wrapper,
            )
            assert_records_close(
                pfo.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pfo._allocations,
                np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]]),
            )
            pfo = vbt.PortfolioOptimizer.from_universal_algo(
                "DynamicCRP",
                prices,
                n=52,
                min_history=8,
            )
            assert_records_close(
                pfo.alloc_records.values,
                np.array([(0, 0, 0)], dtype=alloc_point_dt),
            )
            np.testing.assert_array_equal(
                pfo._allocations,
                universal.algos.DynamicCRP(n=52, min_history=8).run(prices).weights.values[[0]],
            )

    def test_get_allocations(self):
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=[1, 3],
        )
        target_arr = [
            [
                0.13319702814025883,
                0.33810081711389406,
                0.26031768763785473,
                0.2128998389048247,
                0.05548462820316767,
            ],
            [
                0.06528491964469331,
                0.02430844330237927,
                0.3625014516740258,
                0.2515713061862386,
                0.29633387919266296,
            ],
        ]
        assert_frame_equal(
            pfo.get_allocations(squeeze_groups=False),
            pd.DataFrame(
                target_arr,
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.Index(["group", "group"], name="group"),
                        prices.index[[1, 3]],
                    ]
                ),
                columns=prices.columns,
            ),
        )
        assert_frame_equal(
            pfo.get_allocations(squeeze_groups=True),
            pd.DataFrame(
                target_arr,
                index=prices.index[[1, 3]],
                columns=prices.columns,
            ),
        )
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            group_configs=[dict(on=[1, 3], _name="g1"), dict(on=[2, 4], _name="g2")],
        )
        target_arr = np.array(
            [
                [
                    0.13319702814025883,
                    0.33810081711389406,
                    0.26031768763785473,
                    0.2128998389048247,
                    0.05548462820316767,
                ],
                [
                    0.06528491964469331,
                    0.02430844330237927,
                    0.3625014516740258,
                    0.2515713061862386,
                    0.29633387919266296,
                ],
                [
                    0.00928441856775223,
                    0.4374675865753444,
                    0.37546445396096845,
                    0.09577331138241256,
                    0.0820102295135221,
                ],
                [
                    0.10567348701744264,
                    0.17529742718559654,
                    0.302352663027335,
                    0.2488768479913802,
                    0.1677995747782457,
                ],
            ]
        )
        assert_frame_equal(
            pfo.get_allocations(squeeze_groups=False),
            pd.DataFrame(
                target_arr,
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.Index(["g1", "g1", "g2", "g2"], name="group_config"),
                        prices.index[[1, 3, 2, 4]],
                    ]
                ),
                columns=prices.columns,
            ),
        )
        assert_frame_equal(
            pfo.get_allocations(squeeze_groups=True),
            pd.DataFrame(
                target_arr,
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.Index(["g1", "g1", "g2", "g2"], name="group_config"),
                        prices.index[[1, 3, 2, 4]],
                    ]
                ),
                columns=prices.columns,
            ),
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_loc=4,
        )
        assert_frame_equal(
            pfo.get_allocations(squeeze_groups=False),
            pd.DataFrame(
                prices.sum().values[None],
                index=pd.Index(["group"], dtype="object", name="group"),
                columns=prices.columns,
            ),
        )
        assert_frame_equal(
            pfo.get_allocations(squeeze_groups=True),
            pd.DataFrame(
                prices.sum().values[None],
                index=prices.index[[4]],
                columns=prices.columns,
            ),
        )

    def test_fill_allocations(self):
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            on=[1, 3],
        )
        target_arr = [
            [
                0.13319702814025883,
                0.33810081711389406,
                0.26031768763785473,
                0.2128998389048247,
                0.05548462820316767,
            ],
            [
                0.06528491964469331,
                0.02430844330237927,
                0.3625014516740258,
                0.2515713061862386,
                0.29633387919266296,
            ],
        ]
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=pd.MultiIndex.from_arrays(
                [
                    pd.Index(["group", "group", "group", "group", "group"], name="group"),
                    prices.columns,
                ]
            ),
        )
        target_df.iloc[[1, 3]] = target_arr
        assert_frame_equal(
            pfo.fill_allocations(squeeze_groups=False),
            target_df,
        )
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=prices.columns,
        )
        target_df.iloc[[1, 3]] = target_arr
        assert_frame_equal(
            pfo.fill_allocations(squeeze_groups=True),
            target_df,
        )
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            group_configs=[dict(on=[1, 3], _name="g1"), dict(on=[2, 4], _name="g2")],
        )
        target_arr = np.array(
            [
                [
                    0.13319702814025883,
                    0.33810081711389406,
                    0.26031768763785473,
                    0.2128998389048247,
                    0.05548462820316767,
                ],
                [
                    0.06528491964469331,
                    0.02430844330237927,
                    0.3625014516740258,
                    0.2515713061862386,
                    0.29633387919266296,
                ],
                [
                    0.00928441856775223,
                    0.4374675865753444,
                    0.37546445396096845,
                    0.09577331138241256,
                    0.0820102295135221,
                ],
                [
                    0.10567348701744264,
                    0.17529742718559654,
                    0.302352663027335,
                    0.2488768479913802,
                    0.1677995747782457,
                ],
            ]
        )
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=pd.MultiIndex.from_arrays(
                [
                    pd.Index(
                        ["g1", "g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2", "g2"],
                        name="group_config",
                    ),
                    prices.columns.append(prices.columns),
                ]
            ),
        )
        target_df.iloc[[1, 3], 0:5] = target_arr[0:2]
        target_df.iloc[[2, 4], 5:10] = target_arr[2:4]
        assert_frame_equal(pfo.fill_allocations(squeeze_groups=False), target_df)
        assert_frame_equal(pfo.fill_allocations(squeeze_groups=True), target_df)
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            index_loc=4,
        )
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=pd.MultiIndex.from_arrays(
                [
                    pd.Index(["group", "group", "group", "group", "group"], name="group"),
                    prices.columns,
                ]
            ),
        )
        target_df.iloc[[4]] = prices.sum().values
        assert_frame_equal(
            pfo.fill_allocations(squeeze_groups=False),
            target_df,
        )
        target_df = pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=prices.columns,
        )
        target_df.iloc[[4]] = prices.sum().values
        assert_frame_equal(
            pfo.fill_allocations(squeeze_groups=True),
            target_df,
        )

    def test_points_stats(self):
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            group_configs=[dict(on=[1, 3], _name="g1"), dict(on=[2, 4], _name="g2")],
        )
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Records",
                "Mean Allocation: XOM",
                "Mean Allocation: RRC",
                "Mean Allocation: BBY",
                "Mean Allocation: MA",
                "Mean Allocation: PFE",
            ],
            dtype="object",
        )
        assert_series_equal(
            pfo.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    2.0,
                    0.07835996334253675,
                    0.24379356854430356,
                    0.32515906407504597,
                    0.202280326116214,
                    0.1504070779218996,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            pfo.stats(column="g1"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    0.09924097389247608,
                    0.18120463020813665,
                    0.31140956965594024,
                    0.23223557254553162,
                    0.17590925369791532,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(pfo["g1"].stats(), pfo.stats(column="g1"))
        stats_df = pfo.stats(agg_func=None)
        assert stats_df.shape == (2, 9)
        assert_index_equal(stats_df.index, pfo.wrapper.get_columns())
        assert_index_equal(stats_df.columns, stats_index)

    def test_ranges_stats(self):
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            group_configs=[
                dict(index_ranges=[0, 2], _name="g1"),
                dict(index_ranges=[1, 3], _name="g2"),
            ],
        )
        stats_index = pd.Index(
            [
                "Start Index",
                "End Index",
                "Total Duration",
                "Total Records",
                "Coverage",
                "Overlap Coverage",
                "Mean Allocation: XOM",
                "Mean Allocation: RRC",
                "Mean Allocation: BBY",
                "Mean Allocation: MA",
                "Mean Allocation: PFE",
            ],
            dtype="object",
        )
        assert_series_equal(
            pfo.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    1.0,
                    0.4,
                    0.0,
                    108.68882550000001,
                    103.4886705,
                    66.15678550000001,
                    44.069272,
                    27.5600615,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        assert_series_equal(
            pfo.stats(column="g1"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00"),
                    pd.Timestamp("2020-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    1,
                    0.4,
                    0.0,
                    108.348701,
                    103.293606,
                    65.873542,
                    44.059574999999995,
                    27.681569,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        assert_series_equal(pfo["g1"].stats(), pfo.stats(column="g1"))
        stats_df = pfo.stats(agg_func=None)
        assert stats_df.shape == (2, 11)
        assert_index_equal(stats_df.index, pfo.wrapper.get_columns())
        assert_index_equal(stats_df.columns, stats_index)

    def test_plots(self):
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            group_configs=[dict(on=[1, 3], _name="g1"), dict(on=[2, 4], _name="g2")],
        )
        pfo["g1"].plot()
        pfo[["g1"]].plot()
        pfo.plot(column="g1")
        pfo.plots(column="g1")
        with pytest.raises(Exception):
            pfo.plots()

        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            group_configs=[
                dict(index_ranges=[0, 2], _name="g1"),
                dict(index_ranges=[1, 3], _name="g2"),
            ],
        )
        pfo["g1"].plot()
        pfo[["g1"]].plot()
        pfo.plot(column="g1")
        pfo.plots(column="g1")
        with pytest.raises(Exception):
            pfo.plots()


# ############# Portfolio ############# #


class TestPortfolio:
    def test_from_optimizer(self):
        pfo = vbt.PortfolioOptimizer.from_random(
            prices.vbt.wrapper,
            seed=42,
            group_configs=[dict(on=[1, 3], _name="g1"), dict(on=[2, 4], _name="g2")],
        )
        pf = vbt.Portfolio.from_optimizer(prices, pfo)
        allocations = pf.get_asset_value(group_by=False).vbt / pf.value
        np.testing.assert_allclose(
            allocations.values[1][:5],
            pfo.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][:5],
            pfo.allocations.values[1],
        )
        np.testing.assert_allclose(
            allocations.values[2][5:10],
            pfo.allocations.values[2],
        )
        np.testing.assert_allclose(
            allocations.values[4][5:10],
            pfo.allocations.values[3],
        )
        assert_frame_equal(
            vbt.Portfolio.from_optimizer(prices, pfo, pf_method="from_orders", val_price=-np.inf)
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
            vbt.Portfolio.from_optimizer(prices, pfo, pf_method="from_signals", val_price=-np.inf)
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum()
            / prices.iloc[index_slice].sum().sum(),
            vbt.Rep("index_slice"),
            group_configs=[
                dict(index_ranges=[0, 2], _name="g1"),
                dict(index_ranges=[1, 3], _name="g2"),
            ],
        )
        pf = vbt.Portfolio.from_optimizer(prices, pfo)
        allocations = pf.get_asset_value(group_by=False).vbt / pf.value
        np.testing.assert_allclose(
            allocations.values[2][:5],
            pfo.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][5:10],
            pfo.allocations.values[1],
        )
        assert_frame_equal(
            vbt.Portfolio.from_optimizer(prices, pfo, pf_method="from_orders", val_price=-np.inf)
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
            vbt.Portfolio.from_optimizer(prices, pfo, pf_method="from_signals", val_price=-np.inf)
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: prices.iloc[index_slice].sum(),
            vbt.Rep("index_slice"),
            group_configs=[
                dict(index_ranges=[0, 2], _name="g1"),
                dict(index_ranges=[1, 3], _name="g2"),
            ],
        )
        pf = vbt.Portfolio.from_optimizer(prices, pfo, init_cash="auto", size_type="amount")
        allocations = pf.asset_flow
        np.testing.assert_allclose(
            allocations.values[2][:5],
            pfo.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][5:10],
            pfo.allocations.values[1],
        )
        assert_frame_equal(
            vbt.Portfolio.from_optimizer(
                prices, pfo, pf_method="from_orders", init_cash="auto", size_type="amount"
            )
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
            vbt.Portfolio.from_optimizer(
                prices, pfo, pf_method="from_signals", init_cash="auto", size_type="amount"
            )
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
        )
        pfo = vbt.PortfolioOptimizer.from_optimize_func(
            prices.vbt.wrapper,
            lambda index_slice: -prices.iloc[index_slice].sum()
            / prices.iloc[index_slice].sum().sum(),
            vbt.Rep("index_slice"),
            group_configs=[
                dict(index_ranges=[0, 2], _name="g1"),
                dict(index_ranges=[1, 3], _name="g2"),
            ],
        )
        pf = vbt.Portfolio.from_optimizer(prices, pfo)
        allocations = pf.get_asset_value(group_by=False).vbt / pf.value
        np.testing.assert_allclose(
            allocations.values[2][:5],
            pfo.allocations.values[0],
        )
        np.testing.assert_allclose(
            allocations.values[3][5:10],
            pfo.allocations.values[1],
        )
        assert_frame_equal(
            vbt.Portfolio.from_optimizer(prices, pfo, pf_method="from_orders", val_price=-np.inf)
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
            vbt.Portfolio.from_optimizer(prices, pfo, pf_method="from_signals", val_price=-np.inf)
            .get_asset_value(group_by=False)
            .vbt
            / pf.value,
        )
