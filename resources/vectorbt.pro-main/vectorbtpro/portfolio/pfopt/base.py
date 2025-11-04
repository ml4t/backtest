# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the base functions and classes for portfolio optimization."""

import inspect

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexes import combine_indexes, select_levels, stack_indexes
from vectorbtpro.base.indexing import point_idxr_defaults, range_idxr_defaults
from vectorbtpro.base.merging import row_stack_arrays
from vectorbtpro.base.reshaping import (
    broadcast_array_to,
    to_1d_array,
    to_2d_array,
    to_dict,
    to_pd_array,
)
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.data.base import Data
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.enums import RangeStatus
from vectorbtpro.generic.splitting.base import Splitter, Takeable
from vectorbtpro.portfolio.enums import Direction
from vectorbtpro.portfolio.pfopt import nb
from vectorbtpro.portfolio.pfopt.records import AllocPoints, AllocRanges
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.annotations import has_annotatables
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.execution import Task, execute
from vectorbtpro.utils.params import Param, Parameterizer, combine_params
from vectorbtpro.utils.parsing import (
    ann_args_to_args,
    annotate_args,
    flatten_ann_args,
    get_func_arg_names,
    unflatten_ann_args,
)
from vectorbtpro.utils.pickling import pdict
from vectorbtpro.utils.random_ import set_seed_nb
from vectorbtpro.utils.template import CustomTemplate, Rep, RepFunc, substitute_templates
from vectorbtpro.utils.warnings_ import WarningsFiltered, warn, warn_stdout

if tp.TYPE_CHECKING:
    from vectorbtpro.portfolio.base import Portfolio as PortfolioT
else:
    PortfolioT = "vectorbtpro.portfolio.base.Portfolio"
if tp.TYPE_CHECKING:
    from pypfopt.base_optimizer import BaseOptimizer as BaseOptimizerT
else:
    BaseOptimizerT = "pypfopt.base_optimizer.BaseOptimizer"
if tp.TYPE_CHECKING:
    from riskfolio import HCPortfolio as RHCPortfolioT
    from riskfolio import Portfolio as RPortfolioT
else:
    RPortfolioT = "riskfolio.Portfolio"
    RHCPortfolioT = "riskfolio.HCPortfolio"
if tp.TYPE_CHECKING:
    from universal.algo import Algo as AlgoT
    from universal.result import AlgoResult as AlgoResultT
else:
    AlgoT = "universal.algo.Algo"
    AlgoResultT = "universal.result.AlgoResult"

__all__ = [
    "pfopt_func_dict",
    "pypfopt_optimize",
    "riskfolio_optimize",
    "PortfolioOptimizer",
    "PFO",
]

__pdoc__ = {}


# ############# PyPortfolioOpt ############# #


class pfopt_func_dict(pdict):
    """Dictionary subclass containing optimization functions as keys.

    Keys may be function objects, their names, or the special key `_def` to represent the default value.
    """

    pass


def select_pfopt_func_kwargs(
    pypfopt_func: tp.Callable,
    kwargs: tp.Union[None, tp.Kwargs, pfopt_func_dict] = None,
) -> tp.Kwargs:
    """Select keyword arguments corresponding to `pypfopt_func`.

    Determines and returns a dictionary of keyword arguments intended for the provided `pypfopt_func`
    based on the structure of `kwargs`.

    Args:
        pypfopt_func (Callable): Function for which to select keyword arguments.
        kwargs (Union[None, Kwargs, pfopt_func_dict]): Dictionary containing keyword arguments for one or
            multiple functions.

    Returns:
        Kwargs: Dictionary of keyword arguments for `pypfopt_func`.
    """
    if kwargs is None:
        return {}
    if isinstance(kwargs, pfopt_func_dict):
        if pypfopt_func in kwargs:
            _kwargs = kwargs[pypfopt_func]
        elif pypfopt_func.__name__ in kwargs:
            _kwargs = kwargs[pypfopt_func.__name__]
        elif "_def" in kwargs:
            _kwargs = kwargs["_def"]
        else:
            _kwargs = {}
    else:
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, pfopt_func_dict):
                if pypfopt_func in v:
                    _kwargs[k] = v[pypfopt_func]
                elif pypfopt_func.__name__ in v:
                    _kwargs[k] = v[pypfopt_func.__name__]
                elif "_def" in v:
                    _kwargs[k] = v["_def"]
            else:
                _kwargs[k] = v
    return _kwargs


def resolve_pypfopt_func_kwargs(
    pypfopt_func: tp.Callable,
    cache: tp.KwargsLike = None,
    var_kwarg_names: tp.Optional[tp.Iterable[str]] = None,
    used_arg_names: tp.Optional[tp.Set[str]] = None,
    **kwargs,
) -> tp.Kwargs:
    """Resolve keyword arguments for a PyPortfolioOpt optimization function.

    Parses the signature of `pypfopt_func` and, for each accepted parameter, checks for a matching
    name in `kwargs`. If a corresponding argument is not provided, the function attempts to resolve it
    using other inputs or by invoking additional optimization functions.

    The `frequency` parameter is resolved using the global `freq` and `year_freq` values via
    `ReturnsAccessor.get_ann_factor`.

    Any argument in `kwargs` may be wrapped with `pfopt_func_dict` to specify function-specific values.

    Args:
        pypfopt_func (Callable): PyPortfolioOpt optimization function whose signature is parsed.
        cache (KwargsLike): Cache dictionary for storing resolved arguments.
        var_kwarg_names (Optional[Iterable[str]]): Names of variable keyword arguments to process.
        used_arg_names (Optional[Set[str]]): Set accumulating the names of arguments that have been resolved.
        **kwargs: Keyword arguments for parameter resolution.

    Returns:
        Kwargs: Dictionary of resolved keyword arguments for use with `pypfopt_func`.

    !!! note
        When using custom functions, ensure that their parameters are explicitly defined (i.e., avoid variable
        arguments) and adhere to PyPortfolioOpt's naming conventions.

        Functions `market_implied_prior_returns` and `BlackLittermanModel.bl_weights` require `risk_aversion`,
        which differs from similarly named parameters in other functions. To configure it, pass `delta`.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")

    signature = inspect.signature(pypfopt_func)
    kwargs = select_pfopt_func_kwargs(pypfopt_func, kwargs)
    if cache is None:
        cache = {}
    arg_names = get_func_arg_names(pypfopt_func)
    if len(arg_names) == 0:
        return {}
    if used_arg_names is None:
        used_arg_names = set()

    pass_kwargs = dict()

    def _process_arg(arg_name, arg_value):
        orig_arg_name = arg_name
        if pypfopt_func.__name__ in ("market_implied_prior_returns", "bl_weights"):
            if arg_name == "risk_aversion":
                # In some methods, risk_aversion is expected as array and means delta
                arg_name = "delta"

        def _get_kwarg(*args):
            used_arg_names.add(args[0])
            return kwargs.get(*args)

        def _get_prices():
            prices = None
            if "prices" in cache:
                prices = cache["prices"]
            elif "prices" in kwargs:
                if not _get_kwarg("returns_data", False):
                    prices = _get_kwarg("prices")
            return prices

        def _get_returns():
            returns = None
            if "returns" in cache:
                returns = cache["returns"]
            elif "returns" in kwargs:
                returns = _get_kwarg("returns")
            elif "prices" in kwargs and _get_kwarg("returns_data", False):
                returns = _get_kwarg("prices")
            return returns

        def _prices_from_returns():
            from pypfopt.expected_returns import prices_from_returns

            cache["prices"] = prices_from_returns(_get_returns(), _get_kwarg("log_returns", False))
            return cache["prices"]

        def _returns_from_prices():
            from pypfopt.expected_returns import returns_from_prices

            cache["returns"] = returns_from_prices(_get_prices(), _get_kwarg("log_returns", False))
            return cache["returns"]

        if arg_name == "expected_returns":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "expected_returns" not in cache:
                cache["expected_returns"] = resolve_pypfopt_expected_returns(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["expected_returns"]
        elif arg_name == "cov_matrix":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "cov_matrix" not in cache:
                cache["cov_matrix"] = resolve_pypfopt_cov_matrix(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["cov_matrix"]
        elif arg_name == "optimizer":
            if arg_name in kwargs:
                used_arg_names.add(arg_name)
            if "optimizer" not in cache:
                cache["optimizer"] = resolve_pypfopt_optimizer(
                    cache=cache,
                    used_arg_names=used_arg_names,
                    **kwargs,
                )
            pass_kwargs[orig_arg_name] = cache["optimizer"]

        if orig_arg_name not in pass_kwargs:
            if arg_name in kwargs:
                if arg_name == "market_prices":
                    if pypfopt_func.__name__ != "market_implied_risk_aversion" and checks.is_series(
                        _get_kwarg(arg_name)
                    ):
                        pass_kwargs[orig_arg_name] = (
                            _get_kwarg(arg_name).to_frame().copy(deep=False)
                        )
                    else:
                        pass_kwargs[orig_arg_name] = _get_kwarg(arg_name).copy(deep=False)
                else:
                    pass_kwargs[orig_arg_name] = _get_kwarg(arg_name)
            else:
                if arg_name == "frequency":
                    ann_factor = ReturnsAccessor.get_ann_factor(
                        year_freq=_get_kwarg("year_freq", None),
                        freq=_get_kwarg("freq", None),
                    )
                    if ann_factor is not None:
                        pass_kwargs[orig_arg_name] = ann_factor
                elif arg_name == "prices":
                    if "returns_data" in arg_names:
                        if "returns_data" in kwargs:
                            if _get_kwarg("returns_data", False):
                                if _get_returns() is not None:
                                    pass_kwargs[orig_arg_name] = _get_returns()
                                elif _get_prices() is not None:
                                    pass_kwargs[orig_arg_name] = _returns_from_prices()
                            else:
                                if _get_prices() is not None:
                                    pass_kwargs[orig_arg_name] = _get_prices()
                                elif _get_returns() is not None:
                                    pass_kwargs[orig_arg_name] = _prices_from_returns()
                        else:
                            if _get_prices() is not None:
                                pass_kwargs[orig_arg_name] = _get_prices()
                                pass_kwargs["returns_data"] = False
                            elif _get_returns() is not None:
                                pass_kwargs[orig_arg_name] = _get_returns()
                                pass_kwargs["returns_data"] = True
                    else:
                        if _get_prices() is not None:
                            pass_kwargs[orig_arg_name] = _get_prices()
                        elif _get_returns() is not None:
                            pass_kwargs[orig_arg_name] = _prices_from_returns()
                elif arg_name == "returns":
                    if _get_returns() is not None:
                        pass_kwargs[orig_arg_name] = _get_returns()
                    elif _get_prices() is not None:
                        pass_kwargs[orig_arg_name] = _returns_from_prices()
                elif arg_name == "latest_prices":
                    from pypfopt.discrete_allocation import get_latest_prices

                    if _get_prices() is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(
                            _get_prices()
                        )
                    elif _get_returns() is not None:
                        pass_kwargs[orig_arg_name] = cache["latest_prices"] = get_latest_prices(
                            _prices_from_returns()
                        )
                elif arg_name == "delta":
                    if "delta" not in cache:
                        from pypfopt.black_litterman import market_implied_risk_aversion

                        cache["delta"] = resolve_pypfopt_func_call(
                            market_implied_risk_aversion,
                            cache=cache,
                            used_arg_names=used_arg_names,
                            **kwargs,
                        )
                    pass_kwargs[orig_arg_name] = cache["delta"]
                elif arg_name == "pi":
                    if "pi" not in cache:
                        from pypfopt.black_litterman import market_implied_prior_returns

                        cache["pi"] = resolve_pypfopt_func_call(
                            market_implied_prior_returns,
                            cache=cache,
                            used_arg_names=used_arg_names,
                            **kwargs,
                        )
                    pass_kwargs[orig_arg_name] = cache["pi"]

        if orig_arg_name not in pass_kwargs:
            if arg_value.default != inspect.Parameter.empty:
                pass_kwargs[orig_arg_name] = arg_value.default

    for arg_name, arg_value in signature.parameters.items():
        if arg_value.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"Variable positional arguments in {pypfopt_func} cannot be parsed")
        elif arg_value.kind == inspect.Parameter.VAR_KEYWORD:
            if var_kwarg_names is None:
                var_kwarg_names = []
            for var_arg_name in var_kwarg_names:
                _process_arg(var_arg_name, arg_value)
        else:
            _process_arg(arg_name, arg_value)

    return pass_kwargs


def resolve_pypfopt_func_call(pypfopt_func: tp.Callable, **kwargs) -> tp.Any:
    """Resolve keyword arguments and invoke a PyPortfolioOpt function.

    This function resolves parameters by using `resolve_pypfopt_func_kwargs` with the supplied
    `kwargs` and then calls `pypfopt_func` with the resolved arguments.

    Args:
        pypfopt_func (Callable): PyPortfolioOpt function to be called.
        **kwargs: Keyword arguments for `resolve_pypfopt_func_kwargs`.

    Returns:
        Any: Result of calling `pypfopt_func` with the resolved keyword arguments.
    """
    return pypfopt_func(**resolve_pypfopt_func_kwargs(pypfopt_func, **kwargs))


def resolve_pypfopt_expected_returns(
    expected_returns: tp.Union[tp.Callable, tp.AnyArray, str] = "mean_historical_return",
    **kwargs,
) -> tp.AnyArray:
    """Resolve the expected returns.

    This function resolves the expected returns for portfolio optimization using pypfopt.

    Any provided function is invoked using `resolve_pypfopt_func_call`.

    Args:
        expected_returns (Union[Callable, AnyArray, str]): Expected returns indicator, function, or array.

            * If an array, it is returned unchanged.
            * If a callable, it is executed via `resolve_pypfopt_func_call`.
            * If a string, it selects one of the following models:
                * `mean_historical_return`: resolved from `pypfopt.expected_returns.mean_historical_return`
                *  `ema_historical_return`: resolved from `pypfopt.expected_returns.ema_historical_return`
                *  `capm_return`: resolved from `pypfopt.expected_returns.capm_return`
                *  `bl_returns`: resolved from `pypfopt.black_litterman.BlackLittermanModel.bl_returns`
        **kwargs: Keyword arguments for the expected returns function.

    Returns:
        AnyArray: Computed expected returns.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")

    if isinstance(expected_returns, str):
        if expected_returns.lower() == "mean_historical_return":
            from pypfopt.expected_returns import mean_historical_return

            return resolve_pypfopt_func_call(mean_historical_return, **kwargs)
        if expected_returns.lower() == "ema_historical_return":
            from pypfopt.expected_returns import ema_historical_return

            return resolve_pypfopt_func_call(ema_historical_return, **kwargs)
        if expected_returns.lower() == "capm_return":
            from pypfopt.expected_returns import capm_return

            return resolve_pypfopt_func_call(capm_return, **kwargs)
        if expected_returns.lower() == "bl_returns":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pypfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            ).bl_returns()
        import pypfopt.expected_returns

        if hasattr(pypfopt.expected_returns, expected_returns):
            return resolve_pypfopt_func_call(
                getattr(pypfopt.expected_returns, expected_returns), **kwargs
            )
        raise NotImplementedError(f"Return model '{expected_returns}' is not supported")
    if callable(expected_returns):
        return resolve_pypfopt_func_call(expected_returns, **kwargs)
    return expected_returns


def resolve_pypfopt_cov_matrix(
    cov_matrix: tp.Union[tp.Callable, tp.AnyArray, str] = "ledoit_wolf",
    **kwargs,
) -> tp.AnyArray:
    """Resolve the covariance matrix.

    This function resolves the covariance matrix for portfolio optimization using pypfopt.

    Any provided function is invoked using `resolve_pypfopt_func_call`.

    Args:
        cov_matrix (Union[Callable, AnyArray, str]): Covariance matrix indicator, function, or array.

            * If an array, it is returned unchanged.
            * If a callable, it is executed via `resolve_pypfopt_func_call`.
            * If a string, it selects one of the following risk models:
                * `sample_cov`: resolved from `pypfopt.risk_models.sample_cov`
                * `semicovariance` or `semivariance`: resolved from `pypfopt.risk_models.semicovariance`
                * `exp_cov`: resolved from `pypfopt.risk_models.exp_cov`
                * `ledoit_wolf` or `ledoit_wolf_constant_variance`: resolved from
                    `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf` with constant variance shrinkage
                * `ledoit_wolf_single_factor`: resolved from
                    `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf` with single factor shrinkage
                * `ledoit_wolf_constant_correlation`: resolved from
                    `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf` with constant correlation shrinkage
                * `oracle_approximating`: resolved from
                    `pypfopt.risk_models.CovarianceShrinkage.ledoit_wolf` with oracle approximating shrinkage
        **kwargs: Keyword arguments for the covariance model function.

    Returns:
        AnyArray: Computed covariance matrix.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")

    if isinstance(cov_matrix, str):
        if cov_matrix.lower() == "sample_cov":
            from pypfopt.risk_models import sample_cov

            return resolve_pypfopt_func_call(sample_cov, var_kwarg_names=["fix_method"], **kwargs)
        if cov_matrix.lower() == "semicovariance" or cov_matrix.lower() == "semivariance":
            from pypfopt.risk_models import semicovariance

            return resolve_pypfopt_func_call(
                semicovariance, var_kwarg_names=["fix_method"], **kwargs
            )
        if cov_matrix.lower() == "exp_cov":
            from pypfopt.risk_models import exp_cov

            return resolve_pypfopt_func_call(exp_cov, var_kwarg_names=["fix_method"], **kwargs)
        if (
            cov_matrix.lower() == "ledoit_wolf"
            or cov_matrix.lower() == "ledoit_wolf_constant_variance"
        ):
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf()
        if cov_matrix.lower() == "ledoit_wolf_single_factor":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(
                shrinkage_target="single_factor"
            )
        if cov_matrix.lower() == "ledoit_wolf_constant_correlation":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).ledoit_wolf(
                shrinkage_target="constant_correlation"
            )
        if cov_matrix.lower() == "oracle_approximating":
            from pypfopt.risk_models import CovarianceShrinkage

            return resolve_pypfopt_func_call(CovarianceShrinkage, **kwargs).oracle_approximating()
        import pypfopt.risk_models

        if hasattr(pypfopt.risk_models, cov_matrix):
            return resolve_pypfopt_func_call(getattr(pypfopt.risk_models, cov_matrix), **kwargs)
        raise NotImplementedError(f"Risk model '{cov_matrix}' is not supported")
    if callable(cov_matrix):
        return resolve_pypfopt_func_call(cov_matrix, **kwargs)
    return cov_matrix


def resolve_pypfopt_optimizer(
    optimizer: tp.Union[tp.Callable, BaseOptimizerT, str] = "efficient_frontier",
    **kwargs,
) -> BaseOptimizerT:
    """Resolve the PyPortfolioOpt optimizer.

    Resolves the given optimizer to a `pypfopt.base_optimizer.BaseOptimizer` instance based on the provided input.
    The optimizer can be specified as an instance, callable, subclass, or a recognized string identifier.

    Args:
        optimizer (Union[Callable, BaseOptimizerT, str]):
            Either an instance, callable, or subclass of `pypfopt.base_optimizer.BaseOptimizer`.

            Alternatively, one of the following string identifiers is supported:

            * 'efficient_frontier': `pypfopt.efficient_frontier.EfficientFrontier`
            * 'efficient_cdar': `pypfopt.efficient_frontier.EfficientCDaR`
            * 'efficient_cvar': `pypfopt.efficient_frontier.EfficientCVaR`
            * 'efficient_semivariance': `pypfopt.efficient_frontier.EfficientSemivariance`
            * 'black_litterman' or 'bl': `pypfopt.black_litterman.BlackLittermanModel`
            * 'hierarchical_portfolio', 'hrpopt', or 'hrp': `pypfopt.hierarchical_portfolio.HRPOpt`
            * 'cla': `pypfopt.cla.CLA`
        **kwargs: Keyword arguments for the optimization function.

    Returns:
        BaseOptimizer: Instance of `pypfopt.base_optimizer.BaseOptimizer`.

    !!! note
        Resolution is delegated to `resolve_pypfopt_func_call`.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")
    from pypfopt.base_optimizer import BaseOptimizer

    if isinstance(optimizer, str):
        if optimizer.lower() == "efficient_frontier":
            from pypfopt.efficient_frontier import EfficientFrontier

            return resolve_pypfopt_func_call(EfficientFrontier, **kwargs)
        if optimizer.lower() == "efficient_cdar":
            from pypfopt.efficient_frontier import EfficientCDaR

            return resolve_pypfopt_func_call(EfficientCDaR, **kwargs)
        if optimizer.lower() == "efficient_cvar":
            from pypfopt.efficient_frontier import EfficientCVaR

            return resolve_pypfopt_func_call(EfficientCVaR, **kwargs)
        if optimizer.lower() == "efficient_semivariance":
            from pypfopt.efficient_frontier import EfficientSemivariance

            return resolve_pypfopt_func_call(EfficientSemivariance, **kwargs)
        if optimizer.lower() == "black_litterman" or optimizer.lower() == "bl":
            from pypfopt.black_litterman import BlackLittermanModel

            return resolve_pypfopt_func_call(
                BlackLittermanModel,
                var_kwarg_names=["market_caps", "risk_free_rate"],
                **kwargs,
            )
        if (
            optimizer.lower() == "hierarchical_portfolio"
            or optimizer.lower() == "hrpopt"
            or optimizer.lower() == "hrp"
        ):
            from pypfopt.hierarchical_portfolio import HRPOpt

            return resolve_pypfopt_func_call(HRPOpt, **kwargs)
        if optimizer.lower() == "cla":
            from pypfopt.cla import CLA

            return resolve_pypfopt_func_call(CLA, **kwargs)
        import pypfopt

        if hasattr(pypfopt, optimizer):
            return resolve_pypfopt_func_call(getattr(pypfopt, optimizer), **kwargs)
        raise NotImplementedError(f"Optimizer '{optimizer}' is not supported")
    if isinstance(optimizer, type) and issubclass(optimizer, BaseOptimizer):
        return resolve_pypfopt_func_call(optimizer, **kwargs)
    if isinstance(optimizer, BaseOptimizer):
        return optimizer
    raise NotImplementedError(f"Optimizer {optimizer} is not supported")


def pypfopt_optimize(
    target: tp.Optional[tp.Union[tp.Callable, str]] = None,
    target_is_convex: tp.Optional[bool] = None,
    weights_sum_to_one: tp.Optional[bool] = None,
    target_constraints: tp.Optional[tp.List[tp.Kwargs]] = None,
    target_solver: tp.Optional[str] = None,
    target_initial_guess: tp.Optional[tp.Array] = None,
    objectives: tp.Optional[tp.MaybeIterable[tp.Union[tp.Callable, str]]] = None,
    constraints: tp.Optional[tp.MaybeIterable[tp.Callable]] = None,
    sector_mapper: tp.Optional[dict] = None,
    sector_lower: tp.Optional[dict] = None,
    sector_upper: tp.Optional[dict] = None,
    discrete_allocation: tp.Optional[bool] = None,
    allocation_method: tp.Optional[str] = None,
    silence_warnings: tp.Optional[bool] = None,
    ignore_opt_errors: tp.Optional[bool] = None,
    ignore_errors: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Dict[str, float]:
    """Return allocation using PyPortfolioOpt.

    Resolves the optimizer using `resolve_pypfopt_optimizer` with the provided settings and
    additional parameters. Depending on the inputs, it may further resolve expected returns,
    covariance matrix, objectives, constraints, and sector constraints. It then applies the target
    optimization—adding the target as a convex or non-convex objective, extracts the weights, and,
    if requested, converts them to a discrete allocation using the specified method.

    To specify the optimizer, use `optimizer` (see `resolve_pypfopt_optimizer`).
    To specify the expected returns, use `expected_returns` (see `resolve_pypfopt_expected_returns`).
    To specify the covariance matrix, use `cov_matrix` (see `resolve_pypfopt_cov_matrix`).
    All other keyword arguments in `**kwargs` are forwarded to `resolve_pypfopt_func_call`.

    Each objective can be a function, an attribute of `pypfopt.objective_functions`,
    or an iterable of such. Each constraint can be a function or an iterable of such.
    The target can be an attribute of the optimizer or a stand-alone function.
    If `target_is_convex` is True, the target is added as a convex objective;
    otherwise, it is added as a non-convex objective. The keyword arguments
    `weights_sum_to_one` and those starting with `target` are passed to
    `pypfopt.base_optimizer.BaseConvexOptimizer.convex_objective` and
    `pypfopt.base_optimizer.BaseConvexOptimizer.nonconvex_objective` respectively.

    Set `ignore_opt_errors` to True to bypass errors specific to target optimization.
    Set `ignore_errors` to True to bypass all errors, including those raised by user inputs.

    If `discrete_allocation` is True, the function resolves `pypfopt.discrete_allocation.DiscreteAllocation`
    and invokes the specified attribute `allocation_method` on the resulting allocation object.

    All functions in this process are resolved using `resolve_pypfopt_func_call`.

    Args:
        target (Optional[Union[Callable, str]]): Optimization target function or attribute.
        target_is_convex (Optional[bool]): Indicates whether the target should be treated as convex.
        weights_sum_to_one (Optional[bool]): Enforce that portfolio weights sum to one.
        target_constraints (Optional[List[Kwargs]]): Additional constraints passed to the target function.
        target_solver (Optional[str]): Solver to be used for target optimization.
        target_initial_guess (Optional[Array]): Initial guess for the target optimization.
        objectives (Optional[MaybeIterable[Union[Callable, str]]]): Objectives for the optimizer.

            Functions or string references to objectives in `pypfopt.objective_functions`.
        constraints (Optional[MaybeIterable[Callable]]): Constraints to be applied to the optimizer.
        sector_mapper (Optional[dict]): Mapping of assets to sectors for applying sector constraints.
        sector_lower (Optional[dict]): Lower bounds for sector allocations.
        sector_upper (Optional[dict]): Upper bounds for sector allocations.
        discrete_allocation (Optional[bool]): If True, perform discrete allocation using
            `pypfopt.discrete_allocation.DiscreteAllocation`.
        allocation_method (Optional[str]): Method name used to compute discrete allocation.
        silence_warnings (Optional[bool]): Flag to suppress warning messages.
        ignore_opt_errors (Optional[bool]): Ignore errors related to target optimization if True.
        ignore_errors (Optional[bool]): Whether to ignore errors and return an empty dictionary.
        **kwargs: Keyword arguments for PyPortfolioOpt functions through
            `resolve_pypfopt_func_kwargs` and `resolve_pypfopt_func_call`.

    Returns:
        Dict[str, float]: Dictionary mapping asset symbols to their allocated weights.

    !!! info
        For default settings, see `pypfopt` in `vectorbtpro._settings.pfopt`.

    Examples:
        Using mean historical returns, Ledoit-Wolf covariance matrix with constant variance,
        and efficient frontier:

        ```pycon
        >>> from vectorbtpro import *

        >>> data = vbt.YFData.pull(["MSFT", "AMZN", "KO", "MA"])
        ```

        [=100% "100%"]{: .candystripe .candystripe-animate }

        ```pycon
        >>> vbt.pypfopt_optimize(prices=data.get("Close"))
        {'MSFT': 0.13324, 'AMZN': 0.10016, 'KO': 0.03229, 'MA': 0.73431}
        ```

        EMA historical returns and sample covariance:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     expected_returns="ema_historical_return",
        ...     cov_matrix="sample_cov"
        ... )
        {'MSFT': 0.08984, 'AMZN': 0.0, 'KO': 0.91016, 'MA': 0.0}
        ```

        EMA historical returns, efficient Conditional Value at Risk, and other parameters automatically
        passed to their respective functions. Optimized towards lowest CVaR:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     expected_returns="ema_historical_return",
        ...     optimizer="efficient_cvar",
        ...     beta=0.9,
        ...     weight_bounds=(-1, 1),
        ...     target="min_cvar"
        ... )
        {'MSFT': 0.14779, 'AMZN': 0.07224, 'KO': 0.77552, 'MA': 0.00445}
        ```

        Adding custom objectives:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     objectives=["L2_reg"],
        ...     gamma=0.1,
        ...     target="min_volatility"
        ... )
        {'MSFT': 0.22228, 'AMZN': 0.15685, 'KO': 0.28712, 'MA': 0.33375}
        ```

        Adding custom constraints:

        ```pycon
        >>> vbt.pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     constraints=[lambda w: w[data.symbols.index("MSFT")] <= 0.1]
        ... )
        {'MSFT': 0.1, 'AMZN': 0.10676, 'KO': 0.04341, 'MA': 0.74982}
        ```

        Optimizing towards a custom convex objective (to add a non-convex objective,
        set `target_is_convex` to False):

        ```pycon
        >>> import cvxpy as cp

        >>> def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
        ...     log_sum = cp.sum(cp.log(w))
        ...     var = cp.quad_form(w, cov_matrix)
        ...     return var - k * log_sum

        >>> pypfopt_optimize(
        ...     prices=data.get("Close"),
        ...     target=logarithmic_barrier_objective
        ... )
        {'MSFT': 0.24595, 'AMZN': 0.23047, 'KO': 0.25862, 'MA': 0.26496}
        ```
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("pypfopt")
    from cvxpy.error import SolverError
    from pypfopt.exceptions import OptimizationError

    from vectorbtpro._settings import settings

    pypfopt_cfg = dict(settings["pfopt"]["pypfopt"])

    def _resolve_setting(k, v):
        setting = pypfopt_cfg.pop(k)
        if v is None:
            return setting
        return v

    target = _resolve_setting("target", target)
    target_is_convex = _resolve_setting("target_is_convex", target_is_convex)
    weights_sum_to_one = _resolve_setting("weights_sum_to_one", weights_sum_to_one)
    target_constraints = _resolve_setting("target_constraints", target_constraints)
    target_solver = _resolve_setting("target_solver", target_solver)
    target_initial_guess = _resolve_setting("target_initial_guess", target_initial_guess)
    objectives = _resolve_setting("objectives", objectives)
    constraints = _resolve_setting("constraints", constraints)
    sector_mapper = _resolve_setting("sector_mapper", sector_mapper)
    sector_lower = _resolve_setting("sector_lower", sector_lower)
    sector_upper = _resolve_setting("sector_upper", sector_upper)
    discrete_allocation = _resolve_setting("discrete_allocation", discrete_allocation)
    allocation_method = _resolve_setting("allocation_method", allocation_method)
    silence_warnings = _resolve_setting("silence_warnings", silence_warnings)
    ignore_opt_errors = _resolve_setting("ignore_opt_errors", ignore_opt_errors)
    ignore_errors = _resolve_setting("ignore_errors", ignore_errors)
    kwargs = merge_dicts(pypfopt_cfg, kwargs)

    if "cache" not in kwargs:
        kwargs["cache"] = {}
    if "used_arg_names" not in kwargs:
        kwargs["used_arg_names"] = set()

    try:
        with WarningsFiltered(entries="ignore" if silence_warnings else None):
            optimizer = kwargs["optimizer"] = resolve_pypfopt_optimizer(**kwargs)

            if objectives is not None:
                if not checks.is_iterable(objectives) or isinstance(objectives, str):
                    objectives = [objectives]
                for objective in objectives:
                    if isinstance(objective, str):
                        import pypfopt.objective_functions

                        objective = getattr(pypfopt.objective_functions, objective)
                    objective_kwargs = resolve_pypfopt_func_kwargs(objective, **kwargs)
                    optimizer.add_objective(objective, **objective_kwargs)
            if constraints is not None:
                if not checks.is_iterable(constraints):
                    constraints = [constraints]
                for constraint in constraints:
                    optimizer.add_constraint(constraint)
            if sector_mapper is not None:
                if sector_lower is None:
                    sector_lower = {}
                if sector_upper is None:
                    sector_upper = {}
                optimizer.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

            try:
                if isinstance(target, str):
                    resolve_pypfopt_func_call(getattr(optimizer, target), **kwargs)
                else:
                    if target_is_convex:
                        optimizer.convex_objective(
                            target,
                            weights_sum_to_one=weights_sum_to_one,
                            **resolve_pypfopt_func_kwargs(target, **kwargs),
                        )
                    else:
                        optimizer.nonconvex_objective(
                            target,
                            objective_args=tuple(
                                resolve_pypfopt_func_kwargs(target, **kwargs).values()
                            ),
                            weights_sum_to_one=weights_sum_to_one,
                            constraints=target_constraints,
                            solver=target_solver,
                            initial_guess=target_initial_guess,
                        )
            except (OptimizationError, SolverError, ValueError) as e:
                if isinstance(
                    e, ValueError
                ) and "expected return exceeding the risk-free rate" not in str(e):
                    raise e
                if ignore_opt_errors:
                    warn(str(e))
                    return {}
                raise e

            weights = kwargs["weights"] = resolve_pypfopt_func_call(
                optimizer.clean_weights, **kwargs
            )
            if discrete_allocation:
                from pypfopt.discrete_allocation import DiscreteAllocation

                allocator = resolve_pypfopt_func_call(DiscreteAllocation, **kwargs)
                return resolve_pypfopt_func_call(getattr(allocator, allocation_method), **kwargs)[0]

            passed_arg_names = set(kwargs.keys())
            passed_arg_names.remove("cache")
            passed_arg_names.remove("used_arg_names")
            passed_arg_names.remove("optimizer")
            passed_arg_names.remove("weights")
            unused_arg_names = passed_arg_names.difference(kwargs["used_arg_names"])
            if len(unused_arg_names) > 0:
                warn(f"Some arguments were not used: {unused_arg_names}")

            if not discrete_allocation:
                weights = {k: 1 if v >= 1 else v for k, v in weights.items()}

        return dict(weights)
    except Exception as e:
        if ignore_errors:
            return {}
        raise e


# ############# Riskfolio-Lib ############# #


def prepare_returns(
    returns: tp.AnyArray2d,
    nan_to_zero: bool = True,
    dropna_rows: bool = True,
    dropna_cols: bool = True,
    dropna_any: bool = True,
) -> tp.Frame:
    """Prepare and clean return data.

    Converts the input returns to a Pandas DataFrame and processes missing and infinite values
    based on specified flags.

    Args:
        returns (AnyArray2d): Two-dimensional array containing return data.

            Expected to be convertible to a Pandas DataFrame.
        nan_to_zero (bool): Replace NaN values with 0.
        dropna_rows (bool): Remove rows that have missing or, if NaN values are replaced, zero values.
        dropna_cols (bool): Remove columns that have missing or, if NaN values are replaced, zero values.
        dropna_any (bool): Drop rows where any entry is missing if NaN values are not replaced.

    Returns:
        Frame: Processed DataFrame of return values.
    """
    returns = to_pd_array(returns)
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a two-dimensional array")
    if returns.size == 0:
        return returns
    if nan_to_zero or dropna_rows or dropna_cols or dropna_any:
        returns = returns.replace([np.inf, -np.inf], np.nan)
    if nan_to_zero:
        returns = returns.fillna(0.0)
    if dropna_rows or dropna_cols:
        if nan_to_zero:
            valid_mask = returns != 0
        else:
            valid_mask = ~returns.isnull()
        if dropna_rows:
            if nan_to_zero or not dropna_any:
                returns = returns.loc[valid_mask.any(axis=1)]
                if returns.size == 0:
                    return returns
        if dropna_cols:
            returns = returns.loc[:, valid_mask.any(axis=0)]
            if returns.size == 0:
                return returns
    if not nan_to_zero and dropna_any:
        returns = returns.dropna()
    return returns


def resolve_riskfolio_func_kwargs(
    riskfolio_func: tp.Callable,
    unused_arg_names: tp.Optional[tp.Set[str]] = None,
    func_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Kwargs:
    """Select keyword arguments for a given riskfolio function.

    Extracts keyword arguments from the provided input that match the parameters of `riskfolio_func`.
    If additional function-specific keyword arguments are provided via `func_kwargs`, they are merged.

    Args:
        riskfolio_func (Callable): Riskfolio function to filter keyword arguments for.
        unused_arg_names (Optional[Set[str]]): Set of argument names to be excluded.

            Matching keys found in kwargs will be removed.
        func_kwargs (KwargsLike): Keyword arguments specific to the riskfolio function.
        **kwargs: Keyword arguments to be filtered.

    Returns:
        Kwargs: Dictionary containing the keyword arguments relevant to `riskfolio_func`.
    """
    func_arg_names = get_func_arg_names(riskfolio_func)
    matched_kwargs = dict()
    for k, v in kwargs.items():
        if k in func_arg_names:
            matched_kwargs[k] = v
            if unused_arg_names is not None:
                if k in unused_arg_names:
                    unused_arg_names.remove(k)
    if func_kwargs is not None:
        return merge_dicts(
            select_pfopt_func_kwargs(riskfolio_func, matched_kwargs),
            select_pfopt_func_kwargs(riskfolio_func, pfopt_func_dict(func_kwargs)),
        )
    return select_pfopt_func_kwargs(riskfolio_func, matched_kwargs)


def resolve_asset_classes(
    asset_classes: tp.Union[None, tp.Frame, tp.Sequence],
    columns: tp.Index,
    col_indices: tp.Optional[tp.Sequence[int]] = None,
) -> tp.Frame:
    """Resolve asset classes for Riskfolio-Lib.

    Converts various asset class specifications into a DataFrame accepted by Riskfolio-Lib.

    Supported formats:

    * None: Uses the bottom-most level of `columns` as assets.
    * Index: Each level in the index is treated as a different asset class set.
    * Nested dict: Each sub-dict represents a different asset class set.
    * Sequence of strings or ints: Matches items against level names in `columns`.
        If the columns have a single level or names are not found, the sequence is used
        directly as one asset class set named "Class".
    * Sequence of dicts: Each dictionary becomes a row in the resulting DataFrame.
    * DataFrame: First column contains assets and subsequent columns represent asset class sets
        (the target format accepted by Riskfolio-Lib).

    Args:
        asset_classes (Union[None, Frame, Sequence]): Asset class information in various supported formats.
        columns (Index): Index of columns from which asset classes are derived.
        col_indices (Optional[Sequence[int]]): Specific indices to select from the asset class data.

    Returns:
        Frame: DataFrame formatted for Riskfolio-Lib containing asset classes.

    !!! note
        If `asset_classes` is neither None nor a DataFrame, the bottom-most level in `columns`
        is renamed to "Assets" and becomes the first column of the new DataFrame.
    """
    if asset_classes is None:
        asset_classes = columns.to_frame().reset_index(drop=True).iloc[:, ::-1]
        asset_classes = asset_classes.rename(columns={asset_classes.columns[0]: "Assets"})
    if not isinstance(asset_classes, pd.DataFrame):
        if isinstance(asset_classes, dict):
            asset_classes = pd.DataFrame(asset_classes)
        elif isinstance(asset_classes, pd.Index):
            asset_classes = asset_classes.to_frame().reset_index(drop=True)
        elif checks.is_sequence(asset_classes) and isinstance(asset_classes[0], int):
            asset_classes = select_levels(columns, asset_classes).to_frame().reset_index(drop=True)
        elif checks.is_sequence(asset_classes) and isinstance(asset_classes[0], str):
            if isinstance(columns, pd.MultiIndex) and set(asset_classes) <= set(columns.names):
                asset_classes = (
                    select_levels(columns, asset_classes).to_frame().reset_index(drop=True)
                )
            else:
                asset_classes = (
                    pd.Index(asset_classes, name="Class").to_frame().reset_index(drop=True)
                )
        else:
            asset_classes = pd.DataFrame.from_records(asset_classes)
        if isinstance(columns, pd.MultiIndex):
            assets = columns.get_level_values(-1)
        else:
            assets = columns
        if col_indices is not None and len(col_indices) > 0:
            asset_classes = asset_classes.iloc[col_indices]
        asset_classes.insert(loc=0, column="Assets", value=assets)
    return asset_classes


def resolve_assets_constraints(constraints: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve asset constraints for Riskfolio-Lib.

    Converts the input constraints into a DataFrame matching the target format for asset constraints
    as required by Riskfolio-Lib. In addition to the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_constraints),
    a sequence of dictionaries is also accepted, with each dictionary representing a row in the DataFrame.
    Missing column names are automatically filled.

    Args:
        constraints (Union[Frame, Sequence]): Asset constraints provided as a DataFrame, dictionary, or
            list of dictionaries.

    Returns:
        Frame: DataFrame structured for asset constraints in Riskfolio-Lib.
    """
    if not isinstance(constraints, pd.DataFrame):
        if isinstance(constraints, dict):
            constraints = pd.DataFrame(constraints)
        else:
            constraints = pd.DataFrame.from_records(constraints)
        constraints.columns = constraints.columns.str.title()
        new_constraints = pd.DataFrame(
            columns=[
                "Disabled",
                "Type",
                "Set",
                "Position",
                "Sign",
                "Weight",
                "Type Relative",
                "Relative Set",
                "Relative",
                "Factor",
            ],
            dtype=object,
        )
        for c in new_constraints.columns:
            if c in constraints.columns:
                new_constraints[c] = constraints[c]
        new_constraints.fillna("", inplace=True)
        new_constraints["Disabled"].replace("", False, inplace=True)
        constraints = new_constraints
    return constraints


def resolve_factors_constraints(constraints: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve factors constraints for Riskfolio-Lib.

    Converts the input factor constraints into a DataFrame matching the target format for factors
    constraints as required by Riskfolio-Lib. In addition to the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_constraints),
    a sequence of dictionaries is also accepted, with each dictionary representing a row in the DataFrame.
    Missing column names are automatically filled.

    Args:
        constraints (Union[Frame, Sequence]): Factor constraints provided as a DataFrame, dictionary, or
            list of dictionaries.

    Returns:
        Frame: DataFrame structured for factor constraints in Riskfolio-Lib.
    """
    if not isinstance(constraints, pd.DataFrame):
        if isinstance(constraints, dict):
            constraints = pd.DataFrame(constraints)
        else:
            constraints = pd.DataFrame.from_records(constraints)
        constraints.columns = constraints.columns.str.title()
        new_constraints = pd.DataFrame(
            columns=[
                "Disabled",
                "Factor",
                "Sign",
                "Value",
                "Relative Factor",
            ],
            dtype=object,
        )
        for c in new_constraints.columns:
            if c in constraints.columns:
                new_constraints[c] = constraints[c]
        new_constraints.fillna("", inplace=True)
        new_constraints["Disabled"].replace("", False, inplace=True)
        constraints = new_constraints
    return constraints


def resolve_assets_views(views: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve asset views for Riskfolio-Lib.

    Converts the input asset views into a DataFrame that complies with the target format for asset views
    as required by Riskfolio-Lib. In addition to the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_views),
    a sequence of dictionaries is also accepted, with each dictionary representing a row in the DataFrame.
    Missing column names are automatically filled.

    Args:
        views (Union[Frame, Sequence]): Asset views provided as a DataFrame, dictionary, or list of dictionaries.

            Each dictionary is converted to a row in a new DataFrame. Missing columns are auto-filled.

    Returns:
        Frame: DataFrame structured for asset views in Riskfolio-Lib.
    """
    if not isinstance(views, pd.DataFrame):
        if isinstance(views, dict):
            views = pd.DataFrame(views)
        else:
            views = pd.DataFrame.from_records(views)
        views.columns = views.columns.str.title()
        new_views = pd.DataFrame(
            columns=[
                "Disabled",
                "Type",
                "Set",
                "Position",
                "Sign",
                "Return",
                "Type Relative",
                "Relative Set",
                "Relative",
            ],
            dtype=object,
        )
        for c in new_views.columns:
            if c in views.columns:
                new_views[c] = views[c]
        new_views.fillna("", inplace=True)
        new_views["Disabled"].replace("", False, inplace=True)
        views = new_views
    return views


def resolve_factors_views(views: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve factors views for Riskfolio-Lib.

    Apart from the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_views),
    the input can also be provided as a sequence of dictionaries.

    Args:
        views (Union[Frame, Sequence]): Factor views provided as a DataFrame, dictionary, or list of dictionaries.

            Each dictionary is converted to a row in a new DataFrame. Missing columns are auto-filled.

    Returns:
        Frame: Resolved DataFrame formatted for Riskfolio-Lib.
    """
    if not isinstance(views, pd.DataFrame):
        if isinstance(views, dict):
            views = pd.DataFrame(views)
        else:
            views = pd.DataFrame.from_records(views)
        views.columns = views.columns.str.title()
        new_views = pd.DataFrame(
            columns=[
                "Disabled",
                "Factor",
                "Sign",
                "Value",
                "Relative Factor",
            ],
            dtype=object,
        )
        for c in new_views.columns:
            if c in views.columns:
                new_views[c] = views[c]
        new_views.fillna("", inplace=True)
        new_views["Disabled"].replace("", False, inplace=True)
        views = new_views
    return views


def resolve_hrp_constraints(constraints: tp.Union[tp.Frame, tp.Sequence]) -> tp.Frame:
    """Resolve HRP constraints for Riskfolio-Lib.

    Apart from the [target format](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.hrp_constraints),
    the input can also be provided as a sequence of dictionaries.

    Args:
        constraints (Union[Frame, Sequence]): HRP constraints provided as a DataFrame, dictionary, or
            list of dictionaries.

            Each dictionary is converted to a row in a new DataFrame. Missing columns are auto-filled.

    Returns:
        Frame: Resolved DataFrame formatted for Riskfolio-Lib.
    """
    if not isinstance(constraints, pd.DataFrame):
        if isinstance(constraints, dict):
            constraints = pd.DataFrame(constraints)
        else:
            constraints = pd.DataFrame.from_records(constraints)
        constraints.columns = constraints.columns.str.title()
        new_constraints = pd.DataFrame(
            columns=[
                "Disabled",
                "Type",
                "Set",
                "Position",
                "Sign",
                "Weight",
            ],
            dtype=object,
        )
        for c in new_constraints.columns:
            if c in constraints.columns:
                new_constraints[c] = constraints[c]
        new_constraints.fillna("", inplace=True)
        new_constraints["Disabled"].replace("", False, inplace=True)
        constraints = new_constraints
    return constraints


def riskfolio_optimize(
    returns: tp.AnyArray2d,
    nan_to_zero: tp.Optional[bool] = None,
    dropna_rows: tp.Optional[bool] = None,
    dropna_cols: tp.Optional[bool] = None,
    dropna_any: tp.Optional[bool] = None,
    factors: tp.Optional[tp.AnyArray2d] = None,
    port: tp.Optional[tp.Union[RPortfolioT, RHCPortfolioT]] = None,
    port_cls: tp.Union[None, str, tp.Type] = None,
    opt_method: tp.Union[None, str, tp.Callable] = None,
    stats_methods: tp.Optional[tp.Sequence[str]] = None,
    model: tp.Optional[str] = None,
    asset_classes: tp.Union[None, tp.Frame, tp.Sequence] = None,
    constraints_method: tp.Optional[str] = None,
    constraints: tp.Union[None, tp.Frame, tp.Sequence] = None,
    views_method: tp.Optional[str] = None,
    views: tp.Union[None, tp.Frame, tp.Sequence] = None,
    solvers: tp.Optional[tp.Sequence[str]] = None,
    sol_params: tp.KwargsLike = None,
    freq: tp.Optional[tp.FrequencyLike] = None,
    year_freq: tp.Optional[tp.FrequencyLike] = None,
    pre_opt: tp.Optional[bool] = None,
    pre_opt_kwargs: tp.KwargsLike = None,
    pre_opt_as_w: tp.Optional[bool] = None,
    func_kwargs: tp.KwargsLike = None,
    silence_warnings: tp.Optional[bool] = None,
    return_port: tp.Optional[bool] = None,
    ignore_errors: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Union[
    tp.Dict[str, float], tp.Tuple[tp.Dict[str, float], tp.Union[RPortfolioT, RHCPortfolioT]]
]:
    """Get allocation using Riskfolio-Lib.

    Compute asset allocation weights using the Riskfolio-Lib library. This function resolves default
    settings, prepares asset returns, optionally pre-optimizes the portfolio, builds the portfolio,
    applies constraints and views, runs statistics, and finally performs the optimization.

    Args:
        returns (AnyArray2d): 2D array or DataFrame containing asset returns.
        nan_to_zero (Optional[bool]): Whether to convert NaN values to zero.
        dropna_rows (Optional[bool]): Whether to drop rows with all NaN/zero values.

            Applied only if `nan_to_zero` is True or `dropna_any` is False.
        dropna_cols (Optional[bool]): Whether to drop columns with all NaN/zero values.
        dropna_any (Optional[bool]): Whether to drop any NaN values.

            Applied only if `nan_to_zero` is False.
        factors (Optional[AnyArray2d]): 2D array or DataFrame containing factor values.
        port (Optional[Union[RPortfolio, RHCPortfolio]]): Already initialized portfolio.
        port_cls (Optional[Union[None, str, Type]]): Portfolio class.

            Supported values:

            * None: Uses `Portfolio`
            * "hc" or "hcportfolio" (case-insensitive): Uses `HCPortfolio`
            * Other string: Uses attribute of `riskfolio`
            * Class: Uses a custom class
        opt_method (Optional[Union[None, str, Callable]]): Optimization method.

            Supported values:

            * None or "optimization": Uses `port.optimization`
            * "wc" or "wc_optimization": Uses `port.wc_optimization`
            * "rp" or "rp_optimization": Uses `port.rp_optimization`
            * "rrp" or "rrp_optimization": Uses `port.rrp_optimization`
            * "owa" or "owa_optimization": Uses `port.owa_optimization`
            * String: Uses the corresponding attribute of `port`
            * Callable: Custom optimization function
        stats_methods (Optional[Sequence[str]]): Sequence of stats methods to call before optimization.

            If not provided, the sequence is automatically populated based on `opt_method` and `model`.
            Additionally, functions with names ending in "_stats" from `func_kwargs` may be added.
        model (Optional[str]): Model used to optimize the portfolio.
        asset_classes (Optional[Union[None, Frame, Sequence]]): Asset classes matrix.

            See `resolve_asset_classes` for possible formats.
        constraints_method (Optional[str]): Constraints method.

            Supported values:

            * "assets" or "assets_constraints": Uses [assets constraints](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_constraints)
            * "factors" or "factors_constraints": Uses [factors constraints](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_constraints)
            * "hrp" or "hrp_constraints": Uses [HRP constraints](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.hrp_constraints)

            If None and a `Portfolio` is used, factors constraints are applied if `factors_stats` is used;
            otherwise, assets constraints are applied. For `HCPortfolio`, HRP constraints are used.
        constraints (Union[None, Frame, Sequence]): Constraints matrix.

            See `resolve_assets_constraints` for assets constraints formats,
            `resolve_factors_constraints` for factors constraints formats, and
            `resolve_hrp_constraints` for HRP constraints formats.
        views_method (Optional[str]): Views method.

            Supported values:

            * "assets" or "assets_views": Uses [assets views](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.assets_views)
            * "factors" or "factors_views": Uses [factors views](https://riskfolio-lib.readthedocs.io/en/latest/constraints.html#ConstraintsFunctions.factors_views)

            If None, defaults to factors views if "blfactors_stats" is present; otherwise, assets views.
        views (Union[None, Frame, Sequence]): Views matrix.

            See `resolve_assets_views` for assets views formats and
            `resolve_factors_views` for factors views formats.
        solvers (Optional[Sequence[str]]): Solvers.
        sol_params (KwargsLike): Solver parameters.
        freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

            Required when using views. See `vectorbtpro.utils.datetime_.infer_index_freq`.
        year_freq (Optional[FrequencyLike]): Year frequency for annualization (e.g., "252 days", "auto").

            Required when using views.
        pre_opt (Optional[bool]): Whether to pre-optimize the portfolio using `pre_opt_kwargs`.
        pre_opt_kwargs (KwargsLike): Keyword arguments for pre-optimization.

            Passed to a recursive call of `riskfolio_optimize`.
        pre_opt_as_w (Optional[bool]): Whether to use the weights from pre-optimization in subsequent optimization.
        func_kwargs (KwargsLike): Keyword arguments for function calls.

            Wrapped with `pfopt_func_dict` and used to override or add parameters.
        silence_warnings (Optional[bool]): Flag to suppress warning messages.
        return_port (Optional[bool]): Whether to return the portfolio along with the allocation weights.
        ignore_errors (Optional[bool]): Whether to ignore errors and return an empty dictionary.
        **kwargs: Keyword arguments for Riskfolio-Lib functions through `resolve_riskfolio_func_kwargs`.

    Returns:
        Union[Dict[str, float], Tuple[Dict[str, float], Union[RPortfolio, RHCPortfolio]]]:
            Allocation weights as a dictionary mapping asset names to weights, or a tuple of the weights
            and the portfolio if `return_port` is True.

    !!! info
        For default settings, see `riskfolio` in `vectorbtpro._settings.pfopt`.

    Examples:
        Classic Mean Risk Optimization:

        ```pycon
        >>> from vectorbtpro import *

        >>> data = vbt.YFData.pull(["MSFT", "AMZN", "KO", "MA"])
        >>> returns = data.close.vbt.to_returns()
        ```

        [=100% "100%"]{: .candystripe .candystripe-animate }

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     method_mu='hist', method_cov='hist', d=0.94,  # assets_stats
        ...     model='Classic', rm='MV', obj='Sharpe', hist=True, rf=0, l=0  # optimization
        ... )
        {'MSFT': 0.26297126323056036,
         'AMZN': 0.13984467450137006,
         'KO': 0.35870315943426767,
         'MA': 0.238480902833802}
        ```

        The same by splitting arguments:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     func_kwargs=dict(
        ...         assets_stats=dict(method_mu='hist', method_cov='hist', d=0.94),
        ...         optimization=dict(model='Classic', rm='MV', obj='Sharpe', hist=True, rf=0, l=0)
        ...     )
        ... )
        {'MSFT': 0.26297126323056036,
         'AMZN': 0.13984467450137006,
         'KO': 0.35870315943426767,
         'MA': 0.238480902833802}
        ```

        Asset constraints:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     constraints=[
        ...         {
        ...             "Type": "Assets",
        ...             "Position": "MSFT",
        ...             "Sign": "<=",
        ...             "Weight": 0.01
        ...         }
        ...     ]
        ... )
        {'MSFT': 0.009999990814976588,
         'AMZN': 0.19788481506569947,
         'KO': 0.4553600308839969,
         'MA': 0.336755163235327}
        ```

        Asset class constraints:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     asset_classes=["C1", "C1", "C2", "C2"],
        ...     constraints=[
        ...         {
        ...             "Type": "Classes",
        ...             "Set": "Class",
        ...             "Position": "C1",
        ...             "Sign": "<=",
        ...             "Weight": 0.1
        ...         }
        ...     ]
        ... )
        {'MSFT': 0.03501297245802569,
         'AMZN': 0.06498702655063979,
         'KO': 0.4756624658301967,
         'MA': 0.4243375351611379}
        ```

        Hierarchical Risk Parity (HRP) Portfolio Optimization:

        ```pycon
        >>> vbt.riskfolio_optimize(
        ...     returns,
        ...     port_cls="HCPortfolio",
        ...     model='HRP',
        ...     codependence='pearson',
        ...     rm='MV',
        ...     rf=0,
        ...     linkage='single',
        ...     max_k=10,
        ...     leaf_order=True
        ... )
        {'MSFT': 0.19091632057853536,
         'AMZN': 0.11069893826556164,
         'KO': 0.28589872132122485,
         'MA': 0.41248601983467814}
        ```
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("riskfolio")
    import riskfolio as rp

    from vectorbtpro._settings import settings

    riskfolio_cfg = dict(settings["pfopt"]["riskfolio"])

    def _resolve_setting(k, v):
        setting = riskfolio_cfg.pop(k)
        if v is None:
            return setting
        return v

    nan_to_zero = _resolve_setting("nan_to_zero", nan_to_zero)
    dropna_rows = _resolve_setting("dropna_rows", dropna_rows)
    dropna_cols = _resolve_setting("dropna_cols", dropna_cols)
    dropna_any = _resolve_setting("dropna_any", dropna_any)
    factors = _resolve_setting("factors", factors)
    port = _resolve_setting("port", port)
    port_cls = _resolve_setting("port_cls", port_cls)
    opt_method = _resolve_setting("opt_method", opt_method)
    stats_methods = _resolve_setting("stats_methods", stats_methods)
    model = _resolve_setting("model", model)
    asset_classes = _resolve_setting("asset_classes", asset_classes)
    constraints_method = _resolve_setting("constraints_method", constraints_method)
    constraints = _resolve_setting("constraints", constraints)
    views_method = _resolve_setting("views_method", views_method)
    views = _resolve_setting("views", views)
    solvers = _resolve_setting("solvers", solvers)
    sol_params = _resolve_setting("sol_params", sol_params)
    freq = _resolve_setting("freq", freq)
    year_freq = _resolve_setting("year_freq", year_freq)
    pre_opt = _resolve_setting("pre_opt", pre_opt)
    pre_opt_kwargs = merge_dicts(riskfolio_cfg.pop("pre_opt_kwargs"), pre_opt_kwargs)
    pre_opt_as_w = _resolve_setting("pre_opt_as_w", pre_opt_as_w)
    func_kwargs = merge_dicts(riskfolio_cfg.pop("func_kwargs"), func_kwargs)
    silence_warnings = _resolve_setting("silence_warnings", silence_warnings)
    return_port = _resolve_setting("return_port", return_port)
    ignore_errors = _resolve_setting("ignore_errors", ignore_errors)
    kwargs = merge_dicts(riskfolio_cfg, kwargs)
    if pre_opt_kwargs is None:
        pre_opt_kwargs = {}
    if func_kwargs is None:
        func_kwargs = {}
    func_kwargs = pfopt_func_dict(func_kwargs)
    unused_arg_names = set(kwargs.keys())

    try:
        with WarningsFiltered(entries="ignore" if silence_warnings else None):
            # Prepare returns
            new_returns = prepare_returns(
                returns,
                nan_to_zero=nan_to_zero,
                dropna_rows=dropna_rows,
                dropna_cols=dropna_cols,
                dropna_any=dropna_any,
            )
            col_indices = [i for i, c in enumerate(returns.columns) if c in new_returns.columns]
            returns = new_returns
            if returns.size == 0:
                return {}

            # Pre-optimize
            if pre_opt:
                w, port = riskfolio_optimize(returns, port=port, return_port=True, **pre_opt_kwargs)
                if pre_opt_as_w:
                    w = pd.DataFrame.from_records([w]).T.rename(columns={0: "weights"})
                    kwargs["w"] = w
                    unused_arg_names.add("w")

            # Build portfolio
            if port_cls is None:
                port_cls = rp.Portfolio
            elif isinstance(port_cls, str) and port_cls.lower() in ("hc", "hcportfolio"):
                port_cls = rp.HCPortfolio
            elif isinstance(port_cls, str):
                port_cls = getattr(rp, port_cls)
            else:
                port_cls = port_cls
            matched_kwargs = resolve_riskfolio_func_kwargs(
                port_cls,
                unused_arg_names=unused_arg_names,
                func_kwargs=func_kwargs,
                **kwargs,
            )
            if port is None:
                port = port_cls(returns, **matched_kwargs)
            else:
                for k, v in matched_kwargs.items():
                    setattr(port, k, v)
            if solvers is not None:
                port.solvers = list(solvers)
            if sol_params is not None:
                port.sol_params = dict(sol_params)
            if factors is not None:
                factors = to_pd_array(factors).dropna()
                port.factors = factors

            # Resolve optimization and stats methods
            if opt_method is None:
                if len(func_kwargs) > 0:
                    for name_or_func in func_kwargs:
                        if isinstance(name_or_func, str):
                            if name_or_func.endswith("optimization"):
                                if opt_method is not None:
                                    raise ValueError(
                                        "Function keyword arguments list multiple optimization methods"
                                    )
                                opt_method = name_or_func
            if opt_method is None:
                opt_method = "optimization"
            if stats_methods is None:
                if len(func_kwargs) > 0:
                    for name_or_func in func_kwargs:
                        if isinstance(name_or_func, str):
                            if name_or_func.endswith("_stats"):
                                if stats_methods is None:
                                    stats_methods = []
                                stats_methods.append(name_or_func)
            if isinstance(port, rp.Portfolio):
                if isinstance(opt_method, str) and opt_method.lower() == "optimization":
                    opt_func = port.optimization
                    if model is None:
                        opt_func_kwargs = select_pfopt_func_kwargs(opt_func, func_kwargs)
                        model = opt_func_kwargs.get("model", "Classic")
                    if model.lower() == "classic":
                        model = "Classic"
                        if stats_methods is None:
                            stats_methods = ["assets_stats"]
                    elif model.lower() == "fm":
                        model = "FM"
                        if stats_methods is None:
                            stats_methods = ["assets_stats", "factors_stats"]
                    elif model.lower() == "bl":
                        model = "BL"
                        if stats_methods is None:
                            stats_methods = ["assets_stats", "blacklitterman_stats"]
                    elif model.lower() in ("bl_fm", "blfm"):
                        model = "BL_FM"
                        if stats_methods is None:
                            stats_methods = ["assets_stats", "factors_stats", "blfactors_stats"]
                elif isinstance(opt_method, str) and opt_method.lower() in (
                    "wc",
                    "wc_optimization",
                ):
                    opt_func = port.wc_optimization
                    if stats_methods is None:
                        stats_methods = ["assets_stats", "wc_stats"]
                elif isinstance(opt_method, str) and opt_method.lower() in (
                    "rp",
                    "rp_optimization",
                ):
                    opt_func = port.rp_optimization
                    if model is None:
                        opt_func_kwargs = select_pfopt_func_kwargs(opt_func, func_kwargs)
                        model = opt_func_kwargs.get("model", "Classic")
                    if model.lower() == "classic":
                        model = "Classic"
                        if stats_methods is None:
                            stats_methods = ["assets_stats"]
                    elif model.lower() == "fm":
                        model = "FM"
                        if stats_methods is None:
                            stats_methods = ["assets_stats", "factors_stats"]
                elif isinstance(opt_method, str) and opt_method.lower() in (
                    "rrp",
                    "rrp_optimization",
                ):
                    opt_func = port.rrp_optimization
                    if model is None:
                        opt_func_kwargs = select_pfopt_func_kwargs(opt_func, func_kwargs)
                        model = opt_func_kwargs.get("model", "Classic")
                    if model.lower() == "classic":
                        model = "Classic"
                        if stats_methods is None:
                            stats_methods = ["assets_stats"]
                    elif model.lower() == "fm":
                        model = "FM"
                        if stats_methods is None:
                            stats_methods = ["assets_stats", "factors_stats"]
                elif isinstance(opt_method, str) and opt_method.lower() in (
                    "owa",
                    "owa_optimization",
                ):
                    opt_func = port.owa_optimization
                    if stats_methods is None:
                        stats_methods = ["assets_stats"]
                elif isinstance(opt_method, str):
                    opt_func = getattr(port, opt_method)
                else:
                    opt_func = opt_method
            else:
                if isinstance(opt_method, str):
                    opt_func = getattr(port, opt_method)
                else:
                    opt_func = opt_method
            if model is not None:
                kwargs["model"] = model
                unused_arg_names.add("model")
            if stats_methods is None:
                stats_methods = []

            # Apply constraints
            if constraints is not None:
                if constraints_method is None:
                    if isinstance(port, rp.Portfolio):
                        if "factors_stats" in stats_methods:
                            constraints_method = "factors"
                        else:
                            constraints_method = "assets"
                    elif isinstance(port, rp.HCPortfolio):
                        constraints_method = "hrp"
                    else:
                        raise ValueError("Constraints method is required")
                if constraints_method.lower() in ("assets", "assets_constraints"):
                    asset_classes = resolve_asset_classes(
                        asset_classes, returns.columns, col_indices
                    )
                    kwargs["asset_classes"] = asset_classes
                    unused_arg_names.add("asset_classes")
                    constraints = resolve_assets_constraints(constraints)
                    kwargs["constraints"] = constraints
                    unused_arg_names.add("constraints")
                    matched_kwargs = resolve_riskfolio_func_kwargs(
                        rp.assets_constraints,
                        unused_arg_names=unused_arg_names,
                        func_kwargs=func_kwargs,
                        **kwargs,
                    )
                    port.ainequality, port.binequality = warn_stdout(rp.assets_constraints)(
                        **matched_kwargs
                    )
                elif constraints_method.lower() in ("factors", "factors_constraints"):
                    if "loadings" not in kwargs:
                        matched_kwargs = resolve_riskfolio_func_kwargs(
                            rp.loadings_matrix,
                            unused_arg_names=unused_arg_names,
                            func_kwargs=func_kwargs,
                            **kwargs,
                        )
                        if "X" not in matched_kwargs:
                            matched_kwargs["X"] = port.factors
                        if "Y" not in matched_kwargs:
                            matched_kwargs["Y"] = port.returns
                        loadings = warn_stdout(rp.loadings_matrix)(**matched_kwargs)
                        kwargs["loadings"] = loadings
                        unused_arg_names.add("loadings")
                    constraints = resolve_factors_constraints(constraints)
                    kwargs["constraints"] = constraints
                    unused_arg_names.add("constraints")

                    matched_kwargs = resolve_riskfolio_func_kwargs(
                        rp.factors_constraints,
                        unused_arg_names=unused_arg_names,
                        func_kwargs=func_kwargs,
                        **kwargs,
                    )
                    port.ainequality, port.binequality = warn_stdout(rp.factors_constraints)(
                        **matched_kwargs
                    )
                elif constraints_method.lower() in ("hrp", "hrp_constraints"):
                    asset_classes = resolve_asset_classes(
                        asset_classes, returns.columns, col_indices
                    )
                    kwargs["asset_classes"] = asset_classes
                    unused_arg_names.add("asset_classes")
                    constraints = resolve_hrp_constraints(constraints)
                    kwargs["constraints"] = constraints
                    unused_arg_names.add("constraints")
                    matched_kwargs = resolve_riskfolio_func_kwargs(
                        rp.hrp_constraints,
                        unused_arg_names=unused_arg_names,
                        func_kwargs=func_kwargs,
                        **kwargs,
                    )
                    port.w_max, port.w_min = warn_stdout(rp.hrp_constraints)(**matched_kwargs)
                else:
                    raise ValueError(f"Constraints method '{constraints_method}' is not supported")

            # Resolve views
            if views is not None:
                if views_method is None:
                    if "blfactors_stats" in stats_methods:
                        views_method = "factors"
                    else:
                        views_method = "assets"
                if views_method.lower() in ("assets", "assets_views"):
                    asset_classes = resolve_asset_classes(
                        asset_classes, returns.columns, col_indices
                    )
                    kwargs["asset_classes"] = asset_classes
                    unused_arg_names.add("asset_classes")
                    views = resolve_assets_views(views)
                    kwargs["views"] = views
                    unused_arg_names.add("views")
                    matched_kwargs = resolve_riskfolio_func_kwargs(
                        rp.assets_views,
                        unused_arg_names=unused_arg_names,
                        func_kwargs=func_kwargs,
                        **kwargs,
                    )
                    P, Q = warn_stdout(rp.assets_views)(**matched_kwargs)
                    ann_factor = ReturnsAccessor.get_ann_factor(year_freq=year_freq, freq=freq)
                    if ann_factor is not None:
                        Q /= ann_factor
                    else:
                        warn("Set frequency and year frequency to adjust expected returns")
                    kwargs["P"] = P
                    unused_arg_names.add("P")
                    kwargs["Q"] = Q
                    unused_arg_names.add("Q")
                elif views_method.lower() in ("factors", "factors_views"):
                    if "loadings" not in kwargs:
                        matched_kwargs = resolve_riskfolio_func_kwargs(
                            rp.loadings_matrix,
                            unused_arg_names=unused_arg_names,
                            func_kwargs=func_kwargs,
                            **kwargs,
                        )
                        if "X" not in matched_kwargs:
                            matched_kwargs["X"] = port.factors
                        if "Y" not in matched_kwargs:
                            matched_kwargs["Y"] = port.returns
                        loadings = warn_stdout(rp.loadings_matrix)(**matched_kwargs)
                        kwargs["loadings"] = loadings
                        unused_arg_names.add("loadings")
                    if "B" not in kwargs:
                        kwargs["B"] = kwargs["loadings"]
                        unused_arg_names.add("B")
                    views = resolve_factors_views(views)
                    kwargs["views"] = views
                    unused_arg_names.add("views")
                    matched_kwargs = resolve_riskfolio_func_kwargs(
                        rp.factors_views,
                        unused_arg_names=unused_arg_names,
                        func_kwargs=func_kwargs,
                        **kwargs,
                    )
                    P_f, Q_f = warn_stdout(rp.factors_views)(**matched_kwargs)
                    ann_factor = ReturnsAccessor.get_ann_factor(year_freq=year_freq, freq=freq)
                    if ann_factor is not None:
                        Q_f /= ann_factor
                    else:
                        warn("Set frequency and year frequency to adjust expected returns")
                    kwargs["P_f"] = P_f
                    unused_arg_names.add("P_f")
                    kwargs["Q_f"] = Q_f
                    unused_arg_names.add("Q_f")
                else:
                    raise ValueError(f"Views method '{constraints_method}' is not supported")

            # Run stats
            for stats_method in stats_methods:
                stats_func = getattr(port, stats_method)
                matched_kwargs = resolve_riskfolio_func_kwargs(
                    stats_func,
                    unused_arg_names=unused_arg_names,
                    func_kwargs=func_kwargs,
                    **kwargs,
                )
                warn_stdout(stats_func)(**matched_kwargs)

            # Run optimization
            matched_kwargs = resolve_riskfolio_func_kwargs(
                opt_func,
                unused_arg_names=unused_arg_names,
                func_kwargs=func_kwargs,
                **kwargs,
            )
            weights = warn_stdout(opt_func)(**matched_kwargs)

            # Post-process weights
            if len(unused_arg_names) > 0:
                warn(f"Some arguments were not used: {unused_arg_names}")
            if weights is None:
                weights = {}
            if isinstance(weights, pd.DataFrame):
                if "weights" not in weights.columns:
                    raise ValueError("Weights column wasn't returned")
                weights = weights["weights"]
        if return_port:
            return dict(weights), port
        return dict(weights)
    except Exception as e:
        if ignore_errors:
            return {}
        raise e


# ############# PortfolioOptimizer ############# #


PortfolioOptimizerT = tp.TypeVar("PortfolioOptimizerT", bound="PortfolioOptimizer")


class PortfolioOptimizer(Analyzable):
    """Class that exposes methods for generating allocations.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        alloc_records (Union[AllocRanges, AllocPoints]): Allocation records associated with the portfolio.
        allocations (Array2d): 2D array containing allocation values.
        **kwargs: Keyword arguments for `vectorbtpro.generic.analyzable.Analyzable`.
    """

    def __init__(
        self,
        wrapper: ArrayWrapper,
        alloc_records: tp.Union[AllocRanges, AllocPoints],
        allocations: tp.Array2d,
        **kwargs,
    ) -> None:
        Analyzable.__init__(
            self,
            wrapper,
            alloc_records=alloc_records,
            allocations=allocations,
            **kwargs,
        )

        self._alloc_records = alloc_records
        self._allocations = allocations

        # Only slices of rows can be selected
        self._range_only_select = True

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[PortfolioOptimizerT],
        *objs: tp.MaybeSequence[PortfolioOptimizerT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Stack multiple `PortfolioOptimizer` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        Args:
            *objs (MaybeSequence[PortfolioOptimizer]): (Additional) `PortfolioOptimizer` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `PortfolioOptimizer` through
                `PortfolioOptimizer.resolve_row_stack_kwargs` and `PortfolioOptimizer.resolve_stack_kwargs`.

        Returns:
            PortfolioOptimizer: New `PortfolioOptimizer` instance with row-stacked attributes.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PortfolioOptimizer):
                raise TypeError(
                    "Each object to be merged must be an instance of PortfolioOptimizer"
                )
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs],
                stack_columns=False,
                **wrapper_kwargs,
            )
        if "alloc_records" not in kwargs:
            alloc_records_type = None
            for obj in objs:
                if alloc_records_type is None:
                    alloc_records_type = type(obj.alloc_records)
                elif not isinstance(obj.alloc_records, alloc_records_type):
                    raise TypeError(
                        "Objects to be merged must have the same type for alloc_records"
                    )
            kwargs["alloc_records"] = alloc_records_type.row_stack(
                *[obj.alloc_records for obj in objs],
                wrapper_kwargs=wrapper_kwargs,
            )
        if "allocations" not in kwargs:
            record_indices = type(kwargs["alloc_records"]).get_row_stack_record_indices(
                *[obj.alloc_records for obj in objs],
                wrapper=kwargs["alloc_records"].wrapper,
            )
            kwargs["allocations"] = row_stack_arrays([obj._allocations for obj in objs])[
                record_indices
            ]

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[PortfolioOptimizerT],
        *objs: tp.MaybeSequence[PortfolioOptimizerT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Stack multiple `PortfolioOptimizer` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        Args:
            *objs (MaybeSequence[PortfolioOptimizer]): (Additional) `PortfolioOptimizer` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `PortfolioOptimizer` through
                `PortfolioOptimizer.resolve_column_stack_kwargs` and `PortfolioOptimizer.resolve_stack_kwargs`.

        Returns:
            PortfolioOptimizer: New `PortfolioOptimizer` instance with column-stacked attributes.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PortfolioOptimizer):
                raise TypeError(
                    "Each object to be merged must be an instance of PortfolioOptimizer"
                )
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                union_index=False,
                **wrapper_kwargs,
            )
        if "alloc_records" not in kwargs:
            alloc_records_type = None
            for obj in objs:
                if alloc_records_type is None:
                    alloc_records_type = type(obj.alloc_records)
                elif not isinstance(obj.alloc_records, alloc_records_type):
                    raise TypeError(
                        "Objects to be merged must have the same type for alloc_records"
                    )
            kwargs["alloc_records"] = alloc_records_type.column_stack(
                *[obj.alloc_records for obj in objs],
                wrapper_kwargs=wrapper_kwargs,
            )
        if "allocations" not in kwargs:
            record_indices = type(kwargs["alloc_records"]).get_column_stack_record_indices(
                *[obj.alloc_records for obj in objs],
                wrapper=kwargs["alloc_records"].wrapper,
            )
            kwargs["allocations"] = row_stack_arrays([obj._allocations for obj in objs])[
                record_indices
            ]

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    def indexing_func(
        self: PortfolioOptimizerT,
        *args,
        wrapper_meta: tp.DictLike = None,
        alloc_wrapper_meta: tp.DictLike = None,
        alloc_records_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Perform indexing on a `PortfolioOptimizer` instance.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.
            wrapper_meta (DictLike): Metadata from the indexing operation on the wrapper.
            alloc_wrapper_meta (DictLike): Metadata from the indexing operation on the allocation wrapper.
            alloc_records_meta (DictLike): Metadata from the indexing operation on the allocation records.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.

        Returns:
            PortfolioOptimizer: New `PortfolioOptimizer` instance with updated indexing.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        if alloc_records_meta is None:
            alloc_records_meta = self.alloc_records.indexing_func_meta(
                *args,
                wrapper_meta=alloc_wrapper_meta,
                **kwargs,
            )
        new_alloc_records = self.alloc_records.indexing_func(
            *args,
            records_meta=alloc_records_meta,
            **kwargs,
        )
        new_allocations = to_2d_array(self._allocations)[alloc_records_meta["new_indices"]]
        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            alloc_records=new_alloc_records,
            allocations=new_allocations,
        )

    def resample(self: PortfolioOptimizerT, *args, **kwargs) -> PortfolioOptimizerT:
        """Perform resampling on a `PortfolioOptimizer` instance.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample`.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample`.

        Returns:
            PortfolioOptimizer: New `PortfolioOptimizer` instance with resampled attributes.
        """
        new_wrapper = self.wrapper.resample(*args, **kwargs)
        new_alloc_records = self.alloc_records.resample(*args, **kwargs)
        return self.replace(
            wrapper=new_wrapper,
            alloc_records=new_alloc_records,
        )

    # ############# Class methods ############# #

    @classmethod
    def run_allocation_group(
        cls,
        wrapper: ArrayWrapper,
        group_configs: tp.List[dict],
        group_index: tp.Index,
        group_idx: int,
        pre_group_func: tp.Optional[tp.Callable] = None,
    ) -> tp.Tuple[tp.RecordArray, tp.Array2d]:
        """Run an allocation group.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            group_configs (List[dict]): List of configurations for allocation groups.
            group_index (Index): Index representing the allocation group.
            group_idx (int): Index specifying which configuration in `group_configs` to use.
            pre_group_func (Optional[Callable]): Function to preprocess and modify the group configuration.

        Returns:
            Tuple[RecordArray, Array2d]: Tuple containing allocation records and a 2D array of allocation values.

        See:
            * `vectorbtpro.portfolio.pfopt.nb.allocate_meta_nb` if `jitted_loop=True`
        """
        group_config = dict(group_configs[group_idx])
        if pre_group_func is not None:
            pre_group_func(group_config)

        allocate_func = group_config.pop("allocate_func")
        every = group_config.pop("every")
        normalize_every = group_config.pop("normalize_every")
        at_time = group_config.pop("at_time")
        start = group_config.pop("start")
        end = group_config.pop("end")
        exact_start = group_config.pop("exact_start")
        on = group_config.pop("on")
        add_delta = group_config.pop("add_delta")
        kind = group_config.pop("kind")
        indexer_method = group_config.pop("indexer_method")
        indexer_tolerance = group_config.pop("indexer_tolerance")
        skip_not_found = group_config.pop("skip_not_found")
        index_points = group_config.pop("index_points")
        rescale_to = group_config.pop("rescale_to")
        jitted_loop = group_config.pop("jitted_loop")
        jitted = group_config.pop("jitted")
        chunked = group_config.pop("chunked")
        template_context = group_config.pop("template_context")
        execute_kwargs = group_config.pop("execute_kwargs")
        args = group_config.pop("args")
        kwargs = group_config

        template_context = merge_dicts(
            dict(
                group_configs=group_configs,
                group_index=group_index,
                group_idx=group_idx,
                wrapper=wrapper,
                allocate_func=allocate_func,
                every=every,
                normalize_every=normalize_every,
                at_time=at_time,
                start=start,
                end=end,
                exact_start=exact_start,
                on=on,
                add_delta=add_delta,
                kind=kind,
                indexer_method=indexer_method,
                indexer_tolerance=indexer_tolerance,
                skip_not_found=skip_not_found,
                index_points=index_points,
                rescale_to=rescale_to,
                jitted_loop=jitted_loop,
                jitted=jitted,
                chunked=chunked,
                execute_kwargs=execute_kwargs,
                args=args,
                kwargs=kwargs,
            ),
            template_context,
        )

        if index_points is None:
            get_index_points_kwargs = substitute_templates(
                dict(
                    every=every,
                    normalize_every=normalize_every,
                    at_time=at_time,
                    start=start,
                    end=end,
                    exact_start=exact_start,
                    on=on,
                    add_delta=add_delta,
                    kind=kind,
                    indexer_method=indexer_method,
                    indexer_tolerance=indexer_tolerance,
                    skip_not_found=skip_not_found,
                ),
                template_context,
                eval_id="get_index_points_defaults",
                strict=True,
            )
            index_points = wrapper.get_index_points(**get_index_points_kwargs)
            template_context = merge_dicts(
                template_context,
                get_index_points_kwargs,
                dict(index_points=index_points),
            )
        else:
            index_points = substitute_templates(
                index_points,
                template_context,
                eval_id="index_points",
                strict=True,
            )
            index_points = to_1d_array(index_points)
            template_context = merge_dicts(template_context, dict(index_points=index_points))

        if jitted_loop:
            allocate_func = substitute_templates(
                allocate_func,
                template_context,
                eval_id="allocate_func",
                strict=True,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            kwargs = substitute_templates(kwargs, template_context, eval_id="kwargs")
            func = jit_reg.resolve_option(nb.allocate_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            _allocations = func(len(wrapper.columns), index_points, allocate_func, *args, **kwargs)
        else:
            tasks = []
            keys = []
            for i in range(len(index_points)):
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_point=index_points[i],
                    ),
                    template_context,
                )
                _allocate_func = substitute_templates(
                    allocate_func,
                    _template_context,
                    eval_id="allocate_func",
                    strict=True,
                )
                _args = substitute_templates(args, _template_context, eval_id="args")
                _kwargs = substitute_templates(kwargs, _template_context, eval_id="kwargs")
                tasks.append(Task(_allocate_func, *_args, **_kwargs))
                if isinstance(wrapper.index, pd.DatetimeIndex):
                    keys.append(
                        dt.readable_datetime(wrapper.index[index_points[i]], freq=wrapper.freq)
                    )
                else:
                    keys.append(str(wrapper.index[index_points[i]]))
            results = execute(tasks, keys=keys, **execute_kwargs)
            _allocations = pd.DataFrame(results, columns=wrapper.columns)
            if isinstance(_allocations.columns, pd.RangeIndex):
                _allocations = _allocations.values
            else:
                _allocations = _allocations[list(wrapper.columns)].values
            if rescale_to is not None:
                _allocations = nb.rescale_allocations_nb(_allocations, rescale_to)

        return nb.prepare_alloc_points_nb(index_points, _allocations, group_idx)

    @classmethod
    def from_allocate_func(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        allocate_func: tp.Callable,
        *args,
        every: tp.Union[None, tp.FrequencyLike, Param] = point_idxr_defaults["every"],
        normalize_every: tp.Union[bool, Param] = point_idxr_defaults["normalize_every"],
        at_time: tp.Union[None, tp.TimeLike, Param] = point_idxr_defaults["at_time"],
        start: tp.Union[None, int, tp.DatetimeLike, Param] = point_idxr_defaults["start"],
        end: tp.Union[None, int, tp.DatetimeLike, Param] = point_idxr_defaults["end"],
        exact_start: tp.Union[bool, Param] = point_idxr_defaults["exact_start"],
        on: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, Param] = point_idxr_defaults["on"],
        add_delta: tp.Union[None, tp.FrequencyLike, Param] = point_idxr_defaults["add_delta"],
        kind: tp.Union[None, str, Param] = point_idxr_defaults["kind"],
        indexer_method: tp.Union[None, str, Param] = point_idxr_defaults["indexer_method"],
        indexer_tolerance: tp.Union[None, str, Param] = point_idxr_defaults["indexer_tolerance"],
        skip_not_found: tp.Union[bool, Param] = point_idxr_defaults["skip_not_found"],
        index_points: tp.Union[None, tp.MaybeSequence[int], Param] = None,
        rescale_to: tp.Union[None, tp.Tuple[float, float], Param] = None,
        parameterizer: tp.Optional[tp.MaybeType[Parameterizer]] = None,
        param_search_kwargs: tp.KwargsLike = None,
        name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
        group_configs: tp.Union[
            None, tp.Dict[tp.Hashable, tp.Kwargs], tp.Sequence[tp.Kwargs]
        ] = None,
        pre_group_func: tp.Optional[tp.Callable] = None,
        jitted_loop: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        template_context: tp.KwargsLike = None,
        group_execute_kwargs: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        random_subset: tp.Optional[int] = None,
        clean_index_kwargs: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations from an allocation function.

        Generate allocation points based on dates and apply the allocation function at each point.
        This method uses `vectorbtpro.base.wrapping.ArrayWrapper.get_index_points` to determine allocation
        indices, making each point available as `index_point` in the template context. It creates allocation
        records of type `vectorbtpro.portfolio.pfopt.records.AllocPoints`, unlike
        `PortfolioOptimizer.from_optimize_func`.

        Templates can use the variables:

        * `i`: Allocation step.
        * `index_point`: Allocation index.

        If `jitted_loop` is True, see `vectorbtpro.portfolio.pfopt.nb.allocate_meta_nb`.

        Args:
            cls (Type[PortfolioOptimizer]): Class to instantiate.
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            allocate_func (Callable): Function that computes allocations.
            *args: Positional arguments for `allocate_func`.
            every (Union[None, FrequencyLike, Param]): See `vectorbtpro.base.indexing.PointIdxr.every`.
            normalize_every (Union[bool, Param]): See `vectorbtpro.base.indexing.PointIdxr.normalize_every`.
            at_time (Union[None, TimeLike, Param]): See `vectorbtpro.base.indexing.PointIdxr.at_time`.
            start (Union[None, int, DatetimeLike, Param]): See `vectorbtpro.base.indexing.PointIdxr.start`.
            end (Union[None, int, DatetimeLike, Param]): See `vectorbtpro.base.indexing.PointIdxr.end`.
            exact_start (Union[bool, Param]): See `vectorbtpro.base.indexing.PointIdxr.exact_start`.
            on (Union[None, int, DatetimeLike, IndexLike, Param]): See `vectorbtpro.base.indexing.PointIdxr.on`.
            add_delta (Union[None, FrequencyLike, Param]): See `vectorbtpro.base.indexing.PointIdxr.add_delta`.
            kind (Union[None, str, Param]): See `vectorbtpro.base.indexing.PointIdxr.kind`.
            indexer_method (Union[None, str, Param]): See `vectorbtpro.base.indexing.PointIdxr.indexer_method`.
            indexer_tolerance (Union[None, str, Param]): See `vectorbtpro.base.indexing.PointIdxr.indexer_tolerance`.
            skip_not_found (Union[bool, Param]): See `vectorbtpro.base.indexing.PointIdxr.skip_not_found`.
            index_points (Union[None, MaybeSequence[int], Param]): Manually specified index points.
            rescale_to (Union[None, Tuple[float, float], Param]): Tuple specifying the rescaling range for allocations.
            parameterizer (Optional[MaybeType[Parameterizer]]): Parameterizer class or instance for handling parameters.

                Defaults to `vectorbtpro.utils.params.Parameterizer`.
            param_search_kwargs (KwargsLike): Keyword arguments for parameter search.

                See `vectorbtpro.utils.params.Parameterizer.find_params_in_obj`.
            name_tuple_to_str (Union[None, bool, Callable]): Flag or function to convert name tuples to
                strings for the parameter index.
            group_configs (Union[None, Dict[Hashable, Kwargs], Sequence[Kwargs]]): Group configuration(s) for allocation.
            pre_group_func (Optional[Callable]): Function to preprocess and modify the group configuration.
            jitted_loop (bool): Flag indicating whether to use a JIT-compiled loop over allocation groups.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            template_context (KwargsLike): Additional context for template substitution.
            group_execute_kwargs (KwargsLike): Keyword arguments for the execution handler of allocation groups.

                See `vectorbtpro.utils.execution.execute`.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            random_subset (Optional[int]): Select a random subset of parameter combinations.

                Set the seed for reproducibility.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `allocate_func`.

        Returns:
            PortfolioOptimizer: Instance of portfolio optimizer containing allocation points and allocations.

        !!! info
            For default settings, see `vectorbtpro._settings.params`.

        Examples:
            Allocate uniformly every day:

            ```pycon
            >>> from vectorbtpro import *

            >>> data = vbt.YFData.pull(
            ...     ["MSFT", "AMZN", "AAPL"],
            ...     start="2010-01-01",
            ...     end="2020-01-01"
            ... )
            >>> close = data.get("Close")

            >>> def uniform_allocate_func(n_cols):
            ...     return np.full(n_cols, 1 / n_cols)

            >>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     uniform_allocate_func,
            ...     close.shape[1]
            ... )
            >>> pfo.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-01-04 00:00:00-05:00  0.333333  0.333333  0.333333
            2010-01-05 00:00:00-05:00  0.333333  0.333333  0.333333
            2010-01-06 00:00:00-05:00  0.333333  0.333333  0.333333
            2010-01-07 00:00:00-05:00  0.333333  0.333333  0.333333
            2010-01-08 00:00:00-05:00  0.333333  0.333333  0.333333
            ...                             ...       ...       ...
            2019-12-24 00:00:00-05:00  0.333333  0.333333  0.333333
            2019-12-26 00:00:00-05:00  0.333333  0.333333  0.333333
            2019-12-27 00:00:00-05:00  0.333333  0.333333  0.333333
            2019-12-30 00:00:00-05:00  0.333333  0.333333  0.333333
            2019-12-31 00:00:00-05:00  0.333333  0.333333  0.333333

            [2516 rows x 3 columns]
            ```

            Allocate randomly every first date of the year:

            ```pycon
            >>> def random_allocate_func(n_cols):
            ...     weights = np.random.uniform(size=n_cols)
            ...     return weights / weights.sum()

            >>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func,
            ...     close.shape[1],
            ...     every="AS-JAN"
            ... )
            >>> pfo.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2011-01-03 00:00:00+00:00  0.160335  0.122434  0.717231
            2012-01-03 00:00:00+00:00  0.071386  0.469564  0.459051
            2013-01-02 00:00:00+00:00  0.125853  0.168480  0.705668
            2014-01-02 00:00:00+00:00  0.391565  0.169205  0.439231
            2015-01-02 00:00:00+00:00  0.115075  0.602844  0.282081
            2016-01-04 00:00:00+00:00  0.244070  0.046547  0.709383
            2017-01-03 00:00:00+00:00  0.316065  0.335000  0.348935
            2018-01-02 00:00:00+00:00  0.422142  0.252154  0.325704
            2019-01-02 00:00:00+00:00  0.368748  0.195147  0.436106
            ```

            Specify index points manually:

            ```pycon
            >>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func,
            ...     close.shape[1],
            ...     index_points=[0, 30, 60]
            ... )
            >>> pfo.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-01-04 00:00:00+00:00  0.257878  0.308287  0.433835
            2010-02-17 00:00:00+00:00  0.090927  0.471980  0.437094
            2010-03-31 00:00:00+00:00  0.395855  0.148516  0.455629
            ```

            Specify allocations manually:

            ```pycon
            >>> def manual_allocate_func(weights):
            ...     return weights

            >>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     manual_allocate_func,
            ...     vbt.RepEval("weights[i]", context=dict(weights=[
            ...         [1, 0, 0],
            ...         [0, 1, 0],
            ...         [0, 0, 1]
            ...     ])),
            ...     index_points=[0, 30, 60]
            ... )
            >>> pfo.allocations
            symbol                     MSFT  AMZN  AAPL
            Date
            2010-01-04 00:00:00+00:00     1     0     0
            2010-02-17 00:00:00+00:00     0     1     0
            2010-03-31 00:00:00+00:00     0     0     1
            ```

            Use Numba-compiled loop:

            ```pycon
            >>> @njit
            ... def random_allocate_func_nb(i, idx, n_cols):
            ...     weights = np.random.uniform(0, 1, n_cols)
            ...     return weights / weights.sum()

            >>> pfo = vbt.PortfolioOptimizer.from_allocate_func(
            ...     close.vbt.wrapper,
            ...     random_allocate_func_nb,
            ...     close.shape[1],
            ...     index_points=[0, 30, 60],
            ...     jitted_loop=True
            ... )
            >>> pfo.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-01-04 00:00:00+00:00  0.231925  0.351085  0.416990
            2010-02-17 00:00:00+00:00  0.163050  0.070292  0.766658
            2010-03-31 00:00:00+00:00  0.497465  0.500215  0.002319
            ```

            !!! hint
                There is little benefit to using a Numba-compiled loop unless rebalancing thousands
                of times. Typically, a regular Python loop with a Numba-compiled allocation function suffices.
        """
        from vectorbtpro._settings import settings

        params_cfg = settings["params"]

        if parameterizer is None:
            parameterizer = params_cfg["parameterizer"]
        if parameterizer is None:
            parameterizer = Parameterizer
        param_search_kwargs = merge_dicts(params_cfg["param_search_kwargs"], param_search_kwargs)
        if group_execute_kwargs is None:
            group_execute_kwargs = {}
        if execute_kwargs is None:
            execute_kwargs = {}
        if clean_index_kwargs is None:
            clean_index_kwargs = {}

        # Prepare group config names
        gc_names = []
        gc_names_none = True
        n_configs = 0
        if group_configs is not None:
            if isinstance(group_configs, dict):
                new_group_configs = []
                for k, v in group_configs.items():
                    v = dict(v)
                    v["_name"] = k
                    new_group_configs.append(v)
                group_configs = new_group_configs
            else:
                group_configs = list(group_configs)
            for i, group_config in enumerate(group_configs):
                group_config = dict(group_config)
                if "args" in group_config:
                    for k, arg in enumerate(group_config.pop("args")):
                        group_config[f"args_{k}"] = arg
                if "kwargs" in group_config:
                    for k, v in enumerate(group_config.pop("kwargs")):
                        group_config[k] = v
                if "_name" in group_config and group_config["_name"] is not None:
                    gc_names.append(group_config.pop("_name"))
                    gc_names_none = False
                else:
                    gc_names.append(n_configs)
                group_configs[i] = group_config
                n_configs += 1
        else:
            group_configs = []

        # Combine parameters
        paramable_kwargs = {
            "every": every,
            "normalize_every": normalize_every,
            "at_time": at_time,
            "start": start,
            "end": end,
            "exact_start": exact_start,
            "on": on,
            "add_delta": add_delta,
            "kind": kind,
            "indexer_method": indexer_method,
            "indexer_tolerance": indexer_tolerance,
            "skip_not_found": skip_not_found,
            "index_points": index_points,
            "rescale_to": rescale_to,
            **{f"args_{i}": args[i] for i in range(len(args))},
            **kwargs,
        }
        param_dct = parameterizer.find_params_in_obj(paramable_kwargs, **param_search_kwargs)
        param_columns = None
        if len(param_dct) > 0:
            param_product, param_columns = combine_params(
                param_dct,
                random_subset=random_subset,
                clean_index_kwargs=clean_index_kwargs,
                name_tuple_to_str=name_tuple_to_str,
            )
            if param_columns is None:
                n_param_configs = len(param_product[list(param_product.keys())[0]])
                param_columns = pd.RangeIndex(stop=n_param_configs, name="param_config")
            product_group_configs = parameterizer.param_product_to_objs(
                paramable_kwargs, param_product
            )
            if len(group_configs) == 0:
                group_configs = product_group_configs
            else:
                new_group_configs = []
                for i in range(len(product_group_configs)):
                    for group_config in group_configs:
                        new_group_config = merge_dicts(product_group_configs[i], group_config)
                        new_group_configs.append(new_group_config)
                group_configs = new_group_configs

        # Build group index
        n_config_params = len(gc_names)
        if param_columns is not None:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = param_columns
            else:
                group_index = combine_indexes(
                    (
                        param_columns,
                        pd.Index(gc_names, name="group_config"),
                    ),
                    **clean_index_kwargs,
                )
        else:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = pd.Index(["group"], name="group")
            else:
                group_index = pd.Index(gc_names, name="group_config")

        # Create group config from arguments if empty
        if len(group_configs) == 0:
            single_group = True
            group_configs.append(dict())
        else:
            single_group = False

        # Resolve each group
        groupable_kwargs = {
            "allocate_func": allocate_func,
            **paramable_kwargs,
            "jitted_loop": jitted_loop,
            "jitted": jitted,
            "chunked": chunked,
            "template_context": template_context,
            "execute_kwargs": execute_kwargs,
        }
        new_group_configs = []
        for group_config in group_configs:
            new_group_config = merge_dicts(groupable_kwargs, group_config)
            _args = ()
            while True:
                if f"args_{len(_args)}" in new_group_config:
                    _args += (new_group_config.pop(f"args_{len(_args)}"),)
                else:
                    break
            new_group_config["args"] = _args
            new_group_configs.append(new_group_config)
        group_configs = new_group_configs

        # Generate allocations
        tasks = []
        for group_idx, group_config in enumerate(group_configs):
            tasks.append(
                Task(
                    cls.run_allocation_group,
                    wrapper=wrapper,
                    group_configs=group_configs,
                    group_index=group_index,
                    group_idx=group_idx,
                    pre_group_func=pre_group_func,
                )
            )
        group_execute_kwargs = merge_dicts(
            dict(show_progress=False if single_group else None), group_execute_kwargs
        )
        results = execute(tasks, keys=group_index, **group_execute_kwargs)
        alloc_points, allocations = zip(*results)

        # Build column hierarchy
        new_columns = combine_indexes((group_index, wrapper.columns), **clean_index_kwargs)

        # Create instance
        wrapper_kwargs = merge_dicts(
            dict(
                index=wrapper.index,
                columns=new_columns,
                ndim=2,
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
                group_select=True,
                grouped_ndim=1 if single_group else 2,
                group_by=group_index.names if group_index.nlevels > 1 else group_index.name,
                allow_enable=False,
                allow_disable=True,
                allow_modify=False,
            ),
            wrapper_kwargs,
        )
        new_wrapper = ArrayWrapper(**wrapper_kwargs)
        alloc_points = AllocPoints(
            ArrayWrapper(
                index=wrapper.index,
                columns=new_wrapper.get_columns(),
                ndim=new_wrapper.get_ndim(),
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
            ),
            np.concatenate(alloc_points),
        )
        allocations = row_stack_arrays(allocations)
        return cls(new_wrapper, alloc_points, allocations)

    @classmethod
    def from_allocations(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        allocations: tp.ArrayLike,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Pick allocations from a flexible array.

        Uses `PortfolioOptimizer.from_allocate_func`.

        If `allocations` is a DataFrame, its index is used as labels. If it is a Series or dict,
        the allocation is applied uniformly across all indices. If the input is neither a DataFrame,
        Series, nor a NumPy array, an attempt is made to convert it to a NumPy array.

        If `allocations` is a NumPy array, the function uses
        `vectorbtpro.portfolio.pfopt.nb.pick_idx_allocate_func_nb` with a Numba-compiled loop.
        Otherwise, a regular Python function is used, where a single element is applied to all
        rows and one-dimensional arrays are broadcast across columns.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            allocations (ArrayLike): Flexible array containing allocation data.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_allocate_func`.

        Returns:
            PortfolioOptimizer: Instance of the portfolio optimizer with the specified allocations.
        """
        if isinstance(allocations, pd.Series):
            allocations = allocations.to_dict()
        if isinstance(allocations, dict):
            allocations = pd.DataFrame([allocations], columns=wrapper.columns)
            allocations = allocations.values
        if isinstance(allocations, pd.DataFrame):
            kwargs = merge_dicts(dict(on=allocations.index, kind="labels"), kwargs)
            allocations = allocations.values
        if not isinstance(allocations, np.ndarray):
            with WarningsFiltered():
                try:
                    new_allocations = np.asarray(allocations)
                    if new_allocations.dtype != object:
                        allocations = new_allocations
                except Exception:
                    pass

        if isinstance(allocations, np.ndarray):

            def _resolve_allocations(index_points):
                target_shape = (len(index_points), len(wrapper.columns))
                return broadcast_array_to(allocations, target_shape, expand_axis=0)

            return cls.from_allocate_func(
                wrapper,
                nb.pick_idx_allocate_func_nb,
                RepFunc(_resolve_allocations),
                jitted_loop=True,
                **kwargs,
            )

        def _pick_allocate_func(index_points, i):
            if not checks.is_sequence(allocations):
                return allocations
            if len(allocations) == 1:
                return allocations[0]
            if len(index_points) != len(allocations):
                raise ValueError(f"Allocation array must have {len(index_points)} rows")
            return allocations[i]

        return cls.from_allocate_func(
            wrapper, _pick_allocate_func, Rep("index_points"), Rep("i"), **kwargs
        )

    @classmethod
    def from_initial(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        allocations: tp.ArrayLike,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Allocate allocations at the first index.

        Uses `PortfolioOptimizer.from_allocations` with `on=0`.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            allocations (ArrayLike): Flexible array containing allocation data.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_allocations`.

        Returns:
            PortfolioOptimizer: Instance of the portfolio optimizer with the initial allocation
                applied at the first index.
        """
        return cls.from_allocations(wrapper, allocations, on=0, **kwargs)

    @classmethod
    def from_filled_allocations(
        cls: tp.Type[PortfolioOptimizerT],
        allocations: tp.AnyArray2d,
        valid_only: bool = True,
        nonzero_only: bool = True,
        unique_only: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Pick allocations from an already filled allocation array.

        Uses `PortfolioOptimizer.from_allocate_func` with a Numba-compiled loop via
        `vectorbtpro.portfolio.pfopt.nb.pick_point_allocate_func_nb`.
        Allocation points are extracted using `vectorbtpro.portfolio.pfopt.nb.get_alloc_points_nb`.

        Args:
            allocations (AnyArray2d): Filled allocation array.
            valid_only (bool): Skip rows where all values are NaN.
            nonzero_only (bool): Skip rows where all values are zero.
            unique_only (bool): Skip rows that are identical to the previous valid row.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Required if allocations is not a DataFrame.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_allocate_func`.

        Returns:
            PortfolioOptimizer: Instance of the portfolio optimizer with the filled allocations.
        """
        if wrapper is None:
            if checks.is_frame(allocations):
                wrapper = ArrayWrapper.from_obj(allocations)
            else:
                raise TypeError("Wrapper is required if allocations is not a DataFrame")
        allocations = to_2d_array(allocations, expand_axis=0)
        if allocations.shape != wrapper.shape_2d:
            raise ValueError("Allocation array must have the same shape as wrapper")
        on = nb.get_alloc_points_nb(
            allocations,
            valid_only=valid_only,
            nonzero_only=nonzero_only,
            unique_only=unique_only,
        )
        kwargs = merge_dicts(dict(on=on), kwargs)
        return cls.from_allocate_func(
            wrapper,
            nb.pick_point_allocate_func_nb,
            allocations,
            jitted_loop=True,
            **kwargs,
        )

    @classmethod
    def from_uniform(
        cls: tp.Type[PortfolioOptimizerT], wrapper: ArrayWrapper, **kwargs
    ) -> PortfolioOptimizerT:
        """Generate uniform allocations.

        Uses `PortfolioOptimizer.from_allocate_func` with a uniform allocation function that
        assigns equal weights to all assets.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_allocate_func`.

        Returns:
            PortfolioOptimizer: Instance of the portfolio optimizer with uniform allocations.
        """

        def _uniform_allocate_func():
            return np.full(wrapper.shape_2d[1], 1 / wrapper.shape_2d[1])

        return cls.from_allocate_func(wrapper, _uniform_allocate_func, **kwargs)

    @classmethod
    def from_random(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        direction: tp.Union[str, int] = "longonly",
        n: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate random allocations.

        Uses `PortfolioOptimizer.from_allocate_func` with a Numba-compiled random allocation function
        (`vectorbtpro.portfolio.pfopt.nb.random_allocate_func_nb`).

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            direction (Union[str, int]): Market direction for allocation, e.g. "longonly".

                Mapped using `vectorbtpro.portfolio.enums.Direction` if provided as a string.
            n (Optional[int]): Number of columns to assign random weights.

                If None, assign weights to all columns.
            seed (Optional[int]): Random seed for deterministic output.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_allocate_func`.

        Returns:
            PortfolioOptimizer: Instance of the portfolio optimizer with random allocations.
        """
        if isinstance(direction, str):
            direction = map_enum_fields(direction, Direction)
        if seed is not None:
            set_seed_nb(seed)
        return cls.from_allocate_func(
            wrapper,
            nb.random_allocate_func_nb,
            wrapper.shape_2d[1],
            direction,
            n,
            jitted_loop=True,
            **kwargs,
        )

    @classmethod
    def from_universal_algo(
        cls: tp.Type[PortfolioOptimizerT],
        algo: tp.Union[str, tp.Type[AlgoT], AlgoT, AlgoResultT],
        S: tp.Optional[tp.AnyArray2d] = None,
        n_jobs: int = 1,
        log_progress: bool = False,
        valid_only: bool = True,
        nonzero_only: bool = True,
        unique_only: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations using Universal Portfolios.

        Uses a universal portfolio algorithm from
        [Universal Portfolios](https://github.com/Marigold/universal-portfolios) to generate allocations.
        The parameter `S` represents price data, while `algo` must be either:

        * a package attribute,
        * a subclass of `universal.algo.Algo`,
        * an instance of `universal.algo.Algo`,
        * or an instance of `universal.result.AlgoResult`.

        Allocation points are extracted using `vectorbtpro.portfolio.pfopt.nb.get_alloc_points_nb`.

        Args:
            algo (Union[str, type, AlgoT, AlgoResult]): Universal portfolio algorithm identifier or instance.
            S (Optional[AnyArray2d]): Price data for running the algorithm.
            n_jobs (int): Number of parallel jobs for algorithm execution.
            log_progress (bool): Flag to enable progress logging.
            valid_only (bool): Skip rows where all values are NaN.
            nonzero_only (bool): Skip rows where all values are zero.
            unique_only (bool): Skip rows that are identical to the previous valid row.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Required if `S` is not a DataFrame.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_allocate_func`.

        Returns:
            PortfolioOptimizer: Instance of the portfolio optimizer with allocations generated
                by the universal algorithm.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("universal")
        from universal.algo import Algo
        from universal.result import AlgoResult

        if wrapper is None:
            if S is None or not checks.is_frame(S):
                raise TypeError("Wrapper is required if allocations is not a DataFrame")
            else:
                wrapper = ArrayWrapper.from_obj(S)

        def _pre_group_func(group_config, _algo=algo):
            _ = group_config.pop("args", ())
            if isinstance(_algo, str):
                import universal.algos

                _algo = getattr(universal.algos, _algo)
            if isinstance(_algo, type) and issubclass(_algo, Algo):
                reserved_arg_names = get_func_arg_names(cls.from_allocate_func)
                algo_keys = set(group_config.keys()).difference(reserved_arg_names)
                algo_kwargs = {}
                for k in algo_keys:
                    algo_kwargs[k] = group_config.pop(k)
                _algo = _algo(**algo_kwargs)
            if isinstance(_algo, Algo):
                if S is None:
                    raise ValueError("S is required")
                _algo = _algo.run(S, n_jobs=n_jobs, log_progress=log_progress)
            if isinstance(_algo, AlgoResult):
                weights = _algo.weights[wrapper.columns].values
            else:
                raise TypeError(f"Algo {_algo} not supported")
            if "on" not in kwargs:
                group_config["on"] = nb.get_alloc_points_nb(
                    weights,
                    valid_only=valid_only,
                    nonzero_only=nonzero_only,
                    unique_only=unique_only,
                )
            group_config["args"] = (weights,)

        return cls.from_allocate_func(
            wrapper,
            nb.pick_point_allocate_func_nb,
            jitted_loop=True,
            pre_group_func=_pre_group_func,
            **kwargs,
        )

    @classmethod
    def run_optimization_group(
        cls,
        wrapper: ArrayWrapper,
        group_configs: tp.List[dict],
        group_index: tp.Index,
        group_idx: int,
        pre_group_func: tp.Optional[tp.Callable] = None,
        silence_warnings: bool = False,
    ) -> tp.Tuple[tp.RecordArray, tp.Array2d]:
        """Run optimization for a group of assets.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            group_configs (List[dict]): List of configurations for optimization groups.
            group_index (Index): Index representing the optimization group.
            group_idx (int): Index specifying which configuration in `group_configs` to use.
            pre_group_func (Callable): Function to preprocess and modify the group configuration.
            silence_warnings (bool): Flag to suppress warning messages.

        Returns:
            Tuple[RecordArray, Array2d]: Tuple containing the allocation records and
                the corresponding 2D array of allocation values.

        See:
            * `vectorbtpro.portfolio.pfopt.nb.optimize_meta_nb` if `jitted_loop=True`
        """
        group_config = dict(group_configs[group_idx])
        if pre_group_func is not None:
            pre_group_func(group_config)

        optimize_func = group_config.pop("optimize_func")
        every = group_config.pop("every")
        normalize_every = group_config.pop("normalize_every")
        split_every = group_config.pop("split_every")
        start_time = group_config.pop("start_time")
        end_time = group_config.pop("end_time")
        lookback_period = group_config.pop("lookback_period")
        start = group_config.pop("start")
        end = group_config.pop("end")
        exact_start = group_config.pop("exact_start")
        fixed_start = group_config.pop("fixed_start")
        closed_start = group_config.pop("closed_start")
        closed_end = group_config.pop("closed_end")
        add_start_delta = group_config.pop("add_start_delta")
        add_end_delta = group_config.pop("add_end_delta")
        kind = group_config.pop("kind")
        skip_not_found = group_config.pop("skip_not_found")
        index_ranges = group_config.pop("index_ranges")
        index_loc = group_config.pop("index_loc")
        rescale_to = group_config.pop("rescale_to")
        alloc_wait = group_config.pop("alloc_wait")
        splitter_cls = group_config.pop("splitter_cls")
        eval_id = group_config.pop("eval_id")
        jitted_loop = group_config.pop("jitted_loop")
        jitted = group_config.pop("jitted")
        chunked = group_config.pop("chunked")
        template_context = group_config.pop("template_context")
        execute_kwargs = group_config.pop("execute_kwargs")
        args = group_config.pop("args")
        kwargs = group_config

        if splitter_cls is None:
            splitter_cls = Splitter

        template_context = merge_dicts(
            dict(
                group_configs=group_configs,
                group_index=group_index,
                group_idx=group_idx,
                wrapper=wrapper,
                optimize_func=optimize_func,
                every=every,
                normalize_every=normalize_every,
                split_every=split_every,
                start_time=start_time,
                end_time=end_time,
                lookback_period=lookback_period,
                start=start,
                end=end,
                exact_start=exact_start,
                fixed_start=fixed_start,
                closed_start=closed_start,
                closed_end=closed_end,
                add_start_delta=add_start_delta,
                add_end_delta=add_end_delta,
                kind=kind,
                skip_not_found=skip_not_found,
                index_ranges=index_ranges,
                index_loc=index_loc,
                rescale_to=rescale_to,
                alloc_wait=alloc_wait,
                splitter_cls=splitter_cls,
                eval_id=eval_id,
                jitted_loop=jitted_loop,
                jitted=jitted,
                chunked=chunked,
                args=args,
                kwargs=kwargs,
                execute_kwargs=execute_kwargs,
            ),
            template_context,
        )

        if index_ranges is None:
            get_index_ranges_defaults = substitute_templates(
                dict(
                    every=every,
                    normalize_every=normalize_every,
                    split_every=split_every,
                    start_time=start_time,
                    end_time=end_time,
                    lookback_period=lookback_period,
                    start=start,
                    end=end,
                    exact_start=exact_start,
                    fixed_start=fixed_start,
                    closed_start=closed_start,
                    closed_end=closed_end,
                    add_start_delta=add_start_delta,
                    add_end_delta=add_end_delta,
                    kind=kind,
                    skip_not_found=skip_not_found,
                    jitted=jitted,
                ),
                template_context,
                eval_id="get_index_ranges_defaults",
                strict=True,
            )
            index_ranges = wrapper.get_index_ranges(**get_index_ranges_defaults)
            template_context = merge_dicts(
                template_context,
                get_index_ranges_defaults,
                dict(index_ranges=index_ranges),
            )
        else:
            index_ranges = substitute_templates(
                index_ranges,
                template_context,
                eval_id="index_ranges",
                strict=True,
            )
            if isinstance(index_ranges, np.ndarray):
                index_ranges = (index_ranges[:, 0], index_ranges[:, 1])
            elif not isinstance(index_ranges[0], np.ndarray) and not isinstance(
                index_ranges[1], np.ndarray
            ):
                index_ranges = to_2d_array(index_ranges, expand_axis=0)
                index_ranges = (index_ranges[:, 0], index_ranges[:, 1])
            template_context = merge_dicts(template_context, dict(index_ranges=index_ranges))
        if index_loc is not None:
            index_loc = substitute_templates(
                index_loc,
                template_context,
                eval_id="index_loc",
                strict=True,
            )
            index_loc = to_1d_array(index_loc)
            template_context = merge_dicts(template_context, dict(index_loc=index_loc))

        if jitted_loop:
            optimize_func = substitute_templates(
                optimize_func,
                template_context,
                eval_id="optimize_func",
                strict=True,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            kwargs = substitute_templates(kwargs, template_context, eval_id="kwargs")
            func = jit_reg.resolve_option(nb.optimize_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            _allocations = func(
                len(wrapper.columns),
                index_ranges[0],
                index_ranges[1],
                optimize_func,
                *args,
                **kwargs,
            )
        else:
            tasks = []
            keys = []
            for i in range(len(index_ranges[0])):
                index_slice = slice(max(0, index_ranges[0][i]), index_ranges[1][i])
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_slice=index_slice,
                        index_start=index_ranges[0][i],
                        index_end=index_ranges[1][i],
                    ),
                    template_context,
                )
                _optimize_func = substitute_templates(
                    optimize_func,
                    _template_context,
                    eval_id="optimize_func",
                    strict=True,
                )
                _args = substitute_templates(args, _template_context, eval_id="args")
                _kwargs = substitute_templates(kwargs, _template_context, eval_id="kwargs")

                if has_annotatables(_optimize_func):
                    ann_args = annotate_args(
                        _optimize_func,
                        _args,
                        _kwargs,
                        attach_annotations=True,
                    )
                    flat_ann_args = flatten_ann_args(ann_args)
                    flat_ann_args = splitter_cls.parse_and_inject_takeables(
                        flat_ann_args, eval_id=eval_id
                    )
                    ann_args = unflatten_ann_args(flat_ann_args)
                    _args, _kwargs = ann_args_to_args(ann_args)

                __args = ()
                for v in _args:
                    if isinstance(v, Takeable) and v.meets_eval_id(eval_id):
                        v = splitter_cls.take_range_from_takeable(
                            v,
                            index_slice,
                            template_context=_template_context,
                            silence_warnings=silence_warnings,
                            index=wrapper.index,
                            freq=wrapper.freq,
                        )
                    __args += (v,)
                __kwargs = {}
                for k, v in _kwargs.items():
                    if isinstance(v, Takeable) and v.meets_eval_id(eval_id):
                        v = splitter_cls.take_range_from_takeable(
                            v,
                            index_slice,
                            template_context=_template_context,
                            silence_warnings=silence_warnings,
                            index=wrapper.index,
                            freq=wrapper.freq,
                        )
                    __kwargs[k] = v

                tasks.append(Task(_optimize_func, *__args, **__kwargs))
                if isinstance(wrapper.index, pd.DatetimeIndex):
                    keys.append(
                        "{} → {}".format(
                            dt.readable_datetime(
                                wrapper.index[index_ranges[0][i]], freq=wrapper.freq
                            ),
                            dt.readable_datetime(
                                wrapper.index[index_ranges[1][i] - 1], freq=wrapper.freq
                            ),
                        )
                    )
                else:
                    keys.append(
                        f"{str(wrapper.index[index_ranges[0][i]])} → {str(wrapper.index[index_ranges[1][i] - 1])}"
                    )
            results = execute(tasks, keys=keys, **execute_kwargs)
            _allocations = pd.DataFrame(results, columns=wrapper.columns)
            if isinstance(_allocations.columns, pd.RangeIndex):
                _allocations = _allocations.values
            else:
                _allocations = _allocations[list(wrapper.columns)].values
            if rescale_to is not None:
                _allocations = nb.rescale_allocations_nb(_allocations, rescale_to)

        if index_loc is None:
            alloc_wait = substitute_templates(
                alloc_wait,
                template_context,
                eval_id="alloc_wait",
                strict=True,
            )
            alloc_idx = index_ranges[1] - 1 + alloc_wait
        else:
            alloc_idx = index_loc
        status = np.where(
            alloc_idx >= len(wrapper.index),
            RangeStatus.Open,
            RangeStatus.Closed,
        )
        return nb.prepare_alloc_ranges_nb(
            index_ranges[0],
            index_ranges[1],
            alloc_idx,
            status,
            _allocations,
            group_idx,
        )

    @classmethod
    def from_optimize_func(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: ArrayWrapper,
        optimize_func: tp.Callable,
        *args,
        every: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults["every"],
        normalize_every: tp.Union[bool, Param] = range_idxr_defaults["normalize_every"],
        split_every: tp.Union[bool, Param] = range_idxr_defaults["split_every"],
        start_time: tp.Union[None, tp.TimeLike, Param] = range_idxr_defaults["start_time"],
        end_time: tp.Union[None, tp.TimeLike, Param] = range_idxr_defaults["end_time"],
        lookback_period: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults[
            "lookback_period"
        ],
        start: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, Param] = range_idxr_defaults[
            "start"
        ],
        end: tp.Union[None, int, tp.DatetimeLike, tp.IndexLike, Param] = range_idxr_defaults["end"],
        exact_start: tp.Union[bool, Param] = range_idxr_defaults["exact_start"],
        fixed_start: tp.Union[bool, Param] = range_idxr_defaults["fixed_start"],
        closed_start: tp.Union[bool, Param] = range_idxr_defaults["closed_start"],
        closed_end: tp.Union[bool, Param] = range_idxr_defaults["closed_end"],
        add_start_delta: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults[
            "add_start_delta"
        ],
        add_end_delta: tp.Union[None, tp.FrequencyLike, Param] = range_idxr_defaults[
            "add_end_delta"
        ],
        kind: tp.Union[None, str, Param] = range_idxr_defaults["kind"],
        skip_not_found: tp.Union[bool, Param] = range_idxr_defaults["skip_not_found"],
        index_ranges: tp.Union[None, tp.MaybeSequence[tp.MaybeSequence[int]], Param] = None,
        index_loc: tp.Union[None, tp.MaybeSequence[int], Param] = None,
        rescale_to: tp.Union[None, tp.Tuple[float, float], Param] = None,
        alloc_wait: tp.Union[int, Param] = 1,
        parameterizer: tp.Optional[tp.MaybeType[Parameterizer]] = None,
        param_search_kwargs: tp.KwargsLike = None,
        name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
        group_configs: tp.Union[
            None, tp.Dict[tp.Hashable, tp.Kwargs], tp.Sequence[tp.Kwargs]
        ] = None,
        pre_group_func: tp.Optional[tp.Callable] = None,
        splitter_cls: tp.Optional[tp.Type[Splitter]] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
        jitted_loop: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        template_context: tp.KwargsLike = None,
        group_execute_kwargs: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        random_subset: tp.Optional[int] = None,
        clean_index_kwargs: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Generate allocations from an optimization function.

        Generates date ranges, executes an optimization function on the data subset corresponding
        to each range, and allocates assets at the end of each range.

        This parameterized method supports testing multiple combinations of arguments. It first checks if any
        argument is wrapped with `vectorbtpro.utils.params.Param` and combines their values. These parameters
        may then be aggregated over `group_configs` if provided. Before execution, the group configuration is
        further processed using `pre_group_func`.

        Date ranges are resolved either by using the provided `index_ranges` or by passing all arguments
        from `every` to `jitted` to `vectorbtpro.base.wrapping.ArrayWrapper.get_index_ranges`.
        The optimization function `optimize_func` is then called on each date range after substituting
        any templates in `*args` and `**kwargs`. To forward reserved arguments such as `jitted` to the
        optimization function, list their names in `forward_args` and `forward_kwargs`.

        !!! note
            Use vectorbtpro's templates to select the current date range
            (available as `index_slice` in the context mapping).

        If `jitted_loop` is True, see `vectorbtpro.portfolio.pfopt.nb.optimize_meta_nb`.
        Otherwise, the optimization function should accept the template-substituted `*args` and `**kwargs`
        and return an array or dictionary with asset allocations.

        Templates can utilize the following variables:

        * `i`: Optimization step
        * `index_start`: Optimization start index (inclusive)
        * `index_end`: Optimization end index (exclusive)
        * `index_slice`: `slice(index_start, index_end)`

        !!! note
            When `jitted_loop` is True and multiple groups exist, utilize templates to substitute
            the current group index (available as `group_idx` in the context mapping).

        All allocations across groups are stacked into a 2D array where columns represent
        assets and rows represent allocations. Date ranges are used to create a record array of type
        `vectorbtpro.portfolio.pfopt.records.AllocRanges` that acts as an indexer for allocations.
        For example, the field `col` indicates the group index for each allocation. This record array
        has its own wrapper that holds groups instead of columns, while the wrapper of the `PortfolioOptimizer`
        instance contains regular columns grouped by groups.

        !!! hint
            Wrap arrays with `vectorbtpro.generic.splitting.base.Takeable` to split them automatically.

        Args:
            wrapper (ArrayWrapper): Array wrapper instance.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            optimize_func (Callable): Optimization function that computes asset allocations
                based on a given date range.
            *args: Positional arguments for `optimize_func`.
            every (Union[None, FrequencyLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.every`.
            normalize_every (Union[bool, Param]): See `vectorbtpro.base.indexing.RangeIdxr.normalize_every`.
            split_every (Union[bool, Param]): See `vectorbtpro.base.indexing.RangeIdxr.split_every`.
            start_time (Union[None, TimeLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.start_time`.
            end_time (Union[None, TimeLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.end_time`.
            lookback_period (Union[None, FrequencyLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.lookback_period`.
            start (Union[None, int, DatetimeLike, IndexLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.start`.
            end (Union[None, int, DatetimeLike, IndexLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.end`.
            exact_start (Union[bool, Param]): See `vectorbtpro.base.indexing.RangeIdxr.exact_start`.
            fixed_start (Union[bool, Param]): See `vectorbtpro.base.indexing.RangeIdxr.fixed_start`.
            closed_start (Union[bool, Param]): See `vectorbtpro.base.indexing.RangeIdxr.closed_start`.
            closed_end (Union[bool, Param]): See `vectorbtpro.base.indexing.RangeIdxr.closed_end`.
            add_start_delta (Union[None, FrequencyLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.add_start_delta`.
            add_end_delta (Union[None, FrequencyLike, Param]): See `vectorbtpro.base.indexing.RangeIdxr.add_end_delta`.
            kind (Union[None, str, Param]): See `vectorbtpro.base.indexing.RangeIdxr.kind`.
            skip_not_found (Union[bool, Param]): See `vectorbtpro.base.indexing.RangeIdxr.skip_not_found`.
            index_ranges (Union[None, MaybeSequence[MaybeSequence[int]], Param]): Manually provided index ranges.
            index_loc (Union[None, MaybeSequence[int], Param]): Specific index locations.
            rescale_to (Union[None, Tuple[float, float], Param]): Tuple specifying the rescaling range for allocations.
            alloc_wait (Union[int, Param]): Wait parameter used during allocation processing.
            parameterizer (Optional[MaybeType[Parameterizer]]): Parameterizer class or instance for handling parameters.

                Defaults to `vectorbtpro.utils.params.Parameterizer`.
            param_search_kwargs (KwargsLike): Keyword arguments for parameter search.

                See `vectorbtpro.utils.params.Parameterizer.find_params_in_obj`.
            name_tuple_to_str (Union[None, bool, Callable]): Flag or function to convert name tuples to
                strings for the parameter index.
            group_configs (Union[None, Dict[Hashable, Kwargs], Sequence[Kwargs]]): Group configuration(s) for optimization.
            pre_group_func (Optional[Callable]): Function to preprocess and modify the group configuration.
            splitter_cls (Optional[Type[Splitter]]): Splitter class to use.

                Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
            eval_id (Optional[Hashable]): Evaluation identifier.
            jitted_loop (bool): Flag indicating whether to use a JIT-compiled loop over optimization groups.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            template_context (KwargsLike): Additional context for template substitution.
            group_execute_kwargs (KwargsLike): Keyword arguments for the execution handler of optimization groups.

                See `vectorbtpro.utils.execution.execute`.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            random_subset (Optional[int]): Select a random subset of parameter combinations.

                Set the seed for reproducibility.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `optimize_func`.

        Returns:
            PortfolioOptimizer: New instance of `PortfolioOptimizer` with generated allocation results.

        !!! info
            For default settings, see `vectorbtpro._settings.params`.

        Examples:
            Allocate once:

            ```pycon
            >>> from vectorbtpro import *

            >>> data = vbt.YFData.pull(
            ...     ["MSFT", "AMZN", "AAPL"],
            ...     start="2010-01-01",
            ...     end="2020-01-01"
            ... )
            >>> close = data.get("Close")

            >>> def optimize_func(df):
            ...     sharpe = df.mean() / df.std()
            ...     return sharpe / sharpe.sum()

            >>> df_arg = vbt.RepEval("close.iloc[index_slice]", context=dict(close=close))
            >>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     end="2015-01-01"
            ... )
            >>> pfo.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2015-01-02 00:00:00+00:00  0.402459  0.309351  0.288191
            ```

            Allocate every first date of the year:

            ```pycon
            >>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN"
            ... )
            >>> pfo.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2011-01-03 00:00:00+00:00  0.480693  0.257317  0.261990
                        2012-01-03 00:00:00+00:00  0.489893  0.215381  0.294727
                        2013-01-02 00:00:00+00:00  0.540165  0.228755  0.231080
                        2014-01-02 00:00:00+00:00  0.339649  0.273996  0.386354
                        2015-01-02 00:00:00+00:00  0.350406  0.418638  0.230956
                        2016-01-04 00:00:00+00:00  0.332212  0.141090  0.526698
                        2017-01-03 00:00:00+00:00  0.390852  0.225379  0.383769
                        2018-01-02 00:00:00+00:00  0.337711  0.317683  0.344606
                        2019-01-02 00:00:00+00:00  0.411852  0.282680  0.305468
            ```

            Specify index ranges manually:

            ```pycon
            >>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     index_ranges=[
            ...         (0, 30),
            ...         (30, 60),
            ...         (60, 90)
            ...     ]
            ... )
            >>> pfo.allocations
            symbol                                     MSFT      AMZN      AAPL
            alloc_group Date
            group       2010-02-16 00:00:00+00:00  0.340641  0.285897  0.373462
                        2010-03-30 00:00:00+00:00  0.596392  0.206317  0.197291
                        2010-05-12 00:00:00+00:00  0.437481  0.283160  0.279358
            ```

            Test multiple combinations of one argument:

            ```pycon
            >>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN",
            ...     start="2015-01-01",
            ...     lookback_period=vbt.Param(["3MS", "6MS"])
            ... )
            >>> pfo.allocations
            symbol                                         MSFT      AMZN      AAPL
            lookback_period Date
            3MS             2016-01-04 00:00:00+00:00  0.282725  0.234970  0.482305
                            2017-01-03 00:00:00+00:00  0.318100  0.269355  0.412545
                            2018-01-02 00:00:00+00:00  0.387499  0.236432  0.376068
                            2019-01-02 00:00:00+00:00  0.575464  0.254808  0.169728
            6MS             2016-01-04 00:00:00+00:00  0.265035  0.198619  0.536346
                            2017-01-03 00:00:00+00:00  0.314144  0.409020  0.276836
                            2018-01-02 00:00:00+00:00  0.322741  0.282639  0.394621
                            2019-01-02 00:00:00+00:00  0.565691  0.234760  0.199549
            ```

            Test multiple cross-argument combinations:

            ```pycon
            >>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func,
            ...     df_arg,
            ...     every="AS-JAN",
            ...     group_configs=[
            ...         dict(start="2015-01-01"),
            ...         dict(start="2019-06-01", every="MS"),
            ...         dict(end="2014-01-01")
            ...     ]
            ... )
            >>> pfo.allocations
            symbol                                      MSFT      AMZN      AAPL
            group_config Date
            0            2016-01-04 00:00:00+00:00  0.332212  0.141090  0.526698
                         2017-01-03 00:00:00+00:00  0.390852  0.225379  0.383769
                         2018-01-02 00:00:00+00:00  0.337711  0.317683  0.344606
                         2019-01-02 00:00:00+00:00  0.411852  0.282680  0.305468
            1            2019-07-01 00:00:00+00:00  0.351461  0.327334  0.321205
                         2019-08-01 00:00:00+00:00  0.418411  0.249799  0.331790
                         2019-09-03 00:00:00+00:00  0.400439  0.374044  0.225517
                         2019-10-01 00:00:00+00:00  0.509387  0.250497  0.240117
                         2019-11-01 00:00:00+00:00  0.349983  0.469181  0.180835
                         2019-12-02 00:00:00+00:00  0.260437  0.380563  0.359000
            2            2012-01-03 00:00:00+00:00  0.489892  0.215381  0.294727
                         2013-01-02 00:00:00+00:00  0.540165  0.228755  0.231080
                         2014-01-02 00:00:00+00:00  0.339649  0.273997  0.386354
            ```

            Use Numba-compiled loop:

            ```pycon
            >>> @njit
            ... def optimize_func_nb(i, from_idx, to_idx, close):
            ...     mean = vbt.nb.nanmean_nb(close[from_idx:to_idx])
            ...     std = vbt.nb.nanstd_nb(close[from_idx:to_idx])
            ...     sharpe = mean / std
            ...     return sharpe / np.sum(sharpe)

            >>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
            ...     close.vbt.wrapper,
            ...     optimize_func_nb,
            ...     np.asarray(close),
            ...     index_ranges=[
            ...         (0, 30),
            ...         (30, 60),
            ...         (60, 90)
            ...     ],
            ...     jitted_loop=True
            ... )
            >>> pfo.allocations
            symbol                         MSFT      AMZN      AAPL
            Date
            2010-02-17 00:00:00+00:00  0.336384  0.289598  0.374017
            2010-03-31 00:00:00+00:00  0.599417  0.207158  0.193425
            2010-05-13 00:00:00+00:00  0.434084  0.281246  0.284670
            ```

            !!! hint
                There is little reason to use a Numba-compiled loop, except when rebalancing thousands of times.
                Typically, a regular Python loop with a Numba-compiled optimization function is sufficient.
        """
        from vectorbtpro._settings import settings

        params_cfg = settings["params"]

        if parameterizer is None:
            parameterizer = params_cfg["parameterizer"]
        if parameterizer is None:
            parameterizer = Parameterizer
        param_search_kwargs = merge_dicts(params_cfg["param_search_kwargs"], param_search_kwargs)
        if group_execute_kwargs is None:
            group_execute_kwargs = {}
        if execute_kwargs is None:
            execute_kwargs = {}
        if clean_index_kwargs is None:
            clean_index_kwargs = {}

        # Prepare group config names
        gc_names = []
        gc_names_none = True
        n_configs = 0
        if group_configs is not None:
            group_configs = list(group_configs)
            for i, group_config in enumerate(group_configs):
                if isinstance(group_configs, dict):
                    new_group_configs = []
                    for k, v in group_configs.items():
                        v = dict(v)
                        v["_name"] = k
                        new_group_configs.append(v)
                    group_configs = new_group_configs
                else:
                    group_configs = list(group_configs)
                if "args" in group_config:
                    for k, arg in enumerate(group_config.pop("args")):
                        group_config[f"args_{k}"] = arg
                if "kwargs" in group_config:
                    for k, v in enumerate(group_config.pop("kwargs")):
                        group_config[k] = v
                if "_name" in group_config and group_config["_name"] is not None:
                    gc_names.append(group_config.pop("_name"))
                    gc_names_none = False
                else:
                    gc_names.append(n_configs)
                group_configs[i] = group_config
                n_configs += 1
        else:
            group_configs = []

        # Combine parameters
        paramable_kwargs = {
            "every": every,
            "normalize_every": normalize_every,
            "split_every": split_every,
            "start_time": start_time,
            "end_time": end_time,
            "lookback_period": lookback_period,
            "start": start,
            "end": end,
            "exact_start": exact_start,
            "fixed_start": fixed_start,
            "closed_start": closed_start,
            "closed_end": closed_end,
            "add_start_delta": add_start_delta,
            "add_end_delta": add_end_delta,
            "kind": kind,
            "skip_not_found": skip_not_found,
            "index_ranges": index_ranges,
            "index_loc": index_loc,
            "rescale_to": rescale_to,
            "alloc_wait": alloc_wait,
            **{f"args_{i}": args[i] for i in range(len(args))},
            **kwargs,
        }
        param_dct = parameterizer.find_params_in_obj(paramable_kwargs, **param_search_kwargs)
        param_columns = None
        if len(param_dct) > 0:
            param_product, param_columns = combine_params(
                param_dct,
                random_subset=random_subset,
                clean_index_kwargs=clean_index_kwargs,
                name_tuple_to_str=name_tuple_to_str,
            )
            if param_columns is None:
                n_param_configs = len(param_product[list(param_product.keys())[0]])
                param_columns = pd.RangeIndex(stop=n_param_configs, name="param_config")
            product_group_configs = parameterizer.param_product_to_objs(
                paramable_kwargs, param_product
            )
            if len(group_configs) == 0:
                group_configs = product_group_configs
            else:
                new_group_configs = []
                for i in range(len(product_group_configs)):
                    for group_config in group_configs:
                        new_group_config = merge_dicts(product_group_configs[i], group_config)
                        new_group_configs.append(new_group_config)
                group_configs = new_group_configs

        # Build group index
        n_config_params = len(gc_names)
        if param_columns is not None:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = param_columns
            else:
                group_index = combine_indexes(
                    (
                        param_columns,
                        pd.Index(gc_names, name="group_config"),
                    ),
                    **clean_index_kwargs,
                )
        else:
            if n_config_params == 0 or (n_config_params == 1 and gc_names_none):
                group_index = pd.Index(["group"], name="group")
            else:
                group_index = pd.Index(gc_names, name="group_config")

        # Create group config from arguments if empty
        if len(group_configs) == 0:
            single_group = True
            group_configs.append(dict())
        else:
            single_group = False

        # Resolve each group
        groupable_kwargs = {
            "optimize_func": optimize_func,
            **paramable_kwargs,
            "splitter_cls": splitter_cls,
            "eval_id": eval_id,
            "jitted_loop": jitted_loop,
            "jitted": jitted,
            "chunked": chunked,
            "template_context": template_context,
            "execute_kwargs": execute_kwargs,
        }
        new_group_configs = []
        for group_config in group_configs:
            new_group_config = merge_dicts(groupable_kwargs, group_config)
            _args = ()
            while True:
                if f"args_{len(_args)}" in new_group_config:
                    _args += (new_group_config.pop(f"args_{len(_args)}"),)
                else:
                    break
            new_group_config["args"] = _args
            new_group_configs.append(new_group_config)
        group_configs = new_group_configs

        # Generate allocations
        tasks = []
        for group_idx, group_config in enumerate(group_configs):
            tasks.append(
                Task(
                    cls.run_optimization_group,
                    wrapper=wrapper,
                    group_configs=group_configs,
                    group_index=group_index,
                    group_idx=group_idx,
                    pre_group_func=pre_group_func,
                )
            )
        group_execute_kwargs = merge_dicts(
            dict(show_progress=False if single_group else None), group_execute_kwargs
        )
        results = execute(tasks, keys=group_index, **group_execute_kwargs)
        alloc_ranges, allocations = zip(*results)

        # Build column hierarchy
        new_columns = combine_indexes((group_index, wrapper.columns), **clean_index_kwargs)

        # Create instance
        wrapper_kwargs = merge_dicts(
            dict(
                index=wrapper.index,
                columns=new_columns,
                ndim=2,
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
                group_select=True,
                grouped_ndim=1 if single_group else 2,
                group_by=group_index.names if group_index.nlevels > 1 else group_index.name,
                allow_enable=False,
                allow_disable=True,
                allow_modify=False,
            ),
            wrapper_kwargs,
        )
        new_wrapper = ArrayWrapper(**wrapper_kwargs)
        alloc_ranges = AllocRanges(
            ArrayWrapper(
                index=wrapper.index,
                columns=new_wrapper.get_columns(),
                ndim=new_wrapper.get_ndim(),
                freq=wrapper.freq,
                column_only_select=False,
                range_only_select=True,
            ),
            np.concatenate(alloc_ranges),
        )
        allocations = row_stack_arrays(allocations)
        return cls(new_wrapper, alloc_ranges, allocations)

    @classmethod
    def from_pypfopt(
        cls: tp.Type[PortfolioOptimizerT],
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Instantiate a portfolio optimizer using `PortfolioOptimizer.from_optimize_func`
        with `pypfopt_optimize` (PyPortfolioOpt).

        Args:
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Must be provided if `prices` and `returns` arguments are not available.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_optimize_func` and `pypfopt_optimize`.

        Returns:
            PortfolioOptimizer: Portfolio optimizer instance.
        """
        if wrapper is None:
            if "prices" in kwargs:
                wrapper = ArrayWrapper.from_obj(kwargs["prices"])
            elif "returns" in kwargs:
                wrapper = ArrayWrapper.from_obj(kwargs["returns"])
            else:
                raise TypeError("Must provide a wrapper if price and returns are not set")
        else:
            checks.assert_instance_of(wrapper, ArrayWrapper, arg_name="wrapper")
        if "prices" in kwargs and not isinstance(kwargs["prices"], CustomTemplate):
            kwargs["prices"] = RepFunc(
                lambda index_slice, _prices=kwargs["prices"]: _prices.iloc[index_slice]
            )
        if "returns" in kwargs and not isinstance(kwargs["returns"], CustomTemplate):
            kwargs["returns"] = RepFunc(
                lambda index_slice, _returns=kwargs["returns"]: _returns.iloc[index_slice]
            )
        return cls.from_optimize_func(wrapper, pypfopt_optimize, **kwargs)

    @classmethod
    def from_riskfolio(
        cls: tp.Type[PortfolioOptimizerT],
        returns: tp.AnyArray2d,
        wrapper: tp.Optional[ArrayWrapper] = None,
        **kwargs,
    ) -> PortfolioOptimizerT:
        """Instantiate a portfolio optimizer using `PortfolioOptimizer.from_optimize_func`
        with `riskfolio_optimize` (Riskfolio-Lib).

        Args:
            returns (AnyArray2d): 2D array representing asset returns data.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Must be provided if `returns` is a template.
            **kwargs: Keyword arguments for `PortfolioOptimizer.from_optimize_func` and `riskfolio_optimize`.

        Returns:
            PortfolioOptimizer: Portfolio optimizer instance.
        """
        if wrapper is None:
            if not isinstance(returns, CustomTemplate):
                wrapper = ArrayWrapper.from_obj(returns)
            else:
                raise TypeError("Must provide a wrapper if returns are a template")
        else:
            checks.assert_instance_of(wrapper, ArrayWrapper, arg_name="wrapper")
        if not isinstance(returns, CustomTemplate):
            returns = RepFunc(lambda index_slice, _returns=returns: _returns.iloc[index_slice])
        return cls.from_optimize_func(wrapper, riskfolio_optimize, returns, **kwargs)

    # ############# Properties ############# #

    @property
    def alloc_records(self) -> tp.Union[AllocRanges, AllocPoints]:
        """Allocation records as ranges (`AllocRanges`) or points (`AllocPoints`).

        Returns:
            Union[AllocRanges, AllocPoints]: Allocation records.
        """
        return self._alloc_records

    def get_allocations(self, squeeze_groups: bool = True) -> tp.Frame:
        """Obtain a DataFrame with allocation groups concatenated along the index axis.

        Args:
            squeeze_groups (bool): If True and the data's grouped ndim is 1,
                group levels are squeezed in the resulting DataFrame.

        Returns:
            DataFrame: DataFrame containing the allocation groups.
        """
        idx_arr = self.alloc_records.get_field_arr("idx")
        group_arr = self.alloc_records.col_arr
        allocations = self._allocations
        if isinstance(self.alloc_records, AllocRanges):
            closed_mask = self.alloc_records.get_field_arr("status") == RangeStatus.Closed
            idx_arr = idx_arr[closed_mask]
            group_arr = group_arr[closed_mask]
            allocations = allocations[closed_mask]
        if squeeze_groups and self.wrapper.grouped_ndim == 1:
            index = self.wrapper.index[idx_arr]
        else:
            index = stack_indexes(
                (self.alloc_records.wrapper.columns[group_arr], self.wrapper.index[idx_arr])
            )
        n_group_levels = self.wrapper.grouper.get_index().nlevels
        columns = self.wrapper.columns.droplevel(tuple(range(n_group_levels))).unique()
        return pd.DataFrame(allocations, index=index, columns=columns)

    @property
    def allocations(self) -> tp.Frame:
        """Allocation DataFrame computed using `PortfolioOptimizer.get_allocations` with default arguments.

        Returns:
            Frame: DataFrame of allocations.
        """
        return self.get_allocations()

    @property
    def mean_allocation(self) -> tp.Series:
        """Mean allocation per column.

        Returns:
            Series: Series representing the mean allocation for each column.
        """
        group_level_names = self.wrapper.grouper.get_index().names
        return (
            self.get_allocations(squeeze_groups=False).groupby(group_level_names).mean().transpose()
        )

    def fill_allocations(
        self,
        dropna: tp.Optional[str] = None,
        fill_value: tp.Scalar = np.nan,
        wrap_kwargs: tp.KwargsLike = None,
        squeeze_groups: bool = True,
    ) -> tp.Frame:
        """Fill an empty DataFrame with allocation values.

        If `dropna` is "all", rows with all NaN values are removed.
        If `dropna` is "head", rows prior to the first allocation are removed.

        Args:
            dropna (Optional[str]): Strategy for handling missing allocations, either "all" or "head".
            fill_value (Scalar): Value used to fill empty DataFrame entries.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            squeeze_groups (bool): If True and the data's grouped ndim is 1,
                group levels are squeezed in the resulting DataFrame.

        Returns:
            DataFrame: DataFrame with filled allocation values.
        """
        if wrap_kwargs is None:
            wrap_kwargs = {}
        out = self.wrapper.fill(fill_value, group_by=False, **wrap_kwargs)
        idx_arr = self.alloc_records.get_field_arr("idx")
        group_arr = self.alloc_records.col_arr
        allocations = self._allocations
        if isinstance(self.alloc_records, AllocRanges):
            status_arr = self.alloc_records.get_field_arr("status")
            closed_mask = status_arr == RangeStatus.Closed
            idx_arr = idx_arr[closed_mask]
            group_arr = group_arr[closed_mask]
            allocations = allocations[closed_mask]
        for g in range(len(self.alloc_records.wrapper.columns)):
            group_mask = group_arr == g
            index_mask = np.full(len(self.wrapper.index), False)
            index_mask[idx_arr[group_mask]] = True
            column_mask = self.wrapper.grouper.get_groups() == g
            out.loc[index_mask, column_mask] = allocations[group_mask]
        if dropna is not None:
            if dropna.lower() == "all":
                out = out.dropna(how="all")
            elif dropna.lower() == "head":
                out = out.iloc[idx_arr.min() :]
            else:
                raise ValueError(f"Invalid dropna: '{dropna}'")
        if squeeze_groups and self.wrapper.grouped_ndim == 1:
            n_group_levels = self.wrapper.grouper.get_index().nlevels
            out = out.droplevel(tuple(range(n_group_levels)), axis=1)
        return out

    @property
    def filled_allocations(self) -> tp.Frame:
        """Allocation DataFrame filled with allocation values by invoking
        `PortfolioOptimizer.fill_allocations` with default parameters.

        Returns:
            Frame: DataFrame of filled allocations.
        """
        return self.fill_allocations()

    # ############# Simulation ############# #

    def simulate(self, close: tp.Union[tp.ArrayLike, Data], **kwargs) -> PortfolioT:
        """Simulate a portfolio using `vectorbtpro.portfolio.base.Portfolio.from_optimizer`
        applied to the current optimizer instance.

        Args:
            close (Union[ArrayLike, Data]): Asset close price data used for simulation.
            **kwargs: Keyword arguments for `vectorbtpro.portfolio.base.Portfolio.from_optimizer`.

        Returns:
            Portfolio: Simulated portfolio instance.
        """
        from vectorbtpro.portfolio.base import Portfolio

        return Portfolio.from_optimizer(close, self, **kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `PortfolioOptimizer.stats`.

        Merges the defaults from `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.pfopt`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        pfopt_stats_cfg = settings["pfopt"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), pfopt_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            total_records=dict(
                title="Total Records", calc_func="alloc_records.count", tags="alloc_records"
            ),
            coverage=dict(
                title="Coverage",
                calc_func="alloc_records.get_coverage",
                overlapping=False,
                check_alloc_ranges=True,
                tags=["alloc_ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="alloc_records.get_coverage",
                overlapping=True,
                check_alloc_ranges=True,
                tags=["alloc_ranges", "coverage"],
            ),
            mean_allocation=dict(
                title="Mean Allocation",
                calc_func="mean_allocation",
                post_calc_func=lambda self, out, settings: to_dict(out, orient="index_series"),
                tags="allocations",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        dropna: tp.Optional[str] = "head",
        line_shape: str = "hv",
        plot_rb_dates: tp.Optional[bool] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot allocations.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            dropna (Optional[str]): Strategy for handling missing allocations, either "all" or "head".

                See `PortfolioOptimizer.fill_allocations`.
            line_shape (str): Shape of the plot line (e.g. "hv").
            plot_rb_dates (Optional[bool]): Whether to plot rebalancing dates.

                Defaults to True if there are no more than 20 rebalancing dates.
            trace_kwargs (KwargsLikeSequence): Keyword arguments for `plotly.graph_objects.Scatter`.
            add_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for each shape.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure containing the allocation plot.

        Examples:
            Continuing with the examples under `PortfolioOptimizer.from_optimize_func`:

            ```pycon
            >>> from vectorbtpro import *

            >>> pfo = vbt.PortfolioOptimizer.from_random(
            ...     vbt.ArrayWrapper(
            ...         index=pd.date_range("2020-01-01", "2021-01-01"),
            ...         columns=["MSFT", "AMZN", "AAPL"],
            ...         ndim=2
            ...     ),
            ...     every="MS",
            ...     seed=40
            ... )
            >>> pfo.plot().show()
            ```

            ![](/assets/images/api/pfopt_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/pfopt_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure

        self_group = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if self_group.alloc_records.count() > 0:
            filled_allocations = self_group.fill_allocations(dropna=dropna).ffill()

            fig = filled_allocations.vbt.areaplot(
                line_shape=line_shape,
                trace_kwargs=trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

            if plot_rb_dates is None or (isinstance(plot_rb_dates, bool) and plot_rb_dates):
                rb_dates = self_group.allocations.index
                if plot_rb_dates is None:
                    plot_rb_dates = len(rb_dates) <= 20
                if plot_rb_dates:
                    add_shape_kwargs = merge_dicts(
                        dict(
                            type="line",
                            line=dict(
                                color=fig.layout.template.layout.plot_bgcolor,
                                dash="dot",
                                width=1,
                            ),
                            xref="x",
                            yref="paper",
                            y0=0,
                            y1=1,
                        ),
                        add_shape_kwargs,
                    )
                    for rb_date in rb_dates:
                        fig.add_shape(x0=rb_date, x1=rb_date, **add_shape_kwargs)
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `PortfolioOptimizer.plots`.

        Merges the defaults from `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.pfopt`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        pfopt_plots_cfg = settings["pfopt"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), pfopt_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            alloc_ranges=dict(
                title="Allocation Ranges",
                plot_func="alloc_records.plot",
                check_alloc_ranges=True,
                tags="alloc_ranges",
            ),
            plot=dict(
                title="Allocations",
                plot_func="plot",
                tags="allocations",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


PortfolioOptimizer.override_metrics_doc(__pdoc__)
PortfolioOptimizer.override_subplots_doc(__pdoc__)

PFO = PortfolioOptimizer
"""Alias for `PortfolioOptimizer`."""

__pdoc__["PFO"] = False
