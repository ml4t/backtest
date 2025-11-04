# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing an adapter for integrating QuantStats with vectorbtpro returns.

!!! note
    Accessors do not utilize caching.

Access the adapter via `ReturnsAccessor`:

```pycon
>>> from vectorbtpro import *
>>> import quantstats as qs

>>> np.random.seed(42)
>>> rets = pd.Series(np.random.uniform(-0.1, 0.1, size=(100,)))
>>> bm_returns = pd.Series(np.random.uniform(-0.1, 0.1, size=(100,)))

>>> rets.vbt.returns.qs.r_squared(benchmark=bm_returns)
0.0011582111228735541
```

This is equivalent to:

```pycon
>>> qs.stats.r_squared(rets, bm_returns)
```

Using the adapter offers two advantages:

* Parameters such as benchmark returns can be defined once rather than passed to every function.
* Vectorbtpro automatically translates parameters from `ReturnsAccessor` for QuantStats functions.

```pycon
>>> # Defaults that vectorbtpro understands
>>> ret_acc = rets.vbt.returns(
...     bm_returns=bm_returns,
...     freq='d',
...     year_freq='365d',
...     defaults=dict(risk_free=0.001)
... )

>>> ret_acc.qs.r_squared()
0.0011582111228735541

>>> ret_acc.qs.sharpe()
-1.9158923252075455

>>> # Defaults that only quantstats understands
>>> qs_defaults = dict(
...     benchmark=bm_returns,
...     periods=365,
...     rf=0.001
... )
>>> ret_acc_qs = rets.vbt.returns.qs(defaults=qs_defaults)

>>> ret_acc_qs.r_squared()
0.0011582111228735541

>>> ret_acc_qs.sharpe()
-1.9158923252075455
```

For example, defaults defined in settings, in `ReturnsAccessor`, and in `QSAdapter` itself are merged
and matched with the function's signature. In particular, the `periods` parameter defaults to
`ReturnsAccessor.ann_factor`, which is based on the `freq` argument, aligning the results from
QuantStats and vectorbtpro.

```pycon
>>> vbt.settings.wrapping['freq'] = 'h'
>>> vbt.settings.returns['year_freq'] = '365d'

>>> rets.vbt.returns.sharpe_ratio()  # ReturnsAccessor
-9.38160953971508

>>> rets.vbt.returns.qs.sharpe()  # quantstats via QSAdapter
-9.38160953971508
```

Arguments can still be overridden by modifying defaults or by providing them directly:

```pycon
>>> rets.vbt.returns.qs(defaults=dict(periods=252)).sharpe()
-1.5912029345745982

>>> rets.vbt.returns.qs.sharpe(periods=252)
-1.5912029345745982

>>> qs.stats.sharpe(rets)
-1.5912029345745982
```
"""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("quantstats")

from inspect import Parameter, getmembers, isfunction, signature

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Configured, merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names, has_variable_kwargs
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "QSAdapter",
]


def attach_qs_methods(cls: tp.Type[tp.T], replace_signature: bool = True) -> tp.Type[tp.T]:
    """Attach QuantStats methods to a class.

    This decorator iterates over functions in QuantStats modules
    (`utils`, `stats`, `plots`, and `reports`) and attaches them as methods
    to the decorated class if they accept a `returns` argument.

    Args:
        cls (Type[T]): Class to which QuantStats methods will be attached.
        replace_signature (bool): Whether to replace the method signature with that of the
            corresponding QuantStats function.

    Returns:
        Type[T]: Decorated class with QuantStats methods attached.

    !!! info
        For default settings, see `vectorbtpro._settings.qs_adapter`.
    """
    try:
        import quantstats as qs
    except AttributeError as e:
        if "'ZMQInteractiveShell' object has no attribute 'magic'" in str(e):
            warn(
                "'ZMQInteractiveShell' object has no attribute 'magic'. "
                "Please downgrade ipython to 7.34.0 or earlier, or uninstall QuantStats."
            )
            return cls
        raise e

    checks.assert_subclass_of(cls, "QSAdapter")

    for module_name in ["utils", "stats", "plots", "reports"]:
        for qs_func_name, qs_func in getmembers(getattr(qs, module_name), isfunction):
            if not qs_func_name.startswith("_") and checks.func_accepts_arg(qs_func, "returns"):
                if module_name == "plots":
                    new_method_name = "plot_" + qs_func_name
                elif module_name == "reports":
                    new_method_name = qs_func_name + "_report"
                else:
                    new_method_name = qs_func_name

                def new_method(
                    self,
                    *,
                    _func: tp.Callable = qs_func,
                    column: tp.Optional[tp.Column] = None,
                    **kwargs,
                ) -> tp.Any:
                    func_arg_names = get_func_arg_names(_func)
                    has_var_kwargs = has_variable_kwargs(_func)
                    defaults = self.defaults

                    if has_var_kwargs:
                        pass_kwargs = dict(kwargs)
                    else:
                        pass_kwargs = {}
                    for arg_name in func_arg_names:
                        if arg_name not in kwargs:
                            if arg_name in defaults:
                                pass_kwargs[arg_name] = defaults[arg_name]
                            elif arg_name == "benchmark":
                                if self.returns_acc.bm_returns is not None:
                                    pass_kwargs["benchmark"] = self.returns_acc.bm_returns
                            elif arg_name == "periods":
                                pass_kwargs["periods"] = int(self.returns_acc.ann_factor)
                            elif arg_name == "periods_per_year":
                                pass_kwargs["periods_per_year"] = int(self.returns_acc.ann_factor)
                        elif not has_var_kwargs:
                            pass_kwargs[arg_name] = kwargs[arg_name]

                    returns = self.returns_acc.select_col_from_obj(
                        self.returns_acc.obj,
                        column=column,
                        wrapper=self.returns_acc.wrapper.regroup(False),
                    )
                    if returns.name is None:
                        returns = returns.rename("Strategy")
                    else:
                        returns = returns.rename(str(returns.name))
                    null_mask = returns.isnull()
                    if "benchmark" in pass_kwargs and pass_kwargs["benchmark"] is not None:
                        benchmark = pass_kwargs["benchmark"]
                        benchmark = self.returns_acc.select_col_from_obj(
                            benchmark,
                            column=column,
                            wrapper=self.returns_acc.wrapper.regroup(False),
                        )
                        if benchmark.name is None:
                            benchmark = benchmark.rename("Benchmark")
                        else:
                            benchmark = benchmark.rename(str(benchmark.name))
                        bm_null_mask = benchmark.isnull()
                        null_mask = null_mask | bm_null_mask
                        benchmark = benchmark.loc[~null_mask]
                        if isinstance(benchmark.index, pd.DatetimeIndex):
                            if benchmark.index.tz is not None:
                                benchmark = benchmark.tz_convert("utc")
                            if benchmark.index.tz is not None:
                                benchmark = benchmark.tz_localize(None)
                        pass_kwargs["benchmark"] = benchmark
                    returns = returns.loc[~null_mask]
                    if isinstance(returns.index, pd.DatetimeIndex):
                        if returns.index.tz is not None:
                            returns = returns.tz_convert("utc")
                        if returns.index.tz is not None:
                            returns = returns.tz_localize(None)

                    signature(_func).bind(returns=returns, **pass_kwargs)
                    return _func(returns=returns, **pass_kwargs)

                if replace_signature:
                    # Replace the function's signature with the original one
                    source_sig = signature(qs_func)
                    new_method_params = tuple(signature(new_method).parameters.values())
                    self_arg = new_method_params[0]
                    column_arg = new_method_params[2]
                    other_args = [
                        (
                            p.replace(kind=Parameter.KEYWORD_ONLY)
                            if p.kind
                            in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                            else p
                        )
                        for p in list(source_sig.parameters.values())[1:]
                    ]
                    source_sig = source_sig.replace(
                        parameters=(self_arg, column_arg) + tuple(other_args)
                    )
                    new_method.__signature__ = source_sig

                new_method.__name__ = new_method_name
                new_method.__module__ = cls.__module__
                new_method.__qualname__ = f"{cls.__name__}.{new_method.__name__}"
                new_method.__doc__ = f"See `quantstats.{module_name}.{qs_func_name}`."
                setattr(cls, new_method_name, new_method)
    return cls


QSAdapterT = tp.TypeVar("QSAdapterT", bound="QSAdapter")


@attach_qs_methods
class QSAdapter(Configured):
    """Adapter class for quantstats.

    Args:
        returns_acc (ReturnsAccessor): Returns accessor instance.
        defaults (KwargsLike): Dictionary of default parameters.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(
        self, returns_acc: ReturnsAccessor, defaults: tp.KwargsLike = None, **kwargs
    ) -> None:
        checks.assert_instance_of(returns_acc, ReturnsAccessor)

        self._returns_acc = returns_acc
        self._defaults = defaults

        Configured.__init__(self, returns_acc=returns_acc, defaults=defaults, **kwargs)

    def __call__(self: QSAdapterT, **kwargs) -> QSAdapterT:
        """Call the instance to update its configuration.

        Args:
            **kwargs: Keyword arguments for `QSAdapter.replace`.

        Returns:
            QSAdapter: New instance with updated configuration.
        """
        return self.replace(**kwargs)

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """Accessor instance.

        This is the main entry point for accessing returns-related methods and properties.

        Returns:
            ReturnsAccessor: Returns accessor instance.
        """
        return self._returns_acc

    @property
    def defaults_mapping(self) -> tp.Dict[str, str]:
        """Mapping of common quantstats argument names to
        `vectorbtpro.returns.accessors.ReturnsAccessor.defaults`.

        This mapping is used to translate parameters from `ReturnsAccessor` to
        QuantStats functions.

        Returns:
            Dict[str, str]: Dictionary mapping common argument names to their corresponding defaults.
        """
        return dict(rf="risk_free", rolling_period="window")

    @property
    def defaults(self) -> tp.Kwargs:
        """Merged default parameters for `QSAdapter`.

        Merges defaults from `vectorbtpro._settings.qs_adapter`, mapped values from
        `vectorbtpro.returns.accessors.ReturnsAccessor.defaults`, and user-provided defaults.

        Returns:
            Kwargs: Merged default settings for plots.

        !!! info
            For default settings, see `defaults` in `vectorbtpro._settings.qs_adapter`.
        """
        from vectorbtpro._settings import settings

        qs_adapter_defaults_cfg = settings["qs_adapter"]["defaults"]

        mapped_defaults = dict()
        for k, v in self.defaults_mapping.items():
            if v in self.returns_acc.defaults:
                mapped_defaults[k] = self.returns_acc.defaults[v]
        return merge_dicts(qs_adapter_defaults_cfg, mapped_defaults, self._defaults)
