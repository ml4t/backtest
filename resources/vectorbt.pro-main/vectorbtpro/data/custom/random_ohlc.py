# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RandomOHLCData` class for generating synthetic OHLC data."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import broadcast_array_to
from vectorbtpro.data import nb
from vectorbtpro.data.custom.synthetic import SyntheticData
from vectorbtpro.ohlcv import nb as ohlcv_nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import substitute_templates

__all__ = [
    "RandomOHLCData",
]

__pdoc__ = {}


class RandomOHLCData(SyntheticData):
    """Data class for synthetic OHLC data generation.

    See:
        * `RandomOHLCData.generate_symbol` for argument details.

    !!! info
        For default settings, see `custom.random_ohlc` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.random_ohlc")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        n_ticks: tp.Optional[tp.ArrayLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        symmetric: tp.Optional[bool] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SymbolData:
        """Generate data for a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            index (Index): Pandas index representing the time stamps.
            n_ticks (Optional[ArrayLike]): Number of ticks per bar.

                Flexible argument that can be provided as a template with a context
                containing `symbol` and `index`.
            start_value (Optional[float]): Initial value at time 0.

                Note that this value does not appear as the first data point.
            mean (Optional[float]): Drift or average percentage change.
            std (Optional[float]): Standard deviation of the percentage change.
            symmetric (Optional[bool]): If True, adjust negative returns to be symmetric with positive ones.
            seed (Optional[int]): Random seed for deterministic output.

                !!! note
                    When using a seed, pass a unique seed per feature or symbol via
                    `vectorbtpro.data.base.feature_dict`, `vectorbtpro.data.base.symbol_dict`,
                    or generally `vectorbtpro.data.base.key_dict`.
            jitted (Optional[JittedOption]): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Additional keyword arguments.

        Returns:
            KeyData: Generated data and a metadata dictionary.

        See:
            * `vectorbtpro.data.nb.generate_random_data_1d_nb` for generating random data.
            * `vectorbtpro.ohlcv.nb.ohlc_every_1d_nb` for aggregating ticks into OHLC bars.
        """
        n_ticks = cls.resolve_custom_setting(n_ticks, "n_ticks")
        template_context = merge_dicts(dict(symbol=symbol, index=index), template_context)
        n_ticks = substitute_templates(n_ticks, template_context, eval_id="n_ticks")
        n_ticks = broadcast_array_to(n_ticks, len(index))
        start_value = cls.resolve_custom_setting(start_value, "start_value")
        mean = cls.resolve_custom_setting(mean, "mean")
        std = cls.resolve_custom_setting(std, "std")
        symmetric = cls.resolve_custom_setting(symmetric, "symmetric")
        seed = cls.resolve_custom_setting(seed, "seed")
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_random_data_1d_nb, jitted)
        ticks = func(np.sum(n_ticks), start_value=start_value, mean=mean, std=std, symmetric=symmetric)
        func = jit_reg.resolve_option(ohlcv_nb.ohlc_every_1d_nb, jitted)
        out = func(ticks, n_ticks)
        return pd.DataFrame(out, index=index, columns=["Open", "High", "Low", "Close"])

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol]["Open"].iloc[-1]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)
