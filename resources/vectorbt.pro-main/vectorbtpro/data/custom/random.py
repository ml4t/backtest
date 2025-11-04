# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RandomData` class for generating synthetic data."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array
from vectorbtpro.data import nb
from vectorbtpro.data.custom.synthetic import SyntheticData
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.random_ import set_seed

__all__ = [
    "RandomData",
]

__pdoc__ = {}


class RandomData(SyntheticData):
    """Data class for synthetic data generation.

    See:
        * `RandomData.generate_key` for argument details.

    !!! info
        For default settings, see `custom.random` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.random")

    @classmethod
    def generate_key(
        cls,
        key: tp.Key,
        index: tp.Index,
        columns: tp.Union[tp.Hashable, tp.IndexLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        symmetric: tp.Optional[bool] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        **kwargs,
    ) -> tp.KeyData:
        """Generate a feature or symbol.

        Args:
            key (Hashable): Feature or symbol identifier.
            index (Index): Pandas index.
            columns (Union[Hashable, IndexLike]): Column names.

                Provide a single value to create a Series.
            start_value (float): Value at time 0.

                Not included as the first value in the output.
            mean (float): Mean of the Gaussian distribution for sampling returns.
            std (float): Standard deviation of the Gaussian distribution for sampling returns.
            symmetric (bool): If True, converts negative returns to be symmetric to positive ones,
                reducing their negative impact.
            seed (int): Random seed for deterministic output.

                !!! note
                    When using a seed, pass a unique seed per feature or symbol via
                    `vectorbtpro.data.base.feature_dict`, `vectorbtpro.data.base.symbol_dict`,
                    or generally `vectorbtpro.data.base.key_dict`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            KeyData: Generated data and a metadata dictionary.

        See:
            `vectorbtpro.data.nb.generate_random_data_nb`
        """
        if checks.is_hashable(columns):
            columns = [columns]
            make_series = True
        else:
            make_series = False
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        start_value = cls.resolve_custom_setting(start_value, "start_value")
        mean = cls.resolve_custom_setting(mean, "mean")
        std = cls.resolve_custom_setting(std, "std")
        symmetric = cls.resolve_custom_setting(symmetric, "symmetric")
        seed = cls.resolve_custom_setting(seed, "seed")
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_random_data_nb, jitted)
        out = func(
            (len(index), len(columns)),
            start_value=to_1d_array(start_value),
            mean=to_1d_array(mean),
            std=to_1d_array(std),
            symmetric=to_1d_array(symmetric),
        )
        if make_series:
            return pd.Series(out[:, 0], index=index, name=columns[0])
        return pd.DataFrame(out, index=index, columns=columns)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        fetch_kwargs = self.select_fetch_kwargs(key)
        fetch_kwargs["start"] = self.select_last_index(key)
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[key].iloc[-2]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, start_value=start_value, **kwargs)
        return self.fetch_symbol(key, start_value=start_value, **kwargs)
