# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a mixin for constructing statistics from performance metrics."""

import inspect
import string
from collections import Counter

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import Wrapping
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import AttrResolverMixin, get_dict_attr
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts
from vectorbtpro.utils.parsing import get_forward_args, get_func_arg_names
from vectorbtpro.utils.tagging import match_tags
from vectorbtpro.utils.template import CustomTemplate, substitute_templates
from vectorbtpro.utils.warnings_ import warn

__all__ = []


class MetaStatsBuilderMixin(type):
    """Metaclass for `StatsBuilderMixin` that provides access to performance metrics configuration."""

    @property
    def metrics(cls) -> Config:
        """Performance metrics configuration used by `StatsBuilderMixin.stats`.

        Returns:
            Config: Performance metrics configuration.
        """
        return cls._metrics


class StatsBuilderMixin(Base, metaclass=MetaStatsBuilderMixin):
    """Mixin class that provides the implementation for `StatsBuilderMixin.stats`.

    Requires subclassing of `vectorbtpro.base.wrapping.Wrapping`.

    !!! info
        For default settings, see `vectorbtpro._settings.stats_builder`.
    """

    _writeable_attrs: tp.WriteableAttrs = {"_metrics"}

    def __init__(self) -> None:
        checks.assert_instance_of(self, Wrapping)

        # Copy writeable attrs
        self._metrics = type(self)._metrics.copy()

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `StatsBuilderMixin.stats`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        return dict(settings=dict(freq=self.wrapper.freq))

    def resolve_stats_setting(
        self,
        value: tp.Optional[tp.Any],
        key: str,
        merge: bool = False,
    ) -> tp.Any:
        """Resolve and return a configuration setting for `StatsBuilderMixin.stats`.

        Args:
            value (Optional[Any]): Provided value for the setting.
            key (str): Key identifying the stats setting.
            merge (bool): Indicates whether to merge the provided value with defaults.

        Returns:
            Any: Resolved configuration setting.

        !!! info
            For default settings, see `vectorbtpro._settings.stats_builder`.
        """
        from vectorbtpro._settings import settings as _settings

        stats_builder_cfg = _settings["stats_builder"]

        if merge:
            return merge_dicts(
                stats_builder_cfg[key],
                self.stats_defaults.get(key, {}),
                value,
            )
        if value is not None:
            return value
        return self.stats_defaults.get(key, stats_builder_cfg[key])

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
        )
    )

    @property
    def metrics(self) -> Config:
        """Metrics configuration for `${cls_name}`.

        This property returns a copy of `${cls_name}._metrics` created during instance initialization.
        Modifications to the returned configuration do not affect the class-level settings.

        ```python
        ${metrics}
        ```

        To modify the metrics, change the configuration in-place, override this property,
        or assign a new value to the instance variable `${cls_name}._metrics`.

        Returns:
            Config: Copy of the metrics configuration for `${cls_name}`.
        """
        return self._metrics

    def stats(
        self,
        metrics: tp.Optional[tp.MaybeIterable[tp.Union[str, tp.Tuple[str, tp.Kwargs]]]] = None,
        tags: tp.Optional[tp.MaybeIterable[str]] = None,
        column: tp.Optional[tp.Column] = None,
        group_by: tp.GroupByLike = None,
        per_column: tp.Optional[bool] = None,
        split_columns: tp.Optional[bool] = None,
        agg_func: tp.Optional[tp.Callable] = np.mean,
        dropna: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        settings: tp.KwargsLike = None,
        filters: tp.KwargsLike = None,
        metric_settings: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Compute various metrics on this object.

        Args:
            metrics (Optional[MaybeIterable[Union[str, Tuple[str, Kwargs]]]]): Metric or metrics to calculate.

                Each element can be one of the following:

                * Metric name (see keys in `StatsBuilderMixin.metrics`).
                * Tuple with a metric name and a settings dictionary as defined in `StatsBuilderMixin.metrics`.
                * Tuple with a metric name and a `vectorbtpro.utils.template.CustomTemplate` instance.
                * Tuple with a metric name and a list of settings dictionaries to be expanded into multiple metrics.

                The settings dictionary can include:

                * `title`: Metric title. Defaults to the metric name.
                * `tags`: Single tag or multiple tags to associate with the metric.
                    The metric is retained if any tag matches those in `tags`.
                * `check_{filter}` and `inv_check_{filter}`: Flags to evaluate the metric against
                    a filter defined in `filters`. True indicates inclusion.
                * `calc_func` (required): Function to compute custom metrics.
                    It should return a scalar for a single column/group, a pandas.Series
                    for multiple columns/groups, or a dictionary of such values for sub-metrics.
                * `resolve_calc_func`: If True, resolve `calc_func` as an attribute of the object
                    if not callable. You can specify the path to this function as a string
                    (see `vectorbtpro.utils.attr_.deep_getattr` for the path format).
                    If `calc_func` is a function, arguments from merged metric settings are matched with
                    arguments in the signature (see below). If False, `calc_func` must accept the resolved
                    object and a dictionary of merged metric settings. Defaults to True.
                * `use_shortcuts`: If True, employ shortcut properties when resolving `calc_func`.
                    Defaults to True.
                * `post_calc_func`: Function to post-process the output of `calc_func`.
                    It must accept the resolved object, the output from `calc_func`, and
                    merged metric settings, and return a valid metric result. Defaults to None.
                * `fill_wrap_kwargs`: If True, populate `wrap_kwargs` with `to_timedelta` and
                    `silence_warnings`. Defaults to False.
                * `apply_to_timedelta`: If True, apply `vectorbtpro.base.wrapping.ArrayWrapper.arr_to_timedelta`
                    on the result. To disable globally, set `to_timedelta` to False in `settings`.
                    Defaults to False.
                * `pass_{arg}`: If True, pass the corresponding argument from settings when found
                    in the function's signature (see below). If False, the argument is not passed.
                    This key is removed if the argument does not exist.
                * `resolve_path_{arg}`: If True, resolve an argument intended as an attribute of the object
                    and used as the first part of `calc_func`'s attribute path. Applies only to
                    optional arguments. Defaults to True.
                    See `vectorbtpro.utils.attr_.AttrResolverMixin.resolve_attr`.
                * `resolve_{arg}`: If True, resolve an argument found in the function's signature
                    as an attribute of the object. Defaults to False.
                    See `vectorbtpro.utils.attr_.AttrResolverMixin.resolve_attr`.
                * `use_shortcuts_{arg}`: If True, use shortcut properties when resolving the argument.
                    Defaults to True.
                * `select_col_{arg}`: If True, select a specific column from an argument that represents
                    an attribute of the object. Defaults to False.
                * `template_context`: Mapping for template substitution in metric settings.
                * Any other keyword argument overrides settings or is passed directly to `calc_func`.

                If `resolve_calc_func` is True, `calc_func` may accept additional arguments, including:

                * Each alias in `vectorbtpro.utils.attr_.AttrResolverMixin.self_aliases`
                    representing the original object (ungrouped, with no column selected).
                * group_by: Not passed if used in resolving the first attribute of `calc_func`'s path,
                    unless `pass_group_by=True` is specified.
                * `column`
                * `metric_name`
                * `agg_func`
                * `silence_warnings`
                * `to_timedelta`: Replaced with True if set to None and a frequency is available.
                * Any argument from `settings`
                * Any attribute of the object intended for resolution
                    (see `vectorbtpro.utils.attr_.AttrResolverMixin.resolve_attr`).

                Pass `metrics='all'` to calculate all supported metrics.
            tags (Optional[MaybeIterable[str]]): Tag or tags to filter metrics.

                See `vectorbtpro.utils.tagging.match_tags`.
            column (Optional[Column]): Identifier of the column to select.

                !!! hint
                    There are two methods to select a column:

                    * `obj['a'].stats()` computes statistics for column 'a' only.
                    * `obj.stats(column='a')` computes statistics for all columns and then selects column 'a'.

                    Use the first method for large datasets or when caching is disabled,
                    and the second when most attributes are cached.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            per_column (Optional[bool]): If True, compute metrics for each column and stack the results.
            split_columns (Optional[bool]): If True and `per_column` is True, split the instance
                into multiple columns; otherwise, iterate over columns and apply `column` to the entire instance.
            agg_func (Optional[Callable]): Function to aggregate computed statistics across columns.

                By default, calculates the mean across columns. If None, returns all columns as a DataFrame.
                The function should accept a pandas.Series and return a scalar. Aggregation is applied
                if `column` is specified or if the object has only one column.

                If overridden by a metric:

                * It takes effect only if the global `agg_func` is not None.
                * Warning is raised if it is None but the calculation returns multiple values.
            dropna (Optional[bool]): If True, omit metrics that are entirely NaN.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.
            template_context (KwargsLike): Additional context for template substitution.

                Merged with `template_context` from `vectorbtpro._settings.stats_builder` and
                `StatsBuilderMixin.stats_defaults`. Applied first to `settings` and then to
                individual metric settings.
            filters (KwargsLike): Filters to apply to metrics.

                Each key is a filter name with its corresponding settings dictionary, which can include:

                * filter_func: Function that takes the resolved object and merged metric settings,
                    returning True or False.
                * warning_message: Warning to display when skipping a metric
                    (supports template substitution).
                * inv_warning_message: Warning for inverse checks.

                These filters are merged with those from `vectorbtpro._settings.stats_builder`
                and `StatsBuilderMixin.stats_defaults`.
            settings (KwargsLike): Global settings and resolution arguments.

                These override values from `vectorbtpro._settings.stats_builder` and
                `StatsBuilderMixin.stats_defaults` and can be extended or overridden
                by metric-specific settings.
            metric_settings (KwargsLike): Overrides for individual metrics.

                Extend or override all global and metric settings on a per-metric basis.

        Returns:
            Optional[SeriesFrame]: Computed metrics as a Pandas Series
                (for single-dimensional output) or DataFrame (for multi-dimensional output).

        !!! info
            For default settings, see `vectorbtpro._settings.stats_builder` and `StatsBuilderMixin.stats_defaults`.

            See `vectorbtpro.utils.template` for template logic.

        !!! hint
            Optional (resolution) arguments are passed only if they appear in the function's signature,
            while mandatory arguments are always passed. Optional arguments are defined via `settings`
            (globally), whereas mandatory arguments can be set using default metric settings or
            `{metric_name}_kwargs`. Overriding optional arguments does not make them mandatory;
            use `pass_{arg}=True` to enforce passing.

        !!! hint
            Resolve and reuse object attributes wherever possible to leverage built-in caching,
            even if global caching is disabled.
        """
        # Compute per column
        if column is None:
            if per_column is None:
                per_column = self.resolve_stats_setting(per_column, "per_column")
            if per_column:
                columns = self.get_item_keys(group_by=group_by)
                if len(columns) > 1:
                    results = []
                    if split_columns:
                        for _, column_self in self.items(group_by=group_by, wrap=True):
                            _args, _kwargs = get_forward_args(column_self.stats, locals())
                            results.append(column_self.stats(*_args, **_kwargs))
                    else:
                        for column in columns:
                            _args, _kwargs = get_forward_args(self.stats, locals())
                            results.append(self.stats(*_args, **_kwargs))
                    return pd.concat(results, keys=columns, axis=1)

        # Resolve defaults
        dropna = self.resolve_stats_setting(dropna, "dropna")
        silence_warnings = self.resolve_stats_setting(silence_warnings, "silence_warnings")
        template_context = self.resolve_stats_setting(
            template_context, "template_context", merge=True
        )
        filters = self.resolve_stats_setting(filters, "filters", merge=True)
        settings = self.resolve_stats_setting(settings, "settings", merge=True)
        metric_settings = self.resolve_stats_setting(metric_settings, "metric_settings", merge=True)

        # Replace templates globally (not used at metric level)
        if len(template_context) > 0:
            sub_settings = substitute_templates(
                settings,
                context=template_context,
                eval_id="sub_settings",
                strict=False,
            )
        else:
            sub_settings = settings

        # Resolve self
        reself = self.resolve_self(
            cond_kwargs=sub_settings,
            impacts_caching=False,
            silence_warnings=silence_warnings,
        )

        # Prepare metrics
        metrics = reself.resolve_stats_setting(metrics, "metrics")
        if metrics == "all":
            metrics = reself.metrics
        if isinstance(metrics, dict):
            metrics = list(metrics.items())
        if isinstance(metrics, (str, tuple)):
            metrics = [metrics]

        # Prepare tags
        tags = reself.resolve_stats_setting(tags, "tags")
        if isinstance(tags, str) and tags == "all":
            tags = None
        if isinstance(tags, (str, tuple)):
            tags = [tags]

        # Bring to the same shape
        new_metrics = []
        for i, metric in enumerate(metrics):
            if isinstance(metric, str):
                metric = (metric, reself.metrics[metric])
            if not isinstance(metric, tuple):
                raise TypeError(f"Metric at index {i} must be either a string or a tuple")
            new_metrics.append(metric)
        metrics = new_metrics

        # Expand metrics
        new_metrics = []
        for i, (metric_name, _metric_settings) in enumerate(metrics):
            if isinstance(_metric_settings, CustomTemplate):
                metric_context = merge_dicts(
                    template_context,
                    dict.fromkeys(reself.self_aliases, reself),
                    dict(
                        column=column,
                        group_by=group_by,
                        metric_name=metric_name,
                        agg_func=agg_func,
                        silence_warnings=silence_warnings,
                        to_timedelta=None,
                    ),
                    settings,
                )
                metric_context = substitute_templates(
                    metric_context,
                    context=metric_context,
                    eval_id="metric_context",
                )
                _metric_settings = _metric_settings.substitute(
                    context=metric_context,
                    strict=True,
                    eval_id="metric",
                )
            if isinstance(_metric_settings, list):
                for __metric_settings in _metric_settings:
                    new_metrics.append((metric_name, __metric_settings))
            else:
                new_metrics.append((metric_name, _metric_settings))
        metrics = new_metrics

        # Handle duplicate names
        metric_counts = Counter(list(map(lambda x: x[0], metrics)))
        metric_i = dict.fromkeys(metric_counts.keys(), -1)
        metrics_dct = {}
        for i, (metric_name, _metric_settings) in enumerate(metrics):
            if metric_counts[metric_name] > 1:
                metric_i[metric_name] += 1
                metric_name = metric_name + "_" + str(metric_i[metric_name])
            metrics_dct[metric_name] = _metric_settings

        # Check metric_settings
        missed_keys = set(metric_settings.keys()).difference(set(metrics_dct.keys()))
        if len(missed_keys) > 0:
            raise ValueError(
                f"Keys {missed_keys} in metric_settings could not be matched with any metric"
            )

        # Merge settings
        opt_arg_names_dct = {}
        custom_arg_names_dct = {}
        resolved_self_dct = {}
        context_dct = {}
        for metric_name, _metric_settings in list(metrics_dct.items()):
            opt_settings = merge_dicts(
                dict.fromkeys(reself.self_aliases, reself),
                dict(
                    column=column,
                    group_by=group_by,
                    metric_name=metric_name,
                    agg_func=agg_func,
                    silence_warnings=silence_warnings,
                    to_timedelta=None,
                ),
                settings,
            )
            _metric_settings = _metric_settings.copy()
            passed_metric_settings = metric_settings.get(metric_name, {})
            merged_settings = merge_dicts(opt_settings, _metric_settings, passed_metric_settings)
            metric_template_context = merged_settings.pop("template_context", {})
            template_context_merged = merge_dicts(template_context, metric_template_context)
            template_context_merged = substitute_templates(
                template_context_merged,
                context=merged_settings,
                eval_id="template_context_merged",
            )
            context = merge_dicts(template_context_merged, merged_settings)
            merged_settings = substitute_templates(
                merged_settings,
                context=context,
                eval_id="merged_settings",
            )

            # Filter by tag
            if tags is not None:
                in_tags = merged_settings.get("tags", None)
                if in_tags is None or not match_tags(tags, in_tags):
                    metrics_dct.pop(metric_name, None)
                    continue

            custom_arg_names = set(_metric_settings.keys()).union(
                set(passed_metric_settings.keys())
            )
            opt_arg_names = set(opt_settings.keys())
            custom_reself = reself.resolve_self(
                cond_kwargs=merged_settings,
                custom_arg_names=custom_arg_names,
                impacts_caching=True,
                silence_warnings=merged_settings["silence_warnings"],
            )

            metrics_dct[metric_name] = merged_settings
            custom_arg_names_dct[metric_name] = custom_arg_names
            opt_arg_names_dct[metric_name] = opt_arg_names
            resolved_self_dct[metric_name] = custom_reself
            context_dct[metric_name] = context

        # Filter metrics
        for metric_name, _metric_settings in list(metrics_dct.items()):
            custom_reself = resolved_self_dct[metric_name]
            context = context_dct[metric_name]
            _silence_warnings = _metric_settings.get("silence_warnings")

            metric_filters = set()
            for k in _metric_settings.keys():
                filter_name = None
                if k.startswith("check_"):
                    filter_name = k[len("check_") :]
                elif k.startswith("inv_check_"):
                    filter_name = k[len("inv_check_") :]
                if filter_name is not None:
                    if filter_name not in filters:
                        raise ValueError(f"Metric '{metric_name}' requires filter '{filter_name}'")
                    metric_filters.add(filter_name)

            for filter_name in metric_filters:
                filter_settings = filters[filter_name]
                _filter_settings = substitute_templates(
                    filter_settings,
                    context=context,
                    eval_id="filter_settings",
                )
                filter_func = _filter_settings["filter_func"]
                warning_message = _filter_settings.get("warning_message", None)
                inv_warning_message = _filter_settings.get("inv_warning_message", None)
                to_check = _metric_settings.get("check_" + filter_name, False)
                inv_to_check = _metric_settings.get("inv_check_" + filter_name, False)

                if to_check or inv_to_check:
                    whether_true = filter_func(custom_reself, _metric_settings)
                    to_remove = (to_check and not whether_true) or (inv_to_check and whether_true)
                    if to_remove:
                        if to_check and warning_message is not None and not _silence_warnings:
                            warn(warning_message)
                        if (
                            inv_to_check
                            and inv_warning_message is not None
                            and not _silence_warnings
                        ):
                            warn(inv_warning_message)

                        metrics_dct.pop(metric_name, None)
                        custom_arg_names_dct.pop(metric_name, None)
                        opt_arg_names_dct.pop(metric_name, None)
                        resolved_self_dct.pop(metric_name, None)
                        context_dct.pop(metric_name, None)
                        break

        # Any metrics left?
        if len(metrics_dct) == 0:
            if not silence_warnings:
                warn("No metrics to calculate")
            return None

        # Compute stats
        arg_cache_dct = {}
        stats_dct = {}
        used_agg_func = False
        for i, (metric_name, _metric_settings) in enumerate(metrics_dct.items()):
            try:
                final_kwargs = _metric_settings.copy()
                opt_arg_names = opt_arg_names_dct[metric_name]
                custom_arg_names = custom_arg_names_dct[metric_name]
                custom_reself = resolved_self_dct[metric_name]

                # Clean up keys
                for k, v in list(final_kwargs.items()):
                    if k.startswith("check_") or k.startswith("inv_check_") or k in ("tags",):
                        final_kwargs.pop(k, None)

                # Get metric-specific values
                _column = final_kwargs.get("column")
                _group_by = final_kwargs.get("group_by")
                _agg_func = final_kwargs.get("agg_func")
                _silence_warnings = final_kwargs.get("silence_warnings")
                if final_kwargs["to_timedelta"] is None:
                    final_kwargs["to_timedelta"] = custom_reself.wrapper.freq is not None
                to_timedelta = final_kwargs.get("to_timedelta")
                title = final_kwargs.pop("title", metric_name)
                calc_func = final_kwargs.pop("calc_func")
                resolve_calc_func = final_kwargs.pop("resolve_calc_func", True)
                post_calc_func = final_kwargs.pop("post_calc_func", None)
                use_shortcuts = final_kwargs.pop("use_shortcuts", True)
                use_caching = final_kwargs.pop("use_caching", True)
                fill_wrap_kwargs = final_kwargs.pop("fill_wrap_kwargs", False)
                if fill_wrap_kwargs:
                    final_kwargs["wrap_kwargs"] = merge_dicts(
                        dict(to_timedelta=to_timedelta, silence_warnings=_silence_warnings),
                        final_kwargs.get("wrap_kwargs", None),
                    )
                apply_to_timedelta = final_kwargs.pop("apply_to_timedelta", False)

                # Resolve calc_func
                if resolve_calc_func:
                    if not callable(calc_func):
                        passed_kwargs_out = {}

                        def _getattr_func(
                            obj: tp.Any,
                            attr: str,
                            args: tp.ArgsLike = None,
                            kwargs: tp.KwargsLike = None,
                            call_attr: bool = True,
                            _final_kwargs: tp.Kwargs = final_kwargs,
                            _opt_arg_names: tp.Set[str] = opt_arg_names,
                            _custom_arg_names: tp.Set[str] = custom_arg_names,
                            _arg_cache_dct: tp.Kwargs = arg_cache_dct,
                            _use_shortcuts: bool = use_shortcuts,
                            _use_caching: bool = use_caching,
                        ) -> tp.Any:
                            if attr in _final_kwargs:
                                return _final_kwargs[attr]
                            if args is None:
                                args = ()
                            if kwargs is None:
                                kwargs = {}

                            if obj is custom_reself:
                                resolve_path_arg = _final_kwargs.pop("resolve_path_" + attr, True)
                                if resolve_path_arg:
                                    if call_attr:
                                        cond_kwargs = {
                                            k: v
                                            for k, v in _final_kwargs.items()
                                            if k in _opt_arg_names
                                        }
                                        out = custom_reself.resolve_attr(
                                            attr,  # do not pass _attr, important for caching
                                            args=args,
                                            cond_kwargs=cond_kwargs,
                                            kwargs=kwargs,
                                            custom_arg_names=_custom_arg_names,
                                            cache_dct=_arg_cache_dct,
                                            use_caching=_use_caching,
                                            passed_kwargs_out=passed_kwargs_out,
                                            use_shortcuts=_use_shortcuts,
                                        )
                                    else:
                                        if isinstance(obj, AttrResolverMixin):
                                            cls_dir = obj.cls_dir
                                        else:
                                            cls_dir = dir(type(obj))
                                        if "get_" + attr in cls_dir:
                                            _attr = "get_" + attr
                                        else:
                                            _attr = attr
                                        out = getattr(obj, _attr)
                                    _select_col_arg = _final_kwargs.pop("select_col_" + attr, False)
                                    if _select_col_arg and _column is not None:
                                        out = custom_reself.select_col_from_obj(
                                            out,
                                            _column,
                                            wrapper=custom_reself.wrapper.regroup(_group_by),
                                        )
                                        passed_kwargs_out["group_by"] = _group_by
                                        passed_kwargs_out["column"] = _column
                                    return out

                            out = getattr(obj, attr)
                            if callable(out) and call_attr:
                                return out(*args, **kwargs)
                            return out

                        calc_func = custom_reself.deep_getattr(
                            calc_func,
                            getattr_func=_getattr_func,
                            call_last_attr=False,
                        )

                        if "group_by" in passed_kwargs_out:
                            if "pass_group_by" not in final_kwargs:
                                final_kwargs.pop("group_by", None)
                        if "column" in passed_kwargs_out:
                            if "pass_column" not in final_kwargs:
                                final_kwargs.pop("column", None)

                    # Resolve arguments
                    if callable(calc_func):
                        func_arg_names = get_func_arg_names(calc_func)
                        for k in func_arg_names:
                            if k not in final_kwargs:
                                resolve_arg = final_kwargs.pop("resolve_" + k, False)
                                use_shortcuts_arg = final_kwargs.pop("use_shortcuts_" + k, True)
                                select_col_arg = final_kwargs.pop("select_col_" + k, False)
                                if resolve_arg:
                                    try:
                                        arg_out = custom_reself.resolve_attr(
                                            k,
                                            cond_kwargs=final_kwargs,
                                            custom_arg_names=custom_arg_names,
                                            cache_dct=arg_cache_dct,
                                            use_caching=use_caching,
                                            use_shortcuts=use_shortcuts_arg,
                                        )
                                    except AttributeError:
                                        continue

                                    if select_col_arg and _column is not None:
                                        arg_out = custom_reself.select_col_from_obj(
                                            arg_out,
                                            _column,
                                            wrapper=custom_reself.wrapper.regroup(_group_by),
                                        )
                                    final_kwargs[k] = arg_out
                        for k in list(final_kwargs.keys()):
                            if k in opt_arg_names:
                                if "pass_" + k in final_kwargs:
                                    if not final_kwargs.get("pass_" + k):  # first priority
                                        final_kwargs.pop(k, None)
                                elif k not in func_arg_names:  # second priority
                                    final_kwargs.pop(k, None)
                        for k in list(final_kwargs.keys()):
                            if k.startswith("pass_") or k.startswith("resolve_"):
                                final_kwargs.pop(k, None)  # cleanup

                        # Call calc_func
                        out = calc_func(**final_kwargs)
                    else:
                        # calc_func is already a result
                        out = calc_func
                else:
                    # Do not resolve calc_func
                    out = calc_func(custom_reself, _metric_settings)

                # Call post_calc_func
                if post_calc_func is not None:
                    out = post_calc_func(custom_reself, out, _metric_settings)

                # Post-process and store the metric
                multiple = True
                if not isinstance(out, dict):
                    multiple = False
                    out = {None: out}
                for k, v in out.items():
                    # Resolve title
                    if multiple:
                        if title is None:
                            t = str(k)
                        else:
                            t = title + ": " + str(k)
                    else:
                        t = title

                    # Check result type
                    if checks.is_any_array(v) and not checks.is_series(v):
                        raise TypeError(
                            "calc_func must return either a scalar for one column/group, "
                            "pd.Series for multiple columns/groups, or a dict of such. "
                            f"Not {type(v)}."
                        )

                    # Handle apply_to_timedelta
                    if apply_to_timedelta and to_timedelta:
                        v = custom_reself.wrapper.arr_to_timedelta(
                            v, silence_warnings=_silence_warnings
                        )

                    # Select column or aggregate
                    if checks.is_series(v):
                        if _column is None and v.shape[0] == 1:
                            v = v.iloc[0]
                        elif _column is not None:
                            v = custom_reself.select_col_from_obj(
                                v,
                                _column,
                                wrapper=custom_reself.wrapper.regroup(_group_by),
                            )
                        elif _agg_func is not None and agg_func is not None:
                            v = _agg_func(v)
                            if _agg_func is agg_func:
                                used_agg_func = True
                        elif _agg_func is None and agg_func is not None:
                            if not _silence_warnings:
                                warn(
                                    f"Metric '{metric_name}' returned multiple values "
                                    "despite having no aggregation function",
                                )
                            continue

                    # Store metric
                    if t in stats_dct:
                        if not _silence_warnings:
                            warn(f"Duplicate metric title '{t}'")
                    stats_dct[t] = v
            except Exception as e:
                warn(f"Metric '{metric_name}' raised an exception")
                raise e

        # Return the stats
        if reself.wrapper.get_ndim(group_by=group_by) == 1:
            sr = pd.Series(
                stats_dct,
                name=reself.wrapper.get_name(group_by=group_by),
                dtype=object,
            )
            if dropna:
                sr.replace([np.inf, -np.inf], np.nan, inplace=True)
                return sr.dropna()
            return sr
        if column is not None:
            sr = pd.Series(stats_dct, name=column, dtype=object)
            if dropna:
                sr.replace([np.inf, -np.inf], np.nan, inplace=True)
                return sr.dropna()
            return sr
        if agg_func is not None:
            if used_agg_func and not silence_warnings:
                warn(
                    f"Object has multiple columns. Aggregated some metrics using {agg_func}. "
                    "Pass either agg_func=None or per_column=True to return statistics per column. "
                    "Pass column to select a single column or group.",
                )
            sr = pd.Series(stats_dct, name="agg_stats", dtype=object)
            if dropna:
                sr.replace([np.inf, -np.inf], np.nan, inplace=True)
                return sr.dropna()
            return sr
        new_index = reself.wrapper.grouper.get_index(group_by=group_by)
        df = pd.DataFrame(stats_dct, index=new_index)
        if dropna:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df.dropna(axis=1, how="all")
        return df

    # ############# Docs ############# #

    @classmethod
    def build_metrics_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Generate and return the metrics documentation string based on a source class.

        Args:
            source_cls (Optional[type]): Source class providing the original configuration.

        Returns:
            str: Generated metrics documentation string with substituted values.
        """
        if source_cls is None:
            source_cls = StatsBuilderMixin
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, "metrics").__doc__),
        ).substitute(
            {"metrics": cls.metrics.prettify_doc(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_metrics_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Override the metrics documentation for the subclass.

        This method updates the provided documentation dictionary by assigning a generated
        metrics documentation string to the key corresponding to the subclass's `metrics` attribute.

        Args:
            __pdoc__ (dict): Dictionary mapping objects to their documentation strings.
            source_cls (Optional[type]): Source class providing the original configuration.

        Returns:
            None
        """
        __pdoc__[cls.__name__ + ".metrics"] = cls.build_metrics_doc(source_cls=source_cls)


__pdoc__ = dict()
StatsBuilderMixin.override_metrics_doc(__pdoc__)
