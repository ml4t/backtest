# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a mixin for building plots using subplots."""

import inspect
import string
from collections import Counter

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexing import ParamLoc
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


class MetaPlotsBuilderMixin(type):
    """Metaclass for `PlotsBuilderMixin`."""

    @property
    def subplots(cls) -> Config:
        """Subplots configuration used by `PlotsBuilderMixin.plots`.

        Returns:
            Config: Dictionary containing the default subplots configuration.
        """
        return cls._subplots


class PlotsBuilderMixin(Base, metaclass=MetaPlotsBuilderMixin):
    """Mixin class that provides plotting configurations via the `plots` functionality.

    This mixin requires that the class is a subclass of `vectorbtpro.base.wrapping.Wrapping`.

    !!! info
        For default settings, see `vectorbtpro._settings.plots_builder`.
    """

    _writeable_attrs: tp.WriteableAttrs = {"_subplots"}

    def __init__(self) -> None:
        checks.assert_instance_of(self, Wrapping)

        # Copy writeable attrs
        self._subplots = type(self)._subplots.copy()

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `PlotsBuilderMixin.plots`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        return dict(settings=dict(freq=self.wrapper.freq))

    def resolve_plots_setting(
        self,
        value: tp.Optional[tp.Any],
        key: str,
        merge: bool = False,
    ) -> tp.Any:
        """Resolve a setting for `PlotsBuilderMixin.plots` by combining defaults and user-provided values.

        Args:
            value (Optional[Any]): Provided value for the setting.
            key (str): Key identifying the specific setting.
            merge (bool): Flag indicating whether to merge the default and provided values.

        Returns:
            Any: Resolved setting based on the merge strategy and default configurations.
        """
        from vectorbtpro._settings import settings as _settings

        plots_builder_cfg = _settings["plots_builder"]

        if merge:
            return merge_dicts(
                plots_builder_cfg[key],
                self.plots_defaults.get(key, {}),
                value,
            )
        if value is not None:
            return value
        return self.plots_defaults.get(key, plots_builder_cfg[key])

    _subplots: tp.ClassVar[Config] = HybridConfig(dict())

    @property
    def subplots(self) -> Config:
        """Subplots configuration for `${cls_name}`.

        ```python
        ${subplots}
        ```

        This property returns a hybrid copy of `${cls_name}._subplots` created at instance initialization,
        ensuring that modifications do not affect the class-level configuration.

        To modify the subplots, update the configuration in-place, override this property, or assign a new value
        to `${cls_name}._subplots` on the instance.

        Returns:
            Config: Hybrid copy of the subplots configuration.
        """
        return self._subplots

    def plots(
        self,
        subplots: tp.Optional[tp.MaybeIterable[tp.Union[str, tp.Tuple[str, tp.Kwargs]]]] = None,
        tags: tp.Optional[tp.MaybeIterable[str]] = None,
        column: tp.Optional[tp.Column] = None,
        group_by: tp.GroupByLike = None,
        per_column: tp.Optional[bool] = None,
        split_columns: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        settings: tp.KwargsLike = None,
        filters: tp.KwargsLike = None,
        subplot_settings: tp.KwargsLike = None,
        show_titles: bool = None,
        show_legend: tp.Optional[bool] = None,
        show_column_label: tp.Optional[bool] = None,
        hide_id_labels: bool = None,
        group_id_labels: bool = None,
        make_subplots_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.Optional[tp.BaseFigure]:
        """Plot various parts of this object.

        Args:
            subplots (Optional[MaybeIterable[Union[str, Tuple[str, Kwargs]]]]): Subplots to plot.

                Each element may be:

                * Subplot name (as defined in `PlotsBuilderMixin.subplots`).
                * Tuple containing a subplot name and a settings dictionary
                    (as defined in `PlotsBuilderMixin.subplots`).
                * Tuple containing a subplot name and a `vectorbtpro.utils.template.CustomTemplate` instance.
                * Tuple containing a subplot name and a list of settings dictionaries
                    to be expanded into multiple subplots.

                The settings dictionary may include:

                * `title`: Subplot title; defaults to the subplot name.
                * `plot_func` (required): Plotting function for custom subplots.
                    It must modify the provided figure `fig` in-place.
                * `xaxis_kwargs`: Layout keyword arguments for the x-axis;
                    defaults to `{'title': 'Index'}`.
                * `yaxis_kwargs`: Layout keyword arguments for the y-axis;
                    defaults to an empty dict.
                * `tags`, `check_{filter}`, `inv_check_{filter}`, `resolve_plot_func`,
                    `pass_{arg}`, `resolve_path_{arg}`, `resolve_{arg}`, and `template_context`:
                    As described in `vectorbtpro.generic.stats_builder.StatsBuilderMixin` for `calc_func`.
                * Any additional keyword argument that overrides settings or is passed directly to `plot_func`.

                If `resolve_plot_func` is True, the plotting function may request extra parameters
                by accepting them or if `pass_{arg}` was found in the settings dictionary:

                * Each alias from `vectorbtpro.utils.attr_.AttrResolverMixin.self_aliases`
                    representing the original object (ungrouped, without column selection).
                * `group_by` (unless already used to resolve the first attribute of `plot_func`;
                    use `pass_group_by=True` to force its inclusion).
                * `column`
                * `subplot_name`
                * `trace_names`: List containing the subplot name (cannot be used in templates).
                * `add_trace_kwargs`: Dict with subplot row and column index.
                * `xref`
                * `yref`
                * `xaxis`
                * `yaxis`
                * `x_domain`
                * `y_domain`
                * `fig`
                * `silence_warnings`
                * Any parameter from `settings`
                * Any attribute of the object intended for resolution
                    (see `vectorbtpro.utils.attr_.AttrResolverMixin.resolve_attr`).

                !!! note
                    Layout-related resolution arguments (such as `add_trace_kwargs`) are unavailable
                    before filtering and cannot be used in templates, though they may still be overridden.

                Pass `subplots='all'` to plot all supported subplots.
            tags (Optional[MaybeIterable[str]]): Tag or tags to filter metrics, as described in
                `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.
            column (Optional[Column]): Identifier of the column to plot, as described in
                `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.
            group_by (GroupByLike): Grouping specification, as described in
                `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.
            per_column (Optional[bool]): Flag indicating whether to plot per column,
                as in `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.
            split_columns (Optional[bool]): Flag indicating whether to split columns,
                as in `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.
            template_context (KwargsLike): Additional context for template substitution.
            filters (KwargsLike): Filters as specified in `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.
            settings (KwargsLike): Settings as specified in `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.
            subplot_settings (KwargsLike): Subplot-specific settings, analogous to
                `metric_settings` in `StatsBuilderMixin`.
            show_titles (bool): Whether to display titles for subplots.
            show_legend (Optional[bool]): Whether to display the legend.

                If None and plotting per column, the value is inferred.
            show_column_label (Optional[bool]): Whether to display the column label next to each legend label.

                If None and plotting per column, the value is inferred.
            hide_id_labels (bool): Whether to hide duplicate legend labels (duplicates have the same name,
                marker style, and line style).
            group_id_labels (bool): Whether to group identical legend labels.
            make_subplots_kwargs (KwargsLike): Keyword arguments for creating subplots.

                See `vectorbtpro.utils.figure.make_subplots`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            Optional[BaseFigure]: Plotly figure containing subplots.

        !!! note
            `PlotsBuilderMixin` and `vectorbtpro.generic.stats_builder.StatsBuilderMixin`
            share similar designs, differing mainly in nomenclature:

            * `plots_defaults` vs `stats_defaults`
            * `subplots` vs `metrics`
            * `subplot_settings` vs `metric_settings`

            See further details in `vectorbtpro.generic.stats_builder.StatsBuilderMixin`.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.
        """
        # Plot per column
        if column is None:
            if per_column is None:
                per_column = self.resolve_plots_setting(per_column, "per_column")
            if per_column:
                columns = self.get_item_keys(group_by=group_by)
                if len(columns) > 1:
                    if split_columns is None:
                        split_columns = self.resolve_plots_setting(split_columns, "split_columns")
                    if show_legend is None:
                        show_legend = self.resolve_plots_setting(show_legend, "show_legend")
                    if show_legend is None:
                        show_legend = False
                    if show_column_label is None:
                        show_column_label = self.resolve_plots_setting(
                            show_column_label, "show_column_label"
                        )
                    if show_column_label is None:
                        show_column_label = True
                    fig = None
                    if split_columns:
                        for _, column_self in self.items(group_by=group_by, wrap=True):
                            _args, _kwargs = get_forward_args(column_self.plots, locals())
                            fig = column_self.plots(*_args, **_kwargs)
                    else:
                        for column in columns:
                            _args, _kwargs = get_forward_args(self.plots, locals())
                            fig = self.plots(*_args, **_kwargs)
                    return fig

        from vectorbtpro._settings import settings as _settings
        from vectorbtpro.utils.figure import get_domain, make_subplots

        plotting_cfg = _settings["plotting"]

        # Resolve defaults
        silence_warnings = self.resolve_plots_setting(silence_warnings, "silence_warnings")
        show_titles = self.resolve_plots_setting(show_titles, "show_titles")
        show_legend = self.resolve_plots_setting(show_legend, "show_legend")
        if show_legend is None:
            show_legend = True
        show_column_label = self.resolve_plots_setting(show_column_label, "show_column_label")
        if show_column_label is None:
            show_column_label = False
        hide_id_labels = self.resolve_plots_setting(hide_id_labels, "hide_id_labels")
        group_id_labels = self.resolve_plots_setting(group_id_labels, "group_id_labels")
        template_context = self.resolve_plots_setting(
            template_context, "template_context", merge=True
        )
        filters = self.resolve_plots_setting(filters, "filters", merge=True)
        settings = self.resolve_plots_setting(settings, "settings", merge=True)
        subplot_settings = self.resolve_plots_setting(
            subplot_settings, "subplot_settings", merge=True
        )
        make_subplots_kwargs = self.resolve_plots_setting(
            make_subplots_kwargs, "make_subplots_kwargs", merge=True
        )
        layout_kwargs = self.resolve_plots_setting(layout_kwargs, "layout_kwargs", merge=True)

        # Replace templates globally (not used at subplot level)
        if len(template_context) > 0:
            sub_settings = substitute_templates(
                settings,
                context=template_context,
                eval_id="sub_settings",
                strict=False,
            )
            sub_make_subplots_kwargs = substitute_templates(
                make_subplots_kwargs,
                context=template_context,
                eval_id="sub_make_subplots_kwargs",
            )
            sub_layout_kwargs = substitute_templates(
                layout_kwargs,
                context=template_context,
                eval_id="sub_layout_kwargs",
            )
        else:
            sub_settings = settings
            sub_make_subplots_kwargs = make_subplots_kwargs
            sub_layout_kwargs = layout_kwargs

        # Resolve self
        reself = self.resolve_self(
            cond_kwargs=sub_settings,
            impacts_caching=False,
            silence_warnings=silence_warnings,
        )

        # Prepare subplots
        if subplots is None:
            subplots = reself.resolve_plots_setting(subplots, "subplots")
        if subplots == "all":
            subplots = reself.subplots
        if isinstance(subplots, dict):
            subplots = list(subplots.items())
        if isinstance(subplots, (str, tuple)):
            subplots = [subplots]

        # Prepare tags
        if tags is None:
            tags = reself.resolve_plots_setting(tags, "tags")
        if isinstance(tags, str) and tags == "all":
            tags = None
        if isinstance(tags, (str, tuple)):
            tags = [tags]

        # Bring to the same shape
        new_subplots = []
        for i, subplot in enumerate(subplots):
            if isinstance(subplot, str):
                subplot = (subplot, reself.subplots[subplot])
            if not isinstance(subplot, tuple):
                raise TypeError(f"Subplot at index {i} must be either a string or a tuple")
            new_subplots.append(subplot)
        subplots = new_subplots

        # Expand subplots
        new_subplots = []
        for i, (subplot_name, _subplot_settings) in enumerate(subplots):
            if isinstance(_subplot_settings, CustomTemplate):
                subplot_context = merge_dicts(
                    template_context,
                    dict.fromkeys(reself.self_aliases, reself),
                    dict(
                        column=column,
                        group_by=group_by,
                        subplot_name=subplot_name,
                        silence_warnings=silence_warnings,
                    ),
                    settings,
                )
                subplot_context = substitute_templates(
                    subplot_context,
                    context=subplot_context,
                    eval_id="subplot_context",
                )
                _subplot_settings = _subplot_settings.substitute(
                    context=subplot_context,
                    strict=True,
                    eval_id="subplot",
                )
            if isinstance(_subplot_settings, list):
                for __subplot_settings in _subplot_settings:
                    new_subplots.append((subplot_name, __subplot_settings))
            else:
                new_subplots.append((subplot_name, _subplot_settings))
        subplots = new_subplots

        # Handle duplicate names
        subplot_counts = Counter(list(map(lambda x: x[0], subplots)))
        subplot_i = dict.fromkeys(subplot_counts.keys(), -1)
        subplots_dct = {}
        for i, (subplot_name, _subplot_settings) in enumerate(subplots):
            if subplot_counts[subplot_name] > 1:
                subplot_i[subplot_name] += 1
                subplot_name = subplot_name + "_" + str(subplot_i[subplot_name])
            subplots_dct[subplot_name] = _subplot_settings

        # Check subplot_settings
        missed_keys = set(subplot_settings.keys()).difference(set(subplots_dct.keys()))
        if len(missed_keys) > 0:
            raise ValueError(
                f"Keys {missed_keys} in subplot_settings could not be matched with any subplot"
            )

        # Merge settings
        opt_arg_names_dct = {}
        custom_arg_names_dct = {}
        resolved_self_dct = {}
        context_dct = {}
        for subplot_name, _subplot_settings in list(subplots_dct.items()):
            opt_settings = merge_dicts(
                dict.fromkeys(reself.self_aliases, reself),
                dict(
                    column=column,
                    group_by=group_by,
                    subplot_name=subplot_name,
                    trace_names=[subplot_name],
                    silence_warnings=silence_warnings,
                ),
                settings,
            )
            _subplot_settings = _subplot_settings.copy()
            passed_subplot_settings = subplot_settings.get(subplot_name, {})
            merged_settings = merge_dicts(opt_settings, _subplot_settings, passed_subplot_settings)
            subplot_template_context = merged_settings.pop("template_context", {})
            template_context_merged = merge_dicts(template_context, subplot_template_context)
            template_context_merged = substitute_templates(
                template_context_merged,
                context=merged_settings,
                eval_id="template_context_merged",
            )
            context = merge_dicts(template_context_merged, merged_settings)
            # safe because we will use substitute_templates again once layout params are known
            merged_settings = substitute_templates(
                merged_settings,
                context=context,
                eval_id="merged_settings",
            )

            # Filter by tag
            if tags is not None:
                in_tags = merged_settings.get("tags", None)
                if in_tags is None or not match_tags(tags, in_tags):
                    subplots_dct.pop(subplot_name, None)
                    continue

            custom_arg_names = set(_subplot_settings.keys()).union(
                set(passed_subplot_settings.keys())
            )
            opt_arg_names = set(opt_settings.keys())
            custom_reself = reself.resolve_self(
                cond_kwargs=merged_settings,
                custom_arg_names=custom_arg_names,
                impacts_caching=True,
                silence_warnings=merged_settings["silence_warnings"],
            )

            subplots_dct[subplot_name] = merged_settings
            custom_arg_names_dct[subplot_name] = custom_arg_names
            opt_arg_names_dct[subplot_name] = opt_arg_names
            resolved_self_dct[subplot_name] = custom_reself
            context_dct[subplot_name] = context

        # Filter subplots
        for subplot_name, _subplot_settings in list(subplots_dct.items()):
            custom_reself = resolved_self_dct[subplot_name]
            context = context_dct[subplot_name]
            _silence_warnings = _subplot_settings.get("silence_warnings")

            subplot_filters = set()
            for k in _subplot_settings.keys():
                filter_name = None
                if k.startswith("check_"):
                    filter_name = k[len("check_") :]
                elif k.startswith("inv_check_"):
                    filter_name = k[len("inv_check_") :]
                if filter_name is not None:
                    if filter_name not in filters:
                        raise ValueError(f"Metric '{subplot_name}' requires filter '{filter_name}'")
                    subplot_filters.add(filter_name)

            for filter_name in subplot_filters:
                filter_settings = filters[filter_name]
                _filter_settings = substitute_templates(
                    filter_settings,
                    context=context,
                    eval_id="filter_settings",
                )
                filter_func = _filter_settings["filter_func"]
                warning_message = _filter_settings.get("warning_message", None)
                inv_warning_message = _filter_settings.get("inv_warning_message", None)
                to_check = _subplot_settings.get("check_" + filter_name, False)
                inv_to_check = _subplot_settings.get("inv_check_" + filter_name, False)

                if to_check or inv_to_check:
                    whether_true = filter_func(custom_reself, _subplot_settings)
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

                        subplots_dct.pop(subplot_name, None)
                        custom_arg_names_dct.pop(subplot_name, None)
                        opt_arg_names_dct.pop(subplot_name, None)
                        resolved_self_dct.pop(subplot_name, None)
                        context_dct.pop(subplot_name, None)
                        break

        # Any subplots left?
        if len(subplots_dct) == 0:
            if not silence_warnings:
                warn("No subplots to plot")
            return None

        # Set up figure
        rows = sub_make_subplots_kwargs.pop("rows", len(subplots_dct))
        cols = sub_make_subplots_kwargs.pop("cols", 1)
        specs = sub_make_subplots_kwargs.pop(
            "specs",
            [[{} for _ in range(cols)] for _ in range(rows)],
        )
        row_col_tuples = []
        for row, row_spec in enumerate(specs):
            for col, col_spec in enumerate(row_spec):
                if col_spec is not None:
                    row_col_tuples.append((row + 1, col + 1))
        shared_xaxes = sub_make_subplots_kwargs.pop("shared_xaxes", True)
        shared_yaxes = sub_make_subplots_kwargs.pop("shared_yaxes", False)
        default_height = plotting_cfg["layout"]["height"]
        default_width = plotting_cfg["layout"]["width"] + 50
        min_space = 10  # space between subplots with no axis sharing
        max_title_spacing = 30
        max_xaxis_spacing = 50
        max_yaxis_spacing = 100
        legend_height = 50
        if show_titles:
            title_spacing = max_title_spacing
        else:
            title_spacing = 0
        if not shared_xaxes and rows > 1:
            xaxis_spacing = max_xaxis_spacing
        else:
            xaxis_spacing = 0
        if not shared_yaxes and cols > 1:
            yaxis_spacing = max_yaxis_spacing
        else:
            yaxis_spacing = 0
        if "height" in sub_layout_kwargs:
            height = sub_layout_kwargs.pop("height")
        else:
            height = default_height + title_spacing
            if rows > 1:
                height *= rows
                height += min_space * rows - min_space
                height += legend_height - legend_height * rows
                if shared_xaxes:
                    height += max_xaxis_spacing - max_xaxis_spacing * rows
        if "width" in sub_layout_kwargs:
            width = sub_layout_kwargs.pop("width")
        else:
            width = default_width
            if cols > 1:
                width *= cols
                width += min_space * cols - min_space
                if shared_yaxes:
                    width += max_yaxis_spacing - max_yaxis_spacing * cols
        if height is not None:
            if "vertical_spacing" in sub_make_subplots_kwargs:
                vertical_spacing = sub_make_subplots_kwargs.pop("vertical_spacing")
            else:
                vertical_spacing = min_space + title_spacing + xaxis_spacing
            if vertical_spacing is not None and vertical_spacing > 1:
                vertical_spacing /= height
            legend_y = 1 + (min_space + title_spacing) / height
        else:
            vertical_spacing = sub_make_subplots_kwargs.pop("vertical_spacing", None)
            legend_y = 1.02
        if width is not None:
            if "horizontal_spacing" in sub_make_subplots_kwargs:
                horizontal_spacing = sub_make_subplots_kwargs.pop("horizontal_spacing")
            else:
                horizontal_spacing = min_space + yaxis_spacing
            if horizontal_spacing is not None and horizontal_spacing > 1:
                horizontal_spacing /= width
        else:
            horizontal_spacing = sub_make_subplots_kwargs.pop("horizontal_spacing", None)
        if show_titles:
            _subplot_titles = []
            for i in range(len(subplots_dct)):
                _subplot_titles.append("$title_" + str(i))
        else:
            _subplot_titles = None
        if fig is None:
            fig = make_subplots(
                rows=rows,
                cols=cols,
                specs=specs,
                shared_xaxes=shared_xaxes,
                shared_yaxes=shared_yaxes,
                subplot_titles=_subplot_titles,
                vertical_spacing=vertical_spacing,
                horizontal_spacing=horizontal_spacing,
                **sub_make_subplots_kwargs,
            )
            sub_layout_kwargs = merge_dicts(
                dict(
                    showlegend=True,
                    width=width,
                    height=height,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=legend_y,
                        xanchor="right",
                        x=1,
                        traceorder="normal",
                    ),
                ),
                sub_layout_kwargs,
            )
            trace_start_idx = 0
        else:
            trace_start_idx = len(fig.data)
        fig.update_layout(**sub_layout_kwargs)

        # Plot subplots
        arg_cache_dct = {}
        for i, (subplot_name, _subplot_settings) in enumerate(subplots_dct.items()):
            try:
                final_kwargs = _subplot_settings.copy()
                opt_arg_names = opt_arg_names_dct[subplot_name]
                custom_arg_names = custom_arg_names_dct[subplot_name]
                custom_reself = resolved_self_dct[subplot_name]
                context = context_dct[subplot_name]

                # Compute figure artifacts
                row, col = row_col_tuples[i]
                xref = "x" if i == 0 else "x" + str(i + 1)
                yref = "y" if i == 0 else "y" + str(i + 1)
                xaxis = "xaxis" + xref[1:]
                yaxis = "yaxis" + yref[1:]
                x_domain = get_domain(xref, fig)
                y_domain = get_domain(yref, fig)
                subplot_layout_kwargs = dict(
                    add_trace_kwargs=dict(row=row, col=col),
                    xref=xref,
                    yref=yref,
                    xaxis=xaxis,
                    yaxis=yaxis,
                    x_domain=x_domain,
                    y_domain=y_domain,
                    fig=fig,
                    pass_fig=True,  # force passing fig
                )
                for k in subplot_layout_kwargs:
                    opt_arg_names.add(k)
                    if k in final_kwargs:
                        custom_arg_names.add(k)
                final_kwargs = merge_dicts(subplot_layout_kwargs, final_kwargs)
                context = merge_dicts(subplot_layout_kwargs, context)
                final_kwargs = substitute_templates(
                    final_kwargs, context=context, eval_id="final_kwargs"
                )

                # Clean up keys
                for k, v in list(final_kwargs.items()):
                    if k.startswith("check_") or k.startswith("inv_check_") or k in ("tags",):
                        final_kwargs.pop(k, None)

                # Get subplot-specific values
                _column = final_kwargs.get("column")
                _group_by = final_kwargs.get("group_by")
                _silence_warnings = final_kwargs.get("silence_warnings")
                title = final_kwargs.pop("title", subplot_name)
                plot_func = final_kwargs.pop("plot_func", None)
                xaxis_kwargs = final_kwargs.pop("xaxis_kwargs", None)
                yaxis_kwargs = final_kwargs.pop("yaxis_kwargs", None)
                resolve_plot_func = final_kwargs.pop("resolve_plot_func", True)
                use_shortcuts = final_kwargs.pop("use_shortcuts", True)
                use_caching = final_kwargs.pop("use_caching", True)

                if plot_func is not None:
                    # Resolve plot_func
                    if resolve_plot_func:
                        if not callable(plot_func):
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
                                    resolve_path_arg = _final_kwargs.pop(
                                        "resolve_path_" + attr,
                                        True,
                                    )
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
                                        _select_col_arg = _final_kwargs.pop(
                                            "select_col_" + attr,
                                            False,
                                        )
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

                            plot_func = custom_reself.deep_getattr(
                                plot_func,
                                getattr_func=_getattr_func,
                                call_last_attr=False,
                            )

                            if "group_by" in passed_kwargs_out:
                                if "pass_group_by" not in final_kwargs:
                                    final_kwargs.pop("group_by", None)
                            if "column" in passed_kwargs_out:
                                if "pass_column" not in final_kwargs:
                                    final_kwargs.pop("column", None)
                        if not callable(plot_func):
                            raise TypeError("plot_func must be callable")

                        # Resolve arguments
                        func_arg_names = get_func_arg_names(plot_func)
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

                        # Call plot_func
                        plot_func(**final_kwargs)
                    else:
                        # Do not resolve plot_func
                        plot_func(custom_reself, _subplot_settings)

                # Update global layout
                for annotation in fig.layout.annotations:
                    if "text" in annotation and annotation["text"] == "$title_" + str(i):
                        annotation.update(text=title)
                subplot_layout = dict()
                subplot_layout[xaxis] = merge_dicts(dict(title="Index"), xaxis_kwargs)
                subplot_layout[yaxis] = merge_dicts(dict(), yaxis_kwargs)
                fig.update_layout(**subplot_layout)
            except Exception as e:
                warn(f"Subplot '{subplot_name}' raised an exception")
                raise e

        # Hide legend labels
        if not show_legend:
            for i in range(trace_start_idx, len(fig.data)):
                fig.data[i].update(showlegend=False)

        # Show column label
        if show_column_label:
            if column is not None:
                _column = column
            else:
                _column = reself.wrapper.get_columns(group_by=group_by)[0]
            for i in range(trace_start_idx, len(fig.data)):
                trace = fig.data[i]
                if trace["name"] is not None:
                    trace.update(name=trace["name"] + f" [{ParamLoc.encode_key(_column)}]")

        # Remove duplicate legend labels
        found_ids = dict()
        unique_idx = trace_start_idx
        for i in range(trace_start_idx, len(fig.data)):
            trace = fig.data[i]
            if trace["showlegend"] is not False and trace["legendgroup"] is None:
                if "name" in trace:
                    name = trace["name"]
                else:
                    name = None
                if "marker" in trace:
                    marker = trace["marker"]
                else:
                    marker = {}
                if "symbol" in marker:
                    marker_symbol = marker["symbol"]
                else:
                    marker_symbol = None
                if "color" in marker:
                    marker_color = marker["color"]
                else:
                    marker_color = None
                if "line" in trace:
                    line = trace["line"]
                else:
                    line = {}
                if "dash" in line:
                    line_dash = line["dash"]
                else:
                    line_dash = None
                if "color" in line:
                    line_color = line["color"]
                else:
                    line_color = None

                id = (name, marker_symbol, marker_color, line_dash, line_color)
                if id in found_ids:
                    if hide_id_labels:
                        trace.update(showlegend=False)
                    if group_id_labels:
                        trace.update(legendgroup=found_ids[id])
                else:
                    if group_id_labels:
                        trace.update(legendgroup=unique_idx)
                    found_ids[id] = unique_idx
                    unique_idx += 1

        # Hide identical legend labels
        if hide_id_labels:
            legendgroups = set()
            for i in range(trace_start_idx, len(fig.data)):
                trace = fig.data[i]
                if trace["legendgroup"] is not None:
                    if trace["showlegend"]:
                        if trace["legendgroup"] in legendgroups:
                            trace.update(showlegend=False)
                        else:
                            legendgroups.add(trace["legendgroup"])

        # Remove all except the last title if sharing the same axis
        if shared_xaxes:
            i = 0
            for row in range(rows):
                for col in range(cols):
                    if specs[row][col] is not None:
                        xaxis = "xaxis" if i == 0 else "xaxis" + str(i + 1)
                        if row < rows - 1:
                            fig.layout[xaxis].update(title=None)
                        i += 1
        if shared_yaxes:
            i = 0
            for row in range(rows):
                for col in range(cols):
                    if specs[row][col] is not None:
                        yaxis = "yaxis" if i == 0 else "yaxis" + str(i + 1)
                        if col > 0:
                            fig.layout[yaxis].update(title=None)
                        i += 1

        # Return the figure
        return fig

    @classmethod
    def build_subplots_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build subplots documentation.

        Args:
            source_cls (Optional[type]): Source class providing the original configuration.

                If not provided, defaults to using `PlotsBuilderMixin`.

        Returns:
            str: Generated documentation string for subplots.
        """
        if source_cls is None:
            source_cls = PlotsBuilderMixin
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, "subplots").__doc__),
        ).substitute(
            {"subplots": cls.subplots.prettify_doc(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_subplots_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Override subplots documentation for the subclass.

        Args:
            __pdoc__ (dict): Dictionary mapping objects to their documentation strings.
            source_cls (Optional[type]): Source class providing the original configuration.

                If not provided, defaults to using `PlotsBuilderMixin`.

        Returns:
            None
        """
        __pdoc__[cls.__name__ + ".subplots"] = cls.build_subplots_doc(source_cls=source_cls)


__pdoc__ = dict()
PlotsBuilderMixin.override_subplots_doc(__pdoc__)
