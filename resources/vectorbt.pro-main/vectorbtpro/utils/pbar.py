# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for progress bars.

!!! info
    For default settings, see `vectorbtpro._settings.pbar`.
"""

from functools import wraps
from numbers import Number
from time import time as utc_time

from tqdm.std import tqdm

from vectorbtpro import _typing as tp
from vectorbtpro.registries.pbar_registry import PBarRegistry, pbar_reg
from vectorbtpro.utils.attr_ import MISSING
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "ProgressBar",
    "ProgressHidden",
    "with_progress_hidden",
    "ProgressShown",
    "with_progress_shown",
]


class ProgressBar(Base):
    """Class for managing a progress bar.

    Args:
        iterable (Optional[Iterable]): Iterable for tracking progress.
        bar_id (Optional[Hashable]): Identifier for the progress bar.
        bar_type (Optional[str]): Type of progress bar to use.

            * `tqdm_auto`
            * `tqdm_notebook`
            * `tqdm_gui`
            * `tqdm`
        force_open_bar (Optional[bool]): Flag to force the progress bar to open.
        reuse (Optional[bool]): Flag indicating whether the progress bar can be reused.
        disable (Optional[bool]): Flag to disable the progress bar.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        show_progress_desc (Optional[bool]): Flag indicating if the description is shown.
        prefix (Union[None, str, dict]): Prefix for the progress bar description.
        postfix (Union[None, str, dict]): Postfix for the progress bar description.
        desc_kwargs (KwargsLike): Keyword arguments for customizing the description.
        registry (Optional[PBarRegistry]): Registry for managing progress bar instances.
        silence_warnings (Optional[bool]): Flag to suppress warning messages.
        **kwargs: Keyword arguments for the progress bar constructor.

    !!! info
        For default settings, see `vectorbtpro._settings.pbar`.
    """

    def __init__(
        self,
        iterable: tp.Optional[tp.Iterable] = None,
        bar_id: tp.Optional[tp.Hashable] = None,
        bar_type: tp.Optional[str] = None,
        force_open_bar: tp.Optional[bool] = None,
        reuse: tp.Optional[bool] = None,
        disable: tp.Optional[bool] = None,
        show_progress: tp.Optional[bool] = None,
        show_progress_desc: tp.Optional[bool] = None,
        prefix: tp.Union[None, str, dict] = None,
        postfix: tp.Union[None, str, dict] = None,
        desc_kwargs: tp.KwargsLike = None,
        registry: tp.Optional[PBarRegistry] = pbar_reg,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        if bar_type is None:
            bar_type = pbar_cfg["type"]
        if bar_type.lower() == "tqdm_auto":
            from tqdm.auto import tqdm as bar_type
        elif bar_type.lower() == "tqdm_notebook":
            from tqdm.notebook import tqdm as bar_type
        elif bar_type.lower() == "tqdm_gui":
            from tqdm.gui import tqdm as bar_type
        elif bar_type.lower() == "tqdm":
            from tqdm import tqdm as bar_type
        else:
            raise ValueError(f"Invalid bar_type: '{bar_type}'")
        if force_open_bar is None:
            force_open_bar = pbar_cfg["force_open_bar"]
        if reuse is None:
            reuse = pbar_cfg["reuse"]
        if disable is not None:
            if show_progress is not None:
                if disable or not show_progress:
                    show_progress = False
                else:
                    show_progress = True
            else:
                show_progress = not disable
        else:
            if pbar_cfg["disable"]:
                show_progress = False
            if show_progress is None:
                show_progress = True
        if pbar_cfg["disable_machinery"]:
            show_progress = False
        kwargs = merge_dicts(pbar_cfg["kwargs"], kwargs)
        if pbar_cfg["disable_desc"] or pbar_cfg["disable_machinery"]:
            show_progress_desc = False
        if show_progress_desc is None:
            show_progress_desc = True
        if prefix is not None:
            kwargs["desc"] = self.prepare_desc(prefix)
        if postfix is not None:
            kwargs["postfix"] = self.prepare_desc(postfix)
        desc_kwargs = merge_dicts(pbar_cfg["desc_kwargs"], desc_kwargs)
        if pbar_cfg["disable_registry"] or pbar_cfg["disable_machinery"]:
            registry = None
        if silence_warnings is None:
            silence_warnings = pbar_cfg["silence_warnings"]

        self._iterable = iterable
        self._bar_id = bar_id
        self._bar_type = bar_type
        self._force_open_bar = force_open_bar
        self._reuse = reuse
        self._show_progress = show_progress
        self._kwargs = kwargs
        self._show_progress_desc = show_progress_desc
        self._desc_kwargs = desc_kwargs
        self._registry = registry
        self._silence_warnings = silence_warnings

        self._bar = None
        self._open_time = None
        self._update_time = None
        self._refresh_time = None
        self._close_time = None

    @property
    def bar_id(self) -> tp.Optional[tp.Hashable]:
        """Identifier of the progress bar.

        Returns:
            Optional[Hashable]: Unique identifier for the progress bar if available; otherwise, None.
        """
        return self._bar_id

    @property
    def bar_type(self) -> tp.Type[tqdm]:
        """Type of the progress bar.

        Returns:
            Type[tqdm]: Class/type used for creating the progress bar instance.
        """
        return self._bar_type

    @property
    def force_open_bar(self) -> bool:
        """Indicates whether the progress bar is forced open even if progress is not shown.

        Returns:
            bool: True if the progress bar is forced to open; otherwise, False.
        """
        return self._force_open_bar

    @property
    def reuse(self) -> bool:
        """Indicates whether the progress bar can be reused.

        Returns:
            bool: True if the progress bar instance can be reused; otherwise, False.
        """
        return self._reuse

    @property
    def show_progress(self) -> bool:
        """Indicates whether the progress bar is shown.

        Returns:
            bool: True if the progress bar is set to be displayed; otherwise, False.
        """
        return self._show_progress

    @property
    def iterable(self) -> tp.Optional[tp.Iterable]:
        """Iterable used to track progress.

        Returns:
            Optional[Iterable]: Iterable associated with the progress bar, or None if not provided.
        """
        return self._iterable

    @property
    def kwargs(self) -> tp.Kwargs:
        """Keyword arguments for progress bar initialization.

        Returns:
            Kwargs: Dictionary of keyword arguments used when initializing the progress bar.
        """
        return self._kwargs

    @property
    def show_progress_desc(self) -> bool:
        """Indicates whether the progress bar description is shown.

        Returns:
            bool: True if progress bar description is enabled; otherwise, False.
        """
        return self._show_progress_desc

    @property
    def desc_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for setting the progress bar's description.

        Returns:
            Kwargs: Dictionary of keyword arguments for customizing the progress bar's description.
        """
        return self._desc_kwargs

    @property
    def registry(self) -> tp.Optional[PBarRegistry]:
        """Progress bar registry managing progress bar instances.

        Returns:
            Optional[PBarRegistry]: Registry instance responsible for progress bar management,
                or None if registry operations are disabled.
        """
        return self._registry

    @property
    def silence_warnings(self) -> bool:
        """Indicates whether warnings are silenced.

        Returns:
            bool: True if warnings should be silenced; otherwise, False.
        """
        return self._silence_warnings

    @property
    def bar(self) -> tp.Optional[tqdm]:
        """Current progress bar instance.

        Returns:
            Optional[tqdm]: Active progress bar instance, or None if not set.
        """
        return self._bar

    @property
    def open_time(self) -> tp.Optional[int]:
        """Timestamp indicating when the progress bar was opened.

        Returns:
            Optional[int]: Unix timestamp when the progress bar was opened, or None if not yet opened.
        """
        return self._open_time

    @property
    def update_time(self) -> tp.Optional[int]:
        """Timestamp indicating when the progress bar was last updated.

        Returns:
            Optional[int]: Unix timestamp of the last update, or None if never updated.
        """
        return self._update_time

    @property
    def refresh_time(self) -> tp.Optional[int]:
        """Timestamp indicating when the progress bar was last refreshed.

        Returns:
            Optional[int]: Unix timestamp when the progress bar was refreshed, or None if never refreshed.
        """
        return self._refresh_time

    @property
    def close_time(self) -> tp.Optional[int]:
        """Timestamp indicating when the progress bar was closed.

        Returns:
            Optional[int]: Unix timestamp when the progress bar was closed, or None if still active.
        """
        return self._close_time

    def set_bar(self, bar: tp.Optional[tqdm] = None) -> None:
        """Initialize the progress bar instance.

        Args:
            bar (Optional[tqdm]): Alternative progress bar instance to set.

                If not provided, a new progress bar is created.

        Returns:
            None
        """
        if bar is None:
            bar = self.bar_type(self.iterable, disable=not self.show_progress, **self.kwargs)
        self._bar = bar

    def remove_bar(self) -> None:
        """Remove the current progress bar instance.

        Returns:
            None
        """
        self._bar = None

    def reset(self) -> None:
        """Reset the progress bar instance by updating its configuration and refreshing parent progress bars.

        Returns:
            None
        """
        for pbar in self.registry.get_parent_instances(self):
            if not pbar.disabled and not pbar.displayed:
                pbar.refresh()  # refresh parents first
        self.bar.iterable = self.iterable
        self.bar.delay = 0
        self.bar.reset(total=self.kwargs.get("total", None))  # tqdm-forced refresh

    def open(self, reuse: tp.Optional[bool] = None) -> None:
        """Open the progress bar instance.

        Args:
            reuse (Optional[bool]): Flag indicating whether the progress bar can be reused.

        Returns:
            None
        """
        if self.bar is None:
            self._open_time = utc_time()
            if self.registry is None:
                self.set_bar()
            else:
                if reuse is None:
                    reuse = self.reuse
                if reuse and self.registry.has_conflict(self):
                    new_bar_id = self.registry.generate_bar_id()
                    if isinstance(self.bar_id, str):
                        bar_id = f"'{self.bar_id}'"
                    else:
                        bar_id = self.bar_id
                    if not self.silence_warnings:
                        warn(
                            f"Two active progress bars share the same bar id {bar_id}. "
                            f"Setting bar id of the new progress bar to '{new_bar_id}'."
                        )
                    self._bar_id = new_bar_id
                pending_bar = self.registry.get_pending_instance(self)
                if reuse and pending_bar is not None:
                    self.set_bar(bar=pending_bar.bar)
                    pending_bar.remove_bar()
                    self.registry.deregister_instance(pending_bar)
                    self.reset()
                else:
                    if self.bar_id is None:
                        self._bar_id = self.registry.generate_bar_id()
                    self.set_bar()
                self.registry.register_instance(self)

    def close(self, reuse: tp.Optional[bool] = None, close_children: bool = True) -> None:
        """Close the progress bar instance.

        Args:
            reuse (Optional[bool]): Flag indicating whether the progress bar can be reused.
            close_children (bool): Whether to close child progress bar instances.

        Returns:
            None
        """
        if self.bar is not None:
            self._close_time = utc_time()
            if self.registry is None:
                if not self.disabled:
                    self.bar.close()
                self.remove_bar()
            else:
                if reuse is None:
                    if not self.displayed:
                        reuse = False
                    else:
                        reuse = self.reuse
                parent_pbar = self.registry.get_parent_instance(self)
                if not reuse or parent_pbar is None:
                    if close_children:
                        for pbar in self.registry.get_child_instances(self):
                            pbar.close(reuse=False, close_children=False)
                    if not self.disabled:
                        self.bar.close()
                    self.remove_bar()
                    self.registry.deregister_instance(self)

    @property
    def active(self) -> bool:
        """Indicates whether the progress bar is active (i.e., created but not closed).

        Returns:
            bool: True if the progress bar instance exists and has not been closed; otherwise, False.
        """
        return self.bar is not None and self.close_time is None

    @property
    def pending(self) -> bool:
        """Indicates whether the progress bar has been created and subsequently closed.

        Returns:
            bool: True if the progress bar exists and has been closed (pending), otherwise False.
        """
        return self.bar is not None and self.close_time is not None

    @property
    def disabled(self) -> bool:
        """Indicates whether the progress bar is disabled.

        Returns:
            bool: True if progress display is disabled, either by configuration or
                due to the underlying bar state; False if the progress bar is enabled and open.

        Raises:
            ValueError: If the progress bar has not been opened when attempting to check its status.
        """
        if not self.show_progress:
            return True
        if self.bar is None:
            if self.bar_id is not None:
                if isinstance(self.bar_id, str):
                    bar_id = f"'{self.bar_id}'"
                else:
                    bar_id = self.bar_id
                raise ValueError(
                    f'Progress bar with bar id {bar_id} must be opened first. Use "with" statement.'
                )
            else:
                raise ValueError('Progress bar must be opened first. Use "with" statement.')
        if self.bar.disable:
            return True
        return False

    @property
    def displayed(self) -> bool:
        """Indicates whether the progress bar is currently displayed.

        Returns:
            bool: True if the progress bar is enabled and has been refreshed or
                immediately displayed (if delay is zero); otherwise, False.
        """
        if self.disabled:
            return False
        if self.refresh_time is not None:
            return True
        if self.bar.delay == 0:
            return True
        delay_end_t = self.bar.start_t + self.bar.delay
        return self.bar.last_print_t >= delay_end_t

    @property
    def should_display(self) -> bool:
        """Indicates whether the progress bar should be displayed based on its defined delay.

        Returns:
            bool: True if the progress bar is enabled and either no delay is set or
                the delay period has elapsed; otherwise, False.
        """
        if self.disabled:
            return False
        if self.bar.delay == 0:
            return True
        delay_end_t = self.bar.start_t + self.bar.delay
        if utc_time() >= delay_end_t:
            return True
        return False

    def refresh(self) -> None:
        """Refresh the progress bar.

        Returns:
            None
        """
        self.bar.refresh()
        self._refresh_time = utc_time()

    def before_update(self) -> None:
        """Execute necessary actions before updating the progress bar.

        Returns:
            None
        """
        if self.disabled:
            return
        if self.registry is not None:
            displayed = self.displayed
            should_display = self.should_display
            for pbar in self.registry.get_parent_instances(self):
                if not pbar.disabled and not pbar.displayed:
                    if (displayed or should_display) or pbar.should_display:
                        pbar.refresh()

    def update(self, n: int = 1) -> None:
        """Update the progress bar by the specified number of iterations.

        Args:
            n (int): Number of iterations to update.

        Returns:
            None
        """
        if self.disabled:
            return
        self.before_update()
        self.bar.update(n=n)
        self._update_time = utc_time()
        self.after_update()

    def update_to(self, n: int) -> None:
        """Update the progress bar to the given iteration count.

        Args:
            n (int): Target iteration count.

        Returns:
            None
        """
        if self.disabled:
            return
        self.update(n=n - self.bar.n)

    def after_update(self) -> None:
        """Execute necessary actions after updating the progress bar.

        Returns:
            None
        """
        if self.disabled:
            return

    @classmethod
    def format_num(cls, n: float) -> str:
        """Format the given number using up to three significant digits.

        Args:
            n (float): Number to format.

        Returns:
            str: Formatted string representation of the number.
                If the formatted string is shorter than the original, it is returned;
                otherwise, the original string is returned.
        """
        f = f"{n:.3g}".replace("+0", "+").replace("-0", "-")
        n = str(n)
        return f if len(f) < len(n) else n

    @classmethod
    def prepare_desc(cls, desc: tp.Union[None, str, dict]) -> str:
        """Prepare and format the description.

        If `desc` is a dictionary, its key-value pairs are concatenated into a comma-separated string.
        If `desc` is None or missing, an empty string is returned.

        Args:
            desc (Union[None, str, dict]): Description to prepare.

        Returns:
            str: Formatted description string.
        """
        if desc is None or desc is MISSING:
            return ""
        if isinstance(desc, dict):
            new_desc = []
            for k, v in desc.items():
                if v is MISSING:
                    if k not in (None, MISSING):
                        if not isinstance(k, str):
                            k = str(k)
                        k = k.strip()
                        new_desc.append(k)
                else:
                    if isinstance(v, Number):
                        v = cls.format_num(v)
                    if not isinstance(v, str):
                        v = str(v)
                    v = v.strip()
                    if k not in (None, MISSING):
                        if not isinstance(k, str):
                            k = str(k)
                        k = k.strip()
                        new_desc.append(k + "=" + v)
                    else:
                        new_desc.append(v)
            return ", ".join(new_desc)
        return str(desc)

    def set_prefix(
        self, desc: tp.Union[None, str, dict], refresh: tp.Optional[bool] = None
    ) -> None:
        """Set the progress bar's prefix.

        The description is prepared using `ProgressBar.prepare_desc` before being set.

        Args:
            desc (Union[None, str, dict]): Description to set as prefix.
            refresh (Optional[bool]): Determines whether to refresh the bar after setting the prefix.

        Returns:
            None
        """
        if self.disabled or not self.show_progress_desc:
            return
        desc = self.prepare_desc(desc)
        self.bar.set_description_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_prefix_str(self, desc: str, refresh: tp.Optional[bool] = None) -> None:
        """Set the progress bar's prefix without preparing the description.

        Args:
            desc (str): Prefix string to set.
            refresh (Optional[bool]): Determines whether to refresh the bar after setting the prefix.

        Returns:
            None
        """
        if self.disabled or not self.show_progress_desc:
            return
        self.bar.set_description_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_postfix(
        self, desc: tp.Union[None, str, dict], refresh: tp.Optional[bool] = None
    ) -> None:
        """Set the progress bar's postfix.

        The description is prepared using `ProgressBar.prepare_desc` before being set.

        Args:
            desc (Union[None, str, dict]): Description to set as postfix.
            refresh (Optional[bool]): Determines whether to refresh the bar after setting the postfix.

        Returns:
            None
        """
        if self.disabled or not self.show_progress_desc:
            return
        desc = self.prepare_desc(desc)
        self.bar.set_postfix_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_postfix_str(self, desc: str, refresh: tp.Optional[bool] = None) -> None:
        """Set the progress bar's postfix without preparing the description.

        Args:
            desc (str): Postfix string to set.
            refresh (Optional[bool]): Determines whether to refresh the bar after setting the postfix.

        Returns:
            None
        """
        if self.disabled or not self.show_progress_desc:
            return
        self.bar.set_postfix_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_description(
        self,
        desc: tp.Union[None, str, dict],
        as_postfix: tp.Optional[bool] = None,
        refresh: tp.Optional[bool] = None,
    ) -> None:
        """Set the progress bar's description.

        Depending on the `as_postfix` setting in `ProgressBar.desc_kwargs`, this method calls either
        `ProgressBar.set_postfix` (if True) or `ProgressBar.set_prefix` (if False).

        Args:
            desc (Union[None, str, dict]): Description to set.
            as_postfix (Optional[bool]): Determines whether to set the description as postfix.
            refresh (Optional[bool]): Determines whether to refresh the bar after setting the description.

        Returns:
            None
        """
        if self.disabled or not self.show_progress_desc:
            return
        if as_postfix is None:
            as_postfix = self.desc_kwargs.get("as_postfix", True)
        if as_postfix:
            self.set_postfix(desc, refresh=refresh)
        else:
            self.set_prefix(desc, refresh=refresh)

    def set_description_str(
        self,
        desc: str,
        as_postfix: tp.Optional[bool] = None,
        refresh: tp.Optional[bool] = None,
    ) -> None:
        """Set the progress bar's description without preparing the provided string.

        Depending on the `as_postfix` setting in `ProgressBar.desc_kwargs`, this method calls either
        `ProgressBar.set_postfix_str` (if True) or `ProgressBar.set_prefix_str` (if False).

        Args:
            desc (str): Description string to set.
            as_postfix (Optional[bool]): Determines whether to set the description as postfix.
            refresh (Optional[bool]): Determines whether to refresh the bar after setting the description.

        Returns:
            None
        """
        if self.disabled or not self.show_progress_desc:
            return
        if as_postfix is None:
            as_postfix = self.desc_kwargs.get("as_postfix", True)
        if as_postfix:
            self.set_postfix_str(desc, refresh=refresh)
        else:
            self.set_prefix_str(desc, refresh=refresh)

    def enter(self, **kwargs) -> tp.Self:
        """Enter the progress bar context.

        If configured to show progress or forced to open, the progress bar is opened with
        the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments for `ProgressBar.open`.

        Returns:
            Self: Progress bar instance.
        """
        if self.show_progress or self.force_open_bar:
            self.open(**kwargs)
        return self

    def __enter__(self) -> tp.Self:
        return self.enter()

    def exit(self, **kwargs) -> None:
        """Exit the progress bar.

        Args:
            **kwargs: Keyword arguments for `ProgressBar.close`.

        Returns:
            None
        """
        self.close(**kwargs)

    def __exit__(self, *args) -> None:
        self.exit()

    def __len__(self) -> int:
        return self.bar.__len__()

    def __contains__(self, item: tp.Any) -> bool:
        return self.bar.__contains__(item)

    def iter(self) -> tp.Iterator:
        """Return an iterator over the progress bar's iterable.

        Yields:
            Any: Next object in the iterable.
        """
        for obj in self.iterable:
            yield obj

    def __iter__(self) -> tp.Iterator:
        if self.bar is None:
            if self.show_progress or self.force_open_bar:
                self.open()
        if self.disabled:
            return self.iter()
        for i, obj in enumerate(self.bar.__iter__()):
            if i > 0:
                self._update_time = utc_time()
                self.after_update()
            yield obj
            self.before_update()
        self._update_time = utc_time()
        self.after_update()
        self.close()

    def __del__(self) -> None:
        self.close()


class ProgressHidden(Base):
    """Context manager to temporarily hide progress globally.

    Args:
        disable_registry (bool): Flag to disable registry operations.
        disable_machinery (bool): Flag to disable progress machinery.

    Returns:
        None

    !!! info
        For default settings, see `vectorbtpro._settings.pbar`.
    """

    def __init__(self, disable_registry: bool = True, disable_machinery: bool = True) -> None:
        self._disable_registry = disable_registry
        self._disable_machinery = disable_machinery
        self._init_settings = None

    @property
    def disable_registry(self) -> bool:
        """Indicates whether registry operations are disabled.

        Returns:
            bool: True if registry operations are disabled; otherwise, False.
        """
        return self._disable_registry

    @property
    def disable_machinery(self) -> bool:
        """Indicates whether progress machinery is disabled.

        Returns:
            bool: True if progress machinery is disabled; otherwise, False.
        """
        return self._disable_machinery

    @property
    def init_settings(self) -> tp.Kwargs:
        """Dictionary containing the initial progress settings captured upon entering the context.

        Returns:
            Kwargs: Dictionary of keyword arguments containing the initial progress settings.
        """
        return self._init_settings

    def __enter__(self) -> tp.Self:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        self._init_settings = dict(
            disable=pbar_cfg["disable"],
            disable_registry=pbar_cfg["disable_registry"],
            disable_machinery=pbar_cfg["disable_machinery"],
        )

        pbar_cfg["disable"] = True
        pbar_cfg["disable_registry"] = self.disable_registry
        pbar_cfg["disable_machinery"] = self.disable_machinery

        return self

    def __exit__(self, *args) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        pbar_cfg["disable"] = self.init_settings["disable"]
        pbar_cfg["disable_registry"] = self.init_settings["disable_registry"]
        pbar_cfg["disable_machinery"] = self.init_settings["disable_machinery"]


def with_progress_hidden(*args) -> tp.Callable:
    """Return a decorator that runs a function within a `ProgressHidden` context.

    Args:
        func (Callable): Function to be decorated.

    Returns:
        Callable: Decorated function that executes with progress hidden.
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with ProgressHidden():
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


class ProgressShown(Base):
    """Context manager to temporarily show progress globally.

    Args:
        enable_registry (bool): Flag to enable registry operations.
        enable_machinery (bool): Flag to enable progress machinery.

    !!! info
        For default settings, see `vectorbtpro._settings.pbar`.
    """

    def __init__(self, enable_registry: bool = True, enable_machinery: bool = True) -> None:
        self._enable_registry = enable_registry
        self._enable_machinery = enable_machinery
        self._init_settings = None

    @property
    def enable_registry(self) -> bool:
        """Indicates whether registry operations are enabled.

        Returns:
            bool: True if registry operations are enabled; otherwise, False.
        """
        return self._enable_registry

    @property
    def enable_machinery(self) -> bool:
        """Indicates whether progress machinery is enabled.

        Returns:
            bool: True if progress machinery is enabled; otherwise, False.
        """
        return self._enable_machinery

    @property
    def init_settings(self) -> tp.Kwargs:
        """Dictionary containing the initial progress settings captured upon entering the context.

        Returns:
            Kwargs: Dictionary of keyword arguments containing the initial progress settings.
        """
        return self._init_settings

    def __enter__(self) -> tp.Self:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        self._init_settings = dict(
            disable=pbar_cfg["disable"],
            disable_registry=pbar_cfg["disable_registry"],
            disable_machinery=pbar_cfg["disable_machinery"],
        )

        pbar_cfg["disable"] = False
        pbar_cfg["disable_registry"] = not self.enable_registry
        pbar_cfg["disable_machinery"] = not self.enable_machinery

        return self

    def __exit__(self, *args) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        pbar_cfg["disable"] = self.init_settings["disable"]
        pbar_cfg["disable_registry"] = self.init_settings["disable_registry"]
        pbar_cfg["disable_machinery"] = self.init_settings["disable_machinery"]


def with_progress_shown(*args) -> tp.Callable:
    """Return a decorator that runs a function within a `ProgressShown` context.

    Args:
        func (Callable): Function to be decorated.

    Returns:
        Callable: Decorated function that executes with progress shown.
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with ProgressShown():
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
