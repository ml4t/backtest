# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes for scheduling data updates."""

import logging

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import Data
from vectorbtpro.utils.config import Configured, merge_dicts
from vectorbtpro.utils.schedule_ import ScheduleManager

__all__ = [
    "DataUpdater",
]

logger = logging.getLogger(__name__)


class DataUpdater(Configured):
    """Class for scheduling data updates.

    Args:
        data (Data): Data instance.
        update_kwargs (KwargsLike): Default configuration for `DataSaver.update`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(
        self,
        data: Data,
        schedule_manager: tp.Optional[ScheduleManager] = None,
        update_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        if schedule_manager is None:
            schedule_manager = ScheduleManager()
        Configured.__init__(
            self,
            data=data,
            schedule_manager=schedule_manager,
            update_kwargs=update_kwargs,
            **kwargs,
        )

        self._data = data
        self._schedule_manager = schedule_manager
        self._update_kwargs = update_kwargs

    @property
    def data(self) -> Data:
        """Data instance associated with the updater.

        Returns:
            Data: Instance of `vectorbtpro.data.base.Data`.
        """
        return self._data

    @property
    def schedule_manager(self) -> ScheduleManager:
        """Schedule manager instance used for scheduling update jobs.

        Returns:
            ScheduleManager: Instance of `vectorbtpro.utils.schedule_.ScheduleManager`.
        """
        return self._schedule_manager

    @property
    def update_kwargs(self) -> tp.KwargsLike:
        """Default configuration for `DataSaver.update`.

        These arguments are merged with those provided during scheduled updates.

        Returns:
            KwargsLike: Default configuration for data updates.
        """
        return self._update_kwargs

    def update(self, **kwargs) -> None:
        """Update the data instance.

        Override this method to incorporate any pre- or post-processing steps during the update.

        To cancel further scheduled updates, raise `vectorbtpro.utils.schedule_.CancelledError`.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.data.base.Data.update`.

        Returns:
            None
        """
        # In case the method was called by the user
        kwargs = merge_dicts(self.update_kwargs, kwargs)

        self._data = self.data.update(**kwargs)
        self.update_config(data=self.data)
        new_index = self.data.wrapper.index
        logger.info(f"New data has {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def update_every(
        self,
        *args,
        to: tp.Optional[int] = None,
        tags: tp.Optional[tp.Iterable[tp.Hashable]] = None,
        in_background: bool = False,
        replace: bool = True,
        start: bool = True,
        start_kwargs: tp.KwargsLike = None,
        **update_kwargs,
    ) -> None:
        """Schedule `DataUpdater.update` as a periodic job.

        For details on `*args`, `to`, and `tags`, refer to `vectorbtpro.utils.schedule_.ScheduleManager.every`.

        Args:
            *args: Positional arguments for `vectorbtpro.utils.schedule_.ScheduleManager.every`.
            to (int): Upper time boundary for scheduling updates.
            tags (Optional[Iterable[Hashable]]): Tags used to identify the scheduled job.
            in_background (bool): If True, the job runs as an asyncio task and
                can be stopped using `vectorbtpro.utils.schedule_.ScheduleManager.stop`
            replace (bool): If True, remove existing jobs with matching tags.
            start (bool): If True, start the job immediately after scheduling.
            start_kwargs (KwargsLike): Keyword arguments for starting the scheduler.

                See `vectorbtpro.utils.schedule_.ScheduleManager.start`.
            **update_kwargs: Keyword arguments for `vectorbtpro.utils.schedule_.AsyncJob.do`.

        Returns:
            None
        """
        if replace:
            self.schedule_manager.clear_jobs(tags)
        update_kwargs = merge_dicts(self.update_kwargs, update_kwargs)
        self.schedule_manager.every(*args, to=to, tags=tags).do(self.update, **update_kwargs)
        if start:
            if start_kwargs is None:
                start_kwargs = {}
            if in_background:
                self.schedule_manager.start_in_background(**start_kwargs)
            else:
                self.schedule_manager.start(**start_kwargs)
