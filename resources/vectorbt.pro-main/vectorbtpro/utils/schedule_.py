# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for scheduling jobs."""

import asyncio
import inspect
import logging
import random
import time
from datetime import datetime, timedelta
from datetime import time as dt_time

from schedule import CancelJob, Job, ScheduleError, Scheduler

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.base import Base

__all__ = [
    "AsyncJob",
    "AsyncScheduler",
    "CancelledError",
    "ScheduleManager",
]

logger = logging.getLogger(__name__)


class CustomScheduler(Scheduler, Base):
    """Class for custom scheduling."""

    def __init__(self) -> None:
        super(CustomScheduler, self).__init__()


CustomJobT = tp.TypeVar("CustomJobT", bound="CustomJob")


class CustomJob(Job, Base):
    """Class for custom job scheduling.

    Args:
        interval (int): Interval between job executions.
        scheduler (Optional[Scheduler]): Scheduler instance managing the job.
    """

    def __init__(self, interval: int, scheduler: tp.Optional[Scheduler] = None) -> None:
        super(CustomJob, self).__init__(interval, scheduler)
        self._zero_offset = False
        self._force_missed_run = False
        self.future_run = None

    @property
    def zero_offset(self: CustomJobT) -> CustomJobT:
        """Job instance with zero offset scheduling enabled.

        Returns:
            CustomJob: Current job instance with zero offset enabled.
        """
        self._zero_offset = True
        return self

    @property
    def force_missed_run(self: CustomJobT) -> CustomJobT:
        """Job instance with forced missed run scheduling enabled.

        Returns:
            CustomJob: Current job instance with forced missed run enabled.
        """
        self._force_missed_run = True
        return self

    @property
    def modulo(self) -> int:
        """Remainder of the next scheduled run time's corresponding unit divided by the interval.

        Returns:
            int: Remainder computed based on the time unit of the next run.
        """
        if self.unit == "seconds":
            return self.next_run.second % self.interval
        if self.unit == "minutes":
            return self.next_run.minute % self.interval
        if self.unit == "hours":
            return self.next_run.hour % self.interval
        if self.unit == "days":
            return self.next_run.day % self.interval

    def _schedule_next_run(self) -> None:
        super(CustomJob, self)._schedule_next_run()

        if self.latest is None and self._zero_offset:
            if self.modulo != 0:
                self.next_run -= timedelta(**{self.unit: self.modulo})

        if self.future_run and self.future_run < self.next_run and self._force_missed_run:
            self.next_run, self.future_run = self.future_run, self.next_run
        else:
            if self.latest is not None:
                if not (self.latest >= self.interval):
                    raise ScheduleError("`latest` is greater than `interval`")
                interval = random.randint(self.interval, self.latest)
            else:
                interval = self.interval
            period = timedelta(**{self.unit: interval})
            self.future_run = self.next_run + period


class CancelledError(asyncio.CancelledError):
    """Exception indicating that an operation has been cancelled."""

    pass


class AsyncJob(CustomJob):
    """Class for asynchronous custom jobs."""

    async def async_run(self) -> tp.Any:
        """Asynchronously execute the job function.

        Runs the job function, updates the last run timestamp, and schedules the next run.

        Returns:
            Any: Result returned by the job function.
        """
        logger.info("Running job %s", self)
        ret = self.job_func()
        if inspect.isawaitable(ret):
            ret = await ret
        self.last_run = datetime.now()
        self._schedule_next_run()
        return ret


class AsyncScheduler(CustomScheduler):
    """Class for asynchronous custom scheduling."""

    async def async_run_pending(self) -> None:
        """Asynchronously run all pending jobs.

        Identifies jobs that are ready to run and concurrently executes them.

        Returns:
            None
        """
        runnable_jobs = (job for job in self.jobs if job.should_run)
        await asyncio.gather(*[self._async_run_job(job) for job in runnable_jobs])

    async def async_run_all(self, delay_seconds: int = 0) -> None:
        """Asynchronously execute all scheduled jobs with an optional delay between jobs.

        Args:
            delay_seconds (int): Delay in seconds between consecutive job executions.

        Returns:
            None
        """
        logger.info(
            "Running *all* %i jobs with %is delay in-between", len(self.jobs), delay_seconds
        )
        for job in self.jobs[:]:
            await self._async_run_job(job)
            await asyncio.sleep(delay_seconds)

    async def _async_run_job(self, job: AsyncJob) -> None:
        """Asynchronously execute a job.

        Asynchronous version of `CustomScheduler.run_job`.

        Args:
            job (AsyncJob): Asynchronous job.

        Returns:
            None
        """
        ret = await job.async_run()
        if isinstance(ret, CancelJob) or ret is CancelJob:
            self.cancel_job(job)

    def every(self, interval: int = 1) -> AsyncJob:
        """Schedule a new periodic asynchronous job.

        Args:
            interval (int): Interval between job executions.

        Returns:
            AsyncJob: Newly scheduled asynchronous job.
        """
        job = AsyncJob(interval, self)
        return job


class ScheduleManager(Base):
    """Class for managing `CustomScheduler` jobs.

    Args:
        scheduler (Optional[AsyncScheduler]): Scheduler instance to be used.

            If not provided, an `AsyncScheduler` instance is created.
    """

    units: tp.ClassVar[tp.Tuple[str, ...]] = (
        "second",
        "seconds",
        "minute",
        "minutes",
        "hour",
        "hours",
        "day",
        "days",
        "week",
        "weeks",
    )
    """Time units accepted by the scheduler."""

    weekdays: tp.ClassVar[tp.Tuple[str, ...]] = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )
    """Valid weekdays for scheduling jobs."""

    def __init__(self, scheduler: tp.Optional[AsyncScheduler] = None) -> None:
        if scheduler is None:
            scheduler = AsyncScheduler()
        checks.assert_instance_of(scheduler, AsyncScheduler)

        self._scheduler = scheduler
        self._async_task = None

    @property
    def scheduler(self) -> AsyncScheduler:
        """Scheduler instance used for scheduling jobs.

        Returns:
            AsyncScheduler: Scheduler instance.
        """
        return self._scheduler

    @property
    def async_task(self) -> tp.Optional[asyncio.Task]:
        """Current asynchronous task, if any.

        Returns:
            Optional[asyncio.Task]: Current asynchronous task.
        """
        return self._async_task

    def every(
        self,
        *args,
        to: tp.Optional[int] = None,
        zero_offset: bool = False,
        force_missed_run: bool = False,
        tags: tp.Optional[tp.Iterable[tp.Hashable]] = None,
    ) -> AsyncJob:
        """Create a new asynchronous job that runs at a specified interval.

        Positional arguments determine scheduling parameters in a strict order:

        * interval: int or timedelta specifying the time interval.
        * unit: str from `ScheduleManager.units` indicating the time unit.
        * start_day: str from `ScheduleManager.weekdays` indicating the starting weekday.
        * at: str or datetime.time specifying the execution time.

        This method utilizes the `schedule` package for job scheduling.

        Args:
            *args: Positional arguments for specifying scheduling parameters.

                They must be provided in the strict order:

                * interval: int or timedelta specifying the time interval.
                * unit: str from `ScheduleManager.units` indicating the time unit.
                * start_day: str from `ScheduleManager.weekdays` indicating the starting weekday.
                * at: str or datetime.time specifying a specific execution time.
            to (Optional[int]): Specifies an end parameter for the schedule.
            zero_offset (bool): When True, configures the job to use a zero offset.
            force_missed_run (bool): When True, forces the job to run if a scheduled execution was missed.
            tags (Optional[Iterable[Hashable]]): Tags used to identify the scheduled job.

        Returns:
            AsyncJob: Configured asynchronous job.

        Examples:
            ```pycon
            >>> from vectorbtpro import *

            >>> def job_func(message="I'm working..."):
            ...     print(message)

            >>> my_manager = vbt.ScheduleManager()

            >>> my_manager.every().do(job_func, message="Hello")
            Every 1 second do job_func(message='Hello') (last run: [never], next run: 2021-03-18 19:06:47)

            >>> my_manager.every(10, 'minutes').do(job_func)
            Every 10 minutes do job_func() (last run: [never], next run: 2021-03-18 19:16:46)

            >>> my_manager.every(10, 'minutes', ':00', zero_offset=True).do(job_func)
            Every 10 minutes at 00:00:00 do job_func() (last run: [never], next run: 2022-08-18 16:10:00)

            >>> my_manager.every('hour').do(job_func)
            Every 1 hour do job_func() (last run: [never], next run: 2021-03-18 20:06:46)

            >>> my_manager.every('hour', '00:00').do(job_func)
            Every 1 hour at 00:00:00 do job_func() (last run: [never], next run: 2021-03-18 20:00:00)

            >>> my_manager.every(4, 'hours', '00:00').do(job_func)
            Every 4 hours at 00:00:00 do job_func() (last run: [never], next run: 2021-03-19 00:00:00)

            >>> my_manager.every('10:30').do(job_func)
            Every 1 day at 10:30:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

            >>> my_manager.every('hour', '00:00').do(job_func)
            Every 1 hour at 00:00:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

            >>> my_manager.every(4, 'hour', '00:00').do(job_func)
            Every 4 hours at 00:00:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

            >>> my_manager.every('day', '10:30').do(job_func)
            Every 1 day at 10:30:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

            >>> my_manager.every('day', time(9, 30, tzinfo="utc")).do(job_func)
            Every 1 day at 10:30:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

            >>> my_manager.every('monday').do(job_func)
            Every 1 week do job_func() (last run: [never], next run: 2021-03-22 19:06:46)

            >>> my_manager.every('wednesday', '13:15').do(job_func)
            Every 1 week at 13:15:00 do job_func() (last run: [never], next run: 2021-03-24 13:15:00)

            >>> my_manager.every('minute', ':17').do(job_func)
            Every 1 minute at 00:00:17 do job_func() (last run: [never], next run: 2021-03-18 19:07:17)

            >>> my_manager.start()
            ```

            You can still use the chained approach as done by `schedule`:

            ```pycon
            >>> my_manager.every().minute.at(':17').do(job_func)
            Every 1 minute at 00:00:17 do job_func() (last run: [never], next run: 2021-03-18 19:07:17)
            ```
        """
        # Parse arguments
        interval = 1
        unit = None
        start_day = None
        at = None

        def _is_arg_interval(arg):
            return isinstance(arg, (int, timedelta))

        def _is_arg_unit(arg):
            return isinstance(arg, str) and arg in self.units

        def _is_arg_start_day(arg):
            return isinstance(arg, str) and arg in self.weekdays

        def _is_arg_at(arg):
            return (isinstance(arg, str) and ":" in arg) or isinstance(arg, dt_time)

        expected_args = ["interval", "unit", "start_day", "at"]
        for i, arg in enumerate(args):
            if "interval" in expected_args and _is_arg_interval(arg):
                interval = arg
                expected_args = expected_args[expected_args.index("interval") + 1 :]
                continue
            if "unit" in expected_args and _is_arg_unit(arg):
                unit = arg
                expected_args = expected_args[expected_args.index("unit") + 1 :]
                continue
            if "start_day" in expected_args and _is_arg_start_day(arg):
                start_day = arg
                expected_args = expected_args[expected_args.index("start_day") + 1 :]
                continue
            if "at" in expected_args and _is_arg_at(arg):
                at = arg
                expected_args = expected_args[expected_args.index("at") + 1 :]
                continue
            raise ValueError(f"Arg at index {i} is unexpected")

        if at is not None:
            if unit is None and start_day is None:
                unit = "days"
        if unit is None and start_day is None:
            unit = "seconds"

        job = self.scheduler.every(interval)
        if unit is not None:
            job = getattr(job, unit)
        if start_day is not None:
            job = getattr(job, start_day)
        if at is not None:
            if isinstance(at, dt_time):
                if job.unit == "days" or job.start_day:
                    if at.tzinfo is not None:
                        at = dt.tzaware_to_naive_time(at, None)
                at = at.isoformat()
                if job.unit == "hours":
                    at = ":".join(at.split(":")[1:])
                if job.unit == "minutes":
                    at = ":" + at.split(":")[2]
            job = job.at(at)
        if to is not None:
            job = job.to(to)
        if tags is not None:
            if not isinstance(tags, tuple):
                tags = (tags,)
            job = job.tag(*tags)
        if zero_offset:
            job = job.zero_offset
        if force_missed_run:
            job = job.force_missed_run

        return job

    def start(self, sleep: int = 1, clear_after: bool = False) -> None:
        """Run pending jobs in a loop.

        Args:
            sleep (int): Time in seconds to sleep between job checks.
            clear_after (bool): Clear scheduled jobs after stopping if True.

        Returns:
            None
        """
        logger.info("Starting schedule manager with jobs %s", str(self.scheduler.jobs))
        try:
            while True:
                self.scheduler.run_pending()
                time.sleep(sleep)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Stopping schedule manager")
        if clear_after:
            self.scheduler.clear()

    async def async_start(self, sleep: int = 1, clear_after: bool = False) -> None:
        """Run pending jobs in a loop asynchronously.

        Args:
            sleep (int): Time in seconds to sleep between job checks.
            clear_after (bool): Clear scheduled jobs after stopping if True.

        Returns:
            None
        """
        logger.info(
            "Starting schedule manager in the background with jobs %s", str(self.scheduler.jobs)
        )
        logger.info("Jobs: %s", str(self.scheduler.jobs))
        try:
            while True:
                await self.scheduler.async_run_pending()
                await asyncio.sleep(sleep)
        except asyncio.CancelledError:
            logger.info("Stopping schedule manager")
        if clear_after:
            self.scheduler.clear()

    def done_callback(self, async_task: asyncio.Task) -> None:
        """Handle completion of an asynchronous task.

        Args:
            async_task (Task): Asynchronous task that has completed.

        Returns:
            None
        """
        logger.info(async_task)

    def start_in_background(self, **kwargs) -> None:
        """Start the asynchronous schedule manager in the background.

        Args:
            **kwargs: Keyword arguments for `ScheduleManager.async_start`.

        Returns:
            None
        """
        async_task = asyncio.create_task(self.async_start(**kwargs))
        async_task.add_done_callback(self.done_callback)
        logger.info(async_task)
        self._async_task = async_task

    @property
    def async_task_running(self) -> bool:
        """Indicates whether the asynchronous task is currently running.

        Returns:
            bool: True if the asynchronous task is running, False otherwise.
        """
        return self.async_task is not None and not self.async_task.done()

    def stop(self) -> None:
        """Stop the asynchronous task if it is running.

        Returns:
            None
        """
        if self.async_task_running:
            self.async_task.cancel()

    def clear_jobs(self, tags: tp.Optional[tp.Iterable[tp.Hashable]] = None) -> None:
        """Delete scheduled jobs filtered by given tags, or all jobs if no tags are provided.

        Args:
            tags (Optional[Iterable[Hashable]]): Tags used to identify the scheduled job.

        Returns:
            None
        """
        if tags is None:
            self.scheduler.clear()
        else:
            tags = set(tags)
            logger.debug('Deleting all jobs tagged "%s"', tags)
            self.scheduler.jobs[:] = (job for job in self.scheduler.jobs if tags == job.tags)
