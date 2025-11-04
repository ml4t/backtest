# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a global registry for managing progress bars."""

import uuid

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

if tp.TYPE_CHECKING:
    from vectorbtpro.utils.pbar import ProgressBar as ProgressBarT
else:
    ProgressBarT = "ProgressBar"

__all__ = [
    "PBarRegistry",
    "pbar_reg",
]


class PBarRegistry(Base):
    """Class for registering and managing `vectorbtpro.utils.pbar.ProgressBar`
    instances by their unique bar ids."""

    @classmethod
    def generate_bar_id(cls) -> tp.Hashable:
        """Generate a unique progress bar identifier.

        Returns:
            Hashable: Unique identifier for a progress bar.
        """
        return str(uuid.uuid4())

    def __init__(self):
        self._instances = {}

    @property
    def instances(self) -> tp.Dict[tp.Hashable, ProgressBarT]:
        """Dictionary of registered progress bar instances.

        Returns:
            Dict[Hashable, ProgressBar]: Dictionary mapping unique bar ids to their
                corresponding progress bar instances.
        """
        return self._instances

    def register_instance(self, instance: ProgressBarT) -> None:
        """Register a progress bar instance.

        Args:
            instance (ProgressBar): Progress bar instance to be registered.

        Returns:
            None
        """
        self.instances[instance.bar_id] = instance

    def deregister_instance(self, instance: ProgressBarT) -> None:
        """Deregister a progress bar instance.

        Args:
            instance (ProgressBar): Progress bar instance to be deregistered.

        Returns:
            None
        """
        if instance.bar_id in self.instances:
            del self.instances[instance.bar_id]

    def has_conflict(self, instance: ProgressBarT) -> bool:
        """Check for an active progress bar instance conflict.

        Args:
            instance (ProgressBar): Progress bar instance to check for conflicts.

        Returns:
            bool: True if another active progress bar with the same bar id exists, otherwise False.
        """
        if instance.bar_id is None:
            return False
        for inst in self.instances.values():
            if inst is not instance and inst.bar_id == instance.bar_id and inst.active:
                return True
        return False

    def get_last_active_instance(self) -> tp.Optional[ProgressBarT]:
        """Return the last active progress bar instance.

        Returns:
            Optional[ProgressBar]: Most recently active progress bar instance,
                or None if no active instance exists.
        """
        max_open_time = None
        last_active_instance = None
        for inst in self.instances.values():
            if inst.active:
                if max_open_time is None or inst.open_time > max_open_time:
                    max_open_time = inst.open_time
                    last_active_instance = inst
        return last_active_instance

    def get_first_pending_instance(self) -> tp.Optional[ProgressBarT]:
        """Return the first pending progress bar instance opened after the last active instance.

        Returns:
            Optional[ProgressBar]: First pending progress bar instance if found, otherwise None.
        """
        last_active_instance = self.get_last_active_instance()
        if last_active_instance is None:
            return None
        min_open_time = None
        first_pending_instance = None
        for inst in self.instances.values():
            if inst.pending and inst.open_time > last_active_instance.open_time:
                if min_open_time is None or inst.open_time < min_open_time:
                    min_open_time = inst.open_time
                    first_pending_instance = inst
        return first_pending_instance

    def get_pending_instance(self, instance: ProgressBarT) -> tp.Optional[ProgressBarT]:
        """Return a pending progress bar instance corresponding to the given instance.

        If the instance has a non-None bar id, this method searches for another pending
        instance with the same id. If the bar id is None, it returns the first pending progress
        bar instance that was opened after the last active instance.

        Args:
            instance (ProgressBar): Progress bar instance to search a pending instance for.

        Returns:
            Optional[ProgressBar]: Matching pending progress bar instance if found, otherwise None.
        """
        if instance.bar_id is not None:
            for inst in self.instances.values():
                if inst is not instance and inst.pending:
                    if inst.bar_id == instance.bar_id:
                        return inst
            return None
        last_active_instance = self.get_last_active_instance()
        if last_active_instance is None:
            return None
        min_open_time = None
        first_pending_instance = None
        for inst in self.instances.values():
            if inst.pending and inst.open_time > last_active_instance.open_time:
                if min_open_time is None or inst.open_time < min_open_time:
                    min_open_time = inst.open_time
                    first_pending_instance = inst
        return first_pending_instance

    def get_parent_instances(self, instance: ProgressBarT) -> tp.List[ProgressBarT]:
        """Return active parent progress bar instances of the given instance.

        Args:
            instance (ProgressBar): Progress bar instance whose parent instances are to be retrieved.

        Returns:
            List[ProgressBar]: List of active parent progress bar instances.
        """
        parent_instances = []
        for inst in self.instances.values():
            if inst is not instance and inst.active:
                if inst.open_time < instance.open_time:
                    parent_instances.append(inst)
        return parent_instances

    def get_parent_instance(self, instance: ProgressBarT) -> tp.Optional[ProgressBarT]:
        """Return the most recent active parent progress bar instance of the given instance.

        Args:
            instance (ProgressBar): Progress bar instance for which to find the parent instance.

        Returns:
            Optional[ProgressBar]: Active parent progress bar instance with the latest open time,
            or None if none exist.
        """
        max_open_time = None
        parent_instance = None
        for inst in self.get_parent_instances(instance):
            if max_open_time is None or inst.open_time > max_open_time:
                max_open_time = inst.open_time
                parent_instance = inst
        return parent_instance

    def get_child_instances(self, instance: ProgressBarT) -> tp.List[ProgressBarT]:
        """Return active or pending child progress bar instances of the given instance.

        Args:
            instance (ProgressBar): Progress bar instance whose child instances are to be retrieved.

        Returns:
            List[ProgressBar]: List of child progress bar instances.
        """
        child_instances = []
        for inst in self.instances.values():
            if inst is not instance and (inst.active or inst.pending):
                if inst.open_time > instance.open_time:
                    child_instances.append(inst)
        return child_instances

    def clear_instances(self) -> None:
        """Clear all registered progress bar instances.

        Returns:
            None
        """
        self.instances.clear()


pbar_reg = PBarRegistry()
"""Default registry instance of `PBarRegistry` for managing progress bar instances."""
