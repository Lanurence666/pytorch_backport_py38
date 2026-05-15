"""Hook system for Dynamo's guard functionality.

This module provides a way to register callback functions that are triggered during
guard-related operations.

The Hooks class manages two types of hook functions:
- guard_export_fn: Called when guards need to be exported, taking a GuardsSet as input
- guard_fail_fn: Called when a guard check fails, taking a GuardFail object as input
These hooks enable customization of guard export and failure handling behaviors.
"""

from typing import Callable, Optional, Sequence, Set
import dataclasses


from torch._guards import GuardsSet

from .types import GuardFail, GuardFilterEntry


@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[GuardsSet], None]]= None
    guard_fail_fn: Optional[Callable[[GuardFail], None]]= None
    guard_filter_fn: Optional[Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]]= (
        None
    )
