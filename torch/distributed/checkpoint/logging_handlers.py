from __future__ import annotations

import logging

from torch.distributed.logging_handlers import _log_handlers
from typing import List


__all__: List[str] = []

DCP_LOGGER_NAME = "dcp_logger"

_log_handlers.update(
    {
        DCP_LOGGER_NAME: logging.NullHandler(),
    }
)
