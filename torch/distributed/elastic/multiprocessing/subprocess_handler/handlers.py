# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler import (
    SubprocessHandler,
)
from torch.numa.binding import NumaOptions
from typing import Dict, Optional


__all__ = ["get_subprocess_handler"]


def get_subprocess_handler(
    entrypoint: str,
    args: tuple,
    env: Dict[str, str],
    stdout: str,
    stderr: str,
    local_rank_id: int,
    numa_options: Optional[NumaOptions] = None
) -> SubprocessHandler:
    return SubprocessHandler(
        entrypoint=entrypoint,
        args=args,
        env=env,
        stdout=stdout,
        stderr=stderr,
        local_rank_id=local_rank_id,
        numa_options=numa_options,
    )
