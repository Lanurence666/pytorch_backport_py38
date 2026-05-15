from __future__ import annotations

import os
import site
import sys

import torch
from typing import List


def _prefix_regex() -> List[str]:
    raw_paths = (
        site.getsitepackages()
        + sys.path
        + [site.getuserbase()]
        + [site.getusersitepackages()]
        + [os.path.dirname(os.path.dirname(torch.__file__))]
    )

    path_prefixes = sorted({os.path.abspath(i) for i in raw_paths}, reverse=True)
    if not all(isinstance(i, str) for i in path_prefixes):
        raise AssertionError("all path_prefixes must be strings")
    return [i + os.sep for i in path_prefixes]
