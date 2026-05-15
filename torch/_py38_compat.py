from __future__ import annotations

import sys

_PY38 = sys.version_info < (3, 9)


def _removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def _removesuffix(s, suffix):
    if s.endswith(suffix):
        return s[:-len(suffix)] if suffix else s
    return s


def _patch_str_methods():
    if not _PY38:
        return

    try:
        if not hasattr(str, "removeprefix"):
            str.removeprefix = _removeprefix
    except TypeError:
        pass

    try:
        if not hasattr(str, "removesuffix"):
            str.removesuffix = _removesuffix
    except TypeError:
        pass


def _patch_importlib_metadata():
    pass


def apply_patches():
    _patch_str_methods()
    _patch_importlib_metadata()


if sys.version_info < (3, 9):
    apply_patches()
