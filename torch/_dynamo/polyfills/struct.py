from __future__ import annotations
"""
Python polyfills for struct
"""


import struct
from typing import Any, Tuple, Union
from typing_extensions import Buffer

from ..decorators import substitute_in_graph


__all__ = [
    "pack",
    "unpack",
]


@substitute_in_graph(struct.pack, can_constant_fold_through=True)  # type: ignore[arg-type]
def pack(fmt: Union[bytes, str], /, *v: Any) -> bytes:
    return struct.pack(fmt, *v)


@substitute_in_graph(struct.unpack, can_constant_fold_through=True)  # type: ignore[arg-type]
def unpack(format: Union[bytes, str], buffer: Buffer, /) -> Tuple[Any, ...]:
    return struct.unpack(format, buffer)
