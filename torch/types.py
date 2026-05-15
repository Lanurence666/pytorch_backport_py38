from __future__ import annotations

import os
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from typing import Any, Dict, IO, List, Optional, Sequence, TYPE_CHECKING, Tuple, Type, Union
from typing_extensions import Self

from torch import (  # noqa: F401
    device as _device,
    DispatchKey,
    dtype as _dtype,
    layout as _layout,
    qscheme as _qscheme,
    Size,
    SymBool,
    SymFloat,
    SymInt,
    Tensor,
)


if TYPE_CHECKING:
    from torch.autograd.graph import GradientEdge


__all__ = ["Number", "Device", "FileLike", "Storage"]

_TensorOrTensors = Union[Tensor, Sequence[Tensor]]
_TensorOrOptionalTensors = Union[Tensor, Sequence[Optional[Tensor]]]
_TensorOrTensorsOrGradEdge = Union[
    Tensor,
    Sequence[Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]

_size = Union[Size, List[int], Tuple[int, ...]]
_symsize = Union[Size, Sequence[Union[int, SymInt]]]
_dispatchkey = Union[str, DispatchKey]

IntLikeType = Union[int, SymInt]
FloatLikeType = Union[float, SymFloat]
BoolLikeType = Union[bool, SymBool]

py_sym_types = (SymInt, SymFloat, SymBool)
PySymType = Union[SymInt, SymFloat, SymBool]

Number = Union[int, float, bool]
_Number = (int, float, bool)

FileLike = Union[str, os.PathLike, IO[bytes]]

Device = Optional[Union[_device, str, int]]


class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
        raise NotImplementedError

    def _new_shared(self, size: int) -> Self:
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> int:
        raise NotImplementedError

    def is_shared(self) -> bool:
        raise NotImplementedError

    def share_memory_(self) -> Self:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> Self:
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> Self:
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> Self:
        raise NotImplementedError
