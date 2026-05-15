from __future__ import annotations
"""Backward compatibility module for torch.onnx.symbolic_opset10."""



__all__: List[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import *  # noqa: F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (  # noqa: F401
    _slice,
)
