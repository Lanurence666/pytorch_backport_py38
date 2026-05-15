from __future__ import annotations
"""Context management for PyTorch Inductor runtime caching.

This module provides context classes for collecting configuration and environment
information used in caching decisions for PyTorch's Inductor runtime.
"""

import json
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Sequence
from functools import lru_cache
from hashlib import sha256
from typing_extensions import override, TypedDict

import torch
from typing import Dict, Optional, Sequence, Type, Union


class _Context(ABC):
    """Abstract base class for context providers.

    Context providers collect specific configuration and environment information
    that affects compilation and runtime behavior.
    """

    @staticmethod
    @abstractmethod
    def forms_of_context() -> Sequence[str]:
        """Return a sequence of context form names provided by this context class.

        Returns:
            A sequence of strings representing the available context forms.
        """


class _RuntimeContext(_Context):
    """Context provider for runtime configuration and environment settings.

    Collects configuration settings that affect runtime behavior but not
    compilation, such as Inductor configs, determinism settings, and CUDA
    matmul precision configurations.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the runtime context forms provided by this class.

        Returns:
            A sequence containing the available runtime context forms:
            - "inductor_configs": PyTorch Inductor configuration settings
            - "torch_determinism_configs": Deterministic algorithm settings
            - "cuda_matmul_precision_configs": CUDA matrix multiplication precision settings
        """
        return (
            "inductor_configs",
            "torch_determinism_configs",
            "cuda_matmul_precision_configs",
        )

    @staticmethod
    def inductor_configs() -> Dict[str, object]:
        """Get portable Inductor configuration settings.

        Returns:
            A dictionary containing Inductor configuration settings,
            including private configs.
        """
        from torch._inductor import config

        return config.save_config_portable(ignore_private_configs=False)

    @staticmethod
    def torch_determinism_configs() -> Dict[str, object]:
        """Get PyTorch deterministic algorithm configuration settings.

        Returns:
            A dictionary containing deterministic algorithm settings:
            - Whether deterministic algorithms are enabled
            - Whether deterministic algorithm warnings are enabled
            - Fill uninitialized memory setting
        """
        return {
            "torch.are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "torch.is_deterministic_algorithms_warn_only_enabled": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "torch.utils.deterministic.fill_uninitialized_memory": getattr(
                torch.utils.deterministic, "fill_uninitialized_memory", None
            ),
        }

    @staticmethod
    def cuda_matmul_precision_configs() -> Dict[str, object]:
        """Get CUDA matrix multiplication precision configuration settings.

        Returns:
            A dictionary containing CUDA matmul precision settings:
            - FP32 precision setting
            - FP16 reduced precision reduction allowance
            - BF16 reduced precision reduction allowance
        """
        return {
            "torch.backends.cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        }


class _CompileContext(_Context):
    """Context provider for compilation-related configuration and environment settings.

    Collects information that affects compilation behavior, such as PyTorch and Triton
    versions, runtime environment, and accelerator properties.
    """

    @override
    @staticmethod
    def forms_of_context() -> Sequence[str]:
        """Return the compile context forms provided by this class.

        Returns:
            A sequence containing the available compile context forms:
            - "torch_version_hash": PyTorch version hash
            - "triton_version_hash": Triton version hash (if available)
            - "runtime": Runtime type (CUDA/HIP/None)
            - "runtime_version": Runtime version string
            - "accelerator_properties": GPU/accelerator properties
        """
        return (
            "torch_version_hash",
            "triton_version_hash",
            "runtime",
            "runtime_version",
            "accelerator_properties",
        )

    @lru_cache(maxsize=None)
    @staticmethod
    def torch_version_hash() -> str:
        """Get base64-encoded PyTorch version hash.

        Returns:
            A base64-encoded string representing the PyTorch version hash.
        """
        from torch._inductor.codecache import torch_key

        return b64encode(torch_key()).decode()

    @lru_cache(maxsize=None)
    @staticmethod
    def triton_version_hash() -> Optional[str]:
        """Get Triton version key if Triton is available.

        Returns:
            Triton version key if Triton is available, None otherwise.
        """
        from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

        return triton_key() if HAS_TRITON else None

    @lru_cache(maxsize=None)
    @staticmethod
    def runtime() -> Optional[str]:
        """Determine the runtime type based on available backends.

        Returns:
            "CUDA" if CUDA is available, "HIP" if HIP is available, None otherwise.
        """
        return "CUDA" if torch.version.cuda else "HIP" if torch.version.hip else None

    @lru_cache(maxsize=None)
    @staticmethod
    def runtime_version() -> Optional[str]:
        """Get the version string for the detected runtime.

        Returns:
            Version string for the current runtime (CUDA or HIP), or None if
            no supported runtime is detected.
        """
        return {
            "CUDA": torch.version.cuda,
            "HIP": torch.version.hip,
            "None": None,
        }.get(_CompileContext.runtime() or "None")

    @lru_cache(maxsize=None)
    @staticmethod
    def accelerator_properties() -> Optional[str]:
        """Get string representation of CUDA device properties.

        Returns:
            String representation of CUDA device properties if a runtime is
            available, None otherwise.
        """
        return (
            repr(torch.cuda.get_device_properties())
            if _CompileContext.runtime() and torch.cuda.is_available()
            else None
        )


class SelectedRuntimeContext(TypedDict):
    inductor_configs: bool
    torch_determinism_configs: bool
    cuda_matmul_precision_configs: bool


class SelectedCompileContext(TypedDict):
    torch_version_hash: bool
    triton_version_hash: bool
    runtime: bool
    runtime_version: bool
    accelerator_properties: bool


class IsolationSchema(TypedDict):
    """Schema for specifying which context forms to include in cache isolation.

    Attributes:
        runtime_context: Either True (include all runtime context), False (exclude all),
                        or a SelectedRuntimeContext dict specifying which forms to include.
        compile_context: Either True (include all compile context), False (exclude all),
                        or a SelectedCompileContext dict specifying which forms to include.
    """

    runtime_context: Union[SelectedRuntimeContext, bool]
    compile_context: Union[SelectedCompileContext, bool]


_DEFAULT_ISOLATION_SCHEMA: IsolationSchema = IsolationSchema(
    runtime_context=True, compile_context=True
)


def _collect_runtime_context(
    selection: Union[SelectedRuntimeContext, bool,]
) -> Optional[Dict[str, object]]:
    """Collect runtime context based on selection.

    Args:
        selection: True to include all, False to exclude all, or a dict
                  specifying which forms to include.

    Returns:
        Dictionary of selected context data, or None if excluded.
    """
    return {
        form: getattr(_RuntimeContext, form)()
        for form in _RuntimeContext.forms_of_context()
        if selection is True or (selection and selection.get(form, False))
    } or None


def _collect_compile_context(
    selection: Union[SelectedCompileContext, bool,]
) -> Optional[Dict[str, object]]:
    """Collect compile context based on selection.

    Args:
        selection: True to include all, False to exclude all, or a dict
                  specifying which forms to include.

    Returns:
        Dictionary of selected context data, or None if excluded.
    """
    return {
        form: getattr(_CompileContext, form)()
        for form in _CompileContext.forms_of_context()
        if selection is True or (selection and selection.get(form, False))
    } or None


def _isolation_context(
    ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA,
) -> Dict[str, object]:
    """Generate context data based on the isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A dictionary containing the selected context data with keys
        "runtime_context" and "compile_context", where each value is
        either None (if excluded) or a dict of context form data.
    """
    return {
        "runtime_context": _collect_runtime_context(ischema["runtime_context"]),
        "compile_context": _collect_compile_context(ischema["compile_context"]),
    }


def _isolation_key(ischema: IsolationSchema = _DEFAULT_ISOLATION_SCHEMA) -> str:
    """Generate a unique key for the given isolation schema.

    Args:
        ischema: Schema specifying which context forms to include.
                Defaults to including all runtime and compile context.

    Returns:
        A 32-character hexadecimal string that uniquely identifies
        the context specified by the isolation schema.
    """
    return sha256(
        json.dumps(_isolation_context(ischema), sort_keys=True).encode()
    ).hexdigest()[:32]
