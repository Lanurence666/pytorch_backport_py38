from __future__ import annotations

from typing import Dict, Union
import os
import platform
import subprocess

from .setup_helpers.cmake import CMake, USE_NINJA
from .setup_helpers.env import check_negative_env_flag, IS_64BIT, IS_WINDOWS


def _get_vc_env(vc_arch: str) -> Dict[str, str]:
    try:
        from setuptools import distutils  # type: ignore[import,attr-defined]

        return distutils._msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return]
    except AttributeError:
        from setuptools._distutils import (
            _msvccompiler,  # type: ignore[import,attr-defined]
        )

        return _msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return,attr-defined]


def _overlay_windows_vcvars(env: Dict[str, str]) -> Dict[str, str]:
    vc_arch = "x64" if IS_64BIT else "x86"

    if platform.machine() == "ARM64":
        vc_arch = "x64_arm64"

        # First Win11 Windows on Arm build version that supports x64 emulation
        # is 10.0.22000.
        win11_1st_version = (10, 0, 22000)
        current_win_version = tuple(
            int(version_part) for version_part in platform.version().split(".")
        )
        if current_win_version < win11_1st_version:
            vc_arch = "x86_arm64"
            print(
                "Warning: 32-bit toolchain will be used, but 64-bit linker "
                "is recommended to avoid out-of-memory linker error!"
            )
            print(
                "Warning: Please consider upgrading to Win11, where x64 "
                "emulation is enabled!"
            )

    vc_env = _get_vc_env(vc_arch)
    vc_env = {k.upper(): v for k, v in vc_env.items()}

    msvc_v142_root = os.path.join(
        "C:", os.sep, "Program Files (x86)",
        "Microsoft Visual Studio", "2022", "BuildTools",
        "VC", "Tools", "MSVC", "14.29.30133"
    )
    msvc_aux_include = os.path.join(
        "C:", os.sep, "Program Files (x86)",
        "Microsoft Visual Studio", "2022", "BuildTools",
        "VC", "Auxiliary", "VS", "include"
    )
    sdk_version = "10.0.26100.0"
    sdk_include = os.path.join(
        "C:", os.sep, "Program Files (x86)", "Windows Kits", "10", "Include"
    )
    sdk_lib = os.path.join(
        "C:", os.sep, "Program Files (x86)", "Windows Kits", "10", "Lib"
    )
    if os.path.exists(msvc_v142_root):
        include_dirs = [os.path.join(msvc_v142_root, "include")]
        if os.path.exists(msvc_aux_include):
            include_dirs.append(msvc_aux_include)
        include_dirs.extend([
            os.path.join(sdk_include, sdk_version, "ucrt"),
            os.path.join(sdk_include, sdk_version, "um"),
            os.path.join(sdk_include, sdk_version, "shared"),
        ])
        vc_env["INCLUDE"] = ";".join(include_dirs)
        vc_env["LIB"] = ";".join([
            os.path.join(msvc_v142_root, "lib", "x64"),
            os.path.join(sdk_lib, sdk_version, "ucrt", "x64"),
            os.path.join(sdk_lib, sdk_version, "um", "x64"),
        ])
        vc_env["VCTOOLSINSTALLDIR"] = msvc_v142_root + "\\"

    for k, v in env.items():
        uk = k.upper()
        if uk == "PYTHONPATH" and uk in vc_env:
            vc_env[uk] = v + os.pathsep + vc_env[uk]
        elif uk in ("VCTOOLSINSTALLDIR", "VCTOOLSVERSION"):
            vc_env[uk] = v
        elif uk not in vc_env:
            vc_env[uk] = v
    return vc_env


def _create_build_env() -> Dict[str, str]:
    # XXX - our cmake file sometimes looks at the system environment
    # and not cmake flags!
    # you should NEVER add something to this list. It is bad practice to
    # have cmake read the environment
    my_env = os.environ.copy()
    if IS_WINDOWS and USE_NINJA:
        # When using Ninja under Windows, the gcc toolchain will be chosen as
        # default. But it should be set to MSVC as the user's first choice.
        my_env = _overlay_windows_vcvars(my_env)
        my_env.setdefault("CC", "cl")
        my_env.setdefault("CXX", "cl")
    return my_env


def build_pytorch(
    version: Union[str, None],
    cmake_python_library: Union[str, None],
    build_python: bool,
    rerun_cmake: bool,
    cmake_only: bool,
    cmake: CMake,
) -> None:
    my_env = _create_build_env()
    build_test = not check_negative_env_flag("BUILD_TEST")
    cmake.generate(
        version, cmake_python_library, build_python, build_test, my_env, rerun_cmake
    )
    if cmake_only:
        return
    build_custom_step = os.getenv("BUILD_CUSTOM_STEP")
    if build_custom_step:
        try:
            output = subprocess.check_output(
                build_custom_step,
                shell=True,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print("Command output:")
            print(output)
        except subprocess.CalledProcessError as e:
            print("Command failed with return code:", e.returncode)
            print("Output (stdout and stderr):")
            print(e.output)
            raise
    cmake.build(my_env)
