@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

cd /d E:\AI_FUWANGGEZHANG\pytorch_backport_py38\build

ninja -j2 2>&1

echo.
echo === Build complete with exit code: %ERRORLEVEL% ===
echo.
