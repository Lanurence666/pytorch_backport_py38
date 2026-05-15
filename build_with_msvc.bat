@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main\build
del /f .ninja_deps 2>nul
ninja
