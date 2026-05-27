@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

cd /d E:\AI_FUWANGGEZHANG\pytorch_backport_py38\build

cmake -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DUSE_DISTRIBUTED=ON ^
  -DUSE_GLOO=ON ^
  -DUSE_NCCL=OFF ^
  -DUSE_TENSORPIPE=OFF ^
  -DUSE_CUDA=ON ^
  -DBUILD_PYTORCH=ON ^
  -DBUILD_TEST=OFF ^
  -DPYTHON_EXECUTABLE="C:\Users\lanfangzheng\AppData\Local\Programs\Python\Python38\python.exe" ^
  -DCMAKE_INSTALL_PREFIX="E:\AI_FUWANGGEZHANG\py38build\torch_site_packages" ^
  ..

echo.
echo === CMake configuration complete ===
echo.
