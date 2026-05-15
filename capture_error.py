import subprocess
import os

os.chdir(r'E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main\build')

env = os.environ.copy()
env['PATH'] = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;' + env.get('PATH', '')
env['INCLUDE'] = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\winrt'
env['LIB'] = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64'

result = subprocess.run(
    ['ninja', '-j1', 'caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/cuda/BinaryMiscOpsKernels.cu.obj'],
    capture_output=True,
    text=True,
    env=env
)

with open(r'E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main\build_err_output.txt', 'w', encoding='utf-8') as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)

print(f"Exit code: {result.returncode}")
print("Output written to build_err_output.txt")
