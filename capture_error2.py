import subprocess
import os

os.chdir(r'E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main\build')

result = subprocess.run(
    ['ninja', '-j1', 'caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/cuda/Bucketization.cu.obj'],
    capture_output=True,
    text=True
)

with open(r'E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main\build_err_bucket.txt', 'w', encoding='utf-8') as f:
    f.write("=== STDERR ===\n")
    f.write(result.stderr)
    f.write("\n=== STDOUT ===\n")
    f.write(result.stdout)

print(f"Exit code: {result.returncode}")
