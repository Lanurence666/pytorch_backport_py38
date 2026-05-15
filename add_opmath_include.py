import os
import re

cuda_dir = r"E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main\aten\src\ATen\native\cuda"

files_needing_include = [
    "DistanceKernel.cu",
    "FractionalMaxPool2d.cu",
    "FractionalMaxPool3d.cu",
    "DilatedMaxPool2d.cu",
    "DilatedMaxPool3d.cu",
    "UnarySpecialOpsKernel.cu",
    "GridSampler.cuh",
    "TensorCompare.cu",
    "BinaryDivFloorKernel.cu",
    "UpSample.cuh",
    "PersistentSoftmax.cuh",
    "SoftMax.cu",
    "ForeachMinMaxFunctors.cuh",
    "BinaryMiscOpsKernels.cu",
    "ActivationThresholdKernel.cu",
    "CompareKernels.cu",
]

for fname in files_needing_include:
    fpath = os.path.join(cuda_dir, fname)
    if not os.path.exists(fpath):
        print(f"SKIP: {fname} not found")
        continue
    
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'OpMathType' in content:
        print(f"SKIP: {fname} already has OpMathType include")
        continue
    
    lines = content.split('\n')
    
    last_include_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#include') and not stripped.startswith('#include "'):
            last_include_idx = i
        elif stripped.startswith('#include "'):
            last_include_idx = i
    
    if last_include_idx == -1:
        print(f"SKIP: {fname} no #include found")
        continue
    
    new_lines = lines[:last_include_idx + 1]
    new_lines.append('#include <ATen/OpMathType.h>')
    new_lines.extend(lines[last_include_idx + 1:])
    
    new_content = '\n'.join(new_lines)
    
    with open(fpath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)
    
    print(f"FIXED: {fname} - added #include <ATen/OpMathType.h> after line {last_include_idx + 1}")

print("\nDone!")
