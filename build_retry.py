import subprocess
import sys
import os
import time

os.chdir(r'E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main\build')

max_retries = 30
for attempt in range(max_retries):
    print(f"\n=== Attempt {attempt + 1}/{max_retries} ===\n")
    
    deps_file = '.ninja_deps'
    if os.path.exists(deps_file):
        try:
            os.remove(deps_file)
            print(f"Deleted {deps_file}")
        except:
            pass
    
    proc = subprocess.Popen(
        ['ninja', '-j2'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    
    output_lines = []
    ninja_io_error = False
    
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
        if 'GetOverlappedResult' in line:
            ninja_io_error = True
    
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()
        proc.wait()
        ninja_io_error = True
    
    if proc.returncode == 0:
        print("\n=== BUILD SUCCEEDED ===")
        sys.exit(0)
    
    if ninja_io_error:
        print(f"\nNinja I/O error detected, retrying in 3 seconds...")
        time.sleep(3)
        continue
    
    last_lines = output_lines[-5:] if output_lines else []
    has_real_error = any('error C' in l or 'LINK :' in l or 'fatal error' in l for l in last_lines)
    
    if has_real_error:
        print(f"\nBuild failed with real compilation error (exit code {proc.returncode})")
        sys.exit(proc.returncode)
    
    print(f"\nBuild stopped (exit code {proc.returncode}), retrying in 3 seconds...")
    time.sleep(3)

print("Max retries reached")
sys.exit(1)
