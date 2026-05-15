import os
import re

BUILTIN_GENERICS = {
    'list': 'List',
    'dict': 'Dict',
    'tuple': 'Tuple',
    'set': 'Set',
    'frozenset': 'FrozenSet',
    'type': 'Type',
}

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    original = source
    
    for builtin, typing_name in BUILTIN_GENERICS.items():
        pattern = r'\b' + builtin + r'\['
        replacement = typing_name + '['
        source = re.sub(pattern, replacement, source)
    
    if source == original:
        return False
    
    needed = set()
    for builtin, typing_name in BUILTIN_GENERICS.items():
        if typing_name + '[' in source:
            needed.add(typing_name)
    
    if not needed:
        return False
    
    existing_typing_match = re.search(r'^from typing import (.+)$', source, re.MULTILINE)
    if existing_typing_match:
        existing_imports_str = existing_typing_match.group(1)
        existing_names = set()
        for part in existing_imports_str.split(','):
            part = part.strip()
            if part:
                existing_names.add(part)
        
        all_names = sorted(existing_names | needed)
        new_import_line = f"from typing import {', '.join(all_names)}"
        old_import_line = f"from typing import {existing_imports_str}"
        source = source.replace(old_import_line, new_import_line, 1)
    else:
        import_line = f"from typing import {', '.join(sorted(needed))}"
        
        lines = source.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from __future__'):
                insert_idx = i + 1
            elif line.startswith('import ') or line.startswith('from '):
                if not line.startswith('from __future__'):
                    insert_idx = i
                    break
            elif line.strip() and not line.startswith('#'):
                insert_idx = i
                break
        
        lines.insert(insert_idx, import_line)
        source = '\n'.join(lines)
    
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(source)
    
    return True

os.chdir(r"E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main")
count = 0
for root_dir in ['tools', 'torchgen']:
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if not f.endswith('.py'):
                continue
            filepath = os.path.join(root, f)
            if fix_file(filepath):
                count += 1
                print(f"Fixed: {filepath}")

print(f"\nTotal files fixed: {count}")
