import ast
import os
import sys
import re

BUILTIN_MAP = {
    'list': 'List',
    'dict': 'Dict',
    'tuple': 'Tuple',
    'set': 'Set',
    'frozenset': 'FrozenSet',
    'type': 'Type',
}

COLLECTIONS_MAP = {
    'Counter': ('collections.Counter', 'Counter'),
    'defaultdict': ('collections.defaultdict', 'DefaultDict'),
    'OrderedDict': ('collections.OrderedDict', 'OrderedDict'),
    'Deque': ('collections.deque', 'Deque'),
}

def find_subscript_locations(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    locations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                name = node.value.id
                if name in BUILTIN_MAP or name in COLLECTIONS_MAP:
                    locations.append({
                        'line': node.value.lineno,
                        'col': node.value.col_offset,
                        'end_col': node.value.end_col_offset,
                        'old_name': name,
                        'new_name': BUILTIN_MAP[name] if name in BUILTIN_MAP else COLLECTIONS_MAP[name][1],
                    })
    
    return locations

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    source = ''.join(lines)
    locations = find_subscript_locations(source)
    
    if not locations:
        return False
    
    needed_typing = set()
    
    for loc in sorted(locations, key=lambda x: (x['line'], x['col']), reverse=True):
        line_idx = loc['line'] - 1
        line = lines[line_idx]
        col_start = loc['col']
        col_end = loc['end_col']
        
        new_name = loc['new_name']
        old_name = loc['old_name']
        
        line = line[:col_start] + new_name + line[col_end:]
        lines[line_idx] = line
        
        needed_typing.add(new_name)
        
        if old_name in COLLECTIONS_MAP:
            collections_name = COLLECTIONS_MAP[old_name][0]
            typing_name = COLLECTIONS_MAP[old_name][1]
            for i, l in enumerate(lines):
                if f'from collections import {old_name}' in l or f'from collections import {old_name},' in l or f', {old_name}' in l:
                    pass
    
    for old_name in COLLECTIONS_MAP:
        typing_name = COLLECTIONS_MAP[old_name][1]
        if typing_name in needed_typing:
            for i, line in enumerate(lines):
                pattern = f'from collections import '
                if line.strip().startswith(pattern):
                    imports_str = line.strip().replace(pattern, '')
                    import_list = [x.strip() for x in imports_str.rstrip(')').split(',')]
                    import_list = [x for x in import_list if x != old_name and x]
                    if import_list:
                        lines[i] = f'from collections import {", ".join(import_list)}\n'
                    else:
                        lines[i] = ''
                    break
    
    if needed_typing:
        existing_typing_line = None
        existing_typing_idx = None
        for i, line in enumerate(lines):
            if line.startswith('from typing import '):
                existing_typing_line = line
                existing_typing_idx = i
                break
        
        if existing_typing_line is not None:
            existing_str = existing_typing_line.replace('from typing import ', '').strip().rstrip('\n')
            existing_names = set(n.strip() for n in existing_str.split(',') if n.strip())
            all_names = sorted(existing_names | needed_typing)
            lines[existing_typing_idx] = f'from typing import {", ".join(all_names)}\n'
        else:
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
            
            lines.insert(insert_idx, f'from typing import {", ".join(sorted(needed_typing))}\n')
    
    new_source = ''.join(lines)
    if new_source == source:
        return False
    
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_source)
    
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
