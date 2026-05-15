import ast
import os
import sys

BUILTIN_GENERICS = {'list': 'List', 'dict': 'Dict', 'tuple': 'Tuple', 'set': 'Set', 'frozenset': 'FrozenSet', 'type': 'Type'}

def has_future_annotations(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            for alias in node.names:
                if alias.name == 'annotations':
                    return True
    return False

def needs_typing_import(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    
    needed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in BUILTIN_GENERICS:
                needed.add(BUILTIN_GENERICS[node.value.id])
    return needed

def fix_source(source, filepath):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, False
    
    needed = needs_typing_import(source)
    if not needed:
        return source, False
    
    has_future = has_future_annotations(source)
    
    lines = source.split('\n')
    modified = False
    
    replacements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in BUILTIN_GENERICS:
                line_idx = node.lineno - 1
                col_start = node.value.col_offset
                col_end = node.value.end_col_offset if hasattr(node.value, 'end_col_offset') else col_start + len(node.value.id)
                
                old_name = node.value.id
                new_name = BUILTIN_GENERICS[old_name]
                
                line = lines[line_idx]
                prefix = line[:col_start]
                suffix = line[col_end:]
                
                if has_future:
                    is_annotation = False
                    for parent in ast.walk(tree):
                        pass
                    
                new_line = prefix + new_name + suffix
                if new_line != line:
                    replacements.append((line_idx, line, new_line))
                    modified = True
    
    for line_idx, old_line, new_line in replacements:
        lines[line_idx] = new_line
    
    if modified:
        typing_imports = sorted(needed)
        import_line = f"from typing import {', '.join(typing_imports)}"
        
        existing_typing = None
        typing_line_idx = None
        for i, line in enumerate(lines):
            if line.startswith('from typing import '):
                existing_typing = line
                typing_line_idx = i
                break
        
        if existing_typing:
            existing_names = set(n.strip() for n in existing_typing.replace('from typing import ', '').split(','))
            all_names = sorted(existing_names | needed)
            lines[typing_line_idx] = f"from typing import {', '.join(all_names)}"
        else:
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('from __future__'):
                    insert_idx = i + 1
                    continue
                if line.startswith('import ') or line.startswith('from '):
                    insert_idx = i
                    break
                if line.strip() == '' and i > 0:
                    continue
            
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
    
    return '\n'.join(lines), modified

def process_directory(root_dir):
    count = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ('__pycache__',)]
        for f in files:
            if not f.endswith('.py'):
                continue
            filepath = os.path.join(root, f)
            try:
                with open(filepath, 'r', encoding='utf-8') as fh:
                    source = fh.read()
            except:
                continue
            
            new_source, modified = fix_source(source, filepath)
            if modified:
                with open(filepath, 'w', encoding='utf-8', newline='\n') as fh:
                    fh.write(new_source)
                count += 1
                print(f"Fixed: {filepath}")
    
    return count

os.chdir(r"E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main")
total = 0
for d in ['tools', 'torchgen']:
    total += process_directory(d)
print(f"\nTotal files fixed: {total}")
