import ast
import os
import sys
from typing import Union as TypingUnion

PY38 = sys.version_info < (3, 10)

def find_type_union_locations(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    locations = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if isinstance(node.left, (ast.Name, ast.Attribute, ast.Subscript, ast.Constant, ast.BinOp)) and \
               isinstance(node.right, (ast.Name, ast.Attribute, ast.Subscript, ast.Constant, ast.BinOp)):
                is_annotation = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.AnnAssign) and parent.annotation is node:
                        is_annotation = True
                        break
                    if isinstance(parent, ast.arg) and parent.annotation is node:
                        is_annotation = True
                        break
                    if isinstance(parent, ast.FunctionDef):
                        if parent.returns is node:
                            is_annotation = True
                            break
                
                locations.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'end_line': node.end_lineno,
                    'end_col': node.end_col_offset,
                    'node': node,
                })
    
    return locations

def convert_union_to_string(node, source_lines):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Attribute):
        return f"{convert_union_to_string(node.value, source_lines)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value = convert_union_to_string(node.value, source_lines)
        slice_str = convert_slice_to_string(node.slice, source_lines)
        return f"{value}[{slice_str}]"
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = convert_union_to_string(node.left, source_lines)
        right = convert_union_to_string(node.right, source_lines)
        return f"Union[{left}, {right}]"
    elif isinstance(node, ast.Tuple):
        elts = [convert_union_to_string(e, source_lines) for e in node.elts]
        return ", ".join(elts)
    else:
        line = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
        return line[node.col_offset:node.end_col_offset].strip() if hasattr(node, 'end_col_offset') else "???"

def convert_slice_to_string(node, source_lines):
    if isinstance(node, ast.Index):
        return convert_union_to_string(node.value, source_lines)
    elif isinstance(node, ast.Tuple):
        elts = [convert_union_to_string(e, source_lines) for e in node.elts]
        return ", ".join(elts)
    else:
        return convert_union_to_string(node, source_lines)

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    
    source_lines = source.split('\n')
    
    replacements = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left_is_type = isinstance(node.left, (ast.Name, ast.Attribute, ast.Subscript, ast.Constant, ast.BinOp))
            right_is_type = isinstance(node.right, (ast.Name, ast.Attribute, ast.Subscript, ast.Constant, ast.BinOp))
            
            if left_is_type and right_is_type:
                if isinstance(node.left, ast.Constant) and not isinstance(node.left.value, type(None)):
                    continue
                if isinstance(node.right, ast.Constant) and not isinstance(node.right.value, type(None)):
                    continue
                
                new_str = convert_union_to_string(node, source_lines)
                if '???' in new_str:
                    continue
                
                replacements.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'end_line': node.end_lineno,
                    'end_col': node.end_col_offset,
                    'new_str': new_str,
                })
    
    if not replacements:
        return False
    
    replacements.sort(key=lambda x: (x['line'], x['col']), reverse=True)
    
    for r in replacements:
        line_idx = r['line'] - 1
        end_line_idx = r['end_line'] - 1
        
        if r['line'] == r['end_line']:
            line = source_lines[line_idx]
            source_lines[line_idx] = line[:r['col']] + r['new_str'] + line[r['end_col']:]
        else:
            first_line = source_lines[line_idx]
            last_line = source_lines[end_line_idx]
            new_content = first_line[:r['col']] + r['new_str'] + last_line[r['end_col']:]
            source_lines[line_idx:end_line_idx + 1] = [new_content]
    
    new_source = '\n'.join(source_lines)
    
    needs_union = 'Union[' in new_source
    if needs_union:
        has_union_import = False
        lines = new_source.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from typing import ') and 'Union' in line:
                has_union_import = True
                break
        
        if not has_union_import:
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
            
            existing_typing = False
            for i, line in enumerate(lines):
                if line.startswith('from typing import '):
                    existing = line.replace('from typing import ', '').strip()
                    lines[i] = f"from typing import Union, {existing}"
                    existing_typing = True
                    break
            
            if not existing_typing:
                lines.insert(insert_idx, "from typing import Union")
            
            new_source = '\n'.join(lines)
    
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
            try:
                if fix_file(filepath):
                    count += 1
                    print(f"Fixed: {filepath}")
            except Exception as e:
                print(f"Error fixing {filepath}: {e}")

print(f"\nTotal files fixed: {count}")
