import ast
import os
import sys

def collect_union_elements(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return collect_union_elements(node.left) + collect_union_elements(node.right)
    else:
        return [node]

def node_to_source(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        if node.value is None:
            return 'None'
        return repr(node.value)
    elif isinstance(node, ast.Attribute):
        return f"{node_to_source(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value = node_to_source(node.value)
        sl = node_to_source(node.slice) if not isinstance(node.slice, ast.Tuple) else ', '.join(node_to_source(e) for e in node.slice.elts)
        if isinstance(node.slice, ast.Index):
            if isinstance(node.slice.value, ast.Tuple):
                sl = ', '.join(node_to_source(e) for e in node.slice.value.elts)
            else:
                sl = node_to_source(node.slice.value)
        return f"{value}[{sl}]"
    elif isinstance(node, ast.List):
        return '[' + ', '.join(node_to_source(e) for e in node.elts) + ']'
    else:
        return None

def find_top_level_unions(tree):
    results = []
    
    def visit(node, parent=None, field=None, idx=None):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            is_nested = isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.BitOr)
            if not is_nested:
                results.append(node)
            return
        
        for child in ast.iter_child_nodes(node):
            visit(child, node)
    
    visit(tree)
    return results

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    
    top_level_unions = find_top_level_unions(tree)
    
    if not top_level_unions:
        return False
    
    replacements = []
    for union_node in top_level_unions:
        elements = collect_union_elements(union_node)
        
        element_strs = []
        valid = True
        for elem in elements:
            s = node_to_source(elem)
            if s is None:
                valid = False
                break
            element_strs.append(s)
        
        if not valid:
            continue
        
        new_str = f"Union[{', '.join(element_strs)}]"
        
        replacements.append({
            'line': union_node.lineno,
            'col': union_node.col_offset,
            'end_line': union_node.end_lineno,
            'end_col': union_node.end_col_offset,
            'new_str': new_str,
        })
    
    if not replacements:
        return False
    
    source_lines = source.split('\n')
    
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
    
    if 'Union[' in new_source:
        has_union_import = False
        lines = new_source.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from typing import ') and 'Union' in line:
                has_union_import = True
                break
        
        if not has_union_import:
            existing_typing_idx = None
            for i, line in enumerate(lines):
                if line.startswith('from typing import '):
                    existing = line.replace('from typing import ', '').strip()
                    lines[i] = f"from typing import Union, {existing}"
                    existing_typing_idx = i
                    break
            
            if existing_typing_idx is None:
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('from __future__'):
                        insert_idx = i + 1
                    elif (line.startswith('import ') or line.startswith('from ')) and not line.startswith('from __future__'):
                        insert_idx = i
                        break
                    elif line.strip() and not line.startswith('#'):
                        insert_idx = i
                        break
                
                lines.insert(insert_idx, "from typing import Union")
            
            new_source = '\n'.join(lines)
    
    if new_source == source:
        return False
    
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_source)
    
    return True

os.chdir(r"E:\AI_FUWANGGEZHANG\pytorch-main\pytorch-main")
count = 0
errors = []
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
                errors.append(f"Error fixing {filepath}: {e}")

print(f"\nTotal files fixed: {count}")
if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  {e}")
