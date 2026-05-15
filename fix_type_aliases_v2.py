from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


class TypeAliasFinder(ast.NodeVisitor):
    def __init__(self):
        self.type_aliases = []

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if self._has_bitor(node.value):
                self.type_aliases.append(
                    (node.lineno, node.targets[0].id, node.value)
                )
        self.generic_visit(node)

    def _has_bitor(self, node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return True
        for child in ast.iter_child_nodes(node):
            if self._has_bitor(child):
                return True
        return False


def bitor_to_union_str(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = bitor_to_union_str(node.left)
        right = bitor_to_union_str(node.right)
        if right == "None":
            return f"Optional[{left}]"
        return f"Union[{left}, {right}]"
    elif isinstance(node, ast.Subscript):
        return ast.unparse(node)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return ast.unparse(node)
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Tuple):
        return ast.unparse(node)
    else:
        return ast.unparse(node)


def fix_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    finder = TypeAliasFinder()
    finder.visit(tree)

    if not finder.type_aliases:
        return False

    lines = source.split("\n")
    replacements = {}
    for lineno, name, value_node in finder.type_aliases:
        new_expr = bitor_to_union_str(value_node)
        old_line = lines[lineno - 1]
        indent = old_line[: len(old_line) - len(old_line.lstrip())]
        new_line = f"{indent}{name} = {new_expr}"
        if old_line != new_line:
            replacements[lineno] = (old_line, new_line)

    if not replacements:
        return False

    for lineno, (old, new) in replacements.items():
        lines[lineno - 1] = new

    has_typing_import = False
    typing_line_idx = -1
    existing_imports = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("from typing import"):
            has_typing_import = True
            typing_line_idx = i
            import_part = stripped[len("from typing import "):]
            for imp in import_part.split(","):
                existing_imports.add(imp.strip())
            break

    needed = {"Union", "Optional"} - existing_imports
    if needed:
        if has_typing_import:
            all_imports = existing_imports | needed
            import_str = ", ".join(sorted(all_imports))
            lines[typing_line_idx] = f"from typing import {import_str}"
        else:
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("from __future__"):
                    insert_idx = i + 1
                elif stripped and not stripped.startswith(("from __future__", "#", '"""', "'''")):
                    break
            lines.insert(insert_idx, "from typing import Union, Optional")

    new_source = "\n".join(lines)

    try:
        ast.parse(new_source)
    except SyntaxError as e:
        print(f"  SKIP (syntax error after fix): {filepath}: {e}")
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_source)

    print(f"Fixed: {filepath} ({len(replacements)} aliases)")
    return True


fixed = 0
for subdir in ["torchgen", "tools"]:
    target = os.path.join(ROOT, subdir)
    if not os.path.isdir(target):
        continue
    for root, dirs, files in os.walk(target):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(root, fn)
            if fix_file(fp):
                fixed += 1

print(f"Total fixed: {fixed}")
