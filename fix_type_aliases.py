from __future__ import annotations

import os
import re

ROOT = os.path.dirname(__file__)

TYPE_ALIAS_PATTERN = re.compile(
    r"^(\w+)\s*=\s*(.+?\|.+?)\s*$", re.MULTILINE
)

UNION_IMPORTS = {}

def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as fh:
        source = fh.read()

    original = source
    lines = source.split("\n")
    new_lines = []
    needs_union = False
    union_names = set()

    for line in lines:
        stripped = line.strip()
        m = TYPE_ALIAS_PATTERN.match(stripped)
        if m and not stripped.startswith("#") and not stripped.startswith("def "):
            alias_name = m.group(1)
            type_expr = m.group(2)
            new_expr = convert_union_expr(type_expr)
            if new_expr != type_expr:
                needs_union = True
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}{alias_name} = {new_expr}")
                continue
        new_lines.append(line)

    if needs_union:
        source = "\n".join(new_lines)
        source = add_union_import(source)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(source)
        print(f"Fixed type aliases: {filepath}")
        return True
    return False

def convert_union_expr(expr):
    if "|" not in expr:
        return expr

    parts = []
    depth = 0
    current = ""
    for ch in expr:
        if ch in ("(", "[", "{"):
            depth += 1
            current += ch
        elif ch in (")", "]", "}"):
            depth -= 1
            current += ch
        elif ch == "|" and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    if len(parts) < 2:
        return expr

    has_none = False
    non_none_parts = []
    for p in parts:
        if p == "None":
            has_none = True
        else:
            non_none_parts.append(p)

    if len(non_none_parts) == 1 and has_none:
        return f"Optional[{non_none_parts[0]}]"
    elif has_none:
        inner = ", ".join(non_none_parts)
        return f"Optional[Union[{inner}]]"
    else:
        inner = ", ".join(parts)
        return f"Union[{inner}]"

def add_union_import(source):
    has_typing_import = False
    typing_line_idx = -1
    existing_imports = set()
    lines = source.split("\n")

    for i, line in enumerate(lines):
        if re.match(r"^from typing import", line):
            has_typing_import = True
            typing_line_idx = i
            imports_str = line[len("from typing import "):]
            for imp in imports_str.split(","):
                existing_imports.add(imp.strip())
            break

    needed = {"Union", "Optional"} - existing_imports
    if needed:
        if has_typing_import:
            all_imports = existing_imports | needed
            import_str = ", ".join(sorted(all_imports))
            lines[typing_line_idx] = f"from typing import {import_str}"
        else:
            future_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("from __future__"):
                    future_idx = i + 1
                elif line.strip() and not line.strip().startswith(("from __future__", "#", '"""', "'''")):
                    break
            lines.insert(future_idx, "from typing import Union, Optional")

    return "\n".join(lines)

fixed_count = 0
for subdir in ["torchgen", "tools"]:
    target_dir = os.path.join(ROOT, subdir)
    if not os.path.isdir(target_dir):
        continue
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            filepath = os.path.join(root, f)
            if process_file(filepath):
                fixed_count += 1

print(f"Total files fixed: {fixed_count}")
