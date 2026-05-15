from __future__ import annotations

import os
import re

torchgen_dir = os.path.join(os.path.dirname(__file__), "torchgen")
fixed_count = 0

for root, dirs, files in os.walk(torchgen_dir):
    for f in files:
        if not f.endswith(".py"):
            continue
        filepath = os.path.join(root, f)
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()

        original = source
        has_future = "from __future__ import annotations" in source

        if not re.search(r"\w+\s*\|\s*\w+", source):
            continue

        if not has_future:
            lines = source.split("\n")
            insert_pos = 0
            in_docstring = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if in_docstring:
                    if '"""' in stripped or "'''" in stripped:
                        in_docstring = False
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if stripped.count('"""') < 2 and stripped.count("'''") < 2:
                        in_docstring = True
                    continue
                if stripped.startswith("#") or stripped == "":
                    insert_pos = i + 1
                    continue
                break

            lines.insert(insert_pos, "from __future__ import annotations")
            source = "\n".join(lines)

        if source != original:
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(source)
            fixed_count += 1
            print(f"Fixed: {filepath}")

print(f"Total files fixed: {fixed_count}")
