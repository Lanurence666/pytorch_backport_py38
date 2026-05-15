from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Union


class CoverageRecord(NamedTuple):
    filepath: str
    covered_lines: List[int]
    uncovered_lines: Union[List[int], None] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "covered_lines": self.covered_lines,
            "uncovered_lines": self.uncovered_lines,
        }
