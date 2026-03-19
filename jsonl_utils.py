from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {path}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump() if hasattr(row, "model_dump") else row
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")