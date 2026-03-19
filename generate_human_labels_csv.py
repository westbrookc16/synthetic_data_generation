from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TOP_LEVEL_FIELDS = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
    "overall_failed",
]

QUALITY_FIELDS = [
    "answer_coherence",
    "step_actionability",
    "tool_realism",
    "safety_specificity",
    "tip_usefulness",
    "problem_answer_alignment",
    "appropriate_scope",
    "category_accuracy",
]

BASE_FIELDS = [
    "id",
    "category",
    "prompt",
    "question",
    "equipment_problem",
    "answer",
    "tools_required",
    "steps",
    "safety_info",
    "tips",
]


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


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def flatten_row(row: dict[str, Any]) -> dict[str, str]:
    flattened = {field: stringify(row.get(field)) for field in BASE_FIELDS}
    if not flattened["category"]:
        flattened["category"] = stringify(row.get("prompt"))
    for field in TOP_LEVEL_FIELDS:
        flattened[field] = stringify(row.get(field))

    quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
    for field in QUALITY_FIELDS:
        flattened[f"quality_{field}"] = stringify(quality.get(field))

    flattened["notes"] = stringify(row.get("notes"))
    return flattened


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = BASE_FIELDS + TOP_LEVEL_FIELDS + [f"quality_{field}" for field in QUALITY_FIELDS] + ["notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CSV review sheet from human_labels JSONL.")
    parser.add_argument("--input", default="human_labels.jsonl", help="Input human label JSONL file.")
    parser.add_argument("--output", default="human_labels_review.csv", help="Output CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = read_jsonl(input_path)
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} review row(s) to {output_path}")


if __name__ == "__main__":
    main()

