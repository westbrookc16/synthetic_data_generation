from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABEL_FIELDS = [
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


def build_template_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "category": record.get("category", record.get("prompt")),
        "prompt": record.get("prompt", ""),
        "question": record.get("question"),
        "equipment_problem": record.get("equipment_problem"),
        "answer": record.get("answer"),
        "tools_required": record.get("tools_required"),
        "steps": record.get("steps"),
        "safety_info": record.get("safety_info"),
        "tips": record.get("tips"),
        **{field: None for field in LABEL_FIELDS},
        "quality": {field: None for field in QUALITY_FIELDS},
        "notes": "",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a human-labeling JSONL template from DIY repair records."
    )
    parser.add_argument("--input", default="diy_repair_data.jsonl", help="Source record JSONL file.")
    parser.add_argument("--output", default="human_labels.jsonl", help="Output template JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = [build_template_row(record) for record in read_jsonl(input_path)]
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} labeling template row(s) to {output_path}")


if __name__ == "__main__":
    main()

