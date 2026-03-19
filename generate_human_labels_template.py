from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from jsonl_utils import read_jsonl, write_jsonl
from label_fields import QUALITY_LABEL_FIELDS, TOP_LEVEL_LABEL_FIELDS


LABEL_FIELDS = TOP_LEVEL_LABEL_FIELDS
QUALITY_FIELDS = QUALITY_LABEL_FIELDS

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

