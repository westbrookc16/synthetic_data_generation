from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from jsonl_utils import read_jsonl
from label_fields import BASE_RECORD_FIELDS, QUALITY_LABEL_FIELDS, REVIEW_CSV_FIELDS, TOP_LEVEL_LABEL_FIELDS


TOP_LEVEL_FIELDS = TOP_LEVEL_LABEL_FIELDS
QUALITY_FIELDS = QUALITY_LABEL_FIELDS
BASE_FIELDS = BASE_RECORD_FIELDS

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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_CSV_FIELDS)
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

