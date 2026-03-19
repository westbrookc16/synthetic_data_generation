from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from generate_human_labels_csv import BASE_FIELDS, QUALITY_FIELDS, TOP_LEVEL_FIELDS


LIST_FIELDS = ["tools_required", "steps", "tips"]
CSV_FIELDS = BASE_FIELDS + TOP_LEVEL_FIELDS + [f"quality_{field}" for field in QUALITY_FIELDS] + ["notes"]


def parse_required_int(value: str | None, field: str, path: Path, row_number: int) -> int:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"Missing required value for '{field}' on row {row_number} in {path}")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer value for '{field}' on row {row_number} in {path}: {value!r}"
        ) from exc


def parse_binary_label(value: str | None, field: str, path: Path, row_number: int) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    if text in {"0", "1"}:
        return int(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid binary label for '{field}' on row {row_number} in {path}: {value!r}. "
            "Expected 0, 1, or blank."
        ) from exc
    if isinstance(parsed, bool):
        return int(parsed)
    if isinstance(parsed, int) and parsed in {0, 1}:
        return parsed
    raise ValueError(
        f"Invalid binary label for '{field}' on row {row_number} in {path}: {value!r}. "
        "Expected 0, 1, or blank."
    )


def parse_list_field(value: str | None, field: str, path: Path, row_number: int) -> list[Any]:
    text = (value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON list for '{field}' on row {row_number} in {path}: {value!r}"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected '{field}' to be a JSON list on row {row_number} in {path}")
    return parsed


def parse_csv_row(row: dict[str, str | None], path: Path, row_number: int) -> dict[str, Any]:
    parsed_row: dict[str, Any] = {}
    for field in BASE_FIELDS:
        parsed_row[field] = row.get(field) or ""

    parsed_row["id"] = parse_required_int(row.get("id"), "id", path, row_number)
    for field in LIST_FIELDS:
        parsed_row[field] = parse_list_field(row.get(field), field, path, row_number)

    if not parsed_row["category"].strip():
        parsed_row["category"] = parsed_row["prompt"]

    for field in TOP_LEVEL_FIELDS:
        parsed_row[field] = parse_binary_label(row.get(field), field, path, row_number)

    parsed_row["quality"] = {
        field: parse_binary_label(row.get(f"quality_{field}"), f"quality_{field}", path, row_number)
        for field in QUALITY_FIELDS
    }
    parsed_row["notes"] = row.get("notes") or ""
    return parsed_row


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        missing_fields = [field for field in CSV_FIELDS if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(f"CSV file is missing required columns: {', '.join(missing_fields)}")

        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            if all(not (value or "").strip() for value in row.values()):
                continue
            rows.append(parse_csv_row(row, path, row_number))
        return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a labeled CSV review sheet back to JSONL.")
    parser.add_argument("--input", default="human_labels_review.csv", help="Input labeled CSV file.")
    parser.add_argument("--output", default="human_labels.jsonl", help="Output JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = read_csv(input_path)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} JSONL row(s) to {output_path}")


if __name__ == "__main__":
    main()