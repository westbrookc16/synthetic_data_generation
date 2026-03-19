from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from jsonl_utils import read_jsonl
from label_fields import COMPARISON_LABEL_FIELDS, QUALITY_LABEL_FIELDS, TOP_LEVEL_LABEL_FIELDS


LABEL_FIELDS = COMPARISON_LABEL_FIELDS

def require_binary_field(row: dict[str, Any], field: str, path: Path, record_id: int) -> int:
    value = row.get(field)
    if value is None and "." in field:
        parent_field, nested_field = field.split(".", 1)
        nested_value = row.get(parent_field)
        if isinstance(nested_value, dict):
            value = nested_value.get(nested_field)
        if value is None:
            value = row.get(nested_field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    raise ValueError(
        f"Record id={record_id} in {path} has invalid value for '{field}': {value!r}. "
        "Expected 0 or 1."
    )


def normalize_row(row: dict[str, Any], path: Path) -> tuple[int, dict[str, int]]:
    record_id = row.get("id")
    if not isinstance(record_id, int):
        raise ValueError(f"Every record in {path} must have an integer 'id'. Got: {record_id!r}")
    normalized_row = {field: require_binary_field(row, field, path, record_id) for field in LABEL_FIELDS}
    return record_id, normalized_row


def normalize_rows(rows: list[dict[str, Any]], path: Path) -> dict[int, dict[str, int]]:
    normalized: dict[int, dict[str, int]] = {}
    for row in rows:
        record_id, normalized_row = normalize_row(row, path)
        if record_id in normalized:
            raise ValueError(f"Duplicate id={record_id} found in {path}")
        normalized[record_id] = normalized_row
    return normalized


def index_rows_by_id(rows: list[dict[str, Any]], path: Path) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("id")
        if not isinstance(record_id, int):
            raise ValueError(f"Every record in {path} must have an integer 'id'. Got: {record_id!r}")
        if record_id in indexed:
            raise ValueError(f"Duplicate id={record_id} found in {path}")
        indexed[record_id] = row
    return indexed


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_field_metrics(
    human_rows: dict[int, dict[str, int]],
    judge_rows: dict[int, dict[str, int]],
    ids: list[int],
    field: str,
) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for record_id in ids:
        human_value = human_rows[record_id][field]
        judge_value = judge_rows[record_id][field]
        if human_value == 1 and judge_value == 1:
            tp += 1
        elif human_value == 0 and judge_value == 0:
            tn += 1
        elif human_value == 0 and judge_value == 1:
            fp += 1
        else:
            fn += 1

    accuracy = safe_divide(tp + tn, len(ids))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def collect_mismatches(
    human_rows: dict[int, dict[str, int]],
    judge_rows: dict[int, dict[str, int]],
    human_source_rows: dict[int, dict[str, Any]],
    ids: list[int],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for record_id in ids:
        differing_fields = [
            field for field in LABEL_FIELDS if human_rows[record_id][field] != judge_rows[record_id][field]
        ]
        if differing_fields:
            mismatches.append(
                {
                    "id": record_id,
                    "category": human_source_rows[record_id].get("category", ""),
                    "question": human_source_rows[record_id].get("question", ""),
                    "fields": differing_fields,
                    "human": {field: human_rows[record_id][field] for field in LABEL_FIELDS},
                    "judge": {field: judge_rows[record_id][field] for field in LABEL_FIELDS},
                }
            )
    return mismatches


def print_report(summary: dict[str, Any], max_mismatches: int) -> None:
    print("Comparison summary")
    print(f"- Human label records: {summary['human_record_count']}")
    print(f"- Judge result records: {summary['judge_record_count']}")
    print(f"- Shared ids compared: {summary['shared_record_count']}")
    print(f"- Missing from judge results: {summary['missing_from_judge']}")
    print(f"- Missing from human labels: {summary['missing_from_humans']}")
    print(f"- Exact match rate: {summary['exact_match_rate']:.3f}")

    print("\nPer-field metrics")
    header = (
        f"{'field':34} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} "
        f"{'tp':>4} {'tn':>4} {'fp':>4} {'fn':>4}"
    )
    print(header)
    print("-" * len(header))
    for field in LABEL_FIELDS:
        metrics = summary["metrics"][field]
        print(
            f"{field:34} {metrics['accuracy']:7.3f} {metrics['precision']:7.3f} "
            f"{metrics['recall']:7.3f} {metrics['f1']:7.3f} {metrics['tp']:4d} "
            f"{metrics['tn']:4d} {metrics['fp']:4d} {metrics['fn']:4d}"
        )

    mismatches = summary["mismatches"]
    print(f"\nMismatched records: {len(mismatches)}")
    for mismatch in mismatches[:max_mismatches]:
        print(f"- id={mismatch['id']}: {', '.join(mismatch['fields'])}")


def write_csv_report(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = ["field", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for field in LABEL_FIELDS:
            metrics = summary["metrics"][field]
            writer.writerow(
                {
                    "field": field,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
            )


def write_mismatches_csv_report(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = ["id", "category", "question", "field", "human_value", "judge_value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mismatch in summary["mismatches"]:
            record_id = mismatch["id"]
            category = mismatch.get("category", "")
            question = mismatch.get("question", "")
            human_values = mismatch["human"]
            judge_values = mismatch["judge"]
            for field in mismatch["fields"]:
                writer.writerow(
                    {
                        "id": record_id,
                        "category": category,
                        "question": question,
                        "field": field,
                        "human_value": human_values[field],
                        "judge_value": judge_values[field],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human labels against judge_results-style JSONL by record id."
    )
    parser.add_argument(
        "--human-labels",
        default="human_labels.jsonl",
        help="Path to human-labeled JSONL file.",
    )
    parser.add_argument(
        "--judge-results",
        default="judge_results.jsonl",
        help="Path to judge results JSONL file.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=10,
        help="Maximum number of mismatched record ids to print.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the comparison summary as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path to write per-field comparison metrics as CSV.",
    )
    parser.add_argument(
        "--mismatches-csv",
        help="Optional path to write mismatched record fields as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_path = Path(args.human_labels)
    judge_path = Path(args.judge_results)

    if args.max_mismatches < 0:
        raise ValueError("--max-mismatches must be >= 0")
    if not human_path.exists():
        raise FileNotFoundError(f"Human labels file not found: {human_path}")
    if not judge_path.exists():
        raise FileNotFoundError(f"Judge results file not found: {judge_path}")

    human_records = read_jsonl(human_path)
    judge_records = read_jsonl(judge_path)
    human_rows = normalize_rows(human_records, human_path)
    judge_rows = normalize_rows(judge_records, judge_path)
    human_source_rows = index_rows_by_id(human_records, human_path)

    shared_ids = sorted(set(human_rows) & set(judge_rows))
    if not shared_ids:
        raise ValueError("No shared record ids found between the two files.")

    missing_from_judge = sorted(set(human_rows) - set(judge_rows))
    missing_from_humans = sorted(set(judge_rows) - set(human_rows))
    mismatches = collect_mismatches(human_rows, judge_rows, human_source_rows, shared_ids)
    exact_matches = len(shared_ids) - len(mismatches)

    summary = {
        "human_record_count": len(human_rows),
        "judge_record_count": len(judge_rows),
        "shared_record_count": len(shared_ids),
        "missing_from_judge": missing_from_judge,
        "missing_from_humans": missing_from_humans,
        "exact_match_rate": safe_divide(exact_matches, len(shared_ids)),
        "metrics": {
            field: compute_field_metrics(human_rows, judge_rows, shared_ids, field) for field in LABEL_FIELDS
        },
        "mismatches": mismatches,
    }

    print_report(summary, args.max_mismatches)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {output_path}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        write_csv_report(output_path, summary)
        print(f"Wrote CSV report to {output_path}")

    if args.mismatches_csv:
        output_path = Path(args.mismatches_csv)
        write_mismatches_csv_report(output_path, summary)
        print(f"Wrote mismatches CSV to {output_path}")


if __name__ == "__main__":
    main()