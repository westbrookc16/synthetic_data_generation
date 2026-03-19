import csv
import tempfile
import unittest
from pathlib import Path

from compare_judgments import LABEL_FIELDS, write_csv_report, write_mismatches_csv_report


class CompareJudgmentsCsvReportTests(unittest.TestCase):
    def test_write_csv_report_outputs_expected_headers_and_rows(self) -> None:
        summary = {
            "metrics": {
                field: {
                    "accuracy": 1.0 if field == "incomplete_answer" else 0.5,
                    "precision": 1.0 if field == "incomplete_answer" else 0.25,
                    "recall": 1.0 if field == "incomplete_answer" else 0.75,
                    "f1": 1.0 if field == "incomplete_answer" else 0.375,
                    "tp": 3 if field == "incomplete_answer" else 1,
                    "tn": 7 if field == "incomplete_answer" else 2,
                    "fp": 0 if field == "incomplete_answer" else 3,
                    "fn": 0 if field == "incomplete_answer" else 4,
                }
                for field in LABEL_FIELDS
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "comparison.csv"
            write_csv_report(csv_path, summary)

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            rows[0].keys(),
            {"field", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"},
        )
        self.assertEqual(len(rows), len(LABEL_FIELDS))
        self.assertEqual(rows[0]["field"], "incomplete_answer")
        self.assertEqual(rows[0]["accuracy"], "1.0")
        self.assertEqual(rows[0]["tp"], "3")

        tip_row = next(row for row in rows if row["field"] == "quality.tip_usefulness")
        self.assertEqual(tip_row["precision"], "0.25")
        self.assertEqual(tip_row["fn"], "4")

    def test_write_mismatches_csv_report_outputs_one_row_per_difference(self) -> None:
        summary = {
            "mismatches": [
                {
                    "id": 14,
                    "category": "appliance",
                    "question": "Why is my dryer running but not heating?",
                    "fields": ["overall_failed", "quality.safety_specificity"],
                    "human": {
                        field: (0 if field != "quality.safety_specificity" else 0) for field in LABEL_FIELDS
                    },
                    "judge": {
                        field: (0 if field != "quality.safety_specificity" else 1) for field in LABEL_FIELDS
                    },
                },
                {
                    "id": 24,
                    "category": "plumbing",
                    "question": "How do I stop a faucet from dripping?",
                    "fields": ["poor_quality_tips"],
                    "human": {field: 0 for field in LABEL_FIELDS},
                    "judge": {field: (1 if field == "poor_quality_tips" else 0) for field in LABEL_FIELDS},
                },
            ]
        }
        summary["mismatches"][0]["judge"]["overall_failed"] = 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "mismatches.csv"
            write_mismatches_csv_report(csv_path, summary)

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {
            "id": "14",
            "category": "appliance",
            "question": "Why is my dryer running but not heating?",
            "field": "overall_failed",
            "human_value": "0",
            "judge_value": "1",
        })
        self.assertEqual(rows[1], {
            "id": "14",
            "category": "appliance",
            "question": "Why is my dryer running but not heating?",
            "field": "quality.safety_specificity",
            "human_value": "0",
            "judge_value": "1",
        })
        self.assertEqual(rows[2], {
            "id": "24",
            "category": "plumbing",
            "question": "How do I stop a faucet from dripping?",
            "field": "poor_quality_tips",
            "human_value": "0",
            "judge_value": "1",
        })


if __name__ == "__main__":
    unittest.main()