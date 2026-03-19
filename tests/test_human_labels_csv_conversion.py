import tempfile
import unittest
from pathlib import Path

from convert_human_labels_csv_to_jsonl import read_csv
from generate_human_labels_csv import write_csv


class HumanLabelsCsvConversionTests(unittest.TestCase):
    def test_read_csv_restores_lists_nested_quality_and_labels(self) -> None:
        row = {
            "id": 7,
            "category": "plumbing",
            "prompt": "System prompt...",
            "question": "How do I fix a leaking shutoff valve?",
            "equipment_problem": "Leaking shutoff valve",
            "answer": "Tighten the packing nut slightly and replace the packing if needed.",
            "tools_required": ["adjustable wrench", "bucket"],
            "steps": ["1. Turn off the main water supply.", "2. Tighten the packing nut."],
            "safety_info": "Turn off the water supply before loosening fittings.",
            "tips": ["Dry the area first so you can spot the leak source."],
            "incomplete_answer": 0,
            "safety_violations": 0,
            "unrealistic_tools": 0,
            "overcomplicated_solution": 0,
            "missing_context": 0,
            "poor_quality_tips": 1,
            "overall_failed": 1,
            "quality": {
                "answer_coherence": 0,
                "step_actionability": 0,
                "tool_realism": 0,
                "safety_specificity": 0,
                "tip_usefulness": 1,
                "problem_answer_alignment": 0,
                "appropriate_scope": 0,
                "category_accuracy": 0,
            },
            "notes": "Tips are too generic.",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labels.csv"
            write_csv(csv_path, [row])
            self.assertEqual(read_csv(csv_path), [row])

    def test_read_csv_keeps_blank_labels_as_none(self) -> None:
        row = {
            "id": 11,
            "category": "appliance",
            "prompt": "Full prompt text",
            "question": "Why is my fridge warm?",
            "equipment_problem": "Fridge not cooling",
            "answer": "Clean the condenser coils and confirm the fan is running.",
            "tools_required": ["vacuum", "coil brush"],
            "steps": ["1. Unplug the fridge.", "2. Clean the condenser coils."],
            "safety_info": "Unplug the refrigerator before cleaning near moving parts.",
            "tips": ["Photograph the panel before removing it."],
            "incomplete_answer": None,
            "safety_violations": None,
            "unrealistic_tools": None,
            "overcomplicated_solution": None,
            "missing_context": None,
            "poor_quality_tips": None,
            "overall_failed": None,
            "quality": {
                "answer_coherence": None,
                "step_actionability": None,
                "tool_realism": None,
                "safety_specificity": None,
                "tip_usefulness": None,
                "problem_answer_alignment": None,
                "appropriate_scope": None,
                "category_accuracy": None,
            },
            "notes": "",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labels.csv"
            write_csv(csv_path, [row])
            self.assertEqual(read_csv(csv_path), [row])


if __name__ == "__main__":
    unittest.main()