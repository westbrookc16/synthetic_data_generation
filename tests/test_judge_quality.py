import unittest
from pathlib import Path

from pydantic import ValidationError

from compare_judgments import normalize_row
from judge import (
    JudgeResult,
    QualityMetrics,
    build_judge_prompt,
    build_schema_validation_failure_result,
    compute_overall_failed,
)
from model import DIYRepairQA


class JudgeQualityTests(unittest.TestCase):
    def test_build_judge_prompt_includes_quality_shape_and_rules(self) -> None:
        prompt = build_judge_prompt({"id": 7, "category": "plumbing", "question": "How do I fix a leaking faucet?"})

        self.assertIn('"quality": {', prompt)
        self.assertIn('"answer_coherence": 0', prompt)
        self.assertIn('"category_accuracy": 0', prompt)
        self.assertIn("any top-level failure field is 1 or any quality field is 1", prompt)

    def test_compute_overall_failed_counts_quality_failures(self) -> None:
        result = JudgeResult(
            id=1,
            incomplete_answer=0,
            safety_violations=0,
            unrealistic_tools=0,
            overcomplicated_solution=0,
            missing_context=0,
            poor_quality_tips=0,
            overall_failed=0,
            notes="quality failed",
            quality=QualityMetrics(
                answer_coherence=0,
                step_actionability=0,
                tool_realism=0,
                safety_specificity=0,
                tip_usefulness=1,
                problem_answer_alignment=0,
                appropriate_scope=0,
                category_accuracy=0,
            ),
        )

        self.assertEqual(compute_overall_failed(result), 1)

    def test_schema_validation_failure_result_includes_quality_failures(self) -> None:
        bad_row = {"id": 4, "question": "too short"}
        with self.assertRaises(ValidationError) as ctx:
            DIYRepairQA.model_validate(bad_row)

        result = build_schema_validation_failure_result(bad_row, 4, ctx.exception)

        self.assertEqual(result.id, 4)
        self.assertEqual(result.overall_failed, 1)
        self.assertEqual(result.quality.model_dump(), {
            "answer_coherence": 1,
            "step_actionability": 1,
            "tool_realism": 1,
            "safety_specificity": 1,
            "tip_usefulness": 1,
            "problem_answer_alignment": 1,
            "appropriate_scope": 1,
            "category_accuracy": 1,
        })

    def test_compare_normalize_row_reads_nested_quality_fields(self) -> None:
        record_id, normalized = normalize_row(
            {
                "id": 9,
                "incomplete_answer": 0,
                "safety_violations": 0,
                "unrealistic_tools": 0,
                "overcomplicated_solution": 0,
                "missing_context": 0,
                "poor_quality_tips": 0,
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
            },
            Path("judge_results.jsonl"),
        )

        self.assertEqual(record_id, 9)
        self.assertEqual(normalized["quality.tip_usefulness"], 1)
        self.assertEqual(normalized["quality.answer_coherence"], 0)

    def test_model_accepts_legacy_prompt_field_and_exposes_category(self) -> None:
        record = DIYRepairQA.model_validate(
            {
                "id": 1,
                "prompt": "plumbing",
                "model": "gpt-4.1-mini",
                "question": "How do I stop a sink from dripping overnight?",
                "answer": "Turn off the water supply, disassemble the faucet, replace the worn washer, and reassemble carefully.",
                "equipment_problem": "Dripping sink faucet",
                "tools_required": ["Screwdriver", "Adjustable wrench", "Replacement washer"],
                "steps": [
                    "1. Turn off the shutoff valves under the sink.",
                    "2. Remove the handle and take out the worn washer.",
                    "3. Install the new washer and test for leaks.",
                ],
                "safety_info": "Turn off the water supply first to avoid spraying water and slippery surfaces.",
                "tips": ["Take a photo before disassembly so reassembly is easier."],
            }
        )

        self.assertEqual(record.category, "plumbing")
        self.assertEqual(record.prompt, "plumbing")

    def test_model_keeps_separate_category_and_full_prompt(self) -> None:
        record = DIYRepairQA.model_validate(
            {
                "id": 2,
                "category": "plumbing",
                "prompt": "System prompt:\nYou are a plumbing expert.\n\nUser prompt:\nReturn one JSON record.",
                "model": "gpt-4.1-mini",
                "question": "How do I stop a sink from dripping overnight?",
                "answer": "Turn off the water supply, disassemble the faucet, replace the worn washer, and reassemble carefully.",
                "equipment_problem": "Dripping sink faucet",
                "tools_required": ["Screwdriver", "Adjustable wrench", "Replacement washer"],
                "steps": [
                    "1. Turn off the shutoff valves under the sink.",
                    "2. Remove the handle and take out the worn washer.",
                    "3. Install the new washer and test for leaks.",
                ],
                "safety_info": "Turn off the water supply first to avoid spraying water and slippery surfaces.",
                "tips": ["Take a photo before disassembly so reassembly is easier."],
            }
        )

        self.assertEqual(record.category, "plumbing")
        self.assertIn("System prompt:", record.prompt)


if __name__ == "__main__":
    unittest.main()

