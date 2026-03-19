from __future__ import annotations

BASE_RECORD_FIELDS = [
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

TOP_LEVEL_LABEL_FIELDS = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
    "overall_failed",
]

QUALITY_LABEL_FIELDS = [
    "answer_coherence",
    "step_actionability",
    "tool_realism",
    "safety_specificity",
    "tip_usefulness",
    "problem_answer_alignment",
    "appropriate_scope",
    "category_accuracy",
]

QUALITY_REVIEW_CSV_FIELDS = [f"quality_{field}" for field in QUALITY_LABEL_FIELDS]
REVIEW_CSV_FIELDS = BASE_RECORD_FIELDS + TOP_LEVEL_LABEL_FIELDS + QUALITY_REVIEW_CSV_FIELDS + ["notes"]
COMPARISON_LABEL_FIELDS = TOP_LEVEL_LABEL_FIELDS + [f"quality.{field}" for field in QUALITY_LABEL_FIELDS]