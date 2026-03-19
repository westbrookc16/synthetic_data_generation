from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError

from env_utils import load_env_file
from jsonl_utils import read_jsonl, write_jsonl
from model import DIYRepairQA
from prompts import judge_prompt_config


class QualityMetrics(BaseModel):
    answer_coherence: int = Field(
        ..., ge=0, le=1, description="The answer should read as a complete, natural response."
    )
    step_actionability: int = Field(
        ...,
        ge=0,
        le=1,
        description="Steps should be specific enough for a homeowner to follow without guessing.",
    )
    tool_realism: int = Field(
        ..., ge=0, le=1, description="Tools should be realistic for a typical homeowner to have."
    )
    safety_specificity: int = Field(
        ...,
        ge=0,
        le=1,
        description="Safety guidance should name specific hazards and precautions, not generic warnings.",
    )
    tip_usefulness: int = Field(
        ...,
        ge=0,
        le=1,
        description="Tips should be task-specific and add value beyond restating the steps.",
    )
    problem_answer_alignment: int = Field(
        ...,
        ge=0,
        le=1,
        description="The answer should directly address the user question and stated problem.",
    )
    appropriate_scope: int = Field(
        ...,
        ge=0,
        le=1,
        description="The repair should stay within realistic DIY scope and defer unsafe work.",
    )
    category_accuracy: int = Field(
        ...,
        ge=0,
        le=1,
        description="The category should correctly match the repair domain described by the record.",
    )


def build_default_quality_failures() -> QualityMetrics:
    return QualityMetrics(
        answer_coherence=1,
        step_actionability=1,
        tool_realism=1,
        safety_specificity=1,
        tip_usefulness=1,
        problem_answer_alignment=1,
        appropriate_scope=1,
        category_accuracy=1,
    )


class JudgeResult(BaseModel):
    id: int = Field(..., ge=1)
    incomplete_answer: int = Field(..., ge=0, le=1)
    safety_violations: int = Field(..., ge=0, le=1)
    unrealistic_tools: int = Field(..., ge=0, le=1)
    overcomplicated_solution: int = Field(..., ge=0, le=1)
    missing_context: int = Field(..., ge=0, le=1)
    poor_quality_tips: int = Field(..., ge=0, le=1)
    overall_failed: int = Field(..., ge=0, le=1)
    notes: str = Field(..., min_length=3)
    quality: QualityMetrics = Field(..., description="Quality metrics for the answer")


def compute_overall_failed(result: JudgeResult) -> int:
    flags = [
        result.incomplete_answer,
        result.safety_violations,
        result.unrealistic_tools,
        result.overcomplicated_solution,
        result.missing_context,
        result.poor_quality_tips,
        *result.quality.model_dump().values(),
    ]
    return 1 if any(flag == 1 for flag in flags) else 0


def build_schema_validation_failure_result(raw: Any, idx: int, exc: ValidationError) -> JudgeResult:
    record_id = raw.get("id", idx) if isinstance(raw, dict) and isinstance(raw.get("id", idx), int) else idx
    return JudgeResult(
        id=record_id,
        incomplete_answer=1,
        safety_violations=0,
        unrealistic_tools=0,
        overcomplicated_solution=0,
        missing_context=1,
        poor_quality_tips=0,
        overall_failed=1,
        notes=f"Schema validation failed before LLM judging: {exc.errors()[0].get('msg', 'Schema validation error')}",
        quality=build_default_quality_failures(),
    )


def build_judge_prompt(record: dict[str, Any]) -> str:
    quality_shape = {
        "answer_coherence": 0,
        "step_actionability": 0,
        "tool_realism": 0,
        "safety_specificity": 0,
        "tip_usefulness": 0,
        "problem_answer_alignment": 0,
        "appropriate_scope": 0,
        "category_accuracy": 0,
    }
    result_shape = {
        "id": record.get("id", 0),
        "incomplete_answer": 0,
        "safety_violations": 0,
        "unrealistic_tools": 0,
        "overcomplicated_solution": 0,
        "missing_context": 0,
        "poor_quality_tips": 0,
        "overall_failed": 0,
        "notes": "Short reasoning for the flags.",
        "quality": quality_shape,
    }
    return (
        "Evaluate this DIY repair record:\n"
        f"{json.dumps(record, indent=2)}\n\n"
        "Top-level failure mode definitions (0 = pass, 1 = fail):\n"
        "- incomplete_answer: Answer lacks enough detail to complete the repair.\n"
        "- safety_violations: Missing or incorrect safety warnings for hazardous tasks.\n"
        "- unrealistic_tools: Uses professional/specialized tools not typical at home.\n"
        "- overcomplicated_solution: Pushes professional service for simple DIY tasks.\n"
        "- missing_context: Missing problem context needed to understand the repair.\n"
        "- poor_quality_tips: Tips are vague, generic, or not useful.\n\n"
        "Nested quality metric definitions inside the `quality` object (0 = pass, 1 = fail):\n"
        "- answer_coherence: The answer should read naturally, not like a stitched list of fields.\n"
        "- step_actionability: Steps should be concrete, observable, and specific enough to follow.\n"
        "- tool_realism: Tools should be realistic for a typical homeowner to already own.\n"
        "- safety_specificity: Safety guidance should name the actual hazard and exact precaution.\n"
        "- tip_usefulness: Tips should add non-obvious, task-specific value beyond the steps.\n"
        "- problem_answer_alignment: The answer should directly address the stated question/problem.\n"
        "- appropriate_scope: Unsafe or non-DIY work should be clearly deferred to a professional.\n"
        "- category_accuracy: The record's category field should correctly match the repair domain.\n\n"
        "Return exactly one JSON object with this shape:\n"
        f"{json.dumps(result_shape, indent=2)}\n\n"
        "Rules:\n"
        "- each top-level failure field must be binary: 0 (pass) or 1 (fail)\n"
        "- each field inside quality must be binary: 0 (pass) or 1 (fail)\n"
        "- overall_failed must be 1 if any top-level failure field is 1 or any quality field is 1; else 0\n"
        "- notes should briefly justify the major flag(s)\n"
        "- no markdown, no extra keys, no commentary\n"
    )


def request_judgment(
    client: OpenAI,
    model: str,
    record: dict[str, Any],
    max_retries: int = 6,
    base_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 20.0,
) -> JudgeResult:
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": judge_prompt_config["prompt"]},
                    {"role": "user", "content": build_judge_prompt(record)},
                ],
            )
            text = response.output_text.strip()
            result = JudgeResult.model_validate(json.loads(text))
            result.overall_failed = compute_overall_failed(result)
            print(f"judged id {result.id}")
            return result
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIStatusError) as exc:
            status_code = getattr(exc, "status_code", None)
            is_transient_status = status_code in {408, 409, 425, 429, 500, 502, 503, 504}
            is_transient = (
                isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError))
                or is_transient_status
            )
            if not is_transient or attempt >= max_retries:
                raise

            delay = min(max_backoff_seconds, base_backoff_seconds * (2**attempt))
            delay += random.uniform(0, 0.35 * base_backoff_seconds)
            print(
                f"Transient OpenAI error ({type(exc).__name__}, status={status_code}). "
                f"Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt >= max_retries:
                raise ValueError(f"Judge output was invalid after retries: {exc}") from exc
            time.sleep(0.75 + random.uniform(0, 0.35))

    raise RuntimeError("Failed to get judgment response.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge for DIY repair records.")
    parser.add_argument("--input", default="diy_repair_data.jsonl", help="Input JSONL file.")
    parser.add_argument("--output", default="judge_results.jsonl", help="Output JSONL file.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        help="OpenAI judge model name (OPENAI_JUDGE_MODEL, OPENAI_MODEL, or gpt-4.1-mini).",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.25,
        help="Delay between judge calls.",
    )
    parser.add_argument(
        "--retry-max",
        type=int,
        default=6,
        help="Max retries for transient OpenAI errors.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()

    if args.request_delay_seconds < 0:
        raise ValueError("--request-delay-seconds must be >= 0")
    if args.retry_max < 0:
        raise ValueError("--retry-max must be >= 0")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_records = read_jsonl(input_path)
    client = OpenAI(api_key=api_key)
    judged_rows: list[dict[str, Any]] = []

    for idx, raw in enumerate(raw_records, start=1):
        try:
            record = DIYRepairQA.model_validate(raw)
        except ValidationError as exc:
            judged_rows.append(build_schema_validation_failure_result(raw, idx, exc).model_dump())
            continue

        result = request_judgment(
            client=client,
            model=args.model,
            record=record.model_dump(),
            max_retries=args.retry_max,
        )
        judged_rows.append(result.model_dump())

        if args.request_delay_seconds > 0:
            time.sleep(args.request_delay_seconds)

    output_path = Path(args.output)
    write_jsonl(output_path, judged_rows)

    failed = sum(1 for row in judged_rows if row["overall_failed"] == 1)
    total = len(judged_rows)
    passed = total - failed
    print(f"Judged {total} record(s). Passed: {passed}. Failed: {failed}. Output: {output_path}")


if __name__ == "__main__":
    main()
