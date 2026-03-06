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

from model import DIYRepairQA
from prompts import judge_prompt_config


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


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


def compute_overall_failed(result: JudgeResult) -> int:
    flags = [
        result.incomplete_answer,
        result.safety_violations,
        result.unrealistic_tools,
        result.overcomplicated_solution,
        result.missing_context,
        result.poor_quality_tips,
    ]
    return 1 if any(flag == 1 for flag in flags) else 0


def build_judge_prompt(record: dict[str, Any]) -> str:
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
    }
    return (
        "Evaluate this DIY repair record:\n"
        f"{json.dumps(record, indent=2)}\n\n"
        "Failure mode definitions:\n"
        "- incomplete_answer: Answer lacks enough detail to complete the repair.\n"
        "- safety_violations: Missing or incorrect safety warnings for hazardous tasks.\n"
        "- unrealistic_tools: Uses professional/specialized tools not typical at home.\n"
        "- overcomplicated_solution: Pushes professional service for simple DIY tasks.\n"
        "- missing_context: Missing problem context needed to understand the repair.\n"
        "- poor_quality_tips: Tips are vague, generic, or not useful.\n\n"
        "Return exactly one JSON object with this shape:\n"
        f"{json.dumps(result_shape, indent=2)}\n\n"
        "Rules:\n"
        "- each failure field must be binary: 0 (pass) or 1 (fail)\n"
        
        "- overall_failed must be 1 if any failure field is 1; else 0\n"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            try:
                records.append(json.loads(row))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


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
            judged_rows.append(
                JudgeResult(
                    id=raw.get("id", idx) if isinstance(raw.get("id", idx), int) else idx,
                    incomplete_answer=1,
                    safety_violations=0,
                    unrealistic_tools=0,
                    overcomplicated_solution=0,
                    missing_context=1,
                    poor_quality_tips=0,
                    overall_failed=1,
                    notes=f"Schema validation failed before LLM judging: {exc.errors()[0].get('msg', 'Schema validation error')}",
                ).model_dump()
            )
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
