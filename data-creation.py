from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
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
from pydantic import BaseModel, ValidationError

from model import DIYRepairQA
from prompts import prompt_configs
import instructor


class GeneratedDIYRepairQA(BaseModel):
    question: str
    answer: str
    equipment_problem: str
    tools_required: list[str]
    steps: list[str]
    safety_info: str
    tips: list[str]


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


def build_user_prompt(recent_questions: list[str], max_recent: int = 15) -> str:
    schema = {
        "id": 1,
        "question": "string",
        "answer": "string",
        "equipment_problem": "string",
        "tools_required": ["string"],
        "steps": ["1. string", "2. string"],
        "safety_info": "string",
        "tips": ["string"],
    }
    return (
        "Return exactly one JSON object matching this shape:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Requirements:\n"
        "- Create realistic homeowner repair question/answer pairs that strictly follow "
        "the required schema.\n"
        "- The question must be meaningfully different from earlier questions in this same category.\n"
        "- The question must be a novel scenario and wording.\n"
        "- steps must be ordered and numbered like '1. ...', '2. ...'\n"
        "- tools_required and tips must be realistic and practical\n"
        "- no markdown, no extra keys, no commentary\n"
        "\n"
        "Do not repeat or closely paraphrase any of these prior questions:\n"
        f"{json.dumps(recent_questions[-max_recent:], indent=2)}\n"
    )


def build_full_generation_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "System prompt:\n"
        f"{system_prompt}\n\n"
        "User prompt:\n"
        f"{user_prompt}"
    )


def normalize_question_text(question: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", question.lower())
    return " ".join(cleaned.split())


def classify_question_similarity(
    candidate_question: str,
    global_normalized_questions: set[str],
    category_normalized_questions: list[str],
    similarity_threshold: float,
) -> str | None:
    normalized_candidate = normalize_question_text(candidate_question)
    if normalized_candidate in global_normalized_questions:
        return "duplicate"

    is_near_duplicate = any(
        difflib.SequenceMatcher(a=normalized_candidate, b=existing).ratio()
        >= similarity_threshold
        for existing in category_normalized_questions
    )
    if is_near_duplicate:
        return "near-duplicate"
    return None


def request_one_record(
    client: instructor.Instructor,
    model: str,
    prompt_config: dict[str, str],
    recent_questions: list[str],
    temperature: float,
    top_p: float,
    max_retries: int = 6,
    base_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 20.0,
) -> dict[str, Any]:
    system_prompt = prompt_config["prompt"]
    user_prompt = build_user_prompt(recent_questions)
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                response_model=GeneratedDIYRepairQA,
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
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

    raw = response.model_dump()
    raw["prompt"] = build_full_generation_prompt(system_prompt, user_prompt)
    return raw


def generate_records(
    client: instructor.Instructor,
    model: str,
    count: int,
    temperature: float,
    top_p: float,
    question_similarity_threshold: float,
    request_delay_seconds: float = 0.25,
    retry_max: int = 6,
) -> list[DIYRepairQA]:
    items: list[DIYRepairQA] = []
    global_normalized_questions: set[str] = set()
    category_normalized_questions: dict[str, list[str]] = {
        config["type"]: [] for config in prompt_configs
    }
    category_recent_questions: dict[str, list[str]] = {
        config["type"]: [] for config in prompt_configs
    }
    attempts = 0
    max_attempts = count * 4

    while len(items) < count and attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} (accepted: {len(items)}/{count})")
        prompt_config = prompt_configs[len(items) % len(prompt_configs)]
        current_category = prompt_config["type"]
        try:
            raw = request_one_record(
                client,
                model,
                prompt_config=prompt_config,
                recent_questions=category_recent_questions[current_category],
                temperature=temperature,
                top_p=top_p,
                max_retries=retry_max,
            )
            raw["id"] = len(items) + 1
            raw["model"] = model
            raw["category"] = current_category
            item = DIYRepairQA.model_validate(raw)

            similarity_result = classify_question_similarity(
                candidate_question=item.question,
                global_normalized_questions=global_normalized_questions,
                category_normalized_questions=category_normalized_questions[current_category],
                similarity_threshold=question_similarity_threshold,
            )
            if similarity_result is not None:
                print(f"Rejected candidate ({similarity_result} question in {current_category}): {item.question}")
                continue

            normalized_question = normalize_question_text(item.question)
            global_normalized_questions.add(normalized_question)
            category_normalized_questions[current_category].append(normalized_question)
            category_recent_questions[current_category].append(item.question)
            items.append(item)
            print(f"Accepted record id={item.id} ({len(items)}/{count})")
        except (ValueError, ValidationError):
            print("Rejected candidate (invalid JSON or schema mismatch).")
            continue
        finally:
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)

    if len(items) < count:
        raise RuntimeError(
            f"Only generated {len(items)} valid record(s) out of {count} requested "
            f"after {attempts} attempt(s)."
        )

    return items


def write_jsonl(path: Path, records: list[DIYRepairQA]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validated DIY repair data.")
    parser.add_argument("--count", type=int, default=5, help="Number of records to generate.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name (default: OPENAI_MODEL or gpt-4.1-mini).",
    )
    parser.add_argument(
        "--output",
        default="diy_repair_data.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.25,
        help="Delay between requests to smooth API throughput.",
    )
    parser.add_argument(
        "--retry-max",
        type=int,
        default=6,
        help="Max retries for transient OpenAI API errors (rate limits, timeouts).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (higher is more varied output).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter for output diversity.",
    )
    parser.add_argument(
        "--question-similarity-threshold",
        type=float,
        default=0.90,
        help="Reject near-duplicate questions at or above this similarity ratio.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.request_delay_seconds < 0:
        raise ValueError("--request-delay-seconds must be >= 0")
    if args.retry_max < 0:
        raise ValueError("--retry-max must be >= 0")
    if not 0 <= args.temperature <= 2:
        raise ValueError("--temperature must be between 0 and 2")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in the range (0, 1]")
    if not 0 < args.question_similarity_threshold <= 1:
        raise ValueError("--question-similarity-threshold must be in the range (0, 1]")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    openai_client = OpenAI(api_key=api_key)
    client = instructor.from_openai(openai_client)
    records = generate_records(
        client=client,
        model=args.model,
        count=args.count,
        temperature=args.temperature,
        top_p=args.top_p,
        question_similarity_threshold=args.question_similarity_threshold,
        request_delay_seconds=args.request_delay_seconds,
        retry_max=args.retry_max,
    )
    output_path = Path(args.output)
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} validated record(s) to {output_path}")


if __name__ == "__main__":
    main()
