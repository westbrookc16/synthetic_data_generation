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
from pydantic import ValidationError

from model import DIYRepairQA
from prompts import prompt_config


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


def build_user_prompt() -> str:
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
        "- steps must be ordered and numbered like '1. ...', '2. ...'\n"
        "- tools_required and tips must be realistic and practical\n"
        "- no markdown, no extra keys, no commentary\n"
    )


def request_one_record(
    client: OpenAI,
    model: str,
    max_retries: int = 6,
    base_backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 20.0,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": prompt_config["prompt"]},
                    {"role": "user", "content": build_user_prompt()},
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

    text = response.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {text}") from exc


def generate_records(
    client: OpenAI,
    model: str,
    count: int,
    request_delay_seconds: float = 0.25,
    retry_max: int = 6,
) -> list[DIYRepairQA]:
    items: list[DIYRepairQA] = []
    attempts = 0
    max_attempts = count * 4

    while len(items) < count and attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} (accepted: {len(items)}/{count})")
        try:
            raw = request_one_record(client, model, max_retries=retry_max)
            raw["id"] = len(items) + 1
            item = DIYRepairQA.model_validate(raw)
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

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    client = OpenAI(api_key=api_key)
    records = generate_records(
        client=client,
        model=args.model,
        count=args.count,
        request_delay_seconds=args.request_delay_seconds,
        retry_max=args.retry_max,
    )
    output_path = Path(args.output)
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} validated record(s) to {output_path}")


if __name__ == "__main__":
    main()
