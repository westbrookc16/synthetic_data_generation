from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_step(name: str, command: list[str]) -> None:
    print(f"\n== {name} ==")
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DIY repair records and then judge them in one command."
    )
    parser.add_argument("--count", type=int, default=5, help="Number of records to generate.")
    parser.add_argument(
        "--generator-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="Model used by data-creation.py.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        help="Model used by judge.py.",
    )
    parser.add_argument(
        "--data-output",
        default="diy_repair_data.jsonl",
        help="Output JSONL path for generated records.",
    )
    parser.add_argument(
        "--judge-output",
        default="judge_results.jsonl",
        help="Output JSONL path for judge results.",
    )
    parser.add_argument(
        "--generate-request-delay-seconds",
        type=float,
        default=0.25,
        help="Delay between generation requests.",
    )
    parser.add_argument(
        "--judge-request-delay-seconds",
        type=float,
        default=0.25,
        help="Delay between judge requests.",
    )
    parser.add_argument(
        "--generate-retry-max",
        type=int,
        default=6,
        help="Max retries for transient generation API failures.",
    )
    parser.add_argument(
        "--judge-retry-max",
        type=int,
        default=6,
        help="Max retries for transient judge API failures.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Generation sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Generation nucleus sampling parameter.",
    )
    parser.add_argument(
        "--question-similarity-threshold",
        type=float,
        default=0.90,
        help="Near-duplicate question rejection threshold during generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generate_command = [
        sys.executable,
        "data-creation.py",
        "--count",
        str(args.count),
        "--model",
        args.generator_model,
        "--output",
        args.data_output,
        "--request-delay-seconds",
        str(args.generate_request_delay_seconds),
        "--retry-max",
        str(args.generate_retry_max),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--question-similarity-threshold",
        str(args.question_similarity_threshold),
    ]

    judge_command = [
        sys.executable,
        "judge.py",
        "--input",
        args.data_output,
        "--output",
        args.judge_output,
        "--model",
        args.judge_model,
        "--request-delay-seconds",
        str(args.judge_request_delay_seconds),
        "--retry-max",
        str(args.judge_retry_max),
    ]

    run_step("Generating dataset", generate_command)
    run_step("Judging dataset", judge_command)

    print("\nPipeline complete.")
    print(f"Generated records: {args.data_output}")
    print(f"Judge results: {args.judge_output}")


if __name__ == "__main__":
    main()