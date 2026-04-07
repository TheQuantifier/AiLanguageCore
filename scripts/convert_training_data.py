import argparse
import json
from pathlib import Path
import sys


SYSTEM_PROMPT = """Return exactly one JSON object with keys `response_type`, `reason`, and `response`.

Allowed `response_type` values:
- DIRECT_ANSWER
- CLARIFICATION
- TOOL_NEEDED
- OUT_OF_SCOPE

Rules:
- Output JSON only.
- No markdown, no extra keys, no arrays.
- `response_type`, `reason`, and `response` must agree with each other.
- DIRECT_ANSWER: answer simple definitions or static questions directly.
- CLARIFICATION: ask a question when the request is missing the object, target, or options.
- TOOL_NEEDED: use when live, external, account-specific, location-based, or exact-calculation data would be required.
- OUT_OF_SCOPE: refuse unsafe, illegal, or restricted personal-advice requests.
- Do not use CLARIFICATION for clear definition prompts.
- Do not use OUT_OF_SCOPE for ordinary lookups.
- Do not use TOOL_NEEDED for harmful or illegal requests.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert processed datasets into chat-format JSONL for SFT."
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/processed/train.json"),
        help="Path to the processed train split.",
    )
    parser.add_argument(
        "--validation-input",
        type=Path,
        default=Path("data/processed/validation.json"),
        help="Path to the processed validation split.",
    )
    parser.add_argument(
        "--benchmark-input",
        type=Path,
        default=Path("data/processed/benchmark.json"),
        help="Path to the processed benchmark split.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/processed/train_sft.jsonl"),
        help="Output path for train SFT examples.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("data/processed/validation_sft.jsonl"),
        help="Output path for validation SFT examples.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("data/processed/benchmark_sft.jsonl"),
        help="Output path for benchmark SFT examples.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def build_assistant_json(record: dict) -> str:
    payload = {
        "response_type": record["response_type"],
        "reason": record["reason"],
        "response": record["response"],
    }
    return json.dumps(payload, ensure_ascii=True)


def convert_record(record: dict) -> dict:
    return {
        "id": record["id"],
        "source": record["source"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["user_input"]},
            {"role": "assistant", "content": build_assistant_json(record)},
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(1.0, current / total))
    filled = int(width * ratio)
    if filled >= width:
        bar = "=" * width
    elif filled <= 0:
        bar = "-" * width
    else:
        bar = ("=" * max(0, filled - 1)) + ">" + ("-" * (width - filled))
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%)"


def main() -> int:
    args = parse_args()
    datasets = [
        (args.train_input, args.train_output),
        (args.validation_input, args.validation_output),
        (args.benchmark_input, args.benchmark_output),
    ]

    for input_path, output_path in datasets:
        records = load_records(input_path)
        converted = []
        print(f"Converting {len(records)} records from {input_path}")
        for index, record in enumerate(records, start=1):
            converted.append(convert_record(record))
            sys.stdout.write("\r" + render_progress_bar(index, len(records)) + " " * 8)
            sys.stdout.flush()
        write_jsonl(output_path, converted)
        sys.stdout.write("\n")
        print(f"Wrote {len(converted)} records to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
