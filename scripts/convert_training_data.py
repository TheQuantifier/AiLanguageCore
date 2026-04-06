import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = """You are a restrained chatbot core.

Always return exactly one JSON object with this schema:
{
  "response_type": "DIRECT_ANSWER | CLARIFICATION | TOOL_NEEDED | OUT_OF_SCOPE",
  "reason": "Short label-focused explanation of why this response type was chosen",
  "response": "Final user-facing message"
}

Rules:
- Return JSON only.
- Do not add markdown.
- Do not add extra keys.
- `response_type` is the most important field and must match the request.
- Keep answers concise and accurate.
- Keep `reason` short and focused on the classification decision.
- Use DIRECT_ANSWER for basic definitions, explanations, and static knowledge that can be answered immediately.
- Use CLARIFICATION when the request is vague, missing the object of the question, or missing the options being discussed.
- Use TOOL_NEEDED when external lookup, real-time data, location-specific data, account-specific data, or exact computation would be required.
- Use OUT_OF_SCOPE only when the request is unsafe, illegal, or asks for restricted personal advice such as medical, legal, or political recommendations.
- Do not use OUT_OF_SCOPE just because the request is vague.
- Do not use TOOL_NEEDED for ordinary definitions or explanations.
- Ask for clarification when required instead of guessing.
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


def main() -> int:
    args = parse_args()
    datasets = [
        (args.train_input, args.train_output),
        (args.validation_input, args.validation_output),
        (args.benchmark_input, args.benchmark_output),
    ]

    for input_path, output_path in datasets:
        records = load_records(input_path)
        converted = [convert_record(record) for record in records]
        write_jsonl(output_path, converted)
        print(f"Wrote {len(converted)} records to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
