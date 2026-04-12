import argparse
import json
from pathlib import Path
import re
import sys


LABEL_ONLY_SYSTEM_PROMPT = "Reply with exactly one label: DIRECT_ANSWER, CLARIFICATION, TOOL_NEEDED, or OUT_OF_SCOPE."
FULL_RESPONSE_SYSTEM_PROMPT = (
    "Reply with valid JSON using exactly these keys: "
    "response_type, reason, response. "
    "response_type must be one of DIRECT_ANSWER, CLARIFICATION, TOOL_NEEDED, or OUT_OF_SCOPE. "
    "Keep reason and response to one short, well-formed sentence each."
)
NATIVE_FOCUSED_BENCHMARKS = [
    (
        Path("data/processed/benchmark_stress.json"),
        Path("data/processed/benchmark_stress_native_sft.jsonl"),
    ),
    (
        Path("data/processed/benchmark_stress_v2.json"),
        Path("data/processed/benchmark_stress_v2_native_sft.jsonl"),
    ),
    (
        Path("data/processed/benchmark_account_tool_boundary.json"),
        Path("data/processed/benchmark_account_tool_boundary_native_sft.jsonl"),
    ),
    (
        Path("data/processed/benchmark_medical_refusal_boundary.json"),
        Path("data/processed/benchmark_medical_refusal_boundary_native_sft.jsonl"),
    ),
    (
        Path("data/processed/benchmark_oos_vs_tool_boundary.json"),
        Path("data/processed/benchmark_oos_vs_tool_boundary_native_sft.jsonl"),
    ),
]


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
    parser.add_argument(
        "--extra-train-inputs",
        nargs="*",
        type=Path,
        default=[
            Path("data/curated/native_boundary_boost_v1.json"),
            Path("data/curated/native_boundary_boost_v2.json"),
            Path("data/processed/train_clarification_boundary_pack.json"),
            Path("data/processed/train_direct_answer_tool_boundary_pack.json"),
            Path("data/processed/train_clarification_oos_boundary_pack.json"),
            Path("data/processed/train_oos_ambiguity_boundary_pack.json"),
        ],
        help="Optional extra JSON datasets to merge into training only after removing train/validation/benchmark overlaps.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def load_optional_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_records(path)


def build_label_assistant_target(record: dict) -> str:
    return str(record["response_type"])


def build_full_response_assistant_target(record: dict) -> str:
    payload = {
        "response_type": str(record["response_type"]),
        "reason": build_stage2_reason(record),
        "response": build_stage2_response(record),
    }
    return json.dumps(payload, ensure_ascii=True)


def ensure_terminal_punctuation(text: str, fallback_ending: str = ".") -> str:
    candidate = str(text).strip()
    if not candidate:
        return candidate
    if candidate[-1] not in ".?!":
        candidate += fallback_ending
    return candidate


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def first_sentence(text: str) -> str:
    candidate = collapse_whitespace(text)
    if not candidate:
        return ""
    match = re.match(r"^(.+?[.?!])(?:\s|$)", candidate)
    if match:
        return match.group(1).strip()
    return ensure_terminal_punctuation(candidate)


def first_question_sentence(text: str) -> str:
    candidate = collapse_whitespace(text)
    if not candidate:
        return ""
    match = re.search(r"(.+?\?)", candidate)
    if match:
        return match.group(1).strip()
    return ""


def build_stage2_reason(record: dict) -> str:
    response_type = str(record["response_type"])
    if response_type == "DIRECT_ANSWER":
        return "The request can be answered directly."
    if response_type == "CLARIFICATION":
        return "The request is missing important context."
    if response_type == "TOOL_NEEDED":
        return "The request needs current, external, or account-specific information."
    if response_type == "OUT_OF_SCOPE":
        return "The request is unsafe or not appropriate to help with."
    return first_sentence(record["reason"])


def build_stage2_response(record: dict) -> str:
    response_type = str(record["response_type"])
    original_response = str(record["response"])

    if response_type == "DIRECT_ANSWER":
        candidate = first_sentence(original_response)
        return candidate if candidate else "This can be answered directly."

    if response_type == "CLARIFICATION":
        candidate = first_question_sentence(original_response)
        if candidate:
            return candidate
        return "What specific item or situation are you asking about?"

    if response_type == "TOOL_NEEDED":
        lowered = collapse_whitespace(original_response).lower()
        if "would need" in lowered:
            return first_sentence(original_response)
        return "I would need a tool or live data source to answer that."

    if response_type == "OUT_OF_SCOPE":
        lowered = collapse_whitespace(original_response).lower()
        if lowered.startswith("i can't") or lowered.startswith("i cannot"):
            return first_sentence(original_response)
        return "I can't help with that."

    return first_sentence(original_response)


def normalize_user_input(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text).strip())
    return normalized.lower()


def convert_record_label_only(record: dict) -> dict:
    return {
        "id": record["id"],
        "source": record["source"],
        "messages": [
            {"role": "system", "content": LABEL_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": record["user_input"]},
            {"role": "assistant", "content": build_label_assistant_target(record)},
        ],
    }


def convert_record_full_response(record: dict) -> dict:
    return {
        "id": record["id"],
        "source": record["source"],
        "messages": [
            {"role": "system", "content": FULL_RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": record["user_input"]},
            {"role": "assistant", "content": build_full_response_assistant_target(record)},
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


def merge_extra_train_records(
    train_records: list[dict],
    validation_records: list[dict],
    benchmark_records: list[dict],
    extra_train_inputs: list[Path],
) -> tuple[list[dict], dict]:
    merged_train_records = list(train_records)
    train_keys = {normalize_user_input(record["user_input"]) for record in train_records}
    focused_benchmark_records = []
    for focused_input_path, _ in NATIVE_FOCUSED_BENCHMARKS:
        focused_benchmark_records.extend(load_optional_records(focused_input_path))
    blocked_keys = {
        normalize_user_input(record["user_input"])
        for record in validation_records + benchmark_records + focused_benchmark_records
    }
    stats = {
        "extra_inputs_seen": 0,
        "extra_records_added": 0,
        "extra_records_skipped_existing_train": 0,
        "extra_records_skipped_eval_overlap": 0,
    }

    for input_path in extra_train_inputs:
        extra_records = load_records(input_path)
        stats["extra_inputs_seen"] += len(extra_records)
        for record in extra_records:
            user_input_key = normalize_user_input(record["user_input"])
            if user_input_key in blocked_keys:
                stats["extra_records_skipped_eval_overlap"] += 1
                continue
            if user_input_key in train_keys:
                stats["extra_records_skipped_existing_train"] += 1
                continue
            merged_train_records.append(record)
            train_keys.add(user_input_key)
            stats["extra_records_added"] += 1

    return merged_train_records, stats


def write_native_benchmark_variants() -> None:
    for input_path, output_path in NATIVE_FOCUSED_BENCHMARKS:
        if not input_path.exists():
            continue
        records = load_records(input_path)
        converted = [convert_record_label_only(record) for record in records]
        write_jsonl(output_path, converted)
        print(f"Wrote {len(converted)} records to {output_path}")


def main() -> int:
    args = parse_args()
    train_records = load_records(args.train_input)
    validation_records = load_records(args.validation_input)
    benchmark_records = load_records(args.benchmark_input)

    merged_train_records, merge_stats = merge_extra_train_records(
        train_records,
        validation_records,
        benchmark_records,
        args.extra_train_inputs,
    )

    full_response_datasets = [
        (
            merged_train_records,
            args.train_output,
            f"{args.train_input} + filtered extras" if args.extra_train_inputs else str(args.train_input),
        ),
        (validation_records, args.validation_output, str(args.validation_input)),
        (benchmark_records, args.benchmark_output, str(args.benchmark_input)),
    ]

    explicit_full_response_outputs = [
        (merged_train_records, Path("data/processed/train_full_response_sft.jsonl")),
        (validation_records, Path("data/processed/validation_full_response_sft.jsonl")),
        (benchmark_records, Path("data/processed/benchmark_full_response_sft.jsonl")),
    ]
    explicit_category_prediction_outputs = [
        (merged_train_records, Path("data/processed/train_category_prediction_sft.jsonl")),
        (validation_records, Path("data/processed/validation_category_prediction_sft.jsonl")),
        (benchmark_records, Path("data/processed/benchmark_category_prediction_sft.jsonl")),
    ]

    for records, output_path, source_label in full_response_datasets:
        converted = []
        print(f"Converting {len(records)} records from {source_label}")
        for index, record in enumerate(records, start=1):
            converted.append(convert_record_full_response(record))
            sys.stdout.write("\r" + render_progress_bar(index, len(records)) + " " * 8)
            sys.stdout.flush()
        write_jsonl(output_path, converted)
        sys.stdout.write("\n")
        print(f"Wrote {len(converted)} records to {output_path}")

    for records, output_path in explicit_full_response_outputs:
        converted = [convert_record_full_response(record) for record in records]
        write_jsonl(output_path, converted)
        print(f"Wrote {len(converted)} records to {output_path}")

    for records, output_path in explicit_category_prediction_outputs:
        converted = [convert_record_label_only(record) for record in records]
        write_jsonl(output_path, converted)
        print(f"Wrote {len(converted)} records to {output_path}")

    if args.extra_train_inputs:
        print(
            "Extra train records: "
            f"added={merge_stats['extra_records_added']}, "
            f"skipped_existing_train={merge_stats['extra_records_skipped_existing_train']}, "
            f"skipped_eval_overlap={merge_stats['extra_records_skipped_eval_overlap']}"
        )

    write_native_benchmark_variants()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
