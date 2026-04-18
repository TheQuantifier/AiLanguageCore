import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ALLOWED_RESPONSE_TYPES = {
    "DIRECT_ANSWER",
    "CLARIFICATION",
    "TOOL_NEEDED",
    "OUT_OF_SCOPE",
}

REQUIRED_FIELDS = {
    "user_input",
    "response_type",
    "reason",
    "response",
}

MIN_REASON_LENGTH = 10
MIN_RESPONSE_LENGTH = 10
MIN_REASON_WORDS = 5
GENERIC_PHRASES = [
    "it depends",
    "i can help",
    "that varies",
    "it varies",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate structured training dataset files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more dataset JSON files to validate.",
    )
    parser.add_argument(
        "--strict-extra-fields",
        action="store_true",
        help="Fail if a record contains fields beyond the required schema.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a JSON array.")
    return data


def validate_record(
    record: object, index: int, strict_extra_fields: bool
) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []
    if not isinstance(record, dict):
        return [f"Record {index}: record must be a JSON object."], []

    missing = sorted(REQUIRED_FIELDS - set(record.keys()))
    if missing:
        errors.append(f"Record {index}: missing fields: {', '.join(missing)}")

    if strict_extra_fields:
        extra = sorted(set(record.keys()) - REQUIRED_FIELDS)
        if extra:
            errors.append(f"Record {index}: unexpected fields: {', '.join(extra)}")

    user_input = record.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        errors.append(f"Record {index}: user_input must be a non-empty string")

    response_type = record.get("response_type")
    if response_type not in ALLOWED_RESPONSE_TYPES:
        errors.append(
            f"Record {index}: response_type must be one of {', '.join(sorted(ALLOWED_RESPONSE_TYPES))}"
        )

    reason = record.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append(f"Record {index}: reason must be a non-empty string")
    elif len(reason.strip()) < MIN_REASON_LENGTH:
        errors.append(f"Record {index}: reason too short")
    elif len(reason.strip().split()) < MIN_REASON_WORDS:
        errors.append(f"Record {index}: reason too vague")

    response = record.get("response")
    if not isinstance(response, str) or not response.strip():
        errors.append(f"Record {index}: response must be a non-empty string")
    elif len(response.strip()) < MIN_RESPONSE_LENGTH:
        errors.append(f"Record {index}: response too short")

    if errors:
        return errors, warnings

    normalized_response = response.strip().lower()
    normalized_response_type = str(response_type)
    if normalized_response_type == "CLARIFICATION" and "?" not in response:
        errors.append(f"Record {index}: clarification should ask a question")

    if normalized_response_type == "TOOL_NEEDED" and "would need" not in normalized_response:
        errors.append(
            f"Record {index}: TOOL_NEEDED response should indicate external data requirement"
        )

    if any(phrase in normalized_response for phrase in GENERIC_PHRASES):
        warnings.append(f"Record {index}: response may be too generic")

    return errors, warnings


def summarize_response_types(records: list[dict]) -> Counter:
    return Counter(
        record["response_type"]
        for record in records
        if isinstance(record, dict) and record.get("response_type") in ALLOWED_RESPONSE_TYPES
    )


def find_duplicate_inputs(records: list[dict]) -> list[tuple[str, int]]:
    normalized = [
        record["user_input"].strip().lower()
        for record in records
        if isinstance(record, dict) and isinstance(record.get("user_input"), str)
    ]
    counts = Counter(normalized)
    return sorted(
        ((user_input, count) for user_input, count in counts.items() if count > 1),
        key=lambda item: (-item[1], item[0]),
    )


def validate_dataset(path: Path, strict_extra_fields: bool) -> int:
    if not path.exists():
        print(f"{path}: file does not exist", file=sys.stderr)
        return 1

    try:
        records = load_dataset(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"{path}: invalid dataset file: {exc}", file=sys.stderr)
        return 1

    errors = []
    record_warnings = []
    for index, record in enumerate(records, start=1):
        record_errors, new_warnings = validate_record(record, index, strict_extra_fields)
        errors.extend(record_errors)
        record_warnings.extend(new_warnings)

    duplicates = find_duplicate_inputs(records)
    response_counts = summarize_response_types(records)
    warnings = []

    direct_answer_count = response_counts.get("DIRECT_ANSWER", 0)
    clarification_count = response_counts.get("CLARIFICATION", 0)
    if clarification_count == 0 and direct_answer_count > 0:
        warnings.append("dataset may be imbalanced: no CLARIFICATION examples present")
    elif clarification_count > 0 and direct_answer_count > 2 * clarification_count:
        warnings.append("dataset may be imbalanced: DIRECT_ANSWER count is more than 2x CLARIFICATION")

    print(f"Dataset: {path}")
    print(f"Records: {len(records)}")
    print("Response counts:")
    for response_type in sorted(ALLOWED_RESPONSE_TYPES):
        print(f"  {response_type}: {response_counts.get(response_type, 0)}")

    if duplicates:
        print("Duplicate user_input values:")
        for user_input, count in duplicates[:10]:
            print(f"  {count}x {user_input}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
    else:
        print("Duplicate user_input values: none")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  {warning}")
        for warning in record_warnings:
            print(f"  {warning}")
    elif record_warnings:
        print("Warnings:")
        for warning in record_warnings:
            print(f"  {warning}")
    else:
        print("Warnings: none")

    if errors:
        print("Validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print("Validation: passed")
    return 0


def main() -> int:
    args = parse_args()
    exit_code = 0
    for path in args.paths:
        result = validate_dataset(path, args.strict_extra_fields)
        if result != 0:
            exit_code = result
        print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
