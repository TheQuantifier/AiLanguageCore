import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


ALLOWED_RESPONSE_TYPES = {
    "DIRECT_ANSWER",
    "CLARIFICATION",
    "TOOL_NEEDED",
    "OUT_OF_SCOPE",
}

FIELD_ORDER = [
    "id",
    "user_input",
    "response_type",
    "reason",
    "response",
    "source",
    "quality_score",
]

GENERIC_PHRASES = [
    "it depends",
    "i can help",
    "that varies",
    "it varies",
]

AWKWARD_USER_INPUT_PATTERNS = [
    re.compile(r"\?\s+for me\.$", re.IGNORECASE),
    re.compile(r"\?\s+right now\.$", re.IGNORECASE),
    re.compile(r"\?\s+for me\?$", re.IGNORECASE),
]

CLARIFICATION_FILLER_PATTERNS = [
    re.compile(r"\s+for me\b", re.IGNORECASE),
    re.compile(r"\s+right now\b", re.IGNORECASE),
    re.compile(r"\s+in my situation\b", re.IGNORECASE),
    re.compile(r"\s*i am not sure\b", re.IGNORECASE),
]

OUTPUT_FULL = Path("data/processed/full_dataset.json")
OUTPUT_TRAIN = Path("data/processed/train.json")
OUTPUT_VALIDATION = Path("data/processed/validation.json")
OUTPUT_BENCHMARK = Path("data/processed/benchmark.json")
OUTPUT_REPORT = Path("data/processed/preparation_report.json")
OUTPUT_FIXED_BENCHMARK = Path("data/raw/v1_fixed_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned and split datasets from raw seed and generated data."
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("data/raw/v1_seed_dataset.json"),
        help="Path to the manual seed dataset.",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path("data/raw/generated_dataset.json"),
        help="Path to the generated dataset.",
    )
    parser.add_argument(
        "--fixed-benchmark",
        type=Path,
        default=OUTPUT_FIXED_BENCHMARK,
        help="Path to the fixed benchmark dataset that must be excluded from train and validation.",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=OUTPUT_FULL,
        help="Path for the cleaned full dataset.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=OUTPUT_TRAIN,
        help="Path for the train split.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=OUTPUT_VALIDATION,
        help="Path for the validation split.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=OUTPUT_BENCHMARK,
        help="Path for the benchmark split.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=OUTPUT_REPORT,
        help="Path for the preparation summary report.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio after cleaning.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation ratio after cleaning.",
    )
    parser.add_argument(
        "--benchmark-ratio",
        type=float,
        default=0.1,
        help="Benchmark ratio after cleaning.",
    )
    parser.add_argument(
        "--seed-random",
        type=int,
        default=11,
        help="Random seed used for deterministic splitting.",
    )
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def write_json(path: Path, records: list[dict] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def normalize_terminal_punctuation(text: str) -> str:
    text = re.sub(r"[ \t]+([?.!,])", r"\1", text)
    text = re.sub(r"([?.!,]){2,}", lambda match: match.group(0)[0], text)
    return text.strip()


def normalize_text(text: str) -> str:
    return normalize_terminal_punctuation(normalize_whitespace(text))


def normalized_key(user_input: str) -> str:
    key = normalize_text(user_input).lower()
    key = re.sub(r"\s+", " ", key)
    return key


def near_duplicate_key(user_input: str) -> str:
    key = normalize_text(user_input).lower()
    for pattern in CLARIFICATION_FILLER_PATTERNS:
        key = pattern.sub("", key)
    key = re.sub(r"[?.!,]", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def score_record(record: dict, source_priority: int) -> int:
    reason = record["reason"]
    response = record["response"]
    response_type = record["response_type"]

    score = 0
    score += source_priority * 1000
    score += len(reason)
    score += len(response)
    score += min(len(reason.split()), 20) * 3
    score += min(len(response.split()), 40) * 2

    if response_type == "CLARIFICATION" and "?" in response:
        score += 30
    if response_type == "TOOL_NEEDED" and "would need" in response.lower():
        score += 30
    if not any(phrase in response.lower() for phrase in GENERIC_PHRASES):
        score += 10

    return score


def normalize_record(record: dict, source: str) -> dict:
    clean = {
        "user_input": normalize_text(record["user_input"]),
        "response_type": normalize_text(record["response_type"]).upper(),
        "reason": normalize_text(record["reason"]),
        "response": normalize_text(record["response"]),
        "source": source,
    }
    return clean


def validate_normalized_record(record: dict) -> list[str]:
    errors = []

    for field in ("user_input", "response_type", "reason", "response", "source"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    if record.get("response_type") not in ALLOWED_RESPONSE_TYPES:
        errors.append("response_type is invalid")

    if len(record["reason"]) < 10 or len(record["reason"].split()) < 5:
        errors.append("reason is too weak")

    if len(record["response"]) < 10:
        errors.append("response is too short")

    if any(phrase in record["response"].lower() for phrase in GENERIC_PHRASES):
        errors.append("response is too generic")

    if record["response_type"] == "CLARIFICATION" and "?" not in record["response"]:
        errors.append("clarification response must ask a question")

    if record["response_type"] == "TOOL_NEEDED" and "would need" not in record["response"].lower():
        errors.append("tool_needed response must mention external/tool requirement")

    if record["source"] == "generated":
        for pattern in AWKWARD_USER_INPUT_PATTERNS:
            if pattern.search(record["user_input"]):
                errors.append("generated user_input is awkward and should be removed")
                break

    return errors


def choose_better_record(existing: dict, candidate: dict) -> tuple[dict, str]:
    existing_priority = 2 if existing["source"] == "seed" else 1
    candidate_priority = 2 if candidate["source"] == "seed" else 1

    existing_score = score_record(existing, existing_priority)
    candidate_score = score_record(candidate, candidate_priority)

    if candidate_score > existing_score:
        return candidate, "replaced_with_higher_quality_record"
    return existing, "kept_existing_higher_quality_record"


def keep_clarification_variant(existing: dict, candidate: dict) -> bool:
    return False


def stratified_split(
    records: list[dict], validation_ratio: float, benchmark_ratio: float, seed_random: int
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed_random)
    by_type: dict[str, list[dict]] = {}
    for record in records:
        by_type.setdefault(record["response_type"], []).append(record)

    train: list[dict] = []
    validation: list[dict] = []
    benchmark: list[dict] = []

    for response_type, items in sorted(by_type.items()):
        items_copy = items[:]
        rng.shuffle(items_copy)

        total = len(items_copy)
        benchmark_count = max(1, math.floor(total * benchmark_ratio)) if total >= 3 else 0
        validation_count = max(1, math.floor(total * validation_ratio)) if total >= 3 else 0

        if benchmark_count + validation_count >= total:
            if total >= 3:
                benchmark_count = 1
                validation_count = 1
            elif total == 2:
                benchmark_count = 0
                validation_count = 1
            else:
                benchmark_count = 0
                validation_count = 0

        benchmark.extend(items_copy[:benchmark_count])
        validation.extend(items_copy[benchmark_count : benchmark_count + validation_count])
        train.extend(items_copy[benchmark_count + validation_count :])

    train.sort(key=lambda item: item["id"])
    validation.sort(key=lambda item: item["id"])
    benchmark.sort(key=lambda item: item["id"])
    return train, validation, benchmark


def reorder_record(record: dict) -> dict:
    return {field: record[field] for field in FIELD_ORDER}


def main() -> int:
    args = parse_args()
    if round(args.train_ratio + args.validation_ratio + args.benchmark_ratio, 5) != 1.0:
        raise ValueError("train, validation, and benchmark ratios must sum to 1.0")

    seed_records = load_json_array(args.seed)
    generated_records = load_json_array(args.generated)
    fixed_benchmark_records = load_json_array(args.fixed_benchmark)

    merged: dict[str, list[dict]] = {}
    report = {
        "seed_records": len(seed_records),
        "generated_records": len(generated_records),
        "fixed_benchmark_records": len(fixed_benchmark_records),
        "invalid_records_removed": 0,
        "duplicates_found": 0,
        "near_duplicates_found": 0,
        "duplicate_resolution_events": [],
        "full_dataset_records": 0,
        "train_records": 0,
        "validation_records": 0,
        "benchmark_records": 0,
        "response_type_counts": {},
    }

    for source, records in (("seed", seed_records), ("generated", generated_records)):
        for raw_record in records:
            normalized = normalize_record(raw_record, source)
            errors = validate_normalized_record(normalized)
            if errors:
                report["invalid_records_removed"] += 1
                continue

            exact_key = normalized_key(normalized["user_input"])
            fuzzy_key = near_duplicate_key(normalized["user_input"])
            key = fuzzy_key
            if key in merged:
                existing_records = merged[key]
                exact_match = next(
                    (item for item in existing_records if normalized_key(item["user_input"]) == exact_key),
                    None,
                )

                if exact_match is not None:
                    report["duplicates_found"] += 1
                    chosen, event = choose_better_record(exact_match, normalized)
                    if chosen is not exact_match:
                        existing_records.remove(exact_match)
                        existing_records.append(chosen)
                    report["duplicate_resolution_events"].append(
                        {
                            "user_input": normalized["user_input"],
                            "resolution": event,
                            "kept_source": chosen["source"],
                        }
                    )
                else:
                    report["near_duplicates_found"] += 1
                    retained_variant = False
                    strongest_existing = max(
                        existing_records,
                        key=lambda item: score_record(
                            item, 2 if item["source"] == "seed" else 1
                        ),
                    )

                    has_generated_variant = any(item["source"] == "generated" for item in existing_records)
                    if not has_generated_variant and keep_clarification_variant(strongest_existing, normalized):
                        existing_records.append(normalized)
                        retained_variant = True
                        report["duplicate_resolution_events"].append(
                            {
                                "user_input": normalized["user_input"],
                                "resolution": "retained_clarification_variant",
                                "kept_source": normalized["source"],
                            }
                        )

                    if not retained_variant:
                        chosen, event = choose_better_record(strongest_existing, normalized)
                        if chosen is not strongest_existing:
                            existing_records.remove(strongest_existing)
                            existing_records.append(chosen)
                        report["duplicate_resolution_events"].append(
                            {
                                "user_input": normalized["user_input"],
                                "resolution": event,
                                "kept_source": chosen["source"],
                            }
                        )
            else:
                merged[key] = [normalized]

    cleaned_records = []
    flattened_records = []
    for record_group in merged.values():
        flattened_records.extend(record_group)

    for index, record in enumerate(
        sorted(flattened_records, key=lambda item: item["user_input"].lower()), start=1
    ):
        record["id"] = f"v1-{index:04d}"
        record["quality_score"] = score_record(record, 2 if record["source"] == "seed" else 1)
        cleaned_records.append(reorder_record(record))

    fixed_benchmark_normalized = []
    fixed_benchmark_keys = set()
    for raw_record in fixed_benchmark_records:
        normalized = normalize_record(raw_record, "benchmark")
        errors = validate_normalized_record({**normalized, "source": "seed"})
        if errors:
            raise ValueError(f"Invalid fixed benchmark record for '{normalized['user_input']}': {errors}")
        fixed_benchmark_keys.add(normalized_key(normalized["user_input"]))
        fixed_benchmark_normalized.append(normalized)

    benchmark_records = []
    for index, record in enumerate(
        sorted(fixed_benchmark_normalized, key=lambda item: item["user_input"].lower()), start=1
    ):
        prepared = {
            "id": f"benchmark-{index:04d}",
            "user_input": record["user_input"],
            "response_type": record["response_type"],
            "reason": record["reason"],
            "response": record["response"],
            "source": "benchmark",
            "quality_score": score_record({**record, "source": "seed"}, 2),
        }
        benchmark_records.append(reorder_record(prepared))

    train_validation_pool = [
        record for record in cleaned_records if normalized_key(record["user_input"]) not in fixed_benchmark_keys
    ]

    train_records, validation_records, _unused_benchmark = stratified_split(
        train_validation_pool,
        validation_ratio=args.validation_ratio,
        benchmark_ratio=0.0,
        seed_random=args.seed_random,
    )

    report["full_dataset_records"] = len(train_validation_pool) + len(benchmark_records)
    report["train_records"] = len(train_records)
    report["validation_records"] = len(validation_records)
    report["benchmark_records"] = len(benchmark_records)
    report["response_type_counts"] = dict(
        sorted(Counter(record["response_type"] for record in (train_validation_pool + benchmark_records)).items())
    )

    write_json(args.full_output, train_validation_pool + benchmark_records)
    write_json(args.train_output, train_records)
    write_json(args.validation_output, validation_records)
    write_json(args.benchmark_output, benchmark_records)
    write_json(args.report_output, report)

    print(f"Prepared full dataset: {args.full_output} ({len(cleaned_records)} records)")
    print(f"Train split: {args.train_output} ({len(train_records)} records)")
    print(f"Validation split: {args.validation_output} ({len(validation_records)} records)")
    print(f"Benchmark split: {args.benchmark_output} ({len(benchmark_records)} records)")
    print(f"Report: {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
