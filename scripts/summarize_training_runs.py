import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


DEFAULT_RUNS_DIR = Path("models/runs")
DEFAULT_REPORTS_DIR = Path("experiments")
DEFAULT_CSV_OUT = DEFAULT_REPORTS_DIR / "training_runs_summary.csv"

BENCHMARK_FILE_TO_TYPE = {
    "benchmark_sft.jsonl": "core",
    "benchmark_category_prediction_sft.jsonl": "core",
    "benchmark_response_sft.jsonl": "core",
    "benchmark_full_response_sft.jsonl": "core",
    "benchmark_stress_native_sft.jsonl": "stress",
    "benchmark_stress_v2_native_sft.jsonl": "stress_v2",
    "benchmark_account_tool_boundary_native_sft.jsonl": "account",
    "benchmark_medical_refusal_boundary_native_sft.jsonl": "medical",
    "benchmark_oos_vs_tool_boundary_native_sft.jsonl": "oos_tool",
}

BENCHMARK_FILE_TO_CATEGORY = {
    "benchmark_sft.jsonl": "full_response",
    "benchmark_category_prediction_sft.jsonl": "category_prediction",
    "benchmark_response_sft.jsonl": "response",
    "benchmark_full_response_sft.jsonl": "full_response",
    "benchmark_stress_native_sft.jsonl": "category_prediction",
    "benchmark_stress_v2_native_sft.jsonl": "category_prediction",
    "benchmark_account_tool_boundary_native_sft.jsonl": "category_prediction",
    "benchmark_medical_refusal_boundary_native_sft.jsonl": "category_prediction",
    "benchmark_oos_vs_tool_boundary_native_sft.jsonl": "category_prediction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize training runs and their benchmark reports in a compact table."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing training run folders.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory containing benchmark reports.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of a text table.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_CSV_OUT,
        help="Write the summary rows to a CSV file.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the CSV summary file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_type_from_benchmark_path(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return "unknown"
    return BENCHMARK_FILE_TO_TYPE.get(Path(path_value).name, "custom")


def infer_category_from_benchmark_path(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return "unknown"
    return BENCHMARK_FILE_TO_CATEGORY.get(Path(path_value).name, "custom")


def parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def fallback_training_metrics(status: dict, latest_log: dict) -> tuple[object, object, object]:
    runtime = latest_log.get("train_runtime")
    steps_per_second = latest_log.get("train_steps_per_second")
    samples_per_second = latest_log.get("train_samples_per_second")
    if runtime is not None and steps_per_second is not None and samples_per_second is not None:
        return runtime, steps_per_second, samples_per_second

    started_at = parse_iso_datetime(status.get("started_at"))
    completed_at = parse_iso_datetime(status.get("completed_at")) or parse_iso_datetime(status.get("updated_at"))
    if started_at is None or completed_at is None:
        return runtime, steps_per_second, samples_per_second

    runtime_seconds = max(0.0, (completed_at - started_at).total_seconds())
    if runtime is None:
        runtime = runtime_seconds
    if runtime_seconds <= 0:
        return runtime, steps_per_second, samples_per_second

    if steps_per_second is None:
        global_step = status.get("global_step")
        if isinstance(global_step, (int, float)):
            steps_per_second = float(global_step) / runtime_seconds
    if samples_per_second is None:
        train_examples = status.get("train_examples")
        epoch = status.get("epoch")
        if isinstance(train_examples, (int, float)) and isinstance(epoch, (int, float)):
            samples_per_second = (float(train_examples) * float(epoch)) / runtime_seconds
    return runtime, steps_per_second, samples_per_second


def format_float(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def format_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "-"


def format_int(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "-"


def split_run_timestamp(run_name: str) -> tuple[str, str]:
    parts = run_name.rsplit("-", 2)
    if len(parts) == 3:
        date_part, time_part = parts[1], parts[2]
        if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
            return (
                f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}",
                f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}",
            )
    return "", ""


def infer_type_from_run_name(run_name: str) -> str:
    return "stress" if "-stress-" in run_name else "core"


def build_rows(runs_dir: Path, reports_dir: Path) -> list[dict]:
    rows = []
    for status_path in sorted(runs_dir.rglob("training_status.json"), reverse=True):
        status = load_json(status_path)
        run_dir = status_path.parent
        run_name = run_dir.name
        latest_log = status.get("latest_log", {})
        train_runtime_s, train_steps_per_second, train_samples_per_second = fallback_training_metrics(status, latest_log)
        run_date, run_time = split_run_timestamp(run_name)
        report_path = None
        report_path_value = status.get("benchmark_report")
        if isinstance(report_path_value, str) and report_path_value.strip():
            candidate = Path(report_path_value)
            if candidate.exists():
                report_path = candidate
            else:
                relative_candidate = Path(report_path_value)
                if relative_candidate.exists():
                    report_path = relative_candidate
        if report_path is None:
            candidate = reports_dir / f"benchmark_report-{run_name}.json"
            if candidate.exists():
                report_path = candidate
        report = load_json(report_path) if report_path and report_path.exists() else {}
        training_type = report.get("training_type")
        if not isinstance(training_type, str) or not training_type.strip():
            training_type = infer_type_from_benchmark_path(status.get("benchmark_file"))
        if training_type in {"unknown", "custom"}:
            report_type = infer_type_from_benchmark_path(report.get("benchmark_file"))
            if report_type not in {"unknown", "custom"}:
                training_type = report_type
        if training_type in {"unknown", "custom"}:
            training_type = infer_type_from_run_name(run_name)

        training_category = report.get("training_category")
        if not isinstance(training_category, str) or not training_category.strip():
            training_category = infer_category_from_benchmark_path(status.get("benchmark_file"))

        rows.append(
            {
                "training_category": training_category,
                "training_type": training_type,
                "run": run_name,
                "run_date": run_date,
                "run_time": run_time,
                "status": status.get("status"),
                "epoch": status.get("epoch"),
                "global_step": status.get("global_step"),
                "train_examples": status.get("train_examples"),
                "validation_examples": status.get("validation_examples"),
                "train_runtime_s": train_runtime_s,
                "train_steps_per_second": train_steps_per_second,
                "train_samples_per_second": train_samples_per_second,
                "train_loss": latest_log.get("train_loss"),
                "benchmark_size": report.get("benchmark_size"),
                "valid_json_rate": report.get("valid_json_rate"),
                "response_type_accuracy": report.get("response_type_accuracy"),
                "report_path": str(report_path) if report_path and report_path.exists() else "",
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "training_category",
        "training_type",
        "run",
        "run_date",
        "run_time",
        "status",
        "epoch",
        "global_step",
        "train_examples",
        "validation_examples",
        "train_runtime_s",
        "train_steps_per_second",
        "train_samples_per_second",
        "train_loss",
        "benchmark_size",
        "valid_json_rate",
        "response_type_accuracy",
        "report_path",
    ]
    targets = [path, path.with_name(f"{path.stem}.latest{path.suffix}")]
    last_error = None
    for target in targets:
        try:
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return target
        except PermissionError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return path


def render_table(rows: list[dict], csv_out: Path | None = None) -> str:
    headers = [
        ("training_category", "Category"),
        ("training_type", "Type"),
        ("run", "Run"),
        ("run_date", "Date"),
        ("run_time", "Time"),
        ("status", "Status"),
        ("epoch", "Epoch"),
        ("global_step", "Steps"),
        ("train_examples", "Train"),
        ("validation_examples", "Val"),
        ("train_runtime_s", "Runtime(s)"),
        ("train_steps_per_second", "Step/s"),
        ("train_samples_per_second", "Sample/s"),
        ("train_loss", "TrainLoss"),
        ("benchmark_size", "Bench"),
        ("valid_json_rate", "JSON"),
        ("response_type_accuracy", "Accuracy"),
    ]

    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            {
                "training_category": row["training_category"],
                "training_type": row["training_type"] or "-",
                "run": row["run"],
                "run_date": row["run_date"] or "-",
                "run_time": row["run_time"] or "-",
                "status": row["status"] or "-",
                "epoch": format_float(row["epoch"], 1),
                "global_step": format_int(row["global_step"]),
                "train_examples": format_int(row["train_examples"]),
                "validation_examples": format_int(row["validation_examples"]),
                "train_runtime_s": format_float(row["train_runtime_s"], 1),
                "train_steps_per_second": format_float(row["train_steps_per_second"], 3),
                "train_samples_per_second": format_float(row["train_samples_per_second"], 3),
                "train_loss": format_float(row["train_loss"], 4),
                "benchmark_size": format_int(row["benchmark_size"]),
                "valid_json_rate": format_percent(row["valid_json_rate"]),
                "response_type_accuracy": format_percent(row["response_type_accuracy"]),
            }
        )

    if not rendered_rows:
        lines = ["No native training runs found under models/runs."]
        if csv_out is not None:
            lines.append(f"CSV summary: {csv_out}")
        return "\n".join(lines)

    widths = {}
    for key, label in headers:
        widths[key] = max(
            len(label),
            *(len(str(rendered_row[key])) for rendered_row in rendered_rows),
        )

    lines = []
    lines.append("  ".join(label.ljust(widths[key]) for key, label in headers))
    lines.append("  ".join("-" * widths[key] for key, _ in headers))
    for rendered_row in rendered_rows:
        lines.append(
            "  ".join(str(rendered_row[key]).ljust(widths[key]) for key, _ in headers)
        )

    lines.append("")
    lines.append("Note: benchmark columns reflect the current saved report for each run.")
    lines.append("If you re-ran evaluation with a different benchmark file, the report may have been overwritten.")
    if csv_out is not None:
        lines.append(f"CSV summary: {csv_out}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rows = build_rows(args.runs_dir, args.reports_dir)
    csv_path = None
    if not args.no_csv:
        try:
            csv_path = write_csv(rows, args.csv_out)
        except PermissionError:
            print(
                f"Warning: could not update {args.csv_out} because it is in use. "
                "Close the file and run the summary again."
            )

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=True))
        return 0

    print(render_table(rows, csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
