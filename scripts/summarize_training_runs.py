import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import shutil


DEFAULT_RUNS_DIR = Path("models/runs")
DEFAULT_REPORTS_DIR = Path("experiments")
DEFAULT_CSV_OUT = DEFAULT_REPORTS_DIR / "training_runs_summary.csv"
DEFAULT_NON_FROZEN_RETENTION = 3
FROZEN_TYPES = {"stress_v2", "account", "medical", "oos_tool"}

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
    parser.add_argument(
        "--apply-retention-cleanup",
        action="store_true",
        help="Prune non-frozen run folders and benchmark reports after updating the historical CSV log.",
    )
    parser.add_argument(
        "--retain-non-frozen",
        type=int,
        default=DEFAULT_NON_FROZEN_RETENTION,
        help="How many non-frozen runs and reports to keep saved on disk.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_csv_value(value: object) -> object:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "none":
        return ""
    if lowered == "nan":
        return float("nan")
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


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


def get_run_sort_key(row: dict) -> tuple[str, str, str]:
    run_date = str(row.get("run_date") or "")
    run_time = str(row.get("run_time") or "")
    run_name = str(row.get("run") or "")
    return run_date, run_time, run_name


def infer_type_from_run_name(run_name: str) -> str:
    return "stress" if "-stress-" in run_name else "core"


def is_frozen_type(type_name: object) -> bool:
    return str(type_name or "").strip().lower() in FROZEN_TYPES


def resolve_report_path(path_value: object, reports_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    candidate = Path(path_value)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        repo_candidate = (Path.cwd() / candidate).resolve()
        if repo_candidate.exists():
            return repo_candidate
        reports_candidate = (reports_dir / candidate.name).resolve()
        if reports_candidate.exists():
            return reports_candidate
    return candidate


def load_existing_rows(path: Path, reports_dir: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {key: normalize_csv_value(value) for key, value in raw_row.items()}
            row["run"] = str(row.get("run") or "")
            row["run_date"] = str(row.get("run_date") or "")
            row["run_time"] = str(row.get("run_time") or "")
            report_path = resolve_report_path(row.get("report_path"), reports_dir)
            row["report_path"] = str(report_path) if report_path and report_path.exists() else ""
            rows.append(row)
    return rows


def merge_rows(current_rows: list[dict], existing_rows: list[dict], reports_dir: Path) -> list[dict]:
    merged: dict[str, dict] = {}
    preserve_if_missing_fields = [
        "training_category",
        "training_type",
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
    ]
    for row in existing_rows:
        run_name = str(row.get("run") or "").strip()
        if not run_name:
            continue
        report_path = resolve_report_path(row.get("report_path"), reports_dir)
        row["report_path"] = str(report_path) if report_path and report_path.exists() else ""
        merged[run_name] = row

    for row in current_rows:
        run_name = str(row["run"])
        previous = merged.get(run_name, {})
        merged_row = dict(row)
        for field in preserve_if_missing_fields:
            current_value = merged_row.get(field)
            if current_value in ("", None):
                if previous.get(field) not in ("", None):
                    merged_row[field] = previous[field]
        if merged_row.get("report_path") in ("", None):
            merged_row["report_path"] = ""
        merged[run_name] = merged_row

    rows = list(merged.values())
    rows.sort(key=get_run_sort_key, reverse=True)
    return rows


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
    rows.sort(key=get_run_sort_key, reverse=True)
    return rows


def collect_non_frozen_run_dirs(runs_dir: Path) -> list[tuple[tuple[str, str, str], Path]]:
    candidates: list[tuple[tuple[str, str, str], Path]] = []
    for status_path in runs_dir.rglob("training_status.json"):
        run_dir = status_path.parent
        try:
            status = load_json(status_path)
        except Exception:
            continue
        run_name = run_dir.name
        run_date, run_time = split_run_timestamp(run_name)
        run_type = infer_type_from_benchmark_path(status.get("benchmark_file"))
        if run_type in {"unknown", "custom"}:
            run_type = infer_type_from_run_name(run_name)
        if is_frozen_type(run_type):
            continue
        candidates.append(((run_date, run_time, run_name), run_dir))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def collect_non_frozen_report_paths(reports_dir: Path) -> list[tuple[tuple[str, str, str], Path]]:
    candidates: list[tuple[tuple[str, str, str], Path]] = []
    for report_path in reports_dir.glob("benchmark_report-*.json"):
        try:
            report = load_json(report_path)
        except Exception:
            continue

        benchmark_type = str(report.get("benchmark_type") or "").strip().lower()
        training_type = str(report.get("training_type") or "").strip().lower()
        if is_frozen_type(benchmark_type) or is_frozen_type(training_type):
            continue

        stem_suffix = report_path.stem.removeprefix("benchmark_report-")
        run_date, run_time = split_run_timestamp(stem_suffix)
        sort_key = (run_date, run_time, report_path.name)
        if run_date == "" or run_time == "":
            modified = datetime.fromtimestamp(report_path.stat().st_mtime)
            sort_key = (
                modified.strftime("%Y-%m-%d"),
                modified.strftime("%H:%M:%S"),
                report_path.name,
            )
        candidates.append((sort_key, report_path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def apply_retention_cleanup(runs_dir: Path, reports_dir: Path, retain_non_frozen: int) -> dict[str, int]:
    removed_runs = 0
    removed_reports = 0
    for _sort_key, run_dir in collect_non_frozen_run_dirs(runs_dir)[max(0, retain_non_frozen):]:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=False)
            removed_runs += 1

    for _sort_key, report_path in collect_non_frozen_report_paths(reports_dir)[max(0, retain_non_frozen):]:
        if report_path.exists():
            report_path.unlink()
            removed_reports += 1

    return {
        "removed_runs": removed_runs,
        "removed_reports": removed_reports,
    }


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
    existing_rows = load_existing_rows(args.csv_out, args.reports_dir)
    current_rows = build_rows(args.runs_dir, args.reports_dir)
    rows = merge_rows(current_rows, existing_rows, args.reports_dir)
    csv_path = None
    if not args.no_csv:
        try:
            csv_path = write_csv(rows, args.csv_out)
        except PermissionError:
            print(
                f"Warning: could not update {args.csv_out} because it is in use. "
                "Close the file and run the summary again."
            )

    cleanup_summary = None
    if args.apply_retention_cleanup:
        cleanup_summary = apply_retention_cleanup(
            runs_dir=args.runs_dir,
            reports_dir=args.reports_dir,
            retain_non_frozen=max(0, int(args.retain_non_frozen)),
        )
        current_rows = build_rows(args.runs_dir, args.reports_dir)
        rows = merge_rows(current_rows, rows, args.reports_dir)
        if not args.no_csv:
            try:
                csv_path = write_csv(rows, args.csv_out)
            except PermissionError:
                print(
                    f"Warning: could not update {args.csv_out} after cleanup because it is in use. "
                    "Close the file and run the summary again."
                )

    if args.json:
        payload: object = rows
        if cleanup_summary is not None:
            payload = {
                "rows": rows,
                "cleanup": cleanup_summary,
            }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    output = render_table(rows, csv_path)
    if cleanup_summary is not None:
        output = (
            f"{output}\n\nRetention cleanup: removed {cleanup_summary['removed_runs']} run folder(s) "
            f"and {cleanup_summary['removed_reports']} benchmark report(s)."
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
