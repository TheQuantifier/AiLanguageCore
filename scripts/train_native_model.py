import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROLE_PREFIXES = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small decoder-only transformer from scratch on the project SFT data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("models/configs/v1_native_byte_transformer_config.json"),
        help="Path to the native model training config.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Optional override for config num_train_epochs.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_training_metrics(
    started_at: float,
    global_step: int,
    train_examples: int,
    num_train_epochs: int,
) -> tuple[float, float, float]:
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    steps_per_second = (global_step / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    total_seen_samples = train_examples * num_train_epochs
    samples_per_second = (total_seen_samples / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    return elapsed_seconds, steps_per_second, samples_per_second


def make_run_output_dir(base_output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base_output_dir.parent / f"{base_output_dir.name}-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_output_dir.parent / f"{base_output_dir.name}-{timestamp}-{suffix}"
        suffix += 1
    return candidate


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(torch_module, device_preference: list[str]) -> tuple[object, str]:
    if not device_preference:
        device_preference = ["cuda", "cpu"]

    hip_available = bool(getattr(torch_module.version, "hip", None)) and torch_module.cuda.is_available()
    cuda_available = torch_module.cuda.is_available() and not hip_available

    def load_directml():
        try:
            import torch_directml  # type: ignore
        except ImportError:
            return None
        if not torch_directml.is_available():
            return None
        return torch_directml

    def score_directml_name(name: str) -> tuple[int, int]:
        cleaned = name.replace("\x00", "").strip().lower()
        score = 0
        if "nvidia" in cleaned or "geforce" in cleaned or "rtx" in cleaned:
            score += 50
        if "amd" in cleaned or "radeon" in cleaned:
            score += 40
        if "rx " in cleaned or cleaned.endswith(" rx") or "radeon rx" in cleaned:
            score += 20
        if "graphics" in cleaned and "rx" not in cleaned and "rtx" not in cleaned:
            score -= 25
        if "intel" in cleaned:
            score -= 30
        return score, len(cleaned)

    for item in device_preference:
        normalized = str(item).lower()
        if normalized == "hip" and hip_available:
            return torch_module.device("cuda"), "hip"
        if normalized == "cuda" and cuda_available:
            return torch_module.device("cuda"), "cuda"
        if normalized == "directml":
            torch_directml = load_directml()
            if torch_directml is None:
                continue
            device_count = int(torch_directml.device_count())
            if device_count < 1:
                continue
            best_index = max(
                range(device_count),
                key=lambda index: score_directml_name(str(torch_directml.device_name(index))),
            )
            device_name = str(torch_directml.device_name(best_index)).replace("\x00", "").strip()
            return torch_directml.device(best_index), f"directml:{best_index}:{device_name}"
        if normalized == "cpu":
            return torch_module.device("cpu"), "cpu"

    return torch_module.device("cpu"), "cpu"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: object, fallback: float = -1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_run_benchmark_metrics(run_dir: Path, training_status: dict) -> dict[str, float]:
    metrics = {
        "valid_output_rate": -1.0,
        "response_type_accuracy": -1.0,
        "valid_json_rate": -1.0,
    }
    candidate_paths: list[Path] = []
    benchmark_report = training_status.get("benchmark_report")
    if isinstance(benchmark_report, str) and benchmark_report.strip():
        candidate_paths.append(Path(benchmark_report))
    candidate_paths.append(run_dir / "benchmark_status.json")

    for path in candidate_paths:
        try:
            resolved = path if path.is_absolute() else (run_dir / path).resolve()
            if not resolved.exists():
                continue
            payload = load_json(resolved)
            metrics = {
                "valid_output_rate": safe_float(payload.get("valid_output_rate"), -1.0),
                "response_type_accuracy": safe_float(payload.get("response_type_accuracy"), -1.0),
                "valid_json_rate": safe_float(payload.get("valid_json_rate"), -1.0),
            }
            break
        except Exception:
            continue
    return metrics


def infer_run_category(run_dir: Path) -> str | None:
    training_config_path = run_dir / "training_config.json"
    if training_config_path.exists():
        try:
            training_config = load_json(training_config_path)
            train_file = str(training_config.get("train_file", ""))
            benchmark_file = str(training_config.get("benchmark_file", ""))
            combined = f"{train_file} {benchmark_file}".lower()
            if "full_response" in combined:
                return "full_response"
            if "train_response" in combined or "benchmark_response" in combined:
                return "response"
            if (
                "category_prediction" in combined
                or "train_stress" in combined
                or "benchmark_stress" in combined
            ):
                return "category_prediction"
        except Exception:
            pass

    run_name = run_dir.name
    if "full-response" in run_name:
        return "full_response"
    if run_name.endswith("-response") or "-response-" in run_name:
        return "response"
    if "category-prediction" in run_name:
        return "category_prediction"
    return None


def infer_run_type(run_dir: Path) -> str | None:
    training_config_path = run_dir / "training_config.json"
    if training_config_path.exists():
        try:
            training_config = load_json(training_config_path)
            combined = " ".join(
                str(training_config.get(key, "")) for key in ("train_file", "validation_file", "benchmark_file", "output_dir")
            ).lower()
            if "stress_v2" in combined or "stress-v2" in combined:
                return "stress_v2"
            if "stress" in combined:
                return "stress"
        except Exception:
            pass

    run_name = run_dir.name.lower()
    if "-stress-v2-" in run_name:
        return "stress_v2"
    if "-stress-" in run_name:
        return "stress"
    return "core"


def parse_iso_timestamp(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _load_run_benchmark_metrics(run_dir: Path) -> tuple[float, float, float]:
    status_path = run_dir / "training_status.json"
    if not status_path.exists():
        return -1.0, -1.0, -1.0
    try:
        status = load_json(status_path)
    except Exception:
        return -1.0, -1.0, -1.0

    report_path = status.get("benchmark_report")
    if not report_path:
        return -1.0, -1.0, -1.0
    try:
        report = load_json(Path(report_path))
    except Exception:
        return -1.0, -1.0, -1.0

    valid_output_rate = float(report.get("valid_output_rate", -1.0))
    response_type_accuracy = float(report.get("response_type_accuracy", -1.0))
    valid_json_rate = float(report.get("valid_json_rate", -1.0))
    return valid_output_rate, response_type_accuracy, valid_json_rate


def _run_tokenizer_supports_chars(run_dir: Path, required_chars: set[str] | None) -> bool:
    if not required_chars:
        return True
    tokenizer_path = run_dir / "tokenizer_config.json"
    if not tokenizer_path.exists():
        return False
    try:
        tokenizer = ByteTokenizer.from_config(load_json(tokenizer_path))
    except Exception:
        return False
    return required_chars.issubset(set(tokenizer.chars))


def find_latest_completed_run(
    repo_root: Path,
    type_name: str | None,
    category_name: str | None,
    required_tokenizer_chars: set[str] | None = None,
) -> Path:
    runs_root = repo_root / "models" / "runs"
    status_paths = sorted(
        runs_root.rglob("training_status.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates: list[tuple[float, Path]] = []
    for status_path in status_paths:
        try:
            run_dir = status_path.parent
            if not (run_dir / "model.pt").exists():
                continue
            status = load_json(status_path)
            if str(status.get("status", "")).lower() != "completed":
                continue
            if int(status.get("global_step", 0)) <= 0:
                continue
            if type_name and infer_run_type(run_dir) != type_name:
                continue
            if category_name and infer_run_category(run_dir) != category_name:
                continue
            if not _run_tokenizer_supports_chars(run_dir, required_tokenizer_chars):
                continue
            selection_score = (
                parse_iso_timestamp(status.get("completed_at"))
                or parse_iso_timestamp(status.get("updated_at"))
                or parse_iso_timestamp(status.get("started_at"))
                or float(status_path.stat().st_mtime)
            )
            candidates.append((selection_score, run_dir))
        except Exception:
            continue

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    raise FileNotFoundError(
        f"Could not find a completed training run for type={type_name or '*'} category={category_name or '*'} under {runs_root}"
    )


def find_best_completed_run(
    repo_root: Path,
    type_name: str | None,
    category_name: str | None,
    required_tokenizer_chars: set[str] | None = None,
) -> Path:
    runs_root = repo_root / "models" / "runs"
    status_paths = sorted(runs_root.rglob("training_status.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    candidates: list[tuple[tuple[float, float, float, float, float], Path]] = []
    for status_path in status_paths:
        try:
            run_dir = status_path.parent
            if not (run_dir / "model.pt").exists():
                continue
            status = load_json(status_path)
            if str(status.get("status", "")).lower() != "completed":
                continue
            if int(status.get("global_step", 0)) <= 0:
                continue
            if type_name and infer_run_type(run_dir) != type_name:
                continue
            if category_name and infer_run_category(run_dir) != category_name:
                continue
            if not _run_tokenizer_supports_chars(run_dir, required_tokenizer_chars):
                continue

            valid_output_rate, response_type_accuracy, valid_json_rate = _load_run_benchmark_metrics(run_dir)
            best_validation_loss = float(status.get("best_validation_loss", 1e9))
            completion_time = (
                parse_iso_timestamp(status.get("completed_at"))
                or parse_iso_timestamp(status.get("updated_at"))
                or parse_iso_timestamp(status.get("started_at"))
                or float(status_path.stat().st_mtime)
            )
            selection_score = (
                response_type_accuracy,
                valid_json_rate,
                valid_output_rate,
                -best_validation_loss,
                completion_time,
            )
            candidates.append((selection_score, run_dir))
        except Exception:
            continue

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    raise FileNotFoundError(
        f"Could not find a completed training run for type={type_name or '*'} category={category_name or '*'} under {runs_root}"
    )


def resolve_init_model_path(
    path_value: str | Path | None,
    repo_root: Path,
    required_tokenizer_chars: set[str] | None = None,
) -> Path | None:
    if not path_value:
        return None
    value = str(path_value).strip()
    if not value:
        return None
    if value.startswith("latest:") or value.startswith("best:"):
        mode = value.split(":", 1)[0]
        parts = value.split(":")
        requested_type: str | None = None
        requested_category: str | None = None
        lookup_required_chars = required_tokenizer_chars
        if len(parts) == 2:
            requested_category = parts[1]
        elif len(parts) == 3:
            requested_type = parts[1]
            requested_category = parts[2]
        if requested_category == "category_prediction":
            # Stage-2 intentionally supports tokenizer-vocab mismatch via remapping.
            lookup_required_chars = None
        try:
            if len(parts) == 2:
                selected = None
                if mode == "latest":
                    selected = find_latest_completed_run(
                        repo_root,
                        None,
                        parts[1],
                        required_tokenizer_chars=lookup_required_chars,
                    )
                else:
                    selected = find_best_completed_run(
                        repo_root,
                        None,
                        parts[1],
                        required_tokenizer_chars=lookup_required_chars,
                    )
                if requested_category and infer_run_category(selected) != requested_category:
                    raise FileNotFoundError("No matching checkpoint for requested category.")
                return selected
            if len(parts) == 3:
                selected = None
                if mode == "latest":
                    selected = find_latest_completed_run(
                        repo_root,
                        parts[1],
                        parts[2],
                        required_tokenizer_chars=lookup_required_chars,
                    )
                else:
                    selected = find_best_completed_run(
                        repo_root,
                        parts[1],
                        parts[2],
                        required_tokenizer_chars=lookup_required_chars,
                    )
                if requested_type and infer_run_type(selected) != requested_type:
                    raise FileNotFoundError("No matching checkpoint for requested type.")
                if requested_category and infer_run_category(selected) != requested_category:
                    raise FileNotFoundError("No matching checkpoint for requested category.")
                return selected
        except FileNotFoundError:
            # Keep stage-2 default on category_prediction, but avoid cold starts
            # when type-specific category_prediction checkpoints are absent.
            if len(parts) == 3 and parts[2] == "category_prediction":
                fallback_finder = find_latest_completed_run if mode == "latest" else find_best_completed_run
                try:
                    return fallback_finder(
                        repo_root,
                        None,
                        "category_prediction",
                        required_tokenizer_chars=lookup_required_chars,
                    )
                except FileNotFoundError:
                    return None
            return None
        raise ValueError(
            "init_from_model_path must use latest:<category>, latest:<type>:<category>, "
            "best:<category>, or best:<type>:<category>"
        )
    return resolve_path(value, repo_root)


def render_messages(messages: list[dict], add_generation_prompt: bool) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        prefix = ROLE_PREFIXES[role]
        parts.append(prefix)
        parts.append(message["content"])
        parts.append("\n")
    if add_generation_prompt:
        parts.append(ROLE_PREFIXES["assistant"])
    return "".join(parts)


def extract_example_response_type(answer_text: str) -> str:
    stripped = str(answer_text).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, dict):
        response_type = payload.get("response_type")
        if isinstance(response_type, str) and response_type.strip():
            return response_type.strip()
    return stripped


class ByteTokenizer:
    def __init__(
        self,
        chars: list[str],
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        ordered_chars = []
        seen = set()
        for char in chars:
            if len(char) != 1:
                continue
            if char in seen:
                continue
            ordered_chars.append(char)
            seen.add(char)
        if "?" not in seen:
            ordered_chars.append("?")
            seen.add("?")
        self.chars = ordered_chars
        self.char_to_id = {char: index for index, char in enumerate(self.chars)}
        self.id_to_char = {index: char for char, index in self.char_to_id.items()}
        self.unknown_token_id = self.char_to_id["?"]
        regular_token_limit = len(self.chars)
        self.regular_token_limit = regular_token_limit
        self.pad_token_id = int(regular_token_limit if pad_token_id is None else pad_token_id)
        self.bos_token_id = int(regular_token_limit + 1 if bos_token_id is None else bos_token_id)
        self.eos_token_id = int(regular_token_limit + 2 if eos_token_id is None else eos_token_id)
        self.vocab_size = max(regular_token_limit, self.pad_token_id, self.bos_token_id, self.eos_token_id) + 1

    @classmethod
    def from_config(cls, payload: dict) -> "ByteTokenizer":
        chars = payload.get("chars")
        if chars:
            return cls(
                chars=list(chars),
                pad_token_id=int(payload["pad_token_id"]),
                bos_token_id=int(payload["bos_token_id"]),
                eos_token_id=int(payload["eos_token_id"]),
            )
        regular_token_limit = int(payload.get("regular_token_limit", payload.get("vocab_size", 131) - 3))
        chars = [chr(index) for index in range(regular_token_limit)]
        return cls(
            chars=chars,
            pad_token_id=int(payload["pad_token_id"]),
            bos_token_id=int(payload["bos_token_id"]),
            eos_token_id=int(payload["eos_token_id"]),
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids = [self.char_to_id.get(char, self.unknown_token_id) for char in text]
        if add_special_tokens:
            return [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_char[token_id] for token_id in token_ids if token_id in self.id_to_char)

    def save(self, path: Path) -> None:
        payload = {
            "type": "char_vocab",
            "chars": self.chars,
            "regular_token_limit": self.regular_token_limit,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "vocab_size": self.vocab_size,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")


@dataclass
class Example:
    input_ids: list[int]
    label_ids: list[int]
    response_type: str


def fit_prompt_and_answer(prompt_ids: list[int], answer_ids: list[int], max_seq_length: int) -> tuple[list[int], list[int]]:
    full_input_ids = prompt_ids + answer_ids
    full_label_ids = ([-100] * len(prompt_ids)) + answer_ids
    if len(full_input_ids) <= max_seq_length:
        return full_input_ids, full_label_ids

    if len(answer_ids) >= max_seq_length:
        trimmed_answer = answer_ids[-max_seq_length:]
        return trimmed_answer, trimmed_answer[:]

    prompt_budget = max_seq_length - len(answer_ids)
    trimmed_prompt = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
    input_ids = trimmed_prompt + answer_ids
    label_ids = ([-100] * len(trimmed_prompt)) + answer_ids
    return input_ids, label_ids


def build_examples(rows: list[dict], tokenizer: ByteTokenizer, max_seq_length: int) -> list[Example]:
    examples = []
    for row in rows:
        prompt_text = render_messages(row["messages"][:2], add_generation_prompt=True)
        answer_text = row["messages"][2]["content"]

        prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids, label_ids = fit_prompt_and_answer(prompt_ids, answer_ids, max_seq_length)
        examples.append(
            Example(
                input_ids=input_ids,
                label_ids=label_ids,
                response_type=extract_example_response_type(answer_text),
            )
        )
    return examples


def build_tokenizer_from_rows(*row_sets: list[dict]) -> ByteTokenizer:
    chars = set()
    for prefix in ROLE_PREFIXES.values():
        chars.update(prefix)
    for rows in row_sets:
        for row in rows:
            for message in row["messages"]:
                chars.update(message["content"])
    return ByteTokenizer(chars=sorted(chars))


def collect_required_tokenizer_chars(*row_sets: list[dict]) -> set[str]:
    chars: set[str] = set()
    for prefix in ROLE_PREFIXES.values():
        chars.update(prefix)
    for rows in row_sets:
        for row in rows:
            for message in row["messages"]:
                chars.update(message["content"])
    return chars


def create_batches(examples: list[Example], batch_size: int, shuffle: bool) -> list[list[Example]]:
    items = examples[:]
    if shuffle:
        random.shuffle(items)
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def collate_batch(batch: list[Example], tokenizer: ByteTokenizer, torch_module) -> tuple[object, object]:
    max_length = max(len(example.input_ids) for example in batch)
    input_rows = []
    label_rows = []
    for example in batch:
        pad_count = max_length - len(example.input_ids)
        input_rows.append(example.input_ids + ([tokenizer.pad_token_id] * pad_count))
        label_rows.append(example.label_ids + ([-100] * pad_count))
    input_ids = torch_module.tensor(input_rows, dtype=torch_module.long)
    labels = torch_module.tensor(label_rows, dtype=torch_module.long)
    return input_ids, labels


def compute_response_type_weights(
    examples: list[Example],
    enabled: bool,
    weight_power: float,
) -> dict[str, float]:
    if not enabled or not examples:
        return {}

    counts: dict[str, int] = {}
    for example in examples:
        counts[example.response_type] = counts.get(example.response_type, 0) + 1

    if not counts:
        return {}

    total_examples = len(examples)
    total_classes = len(counts)
    normalized_weights = {
        response_type: ((total_examples / (total_classes * count)) ** weight_power)
        for response_type, count in counts.items()
    }
    mean_weight = sum(normalized_weights.values()) / len(normalized_weights)
    if mean_weight <= 0:
        return {}
    return {
        response_type: (weight / mean_weight)
        for response_type, weight in normalized_weights.items()
    }


def compute_batch_loss(
    logits,
    labels,
    batch: list[Example],
    vocab_size: int,
    class_weights: dict[str, float],
    torch_module,
) -> object:
    shifted_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shifted_labels = labels[:, 1:].reshape(-1)
    if not class_weights:
        return torch_module.nn.functional.cross_entropy(
            shifted_logits,
            shifted_labels,
            ignore_index=-100,
        )

    token_losses = torch_module.nn.functional.cross_entropy(
        shifted_logits,
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape[0], labels.shape[1] - 1)
    valid_mask = (labels[:, 1:] != -100).to(token_losses.dtype)
    token_counts = valid_mask.sum(dim=1).clamp_min(1.0)
    example_losses = (token_losses * valid_mask).sum(dim=1) / token_counts
    weight_values = [
        float(class_weights.get(example.response_type, 1.0))
        for example in batch
    ]
    example_weights = torch_module.tensor(weight_values, dtype=example_losses.dtype, device=example_losses.device)
    return (example_losses * example_weights).sum() / example_weights.sum().clamp_min(1.0)


class TrainingStatusWriter:
    def __init__(self, output_dir: Path, config: dict):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "training_status.json"
        self.state = {
            "status": "initializing",
            "started_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "completed_at": None,
            "pid": os.getpid(),
            "output_dir": str(output_dir),
            "train_file": config["train_file"],
            "validation_file": config["validation_file"],
            "benchmark_file": config.get("benchmark_file", "data/processed/benchmark_sft.jsonl"),
            "global_step": 0,
            "epoch": 0.0,
            "max_steps": None,
            "latest_log": {},
            "error": None,
            "traceback": None,
        }
        self.write()

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        self.state["updated_at"] = utc_now_iso()
        self.write()

    def write(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2, ensure_ascii=True)
            handle.write("\n")


def run_post_training_benchmark(repo_root: Path, output_dir: Path) -> Path:
    report_path = repo_root / "experiments" / f"benchmark_report-{output_dir.name}.json"
    status_path = output_dir / "benchmark_status.json"
    training_config = load_json(output_dir / "training_config.json")
    benchmark_file = resolve_path(
        training_config.get("benchmark_file", "data/processed/benchmark_sft.jsonl"),
        repo_root,
    )
    command = [
        os.sys.executable,
        str(repo_root / "scripts" / "evaluate_native_model.py"),
        "--model-path",
        str(output_dir),
        "--benchmark-file",
        str(benchmark_file),
        "--output-report",
        str(report_path),
        "--status-file",
        str(status_path),
        "--max-new-tokens",
        "384",
    ]
    subprocess.run(command, check=True, cwd=repo_root)
    return report_path


def refresh_training_summary(repo_root: Path, apply_retention_cleanup: bool = False) -> None:
    command = [os.sys.executable, str(repo_root / "scripts" / "summarize_training_runs.py")]
    if apply_retention_cleanup:
        command.append("--apply-retention-cleanup")
    subprocess.run(command, capture_output=True, text=True, cwd=repo_root)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def build_cpu_state_dict(model) -> dict:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def build_token_id_remap(source_tokenizer: ByteTokenizer, target_tokenizer: ByteTokenizer) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []

    for char, source_id in source_tokenizer.char_to_id.items():
        target_id = target_tokenizer.char_to_id.get(char)
        if target_id is not None:
            pairs.append((source_id, target_id))

    special_pairs = [
        (source_tokenizer.pad_token_id, target_tokenizer.pad_token_id),
        (source_tokenizer.bos_token_id, target_tokenizer.bos_token_id),
        (source_tokenizer.eos_token_id, target_tokenizer.eos_token_id),
    ]
    for source_id, target_id in special_pairs:
        pairs.append((int(source_id), int(target_id)))

    unique_pairs: list[tuple[int, int]] = []
    seen_targets: set[int] = set()
    for source_id, target_id in pairs:
        if target_id in seen_targets:
            continue
        seen_targets.add(target_id)
        unique_pairs.append((int(source_id), int(target_id)))
    return unique_pairs


def remap_vocab_matrix(
    source_tensor,
    target_tensor,
    token_pairs: list[tuple[int, int]],
    fallback_source_id: int | None = None,
):
    remapped = target_tensor.clone()
    source_rows = int(source_tensor.shape[0])
    target_rows = int(target_tensor.shape[0])
    copied_targets: set[int] = set()

    if fallback_source_id is not None and 0 <= int(fallback_source_id) < source_rows and target_rows > 0:
        fallback_row = source_tensor[int(fallback_source_id)].to(dtype=target_tensor.dtype, device=target_tensor.device)
        remapped[:] = fallback_row

    for source_id, target_id in token_pairs:
        if source_id < 0 or target_id < 0:
            continue
        if source_id >= source_rows or target_id >= target_rows:
            continue
        remapped[target_id] = source_tensor[source_id].to(dtype=target_tensor.dtype, device=target_tensor.device)
        copied_targets.add(int(target_id))
    return remapped


def transfer_position_embedding(source_tensor, target_tensor):
    transferred = target_tensor.clone()
    if len(source_tensor.shape) != 2 or len(target_tensor.shape) != 2:
        return transferred
    if int(source_tensor.shape[1]) != int(target_tensor.shape[1]):
        return transferred

    rows_to_copy = min(int(source_tensor.shape[0]), int(target_tensor.shape[0]))
    if rows_to_copy <= 0:
        return transferred
    transferred[:rows_to_copy] = source_tensor[:rows_to_copy].to(
        dtype=target_tensor.dtype,
        device=target_tensor.device,
    )
    return transferred


def load_stage2_weights(model, init_model_dir: Path, target_tokenizer: ByteTokenizer, torch_module) -> dict[str, object]:
    state_path = init_model_dir / "model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Initial model checkpoint not found: {state_path}")

    source_state = torch_module.load(state_path, map_location="cpu", weights_only=False)
    source_tokenizer = ByteTokenizer.from_config(load_json(init_model_dir / "tokenizer_config.json"))
    token_pairs = build_token_id_remap(source_tokenizer, target_tokenizer)
    target_state = model.state_dict()
    updated_state = dict(target_state)
    shared_vocab_transfer = None
    source_vocab_tensor = source_state.get("token_embedding.weight")
    if source_vocab_tensor is None:
        source_vocab_tensor = source_state.get("lm_head.weight")
    if (
        source_vocab_tensor is not None
        and "token_embedding.weight" in target_state
        and "lm_head.weight" in target_state
        and len(target_state["token_embedding.weight"].shape) == 2
        and len(source_vocab_tensor.shape) == 2
        and int(target_state["token_embedding.weight"].shape[1]) == int(source_vocab_tensor.shape[1])
    ):
        shared_vocab_transfer = remap_vocab_matrix(
            source_tensor=source_vocab_tensor,
            target_tensor=target_state["token_embedding.weight"],
            token_pairs=token_pairs,
            fallback_source_id=source_tokenizer.unknown_token_id,
        )
        updated_state["token_embedding.weight"] = shared_vocab_transfer
        updated_state["lm_head.weight"] = shared_vocab_transfer
    loaded_names: list[str] = []
    skipped_names: list[str] = []

    for name, tensor in source_state.items():
        if name not in target_state:
            skipped_names.append(name)
            continue
        if name in {"token_embedding.weight", "lm_head.weight"}:
            if shared_vocab_transfer is not None and target_state[name].shape == shared_vocab_transfer.shape:
                loaded_names.append(name)
                continue
        if target_state[name].shape == tensor.shape:
            updated_state[name] = tensor
            loaded_names.append(name)
            continue
        if name == "position_embedding.weight":
            updated_state[name] = transfer_position_embedding(
                source_tensor=tensor,
                target_tensor=target_state[name],
            )
            loaded_names.append(name)
            continue
        skipped_names.append(name)

    model.load_state_dict(updated_state)
    return {
        "loaded_count": len(loaded_names),
        "skipped_count": len(skipped_names),
        "loaded_names": loaded_names,
        "skipped_names": skipped_names,
    }


class DirectMLAdamW:
    def __init__(
        self,
        torch_module,
        params,
        lr: float,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.torch = torch_module
        self.params = [param for param in params if param.requires_grad]
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.state: dict[int, dict[str, object]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)

        with self.torch.no_grad():
            for param in self.params:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("DirectMLAdamW does not support sparse gradients.")

                state = self.state.get(id(param))
                if state is None:
                    state = {
                        "exp_avg": self.torch.zeros_like(param),
                        "exp_avg_sq": self.torch.zeros_like(param),
                    }
                    self.state[id(param)] = state

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(self.beta1).add_(grad, alpha=(1.0 - self.beta1))
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=(1.0 - self.beta2))

                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(self.eps)
                step_size = self.lr / bias_correction1

                if self.weight_decay != 0.0:
                    param.add_(param, alpha=(-self.lr * self.weight_decay))
                param.addcdiv_(exp_avg, denom, value=-step_size)


def create_optimizer(torch_module, model, config: dict, device_label: str):
    optimizer_name = str(config.get("optimizer", "auto")).strip().lower()
    lr = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.01))
    betas = tuple(config.get("adam_betas", [0.9, 0.999]))
    eps = float(config.get("adam_eps", 1e-8))

    if optimizer_name == "auto":
        optimizer_name = "adamw"

    if optimizer_name == "adamw_dml_safe":
        optimizer = DirectMLAdamW(
            torch_module,
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
        )
        return optimizer, "adamw_dml_safe"

    if optimizer_name == "adamw":
        optimizer = torch_module.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            foreach=False,
        )
        return optimizer, "adamw"

    raise ValueError("Unsupported optimizer. Expected one of: auto, adamw, adamw_dml_safe.")


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
    percent = ratio * 100.0
    return f"[{bar}] {current}/{total} steps ({percent:5.1f}%)"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress_block(
    current: int,
    total: int,
    epoch: float,
    epoch_index: int,
    total_epochs: int,
    batch_in_epoch: int,
    batches_per_epoch: int,
    elapsed_seconds: float,
    steps_per_second: float,
    train_loss: float | None = None,
    validation_loss: float | None = None,
) -> None:
    parts = [render_progress_bar(current, total)]
    details = [
        f"epoch={epoch_index}/{total_epochs} ({epoch:.2f})",
        f"batch={batch_in_epoch}/{batches_per_epoch}",
        f"elapsed={format_duration(elapsed_seconds)}",
        f"steps_per_sec={steps_per_second:.3f}",
    ]
    if train_loss is not None:
        details.append(f"train_loss={train_loss:.4f}")
    if validation_loss is not None:
        details.append(f"val_loss={validation_loss:.4f}")
    line = " | ".join(parts + details)
    if sys.stdout.isatty():
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
    else:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config_path = resolve_path(args.config, repo_root)
    config = load_config(config_path)
    if args.num_train_epochs is not None:
        if args.num_train_epochs < 1:
            raise ValueError("--num-train-epochs must be at least 1")
        config["num_train_epochs"] = int(args.num_train_epochs)
    configure_reproducibility(int(config.get("seed", 11)))

    train_file = resolve_path(config["train_file"], repo_root)
    validation_file = resolve_path(config["validation_file"], repo_root)
    output_dir_base = resolve_path(config["output_dir"], repo_root)
    benchmark_file = resolve_path(config.get("benchmark_file", "data/processed/benchmark_sft.jsonl"), repo_root)

    import torch
    import torch.nn.functional as F
    from torch import nn

    device, device_label = detect_device(torch, config.get("device_preference", ["cuda", "cpu"]))
    torch.backends.cudnn.enabled = True
    if device.type == 'cuda':
        use_tf32 = bool(config.get("use_tf32", True))
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = True
    else:
        use_tf32 = False

    use_amp = bool(config.get("use_amp", device.type == 'cuda')) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    train_rows = load_jsonl(train_file)
    validation_rows = load_jsonl(validation_file)
    required_tokenizer_chars = collect_required_tokenizer_chars(train_rows, validation_rows)
    init_model_dir = resolve_init_model_path(
        config.get("init_from_model_path"),
        repo_root,
        required_tokenizer_chars=required_tokenizer_chars,
    )
    tokenizer = None
    tokenizer_reused_from_init = False
    if init_model_dir is not None:
        init_tokenizer_path = init_model_dir / "tokenizer_config.json"
        if init_tokenizer_path.exists():
            tokenizer = ByteTokenizer.from_config(load_json(init_tokenizer_path))
            missing_chars = required_tokenizer_chars.difference(set(tokenizer.chars))
            if missing_chars:
                print(
                    "Tokenizer compatibility: stage-2 tokenizer missing required chars; "
                    f"rebuilding tokenizer with {len(missing_chars)} added chars."
                )
                tokenizer = None
            else:
                tokenizer_reused_from_init = True
    if tokenizer is None:
        tokenizer = build_tokenizer_from_rows(train_rows, validation_rows)
    max_seq_length = int(config["max_seq_length"])
    train_examples = build_examples(train_rows, tokenizer, max_seq_length)
    validation_examples = build_examples(validation_rows, tokenizer, max_seq_length)

    class CausalSelfAttention(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int, dropout: float):
            super().__init__()
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads")
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.qkv = nn.Linear(hidden_size, hidden_size * 3)
            self.proj = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            batch_size, seq_len, hidden_size = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            dropout_p = self.dropout.p if self.training else 0.0
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            return self.proj(attn)

    class Block(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
            super().__init__()
            mlp_hidden = int(hidden_size * mlp_ratio)
            self.norm_1 = nn.LayerNorm(hidden_size)
            self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
            self.norm_2 = nn.LayerNorm(hidden_size)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, hidden_size),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.norm_1(x))
            x = x + self.mlp(self.norm_2(x))
            return x

    class NativeTransformerLM(nn.Module):
        def __init__(self, model_config: dict):
            super().__init__()
            self.vocab_size = int(model_config.get("vocab_size", tokenizer.vocab_size))
            self.max_seq_length = int(model_config["max_seq_length"])
            self.token_embedding = nn.Embedding(self.vocab_size, int(model_config["hidden_size"]))
            self.position_embedding = nn.Embedding(self.max_seq_length, int(model_config["hidden_size"]))
            self.dropout = nn.Dropout(float(model_config["dropout"]))
            self.blocks = nn.ModuleList(
                [
                    Block(
                        hidden_size=int(model_config["hidden_size"]),
                        num_heads=int(model_config["num_heads"]),
                        mlp_ratio=float(model_config["mlp_ratio"]),
                        dropout=float(model_config["dropout"]),
                    )
                    for _ in range(int(model_config["num_layers"]))
                ]
            )
            self.norm = nn.LayerNorm(int(model_config["hidden_size"]))
            self.lm_head = nn.Linear(int(model_config["hidden_size"]), self.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight

        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            if seq_len > self.max_seq_length:
                raise ValueError("sequence length exceeds model max_seq_length")
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = self.dropout(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return self.lm_head(x)

    total_batches_per_epoch = max(1, math.ceil(len(train_examples) / int(config["batch_size"])))
    total_steps = total_batches_per_epoch * int(config["num_train_epochs"])

    output_dir = make_run_output_dir(output_dir_base)
    status_writer = TrainingStatusWriter(output_dir=output_dir, config=config)

    print("Starting native model training")
    print(f"Config: {config_path}")
    print(f"Device: {device_label}")
    print(f"Train file: {train_file}")
    print(f"Validation file: {validation_file}")
    print(f"Benchmark file: {benchmark_file}")
    print(f"Output dir: {output_dir}")
    if tokenizer_reused_from_init:
        print(f"Tokenizer source: stage-2 init model ({init_model_dir})")
    else:
        print("Tokenizer source: rebuilt from train+validation rows")
    print(
        "Run settings: "
        f"epochs={int(config['num_train_epochs'])}, "
        f"batch_size={int(config['batch_size'])}, "
        f"eval_batch_size={int(config['eval_batch_size'])}, "
        f"max_seq_length={max_seq_length}, "
        f"use_amp={use_amp}, "
        f"use_tf32={use_tf32}"
    )
    print(
        "Model settings: "
        f"layers={int(config['num_layers'])}, "
        f"hidden_size={int(config['hidden_size'])}, "
        f"heads={int(config['num_heads'])}, "
        f"dropout={float(config['dropout'])}"
    )
    model = NativeTransformerLM(config).to(device)
    if init_model_dir is not None:
        init_summary = load_stage2_weights(model, init_model_dir, tokenizer, torch)
        config["init_from_model_path_resolved"] = str(init_model_dir)
        print(f"Stage-2 initialization: {init_model_dir}")
        print(
            "Initialized matching weights: "
            f"loaded={init_summary['loaded_count']}, "
            f"skipped={init_summary['skipped_count']} "
            "(token embeddings and output head are remapped when tokenizer vocab differs)"
        )
    elif config.get("init_from_model_path"):
        print(
            "Stage-2 initialization: no matching prior run found for "
            f"{config.get('init_from_model_path')}; starting from scratch."
        )
    optimizer, optimizer_label = create_optimizer(torch, model, config, device_label)
    print(f"Optimizer: {optimizer_label}")
    print(
        "Dataset sizes: "
        f"train_rows={len(train_rows)}, "
        f"validation_rows={len(validation_rows)}, "
        f"train_examples={len(train_examples)}, "
        f"validation_examples={len(validation_examples)}"
    )
    print(
        "Progress calculation: "
        f"total_steps = ceil(train_examples / batch_size) * num_train_epochs = "
        f"{total_batches_per_epoch} * {int(config['num_train_epochs'])} = {total_steps}"
    )
    class_weights = compute_response_type_weights(
        train_examples,
        enabled=bool(config.get("use_class_balanced_loss", False)),
        weight_power=float(config.get("class_weight_power", 1.0)),
    )
    if class_weights:
        print(f"Class-balanced loss weights: {class_weights}")

    status_writer.update(
        status="ready_to_train",
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        device=device_label,
        max_steps=total_steps,
        benchmark_status=None,
    )

    def evaluate_loss(examples: list[Example]) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in create_batches(examples, int(config["eval_batch_size"]), shuffle=False):
                input_ids, labels = collate_batch(batch, tokenizer, torch)
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(input_ids)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Non-finite logits encountered during validation.")
                    loss = F.cross_entropy(
                        logits[:, :-1, :].reshape(-1, model.vocab_size),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                losses.append(float(loss.item()))
                del logits
                del loss
                del input_ids
                del labels
        model.train()
        gc.collect()
        return sum(losses) / len(losses) if losses else 0.0

    global_step = 0
    best_validation_loss = float("inf")
    best_model_state = build_cpu_state_dict(model)
    training_started_at = time.perf_counter()
    initial_validation_loss = evaluate_loss(validation_examples)
    best_validation_loss = initial_validation_loss
    print(f"Initial validation loss (pre-train checkpoint): {initial_validation_loss:.4f}")
    try:
        model.train()
        for epoch_index in range(int(config["num_train_epochs"])):
            epoch_batches = create_batches(train_examples, int(config["batch_size"]), shuffle=True)
            for batch_offset, batch in enumerate(epoch_batches, start=1):
                global_step += 1
                input_ids, labels = collate_batch(batch, tokenizer, torch)
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(input_ids)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(
                            f"Non-finite logits encountered during training at step {global_step}."
                        )
                    loss = compute_batch_loss(
                        logits=logits,
                        labels=labels,
                        batch=batch,
                        vocab_size=model.vocab_size,
                        class_weights=class_weights,
                        torch_module=torch,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss encountered at step {global_step}."
                    )
                loss_value = float(loss.item())
                if use_amp:
                    scaler.scale(loss).backward()
                    grad_clip = float(config.get("grad_clip", 1.0))
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_clip = float(config.get("grad_clip", 1.0))
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                current_epoch = epoch_index + (batch_offset / total_batches_per_epoch)
                if global_step == 1 or global_step % int(config["logging_steps"]) == 0 or global_step == total_steps:
                    elapsed_seconds, steps_per_second, samples_per_second = compute_training_metrics(
                        started_at=training_started_at,
                        global_step=global_step,
                        train_examples=len(train_examples),
                        num_train_epochs=int(config["num_train_epochs"]),
                    )
                    status_writer.update(
                        status="training",
                        global_step=global_step,
                        epoch=round(current_epoch, 4),
                        latest_log={
                            "train_loss": loss_value,
                            "epoch": round(current_epoch, 4),
                            "train_runtime": elapsed_seconds,
                            "train_steps_per_second": steps_per_second,
                            "train_samples_per_second": samples_per_second,
                        },
                    )
                    print_progress_block(
                        current=global_step,
                        total=total_steps,
                        epoch=current_epoch,
                        epoch_index=epoch_index + 1,
                        total_epochs=int(config["num_train_epochs"]),
                        batch_in_epoch=batch_offset,
                        batches_per_epoch=total_batches_per_epoch,
                        elapsed_seconds=elapsed_seconds,
                        steps_per_second=steps_per_second,
                        train_loss=loss_value,
                    )

                if global_step % int(config["eval_steps"]) == 0 or global_step == total_steps:
                    validation_loss = evaluate_loss(validation_examples)
                    elapsed_seconds, steps_per_second, samples_per_second = compute_training_metrics(
                        started_at=training_started_at,
                        global_step=global_step,
                        train_examples=len(train_examples),
                        num_train_epochs=int(config["num_train_epochs"]),
                    )
                    latest_log = dict(status_writer.state.get("latest_log", {}))
                    latest_log["validation_loss"] = validation_loss
                    latest_log["epoch"] = round(current_epoch, 4)
                    latest_log["train_runtime"] = elapsed_seconds
                    latest_log["train_steps_per_second"] = steps_per_second
                    latest_log["train_samples_per_second"] = samples_per_second
                    status_writer.update(
                        status="training",
                        global_step=global_step,
                        epoch=round(current_epoch, 4),
                        latest_log=latest_log,
                    )
                    print_progress_block(
                        current=global_step,
                        total=total_steps,
                        epoch=current_epoch,
                        epoch_index=epoch_index + 1,
                        total_epochs=int(config["num_train_epochs"]),
                        batch_in_epoch=batch_offset,
                        batches_per_epoch=total_batches_per_epoch,
                        elapsed_seconds=elapsed_seconds,
                        steps_per_second=steps_per_second,
                        train_loss=loss_value,
                        validation_loss=validation_loss,
                    )
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_model_state = build_cpu_state_dict(model)

                del logits
                del loss
                del input_ids
                del labels

        total_train_loss = float(status_writer.state.get("latest_log", {}).get("train_loss", 0.0))
        elapsed_seconds, steps_per_second, samples_per_second = compute_training_metrics(
            started_at=training_started_at,
            global_step=global_step,
            train_examples=len(train_examples),
            num_train_epochs=int(config["num_train_epochs"]),
        )
        sys.stdout.write("\n")
        model_path = output_dir / "model.pt"
        torch.save(best_model_state, model_path)
        tokenizer.save(output_dir / "tokenizer_config.json")
        save_json(
            output_dir / "model_config.json",
            {
                "model_type": "native_byte_transformer",
                "vocab_size": tokenizer.vocab_size,
                "max_seq_length": int(config["max_seq_length"]),
                "hidden_size": int(config["hidden_size"]),
                "num_layers": int(config["num_layers"]),
                "num_heads": int(config["num_heads"]),
                "mlp_ratio": float(config["mlp_ratio"]),
                "dropout": float(config["dropout"]),
            },
        )
        save_json(output_dir / "training_config.json", config)
        status_writer.update(
            status="running_benchmark",
            global_step=global_step,
            epoch=float(config["num_train_epochs"]),
            latest_log={
                "train_loss": total_train_loss,
                "validation_loss": best_validation_loss if best_validation_loss != float("inf") else None,
                "epoch": float(config["num_train_epochs"]),
                "train_runtime": elapsed_seconds,
                "train_steps_per_second": steps_per_second,
                "train_samples_per_second": samples_per_second,
            },
            best_validation_loss=None if best_validation_loss == float("inf") else best_validation_loss,
            benchmark_status=str(output_dir / "benchmark_status.json"),
        )
        print("Training loop complete. Starting automatic benchmark evaluation...")
        benchmark_report_path = run_post_training_benchmark(repo_root, output_dir)
        refresh_training_summary(repo_root, apply_retention_cleanup=True)
        status_writer.update(
            status="completed",
            completed_at=utc_now_iso(),
            global_step=global_step,
            epoch=float(config["num_train_epochs"]),
            latest_log={
                "train_loss": total_train_loss,
                "validation_loss": best_validation_loss if best_validation_loss != float("inf") else None,
                "epoch": float(config["num_train_epochs"]),
                "train_runtime": elapsed_seconds,
                "train_steps_per_second": steps_per_second,
                "train_samples_per_second": samples_per_second,
            },
            benchmark_report=str(benchmark_report_path),
            best_validation_loss=None if best_validation_loss == float("inf") else best_validation_loss,
            benchmark_status=str(output_dir / "benchmark_status.json"),
        )
        print(f"Training complete. Saved model to {output_dir}")
        print(f"Benchmark report: {benchmark_report_path}")
        return 0
    except Exception as exc:
        sys.stdout.write("\n")
        status_writer.update(
            status="failed",
            completed_at=utc_now_iso(),
            global_step=global_step,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
