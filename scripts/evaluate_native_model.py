import argparse
import json
from pathlib import Path
import sys
import time


ROLE_PREFIXES = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
}

RESPONSE_TYPES = [
    "DIRECT_ANSWER",
    "CLARIFICATION",
    "TOOL_NEEDED",
    "OUT_OF_SCOPE",
]

BENCHMARK_FILE_TO_TYPE = {
    "benchmark_sft.jsonl": "core",
    "benchmark_category_prediction_sft.jsonl": "core",
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
    "benchmark_full_response_sft.jsonl": "full_response",
    "benchmark_stress_native_sft.jsonl": "category_prediction",
    "benchmark_stress_v2_native_sft.jsonl": "category_prediction",
    "benchmark_account_tool_boundary_native_sft.jsonl": "category_prediction",
    "benchmark_medical_refusal_boundary_native_sft.jsonl": "category_prediction",
    "benchmark_oos_vs_tool_boundary_native_sft.jsonl": "category_prediction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a native byte-level transformer model against a benchmark set."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved native model run directory.")
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path("data/processed/benchmark_sft.jsonl"),
        help="Benchmark SFT JSONL file.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("experiments/benchmark_report.json"),
        help="Output benchmark report path.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Optional JSON file to update with live benchmark progress.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=16,
        help="Minimum number of tokens to generate before EOS is allowed.",
    )
    return parser.parse_args()


def resolve_output_report_path(model_path: Path, output_report: Path) -> Path:
    if output_report != Path("experiments/benchmark_report.json"):
        return output_report
    return output_report.parent / f"benchmark_report-{model_path.name}.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_type_from_benchmark_file(path: Path) -> str:
    return BENCHMARK_FILE_TO_TYPE.get(path.name, "custom")


def infer_category_from_benchmark_file(path: Path) -> str:
    return BENCHMARK_FILE_TO_CATEGORY.get(path.name, "custom")


def infer_training_type_from_model_path(model_path: Path) -> str:
    training_config_path = model_path / "training_config.json"
    if training_config_path.exists():
        try:
            training_config = load_json(training_config_path)
            benchmark_file = training_config.get("benchmark_file")
            if isinstance(benchmark_file, str) and benchmark_file.strip():
                return infer_type_from_benchmark_file(Path(benchmark_file))
        except Exception:
            pass
    if "-stress-" in model_path.name:
        return "stress"
    return "core"


def infer_training_category_from_model_path(model_path: Path) -> str:
    training_config_path = model_path / "training_config.json"
    if training_config_path.exists():
        try:
            training_config = load_json(training_config_path)
            train_file = training_config.get("train_file")
            if isinstance(train_file, str) and "full_response" in train_file:
                return "full_response"
            if isinstance(train_file, str) and "category_prediction" in train_file:
                return "category_prediction"
            benchmark_file = training_config.get("benchmark_file")
            if isinstance(benchmark_file, str) and benchmark_file.strip():
                return infer_category_from_benchmark_file(Path(benchmark_file))
        except Exception:
            pass
    if "full-response" in model_path.name:
        return "full_response"
    return "category_prediction"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_status(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def render_messages(messages: list[dict], add_generation_prompt: bool) -> str:
    parts = []
    for message in messages:
        parts.append(ROLE_PREFIXES[message["role"]])
        parts.append(message["content"])
        parts.append("\n")
    if add_generation_prompt:
        parts.append(ROLE_PREFIXES["assistant"])
    return "".join(parts)


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


def extract_response_type(text: str) -> str | None:
    stripped = text.strip()
    if stripped in RESPONSE_TYPES:
        return stripped
    for response_type in RESPONSE_TYPES:
        if response_type in stripped:
            return response_type
    return None


def parse_expected_payload(text: str) -> dict:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {
            "format": "label_only",
            "response_type": extract_response_type(stripped),
            "reason": None,
            "response": None,
            "raw": stripped,
        }

    if not isinstance(payload, dict):
        return {
            "format": "unknown_json",
            "response_type": None,
            "reason": None,
            "response": None,
            "raw": stripped,
        }

    response_type = payload.get("response_type")
    reason = payload.get("reason")
    response = payload.get("response")
    return {
        "format": "full_response",
        "response_type": response_type if isinstance(response_type, str) else None,
        "reason": reason if isinstance(reason, str) else None,
        "response": response if isinstance(response, str) else None,
        "raw": stripped,
    }


def parse_generated_payload(text: str) -> dict:
    stripped = text.strip()
    if stripped == "":
        return {
            "format": "empty",
            "valid_json": False,
            "valid_output": False,
            "response_type": None,
            "reason": None,
            "response": None,
            "raw": text,
        }

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        predicted_type = extract_response_type(stripped)
        return {
            "format": "label_only_text",
            "valid_json": False,
            "valid_output": predicted_type in RESPONSE_TYPES,
            "response_type": predicted_type,
            "reason": None,
            "response": None,
            "raw": text,
        }

    if not isinstance(payload, dict):
        return {
            "format": "json_non_object",
            "valid_json": False,
            "valid_output": False,
            "response_type": None,
            "reason": None,
            "response": None,
            "raw": text,
        }

    response_type = payload.get("response_type")
    reason = payload.get("reason")
    response = payload.get("response")
    valid_json = (
        isinstance(response_type, str)
        and response_type in RESPONSE_TYPES
        and isinstance(reason, str)
        and reason.strip() != ""
        and isinstance(response, str)
        and response.strip() != ""
    )
    return {
        "format": "full_response_json",
        "valid_json": valid_json,
        "valid_output": valid_json,
        "response_type": response_type if isinstance(response_type, str) else None,
        "reason": reason if isinstance(reason, str) else None,
        "response": response if isinstance(response, str) else None,
        "raw": text,
    }


def detect_device(torch_module) -> tuple[object, str]:
    hip_available = bool(getattr(torch_module.version, "hip", None)) and torch_module.cuda.is_available()
    if hip_available:
        return torch_module.device("cuda"), "hip"
    if torch_module.cuda.is_available():
        return torch_module.device("cuda"), "cuda"
    try:
        import torch_directml  # type: ignore
    except ImportError:
        torch_directml = None
    if torch_directml is not None and torch_directml.is_available() and int(torch_directml.device_count()) > 0:
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

        best_index = max(
            range(int(torch_directml.device_count())),
            key=lambda index: score_directml_name(str(torch_directml.device_name(index))),
        )
        device_name = str(torch_directml.device_name(best_index)).replace("\x00", "").strip()
        return torch_directml.device(best_index), f"directml:{best_index}:{device_name}"
    return torch_module.device("cpu"), "cpu"


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


def print_progress(
    current: int,
    total: int,
    nonempty_output: int,
    valid_output: int,
    correct_response_type: int,
    valid_full_response: int,
    elapsed: float,
) -> None:
    rate = current / elapsed if elapsed > 0 else 0.0
    line = (
        f"{render_progress_bar(current, total)} | "
        f"nonempty_output={nonempty_output} | "
        f"valid_output={valid_output} | "
        f"correct_type={correct_response_type} | "
        f"valid_full={valid_full_response} | "
        f"items_per_sec={rate:.2f}"
    )
    sys.stdout.write("\r" + line + " " * 8)
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    output_report = resolve_output_report_path(args.model_path, args.output_report)
    benchmark_type = infer_type_from_benchmark_file(args.benchmark_file)
    training_type = infer_training_type_from_model_path(args.model_path)
    benchmark_category = infer_category_from_benchmark_file(args.benchmark_file)
    training_category = infer_training_category_from_model_path(args.model_path)

    import torch
    import torch.nn.functional as F  # noqa: F401
    from torch import nn

    device, _device_label = detect_device(torch)
    tokenizer_config_path = args.model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer = ByteTokenizer.from_config(load_json(tokenizer_config_path))
    else:
        tokenizer = ByteTokenizer(chars=[chr(index) for index in range(128)])
    model_config = load_json(args.model_path / "model_config.json")

    class CausalSelfAttention(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int, dropout: float):
            super().__init__()
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
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
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
        def __init__(self):
            super().__init__()
            hidden_size = int(model_config["hidden_size"])
            self.max_seq_length = int(model_config["max_seq_length"])
            self.token_embedding = nn.Embedding(int(model_config["vocab_size"]), hidden_size)
            self.position_embedding = nn.Embedding(self.max_seq_length, hidden_size)
            self.dropout = nn.Dropout(float(model_config["dropout"]))
            self.blocks = nn.ModuleList(
                [
                    Block(
                        hidden_size=hidden_size,
                        num_heads=int(model_config["num_heads"]),
                        mlp_ratio=float(model_config["mlp_ratio"]),
                        dropout=float(model_config["dropout"]),
                    )
                    for _ in range(int(model_config["num_layers"]))
                ]
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, int(model_config["vocab_size"]), bias=False)

        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = self.dropout(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return self.lm_head(x)

    model = NativeTransformerLM()
    state_dict = torch.load(args.model_path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    benchmark_rows = load_jsonl(args.benchmark_file)
    results = []
    correct_response_type = 0
    nonempty_output = 0
    valid_output = 0
    valid_json = 0
    valid_full_response = 0
    started_at = time.perf_counter()
    status_file = args.status_file.resolve() if args.status_file else None

    write_status(
        status_file,
        {
            "status": "running",
            "model_path": str(args.model_path.resolve()),
            "benchmark_file": str(args.benchmark_file.resolve()),
            "output_report": str(output_report.resolve()),
            "current": 0,
            "total": len(benchmark_rows),
            "nonempty_output_count": 0,
            "valid_json_count": 0,
            "valid_output_count": 0,
            "correct_response_type_count": 0,
            "valid_full_response_count": 0,
            "items_per_sec": 0.0,
        },
    )

    print(f"Evaluating {len(benchmark_rows)} benchmark items from {args.benchmark_file}")

    with torch.no_grad():
        for index, row in enumerate(benchmark_rows, start=1):
            prompt_text = render_messages(row["messages"][:2], add_generation_prompt=True)
            prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
            generated_ids = prompt_ids[:]
            generated_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            for generated_token_count in range(args.max_new_tokens):
                current_tensor = generated_tensor[:, -int(model_config["max_seq_length"]) :]
                logits = model(current_tensor)
                next_token_logits = logits[0, -1].clone()
                for token_id in {tokenizer.pad_token_id, tokenizer.bos_token_id}:
                    next_token_logits[token_id] = float("-inf")
                if generated_token_count < int(args.min_new_tokens):
                    next_token_logits[tokenizer.eos_token_id] = float("-inf")
                newline_token_id = tokenizer.char_to_id.get("\n")
                if newline_token_id is not None and generated_token_count < 8:
                    next_token_logits[newline_token_id] = float("-inf")
                next_token_id = int(torch.argmax(next_token_logits).item())
                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                generated_tensor = torch.cat((generated_tensor, next_token_tensor), dim=1)
                if next_token_id == tokenizer.eos_token_id:
                    break

            generated_text = tokenizer.decode(generated_ids[len(prompt_ids) :])
            if generated_text != "":
                nonempty_output += 1
            parsed_generated = parse_generated_payload(generated_text)
            parsed_expected = parse_expected_payload(row["messages"][2]["content"])
            predicted_type = parsed_generated["response_type"]
            expected_type = parsed_expected["response_type"]
            expected_format = parsed_expected["format"]
            is_valid_json = False
            if expected_format == "full_response":
                is_valid_json = bool(parsed_generated["format"] == "full_response_json" and parsed_generated["valid_json"])
            elif expected_format == "label_only":
                is_valid_json = predicted_type in RESPONSE_TYPES
            is_valid_output = is_valid_json
            if is_valid_json:
                valid_json += 1
            if is_valid_output:
                valid_output += 1
            if parsed_generated["format"] == "full_response_json" and parsed_generated["valid_json"]:
                valid_full_response += 1
            if predicted_type == expected_type:
                correct_response_type += 1
            results.append(
                {
                    "id": row["id"],
                    "user_input": row["messages"][1]["content"],
                    "expected_response_type": expected_type,
                    "predicted_response_type": predicted_type,
                    "expected_format": expected_format,
                    "predicted_format": parsed_generated["format"],
                    "valid_json": is_valid_json,
                    "valid_output": is_valid_output,
                    "expected_reason": parsed_expected["reason"],
                    "predicted_reason": parsed_generated["reason"],
                    "expected_response": parsed_expected["response"],
                    "predicted_response": parsed_generated["response"],
                    "raw_generation": generated_text,
                }
            )
            print_progress(
                current=index,
                total=len(benchmark_rows),
                nonempty_output=nonempty_output,
                valid_output=valid_output,
                correct_response_type=correct_response_type,
                valid_full_response=valid_full_response,
                elapsed=time.perf_counter() - started_at,
            )
            elapsed = time.perf_counter() - started_at
            write_status(
                status_file,
                {
                    "status": "running",
                    "model_path": str(args.model_path.resolve()),
                    "benchmark_file": str(args.benchmark_file.resolve()),
                    "output_report": str(output_report.resolve()),
                    "current": index,
                    "total": len(benchmark_rows),
                    "valid_json_count": valid_json,
                    "nonempty_output_count": nonempty_output,
                    "valid_output_count": valid_output,
                    "correct_response_type_count": correct_response_type,
                    "valid_full_response_count": valid_full_response,
                    "items_per_sec": (index / elapsed) if elapsed > 0 else 0.0,
                },
            )

    elapsed_seconds = time.perf_counter() - started_at
    report = {
        "training_type": training_type,
        "training_category": training_category,
        "benchmark_type": benchmark_type,
        "benchmark_category": benchmark_category,
        "benchmark_file": str(args.benchmark_file.resolve()),
        "benchmark_size": len(benchmark_rows),
        "nonempty_output_count": nonempty_output,
        "nonempty_output_rate": nonempty_output / len(benchmark_rows) if benchmark_rows else 0.0,
        "valid_json_count": valid_json,
        "valid_json_rate": valid_json / len(benchmark_rows) if benchmark_rows else 0.0,
        "valid_output_count": valid_output,
        "valid_output_rate": valid_output / len(benchmark_rows) if benchmark_rows else 0.0,
        "valid_full_response_count": valid_full_response,
        "valid_full_response_rate": valid_full_response / len(benchmark_rows) if benchmark_rows else 0.0,
        "correct_response_type_count": correct_response_type,
        "response_type_accuracy": correct_response_type / len(benchmark_rows) if benchmark_rows else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "items_per_sec": (len(benchmark_rows) / elapsed_seconds) if benchmark_rows and elapsed_seconds > 0 else 0.0,
        "results": results,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    write_status(
        status_file,
        {
            "status": "completed",
            "model_path": str(args.model_path.resolve()),
            "benchmark_file": str(args.benchmark_file.resolve()),
            "output_report": str(output_report.resolve()),
            "current": len(benchmark_rows),
            "total": len(benchmark_rows),
            "valid_json_count": valid_json,
            "nonempty_output_count": nonempty_output,
            "valid_output_count": valid_output,
            "correct_response_type_count": correct_response_type,
            "valid_full_response_count": valid_full_response,
            "items_per_sec": report["items_per_sec"],
            "valid_output_rate": report["valid_output_rate"],
            "valid_json_rate": report["valid_json_rate"],
            "response_type_accuracy": report["response_type_accuracy"],
        },
    )
    sys.stdout.write("\n")
    print(f"Wrote benchmark report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
