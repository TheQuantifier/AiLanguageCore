import argparse
import json
from pathlib import Path
import sys
import time


ASCII_TOKEN_LIMIT = 128
PAD_TOKEN_ID = ASCII_TOKEN_LIMIT
BOS_TOKEN_ID = ASCII_TOKEN_LIMIT + 1
EOS_TOKEN_ID = ASCII_TOKEN_LIMIT + 2
VOCAB_SIZE = ASCII_TOKEN_LIMIT + 3

ROLE_PREFIXES = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
        regular_token_limit: int = ASCII_TOKEN_LIMIT,
        pad_token_id: int = PAD_TOKEN_ID,
        bos_token_id: int = BOS_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
    ):
        self.regular_token_limit = int(regular_token_limit)
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = max(self.regular_token_limit, self.pad_token_id, self.bos_token_id, self.eos_token_id) + 1

    @classmethod
    def from_config(cls, payload: dict) -> "ByteTokenizer":
        return cls(
            regular_token_limit=int(payload.get("regular_token_limit", payload.get("vocab_size", ASCII_TOKEN_LIMIT) - 3)),
            pad_token_id=int(payload["pad_token_id"]),
            bos_token_id=int(payload["bos_token_id"]),
            eos_token_id=int(payload["eos_token_id"]),
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids = list(text.encode("ascii", errors="replace"))
        if add_special_tokens:
            return [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids if 0 <= token_id < self.regular_token_limit)


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


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
    valid_json: int,
    correct_response_type: int,
    elapsed: float,
) -> None:
    rate = current / elapsed if elapsed > 0 else 0.0
    line = (
        f"{render_progress_bar(current, total)} | "
        f"nonempty_output={nonempty_output} | "
        f"valid_json={valid_json} | "
        f"correct_type={correct_response_type} | "
        f"items_per_sec={rate:.2f}"
    )
    sys.stdout.write("\r" + line + " " * 8)
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    output_report = resolve_output_report_path(args.model_path, args.output_report)

    import torch
    import torch.nn.functional as F  # noqa: F401
    from torch import nn

    device, _device_label = detect_device(torch)
    tokenizer_config_path = args.model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer = ByteTokenizer.from_config(load_json(tokenizer_config_path))
    else:
        tokenizer = ByteTokenizer()
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
    valid_json = 0
    started_at = time.perf_counter()

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
                next_token_id = int(torch.argmax(next_token_logits).item())
                generated_ids.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                generated_tensor = torch.cat((generated_tensor, next_token_tensor), dim=1)
                if next_token_id == tokenizer.eos_token_id:
                    break

            generated_text = tokenizer.decode(generated_ids[len(prompt_ids) :])
            if generated_text != "":
                nonempty_output += 1
            parsed = extract_json_object(generated_text)
            expected = json.loads(row["messages"][2]["content"])
            is_valid_json = parsed is not None
            if is_valid_json:
                valid_json += 1
            predicted_type = parsed.get("response_type") if parsed else None
            if predicted_type == expected["response_type"]:
                correct_response_type += 1
            results.append(
                {
                    "id": row["id"],
                    "user_input": row["messages"][1]["content"],
                    "expected_response_type": expected["response_type"],
                    "predicted_response_type": predicted_type,
                    "valid_json": is_valid_json,
                    "raw_generation": generated_text,
                }
            )
            print_progress(
                current=index,
                total=len(benchmark_rows),
                nonempty_output=nonempty_output,
                valid_json=valid_json,
                correct_response_type=correct_response_type,
                elapsed=time.perf_counter() - started_at,
            )

    report = {
        "benchmark_size": len(benchmark_rows),
        "valid_json_count": valid_json,
        "valid_json_rate": valid_json / len(benchmark_rows) if benchmark_rows else 0.0,
        "response_type_accuracy": correct_response_type / len(benchmark_rows) if benchmark_rows else 0.0,
        "results": results,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    sys.stdout.write("\n")
    print(f"Wrote benchmark report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
