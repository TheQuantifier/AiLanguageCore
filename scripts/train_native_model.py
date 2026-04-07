import argparse
import json
import math
import os
import random
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PAD_TOKEN_ID = 256
BOS_TOKEN_ID = 257
EOS_TOKEN_ID = 258
VOCAB_SIZE = 259

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
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        device_preference = ["hip", "cuda", "cpu"]

    hip_available = bool(getattr(torch_module.version, "hip", None)) and torch_module.cuda.is_available()
    cuda_available = torch_module.cuda.is_available() and not hip_available

    for item in device_preference:
        normalized = str(item).lower()
        if normalized == "hip" and hip_available:
            return torch_module.device("cuda"), "hip"
        if normalized == "cuda" and cuda_available:
            return torch_module.device("cuda"), "cuda"
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


class ByteTokenizer:
    pad_token_id = PAD_TOKEN_ID
    bos_token_id = BOS_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID
    vocab_size = VOCAB_SIZE

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        if add_special_tokens:
            return [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        payload = bytes(token_id for token_id in token_ids if 0 <= token_id <= 255)
        return payload.decode("utf-8", errors="ignore")

    def save(self, path: Path) -> None:
        payload = {
            "type": "byte_level",
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


def truncate_example(input_ids: list[int], label_ids: list[int], max_seq_length: int) -> tuple[list[int], list[int]]:
    if len(input_ids) <= max_seq_length:
        return input_ids, label_ids
    return input_ids[:max_seq_length], label_ids[:max_seq_length]


def build_examples(rows: list[dict], tokenizer: ByteTokenizer, max_seq_length: int) -> list[Example]:
    examples = []
    for row in rows:
        prompt_text = render_messages(row["messages"][:2], add_generation_prompt=True)
        answer_text = row["messages"][2]["content"]

        prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids = prompt_ids + answer_ids
        label_ids = ([-100] * len(prompt_ids)) + answer_ids
        input_ids, label_ids = truncate_example(input_ids, label_ids, max_seq_length)
        examples.append(Example(input_ids=input_ids, label_ids=label_ids))
    return examples


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


def run_post_training_benchmark(output_dir: Path) -> Path:
    command = [
        os.sys.executable,
        "scripts/evaluate_native_model.py",
        "--model-path",
        str(output_dir),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    report_path = None
    for line in result.stdout.splitlines():
        prefix = "Wrote benchmark report to "
        if line.startswith(prefix):
            report_path = Path(line[len(prefix) :].strip())
            break
    if report_path is None:
        report_path = Path("experiments") / f"benchmark_report-{output_dir.name}.json"
    return report_path


def refresh_training_summary() -> None:
    command = [os.sys.executable, "scripts/summarize_training_runs.py"]
    subprocess.run(command, capture_output=True, text=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_reproducibility(int(config.get("seed", 11)))

    import torch
    import torch.nn.functional as F
    from torch import nn

    device, device_label = detect_device(torch, config.get("device_preference", ["hip", "cuda", "cpu"]))
    tokenizer = ByteTokenizer()

    train_rows = load_jsonl(Path(config["train_file"]))
    validation_rows = load_jsonl(Path(config["validation_file"]))
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
            self.vocab_size = VOCAB_SIZE
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

    base_output_dir = Path(config["output_dir"])
    output_dir = make_run_output_dir(base_output_dir)
    status_writer = TrainingStatusWriter(output_dir=output_dir, config=config)

    model = NativeTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    total_batches_per_epoch = max(1, math.ceil(len(train_examples) / int(config["batch_size"])))
    total_steps = total_batches_per_epoch * int(config["num_train_epochs"])
    status_writer.update(
        status="ready_to_train",
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        device=device_label,
        max_steps=total_steps,
    )

    def evaluate_loss(examples: list[Example]) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in create_batches(examples, int(config["eval_batch_size"]), shuffle=False):
                input_ids, labels = collate_batch(batch, tokenizer, torch)
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.vocab_size),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                losses.append(float(loss.item()))
        model.train()
        return sum(losses) / len(losses) if losses else 0.0

    global_step = 0
    best_validation_loss = float("inf")
    try:
        model.train()
        for epoch_index in range(int(config["num_train_epochs"])):
            for batch in create_batches(train_examples, int(config["batch_size"]), shuffle=True):
                global_step += 1
                input_ids, labels = collate_batch(batch, tokenizer, torch)
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.vocab_size),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                loss.backward()
                grad_clip = float(config.get("grad_clip", 1.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                current_epoch = epoch_index + (global_step / total_batches_per_epoch)
                if global_step == 1 or global_step % int(config["logging_steps"]) == 0 or global_step == total_steps:
                    status_writer.update(
                        status="training",
                        global_step=global_step,
                        epoch=round(current_epoch, 4),
                        latest_log={
                            "train_loss": float(loss.item()),
                            "epoch": round(current_epoch, 4),
                        },
                    )

                if global_step % int(config["eval_steps"]) == 0 or global_step == total_steps:
                    validation_loss = evaluate_loss(validation_examples)
                    latest_log = dict(status_writer.state.get("latest_log", {}))
                    latest_log["validation_loss"] = validation_loss
                    latest_log["epoch"] = round(current_epoch, 4)
                    status_writer.update(
                        status="training",
                        global_step=global_step,
                        epoch=round(current_epoch, 4),
                        latest_log=latest_log,
                    )
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss

        total_train_loss = float(status_writer.state.get("latest_log", {}).get("train_loss", 0.0))
        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        tokenizer.save(output_dir / "tokenizer_config.json")
        save_json(
            output_dir / "model_config.json",
            {
                "model_type": "native_byte_transformer",
                "vocab_size": VOCAB_SIZE,
                "max_seq_length": int(config["max_seq_length"]),
                "hidden_size": int(config["hidden_size"]),
                "num_layers": int(config["num_layers"]),
                "num_heads": int(config["num_heads"]),
                "mlp_ratio": float(config["mlp_ratio"]),
                "dropout": float(config["dropout"]),
            },
        )
        save_json(output_dir / "training_config.json", config)
        benchmark_report_path = run_post_training_benchmark(output_dir)
        refresh_training_summary()
        status_writer.update(
            status="completed",
            completed_at=utc_now_iso(),
            global_step=global_step,
            epoch=float(config["num_train_epochs"]),
            latest_log={
                "train_loss": total_train_loss,
                "validation_loss": best_validation_loss if best_validation_loss != float("inf") else None,
                "epoch": float(config["num_train_epochs"]),
            },
            benchmark_report=str(benchmark_report_path),
            best_validation_loss=None if best_validation_loss == float("inf") else best_validation_loss,
        )
        print(f"Training complete. Saved model to {output_dir}")
        print(f"Benchmark report: {benchmark_report_path}")
        return 0
    except Exception as exc:
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
