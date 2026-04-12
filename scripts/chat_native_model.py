import argparse
import json
from pathlib import Path
import re
import sys


ROLE_PREFIXES = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple chat prompt against a trained native byte-level transformer."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. If omitted, the script starts an interactive REPL.",
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved native model run directory.")
    parser.add_argument("--system-prompt", type=str, default="You are a simple helpful chatbot.", help="System prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument("--min-new-tokens", type=int, default=16, help="Minimum generated tokens before EOS is allowed.")
    parser.add_argument("--interactive", action="store_true", help="Start a simple interactive chat loop.")
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the raw generation instead of extracting the response field from JSON output.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def build_model(model_config: dict, torch_module):
    from torch import nn

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
            attn = torch_module.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
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
            positions = torch_module.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)
            x = self.dropout(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return self.lm_head(x)

    return NativeTransformerLM()


def generate_text(
    *,
    model,
    tokenizer: ByteTokenizer,
    torch_module,
    device,
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
) -> str:
    prompt_text = render_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
    )
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
    generated_ids = prompt_ids[:]
    generated_tensor = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)

    with torch_module.no_grad():
        for generated_token_count in range(max_new_tokens):
            current_tensor = generated_tensor[:, -int(model_config["max_seq_length"]) :]
            logits = model(current_tensor)
            next_token_logits = logits[0, -1].clone()
            for token_id in {tokenizer.pad_token_id, tokenizer.bos_token_id}:
                next_token_logits[token_id] = float("-inf")
            if generated_token_count < min_new_tokens:
                next_token_logits[tokenizer.eos_token_id] = float("-inf")
            newline_token_id = tokenizer.char_to_id.get("\n")
            if newline_token_id is not None and generated_token_count < 8:
                next_token_logits[newline_token_id] = float("-inf")
            next_token_id = int(torch_module.argmax(next_token_logits).item())
            generated_ids.append(next_token_id)
            next_token_tensor = torch_module.tensor([[next_token_id]], dtype=torch_module.long, device=device)
            generated_tensor = torch_module.cat((generated_tensor, next_token_tensor), dim=1)
            if next_token_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids[len(prompt_ids) :]).strip()


def extract_display_text(raw_text: str, show_raw: bool) -> str:
    if show_raw:
        return raw_text
    stripped = raw_text.strip()
    if not stripped:
        return stripped
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return extract_response_like_text(stripped)
    if not isinstance(payload, dict):
        return stripped
    response = payload.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    return extract_response_like_text(stripped)


def extract_response_like_text(text: str) -> str:
    response_match = re.search(r'"response"\s*:\s*"([^"]+)"', text)
    if response_match:
        return response_match.group(1).strip()
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
    if reason_match:
        return reason_match.group(1).strip()
    quoted_segments = re.findall(r'"([^"]{12,})"', text)
    if quoted_segments:
        return max((segment.strip() for segment in quoted_segments), key=len)
    return stripped


def load_runtime(model_path: Path):
    import torch

    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer = ByteTokenizer.from_config(load_json(tokenizer_config_path))
    else:
        tokenizer = ByteTokenizer(chars=[chr(index) for index in range(128)])
    model_config = load_json(model_path / "model_config.json")
    model = build_model(model_config, torch)
    state_dict = torch.load(model_path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    device, device_label = detect_device(torch)
    model = model.to(device)
    model.eval()
    return torch, tokenizer, model_config, model, device, device_label


def run_single_prompt(args: argparse.Namespace, prompt_text: str, runtime: tuple) -> int:
    torch_module, tokenizer, model_config, model, device, _device_label = runtime
    raw_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        device=device,
        model_config=model_config,
        system_prompt=args.system_prompt,
        user_prompt=prompt_text,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
    )
    print(extract_display_text(raw_text, show_raw=args.show_raw))
    return 0


def run_interactive(args: argparse.Namespace, runtime: tuple) -> int:
    _torch_module, _tokenizer, _model_config, _model, _device, device_label = runtime
    print(f"Loaded model from {args.model_path} on {device_label}")
    print("Type a prompt and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt_text = input("chat> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not prompt_text:
            continue
        if prompt_text.lower() in {"exit", "quit"}:
            break
        run_single_prompt(args, prompt_text, runtime)
    return 0


def main() -> int:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    runtime = load_runtime(args.model_path)
    prompt_text = " ".join(args.prompt).strip()
    if args.interactive or not prompt_text:
        return run_interactive(args, runtime)
    return run_single_prompt(args, prompt_text, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
