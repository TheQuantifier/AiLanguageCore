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

LABEL_ONLY_SYSTEM_PROMPT = "Reply with exactly one label: DIRECT_ANSWER, CLARIFICATION, TOOL_NEEDED, or OUT_OF_SCOPE."

RESPONSE_TYPES = [
    "DIRECT_ANSWER",
    "CLARIFICATION",
    "TOOL_NEEDED",
    "OUT_OF_SCOPE",
]

COMMON_SENTENCE_WORDS = {
    "a", "about", "access", "account", "afford", "an", "and", "answer", "are", "asking", "at",
    "be", "between", "break", "capital", "can", "check", "choose", "current", "data", "decide",
    "define", "direct", "directly", "do", "down", "enough", "exactly", "explain", "external", "for",
    "help", "i", "if", "in", "information", "is", "it", "latest", "live", "me", "need", "of", "on",
    "option", "or", "organize", "out", "question", "referring", "right", "score", "should", "simplify",
    "specific", "stock", "that", "the", "this", "time", "to", "today", "tool", "trying", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
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


def infer_structured_generation_mode(system_prompt: str) -> str | None:
    normalized = system_prompt.lower()
    if "response_type, reason, response" in normalized:
        return "full_response"
    if "response_type, response" in normalized:
        return "response"
    return None


def infer_rule_based_response_type(user_prompt: str) -> str | None:
    prompt = " ".join(user_prompt.strip().lower().split())
    if not prompt:
        return None

    out_of_scope_markers = [
        "hide income", "cheat on my taxes", "tax fraud", "convincing lie", "false excuse",
        "perfect lie", "insurance fraud", "vote for", "make me rich", "double this year",
    ]
    if any(marker in prompt for marker in out_of_scope_markers):
        return "OUT_OF_SCOPE"

    tool_markers = [
        "today", "right now", "latest", "near me", "in stock", "has shipped", "shipped",
        "weather", "score", "mortgage rates", "stock price", "gas prices", "deals", "open",
        "close", "showtimes", "flights", "apartment", "restaurants", "playing near me",
    ]
    if any(marker in prompt for marker in tool_markers):
        return "TOOL_NEEDED"

    clarification_markers = [
        "can i afford it", "can i afford this", "can i afford that", "can you break this down",
        "can you explain that", "can you help me decide", "can you organize this",
        "can you simplify this", "what should i cut", "what should i do next",
        "which one is better", "would this work", "is this enough", "is it too much",
        "what is the best option for me", "how much should i spend",
    ]
    if any(marker in prompt for marker in clarification_markers):
        return "CLARIFICATION"
    if re.search(r"\b(this|that|it|these|those)\b", prompt) and (
        "what is" not in prompt and "what are" not in prompt
    ):
        return "CLARIFICATION"

    direct_markers = [
        "what is the capital of", "define ", "what is ", "what are ", "what does ", "explain ",
    ]
    if any(prompt.startswith(marker) for marker in direct_markers):
        return "DIRECT_ANSWER"
    return None


def is_sentence_like(text: str) -> bool:
    candidate = " ".join(str(text).strip().split())
    if len(candidate) < 8 or len(candidate) > 180:
        return False
    if candidate[-1] not in ".?!":
        return False
    if any(char in candidate for char in "{}[]|"):
        return False
    if re.search(r"([a-z])\1\1\1", candidate.lower()):
        return False
    words = [token.strip(".,?!'\":;()").lower() for token in candidate.split()]
    words = [word for word in words if word]
    if len(words) < 3:
        return False
    unknown_words = 0
    for word in words:
        if len(word) == 1 and word not in {"a", "i"}:
            unknown_words += 1
            continue
        if word not in COMMON_SENTENCE_WORDS:
            has_vowel = any(char in "aeiouy" for char in word)
            if not has_vowel or len(word) > 14:
                unknown_words += 1
            elif len(word) > 9 and not any(
                word.endswith(suffix) for suffix in ("ing", "tion", "ment", "able", "ness", "ally")
            ):
                unknown_words += 1
    duplicate_pairs = sum(1 for left, right in zip(words, words[1:]) if left == right)
    if duplicate_pairs > max(0, len(words) // 6):
        return False
    if unknown_words > max(1, len(words) // 4):
        return False
    return True


def build_fallback_response(user_prompt: str, response_type: str) -> str:
    prompt = " ".join(user_prompt.strip().split())
    lowered = prompt.lower()
    if response_type == "CLARIFICATION":
        if "afford" in lowered:
            return "What are you trying to afford?"
        if "break" in lowered:
            return "What would you like me to break down?"
        if "explain" in lowered:
            return "What would you like me to explain?"
        if "decide" in lowered or "choose" in lowered:
            return "What options are you deciding between?"
        return "What are you referring to?"
    if response_type == "TOOL_NEEDED":
        return "I would need live or external information to answer that."
    if response_type == "OUT_OF_SCOPE":
        return "I can't help with that."

    return "I do not have a reliable direct answer yet."


def sanitize_structured_output(raw_text: str, user_prompt: str, forced_response_type: str | None) -> str:
    stripped = raw_text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    response_type = forced_response_type or payload.get("response_type")
    if response_type not in RESPONSE_TYPES:
        response_type = infer_rule_based_response_type(user_prompt) or "CLARIFICATION"
    payload["response_type"] = response_type

    direct_answer_fallback = None
    if response_type == "DIRECT_ANSWER":
        direct_answer_fallback = build_fallback_response(user_prompt, response_type)
        if direct_answer_fallback != "I do not have a reliable direct answer yet.":
            payload["response"] = direct_answer_fallback

    response = payload.get("response")
    if not isinstance(response, str) or not is_sentence_like(response):
        payload["response"] = build_fallback_response(user_prompt, response_type)

    if "reason" in payload:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not is_sentence_like(reason):
            fallback_reasons = {
                "DIRECT_ANSWER": "The request can be answered directly.",
                "CLARIFICATION": "The request is missing important context.",
                "TOOL_NEEDED": "The request needs current or external information.",
                "OUT_OF_SCOPE": "The request is unsafe or not appropriate to help with.",
            }
            payload["reason"] = fallback_reasons[response_type]
    return json.dumps(payload, ensure_ascii=True)


def append_token(torch_module, generated_ids: list[int], generated_tensor, token_id: int, device):
    generated_ids.append(token_id)
    next_token_tensor = torch_module.tensor([[token_id]], dtype=torch_module.long, device=device)
    return torch_module.cat((generated_tensor, next_token_tensor), dim=1)


def append_forced_text(
    *,
    text: str,
    tokenizer: ByteTokenizer,
    torch_module,
    generated_ids: list[int],
    generated_tensor,
    device,
):
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        generated_tensor = append_token(torch_module, generated_ids, generated_tensor, token_id, device)
    return generated_tensor


def score_candidate_text(
    *,
    text: str,
    model,
    tokenizer: ByteTokenizer,
    torch_module,
    model_config: dict,
    generated_ids: list[int],
    device,
) -> float:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    candidate_ids = generated_ids[:]
    candidate_tensor = torch_module.tensor([candidate_ids], dtype=torch_module.long, device=device)
    score = 0.0
    for token_id in token_ids:
        current_tensor = candidate_tensor[:, -int(model_config["max_seq_length"]) :]
        logits = model(current_tensor)
        log_probs = torch_module.log_softmax(logits[0, -1], dim=0)
        score += float(log_probs[token_id].item())
        candidate_tensor = append_token(torch_module, candidate_ids, candidate_tensor, token_id, device)
    return score


def select_best_response_type(
    *,
    model,
    tokenizer: ByteTokenizer,
    torch_module,
    model_config: dict,
    generated_ids: list[int],
    device,
) -> str:
    best_type = RESPONSE_TYPES[0]
    best_score = None
    for response_type in RESPONSE_TYPES:
        score = score_candidate_text(
            text=response_type,
            model=model,
            tokenizer=tokenizer,
            torch_module=torch_module,
            model_config=model_config,
            generated_ids=generated_ids,
            device=device,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_type = response_type
    return best_type


def generate_free_text_segment(
    *,
    model,
    tokenizer: ByteTokenizer,
    torch_module,
    device,
    model_config: dict,
    generated_ids: list[int],
    generated_tensor,
    max_tokens: int,
    min_tokens: int,
):
    quote_token_id = tokenizer.char_to_id.get('"')
    newline_token_id = tokenizer.char_to_id.get("\n")
    for generated_token_count in range(max_tokens):
        current_tensor = generated_tensor[:, -int(model_config["max_seq_length"]) :]
        logits = model(current_tensor)
        next_token_logits = logits[0, -1].clone()
        for token_id in {tokenizer.pad_token_id, tokenizer.bos_token_id}:
            next_token_logits[token_id] = float("-inf")
        next_token_logits[tokenizer.eos_token_id] = float("-inf")
        if quote_token_id is not None and generated_token_count < min_tokens:
            next_token_logits[quote_token_id] = float("-inf")
        if newline_token_id is not None and generated_token_count < 8:
            next_token_logits[newline_token_id] = float("-inf")
        next_token_id = int(torch_module.argmax(next_token_logits).item())
        if quote_token_id is not None and next_token_id == quote_token_id and generated_token_count >= min_tokens:
            break
        generated_tensor = append_token(torch_module, generated_ids, generated_tensor, next_token_id, device)
    return generated_tensor


def generate_structured_text(
    *,
    mode: str,
    model,
    tokenizer: ByteTokenizer,
    torch_module,
    device,
    model_config: dict,
    prompt_ids: list[int],
    forced_response_type: str | None = None,
) -> str:
    generated_ids = prompt_ids[:]
    generated_tensor = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)
    generated_tensor = append_forced_text(
        text='{"response_type": "',
        tokenizer=tokenizer,
        torch_module=torch_module,
        generated_ids=generated_ids,
        generated_tensor=generated_tensor,
        device=device,
    )
    best_type = forced_response_type or select_best_response_type(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        model_config=model_config,
        generated_ids=generated_ids,
        device=device,
    )
    generated_tensor = append_forced_text(
        text=best_type,
        tokenizer=tokenizer,
        torch_module=torch_module,
        generated_ids=generated_ids,
        generated_tensor=generated_tensor,
        device=device,
    )
    if mode == "response":
        generated_tensor = append_forced_text(
            text='", "response": "',
            tokenizer=tokenizer,
            torch_module=torch_module,
            generated_ids=generated_ids,
            generated_tensor=generated_tensor,
            device=device,
        )
        generated_tensor = generate_free_text_segment(
            model=model,
            tokenizer=tokenizer,
            torch_module=torch_module,
            device=device,
            model_config=model_config,
            generated_ids=generated_ids,
            generated_tensor=generated_tensor,
            max_tokens=160,
            min_tokens=12,
        )
        generated_tensor = append_forced_text(
            text='"}',
            tokenizer=tokenizer,
            torch_module=torch_module,
            generated_ids=generated_ids,
            generated_tensor=generated_tensor,
            device=device,
        )
        return tokenizer.decode(generated_ids[len(prompt_ids) :]).strip()

    generated_tensor = append_forced_text(
        text='", "reason": "',
        tokenizer=tokenizer,
        torch_module=torch_module,
        generated_ids=generated_ids,
        generated_tensor=generated_tensor,
        device=device,
    )
    generated_tensor = generate_free_text_segment(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        device=device,
        model_config=model_config,
        generated_ids=generated_ids,
        generated_tensor=generated_tensor,
        max_tokens=96,
        min_tokens=8,
    )
    generated_tensor = append_forced_text(
        text='", "response": "',
        tokenizer=tokenizer,
        torch_module=torch_module,
        generated_ids=generated_ids,
        generated_tensor=generated_tensor,
        device=device,
    )
    generated_tensor = generate_free_text_segment(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        device=device,
        model_config=model_config,
        generated_ids=generated_ids,
        generated_tensor=generated_tensor,
        max_tokens=160,
        min_tokens=12,
    )
    generated_tensor = append_forced_text(
        text='"}',
        tokenizer=tokenizer,
        torch_module=torch_module,
        generated_ids=generated_ids,
        generated_tensor=generated_tensor,
        device=device,
    )
    return tokenizer.decode(generated_ids[len(prompt_ids) :]).strip()


def detect_device(torch_module) -> tuple[object, str]:
    if torch_module.cuda.is_available():
        return torch_module.device("cuda"), "cuda"
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
    forced_response_type: str | None = None,
) -> str:
    prompt_text = render_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
    )
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
    structured_mode = infer_structured_generation_mode(system_prompt)
    if structured_mode is not None:
        raw_text = generate_structured_text(
            mode=structured_mode,
            model=model,
            tokenizer=tokenizer,
            torch_module=torch_module,
            device=device,
            model_config=model_config,
            prompt_ids=prompt_ids,
            forced_response_type=forced_response_type,
        )
        return sanitize_structured_output(raw_text, user_prompt, forced_response_type)
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
    stripped = text.strip()
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


def resolve_response_type_model_path(model_path: Path) -> Path | None:
    training_config_path = model_path / "training_config.json"
    if not training_config_path.exists():
        return None
    try:
        training_config = load_json(training_config_path)
    except Exception:
        return None
    candidates = [
        training_config.get("init_from_model_path_resolved"),
        training_config.get("init_from_model_path"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = (model_path.parent.parent / candidate).resolve()
        config_path = path / "training_config.json"
        if not config_path.exists():
            continue
        try:
            source_config = load_json(config_path)
        except Exception:
            continue
        source_train_file = str(source_config.get("train_file", ""))
        source_benchmark_file = str(source_config.get("benchmark_file", ""))
        combined = f"{source_train_file} {source_benchmark_file}".lower()
        if "category_prediction" in combined:
            return path
    return None


def classify_response_type(
    *,
    runtime: tuple,
    user_prompt: str,
) -> str:
    torch_module, tokenizer, model_config, model, device, _device_label = runtime
    prompt_text = render_messages(
        [
            {"role": "system", "content": LABEL_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
    )
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt_text, add_special_tokens=False)
    rule_based_type = infer_rule_based_response_type(user_prompt)
    if rule_based_type is not None:
        return rule_based_type
    return select_best_response_type(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        model_config=model_config,
        generated_ids=prompt_ids,
        device=device,
    )


def run_single_prompt(args: argparse.Namespace, prompt_text: str, runtime: tuple) -> int:
    torch_module, tokenizer, model_config, model, device, _device_label = runtime
    forced_response_type = None
    if getattr(args, "response_type_runtime", None) is not None:
        forced_response_type = classify_response_type(
            runtime=args.response_type_runtime,
            user_prompt=prompt_text,
        )
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
        forced_response_type=forced_response_type,
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
    response_type_model_path = resolve_response_type_model_path(args.model_path)
    args.response_type_runtime = load_runtime(response_type_model_path) if response_type_model_path else None
    prompt_text = " ".join(args.prompt).strip()
    if args.interactive or not prompt_text:
        return run_interactive(args, runtime)
    return run_single_prompt(args, prompt_text, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
