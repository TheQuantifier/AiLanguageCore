import argparse
import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ALLOWED_RESPONSE_TYPES = {
    "DIRECT_ANSWER",
    "CLARIFICATION",
    "TOOL_NEEDED",
    "OUT_OF_SCOPE",
}

DEFAULT_OUTPUT = Path("data/raw/generated_dataset.json")
DEFAULT_SEED = Path("data/raw/v1_seed_dataset.json")
DEFAULT_PROMPT_TEMPLATE = Path("prompts/teacher_generation_prompt_v1.md")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

DIRECT_ANSWER_TOPICS = [
    "budget",
    "income",
    "expense",
    "savings",
    "goal",
    "priority",
    "plan",
    "category",
    "comparison",
    "needs versus wants",
]

CLARIFICATION_STARTERS = [
    "Help me choose",
    "Is this enough?",
    "Can I afford it?",
    "What should I do next?",
    "Can you explain that?",
    "How much should I spend?",
    "Which one is better?",
    "Can you break this down?",
    "Should I wait?",
    "What is the best option for me?",
]

TOOL_NEEDED_PROMPTS = [
    "Find me the cheapest flights to Seattle next month.",
    "What is the weather in Dallas right now?",
    "Look up Tesla stock price today.",
    "What restaurants are open near me?",
    "Check whether this item is in stock online.",
    "Find the latest news about AMD.",
    "What time does the nearest grocery store close tonight?",
    "Compare today's mortgage rates.",
    "Check whether my package has shipped.",
    "What movies are playing near me tonight?",
]

OUT_OF_SCOPE_PROMPTS = [
    "Should I stop taking my prescription medication?",
    "Tell me how to hack a Wi-Fi network.",
    "Who should I vote for in the election?",
    "Can you guarantee which stock will double this year?",
    "Write a convincing lie for me.",
    "Should I sue my employer?",
    "Tell me the fastest way to get rich with no risk.",
    "How can I cheat on my taxes?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured training examples using a teacher model."
    )
    parser.add_argument("--count", type=int, default=50, help="Number of examples to request.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the JSON file where generated examples will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=DEFAULT_SEED,
        help="Path to the manual seed dataset used for few-shot examples.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Path to the teacher prompt template file.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to a local .env file with TEACHER_* variables.",
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=6,
        help="Number of seed examples to include in the teacher prompt.",
    )
    parser.add_argument(
        "--teacher-model",
        default=os.environ.get("TEACHER_MODEL", DEFAULT_MODEL),
        help="Teacher model name.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("TEACHER_API_BASE_URL", DEFAULT_API_BASE_URL),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="TEACHER_API_KEY",
        help="Environment variable name that stores the API key.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature sent to the teacher model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=250,
        help="Max completion tokens sent to the teacher model.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.5,
        help="Delay in seconds between successful requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for retryable API failures such as rate limits.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Initial backoff in seconds for retryable API failures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts and print them without calling the API.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def load_json_array(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_array(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def sample_seed_examples(seed_path: Path, count: int) -> list[dict]:
    seed_data = load_json_array(seed_path)
    if not seed_data:
        raise ValueError(f"No seed examples found in {seed_path}")
    count = min(count, len(seed_data))
    response_types_seen = set()
    selected = []
    selected_ids = set()
    random.seed(7)
    shuffled = seed_data[:]
    random.shuffle(shuffled)

    for index, item in enumerate(shuffled):
        response_type = item.get("response_type")
        if response_type not in response_types_seen:
            selected.append(item)
            selected_ids.add(index)
            response_types_seen.add(response_type)
        if len(selected) >= count:
            return selected

    for index, item in enumerate(shuffled):
        if len(selected) >= count:
            break
        if index in selected_ids:
            continue
        selected.append(item)

    return selected


def generate_candidate_prompts(count: int) -> list[str]:
    prompts = []

    direct_templates = [
        "What is {topic}?",
        "Explain {topic} in simple words.",
        "Can you define {topic}?",
        "What does {topic} mean?",
    ]

    clarification_templates = [
        "{starter}.",
        "{starter} for me.",
        "{starter} right now.",
    ]

    while len(prompts) < count:
        topic = random.choice(DIRECT_ANSWER_TOPICS)
        prompts.append(random.choice(direct_templates).format(topic=topic))
        if len(prompts) >= count:
            break

        starter = random.choice(CLARIFICATION_STARTERS)
        prompts.append(random.choice(clarification_templates).format(starter=starter))
        if len(prompts) >= count:
            break

        prompts.append(random.choice(TOOL_NEEDED_PROMPTS))
        if len(prompts) >= count:
            break

        prompts.append(random.choice(OUT_OF_SCOPE_PROMPTS))

    random.shuffle(prompts)
    return prompts[:count]


def render_teacher_messages(
    prompt_template: str, few_shot_examples: list[dict], user_input: str
) -> list[dict]:
    examples_json = json.dumps(few_shot_examples, indent=2, ensure_ascii=True)
    user_message = (
        "Use the style and logic of these examples as the standard.\n\n"
        f"Few-shot examples:\n{examples_json}\n\n"
        f"User input:\n{user_input}\n\n"
        "Return one JSON object only."
    )
    return [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": user_message},
    ]


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def validate_record(record: dict, expected_user_input: str) -> dict:
    missing = [key for key in ("user_input", "response_type", "reason", "response") if key not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    response_type = record["response_type"]
    if response_type not in ALLOWED_RESPONSE_TYPES:
        raise ValueError(f"Invalid response_type: {response_type}")

    if not isinstance(record["response"], str) or not record["response"].strip():
        raise ValueError("Response must be a non-empty string")

    if not isinstance(record["reason"], str) or not record["reason"].strip():
        raise ValueError("Reason must be a non-empty string")

    user_input = record["user_input"]
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")

    if user_input.strip() != expected_user_input.strip():
        raise ValueError("Teacher returned a mismatched user_input")

    return {
        "user_input": user_input.strip(),
        "response_type": response_type,
        "reason": record["reason"].strip(),
        "response": record["response"].strip(),
    }


def call_teacher_model(
    api_base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_output_tokens: int,
) -> dict:
    url = api_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected API response shape: {body}") from exc
    return json.loads(strip_code_fences(content))


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    return False


def main() -> int:
    args = parse_args()
    random.seed(11)
    load_env_file(args.env_file)

    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    few_shot_examples = sample_seed_examples(args.seed, args.few_shot_count)
    existing_records = load_json_array(args.output)
    existing_inputs = {item.get("user_input") for item in existing_records}
    prompts = generate_candidate_prompts(args.count * 2)
    prompts = [prompt for prompt in prompts if prompt not in existing_inputs][: args.count]

    if len(prompts) < args.count:
        print(
            f"Warning: only generated {len(prompts)} unique prompts for requested count {args.count}.",
            file=sys.stderr,
        )

    if args.dry_run:
        for index, prompt in enumerate(prompts, start=1):
            messages = render_teacher_messages(prompt_template, few_shot_examples, prompt)
            preview = {
                "prompt_number": index,
                "user_input": prompt,
                "messages": messages,
            }
            print(json.dumps(preview, indent=2, ensure_ascii=True))
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"Missing API key. Set the {args.api_key_env} environment variable or use --dry-run.",
            file=sys.stderr,
        )
        return 1

    generated = []
    for index, prompt in enumerate(prompts, start=1):
        messages = render_teacher_messages(prompt_template, few_shot_examples, prompt)
        attempt = 0
        while True:
            try:
                raw_record = call_teacher_model(
                    api_base_url=args.api_base_url,
                    api_key=api_key,
                    model=args.teacher_model,
                    messages=messages,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                clean_record = validate_record(raw_record, prompt)
                clean_record["teacher_model"] = args.teacher_model
                clean_record["prompt_version"] = args.prompt_template.stem
                clean_record["date_generated"] = time.strftime("%Y-%m-%d")
                generated.append(clean_record)
                print(f"[{index}/{len(prompts)}] generated: {prompt}")
                if args.request_delay > 0:
                    time.sleep(args.request_delay)
                break
            except (ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, socket.timeout) as exc:
                if is_retryable_error(exc) and attempt < args.max_retries:
                    wait_seconds = args.retry_backoff_seconds * (2 ** attempt)
                    attempt += 1
                    print(
                        f"[{index}/{len(prompts)}] retrying after {wait_seconds:.1f}s: {prompt} :: {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(wait_seconds)
                    continue

                print(f"[{index}/{len(prompts)}] failed: {prompt} :: {exc}", file=sys.stderr)
                break

    all_records = existing_records + generated
    save_json_array(args.output, all_records)
    print(f"Saved {len(generated)} new examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
