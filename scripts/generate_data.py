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
DEFAULT_FAILED_RESPONSES_DIR = Path("data/raw/failed_responses")
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_BATCH_SIZE = 100
TOKENS_PER_RECORD_ESTIMATE = 120

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
    "spending limit",
    "emergency fund",
    "financial goal",
    "monthly expense",
    "fixed cost",
    "variable cost",
    "short-term goal",
    "long-term goal",
    "trade-off",
    "opportunity cost",
]

CLARIFICATION_BASES = [
    "Help me choose",
    "Is this enough",
    "Can I afford it",
    "What should I do next",
    "Can you explain that",
    "How much should I spend",
    "Which one is better",
    "Can you break this down",
    "Should I wait",
    "What is the best option for me",
    "Can you organize this",
    "Is it too much",
    "What should I cut",
    "Can you simplify this",
    "Is this a good amount",
]

TOOL_NEEDED_PROMPTS = [
    "Find me the cheapest flights to {city} {timeframe}.",
    "What is the weather in {city} right now?",
    "Look up {company} stock price today.",
    "What restaurants are open near me {timeframe}?",
    "Check whether {product} is in stock online.",
    "Find the latest news about {company}.",
    "What time does the nearest {store_type} close {timeframe}?",
    "Compare today's mortgage rates.",
    "Check whether my {package_type} has shipped.",
    "What movies are playing near me {timeframe}?",
    "Find me an apartment under {price} in {city}.",
    "What are the cheapest gas prices around me {timeframe}?",
    "Can you calculate {large_number_a} times {large_number_b} exactly?",
    "What deals are available for {product_plural} today?",
    "Look up the score of the {sports_team} game today.",
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
    "What is the best way to hide income from the government?",
    "Should I represent myself in court tomorrow?",
    "Tell me which crypto will make me rich next month.",
    "Write a false excuse that my boss will believe.",
]

CITY_OPTIONS = [
    "Seattle",
    "Dallas",
    "Chicago",
    "Denver",
    "Boston",
    "Atlanta",
    "Phoenix",
    "Miami",
    "Portland",
    "Austin",
]

TIMEFRAME_OPTIONS = [
    "right now",
    "tonight",
    "this weekend",
    "next weekend",
    "next month",
    "today",
]

COMPANY_OPTIONS = [
    "Tesla",
    "AMD",
    "NVIDIA",
    "Apple",
    "Microsoft",
    "Amazon",
]

PRODUCT_OPTIONS = [
    "this laptop",
    "that phone",
    "this TV",
    "that gaming console",
    "this refrigerator",
]

PRODUCT_PLURAL_OPTIONS = [
    "laptops",
    "phones",
    "TVs",
    "headphones",
    "gaming consoles",
]

STORE_TYPE_OPTIONS = [
    "grocery store",
    "pharmacy",
    "hardware store",
    "bank branch",
    "coffee shop",
]

PACKAGE_TYPE_OPTIONS = [
    "package",
    "order",
    "delivery",
]

SPORTS_TEAM_OPTIONS = [
    "Lakers",
    "Yankees",
    "Cowboys",
    "Celtics",
    "Dodgers",
]

PRICE_OPTIONS = [
    "$1200",
    "$1500",
    "$1800",
    "$2200",
]

LARGE_NUMBER_OPTIONS = [
    ("73849", "99213"),
    ("54821", "77654"),
    ("91327", "44812"),
]

DIRECT_TEMPLATES = [
    "Define {topic}.",
    "Can you define {topic}?",
    "Explain {topic} in simple words.",
    "What is {topic}?",
    "What does {topic} mean?",
    "Give me a simple explanation of {topic}.",
]

CLARIFICATION_SUFFIXES = [
    "?",
    " for me?",
    " right now?",
    " in my situation?",
    "? I am not sure.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured training examples using a teacher model."
    )
    parser.add_argument("--count", type=int, default=50, help="Number of examples to request.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts to request per API call.",
    )
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
        "--failed-responses-dir",
        type=Path,
        default=DEFAULT_FAILED_RESPONSES_DIR,
        help="Directory where raw failed model responses will be written for inspection.",
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
        default=0,
        help="Max completion tokens sent to the teacher model. Use 0 to omit the limit and let the API decide.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=6.0,
        help="Delay in seconds between successful batches.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Maximum retries for retryable API failures such as rate limits. Default is 0 for low daily quota use.",
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
    prompts = set()
    target_count = max(count, 1)
    max_attempts = target_count * 40
    attempts = 0

    while len(prompts) < target_count and attempts < max_attempts:
        attempts += 1

        topic = random.choice(DIRECT_ANSWER_TOPICS)
        prompts.add(random.choice(DIRECT_TEMPLATES).format(topic=topic))

        clarification_base = random.choice(CLARIFICATION_BASES)
        clarification_suffix = random.choice(CLARIFICATION_SUFFIXES)
        prompts.add(f"{clarification_base}{clarification_suffix}")

        tool_template = random.choice(TOOL_NEEDED_PROMPTS)
        large_number_a, large_number_b = random.choice(LARGE_NUMBER_OPTIONS)
        prompts.add(
            tool_template.format(
                city=random.choice(CITY_OPTIONS),
                timeframe=random.choice(TIMEFRAME_OPTIONS),
                company=random.choice(COMPANY_OPTIONS),
                product=random.choice(PRODUCT_OPTIONS),
                product_plural=random.choice(PRODUCT_PLURAL_OPTIONS),
                store_type=random.choice(STORE_TYPE_OPTIONS),
                package_type=random.choice(PACKAGE_TYPE_OPTIONS),
                sports_team=random.choice(SPORTS_TEAM_OPTIONS),
                price=random.choice(PRICE_OPTIONS),
                large_number_a=large_number_a,
                large_number_b=large_number_b,
            )
        )

        prompts.add(random.choice(OUT_OF_SCOPE_PROMPTS))

    prompt_list = list(prompts)
    random.shuffle(prompt_list)
    return prompt_list[:count]


def render_teacher_messages(
    prompt_template: str,
    few_shot_examples: list[dict],
    user_inputs: list[str] | None = None,
    record_count: int | None = None,
    existing_inputs: set[str] | None = None,
) -> list[dict]:
    examples_json = json.dumps(few_shot_examples, indent=2, ensure_ascii=True)
    if user_inputs is None:
        if record_count is None:
            raise ValueError("record_count is required when user_inputs is omitted")
        existing_sample = sorted(item for item in (existing_inputs or set()) if isinstance(item, str))[:200]
        existing_json = json.dumps(existing_sample, indent=2, ensure_ascii=True)
        user_message = (
            "Use the style and logic of these examples as the standard.\n\n"
            f"Few-shot examples:\n{examples_json}\n\n"
            f"Generate exactly {record_count} new training records.\n"
            "Return exactly one JSON array containing one object per record.\n\n"
            "Each array item must have exactly these fields:\n"
            "- user_input\n"
            "- response_type\n"
            "- reason\n"
            "- response\n\n"
            "Requirements:\n"
            "- Generate realistic user prompts instead of reusing the few-shot examples.\n"
            "- Mix DIRECT_ANSWER, CLARIFICATION, TOOL_NEEDED, and OUT_OF_SCOPE cases.\n"
            "- Keep prompts and responses concise.\n"
            "- Make user_input values meaningfully distinct from each other.\n"
            "- Do not repeat or closely paraphrase any prompt from the avoid list.\n\n"
            f"Avoid list:\n{existing_json}\n\n"
            "Return the JSON array only."
        )
    elif len(user_inputs) == 1:
        user_message = (
            "Use the style and logic of these examples as the standard.\n\n"
            f"Few-shot examples:\n{examples_json}\n\n"
            f"User input:\n{user_inputs[0]}\n\n"
            "Return exactly one JSON object with these fields:\n"
            "- user_input\n"
            "- response_type\n"
            "- reason\n"
            "- response\n\n"
            "Return the JSON object only."
        )
    else:
        user_inputs_json = json.dumps(user_inputs, indent=2, ensure_ascii=True)
        user_message = (
            "Use the style and logic of these examples as the standard.\n\n"
            f"Few-shot examples:\n{examples_json}\n\n"
            "Classify every input independently.\n"
            "Return exactly one JSON array containing one object per input, in the same order.\n\n"
            f"User inputs:\n{user_inputs_json}\n\n"
            "Each array item must have exactly these fields:\n"
            "- user_input\n"
            "- response_type\n"
            "- reason\n"
            "- response\n\n"
            "Return the JSON array only."
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


def extract_json_value(text: str) -> str:
    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ValueError("Model returned empty content")

    decoder = json.JSONDecoder()
    first_non_space = None
    for index, char in enumerate(cleaned):
        if char in "[{":
            first_non_space = index
            break
        if not char.isspace():
            continue

    candidate_starts = []
    if first_non_space is not None:
        candidate_starts.append(first_non_space)

    candidate_starts.extend(
        index for index, char in enumerate(cleaned) if char in "[{"
    )

    seen = set()
    ordered_starts = []
    for index in candidate_starts:
        if index not in seen:
            ordered_starts.append(index)
            seen.add(index)

    for start in ordered_starts:
        try:
            _, end = decoder.raw_decode(cleaned[start:])
            return cleaned[start : start + end]
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not extract JSON value from model response")


def extract_partial_json_array_items(text: str) -> list[dict]:
    cleaned = strip_code_fences(text)
    if not cleaned:
        return []

    start = cleaned.find("[")
    if start == -1:
        return []

    content = cleaned[start + 1 :]
    decoder = json.JSONDecoder()
    items = []
    index = 0

    while index < len(content):
        while index < len(content) and content[index] in " \r\n\t,":
            index += 1

        if index >= len(content) or content[index] == "]":
            break

        try:
            item, consumed = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            break

        if not isinstance(item, dict):
            break

        items.append(item)
        index += consumed

    return items


def save_failed_response(
    directory: Path,
    batch_index: int,
    prompts: list[str],
    raw_content: str,
    response_body: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"batch_{batch_index:03d}_{timestamp}.json"
    payload = {
        "batch_index": batch_index,
        "prompts": prompts,
        "raw_content": raw_content,
        "response_body": response_body,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return path


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


def validate_batch_records(records: object, expected_user_inputs: list[str]) -> list[dict]:
    if len(expected_user_inputs) == 1 and isinstance(records, dict):
        return [validate_record(records, expected_user_inputs[0])]

    if not isinstance(records, list):
        raise ValueError("Teacher response must be a JSON array")

    if len(records) != len(expected_user_inputs):
        raise ValueError(
            f"Teacher returned {len(records)} records for {len(expected_user_inputs)} prompts"
        )

    clean_records = []
    for record, expected_user_input in zip(records, expected_user_inputs):
        if not isinstance(record, dict):
            raise ValueError("Each batch item must be a JSON object")
        clean_records.append(validate_record(record, expected_user_input))

    return clean_records


def validate_generated_batch_records(records: object, existing_inputs: set[str]) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("Teacher response must be a JSON array")

    clean_records = []
    seen_in_batch = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each batch item must be a JSON object")
        clean_record = validate_record(record, record.get("user_input", ""))
        user_input = clean_record["user_input"]
        if user_input in existing_inputs or user_input in seen_in_batch:
            continue
        seen_in_batch.add(user_input)
        clean_records.append(clean_record)

    return clean_records


def try_salvage_partial_batch(
    raw_content: str, expected_user_inputs: list[str]
) -> list[dict]:
    partial_items = extract_partial_json_array_items(raw_content)
    if not partial_items:
        return []

    clean_records = []
    for record, expected_user_input in zip(partial_items, expected_user_inputs):
        clean_records.append(validate_record(record, expected_user_input))

    return clean_records


def extract_message_content(message_content: object) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        text_parts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "\n".join(text_parts).strip()
    return ""


def call_teacher_model(
    api_base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_output_tokens: int,
    expect_array: bool,
) -> dict:
    url = api_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_output_tokens and max_output_tokens > 0:
        payload["max_tokens"] = max_output_tokens
    if not expect_array:
        payload["response_format"] = {"type": "json_object"}
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
        content = extract_message_content(data["choices"][0]["message"].get("content", ""))
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected API response shape: {body}") from exc
    if not content.strip():
        raise ValueError(f"Model returned empty content. Response body: {body[:1000]}")
    extracted = extract_json_value(content)
    return json.loads(extracted), content, body




def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    return False


def generate_records_for_batch(
    batch_index: int,
    batch_count: int,
    args: argparse.Namespace,
    prompt_template: str,
    few_shot_examples: list[dict],
    api_key: str,
    requested_count: int,
    existing_inputs: set[str],
) -> list[dict]:
    messages = render_teacher_messages(
        prompt_template,
        few_shot_examples,
        record_count=requested_count,
        existing_inputs=existing_inputs,
    )
    attempt = 0
    raw_content = ""
    response_body = ""
    expect_array = True
    if args.max_output_tokens and args.max_output_tokens > 0:
        batch_max_output_tokens = max(
            args.max_output_tokens,
            TOKENS_PER_RECORD_ESTIMATE * requested_count,
        )
    else:
        batch_max_output_tokens = 0
    while True:
        try:
            raw_records, raw_content, response_body = call_teacher_model(
                api_base_url=args.api_base_url,
                api_key=api_key,
                model=args.teacher_model,
                messages=messages,
                temperature=args.temperature,
                max_output_tokens=batch_max_output_tokens,
                expect_array=expect_array,
            )
            clean_records = validate_generated_batch_records(raw_records, existing_inputs)
            for clean_record in clean_records:
                clean_record["teacher_model"] = args.teacher_model
                clean_record["prompt_version"] = args.prompt_template.stem
                clean_record["date_generated"] = time.strftime("%Y-%m-%d")
            print(
                f"[batch {batch_index}/{batch_count}] generated {len(clean_records)} records"
            )
            return clean_records
        except (ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, socket.timeout) as exc:
            if is_retryable_error(exc) and attempt < args.max_retries:
                wait_seconds = args.retry_backoff_seconds * (2 ** attempt)
                attempt += 1
                print(
                    f"[batch {batch_index}/{batch_count}] retrying after {wait_seconds:.1f}s :: {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
                continue

            if raw_content:
                try:
                    salvaged_records = validate_generated_batch_records(
                        extract_partial_json_array_items(raw_content),
                        existing_inputs,
                    )
                except ValueError:
                    salvaged_records = []
                if salvaged_records:
                    for clean_record in salvaged_records:
                        clean_record["teacher_model"] = args.teacher_model
                        clean_record["prompt_version"] = args.prompt_template.stem
                        clean_record["date_generated"] = time.strftime("%Y-%m-%d")
                    print(
                        f"[batch {batch_index}/{batch_count}] salvaged {len(salvaged_records)} records from partial batch output",
                        file=sys.stderr,
                    )
                    return salvaged_records

            failed_path = save_failed_response(
                args.failed_responses_dir,
                batch_index,
                [],
                raw_content,
                response_body,
            )
            print(
                f"[batch {batch_index}/{batch_count}] saved failed raw response to {failed_path}",
                file=sys.stderr,
            )
            print(
                f"[batch {batch_index}/{batch_count}] failed for {requested_count} requested records :: {exc}",
                file=sys.stderr,
            )
            return []


def main() -> int:
    args = parse_args()
    random.seed(11)
    load_env_file(args.env_file)
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 1

    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    few_shot_examples = sample_seed_examples(args.seed, args.few_shot_count)
    existing_records = load_json_array(args.output)
    existing_inputs = {item.get("user_input") for item in existing_records}
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        print(
            f"Missing API key. Set the {args.api_key_env} environment variable or use --dry-run.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        batch_count = (args.count + args.batch_size - 1) // args.batch_size
        for batch_index in range(batch_count):
            requested_count = min(args.batch_size, args.count - (batch_index * args.batch_size))
            messages = render_teacher_messages(
                prompt_template,
                few_shot_examples,
                record_count=requested_count,
                existing_inputs=existing_inputs,
            )
            preview = {
                "batch_number": batch_index + 1,
                "batch_size": requested_count,
                "messages": messages,
            }
            print(json.dumps(preview, indent=2, ensure_ascii=True))
        return 0

    generated = []
    batch_count = (args.count + args.batch_size - 1) // args.batch_size
    seen_inputs = set(existing_inputs)
    for batch_index in range(1, batch_count + 1):
        requested_count = min(args.batch_size, args.count - len(generated))
        if requested_count <= 0:
            break
        batch_records = generate_records_for_batch(
            batch_index,
            batch_count,
            args,
            prompt_template,
            few_shot_examples,
            api_key,
            requested_count,
            seen_inputs,
        )
        generated.extend(batch_records)
        seen_inputs.update(item["user_input"] for item in batch_records)
        if args.request_delay > 0:
            time.sleep(args.request_delay)

    all_records = existing_records + generated
    save_json_array(args.output, all_records)
    if len(generated) < args.count:
        print(
            f"Warning: only generated {len(generated)} valid unique examples for requested count {args.count}.",
            file=sys.stderr,
        )
    print(f"Saved {len(generated)} new examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
