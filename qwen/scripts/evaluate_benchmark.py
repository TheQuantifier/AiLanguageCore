import argparse
import json
import os
from pathlib import Path


ALLOWED_RESPONSE_TYPES = {
    "DIRECT_ANSWER",
    "CLARIFICATION",
    "TOOL_NEEDED",
    "OUT_OF_SCOPE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple benchmark evaluation on a trained LoRA adapter or merged model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained model or adapter directory.",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=Path("data/processed/benchmark_sft.jsonl"),
        help="Path to the benchmark SFT dataset.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("qwen/experiments/benchmark_report.json"),
        help="Path to write the evaluation report.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation cap per benchmark example.",
    )
    return parser.parse_args()


def resolve_output_report_path(model_path: Path, output_report: Path) -> Path:
    if output_report != Path("qwen/experiments/benchmark_report.json"):
        return output_report
    return output_report.parent / f"benchmark_report-{model_path.name}.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def configure_local_cache() -> None:
    cache_root = Path(".cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def resolve_model_source(model_ref: str | Path) -> str:
    model_path = Path(model_ref)
    if model_path.exists():
        return str(model_path)

    hf_home = Path(os.environ.get("HF_HOME", ".cache/huggingface"))
    model_cache_dir = hf_home / "hub" / ("models--" + str(model_ref).replace("/", "--"))
    ref_file = model_cache_dir / "refs" / "main"
    if ref_file.exists():
        revision = ref_file.read_text(encoding="utf-8").strip()
        snapshot_dir = model_cache_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return str(snapshot_dir)
    return str(model_ref)


def main() -> int:
    args = parse_args()
    configure_local_cache()
    output_report = resolve_output_report_path(args.model_path, args.output_report)

    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing evaluation dependencies. Install torch, transformers, and peft into the project Python runtime."
        ) from exc

    model_source = resolve_model_source(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model_kwargs = {
        "device_map": "auto" if use_cuda else "cpu",
        "dtype": torch.bfloat16 if use_cuda else torch.float32,
        "trust_remote_code": True,
    }

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=True,
            **model_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=True,
            **model_kwargs,
        )

    benchmark_rows = load_jsonl(args.benchmark_file)
    results = []
    correct_response_type = 0
    valid_json = 0

    for row in benchmark_rows:
        messages = row["messages"][:2]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        parsed = extract_json_object(generated)
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
                "raw_generation": generated,
            }
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

    print(f"Wrote benchmark report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
