import argparse
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a first-pass QLoRA training cycle from a JSON config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("models/v1_qwen2_5_7b_qlora_config.json"),
        help="Path to the training config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_local_cache() -> None:
    cache_root = Path(".cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def resolve_model_source(model_ref: str) -> str:
    model_path = Path(model_ref)
    if model_path.exists():
        return str(model_path)

    hf_home = Path(os.environ.get("HF_HOME", ".cache/huggingface"))
    model_cache_dir = hf_home / "hub" / ("models--" + model_ref.replace("/", "--"))
    ref_file = model_cache_dir / "refs" / "main"
    if ref_file.exists():
        revision = ref_file.read_text(encoding="utf-8").strip()
        snapshot_dir = model_cache_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return str(snapshot_dir)
    return model_ref


def require_training_dependencies() -> None:
    missing = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    try:
        import peft  # noqa: F401
    except ImportError:
        missing.append("peft")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import trl  # noqa: F401
    except ImportError:
        missing.append("trl")

    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install them into the project Python runtime before training."
        )


def pick_torch_dtype(torch_module, config: dict, use_cuda: bool):
    dtype_name = config.get("torch_dtype")
    if dtype_name:
        try:
            return getattr(torch_module, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype in config: {dtype_name}") from exc
    if use_cuda and config.get("bf16", False):
        return torch_module.bfloat16
    if use_cuda and config.get("fp16", False):
        return torch_module.float16
    return torch_module.float32


def estimate_total_steps(config: dict, train_examples: int) -> int:
    batch_size = max(1, int(config["per_device_train_batch_size"]))
    grad_accum = max(1, int(config["gradient_accumulation_steps"]))
    steps_per_epoch = max(1, (train_examples + (batch_size * grad_accum) - 1) // (batch_size * grad_accum))
    epochs = float(config["num_train_epochs"])
    return max(1, int(steps_per_epoch * epochs))


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


def run_post_training_benchmark(output_dir: Path) -> Path:
    command = [
        os.sys.executable,
        "scripts/evaluate_benchmark.py",
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
    command = [
        os.sys.executable,
        "scripts/summarize_training_runs.py",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
        print(f"Warning: failed to refresh training summary: {stderr}")


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
            "base_model": config["base_model"],
            "output_dir": str(output_dir),
            "train_file": config["train_file"],
            "validation_file": config["validation_file"],
            "use_4bit": config.get("load_in_4bit", False),
            "global_step": 0,
            "epoch": 0.0,
            "max_steps": None,
            "logging_steps": config["logging_steps"],
            "eval_steps": config["eval_steps"],
            "save_steps": config["save_steps"],
            "latest_log": {},
            "last_checkpoint": None,
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


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configure_local_cache()
    require_training_dependencies()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    train_file = config["train_file"]
    validation_file = config["validation_file"]
    use_cuda = torch.cuda.is_available()
    base_output_dir = Path(config["output_dir"])
    output_dir = make_run_output_dir(base_output_dir)
    status_writer = TrainingStatusWriter(output_dir=output_dir, config=config)
    model_source = resolve_model_source(config["base_model"])

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": validation_file,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_kwargs = {
        "device_map": "auto" if use_cuda else "cpu",
        "dtype": pick_torch_dtype(torch, config, use_cuda),
    }
    if config.get("load_in_4bit", True):
        if not use_cuda:
            raise RuntimeError(
                "4-bit loading requires CUDA in this training setup. "
                "Use a CPU baseline config with load_in_4bit set to false."
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=config.get("trust_remote_code", False),
        local_files_only=True,
        **model_kwargs,
    )

    def format_example(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = dataset["train"].map(format_example)
    validation_dataset = dataset["validation"].map(format_example)
    estimated_total_steps = estimate_total_steps(config, len(train_dataset))

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        bf16=use_cuda and config.get("bf16", True),
        fp16=use_cuda and config.get("fp16", False),
        warmup_steps=max(1, int(estimated_total_steps * float(config["warmup_ratio"]))),
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to=[],
        max_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        use_cpu=not use_cuda,
    )

    status_writer.update(
        status="ready_to_train",
        train_examples=len(train_dataset),
        validation_examples=len(validation_dataset),
        device="cuda" if use_cuda else "cpu",
        max_steps=estimated_total_steps,
    )

    class StatusCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            status_writer.update(
                status="training",
                global_step=int(state.global_step),
                epoch=float(state.epoch or 0.0),
                max_steps=int(state.max_steps) if state.max_steps is not None else None,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            status_writer.update(
                status="training",
                global_step=int(state.global_step),
                epoch=float(state.epoch or 0.0),
                max_steps=int(state.max_steps) if state.max_steps is not None else None,
                latest_log=logs or {},
            )

        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            status_writer.update(
                status="training",
                global_step=int(state.global_step),
                epoch=float(state.epoch or 0.0),
                last_checkpoint=str(checkpoint_dir),
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            merged = dict(status_writer.state.get("latest_log", {}))
            if metrics:
                merged.update(metrics)
            status_writer.update(
                status="training",
                global_step=int(state.global_step),
                epoch=float(state.epoch or 0.0),
                latest_log=merged,
            )

        def on_train_end(self, args, state, control, **kwargs):
            status_writer.update(
                status="completed",
                completed_at=utc_now_iso(),
                global_step=int(state.global_step),
                epoch=float(state.epoch or 0.0),
            )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[StatusCallback()],
    )
    try:
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        benchmark_report_path = run_post_training_benchmark(output_dir)
        refresh_training_summary()
        status_writer.update(
            status="completed",
            completed_at=utc_now_iso(),
            benchmark_report=str(benchmark_report_path),
        )
        print(f"Training complete. Saved model to {output_dir}")
        print(f"Benchmark report: {benchmark_report_path}")
        return 0
    except Exception as exc:
        status_writer.update(
            status="failed",
            completed_at=utc_now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
