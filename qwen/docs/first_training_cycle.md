# First Training Cycle

## Training Tracks

Keep two tracks:

- local baseline: `Qwen/Qwen2.5-0.5B-Instruct`
- intended GPU run: `Qwen/Qwen2.5-7B-Instruct`

The local baseline is for proving the full training and evaluation loop on CPU. The 7B config remains the intended first real GPU-backed run.

## Local Baseline

Use [v1_local_baseline_config.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/qwen/configs/v1_local_baseline_config.json).

Rationale:

- small enough to be realistic on a CPU-only environment
- same model family as the larger planned run
- good enough to validate SFT formatting, LoRA wiring, save/load behavior, and benchmark evaluation

Training method:

- standard LoRA
- no 4-bit loading
- `float32` on CPU

Command:

```powershell
& '.\.python\python.exe' qwen\scripts\train_lora.py --config qwen\models\configs\v1_local_baseline_config.json
```

This now creates a fresh timestamped run directory under `qwen/models/runs/`, for example
`qwen/models/runs/v1-qwen2.5-0.5b-lora-cpu-20260406-153000`, and prints the exact path when training finishes.

Benchmark evaluation:

```powershell
& '.\.python\python.exe' qwen\scripts\evaluate_benchmark.py --model-path <printed_run_output_dir>
```

## GPU Target

Use [v1_qwen2_5_7b_qlora_config.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/qwen/configs/v1_qwen2_5_7b_qlora_config.json) when a CUDA-capable environment is available.

Rationale:

- 7B-class model is a practical fit for LoRA or QLoRA
- strong instruction-following base
- broad community support for PEFT fine-tuning

Training method:

- QLoRA
- 4-bit loading
- CUDA required

## Dataset Inputs

Training should use:

- `data/processed/train_sft.jsonl`
- `data/processed/validation_sft.jsonl`

Benchmark should remain untouched:

- `data/processed/benchmark_sft.jsonl`

## Required Preparation

Convert the cleaned processed splits into SFT-ready chat examples:

```powershell
& '.\.python\python.exe' scripts\convert_training_data.py
```

## Training Command

For the GPU track:

```powershell
& '.\.python\python.exe' qwen\scripts\train_lora.py --config qwen\models\configs\v1_qwen2_5_7b_qlora_config.json
```

Expected Python packages:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `trl`
- `bitsandbytes`
- `accelerate`

## Benchmark Evaluation

After the selected run completes:

```powershell
& '.\.python\python.exe' qwen\scripts\evaluate_benchmark.py --model-path <run_output_dir>
```

This writes:

- `qwen/experiments/benchmark_report.json`

## What To Check

For the first cycle, focus on:

- valid JSON output rate
- response-type accuracy
- clarification behavior on incomplete prompts
- reduction in hallucinated direct answers
- consistency with the V1 response taxonomy

## Goal Of The First Run

The first run is a pipeline-validation cycle, not a final model.

Success means:

- the training job runs end to end
- the model learns the JSON output shape
- benchmark behavior is directionally better than the untuned base model
- failure modes are visible enough to guide the next dataset iteration
