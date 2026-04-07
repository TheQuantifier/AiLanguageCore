# Architecture Overview

## Purpose

AiLanguageCore is a teacher-student training pipeline for a narrow chatbot core.
The student model is trained to classify each prompt into exactly one response
type and then emit a restrained JSON response.

The root project is now the native-model path: a byte-level decoder-only
transformer trained from scratch on your data.

The old Qwen plus LoRA workflow still exists, but it is isolated under
[qwen](/C:/Users/jhand/Documents/Github/AiLanguageCore/qwen) so it can be removed as a single subtree.

The four response types are:

- `DIRECT_ANSWER`
- `CLARIFICATION`
- `TOOL_NEEDED`
- `OUT_OF_SCOPE`

The behavioral source of truth is [v1_specification.md](/C:/Users/jhand/Documents/Github/AiLanguageCore/docs/v1_specification.md).

## System Flow

```text
teacher API -> raw examples -> cleaned splits -> SFT chat data -> LoRA training
      -> default benchmark -> experiment report -> run summary CSV
```

More explicitly:

```text
data/raw/v1_seed_dataset.json
data/raw/generated_dataset.json
        |
        v
scripts/prepare_dataset.py
        |
        v
data/processed/train.json
data/processed/validation.json
data/processed/benchmark.json
        |
        v
scripts/convert_training_data.py
        |
        v
data/processed/train_sft.jsonl
data/processed/validation_sft.jsonl
data/processed/benchmark_sft.jsonl
        |
        v
scripts/train_native_model.py
        |
        +--> models/runs/<timestamped-run>/
        +--> scripts/evaluate_native_model.py
        |         |
        |         v
        |   experiments/benchmark_report-<run>.json
        |
        +--> scripts/summarize_training_runs.py
                  |
                  v
            experiments/training_runs_summary.csv
```

## Folder Map

### [docs](/C:/Users/jhand/Documents/Github/AiLanguageCore/docs)

Project documentation.

- [v1_specification.md](/C:/Users/jhand/Documents/Github/AiLanguageCore/docs/v1_specification.md): behavioral rules and response taxonomy
- [command_help.md](/C:/Users/jhand/Documents/Github/AiLanguageCore/docs/command_help.md): operational commands and shortcuts
- [architecture_overview.md](/C:/Users/jhand/Documents/Github/AiLanguageCore/docs/architecture_overview.md): this file

### [data/raw](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw)

Unprocessed source material.

- [v1_seed_dataset.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/v1_seed_dataset.json): hand-authored and corrected examples
- [generated_dataset.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/generated_dataset.json): teacher-generated records that passed generation-stage validation
- [v1_fixed_benchmark.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/v1_fixed_benchmark.json): fixed held-out benchmark source
- [failed_responses](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/failed_responses): raw failed API outputs for debugging

### [data/processed](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed)

Cleaned and training-ready artifacts.

- [full_dataset.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/full_dataset.json): cleaned pool plus held-out benchmark
- [train.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/train.json): train split
- [validation.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/validation.json): validation split
- [benchmark.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/benchmark.json): default held-out benchmark
- [train_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/train_sft.jsonl): chat-format training data
- [validation_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/validation_sft.jsonl): chat-format validation data
- [benchmark_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/benchmark_sft.jsonl): chat-format default benchmark
- focused benchmark files for stress and boundary checks
- [preparation_report.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/preparation_report.json): dataset cleaning and split summary

### [prompts](/C:/Users/jhand/Documents/Github/AiLanguageCore/prompts)

Teacher prompt templates.

- [teacher_generation_prompt_v1.md](/C:/Users/jhand/Documents/Github/AiLanguageCore/prompts/teacher_generation_prompt_v1.md): instructions used to ask the teacher model for structured records

### [scripts](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts)

Operational pipeline scripts.

- [generate_data.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/generate_data.py): generate raw labeled records from the teacher API
- [prepare_dataset.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/prepare_dataset.py): clean, dedupe, and split raw data
- [convert_training_data.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/convert_training_data.py): convert splits into SFT chat JSONL
- [train_native_model.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/train_native_model.py): train the native model from scratch and auto-run the default benchmark
- [evaluate_native_model.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/evaluate_native_model.py): evaluate a saved native model against the default benchmark
- [validate_dataset.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/validate_dataset.py): validate dataset schema and quality
- [show_training_status.ps1](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/show_training_status.ps1): watch the live training status file
- [summarize_training_runs.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/summarize_training_runs.py): summarize all runs into a table and CSV

### [models](/C:/Users/jhand/Documents/Github/AiLanguageCore/models)

Native-model configs and run outputs.

- [configs](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/configs): native-model configs
- [runs](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/runs): native-model runs

### [qwen](/C:/Users/jhand/Documents/Github/AiLanguageCore/qwen)

Legacy Qwen/LoRA track, fully isolated from the native model path.

### [experiments](/C:/Users/jhand/Documents/Github/AiLanguageCore/experiments)

Evaluation and reporting outputs.

- benchmark reports per run
- [training_runs_summary.csv](/C:/Users/jhand/Documents/Github/AiLanguageCore/experiments/training_runs_summary.csv): spreadsheet-style summary of all runs

### [.cache](/C:/Users/jhand/Documents/Github/AiLanguageCore/.cache)

Local Hugging Face cache used by training and evaluation.

### [.python](/C:/Users/jhand/Documents/Github/AiLanguageCore/.python)

Project-local Python runtime. All commands use this interpreter so the repo
always runs with the same installed packages.

## Process By Script

### 1. Generate raw records

Run:

```powershell
& $py scripts\generate_data.py --count 50
```

[generate_data.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/generate_data.py) does the following:

- loads `.env` and reads `TEACHER_API_KEY`
- loads few-shot examples from [v1_seed_dataset.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/v1_seed_dataset.json)
- loads the teacher instructions from [teacher_generation_prompt_v1.md](/C:/Users/jhand/Documents/Github/AiLanguageCore/prompts/teacher_generation_prompt_v1.md)
- calls the teacher model through the Google OpenAI-compatible endpoint
- asks for complete JSON records, not just prompts
- validates the returned records
- removes duplicates against existing raw records
- salvages partial records from malformed batch outputs when possible
- appends valid new records to [generated_dataset.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/generated_dataset.json)

If a batch fails, the raw model response is written to
[failed_responses](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/failed_responses).

### 2. Prepare the cleaned dataset

Run:

```powershell
& $py scripts\prepare_dataset.py
```

[prepare_dataset.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/prepare_dataset.py) does the following:

- loads the seed dataset, generated dataset, and fixed benchmark dataset
- normalizes punctuation and whitespace
- validates each record against the V1 schema rules
- removes weak, malformed, or awkward examples
- deduplicates exact and near-duplicate `user_input` values
- prefers stronger records when duplicates conflict
- assigns stable IDs
- removes fixed benchmark prompts from the train/validation pool
- creates deterministic train and validation splits
- keeps the fixed benchmark separate and untouched

Outputs:

- [full_dataset.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/full_dataset.json)
- [train.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/train.json)
- [validation.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/validation.json)
- [benchmark.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/benchmark.json)
- [preparation_report.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/preparation_report.json)

### 3. Convert into SFT chat data

Run:

```powershell
& $py scripts\convert_training_data.py
```

[convert_training_data.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/convert_training_data.py) converts each cleaned record into a chat conversation with:

- one system prompt defining the four response types
- one user message
- one assistant JSON message

This produces the exact format expected by the trainer.

Outputs:

- [train_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/train_sft.jsonl)
- [validation_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/validation_sft.jsonl)
- [benchmark_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/benchmark_sft.jsonl)

### 4. Train the native model

Run:

```powershell
& $py scripts\train_native_model.py --config models\configs\v1_native_byte_transformer_config.json
```

[train_native_model.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/train_native_model.py) does the following:

- loads a native model config from [models/configs](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/configs)
- loads the SFT train and validation datasets
- encodes the data with the project byte-level tokenizer
- trains a small decoder-only transformer from scratch
- writes a new timestamped run directory under [models/runs](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/runs)
- updates a live `training_status.json` file during training

The run folder contains:

- native model weights
- tokenizer config
- model config
- `training_status.json`

### 5. Auto-run the default benchmark

When training succeeds, [train_native_model.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/train_native_model.py) automatically runs [evaluate_native_model.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/evaluate_native_model.py) on the default benchmark.

[evaluate_native_model.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/evaluate_native_model.py):

- loads the saved model or adapter
- loads [benchmark_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/benchmark_sft.jsonl) by default
- generates one answer per held-out benchmark item
- extracts the returned JSON
- checks whether the output is valid JSON
- compares predicted `response_type` against the expected one
- writes a benchmark report to [experiments](/C:/Users/jhand/Documents/Github/AiLanguageCore/experiments)

Primary benchmark metrics:

- valid JSON rate
- response type accuracy

### 6. Summarize all runs

Run:

```powershell
& $py scripts\summarize_training_runs.py
```

[summarize_training_runs.py](/C:/Users/jhand/Documents/Github/AiLanguageCore/scripts/summarize_training_runs.py):

- scans [models/runs](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/runs) for `training_status.json`
- matches each run with its benchmark report
- prints a terminal summary table
- writes [training_runs_summary.csv](/C:/Users/jhand/Documents/Github/AiLanguageCore/experiments/training_runs_summary.csv)

The CSV is the easiest spreadsheet-style view of the whole project history.

## One-Page Walkthrough

### Command: create data

Input:

- your teacher API key
- the current prompt template
- the current seed dataset

Execution:

- generate more raw records
- merge and clean raw data
- convert cleaned data to SFT JSONL

Artifacts produced or updated:

- [generated_dataset.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/raw/generated_dataset.json)
- [train.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/train.json)
- [validation.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/validation.json)
- [benchmark.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/benchmark.json)
- [train_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/train_sft.jsonl)
- [validation_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/validation_sft.jsonl)
- [benchmark_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/benchmark_sft.jsonl)

### Command: train

Input:

- [v1_native_byte_transformer_config.json](/C:/Users/jhand/Documents/Github/AiLanguageCore/models/configs/v1_native_byte_transformer_config.json)
- [train_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/train_sft.jsonl)
- [validation_sft.jsonl](/C:/Users/jhand/Documents/Github/AiLanguageCore/data/processed/validation_sft.jsonl)

Execution:

- create a timestamped run directory
- train the native transformer from scratch
- auto-run the default benchmark
- refresh the run summary CSV

Artifacts produced or updated:

- `models/runs/...` new run folder
- run-local `training_status.json`
- `benchmark_report-<run>.json` in [experiments](/C:/Users/jhand/Documents/Github/AiLanguageCore/experiments)
- [training_runs_summary.csv](/C:/Users/jhand/Documents/Github/AiLanguageCore/experiments/training_runs_summary.csv)

### Command: summarize

Input:

- existing run folders
- existing benchmark reports

Execution:

- scan training history
- aggregate metrics
- write CSV

Artifact produced or updated:

- [training_runs_summary.csv](/C:/Users/jhand/Documents/Github/AiLanguageCore/experiments/training_runs_summary.csv)

## Practical Mental Model

Think of the repo as three layers:

1. `raw behavior data`
2. `cleaned supervised training data`
3. `trained student runs plus benchmark evidence`

Or, more simply:

```text
generate examples -> clean examples -> train model -> measure model -> compare runs
```
