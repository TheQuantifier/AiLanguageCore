# AiLanguageCore

AiLanguageCore is a repository for building Version 1 of a custom foundational
language model: a small, owned chatbot core focused on reliable behavior rather
than broad frontier capability.

The V1 goal is to produce a model that can:

- understand simple user requests
- choose the correct response type
- ask clarifying questions when needed
- recognize when a tool would be required
- avoid hallucination and overclaiming

This project uses a teacher-student workflow. Stronger external models generate
structured behavioral examples, those examples are cleaned into a stable schema,
and a smaller student model is fine-tuned on the curated dataset.

## V1 Scope

Version 1 is intentionally narrow. It is a language-behavior core, not a full
assistant.

In scope:

- simple factual answers
- plain-language explanations
- clarification on incomplete prompts
- response-type routing
- boundary-aware and restrained responses

Out of scope:

- live internet research
- autonomous tool execution
- expert-level legal, medical, or financial advice
- large-model pretraining from scratch

The source of truth for V1 behavior is
[docs/v1_specification.md](docs/v1_specification.md).

## Response Taxonomy

Every model output in V1 should map to one of four classes:

- `DIRECT_ANSWER`
- `CLARIFICATION`
- `TOOL_NEEDED`
- `OUT_OF_SCOPE`

These classes are central to dataset design, evaluation, and later inference.

## Repository Layout

```text
project-root/
|-- docs/           # specs, plans, evaluation notes
|-- data/
|   |-- raw/        # raw teacher outputs and logs
|   `-- processed/  # normalized datasets and splits
|-- prompts/        # teacher prompt templates
|-- scripts/        # data, validation, and evaluation scripts
|-- models/         # model configs and fine-tuning settings
|-- experiments/    # run logs and comparison notes
`-- README.md
```

## Planned Workflow

1. Finalize the V1 specification and output schema.
2. Create teacher prompt templates with stable versions.
3. Generate a seed set of structured examples.
4. Normalize and validate the dataset.
5. Build train, validation, and benchmark splits.
6. Fine-tune the first student model.
7. Evaluate, identify failures, and iterate.

## Immediate Next Steps

- add a machine-readable dataset schema in `data/processed/`
- create teacher prompt templates in `prompts/`
- implement data generation and validation scripts in `scripts/`
- define held-out benchmark sets
- document the first training loop

## Status

This repository is in the project setup phase. The core V1 specification is in
place, and the next stage is to turn that specification into data schemas,
prompt templates, and training/evaluation tooling.

