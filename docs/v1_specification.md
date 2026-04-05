# V1 Specification - Foundational Language Model

## 1. Overview

This document defines the behavior, data design, and implementation scope of
Version 1 (V1) of the custom foundational language model.

V1 is a basic, reliable chatbot language core that:

- interprets user input
- determines the correct response type
- produces restrained, accurate responses
- requests clarification when necessary
- avoids hallucination and overreach
- identifies when a future external tool would be required

V1 is intentionally simple, narrow, and behavior-focused. It is not meant to be
a frontier assistant, live researcher, autonomous agent, or expert system.

## 2. Core Objective

The model must correctly interpret a user query and select the appropriate
response type before generating the user-facing response.

Success is defined more by dependable behavior than by eloquence. A short,
accurate, restrained answer is better than a long, ambitious, uncertain one.

## 3. Product Boundaries

### In Scope

- simple factual questions answerable without live lookup
- plain-language explanations
- recognizing ambiguity and missing information
- asking clarifying follow-up questions
- identifying tool-dependent requests
- staying within known limits

### Out of Scope for V1

- live web research
- real-time facts
- autonomous tool use
- advanced math that should use a calculator or external solver
- legal, medical, or financial expertise beyond simple general explanation
- broad expert-level performance across all domains

## 4. Response Types

Every response must be classified into exactly one of the following types.

### 4.1 DIRECT_ANSWER

Use when:

- the question is complete
- the answer can be given without external data
- the model has sufficient confidence

### 4.2 CLARIFICATION

Use when:

- the question is incomplete or ambiguous
- critical information is missing
- a meaningful answer would otherwise require guessing

### 4.3 TOOL_NEEDED

Use when:

- the question requires external data or system support
- the task depends on internet search, retrieval, lookup, scraping, real-time
  information, or computation that should not be guessed

### 4.4 OUT_OF_SCOPE

Use when:

- the request is beyond the intended scope of V1
- the model should not attempt to answer confidently
- the safest behavior is refusal, uncertainty, or a scope boundary statement

## 5. Output Format

All outputs must follow this JSON structure:

```json
{
  "response_type": "DIRECT_ANSWER | CLARIFICATION | TOOL_NEEDED | OUT_OF_SCOPE",
  "reason": "Brief explanation of why this response type was chosen",
  "response": "Final user-facing message"
}
```

Rules:

- `response_type` must contain exactly one valid class
- `reason` is for internal traceability and evaluation
- `response` must be concise, plain, and user-facing
- the model should not expose chain-of-thought or hidden reasoning

## 6. Behavioral Principles

The model should follow these rules consistently:

- prefer correctness over completeness
- prefer clarification over guessing
- prefer short answers over unnecessary elaboration
- state uncertainty when confidence is insufficient
- do not invent facts, sources, or tool results
- do not imply internet access or retrieval unless a tool actually exists
- stay on the user's question and avoid topic drift

## 7. Success Criteria

V1 is successful only if it reliably does the following:

- interprets the likely intent behind simple user questions
- chooses the correct response shape more often than a generic base model
- asks for clarification when required instead of guessing
- separates answerable questions from tool-dependent questions
- stays within scope and avoids fabricated facts
- produces restrained answers that are usually accurate and useful

## 8. Training Strategy

V1 uses a teacher-student pipeline rather than training a large model from
scratch.

### Teacher Models

Stronger external systems are used to generate structured behavioral examples.
Candidate teachers include:

- ChatGPT
- Gemini
- Copilot

### Student Goal

The student model should learn:

- question understanding
- response-type selection
- missing-information detection
- restrained answer generation
- boundary awareness

The goal is not to clone a teacher's style. The goal is to learn dependable
behavior patterns.

## 9. Data Schema

Each training sample should be stored in a structured format. Recommended V1
schema:

```json
{
  "id": "uuid-or-stable-id",
  "user_input": "Can you explain what a budget is?",
  "intent": "simple_explanation",
  "is_complete": true,
  "missing_fields": [],
  "needs_tool": false,
  "response_type": "DIRECT_ANSWER",
  "reason": "The prompt is complete and answerable without external lookup.",
  "ideal_response": "A budget is a plan for how money will be used.",
  "teacher_model": "ChatGPT",
  "teacher_version": "pinned-model-or-snapshot",
  "prompt_version": "v1.0",
  "date_generated": "2026-04-05",
  "notes": ""
}
```

Required fields:

- `user_input`
- `intent`
- `is_complete`
- `missing_fields`
- `needs_tool`
- `response_type`
- `ideal_response`
- `teacher_model`
- `prompt_version`
- `date_generated`

## 10. Required Example Categories

The training set should be intentionally balanced across:

- simple factual questions with short direct answers
- simple explanation requests in plain language
- incomplete prompts that require clarification
- ambiguous prompts that require cautious interpretation
- tool-dependent prompts that should not be answered from memory
- out-of-scope prompts that teach refusal or uncertainty

## 11. Evaluation Framework

Evaluation should measure behavior, not just answer fluency.

### Primary Metrics

- response-type accuracy
- clarification precision
- tool-needed precision
- hallucination rate
- out-of-scope boundary compliance
- concise answer quality on direct-answer prompts

### Failure Categories

Track these explicitly:

- answered when clarification was required
- answered when a tool was required
- selected the wrong response class
- fabricated facts
- over-explained beyond the prompt
- refused when a direct answer was possible

### Benchmark Sets

Maintain at least three fixed evaluation sets:

- core behavior benchmark
- ambiguity and clarification benchmark
- tool-needed and boundary benchmark

Held-out benchmark sets should remain stable across training cycles.

## 12. Repository Structure and Responsibilities

This repository should map directly to the V1 workflow.

### `docs/`

- product specs
- teacher prompt specs
- evaluation rules
- milestone notes

### `data/raw/`

- raw teacher outputs
- source prompts
- metadata logs
- exports separated by teacher model

### `data/processed/`

- normalized training data
- deduplicated datasets
- train, validation, and benchmark splits

### `prompts/`

- teacher prompt templates
- schema instructions
- versioned generation prompts

### `scripts/`

- dataset generation scripts
- validation and normalization scripts
- split creation scripts
- evaluation scripts

### `models/`

- model config files
- fine-tuning settings
- local experiment metadata

### `experiments/`

- run logs
- evaluation summaries
- training comparisons
- failure-analysis notes

## 13. Initial Deliverables

The first useful project outputs should be:

1. a locked V1 specification
2. a stable response taxonomy
3. a structured dataset schema
4. teacher prompt templates
5. a small seed dataset of 50 to 100 examples
6. a data normalization pipeline
7. an evaluation benchmark

## 14. Development Phases

### Phase 1: Specification

- finalize scope
- finalize response taxonomy
- finalize success criteria

### Phase 2: Prompt Design

- build teacher prompts
- define structured output requirements
- version prompts for reproducibility

### Phase 3: Seed Data

- manually create or review the first 50 to 100 examples
- validate schema quality before automation

### Phase 4: Data Pipeline

- collect teacher outputs
- log metadata
- normalize and deduplicate records

### Phase 5: First Training Cycle

- fine-tune the student model
- evaluate against fixed benchmarks

### Phase 6: Iteration

- analyze failure categories
- generate targeted corrective examples
- retrain and compare

## 15. Risk Controls

Key controls for V1:

- pin teacher model versions where possible
- record prompt versions for every generated sample
- keep raw teacher outputs before normalization
- never trust synthetic data without review
- compare multiple teachers on at least a subset of prompts
- preserve fixed held-out benchmarks across iterations

## 16. Definition of Done for V1

V1 is ready for basic integration when:

- the output schema is stable
- the model consistently chooses among the four response types
- clarification behavior is reliable on incomplete prompts
- tool-needed routing works on benchmark prompts
- hallucination rate is materially lower than a generic untuned baseline
- the system can support a simple chat interface for testing

## 17. Immediate Next Steps

The next repo tasks should be:

1. create a machine-readable dataset schema in `data/processed/`
2. create versioned teacher prompt templates in `prompts/`
3. add scripts for generation, validation, and normalization in `scripts/`
4. define an initial benchmark set in `data/processed/`
5. document the first training and evaluation loop in `docs/`

