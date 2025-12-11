# FAQ Extraction & Adaptive Chatbot


## MultiContext2: FAQ Extraction & Adaptive Chatbot

### Overview
MultiContext2 is a modular framework for FAQ-driven chatbot development. It enables scalable extraction of domain-specific Q&A pairs, advanced NLP analysis, and adaptive conversational AI fine-tuning using OpenAI models.

### Architecture
- `shared_config.py`: Centralized NLP utilities, mappings, and prompt logic.
- `app2.py`: FAQ crawler, extractor, JSONL exporter.
- `check_and_tune.py`: Fine-tuning orchestration, adaptive chatbot.
- `multimodel.py`: Ensemble chatbot, multi-model scoring.

### Setup
- Python 3.9+
- OpenAI API key in `.env`
- Install dependencies: `pip install -r requirements.txt`
- (Recommended) Virtual environment: `python -m venv .venv && .\.venv\Scripts\Activate.ps1`
- (Optional) Download NLTK resources: `python -m nltk.downloader vader_lexicon punkt`

### Usage
- Extract FAQs: `python app2.py` (outputs training data to `FineTuning/`)
- Fine-tune & validate chatbot: `python check_and_tune.py`
- Ensemble chatbot: `python multimodel.py`

### Data Flow
Website URL → app2.py → FineTuning/<site>.jsonl → check_and_tune.py → OpenAI fine-tune → Adaptive/ensemble chatbot session

### Features
- Multi-strategy FAQ extraction, de-duplication
- Shared NLP utilities (sentiment, topic, persona, tone, intent)
- Automated OpenAI JSONL generation, fine-tuning pipeline
- Interactive chatbot loop with multi-dimensional analysis and cost tracking
- Ensemble model support for multi-domain, multi-persona evaluation

### Troubleshooting
- No FAQs extracted: Try alternate URLs or adjust crawl depth
- API key issues: Ensure `.env` is present and valid
- Fine-tuning delays: OpenAI queues may be busy; monitor job status or retry
- Model quality: Combine real and synthetic FAQs for broader coverage
- Sentiment warnings: Ensure NLTK and `vader_lexicon` are installed

### Extending
- Add industries/platforms in `check_and_tune.py`
- Update mappings in `shared_config.py`
- Enhance extraction logic in `app2.py`

### Summary
MultiContext2 delivers a scalable pipeline for FAQ-driven chatbot development, supporting advanced context adaptation, multi-model evaluation, and rapid extension to new domains.

## Overview

This directory contains a production-ready toolkit for building FAQ-driven training data and an adaptive chatbot that tailors every reply by **Domain × Tone × Intent**. The codebase is split into three focused modules that share a single configuration layer, ensuring consistent behaviour while keeping maintenance effort low.

## Features

- Recursive FAQ crawler with 10+ extraction strategies and aggressive de-duplication
- Shared NLP utilities (sentiment, topic, persona, tone detection) backed by NLTK
- Automated JSONL generation following the OpenAI fine-tuning message format
- Fine-tuning pipeline for `gpt-3.5-turbo` with job monitoring and model registry
- Interactive chatbot loop that performs full 9-dimensional analysis before responding
- Built-in cost tracking with per-request warnings and local fallbacks

## Repository Structure

```text
MultiContext2/
├── app2.py                 # Website crawler + FAQ extractor + JSONL exporter
├── check_and_tune.py       # Fine-tuning orchestration + adaptive chatbot
├── shared_config.py        # Shared NLP config, topic maps, persona/tone logic
├── FineTuning/             # Generated training datasets (JSONL)and  Stored fine-tuned model identifiers
└── README.md               # Project documentation (this file)
```

Auxiliary scripts such as `generate_training_data.py` (synthetic data generator) live one directory up and can be used interchangeably with the files produced by `app2.py`.

## Requirements

- Python 3.9+
- OpenAI API access with a valid `OPENAI_API_KEY`
- The packages listed in `requirements.txt`, installed via `pip install -r requirements.txt`

## Setup

1. Create a virtual environment (optional but recommended) and install dependencies:

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

2. Create a `.env` file at the project root with your OpenAI credentials:

  ```env
  OPENAI_API_KEY=sk-your-key
  ```

3. Ensure output folders exist (they are created on demand, but you can pre-create them):

  ```powershell
  mkdir -p FineTuning
  ```

4. (Optional) Pre-download NLTK resources if you are on an offline or firewalled machine:

  ```powershell
  python -m nltk.downloader vader_lexicon punkt
  ```

  The scripts download these packages automatically when missing, but installing them up front avoids warnings on the first run.

## Usage

### 1. Extract FAQs and Generate Training Data (`app2.py`)

```powershell
python app2.py
```

- Provide the starting URL when prompted.
- Configure crawl depth, worker count, and minimum answer length interactively.
- The script crawls same-domain links, extracts FAQ pairs, classifies context, and writes a consolidated JSONL file to `FineTuning/` (one file per domain).

### 2. Fine-tune and Validate the Chatbot (`check_and_tune.py`)

```powershell
python check_and_tune.py
```

- Validates the training JSONL, uploads it to OpenAI, and starts a fine-tuning job.
- Monitors progress until the job succeeds and stores the resulting model ID in `FineTuning/<site_name>_com.txt>`.
- Launches an interactive chat loop that:
  - Detects industry domain, platform tone, user intent, sentiment, topics, and personas.
  - Builds a Domain × Tone × Intent x Persona system prompt (using `shared_config.py`).
  - Warns about estimated API cost and offers a local fallback before each call.

You can rerun the chatbot later; it automatically reuses the saved model ID.

### 3. (Optional) Generate Synthetic Coverage (`generate_training_data.py`)

Use the synthetic generator when you need balanced training data across industries and tones:

```powershell
python ..\generate_training_data.py
```

The output is compatible with `check_and_tune.py`.

## Shared Configuration Highlights (`shared_config.py`)

- `TOPIC_TONE_MAP`: 40+ topic → tone mappings used for nuanced prompt shaping.
- `CONTEXTS`: Canonical system prompts for support, technical, sales, and booking contexts.
- `TONE_GREETINGS`: Greeting templates that match detected tone.
- `analyze_content()`, `advanced_topic_detection()`, `nuanced_tone_detection()`, `persona_expansion()`: Core NLP utilities reused by both executables.
- `advanced_system_prompt_generator()`: Creates unique prompts for every FAQ pair during dataset creation.

Because every module imports from `shared_config.py`, updates to tone/industry logic propagate automatically.

## Data Flow

```text
Website URL → app2.py → FineTuning/<site>.jsonl → check_and_tune.py → OpenAI fine-tune → Adaptive chatbot session
```

During chat sessions, each question passes through a 9-dimensional analysis pipeline covering domain, tone, intent, sentiment, topics, personas, platform, touchpoint, and metadata before a response is generated.

## Cost Controls

- The chatbot estimates token usage and warns about projected cost before calling the OpenAI API.
- Users can choose `yes` (call API), `no` (skip), or `local` (use templated response) for each prompt.
- Session totals are printed at exit, making it easy to track spend.

## Troubleshooting

- **No FAQs extracted:** The site may rely on client-side rendering. Try another URL or reduce crawl depth.
- **`OPENAI_API_KEY` missing:** Confirm `.env` is present and the key is valid.
- **Fine-tuning delayed:** OpenAI queues can be busy; check job status in the dashboard or retry later.
- **Model quality low:** Increase coverage by combining real FAQs with the synthetic generator.
- **Sentiment warnings:** If you see messages about `SentimentIntensityAnalyzer`, ensure `nltk` is installed and the `vader_lexicon` package is available (see setup step 4).

## Extending the System

- Add industries or platforms by editing `INDUSTRY_DOMAINS` and `PLATFORM_TONE_MAP` in `check_and_tune.py`.
- Add new topics or greeting styles in `shared_config.py` to influence both extraction and responses.
- Integrate alternative crawling rules inside `app2.py` if you encounter new FAQ layouts.
