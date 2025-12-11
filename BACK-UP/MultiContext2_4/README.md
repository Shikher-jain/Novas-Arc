# MultiContext2 — FAQ Extraction, Fine‑Tuning, and Ensemble Chatbot

A production‑ready toolkit to:

- Crawl a website and extract FAQ Q/A pairs (`app2.py`)
- Fine‑tune an OpenAI model on those FAQs (`check_and_tune.py`)
- Chat using an ensemble of fine‑tuned models with smart routing and scoring (`multimodel.py`)
- Share NLP utilities and prompt logic (`shared_config.py`)
- Chat through an interactive Streamlit UI with guardrails (`UI_Chatbot.py`)

This guide covers setup, usage, and troubleshooting so anyone can clone the repo and run it.

---

## Quick Start (Windows PowerShell)

```powershell
git clone https://git-codecommit.us-east-2.amazonaws.com/v1/repos/shiker-python-datascience
cd "Faq Extractor\MultiContext2"

python -m venv .venv
\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
$env:OPENAI_API_KEY="sk-..."   # set your key for this session
```

Extract FAQs and create training data:

```powershell
python .\app2.py
```

Fine‑tune and save model ID:

```powershell
python .\check_and_tune.py
```

Run the ensemble chatbot:

```powershell
python .\multimodel.py
```

Launch the Streamlit chat UI:

```powershell
streamlit run .\UI_Chatbot.py
```

---

## Requirements

- Python 3.10+ (Union types like `str | None` are used)
- OpenAI API key in environment or `.env` file

Recommended:

- Virtual environment (venv)
- Stable internet (for crawling and model APIs)

Optional (advanced NLP):

- NLTK data: `vader_lexicon`, `punkt`
- spaCy + model `en_core_web_sm`
- sentence-transformers (semantic scoring)

Install optional extras: see Setup → “(Optional) Install NLP extras” for commands.

---

## Project Structure

```text
MultiContext2/
├── app2.py              # Website crawler → FAQ extractor → JSONL exporter
├── check_and_tune.py    # Fine-tuning orchestration + model ID persistence
├── multimodel.py        # Ensemble chatbot with slash commands
├── shared_config.py     # Shared NLP utilities + prompt generation
├── UI_Chatbot.py        # Streamlit chat UI for fine-tuned models
├── FineTuning/          # Training JSONL and saved fine-tuned model IDs (.txt)
├── analysis/            # Logs and analysis artifacts (e.g., chatbot_logs.jsonl)
├── FLOW/                # Docs (Explanation.txt, overview.md, workflow.md)
└── README.md            # This file
```

---

## How It Works (End‑to‑End)

1) `app2.py` crawls a site, extracts FAQ Q/A pairs with multiple strategies, and writes a JSONL file in `FineTuning/`.
2) `check_and_tune.py` validates the JSONL, fine‑tunes an OpenAI model, and saves the resulting model ID to `FineTuning/<name>.txt`.
3) `multimodel.py` loads all saved model IDs (`FineTuning/*.txt`) and runs an ensemble chat loop, querying multiple models in parallel and picking the best answer.
4) `UI_Chatbot.py` provides a Streamlit front-end to the ensemble pipeline, exposing the same controls via an interactive web UI.

All scripts share analysis and prompt utilities from `shared_config.py`.

---

## Setup

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
```

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

1. Set your OpenAI API key:

- Option A (recommended): create a `.env` file in `MultiContext2/`

```env
OPENAI_API_KEY=sk-your-key
```

- Option B (session only):

```powershell
$env:OPENAI_API_KEY="sk-your-key"
```

Note: Place `.env` in the project root: `MultiContext2/.env`.

1. (Optional) Install NLP extras:

```powershell
pip install spacy sentence-transformers
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon punkt
```

---

## Usage

### 1 Extract FAQs — `app2.py`

```powershell
python .\app2.py
```

- Enter the starting URL when prompted.
- Set crawl depth, workers, and minimum answer length.
- Output: `FineTuning/<site>.jsonl` (training data).

Notes:

- Multiple extraction strategies (JSON‑LD, accordions, details/summary, headings, tables, Q:/A:)
- Deduplicates and cleans Q/A pairs
- Tags each FAQ via `shared_config` (context, tone, intent, persona)
- Creates UNIQUE training prompts per FAQ using `advanced_system_prompt_generator(question, answer, context)`.
- Runtime steering elsewhere uses `generate_system_prompt(..., mode="compact")` to save tokens.

### 2 Fine‑tune and save model — `check_and_tune.py`

Edit `TRAINING_FILE_PATH` at the top of the script (or put your file there), then:

```powershell
python .\check_and_tune.py
```

- Validates JSONL structure.
- Uploads to OpenAI, starts fine‑tuning, monitors status.
- Saves resulting model ID next to the JSONL as `FineTuning/<name>.txt`.

Requirements:

- `OPENAI_API_KEY` must be set
- JSONL must contain messages with roles: `system`, `user`, `assistant`

JSONL format: each line must contain `{"messages": [{"role": "system", "content": ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`.

### 3 Ensemble Chatbot — `multimodel.py`

```powershell
python .\multimodel.py
```

- Loads all model IDs from `FineTuning/*.txt`.
- Interactive chat prompt `You: …`.
- Runs multi‑model inference in parallel and picks the best response.

Slash commands:

- `/models`          → list available model keys (from `FineTuning/*.txt`)
- `/use k1,k2`       → restrict to specific keys this turn (also sets max models)
- `/k N`             → set max models per turn (N ≥ 1)
- `/all`             → use all available models this turn
- `/both`            → toggle showing alternative candidates in logs
- `/style short|medium|long` → adjust target response length & structure
- `/strict [on|off]` → toggle training-grounded conservative mode (no speculation)
- `/exit`            → quit (also `exit`, `quit`)

Examples:

```text
You: /models
You: /use aws_amazon_com
You: /k 1
You: Tell me about yourself
```

Tip:

- `ENSEMBLE_MAX_MODELS` env var controls default per‑turn model limit (default 2).

```powershell
$env:ENSEMBLE_MAX_MODELS="1"
python .\multimodel.py
```

### 4 Streamlit Chat UI — `UI_Chatbot.py`

```powershell
streamlit run .\UI_Chatbot.py
```

- Control panel sidebar groups model selection, conversation style, and logging toggles.
- Multiselect one or more fine-tuned models, or let the ensemble auto-select each turn.
- Quick "Suggest a model" button highlights a random registry entry for fast testing.
- Toggle strict mode, auto answer style, tone-aware greetings, and platform/touchpoint hints.
- Optional expanders reveal generated system prompts, composite prompts, metrics, and candidate scores.
- Chat transcript can be downloaded as JSON; logs append to `analysis/` when enabled.

Strict mode:

- Toggle with `/strict` (switch) or explicitly `/strict on`, `/strict off`.
- When enabled: temperature=0, minimal creativity, answers restricted to content plausibly present in fine‑tuning data; uncertain info → `Not available in training data.`
- When disabled: normal generation instructions guided by `/style` length with moderate creativity and clarity enhancements.

---

## Shared Prompt Generation (`shared_config.py`)

Unified entry: `generate_system_prompt(context=..., mode=...)`

- Modes:
  - `minimal` (smallest): Domain, Intent, Tone + a single “Rules” line
  - `compact` (small): one‑line header + short rules, optional context summary
  - `full` (comprehensive): optimized house‑style Behavior/Format/Safety checklist

Notes:

- Pass `mode` to control size; prefer `minimal` or `compact` to save tokens
- `generate_merged_system_prompt` powers the concise `full` block
- Optional NLP deps (NLTK, spaCy, sentence‑transformers) are guarded; features degrade gracefully

Dataset vs runtime:

- Dataset creation (`app2.py`): `advanced_system_prompt_generator(question, answer, context)` generates unique per‑FAQ system prompts stored in JSONL.
- Runtime (`check_and_tune.py`, `multimodel.py`): `generate_system_prompt(..., mode="compact")` builds a small, efficient steering prompt per request.

Other utilities:

- `analyze_query`: builds `PromptContext` (domain(s), intent, tone, personas, entities, complexity, summary)
- `analyze_content`: sentiment + keyword extraction
- `nuanced_tone_detection`: fine‑grained tone classification (e.g., formal, friendly, empathetic, assertive)
- `advanced_topic_detection`: multi‑label topic detection for routing and prompt shaping
- `persona_expansion`: infer/enrich personas (e.g., expert, teacher, concierge) from query/context
- `evaluate_response`: score a reply for relevance, correctness, completeness
- `update_prompt_with_feedback`: adjust the system prompt using prior errors or user feedback
- `synthesize_output_json`: assemble a compact JSON summary of analysis/answer fields
- `prepend_tone_greeting`: add a greeting aligned to detected tone/persona and context

---

## Troubleshooting

- Missing API key: set `$env:OPENAI_API_KEY` or add `.env`.
- “No model produced a response”: ensure `FineTuning/*.txt` exist and contain valid model IDs.
- Python TypeError for “X | Y”: use Python 3.10+ (or switch to `typing.Optional/Union`).
- NLTK/spaCy warnings: optional; install extras if needed.
- NLTK data missing: `python -m nltk.downloader vader_lexicon punkt`.
- Fine‑tuning errors: validate JSONL and ensure roles `system`/`user`/`assistant` exist.
- Prompts too large/duplicated: use `generate_system_prompt(mode="minimal"|"compact")`.

---

## Security & Cost

- API usage costs: ensemble queries multiple models; reduce `/k` or set `ENSEMBLE_MAX_MODELS` to limit spend.
- Never commit API keys; use `.env` or environment variables.

---

## Contributing

- Keep `shared_config.py` as the single source of truth for mappings and NLP utilities.
- Prefer `generate_system_prompt` at call sites to avoid prompt bloat or duplication.
- Add tests or logs under `analysis/` when extending functionality.

More: See `FLOW/Explanation.txt`, `FLOW/overview.md`, and `FLOW/workflow.md` for architecture diagrams, end‑to‑end workflow, and deep‑dive explanations.
