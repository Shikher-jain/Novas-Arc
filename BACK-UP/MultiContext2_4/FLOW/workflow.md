# MultiContext2 Workflow: Unified File-by-File Logic, Input/Output, and Processing

> Navigation: This file gives per-script inputs/outputs and processing steps. For the narrative, end-to-end overview see `overview.md` in the same `FLOW/` directory.

---

## 1. shared_config.py

**Purpose:** Centralizes all shared logic, configuration, and advanced NLP utilities for FAQ extraction and chatbot systems.

**Main Topics:**

- Topic-to-tone mapping (TOPIC_TONE_MAP)
- Context configuration (CONTEXTS)
- Tone-based greetings (TONE_GREETINGS)
- Sentiment analyzer setup (sia)
- Advanced analysis functions:
  - analyze_content: Sentiment and keyword extraction
  - advanced_topic_detection: Multi-label topic detection
  - nuanced_tone_detection: Advanced tone detection
  - persona_expansion: Multi-label persona detection
  - generate_system_prompt(context, mode): Unified system prompt generator with `minimal`/`compact`/`full` modes
  - generate_merged_system_prompt(...): Concise Behavior/Format/Safety block used by `mode="full"`
  - evaluate_response: Score model responses
  - update_prompt_with_feedback: Adjust prompt based on feedback
  - synthesize_output_json: Output context as JSON
  - prepend_tone_greeting: Add greeting based on detected tone

**Input:**

- Used as an import by all other scripts (no direct user input)

**Output:**

- Provides functions, mappings, and configuration for other modules
- Optional NLP dependencies (NLTK, spaCy, sentence-transformers) are lazily loaded and guarded with fallbacks

---

## 2. app2.py

**Purpose:** Extracts FAQs from websites, analyzes each Q&A for domain/tone/intent/persona, and generates fine-tuning data for chatbot training.

**Main Topics:**

- Web crawling and HTML parsing (BeautifulSoup, requests)
- FAQ detection via keywords, headings, accordions, JSON-LD, etc.
- Deduplication and cleaning of Q&A pairs
- Context detection from URL/content
- Advanced analysis using shared_config.py functions
- System prompt generation for each FAQ
- Data preparation for fine-tuning

**Input:**

- Website URL (root for crawling)
- Optional: crawl depth, worker count, min answer length

**Processing:**

- Crawl site, extract FAQ Q&A pairs
- Detect context (support, technical, sales, booking)
- Clean and deduplicate Q&A pairs
- For each FAQ:
  - Generate UNIQUE training prompt via `advanced_system_prompt_generator(question, answer, context)` (dataset creation)
  - (Runtime steering elsewhere uses `generate_system_prompt(..., mode="compact")` for efficiency)
  - Tag with domain, tone, intent, persona

**Output:**

- Saves processed FAQs as JSONL in FineTuning/ (e.g., FineTuning/domain.jsonl)
- Logs progress and errors

---

## 3. check_and_tune.py

**Purpose:** Fine-tunes OpenAI models with FAQ data and provides an interactive chatbot for validation/testing.

**Main Topics:**

- Training file selection and validation
- Uploading training data to OpenAI
- Fine-tuning job management (start, monitor)
- Saving resulting model ID
- Interactive chatbot loop for validation
- Advanced prompt composition (domain × tone × intent)
- Response analysis and logging

**Input:**

- Training file from FineTuning/ (e.g., rentomojodesk_freshdesk_com.jsonl)

**Processing:**

- Validate JSONL structure
- Upload training file
- Start and monitor fine-tuning job
- Save model ID as .txt next to training file
- Interactive chatbot:
  - Analyze user query (sentiment, domain, tone, intent, persona, platform, touchpoint)
  - Build composite system prompt using `generate_system_prompt(domain=..., tone=..., intent=..., mode)`
  - Call fine-tuned model
  - Return response with analysis metadata

**Output:**

- Model ID saved as .txt in FineTuning/
- Interactive chat responses
- Logs status, errors, and analysis

---

## 4. multimodel.py

**Purpose:** Runs a multi-model ensemble chatbot, querying several fine-tuned models per turn and selecting the best response.

**Main Topics:**

- Model registry loading (FineTuning/*.txt)
- Ensemble querying (multiple models per turn)
- Response scoring and selection
- Chat commands for model control (/models, /use, /k, /all, /both, /style, /strict, /exit)
- Logging of chat turns and model analysis
- Context-aware adaptation (domain/tone/intent/persona + industry/platform/touchpoint) using shared_config
- Style-controlled answer length (`/style short|medium|long`)
- Strict training-grounded mode (`/strict on|off`)

**Input:**

- Loads model IDs from FineTuning/*.txt
- User queries (interactive chat)

**Processing:**

- For each user message:
  - Interpret slash commands (model selection, style, strict mode, etc.)
  - Send query to selected models in parallel
  - Apply strict or relaxed inference config
  - Score and compare responses (keyword overlap + length bonus)
  - Return best answer (optionally show all candidates)
  - Build/refresh steering via `generate_system_prompt(context, mode="compact")`
  - Log each chat turn and model analysis to analysis/chatbot_logs.jsonl

**Output:**

- Interactive ensemble chatbot responses
- Logs of all chat turns and scoring in analysis/chatbot_logs.jsonl

---

## 5. UI_Chatbot.py

**Purpose:** Provides a Streamlit front-end for the ensemble chatbot with quick configuration controls, optional diagnostics, and export utilities.

**Main Topics:**

- Sidebar control panel with grouped expanders (model selection, conversation style, prompts/logging)
- Persistent multiselect for registry entries with optional one-click model suggestions
- Toggles for strict mode, auto answer style, tone-aware greeting, platform/touchpoint auto-detection
- Expanders for system prompt previews, composite prompt summaries, quality metrics, and candidate tables
- JSONL logging into `analysis/` and downloadable chat history payloads

**Input:**

- Model registry from `FineTuning/*.txt`
- User messages entered via Streamlit chat input
- Environment variable `OPENAI_API_KEY`

**Processing:**

- Binds sidebar selections to ensemble controls (mirrors `multimodel.py` capabilities)
- Derives effective platform/touchpoint/answer style based on heuristics and toggles
- Routes prompts through single-model or ensemble pathways, reusing shared helpers
- Persists latest turn summary for quick UI context (cleared with "Clear chat")
- Optionally appends each turn to JSONL logs and renders diagnostics in-place

**Output:**

- Interactive web chat experience
- Optional log entries in `analysis/chat_ui_logs.jsonl` (or custom filenames)
- Downloadable chat transcript as JSON via sidebar button

---

## Data and Logs

- Training data: FineTuning/*.jsonl
- Model IDs: FineTuning/*.txt
- Chat logs (CLI ensemble): analysis/chatbot_logs.jsonl
- Chat logs (Streamlit UI): analysis/chat_ui_logs.jsonl (default) or custom filename from sidebar

---

## Summary Table

| File              | Input(s)                       | Output(s)                                   | Main Logic/Role                                     |
|-------------------|--------------------------------|---------------------------------------------|-----------------------------------------------------|
| app2.py           | Website URL, crawl params      | FineTuning/*.jsonl                          | FAQ extraction, context/tone/intent/persona tagging |
| check_and_tune.py | FineTuning/*.jsonl             | FineTuning/*.txt, interactive chat          | Fine-tuning, adaptive chatbot, model validation     |
| multimodel.py     | FineTuning/*.txt, user queries | analysis/chatbot_logs.jsonl, chat responses | Ensemble chatbot, scoring, style & strict toggles   |
| UI_Chatbot.py     | FineTuning/*.txt, user chat    | analysis/chat_ui_logs.jsonl, UI transcript   | Streamlit UI for ensemble controls & diagnostics    |
| shared_config.py  | (Imported by other scripts)    | (N/A)                                       | Shared NLP, mappings, prompt generation             |

---

**All scripts rely on shared_config.py for shared logic and configuration.**
**All training data and model IDs are stored in FineTuning/.**
**All analysis logs are stored in analysis/.**
