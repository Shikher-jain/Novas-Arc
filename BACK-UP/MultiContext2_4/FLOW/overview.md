# MultiContext2: Unified Step-by-Step Pipeline

> Navigation: This is the narrative, end-to-end view. For a concise per-script breakdown (inputs, processing, outputs) see `workflow.md` in this same folder.

This project builds smart chatbots that answer questions in different styles and for different topics, using data scraped from websites and fine-tuned models. It supports multi-model ensemble querying for robust, context-aware answers.

---

## 1. Shared Configuration (`shared_config.py`)

**Purpose:** Centralizes all shared logic, configuration, and advanced NLP utilities for FAQ extraction and chatbot systems.

**Key Features:**

- Topic-to-Tone Mapping: Matches topics (health, finance, gaming, etc.) to the chatbot's communication style (empathetic, professional, quirky, etc.).
- Context Detection: Decides if a question is about support, sales, technical help, or booking.
- Sentiment & Intent Analysis: Figures out message sentiment and user intent (information, help, purchase, etc.).
- Persona Expansion: Determines chatbot "personality" (expert, helper, teacher, etc.).
- Prompt Generation (Unified): `generate_system_prompt(context, mode)` with `minimal`/`compact`/`full` modes, plus `generate_merged_system_prompt` for concise comprehensive prompts.
- Greeting Selection: Picks greetings matching the tone.
- Response Evaluation: Scores and refines model responses.

**Why is it important?**

- Ensures consistency and avoids code duplication.
- All other scripts import and use these functions.

---

## 2. FAQ Extraction (`app2.py`)

**Purpose:** Crawls websites, extracts FAQ Q&A pairs, analyzes each for domain/tone/intent/persona, and generates fine-tuning data.

**Step-by-Step Process:**

1. User provides a website URL and crawl options.
2. Crawls site, finds FAQ sections using smart detection (headings, accordions, keywords, etc.).
3. Extracts and cleans Q&A pairs, deduplicates data.
4. Detects context, tone, intent, persona for each Q&A using shared_config.py.
5. Generates a unique system prompt for each FAQ using `advanced_system_prompt_generator(...)` (for dataset creation). For runtime, scripts use `generate_system_prompt(..., mode="compact")`.
6. Saves processed FAQs as JSONL in `FineTuning/` for model training.

**Why is this useful?**

- Automates FAQ collection and prepares high-quality training data for AI.

---

## 3. Model Training & Validation (`check_and_tune.py`)

**Purpose:** Fine-tunes OpenAI models with FAQ data and provides an interactive chatbot for validation/testing.

**Step-by-Step Process:**

1. Selects and validates FAQ training data (JSONL format).
2. Uploads data to OpenAI, starts fine-tuning job.
3. Monitors training, saves resulting model ID in `FineTuning/*.txt`.
4. Opens interactive chatbot for validation:
   - Analyzes user queries for context, tone, intent, persona, etc.
   - Builds composite system prompt using `generate_system_prompt(domain=..., tone=..., intent=..., mode)`.
   - Calls fine-tuned model, returns response with analysis metadata.
5. Logs chat turns and analysis for review.

**Why is this useful?**

- Lets you create and test custom chatbots before deployment.

---

## 4. Multi-Model Chatbot (`multimodel.py`)

**Purpose:** Runs an ensemble chatbot, querying several fine-tuned models per turn and selecting the best response.

**Step-by-Step Process:**

1. Loads available model IDs from `FineTuning/*.txt`.
2. Accepts user queries in an interactive chat loop.
3. Sends query to multiple models in parallel.
4. Scores responses for relevance and quality.
5. Returns best answer (optionally shows all candidates).
6. Supports chat commands for model control and logging (/models, /use, /k, /all, /both, /style, /strict, /exit).
7. Logs all chat turns and scoring in `analysis/chatbot_logs.jsonl`.

**Why is this useful?**

- Combines strengths of multiple models for robust, high-quality answers.
- Adapts to domain, tone, and intent; can incorporate richer context when available.
- Can enforce a strict, training-grounded mode for conservative, dataset-faithful responses.

---

## 5. Streamlit Chat UI (`UI_Chatbot.py`)

**Purpose:** Deliver a streamlined, browser-based interface for the ensemble chatbot with configuration helpers and diagnostics.

**Key Highlights:**

- Launch via `streamlit run UI_Chatbot.py` after setting the `OPENAI_API_KEY`.
- Sidebar control panel groups model selection, conversation style, and logging options.
- Multiselect registry entries or auto-pick models; a "Suggest a model" button seeds quick tests.
- Toggle strict grounding, auto answer style, tone-aware greetings, and platform/touchpoint hints.
- Expanders surface generated system prompts, composite prompt previews, quality metrics, and candidate score tables.
- Optional logging appends JSONL records to `analysis/`; chat session can be downloaded from the UI.

---

## 6. How the Whole Pipeline Works

1. Extract FAQs from a website using `app2.py`.
2. Fine-tune models with `check_and_tune.py`.
3. Validate and test models interactively.
4. Run multi-model chat support with `multimodel.py`.
5. All scripts use `shared_config.py` for analysis and prompt logic.

---

## 7. Key Concepts Explained

- **FAQ Extraction:** Automated collection of Q&A pairs from websites.
- **Context/Tone/Intent/Persona:** Detects what help is needed, how to talk, what the user wants, and the AI's "role".
- **System Prompt:** Instructions for the AI before answering.
- **Fine-Tuning:** Training the AI with your own data.
- **Ensemble Chatbot:** Using several models together for the best answer.
- **Industry/Platform/Touchpoint:** Advanced adaptation for business context and user journey.

---

## 8. Data Flow

- Input: Website URL → FAQ Extraction → Cleaned Q&A Data (JSONL)
- Processing: Data Analysis → Context/Tone/Intent/Persona/Industry/Platform/Touchpoint Tagging → System Prompt Generation
- Training: FAQ Data → OpenAI Model Training → Model ID Saved
- Chatting: User Query → Model(s) → Response(s) → Scoring → Best Answer Shown
- Logging: All steps and chat turns are logged for review.

---

## 9. Why Is This Powerful?

- Build chatbots for any domain (support, sales, technical, booking, etc.)
- Flexible: Add new topics, tones, personas, industries, platforms, touchpoints easily.
- Scalable: Works for small or large sites.
- Explainable: Every answer is traceable to its logic and analysis.

---

## 10. Example Usage

1. Run `app2.py` to collect FAQs from your site.
2. Use `check_and_tune.py` to train a chatbot with those FAQs.
3. Test the chatbot interactively.
4. Use `multimodel.py` to compare models and always give users the best answer.

---
