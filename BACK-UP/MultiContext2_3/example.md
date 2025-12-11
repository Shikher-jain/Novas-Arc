
# Multi Context Prompting - Example Scenario (with Style & Strict Modes)

Suppose a user enters the following prompt:

```text
What are the best hotels in New York for a family vacation?
```

## Step-by-step

### Domain, Tone, Intent Detection

- **Domain:** "hospitality" (detected from keywords like "hotel", "vacation")
- **Tone:** "welcoming" (from INDUSTRY_DOMAINS or platform context)
- **Intent:** "exploration" (user is looking for options)

### Composite System Prompt Generation

We now use the unified prompt API in `shared_config.py`:

- `generate_system_prompt(domain=..., tone=..., intent=..., mode)` for runtime steering
- `advanced_system_prompt_generator(question, answer, context)` for dataset creation (unique per FAQ)

For this example at runtime:

- **Domain:** Hospitality expert, focus on comfort and care
- **Tone:** Welcoming, warm language
- **Intent:** Present options and possibilities

---

## Final System Prompt Example (mode="compact")

You are an adaptive, multi-domain customer support assistant. Your goal is to provide accurate, complete, and context-aware responses in a single turn.

Domain: Hospitality expert — speak the guest's language: comfort, convenience, and care.
Intent: Present options and possibilities. Help the user discover what's available.
Tone: Warm and inviting.
Persona: helpful assistant
Context: (no context summary)

Rules: Understand the query; answer accurately and completely; use domain terms; match user tone; anticipate follow-ups; structure clearly (bullets/steps/tables); state uncertainty when needed; self-check for accuracy and relevance.

Output as if written by a domain expert, providing immediate, actionable value in a single response.

### Style Control (`/style`)

At runtime you can request different lengths:

- `/style short`  → ~100–150 words (concise bullets)
- `/style medium` → ~200–300 words (default, structured paragraphs)
- `/style long`   → ~350–500 words (comprehensive, sections or tables if helpful)

Internally this injects a length/structure instruction into the system message without repeating the user question.

### Strict Mode (`/strict`)

Toggle with `/strict` (or explicitly `/strict on` / `/strict off`). When enabled:

- Temperature forced to 0; single deterministic sample
- The assistant must only use information plausibly present in fine‑tuning data
- Unknown / out‑of‑scope queries → respond with: `Not available in training data.`
- Prevents speculative expansions and imaginative examples

Use cases: benchmarking training coverage, auditing hallucinations, creating a conservative Q/A layer.

---

## Model Call

This prompt is sent as the "system" message in the OpenAI API call, along with the user's query. For longer, fully structured prompts, use `mode="full"`; for the smallest footprint, use `mode="minimal"`.

---

## Summary

The final prompt is a concise, context-aware instruction set tailored to the user's domain, tone, and intent, ensuring the model responds as a domain expert. Style and strict toggles adjust generation length and grounding. For training datasets, generate unique per-FAQ prompts with `advanced_system_prompt_generator(question, answer, context)`.
