
"""
Modern Streamlit FAQ Chat UI for Fine-Tuned Assistant

Features:
- Sidebar with icons, sections, tooltips, and auto/manual controls
- Responsive chat area with avatars, color highlights, and energy/tone emojis
- Model selection (manual or auto)
- Customizable greeting, answer style, persona, platform/touchpoint (manual or auto)
- Download chat history (JSON)
- Error/status indicators
- Light/dark mode support

Run:
    streamlit run UI_Chatbot.py

Requires environment variable OPENAI_API_KEY set.
"""


import streamlit as st
import os, json, random, logging
from multimodel import (
    load_model_registry, pick_models_for_ensemble, ensemble_ask, validate_fine_tuned_model,
    ENSEMBLE_MAX_MODELS, detect_industry_domain, detect_platform_context, detect_touchpoint_context, score_response
)
from shared_config import (
    advanced_system_prompt_generator, evaluate_response, prepend_tone_greeting, generate_system_prompt
)
logging.getLogger().setLevel(logging.WARNING)

# --- Modern UI Setup ---
st.set_page_config(page_title="FAQ Chat Assistant", page_icon="💬", layout="wide")

# --- Sidebar ---
api_key = os.getenv("OPENAI_API_KEY")
model_registry = load_model_registry(folder="FineTuning")
all_models = sorted(model_registry.keys())

with st.sidebar:
    st.image("https://img.icons8.com/color/96/faq.png", width=64)
    st.title("FAQ Chat Assistant")
    st.markdown("---")
    st.markdown("### Model Selection")
    auto_model = st.checkbox("Auto-select model(s)", value=True)
    selected_models = st.multiselect("Models", all_models, help="Select models manually if auto is off", disabled=auto_model)
    max_models = st.slider("Max models per response", 1, max(1, len(all_models)), min(2, len(all_models)), disabled=auto_model)
    st.markdown("---")
    st.markdown("### Conversation Settings")
    answer_style = st.selectbox("Answer length", ["auto", "short", "medium", "long"], index=0)
    greeting_style = st.selectbox("Greeting style", ["auto", "formal", "informal"], index=0)
    persona_options = [
        "auto",
        "default",
        "Gen Z",
        "Millennial Professional",
        "Boomer / Traditional",
        "Corporate / Brand Voice",
        "Tech Expert",
        "Customer Success Rep",
        "Educator / Trainer",
        "Playful / Friendly",
        "Minimalist / Direct"
    ]
    persona_descriptions = {
        "Gen Z": "Fun, emoji-heavy, conversational",
        "Millennial Professional": "Friendly but polished",
        "Boomer / Traditional": "Formal, respectful, structured",
        "Corporate / Brand Voice": "Neutral, concise, consistent",
        "Tech Expert": "Analytical, efficient",
        "Customer Success Rep": "Empathetic and helpful",
        "Educator / Trainer": "Explanatory, encouraging",
        "Playful / Friendly": "High energy, positive",
        "Minimalist / Direct": "Ultra-concise, efficient"
    }
    age_group = st.selectbox(
        "Persona",
        persona_options,
        index=0,
        help="Select the response style persona. " + ", ".join([f"{k}: {v}" for k, v in persona_descriptions.items()])
    )
    auto_platform = st.checkbox("Auto-detect platform/touchpoint", value=True)
    platform = st.selectbox("Platform", ["auto", "corporate_website", "support_chat", "ecommerce", "healthcare_portal", "technical_docs", "instagram", "gaming_app", "college_website"], index=0, disabled=auto_platform)
    touchpoint = st.selectbox("Touchpoint", ["auto", "support_page", "home_page", "product_page", "checkout", "faq_section", "feedback_form"], index=0, disabled=auto_platform)
    strict_mode = st.checkbox("Strict dataset grounding", value=False)
    auto_answer_style = st.checkbox("Auto-adjust answer style by energy", value=True)
    mirror_energy = st.checkbox("Mirror user energy", value=True)
    add_tone_greeting = st.checkbox("Add tone-aware greeting", value=True)
    st.markdown("---")
    save_logs = st.checkbox("Save chat logs", value=True)
    log_filename = st.text_input("Log file name", value="chat_ui_logs.jsonl", disabled=not save_logs)
    if st.button("Clear chat"):
        st.session_state.clear()
        st.experimental_rerun()

# --- Main Area ---
st.markdown("## 👋 Welcome to the FAQ Chat Assistant!")
st.markdown("Type your question below and get instant, tone-aware answers from your fine-tuned models.")


# --- Sidebar layout helper ---
def _render_sidebar(api_key: str, model_registry: dict):
    """Render the sidebar UI and return collected configuration values."""
    registry = model_registry or {}
    all_models = list(sorted(registry.keys()))

    # --- Suggestion callback ---
    def suggest_model():
        suggestion = random.choice(all_models)
        st.session_state["sidebar_selected_models"] = [suggestion]
        st.session_state["suggested_model"] = suggestion

    # Only set sidebar_selected_models before widget is created
    if "sidebar_selected_models" not in st.session_state:
        st.session_state["sidebar_selected_models"] = all_models[:1] if all_models else []
    # Do NOT assign to st.session_state["sidebar_selected_models"] after widget is created

    with st.sidebar:
        st.markdown("### Session & API")
        if "last_turn_summary" not in st.session_state:
            st.caption("Start chatting to unlock configuration helpers.")

        if api_key:
            st.success("OPENAI_API_KEY detected.")
        else:
            st.error("OPENAI_API_KEY not set in environment.")

        st.markdown("### Model Selection")
        with st.expander("Model selection", expanded=True):
            if not all_models:
                st.warning("No fine-tuned model IDs found in FineTuning/*.txt")
            else:
                st.info(f"Loaded {len(all_models)} fine-tuned model{'s' if len(all_models) != 1 else ''}.")

            # Suggest button BEFORE multiselect
            suggest_holder = st.empty()
            st.button(
                "Suggest a model",
                disabled=not all_models,
                on_click=suggest_model,
                key="suggest_model_btn_1"
            )
            if st.session_state.get("suggested_model"):
                suggest_holder.info(
                    f"Previous suggestion: `{st.session_state['suggested_model']}`"
                )

            # Only one multiselect for sidebar_selected_models should exist
            use_all = st.checkbox(
                "Use every model this turn",
                value=False,
                help="Override ensemble cap and ping all loaded models.",
                disabled=not all_models,
            )
            slider_cap = max(1, len(all_models)) if all_models else 1
            default_cap = min(ENSEMBLE_MAX_MODELS, slider_cap)
            max_models = st.slider(
                "Max models per response",
                1,
                slider_cap,
                default_cap,
                disabled=use_all or not all_models,
                help="Upper bound for ensemble size when auto-select is used.",
            )

            # suggest_holder = st.empty()
            # if st.button("Suggest a model", disabled=not all_models, key="suggest_model_btn_2"):
            #     suggestion = random.choice(all_models)
            #     st.session_state["sidebar_selected_models"] = [suggestion]
            #     st.session_state["suggested_model"] = suggestion
            #     suggest_holder.success(
            #         f"Selected `{suggestion}` for you. Adjust the multiselect above to add more."
            #     )
            # elif st.session_state.get("suggested_model"):
            #     suggest_holder.info(
            #         f"Previous suggestion: `{st.session_state['suggested_model']}`"
            #     )

        st.markdown("### Conversation Style")
        with st.expander("Conversation style", expanded=True):
            strict_mode = st.checkbox(
                "Strict dataset grounding",
                value=False,
                help="Keep answers tightly aligned with fine-tuning data. Example: 'What is the refund policy?' will only use info from the dataset.",
                key="sidebar_strict_mode"
            )
            answer_style = st.selectbox(
                "Answer length",
                ["default (auto)", "short", "medium", "long"],
                index=0,
                help="Choose a length or leave as 'default (auto)' to let the system decide based on your energy.",
                key="sidebar_answer_style"
            )
            age_group = st.selectbox(
                "Audience persona",
                ["millennial", "genz", "boomer"],
                index=0,
                help="Shape tone and references for the target audience. Example: 'genz' will use more casual language.",
                key="sidebar_age_group"
            )
            greeting_style = st.selectbox(
                "Greeting style",
                value=True,
                help="Infer context from each incoming user prompt. Example: 'I can't checkout' will set platform to 'ecommerce' and touchpoint to 'checkout'.",
                key="sidebar_auto_platform_tp"
            )
            platform = st.selectbox(
                "Platform",
                [
                    "corporate_website",
                    "support_chat",
                    "ecommerce",
                    "healthcare_portal",
                    "technical_docs",
                    "instagram",
                    "gaming_app",
                    "college_website",
                ],
                index=0,
                disabled=auto_platform,
                help="Where is the user interacting? Example: 'ecommerce' for online shopping.",
                key="sidebar_platform"
            )
            touchpoint = st.selectbox(
                "Touchpoint",
                [
                    "support_page",
                    "home_page",
                    "product_page",
                    "checkout",
                    "faq_section",
                    "feedback_form",
                ],
                index=0,
                disabled=auto_platform,
                help="Which part of the platform? Example: 'checkout' for payment issues.",
                key="sidebar_touchpoint"
            )
            auto_answer_style = st.checkbox(
                "Auto-adjust answer style by energy",
                value=True,
                help="Use detected user energy to pick response length automatically. Example: 'I'm super excited!' will get a longer answer.",
                key="sidebar_auto_answer_style"
            )
            mirror_energy = st.checkbox(
                "Mirror user energy",
                value=True,
                help="Make replies match the user's excitement and pacing. Example: If you use lots of exclamation marks, the assistant will too!",
                key="sidebar_mirror_energy"
            )
            add_tone_greeting = st.checkbox(
                "Add tone-aware greeting",
                value=False,
                help="Front-load responses with a greeting that matches detected tone. Example: 'Hey there!' for informal, 'Good afternoon.' for formal.",
                key="sidebar_add_tone_greeting"
            )

        with st.expander("Prompts & logging", expanded=False):
            show_system_prompt = st.checkbox(
                "Show generated system prompt",
                value=False,
                help="Reveal the internal system prompt used for each reply.",
                key="sidebar_show_system_prompt"
            )
            show_composite_prompt = st.checkbox(
                "Show composite system prompt",
                value=False,
                help="Preview the composite prompt derived from detected context.",
                key="sidebar_show_composite_prompt"
            )
            save_logs = st.checkbox(
                "Save chat logs to file (JSONL)",
                value=True,
                help="Append each turn to analysis/<file name>.",
                key="sidebar_save_logs"
            )
            log_filename = st.text_input(
                "Log file name",
                value="chat_ui_logs.jsonl",
                help="Relative to the analysis/ directory.",
                disabled=not save_logs,
                key="sidebar_log_filename"
            )

    clear_chat = st.button("Clear chat", help="Reset conversation history.", key="sidebar_clear_chat")

    return {
        "selected_models": list(st.session_state.get("sidebar_selected_models", [])),
        "max_models": 0 if use_all else max_models,
        "strict_mode": strict_mode,
        "answer_style": answer_style,
        "age_group": age_group,
        "greeting_style": greeting_style,
    "auto_platform_tp": auto_platform,
        "platform": platform,
        "touchpoint": touchpoint,
        "auto_answer_style": auto_answer_style,
        "mirror_energy": mirror_energy,
        "add_tone_greeting": add_tone_greeting,
        "show_system_prompt": show_system_prompt,
        "show_composite_prompt": show_composite_prompt,
        "save_logs": save_logs,
        "log_filename": log_filename,
        "clear_chat": clear_chat,
    }


# --- Local helpers for adaptive behavior ---

def detect_energy(text):
    if not text: return "medium", 0
    score = text.count("!") + sum(w in text.lower() for w in ["great","awesome","excited","amazing","urgent"])
    if score >= 2: return "high", score
    if score == 1: return "medium", score
    return "low", score


def guess_platform(text):
    tl = (text or "").lower()
    if any(k in tl for k in ["buy", "cart", "discount", "sale", "checkout"]): return "ecommerce"
    if any(k in tl for k in ["error", "issue", "help", "support", "not working", "fix"]): return "support_chat"
    if any(k in tl for k in ["api", "docs", "developer", "example", "snippet"]): return "technical_docs"
    if any(k in tl for k in ["instagram", "social", "reel", "story"]): return "instagram"
    return "corporate_website"

def guess_touchpoint(text):
    tl = (text or "").lower()
    if any(k in tl for k in ["price", "pricing", "feature", "compare", "details"]): return "product_page"
    if any(k in tl for k in ["book", "order", "subscribe", "checkout", "reserve"]): return "checkout"
    if any(k in tl for k in ["faq", "how to", "steps", "guide"]): return "faq_section"
    if any(k in tl for k in ["feedback", "complaint", "suggest", "improve"]): return "feedback_form"
    if any(k in tl for k in ["what", "options", "browse", "show", "tell"]): return "home_page"
    if any(k in tl for k in ["error", "issue", "help", "support", "not working", "fix"]): return "support_page"
    return "support_page"


def log_chat(filename, obj):
    try:
        os.makedirs("analysis", exist_ok=True)
        path = os.path.join("analysis", filename or "chat_ui_logs.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# --- Turn helpers to reduce repetition ---

def compute_effective_controls(user_prompt):
    energy_level, energy_score = detect_energy(user_prompt)
    from shared_config import persona_expansion
    # Auto/manual/default logic for all parameters
    # Platform
    if auto_platform or platform == "auto":
        eff_platform = guess_platform(user_prompt)
    elif platform == "default":
        eff_platform = "corporate_website"
    else:
        eff_platform = platform

    # Touchpoint
    if auto_platform or touchpoint == "auto":
        eff_touchpoint = guess_touchpoint(user_prompt)
    elif touchpoint == "default":
        eff_touchpoint = "support_page"
    else:
        eff_touchpoint = touchpoint

    # Answer style
    if auto_answer_style or answer_style == "auto":
        eff_answer_style = "long" if energy_level == "high" else "medium" if energy_level == "medium" else "short"
    elif answer_style == "default":
        eff_answer_style = "medium"
    else:
        eff_answer_style = answer_style

    # Persona
    if age_group == "auto":
        keywords = [w for w in user_prompt.lower().split() if len(w) > 2]
        personas = persona_expansion(keywords)
        eff_age_group = personas[0] if personas else "Millennial Professional"
    elif age_group == "default":
        eff_age_group = "Corporate / Brand Voice"
    else:
        eff_age_group = age_group

    # Greeting style
    if greeting_style == "auto":
        eff_greeting_style = "informal" if energy_level == "high" else "formal"
    elif greeting_style == "default":
        eff_greeting_style = "formal"
    else:
        eff_greeting_style = greeting_style

    return {
        "energy_level": energy_level,
        "energy_score": energy_score,
        "platform": eff_platform,
        "touchpoint": eff_touchpoint,
        "answer_style": eff_answer_style,
        "age_group": eff_age_group,
        "greeting_style": eff_greeting_style,
    }


def run_single_model(model_id, effective):
    result = validate_fine_tuned_model(
        model_id,
        effective["user_prompt"],
        platform=effective["platform"],
        touchpoint=effective["touchpoint"],
        age_group=effective["age_group"],
        answer_style=effective["answer_style"],
        strict_mode=strict_mode,
        energy_level=effective["energy_level"],
    )
    return result.get("response", ""), result.get("analysis", {}), result.get("system_prompt", "")

def run_ensemble(models, effective):
    best, candidates = ensemble_ask(
        models,
        effective["user_prompt"],
        platform=effective["platform"],
        touchpoint=effective["touchpoint"],
        answer_style=effective["answer_style"],
        strict_mode=strict_mode,
        energy_level=effective["energy_level"],
    )
    return best.get("response", ""), best.get("analysis", {}), best.get("system_prompt", ""), candidates


if "messages" not in st.session_state:
    st.session_state["messages"] = []



# --- Render chat history ---
for m in st.session_state["messages"]:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])

# --- Chat input ---
user_prompt = st.chat_input("Type your question...")

def run_chat_flow(user_prompt, chosen_models):
    effective = compute_effective_controls(user_prompt)
    effective["user_prompt"] = user_prompt
    try:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        if not model_registry:
            raise RuntimeError("No fine-tuned models found.")
        if not chosen_models:
            response = "⚠️ No models selected."
            analysis = {}
            system_used = ""
            candidates = None
        elif len(chosen_models) == 1:
            _, model_id = chosen_models[0]
            response, analysis, system_used = run_single_model(model_id, effective)
            candidates = None
        else:
            response, analysis, system_used, candidates = run_ensemble(chosen_models, effective)
        # Tone-aware greeting
        if add_tone_greeting:
            response = prepend_tone_greeting(
                response,
                analysis.get("tone", "neutral"),
                energy=effective["energy_level"],
                age_group=effective["age_group"],
                platform=effective["platform"],
                user_prompt=user_prompt,
                mimic_energy=mirror_energy,
            )
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        if save_logs:
            log_chat(log_filename, {"user_prompt": user_prompt, "response": response})
    except Exception as e:
        err = f"Error: {e}"
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(err)
        st.session_state["messages"].append({"role": "assistant", "content": err})

if user_prompt:
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)
    with st.spinner("Assistant is typing…"):
        if auto_model:
            chosen = pick_models_for_ensemble(model_registry, None, max_models)
        else:
            chosen = pick_models_for_ensemble(model_registry, selected_models or None, max_models)
        run_chat_flow(user_prompt, chosen)


# --- Export chat as JSON ---
with st.sidebar:
    if st.session_state["messages"]:
        export_json = json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2)
        st.download_button(
            label="Download chat JSON",
            data=export_json.encode("utf-8"),
            file_name="chat_history.json",
            mime="application/json",
        )
footer_note = "Session history is in-memory. Use 'Clear chat' to reset."
if save_logs:
    footer_note += f" Logs are being appended to analysis/{log_filename}."
st.caption(footer_note)

