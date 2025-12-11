"""Streamlit-based Chat UI for Fine-Tuned FAQ Assistant.

Features:
 - Model registry loading (FineTuning/*.txt)
 - Multi-model ensemble invocation (optional)
 - Strict vs relaxed mode toggle
 - Answer style selection (short|medium|long)
 - Chat-style interface using st.chat_message and st.chat_input
 - System prompt preview (optional)
 - Candidate score table (optional)

Run:
  streamlit run UI.py

Requires environment variable OPENAI_API_KEY set.
"""

import os
import logging
import json
import streamlit as st
from datetime import datetime

from chatbot import (
    load_model_registry,
    pick_models_for_ensemble,
    ensemble_ask,
    validate_fine_tuned_model,
    ENSEMBLE_MAX_MODELS,
)

from shared import advanced_system_prompt_generator  # For preview only

logging.getLogger().setLevel(logging.WARNING)

st.set_page_config(page_title="FAQ Fine-Tuned Chat", page_icon="💬", layout="wide")
st.title("💬 Fine-Tuned FAQ Chat Assistant")

# --- Logging helpers ---
def _get_log_path(filename: str) -> str:
    base_dir = os.path.dirname(__file__)
    analysis_dir = os.path.join(base_dir, "analysis")
    try:
        os.makedirs(analysis_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(analysis_dir, filename or "chat_ui_logs.jsonl")

def _append_jsonl(path: str, obj: dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# --- Sidebar configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set in environment.")
    model_registry = load_model_registry(folder="FineTuning")
    if not model_registry:
        st.warning("No fine-tuned model IDs found in FineTuning/*.txt")
    all_models = list(sorted(model_registry.keys()))
    selected_models = st.multiselect(
        "Select models (empty = auto pick)", all_models, default=all_models[:1]
    )
    use_all = st.checkbox("Use ALL models this turn", value=False)
    max_models = st.slider(
        "Max models per turn",
        1,
        max(1, len(all_models)) if all_models else 1,
        ENSEMBLE_MAX_MODELS,
    )
    if use_all:
        max_models = 0  # signal 'all'
    strict_mode = st.checkbox("Strict Mode (training-grounded)", value=False)
    answer_style = st.selectbox("Answer Style", ["short", "medium", "long"], index=1)
    age_group = st.selectbox("Age Group", ["millennial", "genz", "boomer"], index=0)
    show_system_prompt = st.checkbox("Show Generated System Prompt", value=False)
    persona_limit = st.number_input("Persona limit (preview)", min_value=1, max_value=3, value=1)
    topic_limit = st.number_input("Topic limit (preview)", min_value=1, max_value=3, value=1)
    save_logs = st.checkbox("Save chat logs to file (JSONL)", value=True)
    log_filename = st.text_input("Log file name", value="chat_ui_logs.jsonl")
    clear_chat = st.button("Clear chat")

if clear_chat or "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]

# Hard stop early if key or models missing to avoid runtime errors
if not os.getenv("OPENAI_API_KEY"):
    st.info("Set OPENAI_API_KEY and refresh to start chatting.")
    st.stop()

if not model_registry:
    st.info("Add fine-tuned model IDs under FineTuning/*.txt and refresh.")
    st.stop()

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=("🧑" if msg["role"] == "user" else "🤖")):
        st.markdown(msg["content"])

# Chat input (bottom)
user_prompt = st.chat_input("Type your question")

if user_prompt:
    # Echo user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)

    try:
        # Choose models
        with st.spinner("Assistant is typing…"):
            chosen = pick_models_for_ensemble(
                model_registry,
                forced_keys=selected_models if selected_models else None,
                max_models=max_models,
            )
        if not chosen:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("No models selected or available.")
            st.session_state.messages.append({"role": "assistant", "content": "No models selected or available."})
        else:
            if len(chosen) == 1:
                # Single model path
                key, model_id = chosen[0]
                with st.spinner("Assistant is typing…"):
                    result = validate_fine_tuned_model(
                        model_id,
                        user_prompt,
                        platform="corporate_website",
                        touchpoint="support_page",
                        age_group=age_group,
                        answer_style=answer_style,
                        strict_mode=strict_mode,
                    )
                assistant_text = result.get("response", "")
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(assistant_text)
                    # Compact context tags
                    a = result.get("analysis", {})
                    if a:
                        tags = [
                            f"Domain: `{a.get('domain','')}`",
                            f"Intent: `{a.get('intent','')}`",
                            f"Tone: `{a.get('tone','')}`",
                        ]
                        # Optional extras if present
                        if a.get("platform"):
                            tags.append(f"Platform: `{a.get('platform')}`")
                        if a.get("touchpoint"):
                            tags.append(f"Touchpoint: `{a.get('touchpoint')}`")
                        st.markdown(" • ".join(tags))
                    if show_system_prompt:
                        with st.expander("System Prompt", expanded=False):
                            sys_preview = advanced_system_prompt_generator(
                                user_prompt,
                                "",
                                context=a.get("domain"),
                                persona_limit=persona_limit,
                                topic_limit=topic_limit,
                            )
                            st.code(sys_preview)
                    with st.expander("Analysis Metadata", expanded=False):
                        st.json(a)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                # Persist log (single model) - minimal fields
                if save_logs:
                    _append_jsonl(
                        _get_log_path(log_filename),
                        {"user_prompt": user_prompt, "response": assistant_text},
                    )
            else:
                # Ensemble path
                with st.spinner("Assistant is typing…"):
                    best, candidates = ensemble_ask(
                        chosen,
                        user_prompt,
                        platform="corporate_website",
                        touchpoint="support_page",
                        answer_style=answer_style,
                        strict_mode=strict_mode,
                    )
                assistant_text = best.get("response", "")
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(assistant_text)
                    # Compact context tags
                    a = best.get("analysis", {})
                    if a:
                        tags = [
                            f"Domain: `{a.get('domain','')}`",
                            f"Intent: `{a.get('intent','')}`",
                            f"Tone: `{a.get('tone','')}`",
                        ]
                        if a.get("platform"):
                            tags.append(f"Platform: `{a.get('platform')}`")
                        if a.get("touchpoint"):
                            tags.append(f"Touchpoint: `{a.get('touchpoint')}`")
                        st.markdown(" • ".join(tags))
                    if show_system_prompt:
                        with st.expander("System Prompt", expanded=False):
                            st.code(best.get("system_prompt", ""))
                    with st.expander("All Candidate Scores", expanded=False):
                        table = [
                            {
                                "model_key": c[0],
                                "model_id": c[1],
                                "score": c[3],
                                "excerpt": c[2].get("response", "")[:160],
                            }
                            for c in candidates
                        ]
                        st.dataframe(table)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                # Persist log (ensemble) - minimal fields
                if save_logs:
                    _append_jsonl(
                        _get_log_path(log_filename),
                        {"user_prompt": user_prompt, "response": assistant_text},
                    )
    except Exception as e:
        err = f"Error: {e}"
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(err)
        st.session_state.messages.append({"role": "assistant", "content": err})

# Export chat as JSON
with st.sidebar:
    if st.session_state.messages:
        export_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download chat JSON",
            data=export_json.encode("utf-8"),
            file_name="chat_history.json",
            mime="application/json",
        )

footer_note = "Session history is in-memory. Use 'Clear chat' to reset."
if 'save_logs' in locals() and save_logs:
    footer_note += f" Logs are being appended to analysis/{log_filename}."
st.caption(footer_note)

