# UI_Chatbot_Simple.py
"""
Simplified Streamlit FAQ Chat UI
Run with:  streamlit run UI_Chatbot_Simple.py
"""

import streamlit as st
import os, json, random
from multimodel import (
    load_model_registry, pick_models_for_ensemble,
    validate_fine_tuned_model, ensemble_ask
)
from shared_config import prepend_tone_greeting

st.set_page_config(page_title="FAQ Chat Assistant", page_icon="💬", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/faq.png", width=64)
    st.title("FAQ Chat Assistant")
    st.markdown("---")

    model_registry = load_model_registry("FineTuning")
    all_models = sorted(model_registry.keys())

    selected_models = st.multiselect("Models", all_models, help="Select fine-tuned models to use")
    max_models = st.slider("Max models", 1, max(1, len(all_models)), 2)
    strict_mode = st.checkbox("Strict grounding", False)
    auto_answer_style = st.checkbox("Auto-adjust by energy", True)
    mirror_energy = st.checkbox("Mirror user energy", True)
    save_logs = st.checkbox("Save chat logs", True)
    log_filename = st.text_input("Log filename", "chat_ui_logs.jsonl", disabled=not save_logs)
    if st.button("Clear chat"):
        st.session_state.clear()
        st.experimental_rerun()

# --- Helper functions ---
def detect_energy(text):
    if not text: return "medium", 0
    score = text.count("!") + sum(w in text.lower() for w in ["great","awesome","excited"])
    if score >= 2: return "high", score
    if score == 1: return "medium", score
    return "low", score

def log_to_file(filename, data):
    path = os.path.join("analysis", filename)
    os.makedirs("analysis", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

# --- Chat UI ---
st.markdown("### 💬 Ask a question about your product or service")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"], avatar="🧑" if m["role"]=="user" else "🤖"):
        st.markdown(m["content"])

user_prompt = st.chat_input("Type your question here...")

if user_prompt:
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)

    energy_level, _ = detect_energy(user_prompt)
    chosen = pick_models_for_ensemble(model_registry, selected_models or None, max_models)

    if not chosen:
        response = "⚠️ No models selected."
    elif len(chosen) == 1:
        model_id = chosen[0][1]
        result = validate_fine_tuned_model(model_id, user_prompt, strict_mode=strict_mode)
        response = result.get("response", "No response.")
    else:
        best, _ = ensemble_ask(chosen, user_prompt, strict_mode=strict_mode)
        response = best.get("response", "No ensemble response.")

    # Tone-aware greeting
    response = prepend_tone_greeting(response, "neutral", energy=energy_level, user_prompt=user_prompt, mimic_energy=mirror_energy)

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})

    if save_logs:
        log_to_file(log_filename, {"user": user_prompt, "assistant": response})

# --- Export ---
if st.session_state["messages"]:
    st.download_button(
        "⬇️ Download chat JSON",
        data=json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json",
    )
