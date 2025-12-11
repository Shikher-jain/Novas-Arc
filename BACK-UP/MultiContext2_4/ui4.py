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
# from streamlit_extras.let_it_rain import rain
from multimodel import (
    load_model_registry, pick_models_for_ensemble, ensemble_ask, validate_fine_tuned_model,
    ENSEMBLE_MAX_MODELS)
from shared_config import prepend_tone_greeting
logging.getLogger().setLevel(logging.WARNING)

# --- Modern UI Setup ---
st.set_page_config(page_title="FAQ Chat Assistant", page_icon="💬", layout="wide")

def safe_rerun():
    """Try to trigger a Streamlit rerun in a couple of compatible ways.

    Some Streamlit versions removed `st.experimental_rerun`. We attempt that
    first, then raise the internal RerunException if available. As a last-ditch
    fallback we set a session flag and call `st.stop()` so the UI halts and the
    next user interaction will re-render with the cleared state.
    """
    try:
        # preferred API (older/newer Streamlit)
        st.experimental_rerun()
        return
    except Exception:
        pass

    try:
        # Internal exception used by Streamlit to trigger reruns
        from streamlit.runtime.scriptrunner.script_runner import RerunException

        raise RerunException()
    except Exception:
        # Fallback: mark a flag and stop the script; next interaction will redraw UI
        st.session_state["_needs_rerun"] = True
        st.stop()

# --- Sidebar scrollbar styling (wider, nicer thumb) ---
# Inject CSS to increase the sidebar scrollbar width and style it for WebKit
# browsers and provide a Firefox fallback using `scrollbar-width`.
st.markdown(
    '''
    <style>
    /* WebKit-based browsers (Chrome, Edge, Safari) */
    section[data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 2px;
        height: 2px;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
        background: transparent;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background-color: rgba(0,0,0,0.35);
        border-radius: 10px;
        border: 3px solid transparent;
        background-clip: padding-box;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
        background-color: rgba(0,0,0,0.55);
    }

    /* Firefox */
    section[data-testid="stSidebar"] {
        scrollbar-width: auto; /* options: auto, thin, none */
        scrollbar-color: rgba(0,0,0,0.35) transparent;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)


api_key = os.getenv("OPENAI_API_KEY")
model_registry = load_model_registry(folder="FineTuning")
all_models = sorted(model_registry.keys())

# --- State Initialization Guards ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "nickname" not in st.session_state:
    st.session_state["nickname"] = ""
if "sidebar_selected_models" not in st.session_state:
    st.session_state["sidebar_selected_models"] = all_models[:1] if all_models else []
if "suggested_model" not in st.session_state:
    st.session_state["suggested_model"] = None
if "assistant_msg_counter" not in st.session_state:
    st.session_state["assistant_msg_counter"] = 0
if "msg_id_seq" not in st.session_state:
    st.session_state["msg_id_seq"] = 0
# ensure saved logging prefs exist
if "save_logs" not in st.session_state:
    st.session_state["save_logs"] = True
if "log_filename" not in st.session_state:
    st.session_state["log_filename"] = "chat_ui_logs.jsonl"

# If a previous safe_rerun() set a flag, perform a deterministic reset now
# at script start so we don't remove Streamlit-internal state mid-run.
if st.session_state.get("_needs_rerun"):
    # Clear the flag first to avoid loops
    st.session_state["_needs_rerun"] = False
    # Reset chat messages and internal counters only (preserve user inputs like nickname
    # and manual model selection). This ensures Clear chat won't wipe the title/input box.
    st.session_state["messages"] = []
    st.session_state["suggested_model"] = None
    st.session_state["assistant_msg_counter"] = 0
    st.session_state["msg_id_seq"] = 0
    # Try to trigger a fresh rerun so UI picks up the cleared state immediately.
    try:
        st.experimental_rerun()
    except Exception:
        # If experimental_rerun isn't available, continue — the cleared state
        # will be visible on the next interaction or render.
        pass

# --- Consolidated Sidebar ---
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/faq.png", width=64)
        st.title("FAQ Chat Assistant 🧑‍💻")
        st.markdown("---")
        with st.expander("🧠 Model Selection", expanded=True):
            auto_model = st.checkbox("🤖 Auto-select model(s)", value=True, help="Let the assistant pick the best model(s) for you.")
            # ensure multiselect shows current chosen models from session state
            selected_models = st.multiselect("🗂️ Models", all_models, default=st.session_state.get("sidebar_selected_models", []), help="Select models manually if auto is off", disabled=auto_model)
            # compute slider bounds and safe default
            max_possible = max(1, len(all_models))
            slider_default = min(ENSEMBLE_MAX_MODELS if ENSEMBLE_MAX_MODELS > 0 else 1, max_possible)
            max_models = st.slider("🔢 Max models per response", 1, max_possible, slider_default, help=f"Choose up to {ENSEMBLE_MAX_MODELS} models", disabled=auto_model)
        st.markdown("---")
        with st.expander("💬 Conversation Settings", expanded=True):
            answer_style = st.selectbox("📝 Answer length", ["auto", "short", "medium", "long"], index=0, help="Choose a length or let the system decide.")
            greeting_style = st.selectbox("👋 Greeting style", ["auto", "formal", "informal"], index=0, help="How should the assistant greet you?")
            persona_options = [
                "auto",
                "default",
                "Gen Z 🧑‍🎤",
                "Millennial Professional 🧑‍💼",
                "Boomer / Traditional 👴",
                "Corporate / Brand Voice 🏢",
                "Tech Expert 👨‍💻",
                "Customer Success Rep 🤝",
                "Educator / Trainer 🧑‍🏫",
                "Playful / Friendly 😄",
                "Minimalist / Direct ⚡"
            ]
            persona_descriptions = {
                "Gen Z 🧑‍🎤": "Fun, emoji-heavy, conversational",
                "Millennial Professional 🧑‍💼": "Friendly but polished",
                "Boomer / Traditional 👴": "Formal, respectful, structured",
                "Corporate / Brand Voice 🏢": "Neutral, concise, consistent",
                "Tech Expert 👨‍💻": "Analytical, efficient",
                "Customer Success Rep 🤝": "Empathetic and helpful",
                "Educator / Trainer 🧑‍🏫": "Explanatory, encouraging",
                "Playful / Friendly 😄": "High energy, positive",
                "Minimalist / Direct ⚡": "Ultra-concise, efficient"
            }
            age_group = st.selectbox(
                "🧑 Persona",
                persona_options,
                index=0,
                help="Select the response style persona. " + ", ".join([f"{k}: {v}" for k, v in persona_descriptions.items()])
            )
            auto_platform = st.checkbox("🛰️ Auto-detect platform/touchpoint", value=True, help="Let the assistant infer platform/touchpoint from your question.")
            platform = st.selectbox("🌐 Platform", ["auto", "corporate_website", "support_chat", "ecommerce", "healthcare_portal", "technical_docs", "instagram", "gaming_app", "college_website"], index=0, disabled=auto_platform)
            touchpoint = st.selectbox("📍 Touchpoint", ["auto", "support_page", "home_page", "product_page", "checkout", "faq_section", "feedback_form"], index=0, disabled=auto_platform)
            strict_mode = st.checkbox("🛡️ Strict dataset grounding", value=False, help="Keep answers tightly aligned with fine-tuning data.")
            auto_answer_style = st.checkbox("⚡ Auto-adjust answer style by energy", value=True, help="Use detected user energy to pick response length automatically.")
            mirror_energy = st.checkbox("🔄 Mirror user energy", value=True, help="Make replies match the user's excitement and pacing.")
            add_tone_greeting = st.checkbox("🎉 Add tone-aware greeting", value=True, help="Front-load responses with a greeting that matches detected tone.")
            # Use an explicit key so the nickname is saved to session_state['nickname']
            nickname = st.text_input("📝 Set your nickname (optional)", value=st.session_state.get("nickname", ""), key="nickname") or "User"
        st.markdown("---")
        with st.expander("🗂️ Logging & Export", expanded=False):
            save_logs = st.checkbox("💾 Save chat logs", value=st.session_state.get("save_logs", True))
            log_filename = st.text_input("🗃️ Log file name", value=st.session_state.get("log_filename", "chat_ui_logs.jsonl"), disabled=not save_logs)
            
            if st.button("Clear chat 🧹"):
                # Clear only conversation-related state. Do NOT call safe_rerun();
                # a button click already triggers a rerun. Avoiding an explicit
                # rerun helps preserve other widget states (nickname, chat input).
                st.session_state["messages"] = []
                st.session_state["suggested_model"] = None
                st.session_state["assistant_msg_counter"] = 0
                st.session_state["msg_id_seq"] = 0
        # Export chat as JSON
        if st.session_state["messages"]:
            export_json = json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2)
            st.download_button(
                label="Download chat JSON",
                data=export_json.encode("utf-8"),
                file_name="chat_history.json",
                mime="application/json",
                key=f"download_chat_json_{len(st.session_state['messages'])}"
            )

    # Save chosen sidebar values back into session_state so other code reads consistent state
    st.session_state["save_logs"] = save_logs
    st.session_state["log_filename"] = log_filename
    st.session_state["sidebar_selected_models"] = selected_models

    return dict(
        auto_model=auto_model,
        selected_models=selected_models,
        max_models=max_models,
        answer_style=answer_style,
        greeting_style=greeting_style,
        age_group=age_group,
        auto_platform=auto_platform,
        platform=platform,
        touchpoint=touchpoint,
        strict_mode=strict_mode,
        auto_answer_style=auto_answer_style,
        mirror_energy=mirror_energy,
        add_tone_greeting=add_tone_greeting,
        nickname=nickname,
        save_logs=save_logs,
        log_filename=log_filename,
    )

# --- Get sidebar values ---
sidebar_values = render_sidebar()
auto_model = sidebar_values["auto_model"]
selected_models = sidebar_values["selected_models"]
max_models = sidebar_values["max_models"]
answer_style = sidebar_values["answer_style"]
greeting_style = sidebar_values["greeting_style"]
age_group = sidebar_values["age_group"]
auto_platform = sidebar_values["auto_platform"]
platform = sidebar_values["platform"]
touchpoint = sidebar_values["touchpoint"]
strict_mode = sidebar_values["strict_mode"]
auto_answer_style = sidebar_values["auto_answer_style"]
mirror_energy = sidebar_values["mirror_energy"]
add_tone_greeting = sidebar_values["add_tone_greeting"]
nickname = sidebar_values["nickname"]
save_logs = sidebar_values["save_logs"]
log_filename = sidebar_values["log_filename"]

# --- Main Area ---
st.markdown("## 👋 Welcome to the FAQ Chat Assistant!")

all_models = ", ".join(selected_models) if selected_models else "the selected model"

st.markdown(f"Hello, **{nickname}**! Type your question **{all_models}** below and get instant, tone-aware answers from your fine-tuned models.")
# --- Local helpers for adaptive behavior ---

def detect_energy(text):
    if not text:
        return "medium", 0
    high_energy_words = [
        "great", "awesome", "excited", "amazing", "urgent", "fantastic", "incredible", "wonderful",
        "super", "love", "wow", "yay", "cool", "excellent", "outstanding", "brilliant", "terrific",
        "spectacular", "unbelievable", "so happy", "so good", "so cool", "can't wait", "thrilled",
        "pumped", "ecstatic", "delighted", "impressive", "rock", "fire", "lit", "best ever", "phenomenal"
    ]
    score = text.count("!") + sum(w in text.lower() for w in high_energy_words)
    if score >= 2:
        return "high", score
    if score == 1:
        return "medium", score
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
        # Deduplicate: read existing logs and update or append
        logs = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    key = (entry.get("user_prompt", ""), entry.get("response", ""))
                    logs[key] = entry
        # Prefer new entry if it has rating or rating_msg
        key = (obj.get("user_prompt", ""), obj.get("response", ""))
        prev = logs.get(key)
        if not prev or (obj.get("rating_msg") or obj.get("rating")):
            logs[key] = obj
        # Write all logs back
        with open(path, "w", encoding="utf-8") as f:
            for entry in logs.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error(f"Failed to write chat log to {filename}: {e}")
        try:
            st.warning(f"Could not save chat log: {e}")
        except Exception:
            pass

# --- Turn helpers to reduce repetition ---

def compute_effective_controls(user_prompt):
    energy_level, energy_score = detect_energy(user_prompt)
    from shared_config import persona_expansion
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

# --- Render chat history ---
for m in st.session_state["messages"]:
    avatar = "🧑" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        # For assistant messages show an in-place feedback widget that updates
        # the existing message dict in session state. Keys are stable per message id.
        if m["role"] == "assistant":
            # stable keys per message id
            msg_id = m.get("id")
            if msg_id is None:
                # defensive: assign one if missing
                msg_id = st.session_state["msg_id_seq"]
                m["id"] = msg_id
                st.session_state["msg_id_seq"] += 1

            rating_key = f"rating_{msg_id}"
            followup_key = f"followup_{msg_id}"

            # Initialize widget state from message if present
            if rating_key not in st.session_state and "rating" in m:
                try:
                    st.session_state[rating_key] = m["rating"]
                except Exception:
                    pass

            rating = st.radio(
                "Your feedback helps us improve!",
                ["👍 Yes, helpful", "👎 No, needs improvement"],
                index=0 if st.session_state.get(rating_key) == "👍 Yes, helpful" else 1 if st.session_state.get(rating_key) == "👎 No, needs improvement" else 0,
                horizontal=True,
                key=rating_key,
            )

            # If user says 'No', show follow-up box
            if rating == "👎 No, needs improvement":
                followup_val = st.text_input("How can we improve this answer?", value=st.session_state.get(followup_key, ""), key=followup_key)
            else:
                followup_val = st.session_state.get(followup_key, "")

            # Persist selections back into the message dict. Log one compact record
            # if either the rating or the follow-up message changes.
            prev_rating = m.get("rating")
            prev_followup = m.get("followup")
            changed = False

            if prev_rating != rating:
                m["rating"] = rating
                changed = True

            if followup_val:
                if prev_followup != followup_val:
                    m["followup"] = followup_val
                    changed = True
            else:
                # remove empty followup to keep logs clean
                if "followup" in m:
                    del m["followup"]
                    if prev_followup is not None:
                        changed = True

            if changed:
                try:
                    if save_logs:
                        log_chat(
                            log_filename,
                            {
                                "user_prompt": m.get("user_prompt", ""),
                                "response": m.get("content", ""),
                                "rating": m.get("rating"),
                                "rating_msg": m.get("followup", ""),
                            },
                        )
                except Exception:
                    # don't let logging failures break the UI
                    pass

# --- Chat input ---
# Give the chat input a stable key so its (unsubmitted) content isn't lost
# by our Clear chat flow that only clears conversation state.
user_prompt = st.chat_input("Type your question...", key="chat_input")

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
        elif len(chosen_models) == 1:
            _, model_id = chosen_models[0]
            response, analysis, system_used = run_single_model(model_id, effective)
        else:
            response, analysis, system_used, candidates = run_ensemble(chosen_models, effective)
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
        msg_id = st.session_state.get("msg_id_seq", 0)
        assistant_msg = {
            "id": msg_id,
            "role": "assistant",
            "content": response,
            "energy": effective["energy_level"],
            "analysis": analysis,
            # store the originating user prompt so feedback/logs can reference it later
            "user_prompt": effective.get("user_prompt", user_prompt),
        }
        st.session_state["messages"].append(assistant_msg)
        st.session_state["msg_id_seq"] = msg_id + 1
        if save_logs:
            log_chat(
                log_filename,
                {
                    "user_prompt": user_prompt,
                    "response": response,
                    "rating": assistant_msg.get("rating"),
                    "rating_msg": assistant_msg.get("followup"),
                },
            )
    except Exception as e:
        err = f"Error: {e}"
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(err)
        st.session_state["messages"].append({"role": "assistant", "content": err})

if user_prompt:
    user_msg_id = st.session_state.get("msg_id_seq", 0)
    user_msg = {"id": user_msg_id, "role": "user", "content": user_prompt}
    st.session_state["messages"].append(user_msg)
    st.session_state["msg_id_seq"] = user_msg_id + 1
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_prompt)
    with st.spinner("🤖 Assistant is typing…"):
        st.markdown("<span style='color:#888;'>Detecting context, selecting model, and generating answer...</span>", unsafe_allow_html=True)
        if auto_model:
            chosen = pick_models_for_ensemble(model_registry, None, max_models)
        else:
            chosen = pick_models_for_ensemble(model_registry, selected_models or None, max_models)
        run_chat_flow(user_prompt, chosen)

# (Sidebar export and footer are now handled inside `render_sidebar()`.)

