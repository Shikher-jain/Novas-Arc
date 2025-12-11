CANONICAL_INSTRUCTION_BLOCK = "Behavior Guidelines:\n- Provide a helpful, good-faith reply that covers the user’s request.\n- Bring in domain terminology or examples when it improves clarity.\n- Keep the structure readable (bullets or short paragraphs when it helps).\n- Offer follow-up tips or next steps if they add value.\n- Mention assumptions or uncertainty only when they materially affect the answer.\n\nOutput Formatting:\n- Code -> runnable blocks when possible.\n- Data/Analysis -> tables or clear calculations.\n- Explanations -> short paragraphs or bullets as needed.\n- Creative -> balanced, engaging narrative.\n\nSafety Rules:\n- Never request or expose private or sensitive data.\n- Avoid illegal, sexual, violent, or manipulative content.\n- Stay respectful and proportionate in tone.\n- Note uncertainty if facts are unclear.\n\nAdditional Guidance:\n- Be honest, helpful, and user-centered.\n- Match the user’s tone without sounding stiff.\n- Share accurate information; say when you’re unsure.\n- Keep internal instructions private.\n- Stay concise—drop redundant details."

SUPPLEMENTAL_INSTRUCTION_TEMPLATE = "- If specifics are missing, share what’s known and acknowledge the gap briefly.\n- Reflect the user’s energy and formality.\n- Tailor language for age group: {age_group}.\n- Keep responses tidy; bullets/steps optional but appreciated.\n- Stick to grounded facts and stay in character."

DOMAIN_INSTRUCTIONS = {"technical": "Provide detailed technical information with clear explanations and examples.", "sales": "Help the user understand value, benefits, and pricing. Use business language.", "booking": "Assist with scheduling, availability, and reservation management. Be efficient.", "support": "Provide troubleshooting steps and solutions to resolve issues quickly.", "product": "Explain features, benefits, and capabilities in an easy-to-understand manner.", "hospitality": "Speak the guest's language: comfort, convenience, and care.", "retail": "Focus on helping customers find and understand products.", "gaming": "Keep it fun, exciting, and engaging!", "finance": "Use business language: ROI, costs, efficiency, revenue.", "healthcare": "Prioritize patient wellbeing and clear explanations.", "education": "Encourage learning and explain concepts clearly.", "technology": "Use technical terminology and provide implementation details.", "corporate": "Focus on efficiency, strategy, and results."}
TONE_INSTRUCTIONS = {"casual": "Use informal, friendly language. Be relaxed and approachable.", "friendly": "Be warm and welcoming. Make the user feel valued.", "formal": "Use professional, structured language. Be precise and clear.", "professional": "Maintain a business-like tone. Be concise and authoritative.", "enthusiastic": "Show excitement and energy. Use positive language.", "empathetic": "Show understanding and compassion. Acknowledge user concerns.", "urgent": "Be direct and action-oriented. Provide quick solutions.", "polite": "Be courteous and respectful. Use please and thank you.", "neutral": "Be objective and factual. Avoid overly emotional language.", "positive": "Use uplifting and encouraging language.", "negative": "Acknowledge concerns but offer constructive solutions.", "quirky": "Be playful, fun, and entertaining.", "technical": "Use precise, detailed technical language.", "welcoming": "Be warm and inviting.", "helpful": "Be supportive and solution-focused."}
INTENT_INSTRUCTIONS = {"exploration": "Show options and possibilities. Help the user discover what's available.", "information": "Provide comprehensive details, comparisons, and explanations.", "conversion": "Focus on value proposition and call-to-action. Guide toward decision.", "support": "Prioritize problem resolution. Provide step-by-step guidance and solutions."}

PROMPT_STYLE_INSTRUCTIONS = {
    "friendly_conversational": {
        "label": "Friendly & Conversational",
        "instruction": "Keep the tone warm, approachable, and lightly enthusiastic. Contractions and friendly phrasing are expected.",
    },
    "professional_clear": {
        "label": "Professional & Clear",
        "instruction": "Use precise, courteous language with clear explanations. Avoid slang and keep structure tight.",
    },
    "curious_engaged": {
        "label": "Curious & Engaged",
        "instruction": "Mirror the inquisitive tone, acknowledge the curiosity, and expand on why the concept matters.",
    },
    "educational_tutorial": {
        "label": "Educational / Tutorial",
        "instruction": "Adopt a patient, step-by-step teaching voice. Set context, explain fundamentals, and build knowledge progressively.",
    },
    "direct_concise": {
        "label": "Direct & Concise",
        "instruction": "Deliver the facts efficiently. Lead with the core answer, then add only the most relevant supporting detail.",
    },
    "neutral": {
        "label": "Neutral",
        "instruction": "Default to a balanced, informative tone that can flex based on subsequent conversation cues.",
    },
}

TECHNICAL_KEYWORDS = {"api","integration","deployment","database","server","framework","library","code","algorithm","model","function","parameter","variable","object","class","method","bug","error","exception","performance","scalability","optimization","architecture","protocol","endpoint","token","authentication","authorization","encryption","compression","latency","throughput","bandwidth","cloud","container","docker","kubernetes","microservice","monolith","distributed","concurrent","parallel","thread","process","memory","cpu","disk","storage","network","socket","port","firewall","load balancer","cache","queue","message","event","stream","batch","pipeline","etl","data","analytics","visualization","dashboard","report","schema","table","row","column","index","key","value","json","xml","csv","yaml","toml","ini","config","settings","environment","secret","vault","monitoring","logging","alert","notification","incident","ticket","support","sla","uptime","downtime","backup","restore","snapshot","replication","failover","high availability","disaster recovery","security","compliance","audit","policy","governance","risk","threat","vulnerability","patch","update","upgrade","release","version","branch","merge","pull request","commit","push","clone","fork","issue","feature","task","story","epic","sprint","kanban","scrum","agile","waterfall","devops","ci","cd","test","unit test","integration test","system test","acceptance test","regression test","performance test","load test","stress test","soak test","smoke test","sanity test","mock","stub","spy","assert","coverage","lint","static analysis","dynamic analysis","profiling","benchmark","trace","debug","log","print","output","input","cli","gui","web","mobile","desktop","app","application","service","daemon","agent","worker","scheduler","cron","timer","event loop","callback","promise","future","async","await","lock","mutex","semaphore","race condition","deadlock","starvation","priority"}

DEFAULT_DOMAIN_MESSAGE = "Default domain guidance:\n- Apply broad knowledge when the domain is unclear.\n- Offer to clarify industry specifics before diving deep.\n- Keep explanations accessible unless the user requests advanced detail."
DEFAULT_TONE_MESSAGE = "Default tone guidance:\n- Lead with a warm, professional voice.\n- Mirror the user's energy without exaggeration.\n- Keep sentences direct and easy to scan."
DEFAULT_INTENT_MESSAGE = "Default intent guidance:\n- Provide the most relevant facts first.\n- Highlight clear next steps or key takeaways.\n- Note any assumptions or limits briefly."

TOPIC_TONE_MAP = {"art": "creative", "automotive": "technical", "career": "motivational", "climate_change": "urgent", "college": "formal", "cryptocurrency": "analytical", "DIY": "creative", "education": "encouraging", "energy": "sustainable", "entertainment": "casual", "environment": "sustainable", "fashion": "trendy", "finance": "professional", "fitness": "motivational", "food": "enthusiastic", "gamer": "casual", "gaming": "excited", "gardening": "peaceful", "government": "official", "health": "empathetic", "history": "narrative", "home_decor": "aesthetic", "hospitality": "welcoming", "insurance": "reassuring", "legal": "authoritative", "literature": "expressive", "logistics": "precise", "manufacturing": "technical", "mental_health": "compassionate", "music": "passionate", "news": "neutral", "parenting": "supportive", "pets": "affectionate", "philosophy": "thoughtful", "photography": "artistic", "politics": "objective", "product": "detailed", "psychology": "insightful", "real_estate": "detailed", "relationships": "understanding", "retail": "helpful", "science": "analytical", "space_exploration": "inspiring", "sports": "energetic", "startup": "innovative", "technology": "informative", "technology_trends": "futuristic", "transport": "efficient", "travel": "adventurous", "videography": "cinematic", "wildlife": "conservationist"}


def evaluate_response(query, response, context=None):
    """
    Evaluate response for relevance, completeness, and confidence.
    """
    relevance = None
    if 'st_model' in globals() and st_model:
        q_emb = st_model.encode(query)
        r_emb = st_model.encode(response)
        relevance = float(st_util.cos_sim(q_emb, r_emb).item())
    completeness = None
    
    if context:
        covered = sum(kw in response for kw in context.domain + context.persona + [context.intent])
        completeness = covered / (len(context.domain) + len(context.persona) + 1)

    confidence = float(len(response)) / 100.0
    if "confident" in response or "certain" in response:
        confidence += 0.1
    
    return {
        "relevance": relevance,
        "completeness": completeness,
        "confidence": min(confidence, 1.0)

    }

def update_prompt_with_feedback(context, scores):
    """
    Refine system prompt using feedback scores.
    """
    base_prompt = context.system_prompt or ""
    adjustments = []

    if scores.get("relevance", 1.0) < 0.5:
        adjustments.append("Please ensure your answer is highly relevant to the user's intent and domain.")
    if scores.get("completeness", 1.0) < 0.7:
        adjustments.append("Make sure to cover all aspects of the user's query.")
    if scores.get("confidence", 1.0) < 0.5:
        adjustments.append("If uncertain, state so and provide the best available information.")

    if not adjustments:
        return base_prompt

    if base_prompt and not base_prompt.endswith("\n"):
        base_prompt += "\n"

    return base_prompt + "\n".join(adjustments)

def synthesize_output_json(context):
    """
    Synthesize the final output as a valid JSON string.
    """
    operational_context = (context.domain[0].strip() if context.domain else "general")

    payload = {
        "domain": list(context.domain or []),
        "intent": context.intent,
        "persona": list(context.persona or []),
        "tone": context.tone,
        "context": operational_context,
        "system_prompt": context.system_prompt,
        "greeting": context.greeting,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

def generate_system_prompt(
    context: 'Optional["PromptContext"]' = None,
    *,
    domain: str | None = None,
    tone: str | None = None,
    intent: str | None = None,
    persona: list[str] | None = None,
    history: list[str] | None = None,
    energy: str = "normal",
    age_group: str = "millennial",
    context_summary: str | None = None,
    prompt_style: str | None = None,
    mode: str = "full",
):
    """Return a minimal, compact, or full system prompt tailored to the current context."""

    def _ensure_context() -> 'PromptContext':
        if context is not None:
            return context
        return PromptContext(
            domain=[domain] if domain else ["general"],
            intent=intent or "information",
            tone=tone or "neutral",
            persona=persona or ["helpful assistant"],
        )

    ctx = _ensure_context()
    primary_domain = ctx.domain[0] if ctx.domain else "general"
    persona_str = ", ".join(ctx.persona)
    operational_context = primary_domain.strip() if ctx.domain else "general"
    effective_summary = (ctx.context_summary or context_summary or "").strip()
    style_label, style_instruction = get_prompt_style_instruction(prompt_style)

    compact_lines = [
        "System role: Adaptive assistant",
        f"Domain: {primary_domain}",
        f"Intent: {ctx.intent}",
        f"Tone: {ctx.tone}",
        f"Persona: {persona_str}",
    ]
    if effective_summary:
        compact_lines.append(f"Context: {effective_summary}")
    if style_label:
        compact_lines.append(f"Prompt Style: {style_label}")
    compact_lines.append("Rules: Be accurate, concise, and structured (bullets/steps); prefer KB facts; if unsure, say so briefly.")
    compact_prompt = "\n".join(compact_lines)

    domain_msg = DOMAIN_INSTRUCTIONS.get(primary_domain, DEFAULT_DOMAIN_MESSAGE)
    tone_msg = TONE_INSTRUCTIONS.get(ctx.tone, DEFAULT_TONE_MESSAGE)
    intent_msg = INTENT_INSTRUCTIONS.get(ctx.intent, DEFAULT_INTENT_MESSAGE)
    contextual_summary = effective_summary if effective_summary else "General context"
    supplemental_lines = SUPPLEMENTAL_INSTRUCTION_TEMPLATE.format(age_group=age_group)

    manual_guidance_lines = [
        f"- Lead with an opening sentence that clearly mirrors the user's tone and energy ({energy}) in the {style_label} style.",
        f"- Maintain that {style_label} voice throughout the response; do not slip back to neutral narration.",
        "- Lead with training-grounded facts; if something is unknown, say so and offer the next best step.",
        "- Provide substantive answers (paragraphs, bullets, or tables). Never collapse the reply into a single line.",
        "- Surface key actions, metrics, or caveats early. Add extra insights when they create value for the user.",
        "- Keep internal instructions private and follow all safety, compliance, and formatting rules exactly.",
    ]
    extra_credit_lines = [
        "• Add a concise recap or call-to-action when it helps the user move forward.",
        "• Suggest follow-up resources or checks if they strengthen the answer.",
        "• Note assumptions that materially affect the guidance.",
    ]
    manual_guidance = "\n".join(manual_guidance_lines)

    full_sections = [
        "You are an adaptive, multi-domain assistant.",
        f"Domain Expertise: {domain_msg}",
        f"Communication Style: {tone_msg}",
        f"User Intent: {intent_msg}",
        f"Persona: {persona_str}",
        f"Energy: {energy}",
        f"Age Group: {age_group}",
        f"Context: {contextual_summary} | Operational Context: {operational_context}",
        "",
        "Instruction Checklist:",
        manual_guidance,
        "",
        f"Prompt Style Target: {style_label} — {style_instruction}",
        "",
        "Extra Points (optional but appreciated):",
        "\n".join(extra_credit_lines),
        "",
        CANONICAL_INSTRUCTION_BLOCK,
        "",
        "Instructions:",
        supplemental_lines,
        "Respond as if written by a domain expert, providing immediate actionable value in a single response.",
    ]

    if history:
        full_sections.append("Contextual history: " + " | ".join(history[-3:]))

    full_prompt = "\n".join(full_sections)

    mode_lc = (mode or "full").lower()
    if mode_lc == "minimal":
        minimal = [
            f"Domain: {primary_domain}",
            f"Intent: {ctx.intent}",
            f"Tone: {ctx.tone}",
            "Rules: accurate, concise, structured; no guessing.",
        ]
        return "\n".join(minimal)
    if mode_lc == "compact":
        return compact_prompt
    return full_prompt

import json
import logging
import random
import re
try:
    import nltk  # Optional; gracefully degrade if unavailable
except Exception:
    nltk = None  # type: ignore
try:
    import spacy  # type: ignore
except Exception:
    spacy = None  # type: ignore
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

nlp_spacy = None  # Lazy-loaded to avoid heavy import at module import time

def _ensure_spacy():
    """Return a cached spaCy pipeline if available, otherwise try to load one."""

    global nlp_spacy

    if nlp_spacy is not None:
        return nlp_spacy

    if spacy is None:  # type: ignore[name-defined]
        return None

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            nlp_spacy = spacy.load(model_name)  # type: ignore[attr-defined]
            return nlp_spacy
        except Exception as exc:  # pragma: no cover - optional model path
            logging.debug("spaCy model %s unavailable: %s", model_name, exc)

    try:
        nlp_spacy = spacy.blank("en")  # type: ignore[attr-defined]
        logging.info("spaCy blank('en') pipeline loaded as fallback.")
    except Exception as exc:  # pragma: no cover - fallback failure
        logging.warning("Failed to initialise spaCy pipeline: %s", exc)
        nlp_spacy = None

    return nlp_spacy

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    st_model = None

try:  # Prefer the high-level import, fall back for older NLTK builds
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
except ImportError:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    except Exception:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        except Exception:
            SentimentIntensityAnalyzer = None  # type: ignore

    try:
        from nltk.tokenize import word_tokenize  # type: ignore
    except Exception:
        # Fallback tokenizer if NLTK is unavailable
        def word_tokenize(text: str):  # type: ignore
            return text.split()

    # Download required NLTK data only if NLTK is available
    if nltk is not None:
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            try:
                nltk.download('vader_lexicon')
            except Exception:
                pass

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt')
            except Exception:
                pass


# Initialize sentiment analyzer (singleton instance)
_sia_instance = None

def _ensure_sentiment_analyzer():
    """Lazily initialise the VADER sentiment analyzer.

    Returns None if the analyser cannot be created (e.g., dependency missing),
    allowing callers to degrade gracefully without raising NameError.
    """

    global _sia_instance

    if _sia_instance is None and SentimentIntensityAnalyzer is not None:  # type: ignore
        try:
            _sia_instance = SentimentIntensityAnalyzer()  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive path
            logging.warning(
                "Sentiment analyzer initialisation failed (%s). Falling back to neutral scores.",
                exc,
            )
            _sia_instance = None

    return _sia_instance

sia = _ensure_sentiment_analyzer()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ========== SHARED CONFIGURATION DICTIONARIES ==========

# Topic to tone mapping (43+ topics with tones)

# Unified context configuration
CONTEXTS = {"support": {"keywords": ["help", "support", "faq", "customer-service"], "system_msg": "You are a customer support assistant. Provide clear and helpful answers to customer questions."}, "technical": {"keywords": ["api", "developer", "docs", "technical", "implementation"], "system_msg": "You are a technical expert. Provide detailed and accurate technical information."}, "sales": {"keywords": ["pricing", "purchase", "product", "order", "buy"], "system_msg": "You are a sales assistant. Help customers understand products and make informed decisions."}, "booking": {"keywords": ["schedule", "appointment", "booking", "reservation"], "system_msg": "You are a booking assistant. Help users schedule and manage their appointments efficiently."}}

# ========== SHARED ANALYSIS FUNCTIONS ==========

def analyze_content(user_message):
    """
    Analyze user message for sentiment and keywords.
    Returns (sentiment_dict, keywords_list)
    
    Used by:
    - app2.py: FAQ analysis and system prompt generation
    - check_and_tune.py: User prompt analysis
    """
    analyzer = sia or _ensure_sentiment_analyzer()
    if analyzer is not None:
        try:
            sentiment = analyzer.polarity_scores(user_message)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Sentiment scoring failed. Falling back to neutral values. Error: %s", exc)
            sentiment = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    else:
        sentiment = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    try:
        tokens = word_tokenize(user_message)
        keywords = [w.lower() for w in tokens if w.isalpha() and len(w) > 3]
    except Exception:
        keywords = user_message.lower().split()
    
    return sentiment, keywords


def classify_prompt_style(user_prompt: str | None) -> str:
    """Heuristically classify the user prompt into a conversational style bucket."""

    if not user_prompt:
        return "neutral"

    text = user_prompt.strip()
    lowered = text.lower()
    words = text.split()

    def _has_any(cues: list[str]) -> bool:
        return any(cue in lowered for cue in cues)

    if _has_any(["let's start", "walk me through", "step by step", "tutorial", "lesson", "basics", "guide me"]):
        return "educational_tutorial"
    if words and len(words) <= 10 and not _has_any(["please", "could you", "would you", "why", "how", "tell me"]):
        return "direct_concise"
    if _has_any(["i've heard", "curious", "wonder", "why do", "why does", "why is", "why are", "why they", "why it", "what are they", "what makes"]):
        return "curious_engaged"
    if text.startswith("hey") or _has_any(["hey", "hi", "hello", "what exactly", "tell me more", "😊", "😀", "👋"]):
        return "friendly_conversational"
    if _has_any(["could you", "would you", "please", "assist", "provide", "kindly", "help me understand", "could i get"]):
        return "professional_clear"
    return "neutral"


def get_prompt_style_instruction(style_key: str) -> tuple[str, str]:
    """Return (label, instruction) text for a given prompt style key."""

    style = PROMPT_STYLE_INSTRUCTIONS.get(style_key or "neutral") or PROMPT_STYLE_INSTRUCTIONS["neutral"]
    return style["label"], style["instruction"]
 
def advanced_topic_detection(keywords, context=None):
    """
    Multi-label topic detection: returns all matched topics.
    
    Features:
    - Detects multiple topics from keywords
    - Falls back to context if no topics detected
    - Adds general product/company domains if appropriate
    
    Args:
        keywords (list): List of keywords to analyze
        context (str): Optional context to use as fallback
    
    Returns:
        list: List of detected topics
    
    Used by: app2.py, check_and_tune.py
    """
    matched_topics = [key for key in TOPIC_TONE_MAP.keys() if key in keywords]
    
    # Add general product/company domains if detected
    general_keywords = {
        "company", "service", "product", "furniture", "appliances", 
        "electronics", "rental", "rent", "mojo", "about", "feature", "benefit"
    }
    if any(gk in keywords for gk in general_keywords):
        for extra in ["product", "sales", "support"]:
            if extra not in matched_topics:
                matched_topics.append(extra)
    
    if not matched_topics and context:
        matched_topics = [context]
    
    if not matched_topics:
        matched_topics = ["general"]
    
    return matched_topics

def nuanced_tone_detection(sentiment_dict, second_sentiment=None):
    """
    Advanced tone detection from sentiment scores.
    
    Args:
        sentiment_dict (dict): Sentiment scores (from analyze_content)
        second_sentiment (dict): Optional second sentiment for comparison
    
    Returns:
        str: Detected tone
    
    Used by:
    - app2.py: With 2 sentiments (question + answer)
    - check_and_tune.py: With 1 sentiment (user message)
    """
    if second_sentiment is None or not isinstance(second_sentiment, dict):
        # Single sentiment (from check_and_tune.py)
        compound = sentiment_dict.get('compound', 0) if isinstance(sentiment_dict, dict) else 0
    else:
        # Two sentiments (from app2.py)
        compound = (sentiment_dict.get('compound', 0) + second_sentiment.get('compound', 0)) / 2
    
    if compound > 0.5:
        return "enthusiastic"
    elif compound > 0.2:
        return "positive"
    elif compound < -0.5:
        return "urgent"
    elif compound < -0.2:
        return "negative"
    else:
        return "neutral"

def persona_expansion(keywords, context=None):
    """
    Multi-label persona detection: returns all matched personas.
    
    Features:
    - Detects multiple relevant personas
    - Context-aware persona selection
    - Comprehensive persona library (11+ personas)
    
    Args:
        keywords (list): List of keywords to analyze
        context (str): Optional context for persona hints
    
    Returns:
        list: List of detected personas
    
    Used by: app2.py, check_and_tune.py
    """
    personas = []
    
    if context == "technical" or "api" in keywords or "developer" in keywords:
        personas.append("technical expert")
    if context == "sales" or "pricing" in keywords or "product" in keywords:
        personas.append("sales assistant")
    if context == "support" or "customer" in keywords or "support" in keywords:
        personas.append("customer support agent")
    if context == "booking" or "schedule" in keywords or "booking" in keywords:
        personas.append("booking assistant")
    if "educate" in keywords or "teach" in keywords or context == "education":
        personas.append("educator")
    if "troubleshoot" in keywords or "error" in keywords or "issue" in keywords:
        personas.append("troubleshooter")
    if "concierge" in keywords or context == "travel":
        personas.append("concierge")
    if "legal" in keywords or context == "legal":
        personas.append("legal advisor")
    if "hospitality" in keywords or context == "hospitality":
        personas.append("hospitality expert")
    if "retail" in keywords or context == "retail":
        personas.append("retail assistant")
    if "government" in keywords or context == "government":
        personas.append("government official")
    
    # Add general company/product personas if detected
    general_keywords = {
        "company", "service", "product", "furniture", "appliances", 
        "electronics", "rental", "rent", "mojo", "about", "feature", "benefit"
    }
    
    if any(gk in keywords for gk in general_keywords):
        for extra in ["customer support agent", "sales assistant", "product expert"]:
            if extra not in personas:
                personas.append(extra)
    
    if not personas:
        personas.append("helpful assistant")
    
    return personas

def advanced_system_prompt_generator(question, answer, context=None):
    """
    Advanced system prompt generator using NLP, context, and extensible rules.
    
    Features:
    - Multi-label topic and persona detection
    - Tone-aware prompt generation
    - Template-based and dynamic generation
    - Context fallback support
    
    Args:
        question (str): User's question
        answer (str): The answer/response to the question
        context (str): Optional context (support, technical, sales, booking)
    
    Returns:
        str: Generated system prompt
    
    Used by: app2.py (for training data generation)
    
    This generates UNIQUE system prompts for each FAQ pair.
    """
    # Analyze both question and answer
    try:
        q_sentiment, q_keywords = analyze_content(question)
    except Exception as e:
        logging.warning(f"NLP analysis failed for question. Error: {e}")
        q_sentiment, q_keywords = {'compound': 0}, []
    
    try:
        a_sentiment, a_keywords = analyze_content(answer)
    except Exception as e:
        logging.warning(f"NLP analysis failed for answer. Error: {e}")
        a_sentiment, a_keywords = {'compound': 0}, []
    
    # Merge keywords
    all_keywords = set(q_keywords + a_keywords)
    if context and context in CONTEXTS:
        all_keywords.update(CONTEXTS[context]["keywords"])
    
    # Multi-label detection
    topics = advanced_topic_detection(all_keywords, context)
    personas = persona_expansion(all_keywords, context)
    tone = nuanced_tone_detection(q_sentiment, a_sentiment)
    
    # Prompt templates for specific combinations
    prompt_template_map = {("technical expert", "casual"): "Hey! I'm your tech pal. I'll explain things simply and keep it chill.", ("technical expert", "enthusiastic"): "Hi! I'm your excited technical expert. Let's dive into this with energy!", ("customer support agent", "casual"): "Hi! I'm your support buddy. Let's solve your issue together, no stress.", ("customer support agent", "empathetic"): "Hi, I'm here for you. I understand your concern and will help you through this.", ("customer support agent", "urgent"): "I'm here to resolve this quickly and efficiently. Let's get this sorted!", ("helpful assistant", "casual"): "Hey there! I'm your friendly assistant. Ask me anything and I'll help out in a relaxed, casual way.", ("helpful assistant", "enthusiastic"): "Hi! I'm your enthusiastic assistant, excited to help you out! Ask away.", ("helpful assistant", "urgent"): "I'm here to help you quickly and efficiently. Let's solve your problem right now!", ("sales assistant", "enthusiastic"): "Hi! I'm excited to help you explore our products and find the perfect fit!", ("troubleshooter", "empathetic"): "Hi, I'm here to help troubleshoot and solve your issue with care and patience."}
    
    # Topic-specific prompts
    topic_prompt_map = {"health": "I'm your caring health assistant. I'll answer with empathy and support for your well-being.", "finance": "You are a finance professional. Give precise, trustworthy, and easy-to-understand financial advice.", "education": "You are an educator. Explain concepts clearly and encourage learning in a supportive way.", "travel": "You are a travel concierge. Offer friendly, adventurous, and helpful travel advice.", "legal": "You are a legal advisor. Provide authoritative, clear, and compliant legal information.", "hospitality": "You are a hospitality expert. Offer welcoming, attentive, and helpful service advice.", "retail": "You are a retail assistant. Provide helpful, friendly, and product-focused answers.", "government": "You are a government official. Provide official, clear, and policy-compliant information.", "technology": "You are a technology expert. Provide clear, detailed, and accurate technical information.", "startup": "You are a startup advisor. Provide innovative, growth-focused, and practical business advice."}
    
    # Score personas/topics and select ONE best persona and ONE best topic to avoid repetition
    def persona_score(p: str) -> int:
        score = 0
        lc = " ".join(all_keywords)
        if context == "support" or any(k in lc for k in ["help", "issue", "support", "error", "not working", "fix"]):
            if p == "customer support agent":
                score += 3
        if context == "sales" or any(k in lc for k in ["buy", "price", "pricing", "order", "subscribe", "plan"]):
            if p == "sales assistant":
                score += 3
        if any(k in lc for k in ["product", "feature", "benefit", "spec", "specification", "compare"]):
            if p in ("product expert", "technical expert"):
                score += 2
        # light bonus for exact context match hints
        if context and p.replace(" ", "") in context.replace(" ", ""):
            score += 1
        return score

    def topic_score(t: str) -> int:
        score = 0
        lc = " ".join(all_keywords)
        if t == "sales" and any(k in lc for k in ["buy", "price", "pricing", "order", "discount", "subscribe", "checkout"]):
            score += 3
        if t == "product" and any(k in lc for k in ["product", "feature", "benefit", "spec", "compare", "details"]):
            score += 3
        return score

    # Choose best persona/topic with deterministic tie-breakers
    persona_order = ["customer support agent", "sales assistant", "product expert", "technical expert", "helpful assistant"]
    topic_order = ["product", "sales", "support", "technology", "general"]
    personas_ranked = sorted(personas[:3] or ["helpful assistant"], key=lambda p: (-persona_score(p), persona_order.index(p) if p in persona_order else 999))
    topics_ranked = sorted(topics[:2] or ["general"], key=lambda t: (-topic_score(t), topic_order.index(t) if t in topic_order else 999))
    best_persona = personas_ranked[0]
    best_topic = topics_ranked[0]

    # Build a single consolidated prompt
    key = (best_persona, tone)
    if key in prompt_template_map:
        base = prompt_template_map[key]
    elif best_topic in topic_prompt_map:
        base = topic_prompt_map[best_topic]
    else:
        base = (
            f"You are a {best_persona} with a {tone} tone. "
            f"Your expertise is in {best_topic}. "
            f"Provide clear, accurate, and helpful answers to user questions."
        )

    # Add two light guardrails consistent with training style
    base += "\nKeep answers concise, structured (bullets/steps), and grounded in known facts; if uncertain, say so briefly."
    canonical = CANONICAL_INSTRUCTION_BLOCK
    supplemental = SUPPLEMENTAL_INSTRUCTION_TEMPLATE.format(age_group="millennial")
    return f"{base}\n\n{canonical}\n\nInstructions:\n{supplemental}"

# ========== INTENT DETECTION FUNCTION ========== 
def detect_intent(query):
    """
    Detect the user's intent (exploration, information, conversion, support, feedback, followup).
    """
    INTENT_KEYWORDS = {"exploration": ["what", "tell me", "show me", "list", "options", "available", "latest", "new", "discover", "explore", "find", "browse", "recommend", "suggest", "demo", "sample", "preview", "tour", "overview"], "information": ["how", "explain", "details", "information", "why", "when", "where", "compare", "define", "clarify", "describe", "meaning", "purpose", "background", "history", "process", "steps", "instruction", "guide", "manual", "policy", "procedure", "specification", "requirement"], "conversion": ["buy", "purchase", "order", "book", "reserve", "subscribe", "sign up", "register", "enroll", "apply", "get", "acquire", "start", "begin", "activate", "upgrade", "download", "checkout", "add to cart", "pay", "payment", "confirm", "complete", "finish", "submit"], "support": ["help", "issue", "problem", "error", "not working", "broken", "fix", "troubleshoot", "support", "assist", "resolve", "repair", "contact", "complaint", "refund", "cancel", "return", "replace", "lost", "forgot", "reset", "recover", "technical", "bug", "fail", "failure", "crash", "hang", "freeze"], "feedback": ["feedback", "suggestion", "review", "rate", "opinion", "comment", "improve", "change", "update", "report", "complain", "recommend", "advise", "critic", "testimonial", "experience", "share", "survey", "poll"], "followup": ["next", "follow up", "continue", "more", "additional", "further", "after", "then", "what's next", "step", "progress", "status", "update", "track", "monitor", "pending", "waiting", "queue"]}
    
    prompt_lower = query.lower()
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        intent_scores[intent] = sum(1 for kw in keywords if kw in prompt_lower)
    best_intent = max(intent_scores, key=intent_scores.get)
    return best_intent if intent_scores[best_intent] > 0 else "information"

# ========== COMPLEXITY ESTIMATION FUNCTION ==========
def estimate_complexity(query):
    """
    Estimate the complexity of a user query based on length, vocabulary, structure, and technical terms.
    Returns: 'low', 'medium', or 'high'
    """
    if not query or not query.strip():
        return "low"

    text = query.strip()
    words = text.split()
    if not words:
        return "low"

    word_count = len(words)
    unique_ratio = len(set(words)) / word_count

    tokens = re.findall(r"[a-zA-Z]{4,}", text)
    technical_hits = sum(1 for token in tokens if token.lower() in TECHNICAL_KEYWORDS)

    logic_terms = {"and", "or", "if", "then", "else", "not", "except"}
    logic_present = any(term in text.lower() for term in logic_terms)

    has_numbers = bool(re.search(r"\d", text))
    has_structural_markers = any(marker in text for marker in (";", ":", "(", ")"))

    complexity_score = 0
    if len(text) > 160 or word_count > 28:
        complexity_score += 1
    if unique_ratio < 0.65:
        complexity_score += 1
    if logic_present:
        complexity_score += 1
    if has_numbers or has_structural_markers:
        complexity_score += 1
    if technical_hits >= 3:
        complexity_score += 2
    elif technical_hits:
        complexity_score += 1

    if len(text) < 45 and technical_hits == 0 and not logic_present:
        return "low"
    if complexity_score >= 4:
        return "high"
    if complexity_score >= 2:
        return "medium"
    return "low"

# ========== TONE-BASED GREETINGS ==========
TONE_GREETINGS = {"casual": ["Hey!", "Hi there!", "Hello!", "Yo!", "Hey friend!", "Hi!", "Hey, how's it going?", "Hey, what's up?", "Glad to help!"], "friendly": ["Hello there!", "Hi, great to see you!", "Welcome!", "Happy to help today!", "Hi! How can I help you today?"], "quirky": ["Yo yo yo!", "What's cookin'?", "Howdy!", "Ahoy!", "Wassup!", "Ready to dive in?"], "fun": ["Let's get this started!", "Woohoo!", "Yay!", "Time for some fun!", "Let's do this!", "Excited to help!"], "formal": ["Greetings.", "Good day.", "Welcome.", "How may I assist you?"], "professional": ["Hello, how can I assist you?", "Welcome—how can I help?", "Good day, I'm here to help.", "Thank you for reaching out."], "enthusiastic": ["Hi! I'm excited to help!", "Hello! Let's get started!", "Awesome—let's solve this together!"], "empathetic": ["Hi, I'm here for you.", "Hello, I understand your concern.", "I'm here to help you through this."], "urgent": ["Let's resolve this quickly.", "I'm on it right away!", "We'll sort this ASAP."], "neutral": ["Hello.", "Hi.", "Welcome.", "How can I help you?"]}

def prepend_tone_greeting(
    response: str,
    tone: str,
    energy: str | None = None,
    age_group: str | None = None,
    platform: str | None = None,
    user_prompt: str | None = None,
    mimic_energy: bool = False,
):
    """Energy-adaptive greeting.

    If mimic_energy is True:
      - low energy: usually drop greeting, answer directly.
      - medium energy: minimal neutral greeting only if tone strongly indicates it; otherwise direct answer.
      - high energy: amplify greeting and keep enthusiastic punctuation.
    Falls back to tone-based greeting lists when mimic_energy is False.
    Backwards compatible for earlier calls (only response + tone).
    """
    tone_lc = (tone or "neutral").strip().lower()
    energy_lc = (energy or "medium").strip().lower()
    prompt = (user_prompt or "").strip()

    # Base greetings set
    greetings = list(TONE_GREETINGS.get(tone_lc, TONE_GREETINGS["neutral"]))

    # Platform nuance (adds candidates, not mandatory)
    if platform:
        p = platform.lower()
        if any(k in p for k in ["instagram", "social", "gaming"]):
            greetings += ["Hey!", "Hi there!", "Let's do this!"]
        elif any(k in p for k in ["corporate", "college", "healthcare", "portal"]):
            greetings += ["Good day.", "Hello, how can I assist you?"]

    # Age group subtle enrichment
    if age_group:
        ag = age_group.lower()
        if ag in ("genz", "millennial") and tone_lc in ("casual", "friendly", "fun", "quirky", "enthusiastic"):
            greetings += ["Hey!", "Hi!", "Welcome!"]
        elif ag == "boomer" and tone_lc in ("professional", "formal", "neutral"):
            greetings += ["Good day.", "Hello."]

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for g in greetings:
        if g not in seen:
            deduped.append(g)
            seen.add(g)
    greetings = deduped or ["Hello."]

    # Energy mimic logic
    if mimic_energy:
        # Low energy or terse question: drop greeting if prompt ends with '?'
        if energy_lc == "low":
            if prompt.endswith("?") or len(prompt) < 60:
                return response  # direct answer, no greeting
            # else pick a calm greeting without exclamation
            calm = [g.replace("!", ".") for g in greetings]
            return f"{random.choice(calm)} {response}".strip()
        if energy_lc == "medium":
            # Medium: minimal greeting only if professional/formal tone, else skip
            if tone_lc in ("professional", "formal", "neutral"):
                base = random.choice(greetings)
                base = base.replace("!", ".")  # moderate
                return f"{base} {response}".strip()
            else:
                return response
        if energy_lc == "high":
            excited = [g if g.endswith("!") else g.rstrip(".") + "!" for g in greetings]
            return f"{random.choice(excited)} {response}".strip()

    # Non-mimic fallback (legacy behavior)
    chosen = random.choice(greetings)
    if energy_lc == "high" and not chosen.endswith("!"):
        chosen = chosen.rstrip(".") + "!"
    elif energy_lc == "low":
        chosen = chosen.replace("!", ".")
    return f"{chosen} {response}".strip()

@dataclass
class PromptContext:
    domain: List[str] = field(default_factory=list)
    intent: str = "general"
    tone: str = "neutral"
    persona: List[str] = field(default_factory=lambda: ["helpful assistant"])
    complexity: str = "medium"
    entities: List[str] = field(default_factory=list)
    context_summary: str = ""
    system_prompt: str = ""
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    greeting: str = ""
        
def analyze_query(query: str) -> PromptContext:
    """
    Deeply analyze user query for domain(s), intent, tone, persona, entities, and context vector.
    Returns a PromptContext dataclass instance.
    """
    # 1. Sentiment & Tone
    sentiment, keywords = analyze_content(query)
    tone = nuanced_tone_detection(sentiment)

    # 2. Domain detection (multi-label)
    domains = advanced_topic_detection(keywords)
    if not domains:
        domains = ["general"]
    
    # 3. Intent detection (simple + advanced)
    intent = detect_intent(query)
    
    # 4. Persona detection
    personas = persona_expansion(keywords)
    if not personas:
        personas = ["helpful assistant"]
    
    # 5. Named Entity Recognition
    entities = []
    nlp = _ensure_spacy()
    if nlp:
        doc = nlp(query)
        entities = [ent.text for ent in doc.ents]
    
    # 6. Complexity estimation
    complexity = estimate_complexity(query)

    # 7. Context vector (embedding)
    context_vector = ""
    if st_model:
        emb = st_model.encode(query)
        context_vector = str(emb.tolist()[:8])  # Truncate for brevity
    
    # 8. Context summary
    context_summary = f"Domains: {domains}, Intent: {intent}, Tone: {tone}, Persona: {personas}, Entities: {entities}, Complexity: {complexity}"
    
    return PromptContext(
        domain=domains,
        intent=intent,
        tone=tone,
        persona=personas,
        complexity=complexity,
        entities=entities,
        context_summary=context_summary,
        system_prompt="",
        quality_scores={},
        greeting=""
    )
