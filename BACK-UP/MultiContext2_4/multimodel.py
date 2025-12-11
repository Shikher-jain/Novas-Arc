import os
import logging
import openai
import json
from dotenv import load_dotenv
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect

# Import shared configuration and functions to eliminate duplication
from shared_config import (
    analyze_content,
    nuanced_tone_detection,
    advanced_system_prompt_generator,
    evaluate_response,
    update_prompt_with_feedback,
    analyze_query,
    synthesize_output_json,
    prepend_tone_greeting,
    generate_system_prompt,
    classify_prompt_style,
    get_prompt_style_instruction,
)

# Fine-tuning configuration
# TRAINING_FILE_PATH = "FineTuning/domain_tone_intent_general_training.jsonl"
# # TRAINING_FILE_PATH = "FineTuning/rentomojodesk_freshdesk_com.jsonl"
# # TRAINING_FILE_PATH = "FineTuning/aws_amazon_com.jsonl"

# BASE_MODEL = "gpt-3.5-turbo"

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Silence noisy third‑party loggers (httpx/OpenAI SDK)
for noisy_logger in [
    "httpx",
    "httpcore",
    "openai",
    "openai._base_client",
    "openai._base_client.http_client",
]:
    nl = logging.getLogger(noisy_logger)
    nl.setLevel(logging.WARNING)
    nl.propagate = False

# ========== ENHANCEMENT 1: EXPANDED INDUSTRY DOMAINS ==========
# Industry-specific keywords for better domain detection (check_and_tune specific)
INDUSTRY_DOMAINS = {
    "hospitality": {
        "keywords": ["hotel", "restaurant", "booking", "reservation", "guest", "stay", "check-in", "checkout", "amenity", "concierge", "lodge", "motel"],
        "tone": "welcoming",
        "description": "Hospitality & Tourism"
    },
    "retail": {
        "keywords": ["store", "shop", "product", "inventory", "purchase", "item", "catalog", "discount", "offer", "sale", "checkout", "cart"],
        "tone": "helpful",
        "description": "Retail & E-Commerce"
    },
    "gaming": {
        "keywords": ["game", "play", "level", "score", "character", "achievement", "quest", "mission", "player", "multiplayer", "console", "app"],
        "tone": "quirky",
        "description": "Gaming & Entertainment"
    },
    "finance": {
        "keywords": ["bank", "account", "transaction", "investment", "loan", "credit", "deposit", "withdraw", "balance", "payment", "interest", "portfolio"],
        "tone": "professional",
        "description": "Finance & Banking"
    },
    "healthcare": {
        "keywords": ["doctor", "patient", "appointment", "prescription", "medical", "hospital", "clinic", "symptom", "treatment", "medicine", "health", "diagnosis"],
        "tone": "empathetic",
        "description": "Healthcare & Medical"
    },
    "education": {
        "keywords": ["student", "course", "class", "lesson", "university", "college", "school", "professor", "assignment", "exam", "learning", "enrollment"],
        "tone": "encouraging",
        "description": "Education & Learning"
    },
    "technology": {
        "keywords": ["code", "api", "software", "system", "database", "server", "deployment", "integration", "developer", "framework", "library", "bug"],
        "tone": "technical",
        "description": "Technology & IT"
    },
    "corporate": {
        "keywords": ["business", "company", "employee", "hr", "management", "meeting", "project", "deadline", "budget", "revenue", "strategy", "report"],
        "tone": "professional",
        "description": "Corporate Business"
    }
}

# ========== ENHANCEMENT 2: PLATFORM-BASED TONE MAPPER ==========
# Map platforms/touchpoints to appropriate tones (check_and_tune specific)
PLATFORM_TONE_MAP = {
    "instagram": {
        "tone": "casual",
        "description": "Instagram/Social Media",
        "style": "friendly, engaging, emoji-friendly"
    },
    "corporate_website": {
        "tone": "professional",
        "description": "Corporate Website",
        "style": "formal, authoritative, structured"
    },
    "gaming_app": {
        "tone": "quirky",
        "description": "Gaming/Entertainment App",
        "style": "fun, playful, energetic"
    },
    "college_website": {
        "tone": "formal",
        "description": "College/University",
        "style": "academic, encouraging, informative"
    },
    "support_chat": {
        "tone": "empathetic",
        "description": "Customer Support Chat",
        "style": "helpful, patient, solution-focused"
    },
    "ecommerce": {
        "tone": "friendly",
        "description": "E-Commerce Platform",
        "style": "persuasive, helpful, product-focused"
    },
    "healthcare_portal": {
        "tone": "empathetic",
        "description": "Healthcare Portal",
        "style": "caring, clear, informative"
    },
    "technical_docs": {
        "tone": "technical",
        "description": "Technical Documentation",
        "style": "precise, detailed, example-based"
    }
}

# ========== ENHANCEMENT 3: TOUCHPOINT-TO-INTENT MAPPER ==========
# Map different touchpoints to expected user intents (check_and_tune specific)
TOUCHPOINT_INTENT_MAP = {
    "home_page": {
        "primary_intent": "exploration",
        "description": "User exploring options and learning about offerings",
        "keywords": ["what", "show", "tell", "options", "available", "browse", "explore"]
    },
    "product_page": {
        "primary_intent": "information",
        "description": "User wants detailed information about specific product/service",
        "keywords": ["details", "how", "features", "specs", "benefits", "compare", "explain"]
    },
    "support_page": {
        "primary_intent": "support",
        "description": "User seeking issue resolution or help",
        "keywords": ["help", "problem", "issue", "error", "fix", "troubleshoot", "not working"]
    },
    "checkout": {
        "primary_intent": "conversion",
        "description": "User ready to make purchase or commit to action",
        "keywords": ["buy", "purchase", "order", "subscribe", "book", "reserve", "checkout"]
    },
    "faq_section": {
        "primary_intent": "information",
        "description": "User looking for quick answers to common questions",
        "keywords": ["how", "what", "when", "where", "why", "steps", "guide"]
    },
    "feedback_form": {
        "primary_intent": "support",
        "description": "User providing feedback or reporting issues",
        "keywords": ["problem", "feedback", "suggest", "improve", "issue", "complaint"]
    }
}

def detect_intent_from_prompt(user_prompt):
    """
    Detect the user's intent (exploration, information, conversion, support).
    """
    INTENT_KEYWORDS = {
        "exploration": ["what", "tell me", "show me", "list", "options", "available", "latest", "new"],
        "information": ["how", "explain", "details", "information", "why", "when", "where", "compare"],
        "conversion": ["buy", "purchase", "order", "book", "reserve", "subscribe", "sign up"],
        "support": ["help", "issue", "problem", "error", "not working", "broken", "fix", "troubleshoot"]
    }
    prompt_lower = user_prompt.lower()
    
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        intent_scores[intent] = sum(1 for kw in keywords if kw in prompt_lower)
    
    best_intent = max(intent_scores, key=intent_scores.get)
    return best_intent if intent_scores[best_intent] > 0 else "information"

def universal_prompt_analysis_and_generation(user_prompt):
    """
    Legacy universal prompt analysis and generation function.
    Returns a simple analysis dict for compatibility.
    """
    return {
        "prompt": user_prompt,
        "length": len(user_prompt),
        "keywords": [w for w in re.findall(r"[a-zA-Z]{4,}", user_prompt.lower())],
        "info": "Legacy universal analysis placeholder."
    }

def detect_domain_from_prompt(user_prompt):
    """
    Detect the domain/context of the user's prompt.
    """
    DOMAIN_KEYWORDS = {
        "technical": ["api", "code", "developer", "implementation", "integration", "technical", "system"],
        "sales": ["price", "pricing", "cost", "discount", "offer", "package", "plan", "buy"],
        "booking": ["book", "schedule", "appointment", "reservation", "availability", "date", "time"],
        "support": ["help", "issue", "problem", "error", "support", "customer service", "assistance"],
        "product": ["product", "feature", "functionality", "capability", "benefit", "advantage"]
    }
    prompt_lower = user_prompt.lower()
    
    domain_scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        domain_scores[domain] = sum(1 for kw in keywords if kw in prompt_lower)
    
    best_domain = max(domain_scores, key=domain_scores.get)
    return best_domain if domain_scores[best_domain] > 0 else "product"

# ========== ENHANCEMENT 4: INDUSTRY DOMAIN DETECTOR ==========
def detect_industry_domain(user_prompt):
    """
    Detect industry-specific domain from user prompt.
    Returns industry domain and relevant metadata.
    """
    prompt_lower = user_prompt.lower()
    industry_scores = {}
    
    for industry, data in INDUSTRY_DOMAINS.items():
        industry_scores[industry] = sum(1 for kw in data["keywords"] if kw in prompt_lower)
    
    best_industry = max(industry_scores, key=industry_scores.get)
    
    if industry_scores[best_industry] > 0:

        print("best industry, industry_scores[best_industry], ",best_industry, INDUSTRY_DOMAINS[best_industry])
        return best_industry, INDUSTRY_DOMAINS[best_industry]
    return "general", {"keywords": [], "tone": "neutral", "description": "General"}

# ========== ENHANCEMENT 5: PLATFORM TONE DETECTOR ==========
def detect_platform_context(platform_name):
    """
    Detect appropriate tone based on platform/touchpoint.
    Returns tone and platform metadata.
    """
    platform_lower = platform_name.lower()
    
    for platform, data in PLATFORM_TONE_MAP.items():
        if platform in platform_lower or platform_name == platform:
            return data["tone"], data
    
    # Default to neutral if platform not recognized
    return "neutral", {"tone": "neutral", "description": "Unknown Platform", "style": "helpful"}

# ========== ENHANCEMENT 6: TOUCHPOINT INTENT DETECTOR ==========
def detect_touchpoint_context(touchpoint_name):
    """
    Detect user intent based on touchpoint/page type.
    Returns primary intent and touchpoint metadata.
    """
    touchpoint_lower = touchpoint_name.lower()
    
    for touchpoint, data in TOUCHPOINT_INTENT_MAP.items():
        if touchpoint in touchpoint_lower or touchpoint_name == touchpoint:
            return data["primary_intent"], data
    
    # Default to information gathering if not recognized
    return "information", {"primary_intent": "information", "description": "Unknown Touchpoint"}
    
def validate_fine_tuned_model(model_id, user_prompt, platform="corporate_website", touchpoint="support_page", age_group="millennial", answer_style: str = "medium", strict_mode: bool = False, energy_level: str = "normal"):
    """
    Validate the fine-tuned model with ADVANCED multi-dimensional Domain × Tone × Intent analysis.
    
    Implements the company's requirement for adaptive communication based on:
    1. DOMAIN (Industry/Business Context) - detected from user prompt
    2. TONE (Communication Style) - detected from platform/touchpoint
    3. INTENT (User Goal) - detected from what user is trying to do
    
    Uses next-level logic from app2.py plus new enhancements:
    - Analyzes sentiment and keywords from the user prompt
    - Detects tone, intent, domain, industry, platform, and touchpoint
    - Generates an advanced system prompt using personas, topics, and dimensions
    - Returns response with full 9-dimensional analysis metadata

    Args:
        model_id (str): The fine-tuned model ID.
        user_prompt (str): The user prompt to test the model.
        platform (str): Platform/channel (instagram, corporate_website, etc.)
        touchpoint (str): Touchpoint type (home_page, product_page, support_page, etc.)

    Returns:
        dict: Contains response, analysis metadata (9 dimensions), and system prompt used
    """
    
    try:
        # ============ ORIGINAL DETECTION (3 dimensions) ============
        # STEP 1: Advanced content analysis
        sentiment, keywords = analyze_content(user_prompt)

        # STEP 2: Multi-dimensional analysis
        tone = nuanced_tone_detection(sentiment)
        intent = detect_intent_from_prompt(user_prompt)
        domain = detect_domain_from_prompt(user_prompt)

        # STYLE CLASSIFICATION (prompt-driven voice guidance)
        prompt_style_key = classify_prompt_style(user_prompt)
        prompt_style_label, prompt_style_instruction = get_prompt_style_instruction(prompt_style_key)

        # ============ NEW ENHANCEMENTS (6 additional dimensions) ============
        # STEP 3: Industry/domain detection
        industry, industry_data = detect_industry_domain(user_prompt)
        
        # STEP 4: Platform-aware tone detection
        platform_tone, platform_data = detect_platform_context(platform)
        
        # STEP 5: Touchpoint-aware intent detection
        touchpoint_intent, touchpoint_data = detect_touchpoint_context(touchpoint)
         
        # ============ COMPOSITE SYSTEM PROMPT GENERATION ============
        # STEP 6: Create unified system prompt (Domain × Tone × Intent + structured + merged)
        composite_system_prompt = generate_system_prompt(
            domain=industry,
            tone=platform_tone,
            intent=touchpoint_intent,
            persona=None,
            energy=energy_level or "normal",
            age_group=age_group,
            prompt_style=prompt_style_key,
        )
        
        # STEP 7: FALLBACK - Also generate traditional advanced prompt for hybrid approach
        keywords_list = list(keywords)
        context = domain  # Use detected domain as context
        
        advanced_system_prompt = advanced_system_prompt_generator(
            user_prompt,
            "",  # Empty answer - just for topic/persona detection
            context
        )
        
        # Prefer the training-style advanced prompt at chat time; merge with guidelines from composite
        def _extract_guidelines_sections(text: str) -> str:
            if not text:
                return ""
            lines = (text or "").splitlines()
            keep_heads = {
                "Behavior Guidelines:",
                "Output Formatting:",
                "Safety Rules:",
                "Instructions:",
            }
            sections = []
            current_head = None
            current = []
            for ln in lines:
                if ln.strip() in keep_heads:
                    if current_head is not None:
                        sections.append(current_head + "\n" + "\n".join(current).strip())
                    current_head = ln.strip()
                    current = []
                    continue
                if current_head is not None:
                    current.append(ln)
            if current_head is not None:
                sections.append(current_head + "\n" + "\n".join(current).strip())
            return "\n\n".join(s for s in sections if s and s.strip())

        guidelines_block = _extract_guidelines_sections(composite_system_prompt)
        if advanced_system_prompt:
            combined = advanced_system_prompt.rstrip()
            if guidelines_block:
                combined += "\n\n" + guidelines_block
            final_system_prompt = combined
            final_system_prompt_source = "Advanced+Guidelines"
        else:
            final_system_prompt = composite_system_prompt
            final_system_prompt_source = "Domain×Tone×Intent Composite"
        
        # STEP 8: Call the model with the final system prompt
        # Build messages: keep instructions in system to reduce echo; user contains only the question
        # Guidance differs between relaxed and strict modes
        energy_note = f" Match the user's energy level: {energy_level or 'normal'}."

        style_note = (
            f" Maintain a {prompt_style_label} voice at every paragraph. {prompt_style_instruction} "
            "Open with a sentence that instantly reflects this tone before sharing facts."
        )

        if strict_mode:
            guidance = (
                "Keep your answer closely aligned with the fine-tuned training data. "
                "Avoid speculation or external knowledge unless it is common-sense connective tissue. "
                "Stay factual and concise; feel free to rephrase for clarity. "
                "If a detail is missing, reply with: 'Not available in training data.' "
                + energy_note
                + style_note
            )
        else:
            guidance = (
                get_generation_instructions(answer_style, age_group)
                + " Prioritise verified training facts, with light paraphrasing or clarifying context when helpful. "
                + "Do not invent new facts, and start directly with the answer without repeating the instructions."
                + energy_note
                + style_note
            ) 

        system_content = final_system_prompt + "\n\n" + guidance

        cfg = STRICT_INFERENCE_CONFIG if strict_mode else INFERENCE_CONFIG

        response = openai.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ],

            temperature=cfg["temperature"],
            presence_penalty=cfg["presence_penalty"],
            frequency_penalty=cfg["frequency_penalty"],
            max_tokens=1500,
        )

        # print("="*40)
        # print(final_system_prompt)
        # print("="*40)

        model_response = response.choices[0].message.content or ""

        # Light post-processing to remove accidental echoes of the question/instructions
        def sanitize_model_response(text: str, original_question: str) -> str:
            t = (text or "").strip()
            oq = (original_question or "").strip()
            if oq and t.startswith(oq):
                t = t[len(oq):].lstrip("\n: -")
            # Remove a known instruction phrase if it leaked
            leak_phrases = [
                "Provide a comprehensive answer",
                "If the solution involves multiple steps, explain them clearly",
                "Start directly with the answer",
            ]
            lines = [ln for ln in t.splitlines() if not any(p in ln for p in leak_phrases)]
            cleaned = "\n".join(lines).strip()
            return cleaned or t

        model_response = sanitize_model_response(model_response, user_prompt)
        model_response = _collapse_redundant_sentences(model_response)
        
        # ============ COMPREHENSIVE ANALYSIS RETURN ============
        # STEP 9: Return response with FULL 9-DIMENSIONAL analysis for transparency
        return {
            "response": model_response,
            "analysis": {
                # Original 3 dimensions
                "tone": tone,
                "intent": intent,
                "domain": domain,
                "energy_level": energy_level or "normal",
                "prompt_style": prompt_style_label,
                "prompt_style_key": prompt_style_key,
                
                # New 6 dimensions from enhancements
                "industry": industry,
                "industry_description": industry_data.get("description", "General"),
                "industry_default_tone": industry_data.get("tone", "neutral"),
                "platform": platform,
                "platform_tone": platform_tone,
                "platform_description": platform_data.get("description", "Unknown Platform"),
                "touchpoint": touchpoint,
                "touchpoint_intent": touchpoint_intent,
                "touchpoint_description": touchpoint_data.get("description", "Unknown Touchpoint"),
                
                # Sentiment and keywords
                "keywords": keywords_list[:5],  # Top 5 keywords
                "sentiment_compound": sentiment.get('compound', 0),
                "sentiment_positive": sentiment.get('pos', 0),
                "sentiment_negative": sentiment.get('neg', 0)
            },
            "system_prompt": final_system_prompt,
            "system_prompt_source": final_system_prompt_source
        }
    except Exception as e:
        logging.error(f"Failed to validate fine-tuned model: {e}")
        raise

def save_model_id(model_id, training_file):
    """
    Save the fine-tuned model ID next to the training file:
    e.g., FineTuning/<name>.jsonl -> FineTuning/<name>.txt
    """
    try:
        output_file = os.path.splitext(training_file)[0] + ".txt"  # FineTuning/<name>.txt
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(model_id)
        logging.info("Model ID saved to %s", output_file)
    except Exception as e:
        logging.error("Failed to save model ID: %s", e)

def select_training_file(folder="."):
    """
    Allow the user to select a training file from the specified folder.

    Args:
        folder (str): The folder to search for training files.

    Returns:
        str: The path to the selected training file.
    """
    try:
        logging.debug("Current directory: %s", os.getcwd())
        logging.debug("Files in directory: %s", os.listdir(folder))

        # List all JSONL files in the folder
        files = [f for f in os.listdir(folder) if f.endswith(".jsonl")]
        if not files:
            logging.error("No JSONL files found in the specified folder. Available: %s", os.listdir(folder))
            raise FileNotFoundError("No JSONL files found in the specified folder.")

        # Display the files for selection
        logging.info("Available training files:")
        for i, file in enumerate(files, start=1):
            logging.info("%d. %s", i, file)

        # Prompt the user to select a file
        while True:
            try:
                choice = int(input("Select a training file by number: "))
                if 1 <= choice <= len(files):
                    return os.path.join(folder, files[choice - 1])
                else:
                    logging.warning("Invalid choice. Please select a valid number.")
            except ValueError:
                logging.warning("Please enter a number.")
    except Exception as e:
        logging.error("Error selecting training file: %s", e)
        raise

# ========== MULTI-MODEL (ENSEMBLE) SUPPORT ==========
def load_model_registry(folder="FineTuning"):
    """
    Load all fine-tuned model IDs saved as FineTuning/<name>.txt
    Returns: { "<name>": "<model-id>", ... }
    """
    registry = {}
    try:
        for path in glob.glob(os.path.join(folder, "*.txt")):
            key = os.path.splitext(os.path.basename(path))[0]
            with open(path, "r", encoding="utf-8") as f:
                mid = f.read().strip()
            if mid:
                registry[key] = mid
    except Exception as e:
        logging.warning("Failed to load model registry: %s", e)
    return registry


# Default max models per turn (env override: ENSEMBLE_MAX_MODELS)
ENSEMBLE_MAX_MODELS = int(os.getenv("ENSEMBLE_MAX_MODELS", "1"))

# Centralized inference configuration for easier tuning/cost control
INFERENCE_CONFIG = {
    "temperature": 0.2,
    "presence_penalty": 0.6,
    "frequency_penalty": 0.8,
}

# Strict mode configuration (training-grounded, conservative)
STRICT_INFERENCE_CONFIG = {
    "temperature": 0.1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

def get_generation_instructions(answer_style: str, age_group: str) -> str:
    """Build concise, reusable generation instructions.

    answer_style controls length/verbosity:
      - short: ~100–150 words
      - medium: ~200–300 words
      - long: ~350–500 words
    """
    style_map = {
        "short": "Provide a detailed answer (~180–240 words).",
        "medium": "Provide a thorough answer (~320–420 words).",
        "long": "Provide an in-depth answer (~520–650 words).",
    }
    style_note = style_map.get(str(answer_style or "").lower(), style_map["medium"])

    return (
        style_note
    + " If the solution involves multiple steps, explain them clearly and in order, expanding on the reasoning behind each step. "
    + "Use bullets, numbered lists, or tables when they improve clarity, and include concrete examples or callouts when helpful. "
    + "Mirror the user's energy appropriately while keeping the explanation rich and engaging. "
    + f"Adapt for the user's age group: {age_group}. "
    + "Avoid repetition and avoid hallucinating facts, but elaborate on context, caveats, and recommended follow-up actions. "
        + "If information is uncertain or unavailable, say so briefly and suggest next steps."
    )

def pick_models_for_ensemble(registry, forced_keys=None, max_models=ENSEMBLE_MAX_MODELS):
    """
    Choose up to max_models from registry. If forced_keys provided, honor those.
    If max_models is None or < 1, use all available models.
    """
    chosen = []
    if forced_keys:
        for k in forced_keys:
            if k in registry:
                chosen.append((k, registry[k]))
    # if max_models <= 0 → use all
    limit = max_models if (max_models and max_models > 0) else len(registry)
    if len(chosen) < limit:
        for k in sorted(registry.keys()):
            if forced_keys and k in forced_keys:
                continue
            chosen.append((k, registry[k]))
            if len(chosen) >= limit:
                break
    return chosen

def _extract_keywords(text):
    try:
        _, kws = analyze_content(text)
        return set(kws)
    except Exception:
        return set(w for w in re.findall(r"[a-zA-Z]{4,}", text.lower()))

def score_response(user_prompt, assistant_text):
    """
    Simple heuristic: keyword overlap Jaccard + length bonus (bounded).
    """
    uk = _extract_keywords(user_prompt)
    ak = _extract_keywords(assistant_text)
    jacc = (len(uk & ak) / max(1, len(uk | ak))) if (uk or ak) else 0.0

    '''
    Jaccard = ∣ Keywordsprompt ​∪ Keywordsresponse​ ∣ / ∣Keywordsprompt ​ ∩ Keywordsresponse​∣
    '''
    length_bonus = min(len(assistant_text) / 600.0, 0.3)  # cap bonus
    return jacc + length_bonus


def _collapse_redundant_sentences(text: str) -> str:
    """Collapse repeated sentences with the same leading phrase into one line."""

    if not text or "\n" not in text:
        return text

    blocks = text.split("\n\n")
    collapsed_blocks = []

    for raw_block in blocks:
        block = raw_block.strip()
        if not block:
            collapsed_blocks.append(raw_block)
            continue

        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            collapsed_blocks.append(raw_block)
            continue
        if any(ln.startswith(('-', '*', '•')) for ln in lines):
            collapsed_blocks.append(raw_block)
            continue

        word_lists = [re.sub(r"[.?!]+$", "", ln).split() for ln in lines]
        if not word_lists or any(len(words) < 3 for words in word_lists):
            collapsed_blocks.append(raw_block)
            continue

        prefix_words = word_lists[0][:]
        for words in word_lists[1:]:
            while prefix_words and words[:len(prefix_words)] != prefix_words:
                prefix_words = prefix_words[:-1]
            if len(prefix_words) < 3:
                break

        if len(prefix_words) < 3:
            collapsed_blocks.append(raw_block)
            continue

        if any(words[:len(prefix_words)] != prefix_words for words in word_lists):
            collapsed_blocks.append(raw_block)
            continue

        suffixes = []
        seen = set()
        for words in word_lists:
            suffix = " ".join(words[len(prefix_words):]).strip(" ,.;")
            if not suffix:
                continue
            key = suffix.lower()
            if key not in seen:
                seen.add(key)
                suffixes.append(suffix)

        if not suffixes:
            collapsed_blocks.append(" ".join(lines))
            continue

        base = " ".join(prefix_words)
        if len(suffixes) == 1:
            sentence = f"{base} {suffixes[0]}".strip()
        else:
            body = ", ".join(suffixes[:-1]) + ", " + suffixes[-1]
            sentence = f"{base} {body}".strip()
        if not sentence.endswith('.'):
            sentence += '.'
        collapsed_blocks.append(sentence)

    return "\n\n".join(collapsed_blocks)

def ensemble_ask(models, user_prompt, platform="corporate_website", touchpoint="support_page", answer_style: str = "medium", strict_mode: bool = False, energy_level: str = "normal"):
    """
    Call multiple models (in parallel), score their answers, return best and all candidates.
    """
    candidates = []

    # Get age_group from caller's frame (main loop)
    frame = inspect.currentframe().f_back
    age_group = frame.f_locals.get('age_group', 'millennial')

    def _call_one(key_mid):
        key, model_id = key_mid
        r = validate_fine_tuned_model(
            model_id,
            user_prompt,
            platform=platform,
            touchpoint=touchpoint,
            age_group=age_group,
            answer_style=answer_style,
            strict_mode=strict_mode,
            energy_level=energy_level,
        )
        sc = score_response(user_prompt, r.get("response", ""))
        return key, model_id, r, sc

    # Parallel calls for speed
    with ThreadPoolExecutor(max_workers=min(len(models), 6)) as ex:
        future_map = {ex.submit(_call_one, km): km for km in models}
        for fut in as_completed(future_map):
            try:
                key, model_id, r, sc = fut.result()
                candidates.append((key, model_id, r, sc))
            except Exception as e:
                key, model_id = future_map[fut]
                logging.warning("Model %s failed: %s", key, e)

    if not candidates:
        raise RuntimeError("No model produced a response.")
    candidates.sort(key=lambda x: x[3], reverse=True)
    best = candidates[0][2]
    return best, candidates

def main():
    # Ensure API key is set
    if not openai.api_key:
        raise RuntimeError("OpenAI API key is not set. Please set it in your environment variables.")

    # Load all available fine-tuned models (FineTuning/*.txt)
    model_registry = load_model_registry(folder="FineTuning")
    logging.info("\n" + "=" * 80)
    logging.info("ADAPTIVE CHAT SUPPORT (Ensemble: use 2 models per turn)")
    logging.info("Models available: %s", ", ".join(sorted(model_registry.keys())) or "(none)")
    logging.info("Commands: /models, /use <key1,key2,...>, /k <N>, /all, /both, /style <short|medium|long>, /strict [on|off], /exit")
    logging.info("=" * 80 + "\n")

    show_all = False
    forced_pair = None
    max_models = ENSEMBLE_MAX_MODELS
    age_group = "millennial"  # Default age group
    answer_style = "medium"   # Default response length/style
    strict_mode = False        # Default: relaxed (non-strict) mode

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            logging.info("Please enter a message.\n")
            continue

        low = user_prompt.lower() if user_prompt else ""
        if low in ("/exit", "exit", "quit"):
            logging.info("Exiting chatbot. Thank you!")
            break
        if low in ("/models", "models"):
            logging.info("Models: %s", ", ".join(sorted(model_registry.keys())))
            continue
        if low.startswith("/use "):
            _, _, keys = user_prompt.partition(" ")
            requested_models = [k.strip() for k in keys.split(",") if k.strip()]
            missing_models = [k for k in requested_models if k not in model_registry]
            if missing_models:
                print(f"Error: The following requested model(s) are not available: {', '.join(missing_models)}.\nAvailable models: {', '.join(sorted(model_registry.keys())) or '(none)'}")
                logging.error("Requested model(s) not found: %s", ', '.join(missing_models))
                forced_pair = None
                continue
            forced_pair = requested_models
            max_models = len(forced_pair) if forced_pair else ENSEMBLE_MAX_MODELS
            logging.info("Using models: %s", ", ".join(forced_pair) if forced_pair else "(none)")
            continue
        if low.startswith("/k "):
            _, _, kstr = user_prompt.partition(" ")
            try:
                max_models = max(1, int(kstr.strip()))
                logging.info("Max models per turn set to: %d", max_models)
            except ValueError:
                logging.warning("Invalid number. Usage: /k 3")
            continue
        if low in ("/all", "all"):
            max_models = 0  # 0 or None → use all
            forced_pair = None
            logging.info("All models will be used per turn.")
            continue
        if low in ("/both", "both"):
            parts = low.split()
            if len(parts) == 1:
                show_all = not show_all
            else:
                val = parts[1]
                if val in ("on","true","1"):
                    show_all = True
                elif val in ("off","false","0"):
                    show_all = False
                else:
                    logging.warning("Usage: /both [on|off]")
                    continue
            logging.info("Show all candidates: %s", show_all)
            continue
        if low.startswith("/style "):
            _, _, style = user_prompt.partition(" ")
            style = style.strip().lower()
            if style in {"short", "medium", "long"}:
                answer_style = style
                logging.info("Answer style set to: %s", answer_style)
            else:
                logging.warning("Invalid style. Use: /style short | medium | long")
            continue
        if low.startswith("/strict"):
            # Forms supported:
            #   /strict (toggle)
            #   /strict on
            #   /strict off
            parts = low.split()
            if len(parts) == 1:
                strict_mode = not strict_mode
            else:
                val = parts[1]
                if val in ("on", "true", "1"):
                    strict_mode = True
                elif val in ("off", "false", "0"):
                    strict_mode = False
                else:
                    logging.warning("Usage: /strict [on|off]")
                    continue
            logging.info("Strict mode %s. %s", 
                         "ENABLED" if strict_mode else "DISABLED", 
                         ("Answers will be grounded strictly in training data." if strict_mode else "Answers may generalize with controlled creativity."))
            continue
        # UNIVERSAL PROMPT ANALYSIS AND GENERATION

        # === ADVANCED MULTI-DOMAIN PROMPT LOGIC ===
        context = analyze_query(user_prompt)
        context.system_prompt = generate_system_prompt(context=context, mode="compact")
        context.greeting = prepend_tone_greeting("", context.tone)
        # Simulate model response (replace with actual model call if available)
        simulated_response = f"[Simulated] {context.system_prompt}"
        context.quality_scores = evaluate_response(user_prompt, simulated_response, context)
        context.system_prompt = update_prompt_with_feedback(context, context.quality_scores)
        output_json = synthesize_output_json(context)
        # logging.info("\n[ADVANCED MULTI-DOMAIN OUTPUT]")
        
        # SHOW on debugging time uncomment
        # logging.info(output_json)

        # Original universal prompt analysis (legacy)
        prompt_analysis = universal_prompt_analysis_and_generation(user_prompt)
        # logging.info("Model Response Accuracy (quality_scores): %s", context.quality_scores)

        # debugging time uncomment
        # logging.info("\n[UNIVERSAL PROMPT ANALYSIS]")
        # logging.info(json.dumps(prompt_analysis, indent=2, ensure_ascii=False))

        try:
            # Choose N models for ensemble (N = max_models or all)
            chosen = pick_models_for_ensemble(model_registry, forced_keys=forced_pair, max_models=max_models)
            if len(chosen) < 1:
                print("Error: No fine-tuned model IDs available. Please check your model selection or registry.")
                logging.error("No fine-tuned model IDs available.")
                continue
            if len(chosen) > 4:
                logging.warning("Using %d models this turn. Cost and latency will increase.", len(chosen))

            best, candidates = ensemble_ask(chosen, user_prompt, answer_style=answer_style, strict_mode=strict_mode)

            # logging.info("\n [%s]:\n%s\n%s", candidates[0][0], "-" * 80, best["response"])
            logging.info("\n \n%s\n%s",  "-" * 80, best["response"])
            logging.info("=" * 80 + "\n")

            if show_all and len(candidates) > 1:
                for key, mid, result, sc in candidates[1:]:
                    logging.info("ALT RESPONSE [%s] (score=%.3f):\n%s\n%s\n", key, sc, "-" * 80, result.get("response", ""))
        except Exception as e:
            print(f"Error: {e}")
            logging.error("Error: %s", e)

        # Log each chatbot turn and model analysis
        if 'candidates' in locals():
            log_chatbot_turn(user_prompt, [c[2]["response"] for c in candidates], context.quality_scores)

def log_chatbot_turn(user_query, model_responses, quality_scores):
    """Log each chatbot turn and model analysis."""
    out_path = os.path.join(ANALYSIS_DIR, "chatbot_logs.jsonl")
    log_entry = {
        "user_query": user_query,
        "model_responses": model_responses,
        "quality_scores": quality_scores
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# Create analysis directory if it doesn't exist
ANALYSIS_DIR = "analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

if __name__ == "__main__":
    main()