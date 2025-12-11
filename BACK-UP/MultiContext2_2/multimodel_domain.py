import os
import time
import logging
import openai
import json
from dotenv import load_dotenv
import re
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect

# Import shared configuration and functions to eliminate duplication
from shared_config import (
    sia,
    TOPIC_TONE_MAP,
    CONTEXTS,
    analyze_content,
    advanced_topic_detection,
    nuanced_tone_detection,
    persona_expansion,
    advanced_system_prompt_generator,
    build_system_prompt,
    evaluate_response,
    update_prompt_with_feedback,
    synthesize_output_json,
    analyze_query,
    prepend_tone_greeting,
    generate_merged_system_prompt
    
)

# Fine-tuning configuration
# TRAINING_FILE_PATH = "FineTuning/domain_tone_intent_general_training.jsonl"
# TRAINING_FILE_PATH = "FineTuning/rentomojodesk_freshdesk_com.jsonl"
TRAINING_FILE_PATH = "FineTuning/aws_amazon_com.jsonl"

BASE_MODEL = "gpt-3.5-turbo"

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
    nl.setLevel(logging.WARNING)  # hide INFO like: HTTP Request: POST ...
    nl.propagate = False

# ========== ADVANCED DICTIONARIES FROM app2.py ==========


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

def detect_tone_from_prompt(user_prompt):
    """
    Detect the tone of the user's prompt (wrapper for nuanced_tone_detection).
    """
    return nuanced_tone_detection(user_prompt)

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

# ========== ENHANCEMENT 7: DOMAIN × TONE × INTENT COMPOSER ==========
# def compose_domain_tone_intent_prompt(domain, tone, intent):
#     """
#     Generate system prompt based on Domain × Tone × Intent combination.
#     This is the core of the company's requirement: adapt to all three dimensions.
#     """
#     domain_instructions = {
#         "hospitality": "You are hospitality expert. Speak the guest's language: comfort, convenience, and care.",
#         "retail": "You are a retail specialist. Focus on helping customers find and understand products.",
#         "gaming": "You are a gaming enthusiast. Keep it fun, exciting, and engaging!",
#         "finance": "You are a finance professional. Use business language: ROI, costs, efficiency, revenue.",
#         "healthcare": "You are a healthcare advisor. Prioritize patient wellbeing and clear explanations.",
#         "education": "You are an educator. Encourage learning and explain concepts clearly.",
#         "technology": "You are a tech expert. Use technical terminology and provide implementation details.",
#         "corporate": "You are a business professional. Focus on efficiency, strategy, and results."
#     }
    
#     tone_behaviors = {
#         "casual": "Use informal, relaxed language. Be like a friend.",
#         "professional": "Be formal, authoritative, and business-like.",
#         "quirky": "Be playful, fun, and entertaining.",
#         "formal": "Use structured, academic language.",
#         "empathetic": "Show understanding and compassion.",
#         "technical": "Use precise, detailed technical language.",
#         "welcoming": "Be warm and inviting.",
#         "helpful": "Be supportive and solution-focused."
#     }
    
#     intent_behaviors = {
#         "exploration": "Present options and possibilities. Help the user discover what's available.",
#         "information": "Provide comprehensive details, comparisons, and step-by-step explanations.",
#         "conversion": "Focus on value proposition and call-to-action. Guide toward decision.",
#         "support": "Prioritize problem resolution. Provide clear steps and solutions."
#     }
    
#     domain_msg = domain_instructions.get(domain, domain_instructions.get("general", "You are a helpful assistant."))
#     tone_msg = tone_behaviors.get(tone, "Be helpful and professional.")
#     intent_msg = intent_behaviors.get(intent, "Provide helpful information.")

#     # Fix: persona_msg and context_summary
#     # If called from validate_fine_tuned_model, pass persona/context as extra kwargs or default
#     import inspect
#     frame = inspect.currentframe().f_back
#     persona_msg = getattr(frame, 'persona_msg', None)
#     context_summary = getattr(frame, 'context_summary', None)
#     if persona_msg is None:
#         persona_msg = "helpful assistant"
#     if context_summary is None:
#         context_summary = "(no context summary)"

#     # Use the shared_config.py version for prompt generation
#     from shared_config import generate_merged_system_prompt
#     # Compose persona and context as needed
#     persona_msg = persona_msg if 'persona_msg' in locals() else "helpful assistant"
#     context_summary = context_summary if 'context_summary' in locals() else "(no context summary)"
#     return generate_merged_system_prompt(domain, tone, intent, persona_msg, "normal", "millennial", instructions=True)

# def generate_adaptive_system_prompt(tone, intent, domain):
#     """
#     Generate a dynamic system prompt based on detected tone, intent, and domain.
#     This is the fallback/simple version. For FAQ responses, use advanced_system_prompt_generator.
#     """
#     domain_instructions = {
#         "technical": "Provide detailed technical information with clear explanations and examples.",
#         "sales": "Help the user understand value, benefits, and pricing. Use business language.",
#         "booking": "Assist with scheduling, availability, and reservation management. Be efficient.",
#         "support": "Provide troubleshooting steps and solutions to resolve issues quickly.",
#         "product": "Explain features, benefits, and capabilities in an easy-to-understand manner."
#     }
    
#     tone_instructions = {
#         "casual": "Use informal, friendly language. Be relaxed and approachable.",
#         "friendly": "Be warm and welcoming. Make the user feel valued.",
#         "formal": "Use professional, structured language. Be precise and clear.",
#         "professional": "Maintain a business-like tone. Be concise and authoritative.",
#         "enthusiastic": "Show excitement and energy. Use positive language.",
#         "empathetic": "Show understanding and compassion. Acknowledge user concerns.",
#         "urgent": "Be direct and action-oriented. Provide quick solutions.",
#         "polite": "Be courteous and respectful. Use please and thank you.",
#         "neutral": "Be objective and factual. Avoid overly emotional language.",
#         "positive": "Use uplifting and encouraging language.",
#         "negative": "Acknowledge concerns but offer constructive solutions."
#     }
    
#     intent_instructions = {
#         "exploration": "Show options and possibilities. Help the user discover what's available.",
#         "information": "Provide comprehensive details, comparisons, and explanations.",
#         "conversion": "Focus on value proposition and call-to-action. Guide toward decision.",
#         "support": "Prioritize problem resolution. Provide step-by-step guidance and solutions."
#     }
    
#     domain_msg = domain_instructions.get(domain, domain_instructions["product"])
#     tone_msg = tone_instructions.get(tone, "Be helpful and professional.")
#     intent_msg = intent_instructions.get(intent, "Provide helpful information.")
    
#     system_prompt = f"""You are an adaptive customer support assistant with the following characteristics:

# **Domain Expertise:** {domain_msg}
# **Communication Style:** {tone_msg}
# **User Intent:** {intent_msg}

# **Your Response Should:**
# - Address the user's specific need based on their intent
# - Match the detected tone of the conversation
# - Provide relevant information for the {domain} domain
# - Be concise, clear, and actionable
# - Build trust and provide value

# Always prioritize the user's satisfaction and needs.
# """
    
#     return system_prompt

# prepend_tone_greeting() imported from shared_config.py
# No need to redefine - uses TONE_GREETINGS from shared config

def validate_jsonl_structure(file_path):
    """
    Validate that the JSONL file has the expected structure for fine-tuning.
    Returns True if valid, False otherwise.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        logging.error(f"Line {i}: Not a JSON object.")
                        return False
                    if "messages" not in data or not isinstance(data["messages"], list):
                        logging.error(f"Line {i}: Missing or invalid 'messages' field.")
                        return False
                    roles = [msg.get("role") for msg in data["messages"]]
                    if "system" not in roles or "user" not in roles or "assistant" not in roles:
                        logging.error(f"Line {i}: Missing required roles in 'messages'.")
                        return False
                except Exception as e:
                    logging.error(f"Line {i}: Invalid JSON or structure. Error: {e}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Failed to read or validate JSONL file: {e}")
        return False
    
def upload_training_file(file_path):
    """
    Upload the training file to OpenAI for fine-tuning using the updated API.
    Validates JSONL structure before upload.

    Args:
        file_path (str): Path to the training file.

    Returns:
        str: The file ID of the uploaded file.
    """
    if not validate_jsonl_structure(file_path):
        raise ValueError("Training file failed JSONL structure validation. Aborting upload.")
    try:
        with open(file_path, "rb") as f:
            response = openai.files.create(
                file=f,
                purpose="fine-tune"
            )
        logging.info(f"Uploaded training file. File ID: {response.id}")
        return response.id
    except Exception as e:
        logging.error(f"Failed to upload training file: {e}")
        raise

def start_fine_tuning(file_id, base_model):
    """
    Start the fine-tuning process using the updated API.

    Args:
        file_id (str): The file ID of the uploaded training file.
        base_model (str): The base model to fine-tune.

    Returns:
        str: The fine-tuning job ID.
    """
    try:
        response = openai.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model
        )
        logging.info(f"Fine-tuning started. Job ID: {response.id}")
        return response.id
    except Exception as e:
        logging.error(f"Failed to start fine-tuning: {e}")
        raise

def monitor_fine_tuning(job_id):
    """
    Monitor the fine-tuning process until completion using the updated API.

    Args:
        job_id (str): The fine-tuning job ID.

    Returns:
        str: The fine-tuned model ID.
    """
    try:
        while True:
            job_status = openai.fine_tuning.jobs.retrieve(job_id)
            status = job_status.status
            logging.info(f"Fine-tuning status: {status}")
            if status in ["succeeded", "failed", "cancelled"]:
                break
            time.sleep(10)

        if status != "succeeded":
            logging.error(f"Fine-tuning failed or was cancelled. Final status: {status}")
            raise RuntimeError("Fine-tuning did not succeed.")

        logging.info(f"Fine-tuning complete. Model ID: {job_status.fine_tuned_model}")
        return job_status.fine_tuned_model
    except Exception as e:
        logging.error(f"Error monitoring fine-tuning: {e}")
        raise

def validate_fine_tuned_model(model_id, user_prompt, platform="corporate_website", touchpoint="support_page"):
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
        tone = nuanced_tone_detection(user_prompt)
        intent = detect_intent_from_prompt(user_prompt)
        domain = detect_domain_from_prompt(user_prompt)
        
        # ============ NEW ENHANCEMENTS (6 additional dimensions) ============
        # STEP 3: Industry/domain detection
        industry, industry_data = detect_industry_domain(user_prompt)
        
        # STEP 4: Platform-aware tone detection
        platform_tone, platform_data = detect_platform_context(platform)
        
        # STEP 5: Touchpoint-aware intent detection
        touchpoint_intent, touchpoint_data = detect_touchpoint_context(touchpoint)
         
        # ============ COMPOSITE SYSTEM PROMPT GENERATION ============
        # STEP 6: Create Domain × Tone × Intent system prompt
        # Use the composite function to combine all dimensions
        composite_system_prompt = compose_domain_tone_intent_prompt(industry, platform_tone, touchpoint_intent)
        
        # STEP 7: FALLBACK - Also generate traditional advanced prompt for hybrid approach
        keywords_list = list(keywords)
        context = domain  # Use detected domain as context
        
        synthetic_qa_context = {
            "question": user_prompt,
            "answer": "",  # Empty for now - we're just using for system prompt generation
            "context": context
        }
        
        advanced_system_prompt = advanced_system_prompt_generator(
            synthetic_qa_context["question"],
            "",  # Empty answer - just for topic/persona detection
            context)
        
        # Use the composite prompt as primary (if available) or fallback to advanced
        final_system_prompt = composite_system_prompt if composite_system_prompt else advanced_system_prompt
        
        # STEP 8: Call the model with the final system prompt
        response = openai.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        model_response = response.choices[0].message.content
        
        # ============ COMPREHENSIVE ANALYSIS RETURN ============
        # STEP 9: Return response with FULL 9-DIMENSIONAL analysis for transparency
        return {
            "response": model_response,
            "analysis": {
                # Original 3 dimensions
                "tone": tone,
                "intent": intent,
                "domain": domain,
                
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
            "system_prompt_source": "Domain×Tone×Intent Composite"
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
ENSEMBLE_MAX_MODELS = int(os.getenv("ENSEMBLE_MAX_MODELS", "2"))

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

def ensemble_ask(models, user_prompt, platform="corporate_website", touchpoint="support_page"):
    """
    Call multiple models (in parallel), score their answers, return best and all candidates.
    """
    candidates = []

    def _call_one(key_mid):
        key, model_id = key_mid
        r = validate_fine_tuned_model(model_id, user_prompt, platform=platform, touchpoint=touchpoint)
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

    # Ensure training file exists
    if not os.path.exists(TRAINING_FILE_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE_PATH}")

    # Load all available fine-tuned models (FineTuning/*.txt)
    model_registry = load_model_registry(folder=os.path.dirname(TRAINING_FILE_PATH) or "FineTuning")
    # Ensure current training file's model is present (train if missing)
    model_id_file = os.path.splitext(TRAINING_FILE_PATH)[0] + ".txt"
    model_id = None
    if os.path.exists(model_id_file):
        with open(model_id_file, "r", encoding="utf-8") as f:
            model_id = f.read().strip()
        key = os.path.splitext(os.path.basename(TRAINING_FILE_PATH))[0]
        if model_id:
            model_registry.setdefault(key, model_id)
        logging.info("Found existing model ID for this training file: %s", model_id)
    else:
        file_id = upload_training_file(TRAINING_FILE_PATH)
        job_id = start_fine_tuning(file_id, BASE_MODEL)
        model_id = monitor_fine_tuning(job_id)
        save_model_id(model_id, TRAINING_FILE_PATH)
        key = os.path.splitext(os.path.basename(TRAINING_FILE_PATH))[0]
        model_registry[key] = model_id
        logging.info("Model trained and ID saved for future use: %s", model_id)

    logging.info("\n" + "=" * 80)
    logging.info("ADAPTIVE CHAT SUPPORT (Ensemble: use 2 models per turn)")
    logging.info("Models available: %s", ", ".join(sorted(model_registry.keys())) or "(none)")
    logging.info("Commands: /models, /use <key1,key2,...>, /k <N>, /all, /both, /exit")
    logging.info("=" * 80 + "\n")

    show_all = False
    forced_pair = None
    max_models = ENSEMBLE_MAX_MODELS

    while True:
        user_prompt = input("You: ").strip()
        if not user_prompt:
            logging.info("Please enter a message.\n")
            continue

        low = user_prompt.lower()
        if low in ("/exit", "exit", "quit"):
            logging.info("Exiting chatbot. Thank you!")
            break
        if low in ("/models", "models"):
            logging.info("Models: %s", ", ".join(sorted(model_registry.keys())))
            continue
        if low.startswith("/use "):
            _, _, keys = user_prompt.partition(" ")
            forced_pair = [k.strip() for k in keys.split(",") if k.strip()]
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
            show_all = not show_all
            logging.info("Show all candidates: %s", show_all)
            continue

        # UNIVERSAL PROMPT ANALYSIS AND GENERATION

        # === ADVANCED MULTI-DOMAIN PROMPT LOGIC ===
        context = analyze_query(user_prompt)
        context.system_prompt = build_system_prompt(context)
        context.greeting = prepend_tone_greeting("", context.tone)
        # Simulate model response (replace with actual model call if available)
        simulated_response = f"[Simulated] {context.system_prompt}"
        context.quality_scores = evaluate_response(user_prompt, simulated_response, context)
        context.system_prompt = update_prompt_with_feedback(context, context.quality_scores)
        output_json = synthesize_output_json(context)
        logging.info("\n[ADVANCED MULTI-DOMAIN OUTPUT]")
        
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
                raise RuntimeError("No fine-tuned model IDs available.")
            if len(chosen) > 4:
                logging.warning("Using %d models this turn. Cost and latency will increase.", len(chosen))

            best, candidates = ensemble_ask(chosen, user_prompt)

            logging.info("\nBEST RESPONSE [%s]:\n%s\n%s", candidates[0][0], "-" * 80, best["response"])
            logging.info("=" * 80 + "\n")

            if show_all and len(candidates) > 1:
                for key, mid, result, sc in candidates[1:]:
                    logging.info("ALT RESPONSE [%s] (score=%.3f):\n%s\n%s\n", key, sc, "-" * 80, result.get("response", ""))
        except Exception as e:
            logging.error("Error: %s", e)

        # Log each chatbot turn and model analysis
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
# Persistent domain selection for chatbot
current_domain = None
def auto_detect_domain(user_input):
    for domain, data in INDUSTRY_DOMAINS.items():
        if any(kw in user_input.lower() for kw in data["keywords"]):
            return domain
    return "general"

def get_domain_settings(domain):
    data = INDUSTRY_DOMAINS.get(domain, {})
    tone = data.get("tone", "neutral")
    intent = "information"
    context = domain
    persona = [f"{domain} expert"] if domain in INDUSTRY_DOMAINS else ["helpful assistant"]
    return tone, intent, context, persona

# Integrate into main chatbot loop
def chatbot_loop():
    global current_domain
    print("Welcome to MultiContext Chatbot!")
    print("Type /<domain> to set domain (e.g., /finance). Type your query to auto-detect domain.")
    print("Type /age <group> to set age group (e.g., /age gen z). Type /exit or quit to leave.\n")
    explicit_age_group = None
    current_max_tokens = 2048
    current_temperature = 0.7
    logging.info("Type /max_tokens <N> to set max tokens, /temp <T> to set temperature.")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "/exit"]:
                print("Goodbye!")
                break
            if user_input.lower() in ["/show", "show"]:
                domain = current_domain if current_domain else "(auto-detect next query)"
                age_group = explicit_age_group if explicit_age_group else "(auto-detect next query)"
                logging.info("\n--- Current Chatbot Settings ---")
                logging.info(f"Domain: {domain}")
                logging.info(f"Age Group: {age_group}")
                logging.info(f"Tone: (auto-detect per query)")
                logging.info(f"Persona: (auto-detect per query)")
                logging.info(f"Energy: (auto-detect per query)")
                logging.info(f"Max Tokens: {current_max_tokens}")
                logging.info(f"Temperature: {current_temperature}")
                logging.info("Instructions: Type /<domain> to set domain, /age <group> to set age group, /show to view settings, /age_group to list age groups, /max_tokens <N>, /temp <T>, /exit to quit.")
                logging.info("-------------------------------\n")
                continue
            if user_input.lower() in ["/age_group", "age_group"]:
                logging.info("\n--- Available Age Groups ---")
                logging.info("gen alpha   : Children/Teens (born ~2010+, modern slang, energetic, brain-rot style)")
                logging.info("gen z       : Young Adults (born ~1997-2012, casual, energetic, modern slang)")
                logging.info("millennial  : Adults (born ~1981-1996, friendly, professional)")
                logging.info("gen x       : Middle-aged (born ~1965-1980, professional, formal)")
                logging.info("boomer      : Seniors (born ~1946-1964, formal, structured)")
                logging.info("---------------------------\n")
                continue
            if user_input.startswith("/age "):
                explicit_age_group = user_input[5:].strip().lower()
                logging.info(f"Age group set to '{explicit_age_group}'. All future queries will use this age group until changed.")
                continue
            if user_input.startswith("/max_tokens "):
                try:
                    current_max_tokens = int(user_input.split(" ")[1])
                    print(f"Max tokens set to {current_max_tokens}.")
                except Exception:
                    print("Usage: /max_tokens <N>")
                continue
            if user_input.startswith("/temp "):
                try:
                    current_temperature = float(user_input.split(" ")[1])
                    print(f"Temperature set to {current_temperature}.")
                except Exception:
                    print("Usage: /temp <T>")
                continue
            if user_input.startswith("/"):
                domain_cmd = user_input[1:].strip().lower()
                if domain_cmd in INDUSTRY_DOMAINS:
                    current_domain = domain_cmd
                    logging.info(f"Domain set to '{domain_cmd}'. All future queries will use this domain until changed.")
                    continue
                else:
                    logging.info(f"Unknown domain '{domain_cmd}'. Please choose a valid domain.")
                    continue

            domain = current_domain if current_domain else auto_detect_domain(user_input)
            tone, intent, context, persona = get_domain_settings(domain)

            alpha_slang = ["slay", "rizz", "sus", "no cap", "bet", "skibidi", "gyatt", "sigma", "brainrot", "npc", "drip", "yeet"]
            genz_slang = ["fam", "vibe", "lit", "savage", "flex", "ghost", "stan", "lowkey", "highkey", "mood", "snatched"]
            millennial_slang = ["adulting", "cringe", "hangry", "binge", "side hustle", "doggo", "selfie", "fomo", "tbh", "basic"]
            genx_slang = ["rad", "chill", "dude", "whatever", "yuppie", "gnarly", "poser", "phat", "wicked"]
            boomer_slang = ["groovy", "far out", "cool cat", "hip", "square", "right on", "dig it", "outta sight"]

            user_lower = user_input.lower()
            if explicit_age_group:
                age_group = explicit_age_group
            else:
                age_group = "millennial"  # default
                if any(word in user_lower for word in alpha_slang):
                    age_group = "gen alpha"
                elif any(word in user_lower for word in genz_slang):
                    age_group = "gen z"
                elif any(word in user_lower for word in millennial_slang):
                    age_group = "millennial"
                elif any(word in user_lower for word in genx_slang):
                    age_group = "gen x"
                elif any(word in user_lower for word in boomer_slang):
                    age_group = "boomer"

            if age_group == "gen alpha":
                persona.append("Gen Alpha slang user")
                tone = "brain-rot, energetic, modern"
            elif age_group == "gen z":
                persona.append("Gen Z slang user")
                tone = "casual, energetic, modern"
            elif age_group == "millennial":
                persona.append("Millennial")
                tone = "friendly, professional"
            elif age_group == "gen x":
                persona.append("Gen X")
                tone = "professional, formal"
            elif age_group == "boomer":
                persona.append("Boomer")
                tone = "formal, structured"

            energy_words = ["!", "OMG", "wow", "awesome", "lit", "hype", "slay", "rizz", "savage", "flex", "yeet"]
            energy = "high" if any(w in user_input for w in energy_words) or user_input.isupper() else "normal"
            if energy == "high":
                tone += ", enthusiastic"

            system_prompt = generate_combined_system_prompt(
                domain=domain,
                tone=tone,
                intent=intent,
                persona=persona,
                energy=energy,
                age_group=age_group,
                instructions=True
            )

            # Simulate OpenAI API call with expanded parameters (for real use, replace with actual API call)
            # Example:
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
            #     max_tokens=current_max_tokens,
            #     temperature=current_temperature,
            #     top_p=1.0,
            #     presence_penalty=0.2,
            #     frequency_penalty=0.1
            # )
            # model_response = response.choices[0].message.content
            # tokens_used = response.usage.total_tokens

            simulated_response = f"[Simulated] {system_prompt}\nResponse: (half-page, {tone}, {intent}, {domain}, {persona})"
            tokens_used = len(simulated_response.split())

            logging.info("\n[ADVANCED MULTI-DOMAIN OUTPUT]")
            logging.info(f"Response: {simulated_response}")
            logging.info(f"Tokens used: {tokens_used}")
            logging.info(f"Tone: {tone}")
            logging.info(f"Intent: {intent}")
            logging.info(f"Domain: {domain}")
            logging.info(f"Persona: {persona}")
            logging.info(f"Energy: {energy}")
            logging.info(f"Age Group: {age_group}")
            logging.info(f"Max Tokens: {current_max_tokens}")
            logging.info(f"Temperature: {current_temperature}")

            # ...existing code for evaluation and feedback...

        except KeyboardInterrupt:
            logging.info("\nSession ended.")
            break 

def generate_shared_system_prompt(domain, tone, intent, persona=None, energy="normal", age_group="millennial", instructions=True):
    """
    Call the shared_config.py version of the merged system prompt generator.
    """
    from shared_config import generate_merged_system_prompt as shared_generate_merged_system_prompt
    return shared_generate_merged_system_prompt(domain, tone, intent, persona, energy, age_group, instructions)

def generate_combined_system_prompt(domain, tone, intent, persona=None, energy="normal", age_group="millennial", instructions=True):
    """
    Combine both local and shared system prompt outputs for maximum context.
    """
    local_prompt = generate_merged_system_prompt(domain, tone, intent, persona, energy, age_group, instructions)
    shared_prompt = generate_shared_system_prompt(domain, tone, intent, persona, energy, age_group, instructions)
    return "[LOCAL]\n" + local_prompt + "\n\n[SHARED]\n" + shared_prompt

# `def generate_merged_system_prompt(domain, tone, intent, persona=None, energy="normal", age_group="millennial", instructions=True):
#     """
#     Generate a unified system prompt including all extracted context and guidelines.
#     Merges advanced and basic prompt logic for a single, perfect output.
#     """
#     # Basic fallback instructions
#     domain_instructions = {
#         "technical": "Provide detailed technical information with clear explanations and examples.",
#         "sales": "Help the user understand value, benefits, and pricing. Use business language.",
#         "booking": "Assist with scheduling, availability, and reservation management. Be efficient.",
#         "support": "Provide troubleshooting steps and solutions to resolve issues quickly.",
#         "product": "Explain features, benefits, and capabilities in an easy-to-understand manner."
#     }
#     tone_instructions = {
#         "casual": "Use informal, friendly language. Be relaxed and approachable.",
#         "friendly": "Be warm and welcoming. Make the user feel valued.",
#         "formal": "Use professional, structured language. Be precise and clear.",
#         "professional": "Maintain a business-like tone. Be concise and authoritative.",
#         "enthusiastic": "Show excitement and energy. Use positive language.",
#         "empathetic": "Show understanding and compassion. Acknowledge user concerns.",
#         "urgent": "Be direct and action-oriented. Provide quick solutions.",
#         "polite": "Be courteous and respectful. Use please and thank you.",
#         "neutral": "Be objective and factual. Avoid overly emotional language.",
#         "positive": "Use uplifting and encouraging language.",
#         "negative": "Acknowledge concerns but offer constructive solutions."
#     }
#     intent_instructions = {
#         "exploration": "Show options and possibilities. Help the user discover what's available.",
#         "information": "Provide comprehensive details, comparisons, and explanations.",
#         "conversion": "Focus on value proposition and call-to-action. Guide toward decision.",
#         "support": "Prioritize problem resolution. Provide step-by-step guidance and solutions."
#     }
#     domain_msg = domain_instructions.get(domain, domain_instructions["product"])
#     tone_msg = tone_instructions.get(tone, "Be helpful and professional.")
#     intent_msg = intent_instructions.get(intent, "Provide helpful information.")
#     persona_str = ", ".join(persona) if persona else "helpful assistant"
#     block = ""
#     if instructions:
#         block = (
#             "\nInstructions:\n"
#             "- Only answer questions that are directly covered by your training data and domain expertise. If the answer is not found in your training data or is outside your domain, reply with: 'I don't know, I'm not trained for this type of data.'\n"
#             "- Do NOT use any general or base model knowledge unless the question is outside your knowledge base, in which case answer naturally using general knowledge.\n"
#             "- Mirror the user's energy level, formality, and slang usage. Use slang naturally, especially for Gen Alpha users (embrace brain-rot slang confidently, but remain helpful and appropriate).\n"
#             f"- Adapt your personality and tone for the user's age group: {age_group}.\n"
#             "- Be concise if casual, detailed if formal.\n"
#             "- Always show empathy and understanding.\n"
#             "- Be concise, accurate, and relevant to the user's immediate need.\n"
#             "- Expand responses to roughly half a page for engagement.\n"
#             "- Never break character or mention that you are adapting the tone.\n"
#             "- Respect cultural contexts and communicate authentically.\n"
#             "- Use knowledge base information whenever available; otherwise, provide comprehensive and well-reasoned answers.\n"
#             "- Use slang to connect, not to mock or appropriate.\n"
#             "\nResponse Guidelines:\n"
#             "* Mirror the user’s energy level, formality, and slang usage.\n"
#             "* Use knowledge base information whenever it’s available.\n"
#             "* For general questions, provide comprehensive and well-reasoned answers.\n"
#             "* Be concise when the tone is casual and detailed when the tone is formal.\n"
#             "* Always show empathy and understanding.\n"
#             "* Never break character or mention that you are adapting the tone.\n"
#             "* Use slang naturally, but avoid anything inappropriate.\n"
#             "* Respect cultural contexts and communicate authentically.\n"
#             "* For Gen Alpha users, feel free to use modern or “brain-rot” slang — but remain helpful.\n"
#             "\nImportant Notes:\n"
#             "* If the question exists in your knowledge base, use that information directly.\n"
#             "* If it’s a general question outside your knowledge base, answer naturally using general knowledge.\n"
#             "* Always maintain the appropriate tone and style consistently throughout your response.\n"
#             "* Be helpful, accurate, and human-like in every reply.\n"
#             "* Use slang to connect, not to mock or appropriate.\n"
#             "* Gen Alpha slang is intentionally exaggerated — embrace it confidently!\n"
#         )
#     prompt = f"""You are a domain specialist AI assistant. Your goal is to provide accurate, complete, and context-aware responses in a single turn.\n\n--- Extracted Context ---\nDomain   : {domain}\nTone     : {tone}\nIntent   : {intent}\nPersona  : {persona_str}\nEnergy   : {energy}\nAge Group: {age_group}\n------------------------\n\n**Domain Expertise:** {domain_msg}\n**Communication Style:** {tone_msg}\n**User Intent:** {intent_msg}\n{block}\n"""
#     return prompt
