import os
import time
import logging
import openai
import json
from dotenv import load_dotenv
import re
import random

# Import shared configuration and functions to eliminate duplication
from shared_config import (
    sia,
    TOPIC_TONE_MAP,
    CONTEXTS,
    analyze_content,
    advanced_topic_detection,
    nuanced_tone_detection,
    persona_expansion,
    advanced_system_prompt_generator
)

# Fine-tuning configuration
# TRAINING_FILE_PATH = "FineTuning/domain_tone_intent_general_training.jsonl"
TRAINING_FILE_PATH = "FineTuning/rentomojodesk_freshdesk_com.jsonl"
# TRAINING_FILE_PATH = "FineTuning/hospitality_training.jsonl"
# TRAINING_FILE_PATH = "FineTuning/technology_training.jsonl"
# TRAINING_FILE_PATH = "FineTuning/education_training.jsonl"
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
def compose_domain_tone_intent_prompt(domain, tone, intent):
    """
    Generate system prompt based on Domain × Tone × Intent combination.
    This is the core of the company's requirement: adapt to all three dimensions.
    """
    domain_instructions = {
        "hospitality": "You are hospitality expert. Speak the guest's language: comfort, convenience, and care.",
        "retail": "You are a retail specialist. Focus on helping customers find and understand products.",
        "gaming": "You are a gaming enthusiast. Keep it fun, exciting, and engaging!",
        "finance": "You are a finance professional. Use business language: ROI, costs, efficiency, revenue.",
        "healthcare": "You are a healthcare advisor. Prioritize patient wellbeing and clear explanations.",
        "education": "You are an educator. Encourage learning and explain concepts clearly.",
        "technology": "You are a tech expert. Use technical terminology and provide implementation details.",
        "corporate": "You are a business professional. Focus on efficiency, strategy, and results."
    }
    
    tone_behaviors = {
        "casual": "Use informal, relaxed language. Be like a friend.",
        "professional": "Be formal, authoritative, and business-like.",
        "quirky": "Be playful, fun, and entertaining.",
        "formal": "Use structured, academic language.",
        "empathetic": "Show understanding and compassion.",
        "technical": "Use precise, detailed technical language.",
        "welcoming": "Be warm and inviting.",
        "helpful": "Be supportive and solution-focused."
    }
    
    intent_behaviors = {
        "exploration": "Present options and possibilities. Help the user discover what's available.",
        "information": "Provide comprehensive details, comparisons, and step-by-step explanations.",
        "conversion": "Focus on value proposition and call-to-action. Guide toward decision.",
        "support": "Prioritize problem resolution. Provide clear steps and solutions."
    }
    
    domain_msg = domain_instructions.get(domain, domain_instructions.get("general", "You are a helpful assistant."))
    tone_msg = tone_behaviors.get(tone, "Be helpful and professional.")
    intent_msg = intent_behaviors.get(intent, "Provide helpful information.")
    
    system_prompt = f"""You are an adaptive customer support assistant optimized for Domain × Tone × Intent communication.

**DOMAIN (Industry/Business Context):** {domain_msg}

**TONE (Communication Style):** {tone_msg}

**INTENT (User Goal):** {intent_msg}

**Your Response Should:**
- Address the user's specific need based on their intent (what they want to do)
- Match the communication style for their platform/context (how to say it)
- Use language and concepts relevant to their industry (domain expertise)
- Be concise, clear, and actionable
- Build trust and provide maximum value

Remember: Different industries require different terminology. A tech person needs APIs and implementation details. A business person needs ROI and efficiency metrics. Always speak in their language."""
    
    return system_prompt

def generate_adaptive_system_prompt(tone, intent, domain):
    """
    Generate a dynamic system prompt based on detected tone, intent, and domain.
    This is the fallback/simple version. For FAQ responses, use advanced_system_prompt_generator.
    """
    domain_instructions = {
        "technical": "Provide detailed technical information with clear explanations and examples.",
        "sales": "Help the user understand value, benefits, and pricing. Use business language.",
        "booking": "Assist with scheduling, availability, and reservation management. Be efficient.",
        "support": "Provide troubleshooting steps and solutions to resolve issues quickly.",
        "product": "Explain features, benefits, and capabilities in an easy-to-understand manner."
    }
    
    tone_instructions = {
        "casual": "Use informal, friendly language. Be relaxed and approachable.",
        "friendly": "Be warm and welcoming. Make the user feel valued.",
        "formal": "Use professional, structured language. Be precise and clear.",
        "professional": "Maintain a business-like tone. Be concise and authoritative.",
        "enthusiastic": "Show excitement and energy. Use positive language.",
        "empathetic": "Show understanding and compassion. Acknowledge user concerns.",
        "urgent": "Be direct and action-oriented. Provide quick solutions.",
        "polite": "Be courteous and respectful. Use please and thank you.",
        "neutral": "Be objective and factual. Avoid overly emotional language.",
        "positive": "Use uplifting and encouraging language.",
        "negative": "Acknowledge concerns but offer constructive solutions."
    }
    
    intent_instructions = {
        "exploration": "Show options and possibilities. Help the user discover what's available.",
        "information": "Provide comprehensive details, comparisons, and explanations.",
        "conversion": "Focus on value proposition and call-to-action. Guide toward decision.",
        "support": "Prioritize problem resolution. Provide step-by-step guidance and solutions."
    }
    
    domain_msg = domain_instructions.get(domain, domain_instructions["product"])
    tone_msg = tone_instructions.get(tone, "Be helpful and professional.")
    intent_msg = intent_instructions.get(intent, "Provide helpful information.")
    
    system_prompt = f"""You are an adaptive customer support assistant with the following characteristics:

**Domain Expertise:** {domain_msg}

**Communication Style:** {tone_msg}

**User Intent:** {intent_msg}

**Your Response Should:**
- Address the user's specific need based on their intent
- Match the detected tone of the conversation
- Provide relevant information for the {domain} domain
- Be concise, clear, and actionable
- Build trust and provide value

Always prioritize the user's satisfaction and needs."""
    
    return system_prompt
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
            context
        )
        
        # Use the composite prompt as primary (if available) or fallback to advanced
        final_system_prompt = composite_system_prompt if composite_system_prompt else advanced_system_prompt
        
        # STEP 8: Call the model with the final system prompt
        response = openai.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
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

def main():
    # Ensure API key is set
    if not openai.api_key:
        raise RuntimeError("OpenAI API key is not set. Please set it in your environment variables.")

    # Ensure training file exists
    if not os.path.exists(TRAINING_FILE_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE_PATH}")

    # Check if model ID already exists for this training file
    model_id_file = os.path.splitext(TRAINING_FILE_PATH)[0] + ".txt"
    model_id = None
    if os.path.exists(model_id_file):
        with open(model_id_file, "r") as f:
            model_id = f.read().strip()
        logging.info(f"Found existing model ID for this training file: {model_id}")
    else:
        # Upload training file
        file_id = upload_training_file(TRAINING_FILE_PATH)
        # Start fine-tuning
        job_id = start_fine_tuning(file_id, BASE_MODEL)
        # Monitor fine-tuning
        model_id = monitor_fine_tuning(job_id)
        # Save the fine-tuned model ID to a file named after the training file
        save_model_id(model_id, TRAINING_FILE_PATH)
        logging.info(f"Model trained and ID saved for future use: {model_id}")

    # Chatbot loop for model validation with adaptive responses
    logging.info("\n" + "=" * 80)
    logging.info("ADAPTIVE CHAT SUPPORT (Domain × Tone × Intent)")
    logging.info("\n" + "=" * 80)
    logging.info("Type 'exit' to quit the chatbot.\n")
    
    while True:
        user_prompt = input("You: ").strip()
        if user_prompt.lower() == "exit":
            logging.info("Exiting chatbot. Thank you!")
            break
        
        if not user_prompt:
            logging.info("Please enter a message.\n")
            continue
        
        try:
            # Get response with advanced analysis
            result = validate_fine_tuned_model(model_id, user_prompt)
            response = result["response"]
            analysis = result["analysis"]
            system_prompt = result["system_prompt"]
            
            # Display the multi-dimensional analysis
            # print("\n" + "="*80)
            # print(f"🎯 PROMPT ANALYSIS")
            # print("="*80)
            # print(f"  Tone:      {analysis['tone'].upper():20s} | Sentiment: {analysis['sentiment_compound']:+.2f}")
            # print(f"  Intent:    {analysis['intent'].upper():20s} | Keywords:  {', '.join(analysis['keywords'])}")
            # print(f"  Domain:    {analysis['domain'].upper():20s}")
            # print("="*80)
            
            # # Display a snippet of the generated system prompt
            # print(f"\nSYSTEM PROMPT (generated based on analysis):")
            # print("-"*80)
            # print(f"{system_prompt[:200]}..." if len(system_prompt) > 200 else system_prompt)
            # print("-"*80)
             
            # Display the response
            logging.info("\nSUPPORT AGENT RESPONSE:")
            logging.info("-" * 80)
            logging.info("%s", response)
            logging.info("=" * 80 + "\n") 
            
        except Exception as e:
            logging.error("Error: %s", e)

if __name__ == "__main__":
    main()