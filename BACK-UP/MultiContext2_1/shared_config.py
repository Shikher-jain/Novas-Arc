# ========== ADVANCED MULTI-DOMAIN CHATBOT FUNCTIONS (Copilot) ========== 
def build_system_prompt(context, history=None):
    """
    Dynamically synthesize a multi-layered system prompt from context and history.
    """
    persona = ", ".join(context.persona)
    domain = ", ".join(context.domain)
    tone = context.tone
    intent = context.intent
    operational_context = infer_operational_context(context.domain)
    prompt = (
        f"You are a {persona} operating within the {domain} domain(s).\n"
        f"Your communication tone should be {tone}.\n"
        f"Primary intent: {intent}.\n"
        f"Operational context: {operational_context}.\n"
        "You must:\n"
        "• Understand the query deeply and precisely.\n"
        "• Provide a complete, factually accurate, and context-optimized answer.\n"
        "• Anticipate related needs and pre-empt follow-up questions.\n"
        "• Balance expertise, empathy, and brevity.\n"
        "• Ensure the user receives all relevant information in a single, comprehensive response.\n"
    )
    if not persona or not domain or not tone:
        prompt += "\nYou are a helpful and adaptive assistant providing safe, neutral, and accurate information."
    if history:
        prompt += "\nContextual history: " + " | ".join(history[-3:])
    return prompt

def infer_operational_context(domains):
    domain_map = {
        "support": "support",
        "technical": "technical",
        "sales": "sales",
        "booking": "booking",
        "education": "education",
        "health": "healthcare",
        "travel": "travel",
        "finance": "finance",
        "government": "government",
        "legal": "legal",
        "entertainment": "entertainment",
        "startup": "startup",
        "hospitality": "hospitality",
        "retail": "retail",
        "product": "product",
    }
    for d in domains:
        if d in domain_map:
            return domain_map[d]
    return "general"

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
        "confidence": confidence
    }

def update_prompt_with_feedback(context, scores):
    """
    Refine system prompt using feedback scores.
    """
    prompt = context.system_prompt
    if scores.get("relevance", 1) < 0.5:
        prompt += "\nPlease ensure your answer is highly relevant to the user's intent and domain."
    if scores.get("completeness", 1) < 0.7:
        prompt += "\nMake sure to cover all aspects of the user's query."
    if scores.get("confidence", 1) < 0.5:
        prompt += "\nIf uncertain, state so and provide best possible information."
    return prompt

def synthesize_output_json(context):
    """
    Synthesize the final output as a valid JSON string.
    """
    import json
    obj = {
        "domain": context.domain,
        "intent": context.intent,
        "persona": context.persona,
        "tone": context.tone,
        "context": infer_operational_context(context.domain),
        "system_prompt": context.system_prompt,
        "greeting": context.greeting
    }
    return json.dumps(obj, ensure_ascii=False)
"""
Shared Configuration and Functions for FAQ Extraction and Chatbot Systems
===========================================================================

This module eliminates code duplication between app2.py (FAQ extraction) 
and check_and_tune.py (chatbot/fine-tuning).

Used by:
- app2.py: For FAQ extraction and training data generation
- check_and_tune.py: For chatbot prompts and model fine-tuning

Benefits:
- Single source of truth for shared configurations
- Easier maintenance and updates
- Better code organization
- Reduced file sizes
"""
import re
import random
import logging
import nltk
from dataclasses import dataclass, field
from typing import List, Dict, Any

try:  # Prefer the high-level import, fall back for older NLTK builds
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
except ImportError:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
    except ImportError:
        SentimentIntensityAnalyzer = None  # type: ignore

from nltk.tokenize import word_tokenize

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
except Exception:
    nlp_spacy = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    st_model = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import openai
except Exception:
    openai = None

try:  # Prefer the high-level import, fall back for older NLTK builds
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
except ImportError:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
    except ImportError:
        SentimentIntensityAnalyzer = None  # type: ignore

from nltk.tokenize import word_tokenize

# ========== NLTK SETUP ==========

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize sentiment analyzer (singleton - reused across modules)
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
TOPIC_TONE_MAP = {
    "art": "creative",
    "automotive": "technical",
    "career": "motivational",
    "climate_change": "urgent",
    "college": "formal",
    "cryptocurrency": "analytical",
    "DIY": "creative",
    "education": "encouraging",
    "energy": "sustainable",
    "entertainment": "casual",
    "environment": "sustainable",
    "fashion": "trendy",
    "finance": "professional",
    "fitness": "motivational",
    "food": "enthusiastic",
    "gamer": "casual",
    "gaming": "excited",
    "gardening": "peaceful",
    "government": "official",
    "health": "empathetic",
    "history": "narrative",
    "home_decor": "aesthetic",
    "hospitality": "welcoming",
    "insurance": "reassuring",
    "legal": "authoritative",
    "literature": "expressive",
    "logistics": "precise",
    "manufacturing": "technical",
    "mental_health": "compassionate",
    "music": "passionate",
    "news": "neutral",
    "parenting": "supportive",
    "pets": "affectionate",
    "philosophy": "thoughtful",
    "photography": "artistic",
    "politics": "objective",
    "product": "detailed",
    "psychology": "insightful",
    "real_estate": "detailed",
    "relationships": "understanding",
    "retail": "helpful",
    "science": "analytical",
    "space_exploration": "inspiring",
    "sports": "energetic",
    "startup": "innovative",
    "technology": "informative",
    "technology_trends": "futuristic",
    "transport": "efficient",
    "travel": "adventurous",
    "videography": "cinematic",
    "wildlife": "conservationist"
}

# Unified context configuration
CONTEXTS = {
    "support": {
        "keywords": ["help", "support", "faq", "customer-service"],
        "system_msg": "You are a customer support assistant. Provide clear and helpful answers to customer questions."
    },
    "technical": {
        "keywords": ["api", "developer", "docs", "technical", "implementation"],
        "system_msg": "You are a technical expert. Provide detailed and accurate technical information."
    },
    "sales": {
        "keywords": ["pricing", "purchase", "product", "order", "buy"],
        "system_msg": "You are a sales assistant. Help customers understand products and make informed decisions."
    },
    "booking": {
        "keywords": ["schedule", "appointment", "booking", "reservation"],
        "system_msg": "You are a booking assistant. Help users schedule and manage their appointments efficiently."
    }
}

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
    except:
        keywords = user_message.lower().split()
    
    return sentiment, keywords

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
    matched_topics = [key for key in TOPIC_TONE_MAP if key in keywords]
    
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
    if second_sentiment is None:
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
    prompt_template_map = {
        ("technical expert", "casual"): "Hey! I'm your tech pal. I'll explain things simply and keep it chill.",
        ("technical expert", "enthusiastic"): "Hi! I'm your excited technical expert. Let's dive into this with energy!",
        ("customer support agent", "casual"): "Hi! I'm your support buddy. Let's solve your issue together, no stress.",
        ("customer support agent", "empathetic"): "Hi, I'm here for you. I understand your concern and will help you through this.",
        ("customer support agent", "urgent"): "I'm here to resolve this quickly and efficiently. Let's get this sorted!",
        ("helpful assistant", "casual"): "Hey there! I'm your friendly assistant. Ask me anything and I'll help out in a relaxed, casual way.",
        ("helpful assistant", "enthusiastic"): "Hi! I'm your enthusiastic assistant, excited to help you out! Ask away.",
        ("helpful assistant", "urgent"): "I'm here to help you quickly and efficiently. Let's solve your problem right now!",
        ("sales assistant", "enthusiastic"): "Hi! I'm excited to help you explore our products and find the perfect fit!",
        ("troubleshooter", "empathetic"): "Hi, I'm here to help troubleshoot and solve your issue with care and patience.",
    }
    
    # Topic-specific prompts
    topic_prompt_map = {
        "health": "I'm your caring health assistant. I'll answer with empathy and support for your well-being.",
        "finance": "You are a finance professional. Give precise, trustworthy, and easy-to-understand financial advice.",
        "education": "You are an educator. Explain concepts clearly and encourage learning in a supportive way.",
        "travel": "You are a travel concierge. Offer friendly, adventurous, and helpful travel advice.",
        "legal": "You are a legal advisor. Provide authoritative, clear, and compliant legal information.",
        "hospitality": "You are a hospitality expert. Offer welcoming, attentive, and helpful service advice.",
        "retail": "You are a retail assistant. Provide helpful, friendly, and product-focused answers.",
        "government": "You are a government official. Provide official, clear, and policy-compliant information.",
        "technology": "You are a technology expert. Provide clear, detailed, and accurate technical information.",
        "startup": "You are a startup advisor. Provide innovative, growth-focused, and practical business advice.",
    }
    
    # Generate prompt parts
    prompt_parts = []
    for persona in personas[:3]:  # Limit to 3 personas
        for topic in topics[:2]:  # Limit to 2 topics
            key = (persona, tone)
            if key in prompt_template_map:
                prompt_parts.append(prompt_template_map[key])
            elif topic in topic_prompt_map:
                prompt_parts.append(topic_prompt_map[topic])
            else:
                prompt_parts.append(
                    f"You are a {persona} with a {tone} tone. "
                    f"Your expertise is in {topic}. "
                    f"Provide clear, accurate, and helpful answers to user questions."
                )
    
    # Deduplicate and join
    system_prompt = "\n".join(dict.fromkeys(prompt_parts))
    
    if not system_prompt or len(system_prompt) < 20:
        # Fallback
        system_prompt = CONTEXTS.get(context, {}).get(
            "system_msg", 
            "You are a helpful assistant."
        )
        logging.warning("System prompt generation resulted in short prompt, using fallback")
    
    return system_prompt

# ========== INTENT DETECTION FUNCTION ========== 
def detect_intent(query):
    """
    Detect the user's intent (exploration, information, conversion, support).
    """
    INTENT_KEYWORDS = {
        "exploration": ["what", "tell me", "show me", "list", "options", "available", "latest", "new"],
        "information": ["how", "explain", "details", "information", "why", "when", "where", "compare"],
        "conversion": ["buy", "purchase", "order", "book", "reserve", "subscribe", "sign up"],
        "support": ["help", "issue", "problem", "error", "not working", "broken", "fix", "troubleshoot"]
    }
    prompt_lower = query.lower()
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        intent_scores[intent] = sum(1 for kw in keywords if kw in prompt_lower)
    best_intent = max(intent_scores, key=intent_scores.get)
    return best_intent if intent_scores[best_intent] > 0 else "information"


# ========== COMPLEXITY ESTIMATION FUNCTION ==========
def estimate_complexity(query):
    """
    Estimate the complexity of a user query based on length, vocabulary, and structure.
    Returns: 'low', 'medium', or 'high'
    """
    if not query:
        return "low"
    length = len(query)
    words = query.split()
    unique_words = set(words)
    complexity_score = 0
    if length > 120 or len(words) > 20:
        complexity_score += 1
    if len(unique_words) / (len(words) + 1e-6) < 0.7:
        complexity_score += 1
    if any(w in query.lower() for w in ["and", "or", "if", "then", "else", "not", "except"]):
        complexity_score += 1
    if complexity_score >= 2:
        return "high"
    elif complexity_score == 1:
        return "medium"
    else:
        return "low"

# ========== TONE-BASED GREETINGS ==========

TONE_GREETINGS = {
    "casual": [
        "Hey!", "Hi there!", "Hello!", "Yo!", "Hey buddy!", "Hey friend!", "Hi!", 
        "Hey, how's it going?", "Hey, what's up?", "Hey, glad to help!"
    ],
    "friendly": [
        "Hello friend!", "Hi, great to see you!", "Hey there!", "Welcome!", 
        "Hi, how can I help you today?", "Hey, happy to assist!"
    ],
    "quirky": [
        "Yo yo yo!", "What's cookin'?", "Howdy partner!", "Ahoy!", "Wassup!", "Ready for some fun?"
    ],
    "fun": [
        "Let's get this party started!", "Woohoo!", "Yay!", "Time for some fun!", 
        "Let's do this!", "Excited to help!"
    ],
    "formal": [
        "Greetings.", "Hello.", "Good day.", "Welcome.", "How may I assist you?"
    ],
    "professional": [
        "Hello, how can I assist you?", "Welcome, let me know your query.", 
        "Good day, I'm here to help.", "Thank you for reaching out."
    ],
    "enthusiastic": [
        "Hi! I'm excited to help!", "Hello! Let's get started!", 
        "Woohoo! Let's solve this together!"
    ],
    "empathetic": [
        "Hi, I'm here for you.", "Hello, I understand your concern.", 
        "Hey, I'm here to help you through this."
    ],
    "urgent": [
        "Let's resolve this quickly.", "I'm on it right away!", "Let's get this sorted ASAP!"
    ],
    "negative": [
        "I'm sorry to hear that.", "Let's see how I can help.", 
        "I'll do my best to assist you."
    ],
    "neutral": [
        "Hello.", "Hi.", "Welcome.", "How can I help you?"
    ]
}


def prepend_tone_greeting(response, tone):
    """
    Prepend an appropriate greeting based on tone.
    Args:
        response (str): The response text
        tone (str): The tone (casual, professional, formal, etc.)
    Returns:
        str: Response with prepended greeting
    """
    greetings = TONE_GREETINGS.get(tone, ["Hello."])
    return f"{random.choice(greetings)} {response}"

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


def estimate_complexity(query):
    """
    Estimate the complexity of a user query based on length, number of keywords, and presence of technical terms.
    Returns: 'low', 'medium', or 'high'
    """
    length = len(query)
    keywords = re.findall(r"[a-zA-Z]{4,}", query)
    technical_terms = [kw for kw in keywords if kw.lower() in [
        "api", "integration", "deployment", "database", "server", "framework", "library", "code", "algorithm", "model", "function", "parameter", "variable", "object", "class", "method", "bug", "error", "exception", "performance", "scalability", "optimization", "architecture", "protocol", "endpoint", "token", "authentication", "authorization", "encryption", "compression", "latency", "throughput", "bandwidth", "cloud", "container", "docker", "kubernetes", "microservice", "monolith", "distributed", "concurrent", "parallel", "thread", "process", "memory", "cpu", "disk", "storage", "network", "socket", "port", "firewall", "load balancer", "cache", "queue", "message", "event", "stream", "batch", "pipeline", "etl", "data", "analytics", "visualization", "dashboard", "report", "schema", "table", "row", "column", "index", "key", "value", "json", "xml", "csv", "yaml", "toml", "ini", "config", "settings", "environment", "variable", "secret", "vault", "monitoring", "logging", "alert", "notification", "incident", "ticket", "support", "sla", "uptime", "downtime", "backup", "restore", "snapshot", "replication", "failover", "high availability", "disaster recovery", "security", "compliance", "audit", "policy", "governance", "risk", "threat", "vulnerability", "patch", "update", "upgrade", "release", "version", "branch", "merge", "pull request", "commit", "push", "clone", "fork", "issue", "bug", "feature", "task", "story", "epic", "sprint", "kanban", "scrum", "agile", "waterfall", "devops", "ci", "cd", "pipeline", "test", "unit test", "integration test", "system test", "acceptance test", "regression test", "performance test", "load test", "stress test", "soak test", "smoke test", "sanity test", "mock", "stub", "spy", "assert", "coverage", "lint", "static analysis", "dynamic analysis", "profiling", "benchmark", "trace", "debug", "log", "print", "output", "input", "cli", "gui", "web", "mobile", "desktop", "app", "application", "service", "daemon", "agent", "worker", "scheduler", "cron", "timer", "event loop", "callback", "promise", "future", "async", "await", "thread", "lock", "mutex", "semaphore", "race condition", "deadlock", "starvation", "priority", "queue", "stack", "heap", "tree", "graph", "node", "edge", "vertex", "weight", "path", "cycle", "traversal", "search", "sort", "filter", "map", "reduce", "fold", "zip", "unzip", "split", "join", "concat", "slice", "substring", "replace", "regex", "pattern", "match", "group", "capture", "substitute", "escape", "quote", "unquote", "parse", "serialize", "deserialize", "marshal", "unmarshal", "encode", "decode", "compress", "decompress", "encrypt", "decrypt", "hash", "sign", "verify", "certificate", "ssl", "tls", "http", "https", "ftp", "smtp", "pop3", "imap", "dns", "dhcp", "ip", "tcp", "udp", "icmp", "arp", "mac", "address", "route", "gateway", "proxy", "vpn", "ssh", "telnet", "rlogin", "scp", "sftp", "rsync", "git", "svn", "hg", "bzr", "cvs", "perforce", "jira", "confluence", "trello", "asana", "slack", "teams", "zoom", "meet", "hangouts", "skype", "webex", "gotomeeting", "bluejeans", "join.me", "go", "python", "java", "javascript", "typescript", "c", "c++", "c#", "ruby", "perl", "php", "swift", "kotlin", "scala", "rust", "go", "dart", "elixir", "erlang", "haskell", "clojure", "lisp", "scheme", "prolog", "fortran", "cobol", "assembly", "bash", "sh", "zsh", "fish", "powershell", "cmd", "bat", "make", "cmake", "ninja", "gradle", "maven", "ant", "sbt", "pip", "conda", "virtualenv", "docker", "kubernetes", "terraform", "ansible", "chef", "puppet", "salt", "packer", "vault", "consul", "nomad", "jenkins", "travis", "circleci", "github actions", "gitlab ci", "bitbucket pipelines", "azure pipelines", "aws codebuild", "aws codepipeline", "aws codecommit", "aws codedeploy", "aws codestar", "aws cloudformation", "aws sam", "aws cdk", "gcp cloud build", "gcp cloud deploy", "gcp cloud functions", "gcp cloud run", "gcp cloud storage", "gcp cloud sql", "gcp cloud spanner", "gcp cloud firestore", "gcp cloud pubsub", "gcp cloud tasks", "gcp cloud scheduler", "gcp cloud monitoring", "gcp cloud logging", "gcp cloud trace", "gcp cloud debug", "gcp cloud error reporting", "gcp cloud profiler", "gcp cloud billing", "gcp cloud identity", "gcp cloud iam", "gcp cloud resource manager", "gcp cloud asset inventory", "gcp cloud security", "gcp cloud compliance", "gcp cloud governance", "gcp cloud risk", "gcp cloud threat", "gcp cloud vulnerability", "gcp cloud patch", "gcp cloud update", "gcp cloud upgrade", "gcp cloud release", "gcp cloud version", "gcp cloud branch", "gcp cloud merge", "gcp cloud pull request", "gcp cloud commit", "gcp cloud push", "gcp cloud clone", "gcp cloud fork", "gcp cloud issue", "gcp cloud bug", "gcp cloud feature", "gcp cloud task", "gcp cloud story", "gcp cloud epic", "gcp cloud sprint", "gcp cloud kanban", "gcp cloud scrum", "gcp cloud agile", "gcp cloud waterfall", "gcp cloud devops", "gcp cloud ci", "gcp cloud cd", "gcp cloud pipeline", "gcp cloud test", "gcp cloud unit test", "gcp cloud integration test", "gcp cloud system test", "gcp cloud acceptance test", "gcp cloud regression test", "gcp cloud performance test", "gcp cloud load test", "gcp cloud stress test", "gcp cloud soak test", "gcp cloud smoke test", "gcp cloud sanity test", "gcp cloud mock", "gcp cloud stub", "gcp cloud spy", "gcp cloud assert", "gcp cloud coverage", "gcp cloud lint", "gcp cloud static analysis", "gcp cloud dynamic analysis", "gcp cloud profiling", "gcp cloud benchmark", "gcp cloud trace", "gcp cloud debug", "gcp cloud log", "gcp cloud print", "gcp cloud output", "gcp cloud input", "gcp cloud cli", "gcp cloud gui", "gcp cloud web", "gcp cloud mobile", "gcp cloud desktop", "gcp cloud app", "gcp cloud application", "gcp cloud service", "gcp cloud daemon", "gcp cloud agent", "gcp cloud worker", "gcp cloud scheduler", "gcp cloud cron", "gcp cloud timer", "gcp cloud event loop", "gcp cloud callback", "gcp cloud promise", "gcp cloud future", "gcp cloud async", "gcp cloud await", "gcp cloud thread", "gcp cloud lock", "gcp cloud mutex", "gcp cloud semaphore", "gcp cloud race condition", "gcp cloud deadlock", "gcp cloud starvation", "gcp cloud priority"]]
    num_technical = len(technical_terms)
    if length < 40 and len(keywords) < 5 and num_technical == 0:
        return "low"
    elif length < 120 and len(keywords) < 15 and num_technical < 3:
        return "medium"
    else:
        return "high"
        
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
    if nlp_spacy:
        doc = nlp_spacy(query)
        entities = [ent.text for ent in doc.ents]
    # 6. Complexity estimation
    complexity = estimate_complexity(query)

    # ========== COMPLEXITY ESTIMATION FUNCTION ==========
    
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

def prepend_tone_greeting(response, tone):
    """
    Prepend an appropriate greeting based on tone.
    Args:
        response (str): The response text
        tone (str): The tone (casual, professional, formal, etc.)
    Returns:
        str: Response with prepended greeting
    Example:
        result = prepend_tone_greeting("Here's the answer...", "casual")
        # Returns: "Hey! Here's the answer..."
    Used by: app2.py, check_and_tune.py
    """
    greetings = TONE_GREETINGS.get(tone, ["Hello."])
    return f"{random.choice(greetings)} {response}"

    """
    Example usage for dynamic multi-domain chatbot prompt refinement:

    query = "What are the best mutual funds for long-term returns?"
    context = analyze_query(query)
    print("Initial PromptContext:", context)

    response = "For long-term returns, consider diversified equity mutual funds such as XYZ Growth Fund and ABC Value Fund. Always review past performance and consult a financial advisor."
    scores = evaluate_response(query, response)
    print("Response Quality Scores:", scores)

    refined_prompt = update_prompt_with_feedback(context, scores)
    print("Refined System Prompt:\n", refined_prompt)
    """

    """
    Example usage for advanced multi-domain chatbot logic:

    query = "What are the best mutual funds for long-term returns?"
    context = analyze_query(query)
    print("Initial PromptContext:", context)

    response = "For long-term returns, consider diversified equity mutual funds such as XYZ Growth Fund and ABC Value Fund. Always review past performance and consult a financial advisor."
    scores = evaluate_response(query, response)
    print("Response Quality Scores:", scores)

    refined_prompt = update_prompt_with_feedback(context, scores)
    print("Refined System Prompt:\n", refined_prompt)

    output_json = synthesize_output_json(context)
    print("Output JSON:\n", output_json)
    """