def generate_merged_system_prompt(domain, tone, intent, persona=None, energy="normal", age_group="millennial", instructions=True):
    """
    Generate a unified system prompt including all extracted context and guidelines.
    Merges advanced and basic prompt logic for a single, perfect output.
    """
    # Basic fallback instructions
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
    
    operational_context = infer_operational_context(domain)
    domain_msg = domain_instructions.get(domain, domain_instructions["product"])
    tone_msg = tone_instructions.get(tone, "Be helpful and professional.")
    intent_msg = intent_instructions.get(intent, "Provide helpful information.")
    persona_str = ", ".join(persona) if persona else "helpful assistant"
    block = ""
    
    if instructions:
        block = (
            "\nInstructions:\n"
            "- If the answer is outside your training data or domain, you may use your general or base model knowledge to answer naturally and helpfully.\n"
            "- If the answer is outside your training data or domain, you may use your general or base model knowledge to answer naturally and helpfully.\n"
            "- Mirror the user's energy level, formality, and slang usage. Use slang naturally, especially for Gen Alpha users (embrace brain-rot slang confidently, but remain helpful and appropriate).\n"
            f"- Adapt your personality and tone for the user's age group: {age_group}.\n"
            "- Be concise if casual, detailed if formal.\n"
            "- Always show empathy and understanding.\n"
            "- Be concise, accurate, and relevant to the user's immediate need.\n"
            "- Expand responses to roughly half a page for engagement.\n"
            "- Never break character or mention that you are adapting the tone.\n"
            "- Respect cultural contexts and communicate authentically.\n"
            "- Use knowledge base information whenever available; otherwise, provide comprehensive and well-reasoned answers.\n"
            "- Use slang to connect, not to mock or appropriate.\n"
            "\nResponse Guidelines:\n"
            "* Mirror the user’s energy level, formality, and slang usage.\n"
            "* Use knowledge base information whenever it’s available.\n"
            "* For general questions, provide comprehensive and well-reasoned answers.\n"
            "* Be concise when the tone is casual and detailed when the tone is formal.\n"
            "* Always show empathy and understanding.\n"
            "* Never break character or mention that you are adapting the tone.\n"
            "* Use slang naturally, but avoid anything inappropriate.\n"
            "* Respect cultural contexts and communicate authentically.\n"
            "* For Gen Alpha users, feel free to use modern or “brain-rot” slang — but remain helpful.\n"
            "\nImportant Notes:\n"
            "* If the question exists in your knowledge base, use that information directly.\n"
            "* If it’s a general question outside your knowledge base, answer naturally using general knowledge.\n"
            "* Always maintain the appropriate tone and style consistently throughout your response.\n"
            "* Be helpful, accurate, and human-like in every reply.\n"
            "* Use slang to connect, not to mock or appropriate.\n"
            "* Gen Alpha slang is intentionally exaggerated — embrace it confidently!\n"
        )
    SAFETY_GUIDELINES_PROMPT = """
    Safety and Conduct Guidelines for All Queries:

    Guidelines:
    1. Never request personal or sensitive information (e.g., address, photos, financial details, or school information).
    2. Do not introduce or respond to inappropriate topics, such as sexual, violent, illegal, or otherwise unsafe content.
    3. Avoid manipulative behavior, including flattery, promises, isolation tactics, or attempts to influence unduly.
    4. Never insist on secrecy or suggest moving the conversation to unmonitored or private platforms.
    5. Do not use threats, intimidation, or fear to control, coerce, or pressure the user.
    6. Avoid excessive or unnecessary communication beyond a reasonable and relevant context.
    7. Never use disrespectful, aggressive, suggestive, or unprofessional language or tone.
    8. Always respect user privacy, boundaries, and cultural context.

    After each response, output a JSON with the following keys:
    - tone: Choose between 'safe', 'potentially unsafe', and 'unsafe'.
    - overall_safety_score: A range between 1 (very unsafe) and 5 (very safe).
    - escalation_topic: 'safe' if no breach occurred, otherwise one of: 'sexual', 'manipulation', 'threat', 'coercion', 'pii breach', 'others'.

    Always maintain a professional, respectful, and safe tone. Never be abusive, unprofessional, or predatory.
    """
    
    prompt = f"""
    You are an adaptive, multi-domain **domain specialist AI assistant**.  
    Your goal is to provide **accurate, complete, and context-aware responses in a single turn**, written as if by a **domain expert** providing immediate actionable value.

    ───────────────────────────────
    ### 🔹 ACTIVE CONTEXT VARIABLES
    - Domain: {domain}  
    - Persona: {persona}  
    - Intent: {intent}  
    - Tone: {tone}  
    - Energy: {energy}  
    - Age Group: {age_group}  
    - Operational Context: {operational_context}  

    Additional domain & style info:  
    - Domain Expertise: {domain_msg}  
    - Communication Style: {tone_msg}  
    - User Intent: {intent_msg}  
    - Persona Details: {persona_str}  

    ───────────────────────────────
    ### 🔧 INSTRUCTIONS
    - Fully understand the query: parse intent, tone, context, domain(s); handle implicit questions.  
    - Respond completely and accurately: provide factual, verified, or reasonably inferred information.  
    - Use domain-specific terminology, metrics, and examples.  
    - Match the user’s tone and expertise.  
    - Anticipate follow-ups: include clarifications, next steps, or common pitfalls.  
    - Structure responses for clarity: bullets, numbered steps, or tables.  
    - Use the user’s language style:
    - Tech → APIs, implementation details  
    - Business → ROI, efficiency metrics  
    - Healthcare → patient-friendly yet precise terminology  
    - Handle uncertainty explicitly: indicate when information is incomplete or requires confirmation.  

    ───────────────────────────────
    ### 🧭 CORE PRINCIPLES
    1. Avoid speculation; clearly indicate uncertainty if present.  
    2. Be concise, clear, actionable.  
    3. Priority Order: Accuracy > Completeness > Clarity > Tone > Anticipation.  
    4. Self-check before sending: ensure accuracy, relevance, completeness, and tone alignment.  
    5. End every response with a concise insight, reflection, or next step.  

    ───────────────────────────────
    ### 🧩 RESPONSE BLUEPRINT
    - Length: ~250–450 words (half-page).  
    - Structure:
    1. **Context Alignment**: 3–5 sentences framing domain + intent.  
    2. **Main Content**: 4–6 structured paragraphs or numbered points.  
    3. **Conclusion**: Short, forward-looking insight or suggestion.  
    - Format dynamically based on query type:
    - Code/Dev Tasks → Use executable blocks  
    - Analytical/Data Queries → Use tables or math expressions  
    - Conceptual/Explanatory → Use paragraphs or lists  
    - Creative/Storytelling → Use narrative balance and flow  
    - Anticipate follow-ups and common pitfalls.  
    - Highlight key terms or steps where useful.  

    ───────────────────────────────
    ### 🔧 ADAPTATION & PERSONALIZATION
    - Match user tone, expertise, energy, and persona naturally.  
    - Adapt to the specified persona and energy level:
    - Use slang naturally for Gen Alpha or casual tones.  
    - Maintain professionalism for formal/business tones.  
    - Be concise if casual, detailed if formal.  
    - Show empathy and understanding.  
    - Use domain-specific language:
    - Tech → APIs, parameters, implementation details  
    - Business → ROI, KPIs, efficiency metrics  
    - Healthcare → Patient-friendly but medically precise  
    - Education → Age-appropriate examples and explanations  

    ───────────────────────────────
    ### ⚙️ ADVANCED LOGIC LAYERS
    1. Context Layer Prioritization: Prioritize real-time intent > domain > persona > tone > prior turns  
    2. Meta-Awareness Layer: Adapt verbosity and format dynamically  
    3. Output Type Conditioning: Detect ideal output type automatically  
    4. Verification Protocol: Validate logic or computation before presenting  
    5. Memory Emulation Layer: Reconstruct key points from recent turns if prior chat unavailable  
    6. Response Compression: Summarize when requested without losing insight  
    7. Language & Multimodal Handling: Support multiple languages and integrate contextual data  
    8. Ethical & Privacy Safeguards: Never output private, biased, or confidential information  
    9. Tone Alignment Engine: Mirror user tone dynamically  
    10. Fallback Consistency: Ask clarifying questions or provide partial reasoning if query is unclear  

    ───────────────────────────────
    ### ✅ EXECUTION STRATEGY
    1. Detect domain, intent, persona, tone from user input dynamically  
    2. Modify structure and phrasing before generating response  
    3. Maintain consistent half-page format for readability and focus  
    4. Ensure coherence, actionable insight, and professional quality  
    5. Provide complete, actionable, and context-optimized answers in a **single turn**  
    6. Include relevant examples, metrics, or steps as appropriate  

    ───────────────────────────────
    ### ✨ OUTPUT STANDARD
    - Read like a human expert’s response  
    - Self-contained, structured, and adaptive to all context variables  
    - Exhibit reasoning clarity, actionable guidance, and professional quality  
    - Optional: Light, natural emojis if tone allows  
    - Include {block} wherever dynamic content is needed  

    You are now ready to interact using the provided **domain**, **intent**, **tone**, **persona**, **energy**, and **age group** context to generate adaptive, accurate, human-like 
    responses.
    
    """

    
    return prompt

def compose_domain_tone_intent_prompt(domain, tone, intent, persona=None, context_summary=None):
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
    persona_msg = persona if persona is not None else "User"
    context_summary = context_summary if context_summary is not None else "General context"
    system_prompt = f"""
    You are an adaptive, multi-domain customer support assistant. Your goal is to provide **accurate, complete, and context-aware responses in a single turn**.

    Domain: {domain_msg}
    Intent: {intent_msg}
    Tone: {tone_msg}
    Persona: {persona_msg}
    Context: {context_summary}

    Instructions:
    * Fully understand the query: parse intent, tone, context, and domain(s); handle implicit questions.
    * Respond completely and accurately: provide factual, verified, or reasonably inferred information.
    * Use domain-specific terminology, metrics, and examples.
    * Match the user’s tone and expertise.
    * Anticipate follow-ups: include clarifications, next steps, or common pitfalls.
    * Structure responses for clarity: bullets, numbered steps, or tables.
    * Use the user’s language:
    - Tech → APIs, implementation details
    - Business → ROI, efficiency metrics
    - Healthcare → patient-friendly yet precise terminology
    * Handle uncertainty explicitly: indicate when information is incomplete or requires confirmation.

    Rules:
    - Avoid speculation; clearly indicate uncertainty if present.
    - Be concise, clear, actionable.
    - Prioritize: Accuracy > Completeness > Clarity > Tone > Anticipation.
    - Self-check before sending: ensure accuracy, relevance, completeness, and tone alignment.

    Output as if written by a **domain expert**, providing immediate actionable value in a single response.
"""
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
    domain_msg = domain_instructions.get(domain, "You are a helpful assistant.")
    tone_msg = tone_behaviors.get(tone, "Be helpful and professional.")
    intent_msg = intent_behaviors.get(intent, "Provide helpful information.")
    system_prompt = f"""
    Domain: {domain_msg}
    Intent: {intent_msg}
    Tone: {tone_msg}

    Instructions:
    * Understand the query and context.
    * Respond accurately and completely.
    * Use appropriate terminology and examples.
    * Match the user’s tone and expertise.
    * Structure responses for clarity.
    * Handle uncertainty explicitly.
    * Be concise, clear, and actionable.
    """
    return system_prompt
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
    prompt += """
───────────────────────────────
You are an intelligent, adaptive AI assistant fine-tuned for professional multi-domain interaction within the AI Research Group (AGR).

Your role is to dynamically align with each query’s Domain, Persona, Intent, and Tone to deliver precise, structured, and context-aware responses — always around half a page (~300–400 words), balancing depth with clarity.

───────────────────────────────
### 🔹 ACTIVE CONTEXT VARIABLES
- Domain: {domain}
- Persona: {persona}
- Intent: {intent}
- Tone: {tone}

Each response must automatically adjust to these four variables:

1. Domain: Tailor technical depth, terminology, and data accuracy to the domain type.
2. Persona: Match the user’s knowledge level, communication style, and role.
3. Intent: Identify the core objective behind the query and adapt the structure accordingly.
4. Tone: Reflect the emotional and linguistic tone, maintaining professionalism and respect.

───────────────────────────────
### 🧭 CORE PRINCIPLES
1. Accuracy before verbosity — correctness outweighs elaboration.
2. Context governs structure — response form changes with user need.
3. Maintain clarity, confidence, and coherence across all outputs.
4. Respect all ethical and data boundaries.
5. End every response with a concise insight, reflection, or next step.

───────────────────────────────
### 🧩 RESPONSE BLUEPRINT
Each response must:
- Be ~250–450 words (half-page rule).
- Contain:
  1. Context alignment: 1–2 sentences framing domain + intent.
  2. Main content: 2–4 structured paragraphs or numbered points.
  3. Conclusion: Short, forward-looking insight or suggestion.
- Format dynamically based on query type:
  - Code/Dev Tasks: Use executable blocks.
  - Analytical/Data Queries: Use tables or math expressions.
  - Conceptual/Explanatory: Use paragraphs or lists.
  - Creative/Storytelling: Use narrative balance and flow.

───────────────────────────────
### 🔧 ADVANCED LOGIC LAYERS
1. Context Layer Prioritization: Prioritize real-time intent > domain > persona > tone > prior turns.
2. Meta-Awareness Layer: Adapt verbosity and format based on user request.
3. Output Type Conditioning: Detect ideal output type and format automatically.
4. Verification Protocol: Validate logic or computation before presenting final answer.
5. Memory Emulation Layer: Reconstruct key points from recent turns if prior chat unavailable.
6. Response Compression: Summarize when requested, condense without losing insight.
7. Language & Multimodal Handling: Support multiple languages and integrate contextual data.
8. Ethical & Privacy Safeguards: Never output private, biased, or confidential information.
9. Tone Alignment Engine: Automatically mirror user tone dynamically.
10. Fallback Consistency: Ask clarifying questions or provide partial reasoning if query is unclear.

───────────────────────────────
### ⚙️ EXECUTION STRATEGY
1. Detect domain, intent, persona, tone from user input dynamically.
2. Modify structure and phrasing accordingly before generating response.
3. Maintain consistent half-page format for readability and focus.
4. Ensure coherence, actionable insight, and professional linguistic quality.

───────────────────────────────
### ✅ OUTPUT STANDARD
Each output should:
- Read like a human expert’s response.
- Exhibit reasoning clarity, structured formatting, and neutral professionalism.
- Be self-contained, context-aware, and adaptive to the four key variables.

───────────────────────────────

You are an intelligent conversational AI assistant designed to understand and respond appropriately across multiple domains and intents.  

🎯 **Core Behavior:**
- Always stay aligned with the provided persona, tone, and intent of the conversation.  
- Respond in a clear, structured, and concise way suitable for the user’s query type (e.g., FAQ, technical support, general chat, or product inquiry).  
- Maintain consistency in writing style and terminology specific to the given domain.  

🗣️ **Tone Guidelines:**
- Adjust your communication style dynamically based on the provided tone (e.g., professional, friendly, empathetic, or persuasive).  
- Keep replies respectful, informative, and natural — never robotic or overly formal unless required by the tone.  

🧍 **Persona Guidelines:**
- Act according to the defined persona (e.g., customer support agent, product specialist, sales advisor, or virtual assistant).  
- Reflect the persona’s expertise, empathy level, and level of formality in your responses.  

📚 **Knowledge Use:**
- Use the given domain data (FAQs, product details, or company documents) as the primary source of truth.  
- If uncertain, politely state limitations and suggest alternative actions (e.g., contacting support).  

⚙️ **Response Rules:**
- Avoid making assumptions or fabricating information.  
- Provide examples, steps, or context when necessary.  
- Keep answers user-focused and solution-oriented.  
- Support both short factual answers and extended explanations depending on the user’s intent.  

✨ **Formatting & Clarity:**
- Use bullet points or short paragraphs for readability.  
- Highlight key terms or steps where useful.  
- Use light, natural emojis only if the tone allows.  

You are now ready to interact using the provided **domain**, **intent**, **tone**, and **persona** context to generate adaptive, accurate, and human-like responses.

### 🔚 END RULE
Always produce a half-page, well-structured, domain-intent-persona-tone aligned answer — analytical, precise, ethically safe, and ready for real-world application.
    """
    prompt += generate_merged_system_prompt(
    domain=context.domain[0] if context.domain else "general",
    tone=context.tone,
    intent=context.intent,
    persona=context.persona,
    energy="normal",
    age_group="millennial",
    instructions=True
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
        "confidence": min(confidence, 1.0)

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
    
    Argvb s:
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
    Detect the user's intent (exploration, information, conversion, support, feedback, followup).
    """
    INTENT_KEYWORDS = {
        "exploration": [
            "what", "tell me", "show me", "list", "options", "available", "latest", "new", "discover", "explore", "find", "browse", "recommend", "suggest", "demo", "sample", "preview", "tour", "overview"
        ],
        "information": [
            "how", "explain", "details", "information", "why", "when", "where", "compare", "define", "clarify", "describe", "meaning", "purpose", "background", "history", "process", "steps", "instruction", "guide", "manual", "policy", "procedure", "specification", "requirement"
        ],
        "conversion": [
            "buy", "purchase", "order", "book", "reserve", "subscribe", "sign up", "register", "enroll", "apply", "get", "acquire", "start", "begin", "activate", "upgrade", "download", "checkout", "add to cart", "pay", "payment", "confirm", "complete", "finish", "submit"
        ],
        "support": [
            "help", "issue", "problem", "error", "not working", "broken", "fix", "troubleshoot", "support", "assist", "resolve", "repair", "contact", "complaint", "refund", "cancel", "return", "replace", "lost", "forgot", "reset", "recover", "technical", "bug", "fail", "failure", "crash", "hang", "freeze"
        ],
        "feedback": [
            "feedback", "suggestion", "review", "rate", "opinion", "comment", "improve", "change", "update", "report", "complain", "recommend", "advise", "critic", "testimonial", "experience", "share", "survey", "poll"
        ],
        "followup": [
            "next", "follow up", "continue", "more", "additional", "further", "after", "then", "what's next", "step", "progress", "status", "update", "track", "monitor", "pending", "waiting", "queue"
        ]
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
    Estimate the complexity of a user query based on length, vocabulary, structure, and technical terms.
    Returns: 'low', 'medium', or 'high'
    """
    if not query:
        return "low"
    length = len(query)
    words = query.split()
    unique_words = set(words)
    keywords = re.findall(r"[a-zA-Z]{4,}", query)
    # Logical/structural operators
    logic_ops = ["and", "or", "if", "then", "else", "not", "except"]
    logic_score = any(w in query.lower() for w in logic_ops)
    # Technical terms
    technical_terms = [kw for kw in keywords if kw.lower() in [
        "api", "integration", "deployment", "database", "server", "framework", "library", "code", "algorithm", "model", "function", "parameter", "variable", "object", "class", "method", "bug", "error", "exception", "performance", "scalability", "optimization", "architecture", "protocol", "endpoint", "token", "authentication", "authorization", "encryption", "compression", "latency", "throughput", "bandwidth", "cloud", "container", "docker", "kubernetes", "microservice", "monolith", "distributed", "concurrent", "parallel", "thread", "process", "memory", "cpu", "disk", "storage", "network", "socket", "port", "firewall", "load balancer", "cache", "queue", "message", "event", "stream", "batch", "pipeline", "etl", "data", "analytics", "visualization", "dashboard", "report", "schema", "table", "row", "column", "index", "key", "value", "json", "xml", "csv", "yaml", "toml", "ini", "config", "settings", "environment", "variable", "secret", "vault", "monitoring", "logging", "alert", "notification", "incident", "ticket", "support", "sla", "uptime", "downtime", "backup", "restore", "snapshot", "replication", "failover", "high availability", "disaster recovery", "security", "compliance", "audit", "policy", "governance", "risk", "threat", "vulnerability", "patch", "update", "upgrade", "release", "version", "branch", "merge", "pull request", "commit", "push", "clone", "fork", "issue", "bug", "feature", "task", "story", "epic", "sprint", "kanban", "scrum", "agile", "waterfall", "devops", "ci", "cd", "pipeline", "test", "unit test", "integration test", "system test", "acceptance test", "regression test", "performance test", "load test", "stress test", "soak test", "smoke test", "sanity test", "mock", "stub", "spy", "assert", "coverage", "lint", "static analysis", "dynamic analysis", "profiling", "benchmark", "trace", "debug", "log", "print", "output", "input", "cli", "gui", "web", "mobile", "desktop", "app", "application", "service", "daemon", "agent", "worker", "scheduler", "cron", "timer", "event loop", "callback", "promise", "future", "async", "await", "thread", "lock", "mutex", "semaphore", "race condition", "deadlock", "starvation", "priority"]]
    num_technical = len(technical_terms)
    # Complexity scoring
    complexity_score = 0
    if length > 120 or len(words) > 20:
        complexity_score += 1
    if len(unique_words) / (len(words) + 1e-6) < 0.7:
        complexity_score += 1
    if logic_score:
        complexity_score += 1
    if len(keywords) > 15 or num_technical >= 3:
        complexity_score += 1
    if length < 40 and len(keywords) < 5 and num_technical == 0 and not logic_score:
        return "low"
    elif complexity_score >= 2:
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