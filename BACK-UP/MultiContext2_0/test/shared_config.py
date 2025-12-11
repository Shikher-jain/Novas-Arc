def build_system_prompt(context, history=None):
    persona=", ".join(context.persona);domain=", ".join(context.domain);tone=context.tone;intent=context.intent;operational_context=infer_operational_context(context.domain)
    prompt=(f"You are a {persona} operating within the {domain} domain(s).\n"+f"Your communication tone should be {tone}.\n"+f"Primary intent: {intent}.\n"+f"Operational context: {operational_context}.\n"+"You must:\n"+"• Understand the query deeply and precisely.\n"+"• Provide a complete, factually accurate, and context-optimized answer.\n"+"• Anticipate related needs and pre-empt follow-up questions.\n"+"• Balance expertise, empathy, and brevity.\n"+"• Ensure the user receives all relevant information in a single, comprehensive response.\n")
    if not persona or not domain or not tone:prompt+="\nYou are a helpful and adaptive assistant providing safe, neutral, and accurate information."
    if history:prompt+="\nContextual history: "+" | ".join(history[-3:])
    return prompt
def infer_operational_context(domains):
    domain_map={"support":"support","technical":"technical","sales":"sales","booking":"booking","education":"education","health":"healthcare","travel":"travel","finance":"finance","government":"government","legal":"legal","entertainment":"entertainment","startup":"startup","hospitality":"hospitality","retail":"retail","product":"product"}
    return next((domain_map[d] for d in domains if d in domain_map),"general")
def evaluate_response(query, response, context=None):
    relevance=float(st_util.cos_sim(st_model.encode(query),st_model.encode(response)).item()) if 'st_model' in globals() and st_model else None
    completeness=(sum(kw in response for kw in context.domain+context.persona+[context.intent])/(len(context.domain)+len(context.persona)+1)) if context else None
    confidence=float(len(response))/100.0+0.1 if "confident" in response or "certain" in response else float(len(response))/100.0
    return{"relevance":relevance,"completeness":completeness,"confidence":min(confidence,1.0)}
def update_prompt_with_feedback(context,scores):
    prompt=context.system_prompt
    prompt+=("\nPlease ensure your answer is highly relevant to the user's intent and domain." if scores.get("relevance",1)<0.5 else"")
    prompt+=("\nMake sure to cover all aspects of the user's query." if scores.get("completeness",1)<0.7 else"")
    prompt+=("\nIf uncertain, state so and provide best possible information." if scores.get("confidence",1)<0.5 else"")
    return prompt
def synthesize_output_json(context):
    import json;return json.dumps({"domain":context.domain,"intent":context.intent,"persona":context.persona,"tone":context.tone,"context":infer_operational_context(context.domain),"system_prompt":context.system_prompt,"greeting":context.greeting},ensure_ascii=False)
import re,random,logging,nltk
from dataclasses import dataclass,field
from typing import List,Dict,Any
try:
    import spacy;nlp_spacy=spacy.load("en_core_web_sm")
except Exception:
    nlp_spacy=None
try:
    from sentence_transformers import SentenceTransformer,util as st_util;st_model=SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    st_model=None
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError:
        SentimentIntensityAnalyzer=None
from nltk.tokenize import word_tokenize
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
_sia_instance = None
def _ensure_sentiment_analyzer():
    global _sia_instance
    _sia_instance=_sia_instance if _sia_instance is not None or SentimentIntensityAnalyzer is None else (SentimentIntensityAnalyzer() if not isinstance(_sia_instance,SentimentIntensityAnalyzer) else _sia_instance)
    return _sia_instance
sia = _ensure_sentiment_analyzer()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
TOPIC_TONE_MAP = {"art": "creative","automotive": "technical","career": "motivational","climate_change": "urgent","college": "formal","cryptocurrency": "analytical","DIY": "creative","education": "encouraging","energy": "sustainable","entertainment": "casual","environment": "sustainable","fashion": "trendy","finance": "professional","fitness": "motivational","food": "enthusiastic","gamer": "casual","gaming": "excited","gardening": "peaceful","government": "official","health": "empathetic","history": "narrative","home_decor": "aesthetic","hospitality": "welcoming","insurance": "reassuring","legal": "authoritative","literature": "expressive","logistics": "precise","manufacturing": "technical","mental_health": "compassionate","music": "passionate","news": "neutral","parenting": "supportive","pets": "affectionate","philosophy": "thoughtful","photography": "artistic","politics": "objective","product": "detailed","psychology": "insightful","real_estate": "detailed","relationships": "understanding","retail": "helpful","science": "analytical","space_exploration": "inspiring","sports": "energetic","startup": "innovative","technology": "informative","technology_trends": "futuristic","transport": "efficient","travel": "adventurous","videography": "cinematic","wildlife": "conservationist"}
CONTEXTS = {"support": {"keywords": ["help", "support", "faq", "customer-service"],"system_msg": "You are a customer support assistant. Provide clear and helpful answers to customer questions."},"technical": {"keywords": ["api", "developer", "docs", "technical", "implementation"],"system_msg": "You are a technical expert. Provide detailed and accurate technical information."    },"sales": {"keywords": ["pricing", "purchase", "product", "order", "buy"],"system_msg": "You are a sales assistant. Help customers understand products and make informed decisions."    },"booking": {"keywords": ["schedule", "appointment", "booking", "reservation"],"system_msg": "You are a booking assistant. Help users schedule and manage their appointments efficiently."}}
def analyze_content(user_message):
    analyzer=sia or _ensure_sentiment_analyzer()
    sentiment=analyzer.polarity_scores(user_message) if analyzer is not None else{"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    keywords=[w.lower() for w in word_tokenize(user_message) if w.isalpha() and len(w)>3] if analyzer is not None else user_message.lower().split()
    return sentiment,keywords
def advanced_topic_detection(keywords, context=None):
    matched_topics = [key for key in TOPIC_TONE_MAP if key in keywords]
    general_keywords = {"company", "service", "product", "furniture", "appliances", "electronics", "rental", "rent", "mojo", "about", "feature", "benefit"}
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
    compound=sentiment_dict.get('compound',0)if second_sentiment is None else(sentiment_dict.get('compound',0)+second_sentiment.get('compound',0))/2
    return "enthusiastic"if compound>0.5 else"positive"if compound>0.2 else"urgent"if compound<-0.5 else"negative"if compound<-0.2 else"neutral"
def persona_expansion(keywords, context=None):
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
    general_keywords = {"company", "service", "product", "furniture", "appliances", "electronics", "rental", "rent", "mojo", "about", "feature", "benefit"}
    if any(gk in keywords for gk in general_keywords):
        for extra in ["customer support agent", "sales assistant", "product expert"]:
            if extra not in personas:
                personas.append(extra)
    if not personas:
        personas.append("helpful assistant")
    return personas
def advanced_system_prompt_generator(question, answer, context=None):
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
    all_keywords = set(q_keywords + a_keywords)
    if context and context in CONTEXTS:
        all_keywords.update(CONTEXTS[context]["keywords"])
    topics = advanced_topic_detection(all_keywords, context)
    personas = persona_expansion(all_keywords, context)
    tone = nuanced_tone_detection(q_sentiment, a_sentiment)
    prompt_template_map = {("technical expert", "casual"): "Hey! I'm your tech pal. I'll explain things simply and keep it chill.",("technical expert", "enthusiastic"): "Hi! I'm your excited technical expert. Let's dive into this with energy!",("customer support agent", "casual"): "Hi! I'm your support buddy. Let's solve your issue together, no stress.",("customer support agent", "empathetic"): "Hi, I'm here for you. I understand your concern and will help you through this.",("customer support agent", "urgent"): "I'm here to resolve this quickly and efficiently. Let's get this sorted!",("helpful assistant", "casual"): "Hey there! I'm your friendly assistant. Ask me anything and I'll help out in a relaxed, casual way.",("helpful assistant", "enthusiastic"): "Hi! I'm your enthusiastic assistant, excited to help you out! Ask away.",("helpful assistant", "urgent"): "I'm here to help you quickly and efficiently. Let's solve your problem right now!",("sales assistant", "enthusiastic"): "Hi! I'm excited to help you explore our products and find the perfect fit!",("troubleshooter", "empathetic"): "Hi, I'm here to help troubleshoot and solve your issue with care and patience.",}
    topic_prompt_map = {"health": "I'm your caring health assistant. I'll answer with empathy and support for your well-being.","finance": "You are a finance professional. Give precise, trustworthy, and easy-to-understand financial advice.","education": "You are an educator. Explain concepts clearly and encourage learning in a supportive way.","travel": "You are a travel concierge. Offer friendly, adventurous, and helpful travel advice.","legal": "You are a legal advisor. Provide authoritative, clear, and compliant legal information.","hospitality": "You are a hospitality expert. Offer welcoming, attentive, and helpful service advice.","retail": "You are a retail assistant. Provide helpful, friendly, and product-focused answers.","government": "You are a government official. Provide official, clear, and policy-compliant information.","technology": "You are a technology expert. Provide clear, detailed, and accurate technical information.","startup": "You are a startup advisor. Provide innovative, growth-focused, and practical business advice.",}
    prompt_parts = []
    for persona in personas[:3]:  # Limit to 3 personas
        for topic in topics[:2]:  # Limit to 2 topics
            key = (persona, tone)
            if key in prompt_template_map:
                prompt_parts.append(prompt_template_map[key])
            elif topic in topic_prompt_map:
                prompt_parts.append(topic_prompt_map[topic])
            else:
                prompt_parts.append(f"You are a {persona} with a {tone} tone. "f"Your expertise is in {topic}. "f"Provide clear, accurate, and helpful answers to user questions.")
    system_prompt = "\n".join(dict.fromkeys(prompt_parts))
    if not system_prompt or len(system_prompt) < 20:
        system_prompt = CONTEXTS.get(context, {}).get("system_msg", "You are a helpful assistant.")
        logging.warning("System prompt generation resulted in short prompt, using fallback")
    return system_prompt
def detect_intent(query):
    INTENT_KEYWORDS = {"exploration": ["what", "tell me", "show me", "list", "options", "available", "latest", "new", "discover", "explore", "find", "browse", "recommend", "suggest", "demo", "sample", "preview", "tour", "overview"],"information": ["how", "explain", "details", "information", "why", "when", "where", "compare", "define", "clarify", "describe", "meaning", "purpose", "background", "history", "process", "steps", "instruction", "guide", "manual", "policy", "procedure", "specification", "requirement"],"conversion": ["buy", "purchase", "order", "book", "reserve", "subscribe", "sign up", "register", "enroll", "apply", "get", "acquire", "start", "begin", "activate", "upgrade", "download", "checkout", "add to cart", "pay", "payment", "confirm", "complete", "finish", "submit"],"support": ["help", "issue", "problem", "error", "not working", "broken", "fix", "troubleshoot", "support", "assist", "resolve", "repair", "contact", "complaint", "refund", "cancel", "return", "replace", "lost", "forgot", "reset", "recover", "technical", "bug", "fail", "failure", "crash", "hang", "freeze"],"feedback": ["feedback", "suggestion", "review", "rate", "opinion", "comment", "improve", "change", "update", "report", "complain", "recommend", "advise", "critic", "testimonial", "experience", "share", "survey", "poll"],"followup": ["next", "follow up", "continue", "more", "additional", "further", "after", "then", "what's next", "step", "progress", "status", "update", "track", "monitor", "pending", "waiting", "queue"]}
    prompt_lower = query.lower()
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        intent_scores[intent] = sum(1 for kw in keywords if kw in prompt_lower)
    best_intent = max(intent_scores, key=intent_scores.get)
    return best_intent if intent_scores[best_intent] > 0 else "information"
def estimate_complexity(query):
    if not query:return"low"
    length=len(query);words=query.split();unique_words=set(words);keywords=re.findall(r"[a-zA-Z]{4,}",query)
    logic_ops=["and","or","if","then","else","not","except"]
    logic_score=any(w in query.lower()for w in logic_ops)
    technical_terms=[kw for kw in keywords if kw.lower()in["api","integration","deployment","database","server","framework","library","code","algorithm","model","function","parameter","variable","object","class","method","bug","error","exception","performance","scalability","optimization","architecture","protocol","endpoint","token","authentication","authorization","encryption","compression","latency","throughput","bandwidth","cloud","container","docker","kubernetes","microservice","monolith","distributed","concurrent","parallel","thread","process","memory","cpu","disk","storage","network","socket","port","firewall","load balancer","cache","queue","message","event","stream","batch","pipeline","etl","data","analytics","visualization","dashboard","report","schema","table","row","column","index","key","value","json","xml","csv","yaml","toml","ini","config","settings","environment","variable","secret","vault","monitoring","logging","alert","notification","incident","ticket","support","sla","uptime","downtime","backup","restore","snapshot","replication","failover","high availability","disaster recovery","security","compliance","audit","policy","governance","risk","threat","vulnerability","patch","update","upgrade","release","version","branch","merge","pull request","commit","push","clone","fork","issue","bug","feature","task","story","epic","sprint","kanban","scrum","agile","waterfall","devops","ci","cd","pipeline","test","unit test","integration test","system test","acceptance test","regression test","performance test","load test","stress test","soak test","smoke test","sanity test","mock","stub","spy","assert","coverage","lint","static analysis","dynamic analysis","profiling","benchmark","trace","debug","log","print","output","input","cli","gui","web","mobile","desktop","app","application","service","daemon","agent","worker","scheduler","cron","timer","event loop","callback","promise","future","async","await","thread","lock","mutex","semaphore","race condition","deadlock","starvation","priority"]]
    num_technical=len(technical_terms);complexity_score=0
    complexity_score+=(1 if length>120 or len(words)>20 else 0)+(1 if len(unique_words)/(len(words)+1e-6)<0.7 else 0)+(1 if logic_score else 0)+(1 if len(keywords)>15 or num_technical>=3 else 0)
    return"low"if length<40 and len(keywords)<5 and num_technical==0 and not logic_score else"high"if complexity_score>=2 else"medium"if complexity_score==1 else"low"
TONE_GREETINGS = {"casual": ["Hey!", "Hi there!", "Hello!", "Yo!", "Hey buddy!", "Hey friend!", "Hi!", "Hey, how's it going?", "Hey, what's up?", "Hey, glad to help!"],"friendly": ["Hello friend!", "Hi, great to see you!", "Hey there!", "Welcome!", "Hi, how can I help you today?", "Hey, happy to assist!"],"quirky": ["Yo yo yo!", "What's cookin'?", "Howdy partner!", "Ahoy!", "Wassup!", "Ready for some fun?"],"fun": ["Let's get this party started!", "Woohoo!", "Yay!", "Time for some fun!", "Let's do this!", "Excited to help!"],"formal": ["Greetings.", "Hello.", "Good day.", "Welcome.", "How may I assist you?"],"professional": ["Hello, how can I assist you?", "Welcome, let me know your query.", "Good day, I'm here to help.", "Thank you for reaching out."],"enthusiastic": ["Hi! I'm excited to help!", "Hello! Let's get started!", "Woohoo! Let's solve this together!"],"empathetic": ["Hi, I'm here for you.", "Hello, I understand your concern.", "Hey, I'm here to help you through this."],"urgent": ["Let's resolve this quickly.", "I'm on it right away!", "Let's get this sorted ASAP!"],"negative": ["I'm sorry to hear that.", "Let's see how I can help.", "I'll do my best to assist you."],"neutral": ["Hello.", "Hi.", "Welcome.", "How can I help you?"]}
def prepend_tone_greeting(response, tone):
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
    sentiment,keywords=analyze_content(query)
    tone=nuanced_tone_detection(sentiment)
    domains=advanced_topic_detection(keywords)or["general"]
    intent=detect_intent(query)
    personas=persona_expansion(keywords)or["helpful assistant"]
    entities=[ent.text for ent in nlp_spacy(query).ents]if nlp_spacy else[]
    complexity=estimate_complexity(query)
    context_vector=str(st_model.encode(query).tolist()[:8])if st_model else""
    context_summary=f"Domains: {domains}, Intent: {intent}, Tone: {tone}, Persona: {personas}, Entities: {entities}, Complexity: {complexity}"
    return PromptContext(domain=domains,intent=intent,tone=tone,persona=personas,complexity=complexity,entities=entities,context_summary=context_summary,system_prompt="",quality_scores={},greeting="")