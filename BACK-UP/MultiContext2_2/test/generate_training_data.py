"""
Domain × Tone × Intent Training Data Generator
==============================================

Generates high-quality, strictly domain-separated training data for adaptive Q&A models.
Features:
- Domain-specific system prompts
- Helpful, domain-aware fallback answers
- Strict separation for each domain's training file
- JSONL output for OpenAI fine-tuning

Usage:
    python generate_training_data.py
"""

import os
import json
import random


# ========== CONFIGURATION DICTIONARIES ==========

INDUSTRY_DOMAINS = {
    "hospitality": {
        "keywords": ["hotel", "restaurant", "booking", "reservation", "guest", "stay", "check-in", "checkout"],
        "tone": "welcoming",
        "description": "Hospitality & Tourism"
    },
    "retail": {
        "keywords": ["store", "shop", "product", "inventory", "purchase", "item", "discount", "sale"],
        "tone": "helpful",
        "description": "Retail & E-Commerce"
    },
    "gaming": {
        "keywords": ["game", "play", "level", "score", "character", "achievement", "quest", "multiplayer"],
        "tone": "quirky",
        "description": "Gaming & Entertainment"
    },
    "finance": {
        "keywords": ["bank", "account", "transaction", "investment", "loan", "credit", "deposit", "balance"],
        "tone": "professional",
        "description": "Finance & Banking"
    },
    "healthcare": {
        "keywords": ["doctor", "patient", "appointment", "prescription", "medical", "hospital", "symptom"],
        "tone": "empathetic",
        "description": "Healthcare & Medical"
    },
    "education": {
        "keywords": ["student", "course", "class", "lesson", "university", "college", "school", "professor"],
        "tone": "encouraging",
        "description": "Education & Learning"
    },
    "technology": {
        "keywords": ["code", "api", "software", "system", "database", "server", "deployment", "bug"],
        "tone": "technical",
        "description": "Technology & IT"
    },
    "corporate": {
        "keywords": ["business", "company", "employee", "management", "meeting", "budget", "revenue"],
        "tone": "professional",
        "description": "Corporate Business"
    }
}

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

TOUCHPOINT_INTENT_MAP = {
    "home_page": {
        "primary_intent": "exploration",
        "description": "User exploring options and learning about offerings",
        "keywords": ["what", "show", "tell", "options", "available"]
    },
    "product_page": {
        "primary_intent": "information",
        "description": "User wants detailed information about specific product/service",
        "keywords": ["details", "how", "features", "specs", "benefits"]
    },
    "support_page": {
        "primary_intent": "support",
        "description": "User seeking issue resolution or help",
        "keywords": ["help", "problem", "issue", "error", "fix"]
    },
    "checkout": {
        "primary_intent": "conversion",
        "description": "User ready to make purchase or commit to action",
        "keywords": ["buy", "purchase", "order", "subscribe", "book"]
    },
    "faq_section": {
        "primary_intent": "information",
        "description": "User looking for quick answers to common questions",
        "keywords": ["how", "what", "when", "where", "why"]
    },
    "feedback_form": {
        "primary_intent": "support",
        "description": "User providing feedback or reporting issues",
        "keywords": ["problem", "feedback", "suggest", "improve", "complaint"]
    }
}

# ========== GENERAL PURPOSE QUESTIONS DATABASE ==========

GENERAL_QUESTIONS = {
    "hospitality": [
        "What are your room types and amenities?",
        "How do I make a reservation?",
        "What's your cancellation policy?",
        "Do you offer group discounts?",
        "What time is check-in and check-out?",
        "Are pets allowed?",
        "Do you have parking facilities?",
        "What restaurants are nearby?"
    ],
    "retail": [
        "Do you have this item in stock?",
        "What's your return policy?",
        "Do you offer free shipping?",
        "How do I use a coupon?",
        "What's the warranty on this product?",
        "Do you have this in other colors?",
        "When are sales happening?",
        "How long does delivery take?"
    ],
    "gaming": [
        "How do I level up faster?",
        "What are the best strategies?",
        "How do I unlock new characters?",
        "Can I play offline?",
        "How do multiplayer matches work?",
        "What rewards can I earn?",
        "Is there a mobile version?",
        "How do I join a clan?"
    ],
    "finance": [
        "What's the interest rate on savings accounts?",
        "How do I open an account?",
        "What are your fees?",
        "How do I transfer money?",
        "Is my money insured?",
        "How do I apply for a loan?",
        "What credit score do I need?",
        "How do I check my balance?"
    ],
    "healthcare": [
        "How do I schedule an appointment?",
        "What insurance do you accept?",
        "What are your office hours?",
        "Do you offer telemedicine?",
        "What should I bring to my appointment?",
        "How long is the wait typically?",
        "What's your cancellation policy?",
        "Do you have emergency services?"
    ],
    "education": [
        "What are the admission requirements?",
        "What's the tuition cost?",
        "What programs do you offer?",
        "What's the average class size?",
        "Do you offer scholarships?",
        "What's your job placement rate?",
        "Can I take courses online?",
        "What's the campus like?"
    ],
    "technology": [
        "How do I install this library?",
        "What's the API endpoint?",
        "How do I authenticate?",
        "What are the rate limits?",
        "Do you have code examples?",
        "What's the response format?",
        "How do I report a bug?",
        "What's your SLA?"
    ],
    "corporate": [
        "What's your company mission?",
        "How do I contact sales?",
        "What are your business hours?",
        "Do you offer enterprise plans?",
        "What's your response time?",
        "Do you have case studies?",
        "What partnerships do you have?",
        "How do I schedule a demo?"
    ]
}

ANSWER_DATABASE = {
    "hospitality": {
        "room_types|amenities": [
            "We offer deluxe rooms with ocean views, standard rooms with garden views, and suites with private balconies. All rooms include complimentary WiFi, flat-screen TV, and premium bedding.",
            "Our rooms range from cozy single rooms to spacious family suites. Amenities include air conditioning, mini-bar, workspace, and 24-hour room service.",
            "Choose from basic rooms, business class, or luxurious suites. Every room features en-suite bathroom, smart TV, and access to our fitness center.",
        ],
        "reservation|booking": [
            "You can reserve online through our website, call our reservations team at 1-800-STAY, or visit our front desk. We offer flexible payment options.",
            "Reservations are easy! Use our mobile app for instant booking, or contact us directly for special requests. Same-day bookings available.",
            "Book now through our website with instant confirmation, or speak with our team for personalized recommendations and group rates.",
        ],
        "cancellation": [
            "Cancellations made 7+ days before arrival receive full refund. Within 7 days, a 50% fee applies. Non-refundable rates are discounted accordingly.",
            "Free cancellation up to 48 hours before check-in. Late cancellations may incur one night's charge. Premium rates have different policies.",
            "Standard rate: free cancellation until 72 hours before arrival. Last-minute rate: no changes allowed. Check your booking confirmation for specifics.",
        ],
        "discounts": [
            "We offer 15% off for group bookings (10+ rooms), senior discounts (15%), and loyalty rewards for returning guests.",
            "Group discounts available for parties of 8+. Students get 10% off with valid ID. AAA members receive special rates.",
            "Extended stay discounts: 10% for 7+ nights, 15% for 30+ nights. Corporate rates available with valid business credentials.",
        ],
        "check_in|check_out": [
            "Check-in is from 3:00 PM, check-out at 11:00 AM. Early check-in and late check-out available for an additional fee based on availability.",
            "Standard check-in: 2:00 PM, check-out: 12:00 PM. Express check-in available at 1:00 PM for a small fee.",
            "Check-in begins at 3:00 PM, check-out by 10:00 AM. Late check-out until 2:00 PM available at 50% of room rate.",
        ],
        "pets": [
            "Yes, we're pet-friendly! Small pets (under 25 lbs) welcome with $50 pet fee per stay. Service animals stay free.",
            "Pets are allowed in designated rooms for a $30 daily fee. We provide pet beds, bowls, and waste bags.",
            "We welcome cats and dogs up to 30 lbs. Pet fee is $25/night. Exotic animals require prior approval.",
        ],
        "parking": [
            "We offer complimentary parking for all guests in our secure lot. Electric vehicle charging available.",
            "Parking is $15/day or free for members. Valet service available for $25/day. Street parking nearby.",
            "Free unlimited parking for guests. Covered parking and valet options available for premium rates.",
        ],
        "restaurants": [
            "Within walking distance: Italian restaurant, Thai cuisine, steakhouse, and café. Room service available 24/7.",
            "We have an in-house restaurant open for breakfast, lunch, and dinner. Popular nearby options include Asian fusion and Mediterranean.",
            "Our partner restaurants offer discounts to guests. Fine dining, casual, and fast-casual options all within 5 minutes.",
        ],
    },
    "retail": {
        "stock|availability": [
            "We have limited units in stock. Check size/color availability in our inventory system. Available for immediate shipping.",
            "In stock and ready to ship! Free standard shipping on orders over $50.",
            "Currently in stock. Limited quantities available. Reserve now to guarantee availability.",
        ],
        "return": [
            "30-day return policy for unworn items with original tags. Full refund to original payment method. Return shipping is free.",
            "Returns accepted within 60 days of purchase. Original packaging and receipt required. Refund processed within 5-7 business days.",
            "Easy returns! 45 days from purchase date. No questions asked. Use our prepaid return label.",
        ]
    },
    "healthcare": {
        "appointment": [
            "Book online 24/7 through our portal. Phone: 1-800-DOC-HELP. Walk-ins available with 2-hour wait average.",
            "Schedule appointments online for next 30 days. Phone booking: Mon-Fri 8am-6pm. Same-day available based on provider.",
            "Online scheduling shows real-time availability. Book appointments 2+ weeks in advance for better times.",
        ],
        "insurance": [
            "We accept 50+ major insurance plans including Aetna, Blue Cross, Cigna, and UnitedHealth. Verify coverage in patient portal.",
            "In-network with most plans. Self-pay rates available. Insurance questions? Call billing at 1-800-BILLING.",
            "Accepted insurance: all major providers. Payment plans available if uninsured. Financial assistance programs offered.",
        ]
    },
    "technology": {
        "install": [
            "Install via: pip install package_name. Or npm install for Node.js. Requires Python 3.7+ or Node 14+.",
            "Installation methods: pip (easiest), conda, or source. Documentation available online.",
            "Quick install: pip install [package]. Detailed instructions in README. Dependencies auto-install.",
        ],
        "api_endpoint": [
            "Base URL: https://api.example.com/v1/. Authentication via header. Rate limit: 1000 requests/hour.",
            "Endpoint format: /api/v2/{resource}/{id}. SSL/TLS required. JSON responses.",
            "Primary endpoint: api.service.com. Endpoints documented at docs.service.com. Multiple regions available.",
        ]
    },
    "education": {
        "admission": [
            "Requirements: High school diploma/GED, 3.0+ GPA, SAT 1050+. Application: $50 fee. Rolling admissions.",
            "Entrance exams required. High school transcripts. Letters of recommendation (2). Personal statement.",
            "Acceptance based on GPA, test scores, and essays. Rolling admission through December. Early decision deadline: November 15.",
        ],
        "tuition": [
            "In-state: $25,000/year. Out-of-state: $45,000/year. Room & board: $15,000/year. Payment plans available.",
            "Tuition: $28k. Fees: $2k. Housing: $12k. Books: $1.5k. Total: ~$43.5k/year.",
            "Annual cost: $30-50k depending on program. Scholarships reduce cost by 20-60%. Financial aid available.",
        ]
    }
}

def get_question_keyword(question, domain):
    """Extract question type keyword from question text."""
    question_lower = question.lower()
    keyword_map = {
        "hospitality": {
            "room": "room_types|amenities",
            "reservation": "reservation|booking",
            "cancel": "cancellation",
            "discount": "discounts",
            "check": "check_in|check_out",
            "pet": "pets",
            "parking": "parking",
            "restaurant": "restaurants",
        },
        "retail": {
            "stock": "stock|availability",
            "return": "return",
        },
        "healthcare": {
            "appointment": "appointment",
            "insurance": "insurance",
        },
        "technology": {
            "install": "install",
            "endpoint": "api_endpoint",
        },
        "education": {
            "admission": "admission",
            "tuition": "tuition",
        },
    }
    domain_keywords = keyword_map.get(domain, {})
    for keyword, answer_key in domain_keywords.items():
        if keyword in question_lower:
            return answer_key
    return list(domain_keywords.values())[0] if domain_keywords else None


def generate_training_record(domain, intent, tone, question, industry_data, touchpoint_data, platform_data):
    """
    Generate a single training record with domain-specific system prompt and helpful, domain-aware fallback answer.
    """
    # Domain-specific system prompt
    domain_prompts = {
        "hospitality": "You are a hospitality support assistant. Your goal is to help guests with empathy and professionalism.",
        "healthcare": "You are a healthcare support assistant. Your goal is to help patients and staff with empathy and clarity.",
        "technology": "You are a technology support assistant. Your goal is to help users solve technical issues with clear, precise guidance.",
        "education": "You are an education support assistant. Your goal is to help students and educators with patience and encouragement.",
    }
    system_prompt = domain_prompts.get(domain, f"You are an expert in {industry_data['description']}. Communicate in a {industry_data['tone']} tone. Style: {platform_data['style']}. User intent: {touchpoint_data['description']}. Provide concise, accurate, and helpful information only.")

    # Get answer from database or fallback
    answer_key = get_question_keyword(question, domain)
    if answer_key and domain in ANSWER_DATABASE and answer_key in ANSWER_DATABASE[domain]:
        answers = ANSWER_DATABASE[domain][answer_key]
        response = random.choice(answers)
    else:
        # Domain-aware fallback answers
        fallback_map = {
            "hospitality": f"Thank you for your question about hospitality. {question.capitalize()} If you need details about reservations, amenities, or policies, please specify.",
            "healthcare": f"Thank you for your healthcare question. {question.capitalize()} For appointments, insurance, or medical info, please clarify your need.",
            "technology": f"Thank you for your technology question. {question.capitalize()} For installation, troubleshooting, or API details, please specify your issue.",
            "education": f"Thank you for your education question. {question.capitalize()} For admissions, courses, or campus info, please clarify your query.",
        }
        response = fallback_map.get(domain, f"Thank you for your question. {question.capitalize()} Please provide more details for a specific answer.")

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }

ANSWER_DATABASE = {
    "hospitality": {
        "room_types|amenities": [
            "We offer deluxe rooms with ocean views, standard rooms with garden views, and suites with private balconies. All rooms include complimentary WiFi, flat-screen TV, and premium bedding.",
            "Our rooms range from cozy single rooms to spacious family suites. Amenities include air conditioning, mini-bar, workspace, and 24-hour room service.",
            "Choose from basic rooms, business class, or luxurious suites. Every room features en-suite bathroom, smart TV, and access to our fitness center.",
        ],
        "reservation|booking": [
            "You can reserve online through our website, call our reservations team at 1-800-STAY, or visit our front desk. We offer flexible payment options.",
            "Reservations are easy! Use our mobile app for instant booking, or contact us directly for special requests. Same-day bookings available.",
            "Book now through our website with instant confirmation, or speak with our team for personalized recommendations and group rates.",
        ],
        "cancellation": [
            "Cancellations made 7+ days before arrival receive full refund. Within 7 days, a 50% fee applies. Non-refundable rates are discounted accordingly.",
            "Free cancellation up to 48 hours before check-in. Late cancellations may incur one night's charge. Premium rates have different policies.",
            "Standard rate: free cancellation until 72 hours before arrival. Last-minute rate: no changes allowed. Check your booking confirmation for specifics.",
        ],
        "discounts": [
            "We offer 15% off for group bookings (10+ rooms), senior discounts (15%), and loyalty rewards for returning guests.",
            "Group discounts available for parties of 8+. Students get 10% off with valid ID. AAA members receive special rates.",
            "Extended stay discounts: 10% for 7+ nights, 15% for 30+ nights. Corporate rates available with valid business credentials.",
        ],
        "check_in|check_out": [
            "Check-in is from 3:00 PM, check-out at 11:00 AM. Early check-in and late check-out available for an additional fee based on availability.",
            "Standard check-in: 2:00 PM, check-out: 12:00 PM. Express check-in available at 1:00 PM for a small fee.",
            "Check-in begins at 3:00 PM, check-out by 10:00 AM. Late check-out until 2:00 PM available at 50% of room rate.",
        ],
        "pets": [
            "Yes, we're pet-friendly! Small pets (under 25 lbs) welcome with $50 pet fee per stay. Service animals stay free.",
            "Pets are allowed in designated rooms for a $30 daily fee. We provide pet beds, bowls, and waste bags.",
            "We welcome cats and dogs up to 30 lbs. Pet fee is $25/night. Exotic animals require prior approval.",
        ],
        "parking": [
            "We offer complimentary parking for all guests in our secure lot. Electric vehicle charging available.",
            "Parking is $15/day or free for members. Valet service available for $25/day. Street parking nearby.",
            "Free unlimited parking for guests. Covered parking and valet options available for premium rates.",
        ],
        "restaurants": [
            "Within walking distance: Italian restaurant, Thai cuisine, steakhouse, and café. Room service available 24/7.",
            "We have an in-house restaurant open for breakfast, lunch, and dinner. Popular nearby options include Asian fusion and Mediterranean.",
            "Our partner restaurants offer discounts to guests. Fine dining, casual, and fast-casual options all within 5 minutes.",
        ],
    },
    "retail": {
        "stock|availability": [
            "We have {quantity} units in stock. Check size/color availability in our inventory system. Available for immediate shipping.",
            "In stock and ready to ship! Free standard shipping on orders over $50.",
            "Currently in stock. Limited quantities available. Reserve now to guarantee availability.",
        ],
        "return": [
            "30-day return policy for unworn items with original tags. Full refund to original payment method. Return shipping is free.",
            "Returns accepted within 60 days of purchase. Original packaging and receipt required. Refund processed within 5-7 business days.",
            "Easy returns! 45 days from purchase date. No questions asked. Use our prepaid return label.",
        ],
        "shipping": [
            "Free shipping on orders over $50! Standard (5-7 days) or expedited (2-3 days) available.",
            "We offer free standard shipping, $5 express, and $10 overnight. Orders ship within 24 hours.",
            "Free shipping to US addresses. International shipping available at calculated rates.",
        ],
        "coupon|discount": [
            "Enter your code at checkout. Discount applies to all items. Valid through end of month. Cannot be combined with other offers.",
            "First-time customers get 20% off with code WELCOME20. Loyalty members get additional 10% off.",
            "Use code SAVE15 for 15% off. Applies to regular and sale items. Expires {expiry_date}.",
        ],
        "warranty": [
            "2-year manufacturer's warranty covers defects. Extended 3-year protection available for $19.99. Accidental damage coverage sold separately.",
            "1-year limited warranty. Protection plan with accidental damage coverage available at checkout.",
            "Lifetime warranty on defects. We stand behind our products 100%. Hassle-free replacement.",
        ],
        "colors|sizes": [
            "Available in 8 colors: black, white, navy, gray, red, blue, green, and burgundy. Sizes XS through 3XL.",
            "We stock black, white, navy, olive, and burgundy. Other colors available through special order (2-3 weeks).",
            "5 color options in stock. Custom colors available. All standard sizes plus extended sizes.",
        ],
        "sales": [
            "Weekly flash sales on Thursdays. Seasonal sales in spring and fall. Subscribe to newsletter for early access.",
            "50% off clearance items! Sale items updated daily. End-of-season sale starts next month.",
            "Daily deals on featured items. Weekend mega sales starting Friday. Email subscribers get 24-hour advance notice.",
        ],
        "delivery": [
            "Standard delivery takes 5-7 business days. Express (2-3 days) or overnight available. Track your order real-time.",
            "Most orders arrive within 3-5 business days. Weekend delivery available in select areas.",
            "Typical delivery is 4-6 days. Expedited options: next-day ($25) or 2-day ($10).",
        ],
    },
    "gaming": {
        "level_up": [
            "Focus on completing daily quests for 2x XP. Grind specific dungeons for fastest progression. Use XP boost items strategically.",
            "Join a guild for bonus XP. Complete side quests for 30% extra experience. Level up every 2-3 hours with optimal strategy.",
            "Farm high-level enemies in zones 5-7. Use experience scrolls from the shop. Average 1-2 levels per hour.",
        ],
        "strategies": [
            "Build tanky character with shield skills. Use crowd control abilities for group battles. Range from distance, heal teammates.",
            "Stack attack speed and critical hit chance. Kite enemies to avoid damage. Use ultimate at 50% enemy health for max damage.",
            "Balanced build: 40% attack, 30% defense, 30% healing. Combo moves for 2x damage. Always maintain high ground advantage.",
        ],
        "characters": [
            "Unlock warrior at level 5, mage at level 10, ranger at level 15, and paladin at level 20. Complete story missions for free characters.",
            "New characters available in premium shop or as quest rewards. Limited-time characters rotate monthly.",
            "Prestige system unlocks legendary characters after level 50. Collect hero shards to unlock faster.",
        ],
        "offline": [
            "Yes! Full offline mode available. Progress syncs when you go online. Offline play available for story and training modes.",
            "Story mode is fully offline. Multiplayer requires online connection. Auto-save every 10 minutes.",
            "Offline campaign available. Multiplayer and events require internet. Download offline content first.",
        ],
        "multiplayer": [
            "1v1 duels, 3v3 team battles, and 5v5 arena matches. Rankings reset weekly. Rewards based on tier placement.",
            "Queue for competitive ranked or casual matches. Average wait: 30 seconds. Cross-platform enabled.",
            "Co-op dungeons for 2-4 players. PvP arenas available. Tournament season with exclusive rewards.",
        ],
        "rewards": [
            "Daily logins: gems, gold, and rare items. Weekly challenges offer premium currency. Seasonal events with legendary loot.",
            "Earn gold from battles, gems from premium tasks, cosmetics from events. Battle pass offers 100+ rewards.",
            "Rewards system: 1st place $100 gems, 2nd $50 gems, 3rd $25 gems. Season pass includes exclusive skins.",
        ],
        "mobile": [
            "Mobile version available on iOS and Android. Cross-save feature. Same content as PC, optimized for touch.",
            "Yes! Full-featured mobile app. Cloud save syncs between devices. Controller support available.",
            "Mobile version launches next month. Pre-register for exclusive rewards. Tablet-optimized for better experience.",
        ],
        "clan": [
            "Go to social menu > create or join clan. Clans unlock shared benefits and weekly bonuses. Max 50 members.",
            "Clans offer 20% experience boost. Clan wars occur weekly. Clan store has exclusive items.",
            "Create: $10k gold investment. Join: request to clan leader. Clan benefits: shared chat, bonus rewards, guild dungeons.",
        ],
    },
    "finance": {
        "interest": [
            "Savings accounts earn 4.5% APY. Money market: 4.75% APY. CDs range from 3-5% depending on term.",
            "Current rates: 4.25% on standard savings, 4.50% on high-yield. Special rates for large deposits ($100k+).",
            "Competitive rates: 4.0% baseline, 4.5% for premium members, 5.0% for VIP accounts with $250k+ balance.",
        ],
        "account": [
            "Online account opens in 5 minutes. Initial deposit: $100 minimum. Free to maintain, no monthly fees.",
            "Open account online with ID verification. Same-day approval. Initial funding: $500 minimum for checking, $100 for savings.",
            "Simple process: 3 steps, 10 minutes. Minimum opening balance $50. No maintenance fees ever.",
        ],
        "fees": [
            "No monthly maintenance fees. ATM fees waived at 50,000+ networks. Overdraft fee: $35 (1 waived/month).",
            "Zero overdraft charges for the first time. Monthly fees: $0. Wire transfer: $15. Foreign transaction: 1.5%.",
            "Competitive pricing: checking $5/month (waived with $1k+ balance), overdraft $25, transfers $10.",
        ],
        "transfer": [
            "Transfers between your accounts: instant. To other banks: 1-3 business days. Wire transfers: same-day (fee: $15).",
            "ACH transfers process in 1-2 days. Internal transfers: immediate. International wire: 2-5 days, $30 fee.",
            "Domestic transfers: 2 hours to 3 days. International: 3-7 days. ACH: 1 day. No hidden fees.",
        ],
        "insured": [
            "Yes! FDIC insured up to $250,000 per account type. Additional coverage available through premium insurance.",
            "Deposits protected by FDIC insurance. $250k per customer per bank. Coverage includes principal and interest.",
            "Full protection: FDIC insurance covers all accounts. Separate coverage for savings vs checking. Peace of mind guaranteed.",
        ],
        "loan": [
            "Personal loans from $1k-$50k at 5.9%-11.9% APR. 12-60 month terms. No collateral required.",
            "Quick approval! Same-day funding available. Rates based on credit score. Terms: 24-84 months.",
            "Competitive rates starting at 4.9%. Flexible terms. Fast approval: 24 hours. Online application.",
        ],
        "credit_score": [
            "Minimum 620 credit score for approval. Better rates available with 700+ scores. No credit history? Consider secured loan.",
            "Typical requirement: 650+. Scores under 600 require co-signer. Improve score with our free monitoring tool.",
            "Need 640+ for best rates. 580-639 possible with higher APR. Build credit with our starter program.",
        ],
        "balance": [
            "Check balance 24/7 via app, website, or call 1-800-BANK. Real-time balance always available.",
            "Real-time balance in mobile app. Website updates within seconds. Phone support available 24/7.",
            "Instant balance check: mobile app (fastest), website, ATM, or phone. Multi-account dashboard available.",
        ],
    },
    "healthcare": {
        "appointment": [
            "Book online 24/7 through our portal. Phone: 1-800-DOC-HELP. Walk-ins available with 2-hour wait average.",
            "Schedule appointments online for next 30 days. Phone booking: Mon-Fri 8am-6pm. Same-day available based on provider.",
            "Online scheduling shows real-time availability. Book appointments 2+ weeks in advance for better times.",
        ],
        "insurance": [
            "We accept 50+ major insurance plans including Aetna, Blue Cross, Cigna, and UnitedHealth. Verify coverage in patient portal.",
            "In-network with most plans. Self-pay rates available. Insurance questions? Call billing at 1-800-BILLING.",
            "Accepted insurance: all major providers. Payment plans available if uninsured. Financial assistance programs offered.",
        ],
        "hours": [
            "Monday-Friday 8:00am-6:00pm, Saturday 9:00am-2:00pm. Emergency clinic open 24/7. Telehealth available weekdays 6am-11pm.",
            "Regular hours: 7am-7pm weekdays, 8am-5pm weekends. Urgent care until midnight. Virtual appointments anytime.",
            "Clinic hours: M-F 8am-8pm, Sat 9am-5pm, Sun 11am-4pm. ER always open. After-hours nurse line: 1-800-NURSE.",
        ],
        "telemedicine": [
            "Yes! Video visits available for most conditions. Book via patient portal. Same-day appointments usually available.",
            "Telehealth consultations for general health, prescriptions, and follow-ups. $39 per visit. Insurance coverage available.",
            "Virtual care available: minor illnesses, medication refills, mental health. Prescription delivery to your pharmacy.",
        ],
        "appointment_prep": [
            "Bring insurance card, ID, and list of current medications. Arrive 15 minutes early for check-in. Update medical history if changed.",
            "Prepare: insurance info, medications list, symptoms timeline, questions for doctor. Allergy information critical.",
            "Bring: insurance card, ID, recent lab results. New patients: arrive 20 mins early. Fasting required for blood work appointments.",
        ],
        "wait": [
            "Average wait time: 15 minutes. Acute care: 30-45 minutes. Book appointment to minimize wait. Real-time wait times available.",
            "Typical wait with appointment: 10-20 minutes. Walk-ins: 45+ minutes. Peak hours: 11am-2pm.",
            "On average 12 minutes with appointment. 1-2 hour waits for walk-ins during peak times. Afternoon appointments faster.",
        ],
        "cancellation": [
            "Cancel online or call 24 hours in advance. No fee if cancelled >24 hours. Late cancellation: $25 fee.",
            "Free cancellation with 24-hour notice. Within 24 hours: $50 fee. Multiple no-shows: appointment privileges suspended.",
            "Cancel anytime online. 48-hour notice: no penalty. Less notice: 50% of visit charge applies.",
        ],
        "emergency": [
            "Yes, 24-hour emergency department. Call 911 or go to ER directly. Life-threatening issues: emergency transport provided.",
            "Full ER services available around the clock. Trauma center designation. Pediatric emergency specialists available.",
            "Emergency department on-site. Critical care ICU. Helicopter transport available. Level 1 trauma center.",
        ],
    },
    "education": {
        "admission": [
            "Requirements: High school diploma/GED, 3.0+ GPA, SAT 1050+. Application: $50 fee. Rolling admissions.",
            "Entrance exams required. High school transcripts. Letters of recommendation (2). Personal statement.",
            "Acceptance based on GPA, test scores, and essays. Rolling admission through {month}. Early decision deadline: {date}.",
        ],
        "tuition": [
            "In-state: $25,000/year. Out-of-state: $45,000/year. Room & board: $15,000/year. Payment plans available.",
            "Tuition: $28k. Fees: $2k. Housing: $12k. Books: $1.5k. Total: ~$43.5k/year.",
            "Annual cost: $30-50k depending on program. Scholarships reduce cost by 20-60%. Financial aid available.",
        ],
        "programs": [
            "Bachelor's: 50+ majors including engineering, business, arts, sciences. Master's: 20 programs. Certificates available.",
            "Undergraduate: business, STEM, humanities, arts. Graduate: MBA, MS, MEd. Online degrees available.",
            "100+ bachelor programs, 40+ masters, 15+ doctoral. Online and on-campus options.",
        ],
        "class_size": [
            "Average class size: 35 students. First-year seminars: 15-20. Upper-level courses: 20-30 students.",
            "Small lectures: 100-150 students. Regular classes: 25-40. Seminars: 10-15 students.",
            "Intro courses average 200 students. Mid-level: 40-60. Advanced: 15-25 students.",
        ],
        "scholarships": [
            "Merit scholarships: $5k-$25k/year based on GPA. Need-based aid available. 70% of students receive aid.",
            "Full-ride scholarships available. Athletic scholarships offered. Financial aid meeting 95% of need.",
            "Scholarships: merit $10-20k, need-based $5-15k, athletic full ride. Apply FAFSA for aid.",
        ],
        "placement": [
            "92% of graduates employed within 6 months. Average starting salary: $65k. Top employers: Google, Microsoft, JPMorgan.",
            "Job placement rate: 95%. Average salary: $60-70k. Career services provides lifetime support.",
            "Placement: 88%. Salary average: $62k. Alumni network: 100k+ in Fortune 500.",
        ],
        "online": [
            "Yes! Full online degree programs available. Same curriculum as on-campus. Asynchronous and live options.",
            "Online courses available for most programs. Synchronous and self-paced options. Same degree awarded.",
            "Complete online bachelor/master available. Flexible schedule. Full-time or part-time.",
        ],
        "campus": [
            "Beautiful 250-acre campus. Modern facilities including new science center. Student center, library, sports complex.",
            "Urban campus location. State-of-the-art labs. Green spaces. Residence halls for 80% of students.",
            "Scenic location, 200+ acres. Modern classrooms, library, fitness center. On-campus housing.",
        ],
    },
    "technology": {
        "install": [
            "Install via: pip install package_name. Or npm install for Node.js. Requires Python 3.7+ or Node 14+.",
            "Installation methods: pip (easiest), conda, or source. Documentation: {docs_link}",
            "Quick install: pip install [package]. Detailed instructions in README. Dependencies auto-install.",
        ],
        "api_endpoint": [
            "Base URL: https://api.example.com/v1/. Authentication via header. Rate limit: 1000 requests/hour.",
            "Endpoint format: /api/v2/{resource}/{id}. SSL/TLS required. JSON responses.",
            "Primary endpoint: api.service.com. Endpoints documented at docs.service.com. Multiple regions available.",
        ],
        "authenticate": [
            "API key authentication. Include in header: Authorization: Bearer {api_key}. Keys generated in dashboard.",
            "OAuth 2.0 supported. JWT tokens. API keys available in admin panel.",
            "Authentication: API key, OAuth, JWT. Tokens expire after 24 hours. Refresh token support.",
        ],
        "rate_limits": [
            "Rate limit: 1000 req/min for free tier, 10000 req/min for premium. Resets every minute. Returns 429 when exceeded.",
            "Free: 100 requests/day. Pro: unlimited. Premium: priority queue. Check headers for remaining quota.",
            "Default: 60 req/min. Premium tiers: 600-6000 req/min. Burst allowed: 10x for 1 second.",
        ],
        "code_examples": [
            "Examples in Python, JavaScript, and cURL available. Full sample applications in GitHub repo.",
            "Documentation includes: REST examples, SDK usage, Webhook setup. Postman collection available.",
            "Code samples: Python, Node.js, Java, Go. Interactive examples: {link}",
        ],
        "response_format": [
            "JSON format. Success: 200 OK. Error: 400/401/500 with error object. Fields: data, status, message.",
            "Response structure: {status, data, pagination}. Timestamps in ISO 8601. Nullable fields omitted.",
            "JSON with metadata. HTTP status codes. Error details in response body. Array pagination with cursors.",
        ],
        "bug_report": [
            "Report bugs: support@example.com or GitHub issues page. Include: version, steps, error message, logs.",
            "Bug tracker: GitHub. Email: bugs@service.com. Include reproduction steps. Expected: response within 24hrs.",
            "File issue on GitHub. Include: OS, version, reproducibility. Critical bugs: email support immediately.",
        ],
        "sla": [
            "SLA: 99.9% uptime. Response time: < 500ms median. Support response: 1 hour for critical, 24 hours standard.",
            "SLA 99.95% uptime guaranteed. Performance: p95 < 1s. Critical support: 24/7. Emergency hotline available.",
            "Enterprise SLA: 99.99% uptime. Support: 1-hour response. 24/7 dedicated engineer.",
        ],
    },
    "corporate": {
        "mission": [
            "Our mission: Deliver innovative solutions that drive business growth. We empower companies through technology and partnerships.",
            "Mission statement: Transform businesses through digital innovation. Our values: integrity, excellence, partnership.",
            "Purpose: Help enterprises achieve their goals through cutting-edge solutions. Core values: quality, innovation, customer-focus.",
        ],
        "sales": [
            "Contact sales: sales@example.com or 1-800-SALES-1. Sales team available Mon-Fri 8am-6pm. Schedule demo.",
            "Sales contact: speak with representative. Live chat available weekdays. Email: enterprise@company.com.",
            "Reach sales team: website form, phone, or email. Response within 24 hours. Demo available immediately.",
        ],
        "hours": [
            "Business hours: Mon-Fri 8:00am-6:00pm EST, Sat 9:00am-2:00pm. Support 24/7 for critical issues.",
            "Hours: M-F 8am-8pm, Sat 10am-4pm. Emergency support available 24/7. Holidays: limited coverage.",
            "Standard: M-F 9am-5pm. Extended hours: M-F 8am-8pm. Premium support: 24/7.",
        ],
        "enterprise": [
            "Enterprise plans starting at $10k/month. Custom solutions available. Dedicated account manager. SLA: 99.99% uptime.",
            "Enterprise tier includes: custom integration, priority support, 24/7 SLA, volume discounts.",
            "Enterprise pricing: negotiable. Includes: white-label, custom features, dedicated support.",
        ],
        "response_time": [
            "Response time: critical (1 hour), high (4 hours), medium (24 hours), low (48 hours).",
            "Standard: 24-hour response. Premium: 4-hour response. Critical: 1-hour response.",
            "Average response: 12 hours. Critical: 2 hours. SLA guarantees: -5% credit if missed.",
        ],
        "case_studies": [
            "Case studies available: Fortune 500 deployments, 40% ROI increase, 60% cost reduction. Download: {link}",
            "Success stories: Samsung, Microsoft, Amazon. Results: 50%+ productivity gain. See website for details.",
            "Customer stories: healthcare, finance, retail sectors. Results: $5M+ savings annually. Whitepaper available.",
        ],
        "partnerships": [
            "Strategic partnerships with Microsoft, AWS, Google Cloud. Integration partners: Salesforce, SAP, Oracle.",
            "Partners: AWS, Google, Microsoft, IBM. Channel partners in 50+ countries. Reseller program available.",
            "Technology partners: all major cloud providers. System integrators: Accenture, Deloitte, others.",
        ],
        "demo": [
            "Schedule demo: book.example.com or call sales. Demo duration: 30 minutes. Personalized to your use case.",
            "Book demo online or call. Live walkthrough of features. Free trial: 14 days. No credit card required.",
            "Demo available immediately. 1-hour interactive session. Custom demo based on industry. Trial: 30 days.",
        ],
    },
}

# ========== TRAINING DATA GENERATOR ==========

def get_question_keyword(question: str, domain: str) -> str:
    """Extract question type keyword from question text (FAST)."""
    question_lower = question.lower()
    
    # Quick keyword mapping for each domain
    keyword_map = {
        "hospitality": {
            "room": "room_types|amenities",
            "reservation": "reservation|booking",
            "cancel": "cancellation",
            "discount": "discounts",
            "check": "check_in|check_out",
            "pet": "pets",
            "parking": "parking",
            "restaurant": "restaurants",
        },
        "retail": {
            "stock": "stock|availability",
            "return": "return",
            "ship": "shipping",
            "coupon": "coupon|discount",
            "warranty": "warranty",
            "color": "colors|sizes",
            "sale": "sales",
            "deliver": "delivery",
        },
        "gaming": {
            "level": "level_up",
            "strateg": "strategies",
            "character": "characters",
            "offline": "offline",
            "multiplayer": "multiplayer",
            "reward": "rewards",
            "mobile": "mobile",
            "clan": "clan",
        },
        "finance": {
            "interest": "interest",
            "account": "account",
            "fee": "fees",
            "transfer": "transfer",
            "insur": "insured",
            "loan": "loan",
            "credit": "credit_score",
            "balance": "balance",
        },
        "healthcare": {
            "appointment": "appointment",
            "insurance": "insurance",
            "hour": "hours",
            "telemedicine": "telemedicine",
            "bring": "appointment_prep",
            "wait": "wait",
            "cancel": "cancellation",
            "emergency": "emergency",
        },
        "education": {
            "admission": "admission",
            "tuition": "tuition",
            "program": "programs",
            "class": "class_size",
            "scholarship": "scholarships",
            "placement": "placement",
            "online": "online",
            "campus": "campus",
        },
        "technology": {
            "install": "install",
            "endpoint": "api_endpoint",
            "authenticate": "authenticate",
            "rate": "rate_limits",
            "example": "code_examples",
            "response": "response_format",
            "bug": "bug_report",
            "sla": "sla",
        },
        "corporate": {
            "mission": "mission",
            "sales": "sales",
            "hour": "hours",
            "enterprise": "enterprise",
            "response": "response_time",
            "case": "case_studies",
            "partner": "partnerships",
            "demo": "demo",
        },
    }
    
    # Find keyword for this domain
    domain_keywords = keyword_map.get(domain, {})
    for keyword, answer_key in domain_keywords.items():
        if keyword in question_lower:
            return answer_key
    
    # Default fallback
    return list(domain_keywords.values())[0] if domain_keywords else None


def generate_training_record(
    domain: str,
    intent: str,
    tone: str,
    question: str,
    industry_data: dict,
    touchpoint_data: dict,
    platform_data: dict
) -> dict:
    """
    Generate a single training record with system prompt and REAL answer.
    OPTIMIZED FOR SPEED - Uses pre-loaded answer database.
    """
    # Create system prompt
    domain_instruction = f"You are a {domain.replace('_', ' ')} specialist. Speak in {industry_data['tone']} tone."
    tone_instruction = f"Your communication style: {platform_data['style']}"
    intent_instruction = f"User Intent: {touchpoint_data['description']}"
    
    system_prompt = f"""{domain_instruction}
{tone_instruction}
{intent_instruction}

Be concise, helpful, and relevant to the user's immediate need."""

    # FAST: Get answer from pre-loaded database
    answer_key = get_question_keyword(question, domain)
    
    if answer_key and domain in ANSWER_DATABASE and answer_key in ANSWER_DATABASE[domain]:
        answers = ANSWER_DATABASE[domain][answer_key]
        response = random.choice(answers)  # Vary answers for robustness
    else:
        # Fallback: generic helpful response
        response = f"Great question! We're here to help. {question.capitalize()} Feel free to ask for more specific information."
    
    # Create training record in OpenAI format
    record = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }
    
    return record


def generate_all_training_data():
    """
    Generate training data for all domain × tone × intent combinations.
    Strictly separates records by domain for later filtering.
    """
    records = []
    for domain, industry_data in INDUSTRY_DOMAINS.items():
        for touchpoint, touchpoint_data in TOUCHPOINT_INTENT_MAP.items():
            for platform, platform_data in PLATFORM_TONE_MAP.items():
                intent = touchpoint_data["primary_intent"]
                tone = platform_data["tone"]
                questions = GENERAL_QUESTIONS.get(domain, ["What can you help me with?"])
                question = random.choice(questions)
                record = generate_training_record(domain, intent, tone, question, industry_data, touchpoint_data, platform_data)
                records.append(record)
    return records


def save_training_data(records: list, output_file: str = "FineTuning/domain_tone_intent_general_training.jsonl"):
    """
    Save training records to JSONL file.
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    return output_file


def add_diverse_examples(records, extra_count=20):
    """
    Add diverse, edge-case examples for robustness.
    """
    diverse_questions = [
        "Can you help me understand the basics?",
        "What's the best way to get started?",
        "Are there any common mistakes I should avoid?",
        "How does this compare to alternatives?",
        "What's the typical timeline for this?",
        "Can you provide examples?",
        "What are the costs involved?",
        "Is there customer support available?",
        "Can I try this before committing?",
        "What are the success rates?",
        "How secure is this?",
        "What's the learning curve?",
        "Are there any limitations I should know?",
        "Can I get a demo?",
        "What do current customers say?",
    ]
    for _ in range(extra_count):
        domain = random.choice(list(INDUSTRY_DOMAINS.keys()))
        platform = random.choice(list(PLATFORM_TONE_MAP.keys()))
        touchpoint = random.choice(list(TOUCHPOINT_INTENT_MAP.keys()))
        question = random.choice(diverse_questions)
        intent = TOUCHPOINT_INTENT_MAP[touchpoint]["primary_intent"]
        tone = PLATFORM_TONE_MAP[platform]["tone"]
        record = generate_training_record(domain, intent, tone, question, INDUSTRY_DOMAINS[domain], TOUCHPOINT_INTENT_MAP[touchpoint], PLATFORM_TONE_MAP[platform])
        records.append(record)
    return records


def validate_training_data(records):
    """
    Validate training data structure.
    """
    if not records:
        print("❌ No records generated!")
        return False
    for i, record in enumerate(records):
        if "messages" not in record:
            print(f"❌ Record {i} missing 'messages' field")
            return False
        messages = record["messages"]
        if len(messages) != 3:
            print(f"❌ Record {i} has {len(messages)} messages, expected 3")
            return False
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            print(f"❌ Record {i} has incorrect roles: {roles}")
            return False
    print(f"✅ All {len(records)} records valid!")
    return True


def generate_domain_training_data(domain, output_file):
    """
    Generate and save training data for a single domain only.
    """
    print(f"\nGenerating training data for domain: {domain}")
    all_records = generate_all_training_data()
    # Filter strictly by domain prompt
    domain_prompt = {
        "hospitality": "hospitality support assistant",
        "healthcare": "healthcare support assistant",
        "technology": "technology support assistant",
        "education": "education support assistant",
    }.get(domain, domain)
    records = [r for r in all_records if domain_prompt in r["messages"][0]["content"].lower()]
    records = add_diverse_examples(records, extra_count=20)
    records = [r for r in records if domain_prompt in r["messages"][0]["content"].lower()]
    if validate_training_data(records):
        save_training_data(records, output_file)
        print(f"✅ Saved {len(records)} records for domain '{domain}' to {output_file}")
    else:
        print(f"❌ Validation failed for domain '{domain}'")


def main():
    print("\n" + "="*60)
    print("DOMAIN × TONE × INTENT TRAINING DATA GENERATOR")
    print("="*60 + "\n")
    records = generate_all_training_data()
    print(f"Generated {len(records)} base training records.")
    records = add_diverse_examples(records, extra_count=50)
    print(f"Total records after diverse examples: {len(records)}")
    if not validate_training_data(records):
        print("Validation failed!")
        return
    output_file = save_training_data(records)
    print(f"Training data saved to: {output_file}")
    # Generate strictly separated domain files
    for domain in ["hospitality", "technology", "education"]:
        domain_file = f"FineTuning/{domain}_training.jsonl"
        generate_domain_training_data(domain, domain_file)
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    print(f"Total records: {len(records)}")
    print(f"Domains: {len(INDUSTRY_DOMAINS)}")
    print(f"Intents: {len(TOUCHPOINT_INTENT_MAP)}")
    print(f"Tones: {len(PLATFORM_TONE_MAP)}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    print("="*60)
    print("Next steps:")
    print("1. Review generated data")
    print("2. Fine-tune model with this data")
    print("3. Test with validate_fine_tuned_model()\n")

if __name__ == "__main__":
    main()
