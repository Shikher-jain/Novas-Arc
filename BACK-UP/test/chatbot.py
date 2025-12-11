import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Utility function to generate system prompt from analysis ---
def generate_system_prompt(analysis_json):
    """
    Converts the analysis JSON into a system prompt for the assistant,
    including your guidance, safety rules, and adaptive persona logic.
    """
    domain = analysis_json.get("Domain", "General")
    persona = analysis_json.get("Suggested_persona", "Helpful assistant")
    age_group = analysis_json.get("Target_age_group", "All ages")
    guidelines = analysis_json.get("Guidelines", "")
    tone = analysis_json.get("Tone", "friendly")
    intent = analysis_json.get("Intent", "inform")

    guidance_block = f"""
───────────────────────────────
### 🧩 Guidance & Adaptive Persona
- Mirror the user's tone, formality, and energy.
- Adapt explanations to the user's age group: {age_group}.
- Show empathy and understanding.
- Be concise if casual, detailed if formal.
- Use slang naturally for Gen Alpha or casual users.
- Provide clear, beginner-friendly examples.
- Follow domain-specific language and terminology.
- Expand responses to roughly half a page.
───────────────────────────────
"""

    safety_block = """
───────────────────────────────
### ⚠️ Safety and Conduct Guidelines
1. Never request personal or sensitive information.
2. Avoid inappropriate topics (sexual, violent, illegal, unsafe).
3. Avoid manipulative behavior or coercion.
4. Respect privacy, boundaries, and cultural context.
5. Always maintain professional, respectful, and safe tone.
───────────────────────────────
"""

    prompt = f"""
You are a {persona} specializing in {domain}.
Audience: {age_group}
Tone: {tone}
Intent: {intent}

Follow these guidelines when responding:
{guidelines}
{guidance_block}
{safety_block}

Provide clear, concise, beginner-friendly explanations.
Use examples, structured lists, or code blocks where helpful.
Always maintain a safe, professional, and empathetic tone.
"""
    return prompt.strip()


# --- Model setup ---
model = "ft:gpt-3.5-turbo-0125:novas-arc-consulting-pvt-ltd::CYRY1ep1"

print("💬 Chatbot is ready! Type 'exit' to quit.\n")

# --- Chat loop ---
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # --- Step 2: Analyze the query ---
    analysis_prompt = f"""
Analyze this user query and identify:
1. Domain
2. Intent
3. Tone
4. Suggested persona for the assistant
5. Target age group
6. Guidelines for answering

User Query: "{user_query}"

Return the result as a JSON object.
"""

    try:
        analysis_resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an intelligent assistant that classifies user intent, context, age group, and persona."},
                {"role": "user", "content": analysis_prompt}
            ]
        )
        analysis_json = json.loads(analysis_resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("Error: The analysis response was not valid JSON. Using default settings.")
        analysis_json = {}

    # --- Step 3: Generate dynamic system prompt ---
    system_prompt = generate_system_prompt(analysis_json)

    # --- Step 4: Generate the final response ---
    final_response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    # --- Step 5: Output ---
    print("\n🤖 Assistant:", final_response.choices[0].message.content, "\n")
