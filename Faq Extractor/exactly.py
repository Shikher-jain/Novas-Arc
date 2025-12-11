import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# === Load API key and model from .env ===
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL")

if not api_key or not model:
    raise ValueError("Missing OPENAI_API_KEY or MODEL in .env file.")

# === Initialize client ===
client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# === Inference Parameters ===
temperature = 0.0    # no randomness — deterministic
top_p = 0.01         # must be >= 0.01; 1.0 means use all tokens deterministically
max_tokens = 300     # maximum length of answer

logging.info(f"Using fine-tuned model: {model} (Type 'exit' or 'quit' to end)")

# === Chat Loop ===
while True:
    user_input = input("🧑‍💻 You: ").strip()
    if user_input.lower() in ["exit", "quit", "q"]:
        logging.info("Exiting chat.")
        break

    if not user_input:
        continue

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system","content":"You must respond exactly as trained. Do not rephrase, summarize, or alter wording."},
                {"role": "user", "content": user_input},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message.content.strip()
        if message:
            logging.info(f"🤖 Model: {message}")
        else:
            logging.warning("No response generated.")

    except Exception as e:
        logging.error(f"Error: {e}")
