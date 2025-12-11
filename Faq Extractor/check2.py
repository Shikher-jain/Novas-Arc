import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Get keys and model from .env
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file.")
if not model:
    raise ValueError("Missing MODEL in .env file.")

# Initialize client
client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configurable parameters
TEMPERATURE = 0.1
MAX_TOKENS = 300
PRESENCE_PENALTY = 0.2
FREQUENCY_PENALTY = 0.2

logging.info(f"Using fine-tuned model: {model}")
logging.info("Type your question (or 'exit' to quit)")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def get_model_response(client, model, user_input):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You must respond exactly as in the training data, without rewording or variation."},
            {"role": "user", "content": user_input}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
    )

while True:
    try:
        user_input = input(" You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            logging.info("Exiting chat. Have a great day!")
            break

        if not user_input:
            logging.info("Please enter a valid question.")
            continue

        # API call with retry
        try:
            response = get_model_response(client, model, user_input)
            model_reply = response.choices[0].message.content.strip()
            logging.info(f" Model: {model_reply}")
        except Exception as api_error:
            logging.error(f"API Error: {api_error}")
            logging.info("Retrying after short delay...")

        # small delay between requests
        # time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Chat interrupted by user. Exiting safely.")
        break

    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        logging.info("Hint: Check internet, API key, or model configuration.")