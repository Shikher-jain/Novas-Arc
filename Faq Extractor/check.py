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

# Simplified parameter dictionary for clarity
PARAMETERS = {
    "temperature": 0.2,  # Sampling temperature; higher = more random
    "max_tokens": 300,  # Maximum tokens for the completion
    "top_p": 1,  # Nucleus sampling threshold
    "n": 1,  # Number of completion choices to generate
    "frequency_penalty": 0.5,  # Penalize tokens based on frequency
    "presence_penalty": 0.0,  # Penalize based on token presence
}

# Update system prompt for better clarity and handling of unrecognized inputs
SYSTEM_PROMPT = (
    "You are a customer support assistant. Your role is to provide responses exactly as they appear in the training data. "
    "Do not reword, summarize, or omit any details. If the input does not match the training data, respond with: "
)

logging.info(f"Using fine-tuned model: {model}")
logging.info("Type your question (or 'exit' to quit)")

# Clarify temperature usage
# TEMPERATURE = 0.1 ensures the model strictly follows the training data with minimal randomness.
# Increase TEMPERATURE for more creative or diverse responses.
TEMPERATURE = 0.2

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def get_model_response(client, model, user_input):
    """
    Get a response from the model with retry logic.
    Retries up to 3 times with exponential backoff in case of API failures.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            **PARAMETERS
        )
        return response
    except TypeError as e:
        logging.error(f"TypeError in API call: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in API call: {e}")
        raise

# Add list view for displaying multiple responses or options
def display_list_view(items):
    """
    Display a list of items in a numbered list view.
    """
    if not items:
        logging.info("No items to display.")
        return

    logging.info("\nList View:")
    for idx, item in enumerate(items, start=1):
        logging.info(f"{idx}. {item}")

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
        except TypeError as api_error:
            logging.error(f"TypeError: {api_error}")
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