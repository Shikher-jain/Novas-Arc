import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import json

# === Load Environment Variables ===
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TRAINING_FILE_PATH = "FineTuning/rentomojodesk_freshdesk_com.jsonl"
BASE_MODEL = "gpt-3.5-turbo"

def validate_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ok = False
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                ok = True
                try:
                    data = json.loads(line)
                except Exception as e:
                    logging.error(f"Line {i}: Invalid JSON - {e}")
                    return False

                if not isinstance(data, dict):
                    logging.error(f"Line {i}: JSON object expected.")
                    return False

                msgs = data.get('messages')
                if not isinstance(msgs, list) or len(msgs) < 2:
                    logging.error(f"Line {i}: 'messages' must be a list with at least user+assistant messages.")
                    return False

                # Validate each message entry
                for j, m in enumerate(msgs, 1):
                    if not isinstance(m, dict):
                        logging.error(f"Line {i}, message {j}: message must be an object.")
                        return False
                    role = m.get('role')
                    content = m.get('content')
                    if role not in ('system', 'user', 'assistant'):
                        logging.error(f"Line {i}, message {j}: invalid or missing role '{role}'.")
                        return False
                    if not isinstance(content, str) or not content.strip():
                        logging.error(f"Line {i}, message {j}: 'content' must be a non-empty string.")
                        return False

                # Optional: warn if messages too long
                total_len = sum(len(m.get('content', '')) for m in msgs if isinstance(m, dict))
                if total_len > 5000:
                    logging.warning(f"Line {i}: Total message length > 5000 characters; consider splitting.")

            if not ok:
                logging.error("Training file is empty.")
                return False
        return True
    except Exception as e:
        logging.error(f"JSONL validation error: {e}")
        return False

# Retry logic for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
def upload_file():
    logging.info("Uploading training file to OpenAI...")
    with open(TRAINING_FILE_PATH, "rb") as fh:
        return client.files.create(
            file=fh,
            purpose="fine-tune"
        )

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=4, max=20))
def start_fine_tune(file_id):
    logging.info("Starting fine-tuning job...")
    return client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL
    )

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=4, max=20))
def get_job_status(job_id):
    return client.fine_tuning.jobs.retrieve(job_id)

try:
    # Validate JSONL before upload
    if not validate_jsonl(TRAINING_FILE_PATH):
        logging.error("Training file failed JSONL validation. Aborting upload.")
        exit()

    # === STEP 1: Upload the training file ===
    upload_response = upload_file()
    file_id = upload_response.id
    logging.info(f"Uploaded File ID: {file_id}")

    # === STEP 2: Start fine-tuning job ===
    fine_tune_job = start_fine_tune(file_id)
    job_id = fine_tune_job.id
    logging.info(f"Fine-tuning started: {job_id}")

    # === STEP 3: Monitor the job ===
    logging.info("Waiting for fine-tuning to complete...")
    while True:
        job_status = get_job_status(job_id)
        status = job_status.status
        logging.info(f"Status: {status}")
        if status in ["succeeded", "failed", "cancelled"]:
            break
        time.sleep(15)

    if status == "succeeded":
        # === STEP 4: Retrieve fine-tuned model ID ===
        fine_tuned_model = job_status.fine_tuned_model
        logging.info(f"Fine-tuning complete! Model ID: {fine_tuned_model}")
    else:
        logging.error(f"Fine-tuning failed or was cancelled. Final status: {status}")
        raise RuntimeError(f"Fine-tuning job failed with status: {status}")

except Exception as e:
    logging.error(f"An error occurred: {e}")
