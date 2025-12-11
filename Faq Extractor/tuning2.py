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

# Update to organize model IDs by training file name in subdirectories
MODEL_ID_DIR = "model_ids"
os.makedirs(MODEL_ID_DIR, exist_ok=True)

def get_model_subdir(training_file):
    # Create a subdirectory based on the training file name (without extension)
    subdir_name = os.path.splitext(os.path.basename(training_file))[0]
    subdir_path = os.path.join(MODEL_ID_DIR, subdir_name)
    os.makedirs(subdir_path, exist_ok=True)
    return subdir_path

def save_model_data(training_file, file_id=None, job_id=None, model_id=None):
    subdir = get_model_subdir(training_file)
    if file_id:
        with open(os.path.join(subdir, "file_id.txt"), "w") as f:
            f.write(file_id)
    if job_id:
        with open(os.path.join(subdir, "job_id.txt"), "w") as f:
            f.write(job_id)
    if model_id:
        with open(os.path.join(subdir, "model_id.txt"), "w") as f:
            f.write(model_id)

def load_model_data(training_file):
    subdir = get_model_subdir(training_file)
    data = {}
    for key in ["file_id", "job_id", "model_id"]:
        file_path = os.path.join(subdir, f"{key}.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data[key] = f.read().strip()
    return data

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

    # Load existing data if available
    model_data = load_model_data(TRAINING_FILE_PATH)

    # === STEP 1: Upload the training file ===
    file_id = model_data.get("file_id")
    if not file_id:
        upload_response = upload_file()
        file_id = upload_response.id
        save_model_data(TRAINING_FILE_PATH, file_id=file_id)
        logging.info(f"Uploaded File ID: {file_id}")
    else:
        logging.info(f"Using previously uploaded File ID: {file_id}")

    # === STEP 2: Start fine-tuning job ===
    job_id = model_data.get("job_id")
    if not job_id:
        fine_tune_job = start_fine_tune(file_id)
        job_id = fine_tune_job.id
        save_model_data(TRAINING_FILE_PATH, job_id=job_id)
        logging.info(f"Fine-tuning started: {job_id}")
    else:
        logging.info(f"Using previously started Job ID: {job_id}")

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

        # Save the model ID with the name of the training file
        save_model_data(TRAINING_FILE_PATH, model_id=fine_tuned_model)
    else:
        logging.error(f"Fine-tuning failed or was cancelled. Final status: {status}")
        raise RuntimeError(f"Fine-tuning job failed with status: {status}")

except Exception as e:
    logging.error(f"An error occurred: {e}")
