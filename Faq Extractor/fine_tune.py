import os
import time
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# === CONFIGURATION ===
training_file_path = "FineTuning/rentomojodesk_freshdesk_com.jsonl"  # your training dataset
base_model = os.getenv("BASE_MODEL", "gpt-3.5-turbo")  # fallback if not in .env

# === DYNAMIC PARAMETER ANALYSIS ===
def analyze_dataset_and_suggest_params(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        dataset_size = len(lines)
        logging.info(f"Training samples detected: {dataset_size}")

        # Dynamic parameter selection logic
        if dataset_size < 50:
            n_epochs = 10
            batch_size = 1
            lr_multiplier = 0.3
        elif dataset_size < 300:
            n_epochs = 6
            batch_size = 2
            lr_multiplier = 0.25
        elif dataset_size < 1000:
            n_epochs = 5
            batch_size = 4
            lr_multiplier = 0.2
        else:
            n_epochs = 3
            batch_size = 8
            lr_multiplier = 0.15

        return n_epochs, batch_size, lr_multiplier
    except Exception as e:
        raise RuntimeError(f"Error analyzing dataset: {e}")

# === UPLOAD TRAINING FILE ===
def upload_training_file(path):
    logging.info("Uploading training file...")
    try:
        with open(path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="fine-tune")
        logging.info(f"File uploaded successfully. File ID: {uploaded_file.id}")
        return uploaded_file.id
    except Exception as e:
        raise RuntimeError(f"File upload failed: {e}")

# === CREATE FINE-TUNING JOB ===
def create_fine_tune_job(file_id, n_epochs, batch_size, learning_rate_multiplier):
    logging.info("Creating fine-tuning job...")
    try:
        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model,
            hyperparameters={
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate_multiplier
            }
        )
        logging.info(f"Fine-tuning started. Job ID: {job.id}")
        return job.id
    except Exception as e:
        raise RuntimeError(f"Failed to create fine-tuning job: {e}")

# === MONITOR JOB STATUS ===
def monitor_job(job_id, check_interval=30):
    logging.info("Monitoring fine-tune job progress...")
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            logging.info(f"Current status: {status}")
            if status in ["succeeded", "failed", "cancelled"]:
                logging.info(f"Final status: {status}")
                if status == "succeeded":
                    logging.info(f"Fine-tuned model name: {job.fine_tuned_model}")
                break
            time.sleep(check_interval)
        except Exception as e:
            logging.error(f"Error fetching job status: {e}")
            time.sleep(check_interval)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        logging.info("Analyzing dataset for parameter optimization...")
        n_epochs, batch_size, lr_multiplier = analyze_dataset_and_suggest_params(training_file_path)

        logging.info(f"Auto-selected parameters:")
        logging.info(f"  Epochs: {n_epochs}")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Learning rate multiplier: {lr_multiplier}")

        file_id = upload_training_file(training_file_path)
        job_id = create_fine_tune_job(file_id, n_epochs, batch_size, lr_multiplier)
        monitor_job(job_id)
    except Exception as e:
        logging.error(f"Fatal Error: {e}")
