import os, json
import time
import logging
import openai
from dotenv import load_dotenv

# TRAINING_FILE_PATH = "FineTuning/aws_amazon_com.jsonl"
# TRAINING_FILE_PATH = "FineTuning/aws_amazon_com1.jsonl"
TRAINING_FILE_PATH = "FineTuning/all_domains.jsonl"
# TRAINING_FILE_PATH = "FineTuning/hospitality_training.jsonl"
# TRAINING_FILE_PATH = "FineTuning/technology_training.jsonl"
# TRAINING_FILE_PATH = "FineTuning/education_training.jsonl"
BASE_MODEL = "gpt-3.5-turbo"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_jsonl_structure(file_path):
    """
    Validate that the JSONL file has the expected structure for fine-tuning.
    Returns True if valid, False otherwise.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        logging.error(f"Line {i}: Not a JSON object.")
                        return False
                    if "messages" not in data or not isinstance(data["messages"], list):
                        logging.error(f"Line {i}: Missing or invalid 'messages' field.")
                        return False
                    roles = [msg.get("role") for msg in data["messages"]]
                    if "system" not in roles or "user" not in roles or "assistant" not in roles:
                        logging.error(f"Line {i}: Missing required roles in 'messages'.")
                        return False
                except Exception as e:
                    logging.error(f"Line {i}: Invalid JSON or structure. Error: {e}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Failed to read or validate JSONL file: {e}")
        return False
    
def upload_training_file(file_path):
    """
    Upload the training file to OpenAI for fine-tuning using the updated API.
    Validates JSONL structure before upload.

    Args:
        file_path (str): Path to the training file.

    Returns:
        str: The file ID of the uploaded file.
    """
    if not validate_jsonl_structure(file_path):
        raise ValueError("Training file failed JSONL structure validation. Aborting upload.")
    try:
        with open(file_path, "rb") as f:
            response = openai.files.create(
                file=f,
                purpose="fine-tune"
            )
        logging.info(f"Uploaded training file. File ID: {response.id}")
        return response.id
    except Exception as e:
        logging.error(f"Failed to upload training file: {e}")
        raise

def start_fine_tuning(file_id, base_model):
    """
    Start the fine-tuning process using the updated API.

    Args:
        file_id (str): The file ID of the uploaded training file.
        base_model (str): The base model to fine-tune.

    Returns:
        str: The fine-tuning job ID.
    """
    try:
        response = openai.fine_tuning.jobs.create(
            training_file=file_id,
            model=base_model
        )
        logging.info(f"Fine-tuning started. Job ID: {response.id}")
        return response.id
    except Exception as e:
        logging.error(f"Failed to start fine-tuning: {e}")
        raise

def monitor_fine_tuning(job_id):
    """
    Monitor the fine-tuning process until completion using the updated API.

    Args:
        job_id (str): The fine-tuning job ID.

    Returns:
        str: The fine-tuned model ID.
    """
    try:
        while True:
            job_status = openai.fine_tuning.jobs.retrieve(job_id)
            status = job_status.status
            logging.info(f"Fine-tuning status: {status}")
            if status in ["succeeded", "failed", "cancelled"]:
                break
            time.sleep(10)

        if status != "succeeded":
            logging.error(f"Fine-tuning failed or was cancelled. Final status: {status}")
            raise RuntimeError("Fine-tuning did not succeed.")

        logging.info(f"Fine-tuning complete. Model ID: {job_status.fine_tuned_model}")
        return job_status.fine_tuned_model
    except Exception as e:
        logging.error(f"Error monitoring fine-tuning: {e}")
        raise

def save_model_id(model_id, training_file):
    """
    Save the fine-tuned model ID next to the training file:
    e.g., FineTuning/<name>.jsonl -> FineTuning/<name>.txt
    """
    try:
        output_file = os.path.splitext(training_file)[0] + ".txt"  # FineTuning/<name>.txt
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(model_id)
        logging.info("Model ID saved to %s", output_file)
    except Exception as e:
        logging.error("Failed to save model ID: %s", e)

def main():
    # Ensure API key is set
    if not openai.api_key:
        raise RuntimeError("OpenAI API key is not set. Please set it in your environment variables.")

    # Ensure training file exists
    if not os.path.exists(TRAINING_FILE_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE_PATH}")

    # Check if model ID already exists for this training file
    model_id_file = os.path.splitext(TRAINING_FILE_PATH)[0] + ".txt"
    model_id = None
    if os.path.exists(model_id_file):
        with open(model_id_file, "r") as f:
            model_id = f.read().strip()
        logging.info(f"Found existing model ID for this training file: {model_id}")
    else:
        # Upload training file
        file_id = upload_training_file(TRAINING_FILE_PATH)
        # Start fine-tuning
        job_id = start_fine_tuning(file_id, BASE_MODEL)
        # Monitor fine-tuning
        model_id = monitor_fine_tuning(job_id)
        # Save the fine-tuned model ID to a file named after the training file
        save_model_id(model_id, TRAINING_FILE_PATH)
        logging.info(f"Model trained and ID saved for future use: {model_id}")

if __name__ == "__main__":
    main()


'''
When you upload a training file, you get a file ID (for the uploaded file).
When you start fine-tuning, you get a job ID (for the fine-tuning process).
When fine-tuning completes successfully, you get one fine-tuned model ID (the actual model you will use).
'''
