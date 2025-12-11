import os
import time
import logging
import openai
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
BASE_MODEL = "gpt-3.5-turbo"

def validate_jsonl_structure(file_path):
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
    try:
        output_file = os.path.splitext(training_file)[0] + ".txt"
        model_id_dir = os.path.join(os.path.dirname(training_file), "..", "model ids")
        model_id_dir = os.path.normpath(model_id_dir)
        os.makedirs(model_id_dir, exist_ok=True)
        out_path = os.path.join(model_id_dir, os.path.basename(output_file))
        with open(out_path, "w") as f:
            f.write(model_id)
        logging.info(f"Model ID saved to {out_path}")
    except Exception as e:
        logging.error(f"Failed to save model ID: {e}")

def select_training_file(folder):
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".jsonl")]
        if not files:
            print("No JSONL files found in the specified folder.")
            raise FileNotFoundError("No JSONL files found in the specified folder.")
        print("Available training files:")
        for i, file in enumerate(files, start=1):
            print(f"{i}. {file}")
        while True:
            try:
                choice = int(input("Select a training file by number: "))
                if 1 <= choice <= len(files):
                    return os.path.join(folder, files[choice - 1])
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Please enter a number.")
    except Exception as e:
        logging.error(f"Error selecting training file: {e}")
        raise

def main():
    if not openai.api_key:
        raise RuntimeError("OpenAI API key is not set. Please set it in your environment variables.")
    folder = input("Enter folder path containing training files (default: FineTuning): ").strip() or "FineTuning"
    training_file = select_training_file(folder)
    model_id_file = os.path.splitext(training_file)[0] + ".txt"
    model_id_dir = os.path.join(os.path.dirname(training_file), "..", "model ids")
    model_id_dir = os.path.normpath(model_id_dir)
    out_path = os.path.join(model_id_dir, os.path.basename(model_id_file))
    model_id = None
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            model_id = f.read().strip()
        logging.info(f"Found existing model ID for this training file: {model_id}")
    else:
        file_id = upload_training_file(training_file)
        job_id = start_fine_tuning(file_id, BASE_MODEL)
        model_id = monitor_fine_tuning(job_id)
        save_model_id(model_id, training_file)
        logging.info(f"Model trained and ID saved for future use: {model_id}")
    print(f"Model ID: {model_id}")

if __name__ == "__main__":
    main()
