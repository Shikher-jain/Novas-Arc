import os
import logging
import openai
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_model_id_from_file(training_file):
    output_file = os.path.splitext(training_file)[0] + ".txt"
    model_id_dir = os.path.join(os.path.dirname(training_file), "..", "model ids")
    model_id_dir = os.path.normpath(model_id_dir)
    out_path = os.path.join(model_id_dir, os.path.basename(output_file))
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Model ID file not found: {out_path}")
    with open(out_path, "r") as f:
        model_id = f.read().strip()
    return model_id

def get_system_prompt(training_file):
    try:
        with open(training_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if "messages" in data and data["messages"]:
                    for msg in data["messages"]:
                        if msg["role"] == "system":
                            return msg["content"]
    except Exception as e:
        logging.warning(f"Could not read system prompt from training file: {e}")
    return "You are a helpful assistant."

def chatbot_loop(model_id, system_prompt):
    print("Type 'exit' to quit the chatbot.")
    while True:
        user_prompt = input("Enter a prompt to test the fine-tuned model:\n> ")
        if user_prompt.strip().lower() == "exit":
            print("Exiting chatbot.")
            break
        try:
            response = openai.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200
            )
            print(f"Model response: {response.choices[0].message.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")

def main():
    training_file = input("Enter the path to the training file used for model ID (e.g., FineTuning/yourfile.jsonl): ").strip()
    if not os.path.exists(training_file):
        print(f"Training file not found: {training_file}")
        return
    try:
        model_id = get_model_id_from_file(training_file)
        print(f"Loaded model ID: {model_id}")
        system_prompt = get_system_prompt(training_file)
        chatbot_loop(model_id, system_prompt)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
