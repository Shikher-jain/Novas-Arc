import json
from collections import OrderedDict


# Path to the chat log file
LOG_PATH = "chat_ui_logs.jsonl"
CLEAN_PATH = "chat_ui_logs_cleaned.jsonl"

# List of params to compare for duplicates. Edit this to select which fields to compare.
# compare_fields = ["user_prompt", "response", "rating", "rating_msg"]
compare_fields = ["user_prompt", "response", ]

def safe_strip(val):
    if val is None:
        return ""
    return str(val).strip()

def is_duplicate(entry1, entry2):
    """
    Returns True if all selected params in compare_fields are the same.
    Handles None values safely.
    """
    for field in compare_fields:
        v1 = entry1.get(field)
        v2 = entry2.get(field)
        # For string fields, compare after stripping
        if isinstance(v1, str) or isinstance(v2, str) or field in ["user_prompt", "response", "rating_msg"]:
            if safe_strip(v1) != safe_strip(v2):
                return False
        else:
            if v1 != v2:
                return False
    return True

def clean_log(input_path=LOG_PATH, output_path=CLEAN_PATH):
    seen = []
    cleaned = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            # Skip entries with unwanted response
            if safe_strip(entry.get("response")) == "Not available in training data.":
                continue
            # Check for duplicate
            if any(is_duplicate(entry, prev) for prev in seen):
                continue
            seen.append(entry)
            cleaned.append(entry)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in cleaned:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Cleaned log written to {output_path}. {len(cleaned)} entries remain.")

if __name__ == "__main__":
    clean_log()
