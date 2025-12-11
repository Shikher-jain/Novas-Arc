# MultiContext2 — FAQ Extraction, Fine‑Tuning, and Ensemble Chatbot

A production‑ready toolkit for advanced FAQ extraction, chatbot fine-tuning, and multi-model ensemble chat.

## What does this code do?
- Crawls websites and extracts FAQ Q/A pairs (`app2.py`).
- Fine-tunes OpenAI models on those FAQs (`check_and_tune.py`).
- Runs an ensemble chatbot using multiple fine-tuned models (`multimodel.py`).
- Provides a modern Streamlit UI (`ui4.py`) for interactive chat.
- Shares NLP utilities and prompt logic (`shared_config.py`).
- Includes workflow documentation in the FLOW folder.

---

## 🚀 How to Run

1. **Clone the repository and enter the module folder:**
   ```powershell
   git clone https://git-codecommit.us-east-2.amazonaws.com/v1/repos/shiker-python-datascience
   cd "shiker-python-datascience/Faq Extractor/MultiContext2"
   ```
2. **Create and activate a virtual environment (recommended):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Set your OpenAI API key:**
   - Option A: Add a `.env` file in `MultiContext2/` with `OPENAI_API_KEY=sk-your-key`
   - Option B: For current session:
     ```powershell
     $env:OPENAI_API_KEY="sk-your-key"
     ```
5. **(Optional) Install NLP extras:**
   ```powershell
   pip install spacy sentence-transformers
   python -m spacy download en_core_web_sm
   python -m nltk.downloader vader_lexicon punkt
   ```
6. **Extract FAQs:**
   ```powershell
   python app2.py
   ```
7. **Fine-tune and save model:**
   ```powershell
   python check_and_tune.py
   ```
8. **Run the ensemble chatbot:**
   ```powershell
   python multimodel.py
   ```
9. **Launch the Streamlit chat UI:**
   ```powershell
   streamlit run ui4.py
   ```

---

## 🛠️ Troubleshooting
- Missing API key: set `$env:OPENAI_API_KEY` or add `.env`.
- “No model produced a response”: ensure `FineTuning/*.txt` exist and contain valid model IDs.
- Python TypeError for “X | Y”: use Python 3.10+.
- NLTK/spaCy warnings: optional; install extras if needed.
- Fine‑tuning errors: validate JSONL and ensure roles `system`/`user`/`assistant` exist.
- For more details, see the [FLOW folder](./FLOW/) for architecture and workflow docs.

---

## Author
**Shikher Jain**
Novas Arc Data Science
