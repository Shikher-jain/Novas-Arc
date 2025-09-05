from bs4 import BeautifulSoup
import streamlit as st
import time
import json
import os
import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

DATA_FILE = "faqs.jsonl"

# Selenium just test

def get_page_source(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # Use an explicit wait instead of time.sleep()
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "main-content"))
        )
    except Exception as e:
        print(f"Timed out waiting for page to load: {e}")
    
    page_source = driver.page_source
    driver.quit()
    return page_source

def get_page_source_(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(5)  # wait for JS to load
    
    page_source = driver.page_source
    driver.quit()
    return page_source

def clean_text(text: str) -> str:
    text = re.sub(r"¶", "", text)
    text = re.sub(r"Â", "", text)
    text = re.sub(r"<\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)  # removes [edit] etc.
    return text.strip()


# Extract FAQ
def extract_faqs_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove noisy tags
    for tag in soup(["script", "style", "nav", "footer", "header", "form","noscript", "iframe", "button", "input", "aside"]):
        tag.decompose()
    
    faqs = []
    texts = [t.get_text(" ", strip=True) for t in soup.find_all(
        ["p", "div", "li", "span", "h2", "h3", "h4", "strong", "b"]
    )]
    
    q, a = None, None
    for text in texts:

        # Detect Question
        if text.lower().startswith("q") or text.endswith("?"):
            if q and a:
                faqs.append({"question":clean_text(q), "answer": clean_text(a)})
            q, a = text, None
        
        # Detect Answer
        elif text.lower().startswith("a") or (q and not a):
            if not a:
                a = text
            else:
                a += " " + text
    
    if q and a:
        faqs.append({"question": q, "answer": a})
    
    # Clean up FAQs
    clean_faqs = []
    for f in faqs:
        question = f["question"].replace("Q:", "").strip()
        answer = f["answer"].replace("A:", "").strip()
        if len(question) > 3 and len(answer) > 3:
            clean_faqs.append({"question": question, "answer": answer})
    
    return clean_faqs

# Save as JSONL (cache + fine-tuning)
def save_faq_cache(faqs, filename=DATA_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        for faq in faqs:
            entry = {
                "messages": [
                    {"role": "user", "content": faq["question"]},
                    {"role": "assistant", "content": faq["answer"]}
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return filename

# Load JSONL 
def load_faq_cache(filename=DATA_FILE):
    faq_store = []
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    q = obj["messages"][0]["content"]
                    a = obj["messages"][1]["content"]
                    faq_store.append({"question": q, "answer": a})
        except Exception:
            return []
    return faq_store


# Streamlit

st.title("URL FAQ Extractor (Fine-Tuning JSONL)")

# Define the file name
urls_file = "urls.txt"

# Check if the urls.txt file exists
if os.path.exists(urls_file):
    # If the file exists, show a message and a button to proceed
    st.info(f"Ready to process URLs from `{urls_file}`")
    
    if st.button("Extract FAQs"):
        all_faqs = []
        start_time = time.time()  # start execution timer

        # Read URLs from the file
        with open(urls_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]

        if urls:
            for url in urls:
                st.info(f"Scraping: {url} ")
                try:
                    html = get_page_source(url)
                    faqs = extract_faqs_from_html(html)
                    if faqs:
                        st.success(f"{len(faqs)} FAQs extracted from {url}")
                        all_faqs.extend(faqs)
                    else:
                        st.warning(f"No FAQs found on {url}")
                except Exception as e:
                    st.error(f"Error with {url}: {str(e)}")

            end_time = time.time()
            execution_time = round(end_time - start_time, 2)

            if all_faqs:
                st.success(f"Total {len(all_faqs)} FAQs extracted from {len(urls)} site(s)")
                st.info(f"Execution Time: {execution_time} seconds")
                
                st.subheader("Extracted Questions")
                for i, faq in enumerate(all_faqs, 1):
                    with st.expander(f"Q{i}: {faq['question']}"):
                        st.write(f"**Answer:** {faq['answer']}")

                filename = save_faq_cache(all_faqs)
                with open(filename, "r", encoding="utf-8") as f:
                    st.download_button("Download All FAQs (JSONL)", f, file_name="faqs.jsonl")
            else:
                st.warning("No FAQs extracted from any URL")
        else:
            st.warning("The `urls.txt` file is empty. Please add URLs to it.")
else:
    # If the file doesn't exist, show a warning and instructions
    st.warning(f"The file `{urls_file}` was not found.")
    st.info(f"Please create a text file named `{urls_file}` in the same directory as your script and add URLs to it, one per line.")


# Streamlit

# st.title("URL FAQ Extractor (Fine-Tuning JSONL)")

# urls_input = st.text_area("Enter Website URLs (comma separated)", "")

# if st.button("Extract FAQs"):
#     if urls_input:
#         urls = [u.strip() for u in urls_input.split(",") if u.strip()]
#         all_faqs = []
        
#         start_time = time.time()  # start execution timer

#         for url in urls:
#             st.info(f"Scraping: {url} ")
#             try:
#                 html = get_page_source(url)
#                 faqs = extract_faqs_from_html(html)
#                 if faqs:
#                     st.success(f"{len(faqs)} FAQs extracted from {url}")
#                     all_faqs.extend(faqs)
#                 else:
#                     st.warning(f"No FAQs found on {url}")
#             except Exception as e:
#                 st.error(f"Error with {url}: {str(e)}")
        
#         end_time = time.time()
#         execution_time = round(end_time - start_time, 2)

#         if all_faqs:
#             st.success(f"Total {len(all_faqs)} FAQs extracted from {len(urls)} site(s)")
#             st.info(f"Execution Time: {execution_time} seconds")

#             # Show only questions list with expandable answers
#             st.subheader("Extracted Questions")
#             for i, faq in enumerate(all_faqs, 1):
#                 with st.expander(f"Q{i}: {faq['question']}"):
#                     st.write(f"**Answer:** {faq['answer']}")

#             filename = save_faq_cache(all_faqs)
#             with open(filename, "r", encoding="utf-8") as f:
#                 st.download_button("Download All FAQs (JSONL)", f, file_name="faqs.jsonl")
#         else:
#             st.warning("No FAQs extracted from any URL")
#     else:
#         st.warning("Please enter at least one URL")
