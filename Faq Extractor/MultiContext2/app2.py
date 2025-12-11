
import os
import re
import time
import json
import logging
import random
from typing import Dict, List, Tuple, Set, Any
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from bs4 import BeautifulSoup
import contractions

# ========== IMPORT SHARED CONFIGURATION ==========
# Eliminates code duplication - all shared functions and configs imported from here
from shared_config import (
    TOPIC_TONE_MAP,
    CONTEXTS,
    advanced_system_prompt_generator
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

"""
Our next task is to train or fine-tune an OpenAI model with multiple contexts.
Here, “different contexts” can refer to:
- Different topics or domains – e.g., customer support, technical help, sales chat
- Different personas or tones – e.g., friendly assistant, formal support agent
- Different user intents – e.g., FAQ answering, booking assistance, product recommendations
A single model can handle all these variations if it is trained or prompted correctly.
"""

# Constants and Configuration
FAQ_KEYWORDS = {
    "frequently", "faq", "faqs", "common questions", "help center",
    "support", "knowledge base", "help", "asked questions"
}

FAQ_KEYWORDS_IN_URL = FAQ_KEYWORDS | {"help", "support", "faq", "knowledge"}


FAQ_KEYWORDS = set([
    keyword for context in CONTEXTS.values() 
    for keyword in context["keywords"]
] + ["faq", "faqs", "questions", "help", "support"])

QUESTION_PATTERN = re.compile(r"\b(what|how|when|where|why|which|who|do|does|did|can|should|is|are|will|there|any)\b.*\?", re.I)


# --- Simple context detection function ---
def detect_context(url, html):
    """
    Detect context and system message based on URL and HTML content.
    Returns (context, system_msg)
    """
    for key, val in CONTEXTS.items():
        # Check keywords in URL
        if any(k in url.lower() for k in val["keywords"]):
            return key, val["system_msg"]
        # Check keywords in HTML
        if html and any(k in html.lower() for k in val["keywords"]):
            return key, val["system_msg"]
    # Default fallback
    return "support", CONTEXTS["support"]["system_msg"]
def clean_answer(text, all_questions):
    if not text:
        return ""
    
    sentences = re.split(r'(?<=[.?!])\s+', text)  # split by sentence
    final = []
    for s in sentences:
        # Stop if this sentence looks like a new question
        if QUESTION_PATTERN.search(s):
            break
        # Stop if sentence contains any known question explicitly
        if any(q.lower() in s.lower() for q in all_questions):
            break
        final.append(s)
    return " ".join(final).strip()

def looks_like_question(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    # Accept if it ends with a question mark
    if text.endswith("?"):
        return True

    # Accept if it starts with common question words or Q-number style
    return bool(
        re.match(
            r"^(q[\.\-\:\)]*\s*\d*\s*|what|how|why|when|where|which|who|can|could|should|may|"
            r"is|are|will|there|any|do|does|did|have|has|had|i\s+am|am\s+i)\b",
            text,
            re.I,
        )
    )

def save_faqs_jsonl(faqs, url, system_msg="You are a customer support assistant. Provide clear and helpful answers to customer questions."):
    """
    Save FAQs to a JSONL file.
    The filename is derived from the domain name in the URL.
    Each record follows the format:
    {
        "messages": [
            {"role": "system", "content": "System message here"},
            {"role": "user", "content": "User's question here"},
            {"role": "assistant", "content": "Assistant's answer here"}
        ]
    }
    """
    # Extract domain name from the URL
    domain = urlparse(url).netloc.replace(".", "_")
    filename = f"{domain}.jsonl"

    with open(filename, "w", encoding="utf-8") as f:
        for faq in faqs:
            record = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": faq['question']},
                    {"role": "assistant", "content": faq['answer']}
                ]
            }
            f.write(json.dumps(record) + "\n")

    logging.info(f"Saved FAQs to {filename}")

def fetch_url(url:  str, timeout: int = 15) -> str:

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    @sleep_and_retry
    @limits(calls=5, period=1)  # 5 requests per second
    def limited_get(url, **kwargs):
        return session.get(url, **kwargs)

    try:
        res = limited_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        res.raise_for_status()
        ctype = res.headers.get("Content-Type", "").lower()
        if "text/html" not in ctype and not url.endswith((".html", ".htm", "/")):
            return ""
        return res.text
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return ""

def same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc

def normalize_url(href: str, base: str) -> str:
    absolute = urljoin(base, href)
    absolute, _ = urldefrag(absolute)  # remove (#fragment)
    return absolute

def remove_abb(text):
    return contractions.fix(text)

def clean_text(text: str) -> str:
    text = re.sub(r"¶", "", text)
    text = re.sub(r"Â", "", text)
    text = re.sub(r"<\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\xa0", " ", text)   
    text = re.sub(r'^(q[\s\.:;\-\)]*\d*\.?\s*)', '', text, flags=re.I)
    text = re.sub(r'^\d+\.\s*', '', text)  # removes leading numbering like "1. "
    text = re.sub(r'\s+\d+\.\s*$', '', text)  # removes trailing numbering like " 1."
    text = re.sub(r"^\s+|\s+$", "", text)  # removes leading and trailing whitespace
    text = re.sub(r"\[.*?\]", " ", text)  # removes [edit] etc.
    text = remove_abb(text)

    # Remove duplicate sentences
    sentences = re.split(r"(?<=[.?!])\s+", text)
    seen, unique = set(), []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    # Rejoin sentences into cleaned text
    return " ".join(unique).strip()

def extract_links(html: str, base_url: str):
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        u = normalize_url(a["href"], base_url)

        # Skip URLs pointing to other domains
        if not same_domain(u, base_url):
            continue

        # Skip static assets or tracking URLs
        if re.search(r"(\.pdf|\.jpg|\.png|\.gif|\.zip|\.mp4|\.css|\.js|\?.*utm|tracking)", u):
            continue

        # Skip URLs that repeat the domain in the path (trap URLs)
        if urlparse(base_url).netloc in u[len(base_url):]:
            continue

        # Keep only URLs likely related to FAQ/help
        if not any(k in u.lower() for k in FAQ_KEYWORDS_IN_URL):
            continue

        links.append(u)

    return list(dict.fromkeys(links))

def deduplicate_faqs(faqs):
    final_faqs = []
    seen = set()
    for f in faqs:
        q = clean_text(f.get("question", ""))
        a = clean_text(f.get("answer", ""))
        if len(q) < 5 or len(a) < 5:
            continue

        # Deduplicate by both question and answer
        key = (q.lower(), a.lower())
        if key not in seen:
            seen.add(key)
            final_faqs.append({"question": q, "answer": clean_answer(a, [q])})
    return final_faqs

def extract_faqs_from_html(html: str):
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script","style","noscript","iframe","template","nav","footer","form","button","input","svg"]):
        tag.decompose()
    faqs = []

    # 1) JSON-LD
    for script in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(script.string or "")
            candidates = data if isinstance(data, list) else [data]

            for d in candidates:
                if not isinstance(d, dict):
                    continue

                # Get @type (can be string or list)
                t = d.get("@type") or d.get("type")
                if t == "FAQPage" or (isinstance(t, list) and "FAQPage" in t):
                    main = d.get("mainEntity") or []
                    for item in main:
                        if not isinstance(item, dict):
                            continue

                        q = item.get("name") or item.get("headline")
                        acc = item.get("acceptedAnswer")
                        ans = acc.get("text") if isinstance(acc, dict) else None

                        if q and ans:
                            faqs.append({
                                "question": clean_text(q),
                                "answer": clean_text(ans)
                            })
        except Exception:
            pass

    # 2) Accordion items (common in many sites)
    for block in soup.find_all("div", class_=lambda x: x and ("accordion-item" in x or "accordion" in x.lower() or "itemExpander_module_expandableSection" in x)):
        q_tag = block.find(["h2", "h3", "h4", "button"])
        a_tag = block.find(["div", "section", "p", "ul", "ol", "article"])
        if q_tag and a_tag:
            q = clean_text(q_tag.get_text(" ", strip=True))
            a = clean_text(a_tag.get_text(" ", strip=True))
            if q and a:
                faqs.append({"question": q, "answer": a})

    # 3) <details>/<summary>
    for det in soup.find_all("details"):
        summary = det.find("summary")
        if summary:
            q = summary.get_text(" ", strip=True)
            content = det.find_all(["p","div","section","article","ul","ol"])
            a = " ".join([clean_text(c.get_text(" ", strip=True)) for c in content]) if content else ""
            if q and (a or q.endswith("?")):
                faqs.append({"question": q, "answer": a})

    # 4) <dl>/<dt>/<dd>
    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                q = dt.get_text(" ", strip=True)
                a = dd.get_text(" ", strip=True)
                if q and a:
                    faqs.append({"question": q, "answer": a})

    # 5) Headings
    for h in soup.find_all(["h2","h3","h4","button"]):
        qtxt = h.get_text(" ", strip=True)
        if not qtxt:
            continue

        if looks_like_question(qtxt):
            nxt = h.find_next_sibling(lambda tag: tag.name in ["p","div","section","article","ul","ol"])
            a = clean_text(nxt.get_text(" ", strip=True)) if nxt else ""
            if qtxt and a:
                faqs.append({"question": qtxt, "answer": a})

    # 6) Tables (Q in first <td>, A in second <td>)
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) == 2:
            q = clean_text(tds[0].get_text(" ", strip=True))
            a = clean_text(tds[1].get_text(" ", strip=True))
            if q.endswith("?") and a:
                faqs.append({"question": q, "answer": a})

    # 7)  Q:/A:
    blocks = [clean_text(t.get_text(" ", strip=True)) for t in soup.find_all(["p","div","li","span","strong","b"])]
    q, a = None, []

    for t in blocks:
        if not t:
            continue
        if t.endswith("?") or re.match(r"^\s*q[:\-]\s*", t, re.I):
            if q and a:
                faqs.append({"question": q, "answer": " ".join(a)})
            q, a = re.sub(r"^\s*q[:\-]\s*", "", t, flags=re.I), []
        else:
            if q is not None:
                if re.match(r"^\s*a[:\-]\s*", t, re.I):
                    t = re.sub(r"^\s*a[:\-]\s*", "", t, flags=re.I)
                a.append(t)
    
    # 8) FAQ sections (h2/h3 as section, h4/p/li as Q/A)
    for h in soup.select("h3"):
        q = clean_text(h.get_text(" ", strip=True))
        if not q.endswith("?"):
            continue
        # Answer is next sibling or next block until something that looks like next question
        nxt = h.find_next_sibling()
        answer_text = ""
        # Collect paragraphs / lists until next h3
        while nxt and nxt.name not in ["h3"]:
            if nxt.name in ["p","div","ul","ol"]:
                answer_text += " " + nxt.get_text(" ", strip=True)
            nxt = nxt.find_next_sibling()
        a = clean_text(answer_text)
        if q and a:
            faqs.append({"question": q, "answer": a})

    # 9) Flexible accordion handling (AWS/Flipkart/others)
    for block in soup.find_all("div", class_=lambda x: x and "accordion" in x.lower()):
        q_tag = block.find(["button","h2","h3","h4"], class_=lambda x: not x or "trigger" in (x or "").lower())
        a_tag = block.find(["div","section","p","ul","ol","article"], class_=lambda x: True)
        if q_tag and a_tag:
            q = clean_text(q_tag.get_text(" ", strip=True))
            a = clean_text(a_tag.get_text(" ", strip=True))
            if q and a:
                faqs.append({"question": q, "answer": a})

    # 10) Each topic is a section, e.g., <div id="topic-1"> ... </div>
    topics = soup.select("div.lb-grid")  
    for topic in topics:  
        questions = topic.find_all(["h3","strong"])  
        for q_tag in questions:  
            question = clean_text(q_tag.get_text(" ", strip=True))  
            answer_parts = []  
            for sib in q_tag.find_all_next():  
                if sib.name in ["h3", "strong"]:  
                    break  
                if sib.name in ["p","div","ul","ol","li"]:  
                    answer_parts.append(clean_text(sib.get_text(" ", strip=True)))  
            answer = " ".join(answer_parts).strip()  
            if question and answer:  
                faqs.append({"question": clean_text(question), "answer": clean_text(answer)})  

    # if q and a:
    #     faqs.append({"question": clean_text(q), "answer": " ".join(clean_text(a))})

    return deduplicate_faqs(faqs)

def process_page(url: str, base_url: str):
    html = fetch_url(url)
    if not html:
        return [], [], None, None
    
    # Extract FAQs from the page
    faqs = extract_faqs_from_html(html)
    
    # Detect context using unified context detection
    context, system_msg = detect_context(url, html)
    
    # Extract links from the page
    links = extract_links(html, base_url)
    for f in faqs:
        soup_ans = BeautifulSoup(f.get("answer", ""), "html.parser")
        for a_tag in soup_ans.find_all("a", href=True):
            u = normalize_url(a_tag["href"], base_url)
            if same_domain(u, base_url) and u not in links:
                links.append(u)
    
    # Deduplicate links
    links = list(dict.fromkeys(links))
    
    return links, faqs, context, system_msg

def crawl_site(root_url: str, max_depth: int, max_workers: int):
    base = root_url
    seen = set()
    seen_lock = Lock()
    all_urls = []
    all_faqs = []
    context_faqs = {}  # Dictionary to store FAQs by context
    failed_urls = []
    frontier = [root_url]

    for depth in tqdm(range(max_depth + 1), desc="Crawl Depth"):
        if not frontier:
            break

        this_batch = []
        with seen_lock:
            for u in frontier:
                if u not in seen:
                    seen.add(u)
                    this_batch.append(u)

        if not this_batch:
            break

        next_frontier = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(process_page, u, base): u for u in this_batch}
            for fut in tqdm(as_completed(future_map), total=len(this_batch), desc=f"Depth {depth}"):
                u = future_map[fut]
                try:
                    links, faqs, context, system_msg = fut.result()
                    all_urls.append(u)
                    all_faqs.extend(faqs)

                    # Store FAQs by context
                    if context not in context_faqs:
                        context_faqs[context] = {"faqs": [], "system_msg": system_msg}
                    context_faqs[context]["faqs"].extend(faqs)

                    next_frontier.extend(links)
                except Exception as e:
                    logging.warning(f"Failed processing {u}: {e}")
                    failed_urls.append(u)

        # Deduplicate next frontier
        frontier = list(dict.fromkeys([x for x in next_frontier if x not in seen and same_domain(x, base)]))

    # Save failed URLs for manual review
    if failed_urls:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            for url in failed_urls:
                f.write(url + "\n")
        logging.info(f"Saved {len(failed_urls)} failed URLs to failed_urls.txt")

    return list(dict.fromkeys(all_urls)), all_faqs, context_faqs

def get_site_name(url):
    """
    Extract a sanitized site name from the URL.
    """
    netloc = urlparse(url).netloc
    return re.sub(r"[^\w]+", "_", netloc)

def prepare_fine_tuning_data(context_faqs, output_folder="FineTuning"):
    """
    Prepare fine-tuning data for OpenAI models.
    Saves JSONL files for each context with variations in tone and persona.
    The filenames are based on the sanitized site name from the URL.
    """
    os.makedirs(output_folder, exist_ok=True)

    for context, data in context_faqs.items():
        faqs = data["faqs"]
        system_msg = data["system_msg"]

        # Extract site name from the first FAQ URL
        if faqs and "url" in faqs[0]:
            site_name = get_site_name(faqs[0]["url"])
        else:
            site_name = context  # Fallback to context name if no URL is available

        output_file = os.path.join(output_folder, f"{site_name}.jsonl")

        records = []
        for faq in faqs:
            record = {
                "prompt": f"{system_msg}\nQ: {faq['question']}\nA:",
                "completion": f" {faq['answer']}"
            }
            records.append(record)

        # Save records to JSONL file
        with open(output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        logging.info(f"Saved fine-tuning data for site '{site_name}' to: {output_file}")

def validate_input(prompt, valid_options=None, default=None):
    """
    Validate user input with optional valid options and default value.
    """
    while True:
        user_input = input(prompt).strip()
        if not user_input and default is not None:
            return default
        if valid_options and user_input not in valid_options:
            print(f"Invalid input. Please choose from: {', '.join(valid_options)}")
            continue
        return user_input




# Update the main function to include input validation and better logging
def main():
    # Instead of predefined sites, we'll use a single starting URL
    start_url = input("Enter starting URL: ")

    # Extract domain from start_url
    domain = urlparse(start_url).netloc.replace(".", "_")

    ft_folder = "FineTuning"
    os.makedirs(ft_folder, exist_ok=True)

    ft_file = os.path.join(ft_folder, f"{domain}.jsonl")
    
    if os.path.exists(ft_file):
        logging.info("Multi-context FAQs already extracted.")
        logging.info(f"Use existing file: {ft_file}")
        return
        
    try:
        max_depth = int(input("Enter crawl depth (default 3): ") or 3)
    except ValueError:
        max_depth = 3
    try:
        max_workers = int(input("Enter max workers (default 12): ") or 12)
    except ValueError:
        max_workers = 12        
    try:
        min_len = int(input("Min answer length (0-2000, default 20): ") or 20)
    except ValueError:
        min_len = 20
        
    logging.info(f"Running crawler for {start_url}...")
    t0 = time.time()
    try:
        urls, all_faqs, context_faqs = crawl_site(start_url, max_depth, max_workers)
    except Exception as e:
        logging.error(f"Crawling failed: {e}")
        return
    dt = round(time.time() - t0, 2)
    
    logging.info(f"Done in {dt} seconds")
    logging.info(f"Crawled {len(urls)} pages")
    logging.info(f"Extracted {len(all_faqs)} FAQs")
    # print(f"Detected {len(context_faqs)} different contexts: {', '.join(context_faqs.keys())}")
    
    # Save fine-tuning data for each context
    all_records = []
    
    # ✅ USE GLOBAL advanced_system_prompt_generator for UNIQUE prompts per FAQ
    logging.info("Generating advanced system prompts for each FAQ...")
    
    for context, data in context_faqs.items():
        faqs = data["faqs"]
        context_prompt_count = 0
        
        for faq in faqs:
            q = faq.get("question", "").strip()
            a = faq.get("answer", "").strip()
            
            if q and a and len(a) >= min_len:
                # Generate UNIQUE system prompt using global advanced function
                system_prompt = advanced_system_prompt_generator(q, a, context)
                
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]
                }
                all_records.append(record)
                context_prompt_count += 1
        
        logging.info(f"Processed {context_prompt_count} {context} FAQs with advanced prompts")

    # Save combined fine-tuning data
    try:
        with open(ft_file, "w", encoding="utf-8") as f:
            for record in all_records:
                try:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    logging.warning(f"Failed to write record: {e}")
        logging.info(f"\nTotal extracted {len(all_faqs)} FAQs across all contexts")
        logging.info(f"Saved fine-tuning data to: {ft_file}")
        logging.info("\nJSONL file created successfully. You can use this file for fine-tuning in a separate process.")
    except Exception as e:
        logging.error(f"Failed to save fine-tuning data: {e}")

if __name__ == "__main__":
    main()
