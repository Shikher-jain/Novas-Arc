import requests
import time
import re
import json
from bs4 import BeautifulSoup

from urllib.parse import urljoin, urlparse, urldefrag

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

FAQ_KEYWORDS_IN_URL = [
    "faq", "faqs", "help", "support", "customer-service",
    "knowledge", "knowledgebase", "kb", "guide",
    "how-to", "howto", "troubleshoot", "troubleshooting",
    "qna", "q&a", "questions", "getting-started", "contact-us"
]

FAQ_TEXT_HINTS = [
    "faq", "frequently asked questions", "how do i", "how to",
    "troubleshoot", "troubleshooting", "common questions",
    "help center", "support", "customer service"
]

def save_faqs_jsonl(faqs, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for faq in faqs:
            q = faq.get("question", "").strip()
            a = faq.get("answer", "").strip()
            if q and a:
                record = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

def fetch_url(url: str, timeout: int = 15) -> str:
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        res.raise_for_status()
        ctype = res.headers.get("Content-Type", "").lower()
        if "text/html" not in ctype and not url.endswith((".html", ".htm", "/")):
            return ""
        return res.text
    except Exception:
        return ""

def same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc

def normalize_url(href: str, base: str) -> str:
    absolute = urljoin(base, href)
    absolute, _ = urldefrag(absolute)  # remove (#fragment)
    return absolute

def filter_links_by_keywords(links):
    return [u for u in links if any(k in u.lower() for k in FAQ_KEYWORDS_IN_URL)]


def remove_abb(data):

    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"there's", "there is", data)
    data = re.sub(r"We're", "We are", data)
    data = re.sub(r"That's", "That is", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"Can't", "Cannot", data)
    data = re.sub(r"wasn't", "was not", data)
    data = re.sub(r"don\x890ªt", "do not", data)    
    data = re.sub(r"aren't", "are not", data)
    data = re.sub(r"isn't", "is not", data)
    data = re.sub(r"What's", "What is", data)
    data = re.sub(r"haven't", "have not", data)
    data = re.sub(r"hasn't", "has not", data)
    data = re.sub(r"There's", "There is", data)
    data = re.sub(r"He's", "He is", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"You're", "You are", data)
    data = re.sub(r"I'M", "I am", data)
    data = re.sub(r"shouldn't", "should not", data)
    data = re.sub(r"wouldn't", "would not", data)
    data = re.sub(r"i'm", "I am", data)
    data = re.sub(r"I\x89Û³m", "I am", data)
    data = re.sub(r"I'm", "I am", data)
    data = re.sub(r"Isn't", "is not", data)
    data = re.sub(r"Here's", "Here is", data)
    data = re.sub(r"you've", "you have", data)
    data = re.sub(r"you\x890ave", "you have", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"couldn't", "could not", data)
    data = re.sub(r"we've", "we have", data)
    data = re.sub(r"it\x89Ûas", "it is", data)
    data = re.sub(r"doesn\x890at", "does not", data)
    data = re.sub(r"It\x890ªs", "It is", data)
    data = re.sub(r"Here\x89Ûas", "Here is", data)
    data = re.sub(r"who's", "who is", data)
    data = re.sub(r"I\x890ªve", "I have", data)
    data = re.sub(r"y'all", "you all", data)
    data = re.sub(r"can\x890at", "cannot", data)
    data = re.sub(r"would've", "would have", data)
    data = re.sub(r"it'll", "it will", data)
    data = re.sub(r"we'11", "we will", data)
    data = re.sub(r"wouldn\x890ªt", "would not", data)
    data = re.sub(r"We've", "We have", data)
    data = re.sub(r"he'11", "he will", data)
    data = re.sub(r"Y'all", "You all", data)
    data = re.sub(r"Weren't", "Were not", data)
    data = re.sub(r"Didn't", "Did not", data)
    data = re.sub(r"they'11", "they will", data)
    data = re.sub(r"they'd", "they would", data)
    data = re.sub(r"DON'T", "DO NOT", data)
    data = re.sub(r"That\x890ªs", "That is", data)
    data = re.sub(r"they've", "they have", data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"should've", "should have", data)
    data = re.sub(r"You\x89Û³re", "You are", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"Don\x890ªt", "Do not", data)
    data = re.sub(r"we'd", "we would", data)
    data = re.sub(r"i'll", "I will", data)    
    data = re.sub(r"weren't", "were not", data)
    data = re.sub(r"They're", "They are", data)
    data = re.sub(r"Can\x890", "Cannot", data)
    data = re.sub(r"you'll", "you will", data)
    data = re.sub(r"I'd", "I would", data)
    data = re.sub(r"let's", "let us", data)    
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"i've", "I have", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"doesn't", "does not", data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"didn't", "did not", data)
    data = re.sub(r"ain't", "am not", data)
    data = re.sub(r"you'll", "you will", data)
    data = re.sub(r"I've", "I have", data)
    data = re.sub(r"Don't", "do not", data)
    data = re.sub(r"I'll", "I will", data)
    data = re.sub(r"I'd", "I would", data)
    data = re.sub(r"Let's", "Let us", data)
    data = re.sub(r"you'd", "You would", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"Ain't", "am not", data)
    data = re.sub(r"Haven't", "Have not", data)
    data = re.sub(r"Could've", "Could have", data)
    data = re.sub(r"youve", "you have", data)
    data = re.sub(r"donårt", "do not", data)
    data = re.sub(r"'", "", data)
    
    return data

def clean_text(text: str) -> str:
    text = re.sub(r"¶", "", text)
    text = re.sub(r"Â", "", text)
    text = re.sub(r"<\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)  # removes [edit] etc.
    text = remove_abb(text)
    return text.strip()

# EXTRACTION 
def extract_links(html: str, base_url: str):
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(a["href"], base_url)
        if same_domain(u, base_url):
            links.append(u)
    links = [u for u in links if not any(u.endswith(ext) for ext in (".pdf",".jpg",".png",".gif",".zip",".mp4",".css",".js"))]
    return list(dict.fromkeys(links))

def extract_faqs_from_html(html: str):

    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    # Drop noise
    for tag in soup(["script","style","noscript","iframe","template","nav","footer","form","button","input","svg"]):
        tag.decompose()

    faqs = []

    # 1) JSON-LD (FAQPage)
    for script in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(script.string or "")
            # Handle dict or list of dicts
            candidates = data if isinstance(data, list) else [data]
            for d in candidates:
                if not isinstance(d, dict):
                    continue
                t = d.get("@type") or d.get("type")
                if t == "FAQPage" or (isinstance(t, list) and "FAQPage" in t):
                    main = d.get("mainEntity") or []
                    for item in main:
                        if isinstance(item, dict) and (item.get("@type") == "Question" or "name" in item):
                            q = item.get("name") or item.get("headline")
                            ans = None
                            acc = item.get("acceptedAnswer")
                            if isinstance(acc, dict):
                                ans = acc.get("text")
                            if q and ans:
                                faqs.append({"question": clean_text(q), "answer": clean_text(ans)})
        except Exception:
            pass

    # 2) details/summary blocks
    for det in soup.find_all("details"):
        summary = det.find("summary")
        if summary:
            q = clean_text(summary.get_text(" ", strip=True))
            # choose first reasonable content element
            content = det.find(["p","div","section","article","ul","ol"])
            a = clean_text(content.get_text(" ", strip=True)) if content else ""
            if q and (a or q.endswith("?")):
                faqs.append({"question": q, "answer": a})

    # 3) dl/dt/dd blocks
    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                q = clean_text(dt.get_text(" ", strip=True))
                a = clean_text(dd.get_text(" ", strip=True))
                if q and a:
                    faqs.append({"question": q, "answer": a})

    # 4) heading + next block (H2/H3/H4)
    for h in soup.find_all(["h2","h3","h4"]):
        qtxt = clean_text(h.get_text(" ", strip=True))
        if not qtxt:
            continue
        looks_like_q = qtxt.endswith("?") or re.search(r"\b(faq|question|help|how to|troubleshoot)\b", qtxt, re.I)
        if looks_like_q:
            nxt = h.find_next_sibling(lambda tag: tag.name in ["p","div","section","article","ul","ol"])
            a = clean_text(nxt.get_text(" ", strip=True)) if nxt else ""
            if qtxt and a:
                faqs.append({"question": qtxt, "answer": a})

    # 5) generic Q:/A: sequence (fallback)
    blocks = [clean_text(t.get_text(" ", strip=True)) for t in soup.find_all(["p","div","li","span","strong","b"])]
    q, a = None, []
    for t in blocks:
        if not t:
            continue
        if t.endswith("?") or re.match(r"^\s*q[:\-]\s*", t, re.I):
            if q and a:
                faqs.append({"question": q, "answer": clean_text(" ".join(a))})
            q, a = re.sub(r"^\s*q[:\-]\s*", "", t, flags=re.I), []
        else:
            if q is not None:
                # stop if block looks like a new heading/question starter
                if re.match(r"^\s*a[:\-]\s*", t, re.I):
                    t = re.sub(r"^\s*a[:\-]\s*", "", t, flags=re.I)
                a.append(t)
    if q and a:
        faqs.append({"question": q, "answer": clean_text(" ".join(a))})

    # de-dup + sanity filter
    unique = {}
    for item in faqs:
        q = clean_text(item.get("question",""))
        a = clean_text(item.get("answer",""))
        if len(q) >= 5 and len(a) >= 5:
            unique[q] = a
    return [{"question": q, "answer": a} for q, a in unique.items()]

def process_page(url: str, base_url: str):
    html = fetch_url(url)
    if not html:
        return [], []
    faqs = extract_faqs_from_html(html)

    links = extract_links(html, base_url)
    links = [u for u in links if same_domain(u, base_url)]
    links = filter_links_by_keywords(links)

    return links, faqs

# Crawler 
def crawl_site(root_url: str, max_depth: int, max_workers: int):
    base = root_url
    seen = set()
    seen_lock = Lock()
    all_urls = []
    all_faqs = []

    frontier = [root_url]

    for depth in range(max_depth + 1):
        if not frontier:
            break

        # make depth-specific unique URLs not seen yet
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
            for fut in as_completed(future_map):
                u = future_map[fut]
                try:
                    links, faqs = fut.result()
                    all_urls.append(u)
                    if faqs:
                        all_faqs.extend(faqs)
                    # accumulate next layer candidates (dedupe later)
                    next_frontier.extend(links)
                except Exception:
                    # swallow and continue
                    pass

        # dedupe next frontier
        if next_frontier:
            # keep only not-seen and same-domain (again)
            nf = []
            with seen_lock:
                for x in dict.fromkeys(next_frontier):
                    if x not in seen and same_domain(x, base):
                        nf.append(x)
            frontier = nf
        else:
            frontier = []

    # final  URLs
    all_urls = list(dict.fromkeys(all_urls))
    
    # final FAQs by question
    faq_map = {}
    for f in all_faqs:
        q = f["question"].strip()
        if q and q not in faq_map:
            faq_map[q] = f["answer"].strip()
    all_faqs = [{"question": q, "answer": a} for q, a in faq_map.items()]
    return all_urls, all_faqs

def main():
    start_url = input("Enter URL: ") 
    try:
        max_depth = int(input("Enter crawl depth (default 2): ") or 2)
    except ValueError:
        max_depth = 2
    try:
        max_workers = int(input("Enter max workers (default 8): ") or 8)
    except ValueError:
        max_workers = 8

    print("🚀 Running crawler...")
    t0 = time.time()
    urls, faqs = crawl_site(start_url, max_depth, max_workers)
    dt = round(time.time() - t0, 2)

    print(f"\n✅ Done in {dt} seconds")
    print(f"📄 Crawled {len(urls)} pages")
    print(f"❓ Extracted {len(faqs)} FAQs\n")

    # Min length filter
    try:
        min_len = int(input("Min answer length (0-2000, default 20): ") or 20)
    except ValueError:
        min_len = 20

    filtered = [f for f in faqs if len(f["answer"]) >= min_len]

    # Save results
    with open("filtered_faqs.jsonl", "w", encoding="utf-8") as f:
        for faq in filtered:
            f.write(json.dumps(faq, ensure_ascii=False) + "\n")

    save_faqs_jsonl(filtered, "train_faq.jsonl")

    print("💾 Saved raw FAQs → filtered_faqs.jsonl")
    print("💾 Saved fine-tune FAQs → train_faq.jsonl")

    # Show sample
    print("\n--- Sample FAQs ---")
    for i, f in enumerate(filtered[:5], 1):
        print(f"Q{i}: {f['question']}")
        print(f"A{i}: {f['answer']}\n")

if __name__ == "__main__":
    main()
