import streamlit as st
import requests, re, time, json
from bs4 import BeautifulSoup

from urllib.parse import urljoin, urlparse, urldefrag

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


FAQ_KEYWORDS_IN_URL = [
    "faq", "faqs", "help", "support", "customer-service",
    "knowledge", "knowledgebase", "kb", "guide",
    "how-to", "howto", "troubleshoot", "troubleshooting",
    "qna", "q&a", "questions", "getting-started", "contact-us"
]

# For on-page detection
FAQ_TEXT_HINTS = [
    "faq", "frequently asked questions", "how do i", "how to",
    "troubleshoot", "troubleshooting", "common questions",
    "help center", "support", "customer service"
]

# saved in jsonl format
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

# fetching
def fetch_url(url: str, timeout: int = 15) -> str:
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "}, timeout=timeout)
        res.raise_for_status()
        # basic content-type sanity check
        ctype = res.headers.get("Content-Type", "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype and not url.endswith((".html",".htm","/")):
            return ""
        return res.text
    except Exception:
        return ""
 
# Utilities
def same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc

def normalize_url(href: str, base: str) -> str:
    absolute = urljoin(base, href)
    absolute, _ = urldefrag(absolute)  # drop #fragments
    return absolute

def filter_links_by_keywords(links):
    L = []
    for u in links:
        lu = u.lower()
        if any(k in lu for k in FAQ_KEYWORDS_IN_URL):
            L.append(u)
    return list(dict.fromkeys(L))  #  keep order

# Extraction 
def extract_links(html: str, base_url: str):
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(a["href"], base_url)
        if same_domain(u, base_url):
            links.append(u)
    # light pruning of obvious non-pages
    links = [u for u in links if not any(u.endswith(ext) for ext in (".pdf",".jpg",".png",".gif",".svg",".zip",".rar",".7z",".mp4",".mp3",".css",".js"))]
    # dedupe
    return list(dict.fromkeys(links)) 

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

# Page Processor
def process_page(url: str, base_url: str):

    html = fetch_url(url)
    if not html:
        return [], []

    # quick on-page hint: if page contains lots of faq hints, we still expand links
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


# Streamlit
st.set_page_config(page_title="Universal FAQ Crawler", layout="wide")
st.title("Universal FAQ Extractor")

col1, col2, col3 = st.columns([3,1,1])
with col1:
    start_url = st.text_input("Website (start at a page likely to link to Help/FAQ):")

with col2:
    max_depth = st.slider("Crawl Depth", 0, 5, 1,help="0 = only the start page; 1 = + its FAQ-like links; etc.")

with col3:
    max_workers = st.slider("Max Threads", 1, 30, 10,help="Higher is faster but heavier. 8–16 is typical on a laptop.")

run = st.button("Start Extraction")

if run:
    if not start_url.startswith("http"):
        start_url = "https://" + start_url

    st.info("Running...")
    t0 = time.time()
    urls, faqs = crawl_site(start_url, max_depth=max_depth, max_workers=max_workers)
    dt = round(time.time() - t0, 2)

    st.success(f"Done in {dt} seconds. Crawled {len(urls)} pages. Extracted {len(faqs)} FAQs.")

    # Search UI over FAQs 
    st.subheader("Search FAQs")
    qcol1, qcol2 = st.columns([2,1])
    with qcol1:
        query = st.text_input("Search terms (filters by question text)", value="")
    with qcol2:
        min_len = st.number_input("Min answer length", min_value=0, max_value=2000, value=20, step=5)

    filtered = faqs
    if query:
        ql = query.lower()
        filtered = [f for f in filtered if ql in f["question"].lower()]
    if min_len > 0:
        filtered = [f for f in filtered if len(f["answer"]) >= min_len]

    # Questions selector
    st.markdown(f"**Matching FAQs: {len(filtered)}**")

    if filtered:
        questions = [f["question"] for f in filtered]

        if "selected_q" not in st.session_state:
            st.session_state.selected_q = questions[0]
        selected_q = st.selectbox("Select a question to view the answer:", questions, key="selected_q")
        sel = next((f for f in filtered if f["question"] == st.session_state.selected_q))

        # selected_q = st.selectbox("Select a question to view the answer:", questions)
        # sel = next((f for f in filtered if f["question"] == selected_q), None)              

        if sel:
            st.markdown("---")
            st.markdown("### Answer")
            st.write(sel["answer"])

    expander = st.expander("Show Raw Data")
    with expander:
        st.write(filtered)

    # Optional: show raw tables (collapsed)
    with st.expander("Crawled URLs"):
        st.write(len(urls))
        st.dataframe(urls, use_container_width=True)

    with st.expander("All Extracted FAQs"):
        st.write(len(faqs))
        for i, f in enumerate(faqs, 1):
            st.markdown(f"**Q{i}. {f['question']}**")
            st.write(f"{f['answer']}\n")
            st.markdown("---")

    save_faqs_jsonl(filtered,"filtered_faqs.jsonl")