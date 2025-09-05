import streamlit as st
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import time
from urllib.parse import urljoin, urlparse

st.set_page_config(page_title="FAQ Extractor", layout="wide")

# --------------------------
# Helper functions
# --------------------------

def fetch_page(url):
    """Fetch page safely with timeout"""
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200 and "text/html" in res.headers.get("Content-Type", ""):
            return res.text
    except Exception:
        return None
    return None


def extract_links(base_url, html, keyword_filters=None):
    """Extract all links from HTML page, filter if keyword provided"""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        # same domain only
        if urlparse(href).netloc == urlparse(base_url).netloc:
            if not keyword_filters or any(k in href.lower() for k in keyword_filters):
                links.append(href.split("#")[0])  # remove fragments
    return list(set(links))


def extract_faqs(html):
    """Extract FAQs from page"""
    faqs = []
    soup = BeautifulSoup(html, "html.parser")

    # Look for Q/A style
    for q in soup.find_all(["h2", "h3", "h4"]):
        text = q.get_text(" ", strip=True)
        if "?" in text:
            ans = ""
            sib = q.find_next_sibling()
            if sib:
                ans = sib.get_text(" ", strip=True)
            if text and ans:
                faqs.append({"question": text, "answer": ans})
    return faqs


def crawl_site(start_url, max_depth=2, max_workers=10, keyword_filters=None):
    visited = set()
    to_visit = [(start_url, 0)]
    urls = []
    faqs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while to_visit:
            futures = {}
            for (u, depth) in list(to_visit):
                if u in visited or depth > max_depth:
                    continue
                visited.add(u)
                futures[executor.submit(fetch_page, u)] = (u, depth)
                to_visit.remove((u, depth))

            for future in concurrent.futures.as_completed(futures):
                url, depth = futures[future]
                html = future.result()
                if not html:
                    continue
                urls.append(url)

                # Extract FAQs
                faqs.extend(extract_faqs(html))

                # Extract more links
                if depth < max_depth:
                    new_links = extract_links(url, html, keyword_filters)
                    for nl in new_links:
                        if nl not in visited:
                            to_visit.append((nl, depth + 1))

    return urls, faqs


# --------------------------
# Streamlit UI
# --------------------------

st.title("🕷️ Smart FAQ Extractor")

start_url = st.text_input("Enter website URL", "https://www.flipkart.com")
max_depth = st.slider("Crawl Depth", 1, 3, 2)
max_workers = st.slider("Max Threads", 5, 50, 15, step=5)

kw_str = st.text_input("Keyword filters (comma-separated)", "faq,help,support")
keyword_filters = [k.strip().lower() for k in kw_str.split(",") if k.strip()]

run = st.button("🚀 Run Crawl")

# --------------------------
# Run Crawl
# --------------------------

if run:
    if not start_url.startswith("http"):
        start_url = "https://" + start_url

    st.info("Running multithreaded crawl with keyword-filtered expansion…")
    t0 = time.time()
    urls, faqs = crawl_site(start_url, max_depth=max_depth, max_workers=max_workers, keyword_filters=keyword_filters)
    dt = round(time.time() - t0, 2)

    st.session_state["urls"] = urls
    st.session_state["faqs"] = faqs
    st.session_state["crawl_time"] = dt

# --------------------------
# Display Results (Session Persisted)
# --------------------------

if "faqs" in st.session_state:
    urls = st.session_state["urls"]
    faqs = st.session_state["faqs"]
    dt = st.session_state["crawl_time"]

    st.success(f"✅ Done in {dt} seconds. Crawled {len(urls)} pages. Extracted {len(faqs)} FAQs.")

    # ---- Search UI over FAQs ----
    st.subheader("🔎 Search FAQs")
    qcol1, qcol2 = st.columns([2, 1])
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

    st.markdown(f"**Matching FAQs: {len(filtered)}**")
    if filtered:
        questions = [f["question"] for f in filtered]
        selected_q = st.selectbox("Select a question to view the answer:", questions, key="faq_select")
        sel = next((f for f in filtered if f["question"] == selected_q), None)
        if sel:
            st.markdown("---")
            st.markdown("### Answer")
            st.write(sel["answer"])

    with st.expander("📄 Crawled URLs"):
        st.write(len(urls))
        st.dataframe(urls, use_container_width=True)

    with st.expander("🧾 All Extracted FAQs"):
        st.write(len(faqs))
        for i, f in enumerate(faqs, 1):
            st.markdown(f"**Q{i}. {f['question']}**")
            st.write(f"{f['answer']}\n")
            st.markdown("---")
