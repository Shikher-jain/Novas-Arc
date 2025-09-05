import streamlit as st
import requests, json, re, time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ---------------- Utility ----------------
def fetch_url(url, use_selenium=False):
    # Fetch page content using Requests (fallback Selenium for dynamic pages)
    try:
        res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        return res.text
    except:
        if use_selenium:
            try:
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                driver = webdriver.Chrome(options=chrome_options)
                driver.get(url)
                time.sleep(3)
                html = driver.page_source
                driver.quit()
                return html
            except Exception as e:
                st.error(f"Selenium failed for {url}: {e}")
                return ""
        return ""

def extract_faqs(html, page_url):
    # Extract FAQs (Q/A pairs) from a given HTML page
    soup = BeautifulSoup(html, "html.parser")
    faqs = []

    # H2/H3 as question + next paragraph as answer
    for q in soup.find_all(["h2", "h3"]):
        if re.search(r"(faq|question|q:|\?)", q.get_text(), re.I):
            answer = q.find_next("p")
            faqs.append({
                "prompt": q.get_text(strip=True),
                "completion": answer.get_text(strip=True) if answer else ""
            })

    # Schema.org FAQ markup
    for faq in soup.find_all(attrs={"itemscope": True}):
        if faq.get("itemtype") and "FAQPage" in faq.get("itemtype"):
            for item in faq.find_all(attrs={"itemprop": "mainEntity"}):
                q = item.find(attrs={"itemprop": "name"})
                a = item.find(attrs={"itemprop": "acceptedAnswer"})
                if q and a:
                    faqs.append({
                        "prompt": q.get_text(strip=True),
                        "completion": a.get_text(strip=True)
                    })

    return faqs

def extract_links(html, base_url):
    # Extract sublinks from page
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        link = urljoin(base_url, a['href'])
        if urlparse(link).netloc == urlparse(base_url).netloc:
            links.append(link)
    return list(set(links))

def crawl(url, seen, depth, max_depth, urls_list, faqs_list):
    # Recursive crawler to fetch FAQs + sublinks
    if url in seen or depth > max_depth:
        return
    seen.add(url)

    html = fetch_url(url)
    if not html:
        return

    urls_list.append(url)  # Save URL
    faqs = extract_faqs(html, url)  # Extract FAQs
    faqs_list.extend(faqs)

    links = extract_links(html, url)
    for link in links:
        crawl(link, seen, depth + 1, max_depth, urls_list, faqs_list)

# ---------------- Streamlit UI ----------------
st.title("Website FAQ Extractor")


home_url = st.text_input("Enter Website URL", "https://www.aboutamazon.com")
max_depth = st.slider("Sublink Crawl Depth", 1, 3, 2)

if st.button("Start Extraction"):
    start_time = time.time()
    st.info("Extracting FAQs and sublinks... please wait")

    seen = set()
    urls_list, faqs_list = [], []

    crawl(home_url, seen, 0, max_depth, urls_list, faqs_list)

    # Save URLs JSON
    with open("urls.json", "w", encoding="utf-8") as f:
        json.dump(urls_list, f, indent=2, ensure_ascii=False)

    # Save FAQs JSONL
    with open("faqs.jsonl", "w", encoding="utf-8") as f:
        for faq in faqs_list:
            f.write(json.dumps(faq, ensure_ascii=False) + "\n")

    end_time = time.time()
    exec_time = round(end_time - start_time, 2)

    st.success(f"Extraction complete in {exec_time} seconds. Found {len(urls_list)} URLs and {len(faqs_list)} FAQs.")

    # Download buttons
    st.download_button("Download URLs JSON", open("urls.json", "rb"), "urls.json")
    st.download_button("Download FAQs JSONL", open("faqs.jsonl", "rb"), "faqs.jsonl")

    # Show all FAQs in a single expander
    with st.expander("All Extracted FAQs"):
        for i, faq in enumerate(faqs_list, 1):
            st.markdown(f"**Q{i}: {faq['prompt']}**")
            st.markdown(f"**A{i}: {faq['completion']}**")
            st.markdown("---")
