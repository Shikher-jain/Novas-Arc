import requests, gzip, time, json, re
import pandas as pd
from io import BytesIO
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin

# -------------------
# Step 1: Detect Sitemap(s)
# -------------------
def find_sitemaps(base_url):
    if not base_url.endswith("/"):
        base_url += "/"

    robots_url = urljoin(base_url, "robots.txt")
    headers = {"User-Agent": "Mozilla/5.0"}
    sitemaps = []

    try:
        res = requests.get(robots_url, headers=headers, timeout=15)
        if res.status_code == 200:
            for line in res.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm_url = line.split(":", 1)[1].strip()
                    sitemaps.append(sm_url)
    except Exception as e:
        print("robots.txt not found:", e)

    # fallback: try default sitemap.xml
    if not sitemaps:
        fallback = urljoin(base_url, "sitemap.xml")
        try:
            res = requests.get(fallback, headers=headers, timeout=10)
            if res.status_code == 200:
                sitemaps.append(fallback)
        except:
            pass

    return sitemaps


# -------------------
# Step 2: Parse Sitemap
# -------------------
def parse_sitemap(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=20)

    if res.status_code != 200:
        return []

    if url.endswith(".gz"):
        xml_content = gzip.decompress(res.content).decode("utf-8")
        df = pd.read_xml(BytesIO(xml_content.encode("utf-8")))
    else:
        df = pd.read_xml(BytesIO(res.content))

    return df['loc'].dropna().tolist()


# -------------------
# Step 3: Fetch Page (requests → fallback Selenium)
# -------------------
def fetch_page(url, use_selenium=False):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if not use_selenium:
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 200:
                return res.text
        else:
            options = Options()
            options.add_argument("--headless")
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            time.sleep(3)
            html = driver.page_source
            driver.quit()
            return html
    except:
        pass
    return ""


# -------------------
# Step 4: Extract FAQs
# -------------------
def extract_faqs(html):
    soup = BeautifulSoup(html, "html.parser")
    faqs = []

    # Generic FAQ container detection
    faq_blocks = soup.find_all(["div","section"], class_=lambda x: x and "faq" in x.lower())

    for block in faq_blocks:
        questions = block.find_all(["h2","h3","h4","p","span"], string=True)
        for q in questions:
            q_text = q.get_text(strip=True)
            if not q_text or len(q_text.split()) < 3:
                continue
            ans_tag = q.find_next_sibling(["p","div","span"])
            a_text = ans_tag.get_text(strip=True) if ans_tag else ""
            if a_text:
                faqs.append({"prompt": q_text, "completion": " " + a_text})

    return faqs


# -------------------
# Step 5: Main Process
# -------------------
def process_website(base_url, output_file="faqs.jsonl", limit_urls=50):
    sitemaps = find_sitemaps(base_url)
    if not sitemaps:
        print("No sitemap found for", base_url)
        return

    print("Sitemaps found:", sitemaps)

    all_faqs = []
    start = time.time()

    for sitemap in sitemaps:
        urls = parse_sitemap(sitemap)
        print(f"From {sitemap} → {len(urls)} URLs")

        for url in urls[:limit_urls]:
            html = fetch_page(url)
            if not html:
                html = fetch_page(url, use_selenium=True)

            faqs = extract_faqs(html)
            if faqs:
                print(f"✅ {len(faqs)} FAQs found in {url}")
                all_faqs.extend(faqs)

    # save jsonl
    with open(output_file, "w", encoding="utf-8") as f:
        for faq in all_faqs:
            f.write(json.dumps(faq, ensure_ascii=False) + "\n")

    print(f"Done ✅ Extracted {len(all_faqs)} FAQs in {time.time()-start:.2f} sec")#!/usr/bin/env python3
"""
optimized_faq_crawler.py

Usage:
    python optimized_faq_crawler.py https://example.com --max-urls 500 --workers 10 --webdriver /path/to/chromedriver

Features:
 - Auto detect sitemap(s) from robots.txt or fallback /sitemap.xml
 - Parse .xml and .xml.gz sitemaps
 - Filter URLs by keywords to reduce work
 - Requests-first (fast); fallback to Selenium only when needed
 - Parallel processing with ThreadPoolExecutor
 - Save visited URLs -> urls.json and FAQs -> faqs.jsonl
"""

import argparse
import gzip
import json
import time
import re
from io import BytesIO
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService

# -----------------------
# Configuration / Defaults
# -----------------------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}
SITEMAP_KEYWORDS = ["faq", "faqs", "question", "qa", "help", "support", "product"]
HTML_QUICK_CHECK_KEYWORDS = ["faq", "question", "help", "support", "q:"]  # used for quick 'requests' check
REQUEST_TIMEOUT = 15

# -----------------------
# Utilities
# -----------------------
def find_sitemaps(base_url):
    # Find sitemaps via robots.txt; fallback to /sitemap.xml
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urljoin(base, "/robots.txt")
    sitemaps = []
    try:
        r = requests.get(robots_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm = line.split(":", 1)[1].strip()
                    if sm:
                        sitemaps.append(sm)
    except Exception:
        pass
    if not sitemaps:
        fallback = urljoin(base, "/sitemap.xml")
        try:
            r = requests.head(fallback, headers=HEADERS, timeout=8)
            if r.status_code == 200:
                sitemaps.append(fallback)
        except Exception:
            pass
    return sitemaps

def fetch_bytes(url):
    # Fetch bytes with headers
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def parse_sitemap_content(content_bytes):
    # Parse sitemap XML bytes and return list of URLs (handles sitemap-index recursively)
    urls = []
    try:
        # detect gzip magic and decompress
        if content_bytes[:2] == b"\x1f\x8b":
            content_bytes = gzip.decompress(content_bytes)
    except Exception:
        pass
    try:
        root = ET.fromstring(content_bytes)
    except Exception:
        # fallback: crude regex to get <loc> tags
        try:
            text = content_bytes.decode("utf-8", errors="ignore")
            urls = re.findall(r"<loc>(.*?)</loc>", text, flags=re.IGNORECASE)
        except Exception:
            urls = []
        return urls

    # handle namespaces
    ns = ""
    m = re.match(r"\{(.*)\}", root.tag)
    if m:
        ns = "{" + m.group(1) + "}"

    # sitemap index
    for s in root.findall(f".//{ns}sitemap/{ns}loc"):
        if s.text:
            sub_bytes = fetch_bytes(s.text.strip())
            if sub_bytes:
                urls.extend(parse_sitemap_content(sub_bytes))

    # urlset
    for loc in root.findall(f".//{ns}url/{ns}loc"):
        if loc.text:
            urls.append(loc.text.strip())

    return urls

def collect_sitemap_urls(sitemap_urls, max_urls=None):
    # Given a list of sitemap urls, return deduped list of contained page URLs
    all_urls = []
    seen = set()
    for sm in sitemap_urls:
        content = fetch_bytes(sm)
        if not content:
            continue
        urls = parse_sitemap_content(content)
        for u in urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
                if max_urls and len(all_urls) >= max_urls:
                    return all_urls
    return all_urls

# -----------------------
# Filtering
# -----------------------
def filter_urls_by_keywords(urls, keywords=None):
    # Keep only URLs containing any of keywords (case-insensitive)
    if not keywords:
        return urls
    kws = [k.lower() for k in keywords]
    filtered = []
    for u in urls:
        low = u.lower()
        if any(k in low for k in kws):
            filtered.append(u)
    return filtered

# -----------------------
# FAQ extraction heuristics
# -----------------------
def extract_jsonld_faqs(soup):
    faqs = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or script.get_text()
            if not raw:
                continue
            parsed = json.loads(raw.strip())
            docs = parsed if isinstance(parsed, list) else [parsed]
            for doc in docs:
                if isinstance(doc, dict) and ("FAQ" in str(doc.get("@type", "")) or "FAQPage" in str(doc.get("@type", ""))):
                    items = doc.get("mainEntity") or doc.get("mainEntity", [])
                    if isinstance(items, list):
                        for item in items:
                            q = item.get("name") or item.get("question") or item.get("text")
                            aobj = item.get("acceptedAnswer") or item.get("answer")
                            if isinstance(aobj, dict):
                                a = aobj.get("text")
                            else:
                                a = aobj
                            if q and a:
                                faqs.append((q.strip(), a.strip()))
                # sometimes @graph present
                if isinstance(doc, dict) and "@graph" in doc:
                    for entry in doc["@graph"]:
                        if entry.get("@type", "").lower().startswith("faq"):
                            items = entry.get("mainEntity", [])
                            for item in items:
                                q = item.get("name")
                                aobj = item.get("acceptedAnswer") or item.get("answer")
                                if isinstance(aobj, dict):
                                    a = aobj.get("text")
                                else:
                                    a = aobj
                                if q and a:
                                    faqs.append((q.strip(), a.strip()))
        except Exception:
            continue
    return faqs

def extract_html_faqs(soup):
    faqs = []
    # 1) <details><summary>
    for d in soup.find_all("details"):
        s = d.find("summary")
        if s:
            q = s.get_text(" ", strip=True)
            rest = d.get_text(" ", strip=True).replace(q, "", 1).strip()
            if q and rest:
                faqs.append((q, rest))

    # 2) common FAQ container detection by class / id
    candidates = []
    for tag in soup.find_all(["div", "section", "article"]):
        cls = " ".join(tag.get("class") or []) if tag.get("class") else ""
        idv = tag.get("id") or ""
        if re.search(r"faq|faqs|question|qa|qna|help|support", cls + " " + idv, re.I):
            candidates.append(tag)
    for block in candidates:
        # find heading/qa pairs inside block
        for qtag in block.find_all(["h2", "h3", "h4", "dt", "p", "strong"]):
            qtext = qtag.get_text(" ", strip=True)
            if not qtext or len(qtext.split()) < 2:
                continue
            # answer candidates: next sibling paragraphs or next div
            ans_tag = qtag.find_next(["p", "div", "dd", "span"])
            atext = ans_tag.get_text(" ", strip=True) if ans_tag else ""
            if atext:
                faqs.append((qtext, atext))

    # 3) headings that end with '?'
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        t = h.get_text(" ", strip=True)
        if t.endswith("?") or re.search(r"\b(what|why|how|when|where|who)\b", t, re.I):
            # gather following sibling(s)
            ans_parts = []
            sib = h.find_next_sibling()
            steps = 0
            while sib and steps < 6:
                if sib.name and re.match(r"h[1-6]", sib.name, re.I):
                    break
                if sib.get_text(strip=True):
                    ans_parts.append(sib.get_text(" ", strip=True))
                sib = sib.find_next_sibling()
                steps += 1
            ans = " ".join(ans_parts).strip()
            if ans:
                faqs.append((t, ans))

    # dedupe preserving order
    seen = set()
    dedup = []
    for q, a in faqs:
        key = (re.sub(r"\s+", " ", q).strip().lower(), re.sub(r"\s+", " ", a).strip().lower())
        if key not in seen:
            seen.add(key)
            dedup.append((q.strip(), a.strip()))
    return dedup

# -----------------------
# Requests-first quick check + parse
# -----------------------
def requests_quick_fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None, False
        html = r.text
        low = html.lower()
        if any(k in low for k in HTML_QUICK_CHECK_KEYWORDS):
            # parse & extract FAQs from static HTML
            soup = BeautifulSoup(html, "html.parser")
            faqs = extract_jsonld_faqs(soup) + extract_html_faqs(soup)
            return html, bool(faqs)
        else:
            return html, False
    except Exception:
        return None, False

# -----------------------
# Selenium fallback per-URL (creates its own webdriver instance)
# -----------------------
def selenium_fetch_and_extract(url, webdriver_path=None, wait_seconds=2):
    # configure options to block images/css/fonts to speed up
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2,
        "profile.default_content_setting_values.notifications": 2
    }
    chrome_opts.add_experimental_option("prefs", chrome_prefs)
    chrome_opts.add_argument(f"user-agent={USER_AGENT}")

    # instantiate driver (per call); in heavy runs you may want a pool of drivers
    try:
        if webdriver_path:
            service = ChromeService(webdriver_path)
            driver = webdriver.Chrome(service=service, options=chrome_opts)
        else:
            driver = webdriver.Chrome(options=chrome_opts)
    except Exception as e:
        # fallback to returning nothing if webdriver fails
        print(f"[selenium] webdriver init failed for {url}: {e}")
        return []

    try:
        driver.set_page_load_timeout(30)
        driver.get(url)
        time.sleep(wait_seconds)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        faqs = extract_jsonld_faqs(soup) + extract_html_faqs(soup)
        driver.quit()
        return faqs
    except Exception as e:
        try:
            driver.quit()
        except Exception:
            pass
        print(f"[selenium] failed for {url}: {e}")
        return []

# -----------------------
# Worker to process a single URL (requests-first, returns (url, faqs, need_selenium_flag))
# -----------------------
def process_url_requests_phase(url):
    html, found_static = requests_quick_fetch(url)
    if found_static:
        soup = BeautifulSoup(html, "html.parser")
        faqs = extract_jsonld_faqs(soup) + extract_html_faqs(soup)
        return url, faqs, False
    # if we got HTML but no FAQ markers, still mark for Selenium fallback
    if html:
        # do lightweight heuristic: presence of common FAQ containers
        low = html.lower()
        if any(k in low for k in HTML_QUICK_CHECK_KEYWORDS):
            soup = BeautifulSoup(html, "html.parser")
            faqs = extract_jsonld_faqs(soup) + extract_html_faqs(soup)
            if faqs:
                return url, faqs, False
        return url, [], True
    # failed to fetch at all -> mark for selenium as last resort
    return url, [], True

# -----------------------
# Main pipeline
# -----------------------
def run_pipeline(base_url, max_urls=None, workers=10, webdriver_path=None, max_selenium_workers=4):
    start_time = time.time()
    base_url_parsed = urlparse(base_url)
    base = f"{base_url_parsed.scheme}://{base_url_parsed.netloc}"

    # 1) find sitemaps
    sitemaps = find_sitemaps(base)
    if not sitemaps:
        print("No sitemaps found for", base)
        return

    print("Discovered sitemaps:", sitemaps)

    # 2) collect sitemap urls (respect max_urls)
    all_urls = collect_sitemap_urls(sitemaps, max_urls=max_urls)
    print(f"Total URLs in sitemaps: {len(all_urls)}")

    # 3) filter URLs by keywords to reduce workload
    filtered = filter_urls_by_keywords(all_urls, SITEMAP_KEYWORDS)
    if not filtered:
        # fallback: if filtering removed everything, keep some of all_urls up to max_urls
        filtered = all_urls[:max_urls] if max_urls else all_urls
    print(f"URLs after keyword filter: {len(filtered)} (will process up to max limit)")

    # limit final set
    if max_urls:
        filtered = filtered[:max_urls]

    # 4) requests-first parallel phase
    results = []
    need_selenium = []
    visited_urls = []
    faqs_list = []

    print("Starting requests-first phase with workers =", workers)
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(process_url_requests_phase, u): u for u in filtered}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                url, faqs, requires_js = fut.result()
                visited_urls.append(url)
                if faqs:
                    # convert list of tuples to dicts
                    for q, a in faqs:
                        faqs_list.append({"prompt": q, "completion": a})
                if requires_js:
                    need_selenium.append(url)
            except Exception as e:
                print("Error processing (requests) url", u, e)

    print(f"Requests phase done. static-faqs: {len(faqs_list)}; need selenium for {len(need_selenium)} URLs")

    # 5) Selenium fallback phase (smaller pool)
    if need_selenium:
        print("Starting selenium fallback phase with workers =", max_selenium_workers)
        with ThreadPoolExecutor(max_workers=max_selenium_workers) as exe:
            futures = {exe.submit(selenium_fetch_and_extract, u, webdriver_path): u for u in need_selenium}
            for fut in as_completed(futures):
                u = futures[fut]
                try:
                    faqs = fut.result()
                    visited_urls.append(u)
                    if faqs:
                        for q, a in faqs:
                            faqs_list.append({"prompt": q, "completion": a})
                except Exception as e:
                    print("Error in selenium fallback for", u, e)

    # dedupe faqs by lower-cased q+a
    seen = set()
    dedup_faqs = []
    for item in faqs_list:
        k = (re.sub(r"\s+"," ", item["prompt"]).strip().lower(),
             re.sub(r"\s+"," ", item["completion"]).strip().lower())
        if k not in seen:
            seen.add(k)
            dedup_faqs.append(item)

    # write urls.json and faqs.jsonl
    urls_out = "urls.json"
    faqs_out = "faqs.jsonl"
    with open(urls_out, "w", encoding="utf-8") as fh:
        json.dump(list(dict.fromkeys(visited_urls)), fh, indent=2, ensure_ascii=False)
    with open(faqs_out, "w", encoding="utf-8") as fh:
        for obj in dedup_faqs:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print("Run complete.")
    print(f"Total visited URLs: {len(set(visited_urls))}")
    print(f"Total unique FAQ pairs extracted: {len(dedup_faqs)}")
    print(f"Saved: {urls_out}, {faqs_out}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

# -----------------------
# CLI
# -----------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Optimized FAQ crawler (sitemap -> requests-first -> selenium fallback).")
#     parser.add_argument("site", help="Base website URL (e.g. https://www.flipkart.com)")
#     parser.add_argument("--max-urls", type=int, default=200, help="Maximum number of URLs to process (default 200)")
#     parser.add_argument("--workers", type=int, default=10, help="Thread workers for requests phase (default 10)")
#     parser.add_argument("--selenium-workers", type=int, default=3, help="Thread workers for selenium fallback (default 3)")
#     parser.add_argument("--webdriver", type=str, default=None, help="Path to chromedriver executable (optional)")
#     args = parser.parse_args()

#     run_pipeline(args.site, max_urls=args.max_urls, workers=args.workers, webdriver_path=args.webdriver, max_selenium_workers=args.selenium_workers)

#     print(f"Output saved to {output_file}")

if __name__ == "__main__":
        process_website(input("Enter website URL: "), "faqs.jsonl")

