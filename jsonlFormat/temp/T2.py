import streamlit as st
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import json
import re
import time
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# --- Scraping and Sitemap Functions ---

def find_sitemap_url(base_url):
    """Finds the sitemap.xml URL by checking robots.txt and common locations."""
    try:
        robots_url = urljoin(base_url, '/robots.txt')
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            for line in response.text.splitlines():
                if line.lower().startswith('sitemap:'):
                    return line.split(':', 1)[1].strip()
        sitemap_common_urls = [urljoin(base_url, '/sitemap.xml'), urljoin(base_url, '/sitemap_index.xml')]
        for sitemap_url in sitemap_common_urls:
            response = requests.head(sitemap_url, timeout=5)
            if response.status_code == 200:
                return sitemap_url
    except requests.RequestException:
        pass
    return None

def get_all_urls_from_sitemap(sitemap_url):
    """Parses a sitemap.xml to extract all URLs, handling sitemap index files."""
    urls = set()
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'xml')
        sitemap_tags = soup.find_all('sitemap')
        if sitemap_tags:
            for sitemap in sitemap_tags:
                loc = sitemap.find('loc').text
                urls.update(get_all_urls_from_sitemap(loc))
        else:
            loc_tags = soup.find_all('loc')
            urls.update(loc.text for loc in loc_tags)
    except requests.RequestException as e:
        st.error(f"Sitemap ko fetch karne mein error: {e}")
    return urls

def get_faqs_with_schema(soup):
    """Tries to find FAQs using Schema.org JSON-LD markup."""
    try:
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "FAQPage":
                faqs = []
                for item in data.get("mainEntity", []):
                    if item.get("@type") == "Question":
                        question = item.get("name", "").strip()
                        answer = item.get("acceptedAnswer", {}).get("text", "").strip()
                        if question and answer:
                            faqs.append({'question': question, 'answer': answer})
                if faqs: return faqs
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return None

def get_faqs_heuristically(soup):
    """Finds FAQs by searching for common HTML patterns."""
    faqs = []
    for details_tag in soup.find_all('details'):
        summary = details_tag.find('summary')
        content = details_tag.find(['p', 'div', 'li'])
        if summary and content:
            faqs.append({'question': summary.get_text(strip=True), 'answer': content.get_text(strip=True)})
    if faqs: return faqs
    for header in soup.find_all(['h2', 'h3', 'h4']):
        next_element = header.find_next_sibling()
        if next_element and (next_element.name in ['p', 'div']) and ('?' in header.get_text() or re.search(r'question|faq|help', header.get_text(), re.I)):
            faqs.append({'question': header.get_text(strip=True), 'answer': next_element.get_text(strip=True)})
    if faqs: return faqs
    for dt_tag in soup.find_all('dt'):
        dd_tag = dt_tag.find_next_sibling('dd')
        if dd_tag:
            faqs.append({'question': dt_tag.get_text(strip=True), 'answer': dd_tag.get_text(strip=True)})
    if faqs: return faqs
    return None

def scrape_page_thread_safe(url):
    """Scrapes a single URL for FAQs in a thread-safe manner."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        faqs = get_faqs_with_schema(soup) or get_faqs_heuristically(soup)
        if faqs: return faqs
    except requests.exceptions.RequestException:
        pass
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('log-level=3')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        faqs = get_faqs_with_schema(soup) or get_faqs_heuristically(soup)
        if faqs: return faqs
    except Exception as e:
        print(f"Dynamic scrape failed for {url}: {e}")
    finally:
        if driver:
            driver.quit()
    return None

def scrape_faqs_concurrently(base_url):
    """Orchestrates the entire scraping process concurrently."""
    sitemap_url = find_sitemap_url(base_url)
    if not sitemap_url:
        st.error("Is website ke liye koi sitemap nahi mila.")
        return None
    
    all_urls = get_all_urls_from_sitemap(sitemap_url)
    faq_urls = [url for url in all_urls if any(kw in url.lower() for kw in ['faq', 'help', 'support', 'qanda', 'questions'])]
    
    st.info(f"Sitemap mein {len(all_urls)} URLs mile. {len(faq_urls)} potential FAQ pages ko check kiya ja raha hai.")

    all_faqs = []
    MAX_WORKERS = 5
    
    with st.spinner("Concurrent scraping in progress..."):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(scrape_page_thread_safe, faq_urls)
            
            for faqs in results:
                if faqs:
                    all_faqs.extend(faqs)
    
    return all_faqs

# --- Streamlit UI ---
st.set_page_config(page_title="Universal FAQ Scraper", layout="centered")

st.title("🌐 Universal FAQ Scraper")
st.markdown("URL enter karein aur scraper FAQs automatically dhoond kar nikalega.")

with st.form("faq_form"):
    base_url = st.text_input("Website URL", "https://www.apple.com/in/")
    submitted = st.form_submit_button("FAQs Dhoondo")

if submitted and base_url:
    start_time = time.time()
    all_faqs = scrape_faqs_concurrently(base_url)
    end_time = time.time()
    execution_time = end_time - start_time
    
    if all_faqs:
        st.success("FAQs mil gaye!")
        st.info(f"Total Execution Time: {execution_time:.2f} seconds")
        
        jsonl_output = ""
        for faq in all_faqs:
            jsonl_output += json.dumps({
                "question": faq["question"],
                "answer": faq["answer"]
            }) + "\n"
        
        st.download_button(
            label="Download FAQs as JSONL",
            data=jsonl_output,
            file_name="scraped_faqs.jsonl",
            mime="application/jsonl"
        )
        
        st.subheader("Extracted FAQs")
        questions = [faq["question"] for faq in all_faqs]
        question_dict = {faq["question"]: faq["answer"] for faq in all_faqs}

        selected_question = st.selectbox(
            "Select a question to see the answer:",
            questions
        )
        
        if selected_question:
            st.markdown("---")
            st.subheader("Answer")
            st.write(question_dict[selected_question])
    else:
        st.warning("Koi FAQs nahi mil paye.")