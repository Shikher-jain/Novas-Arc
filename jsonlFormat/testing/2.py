# step2_extract_faqs.py

import requests, json, re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

def extract_faq_from_jsonld(soup):
    faqs = []
    scripts = soup.find_all('script', type='application/ld+json')
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                data = data[0]
            if isinstance(data, dict) and data.get("@type") in ["FAQPage", "QAPage"]:
                for item in data.get("mainEntity", []):
                    if item.get("@type") == "Question":
                        q = item.get("name", "").strip()
                        a = item.get("acceptedAnswer", {}).get("text", "").strip()
                        if q and a:
                            faqs.append({"question": q, "answer": a})
        except:
            continue
    return faqs

def extract_faq_heuristic(soup):
    faqs = []
    for d in soup.find_all('details'):
        s = d.find('summary')
        c = d.find(['p', 'div', 'li'])
        if s and c:
            faqs.append({"question": s.get_text(strip=True), "answer": c.get_text(strip=True)})
    for h in soup.find_all(['h2','h3','h4']):
        n = h.find_next_sibling()
        if n and n.name in ['p','div'] and ('?' in h.get_text() or re.search(r'faq|help|question', h.get_text(), re.I)):
            faqs.append({"question": h.get_text(strip=True), "answer": n.get_text(strip=True)})
    for dt in soup.find_all('dt'):
        dd = dt.find_next_sibling('dd')
        if dd:
            faqs.append({"question": dt.get_text(strip=True), "answer": dd.get_text(strip=True)})
    return faqs

def scrape_faqs(url, driver):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        faqs = extract_faq_from_jsonld(soup) or extract_faq_heuristic(soup)
        if faqs: return faqs
    except:
        pass

    # dynamic fallback
    try:
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        faqs = extract_faq_from_jsonld(soup) or extract_faq_heuristic(soup)
        return faqs
    except:
        return None

if __name__ == "__main__":
    with open("urls.txt", "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f]

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    all_faqs = []
    for url in urls:
        print(f"🔎 Scraping {url}")
        faqs = scrape_faqs(url, driver)
        if faqs:
            all_faqs.extend(faqs)

    driver.quit()

    with open("faqs.jsonl", "w", encoding="utf-8") as f:
        for faq in all_faqs:
            f.write(json.dumps(faq, ensure_ascii=False) + "\n")

    print(f"✅ Extracted {len(all_faqs)} FAQs. Saved to faqs.jsonl")
